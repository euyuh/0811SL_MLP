import torch
import torch.nn as nn
from torch.nn import functional as F
# ==================== EMA Config ====================
ACT_EMA = {}   # activation range
BN_EMA = {}    # BN running stats
EMA_MOMENTUM = 0.9
# ==================== Quantization Scheme Config ====================
class QuantizeScheme(object):
    def __init__(self):
        self.scheme = 'google'
        self.subscheme = 'per_channel'
        self.is_scale_pow = True
        self.weight_bits = 8
        self.act_bits = 8

quan_scheme = QuantizeScheme()



# 放到 quant_utils.py 中合适位置（已使用 BN_EMA, EMA_MOMENTUM 全局）


class CustomBatchNorm1d(nn.Module):
    """
    Custom BatchNorm1d that:
      - 在训练时计算本批次 mean/var，
      - 立即把本批次统计用 EMA 合并到全局 BN_EMA（按 key）【用于后续的 BN-folding/导出】，
      - 可选地使用更新后的 EMA 值来进行本次 forward 的归一化（paper-style）。
    参数:
      num_features: 通道数
      eps, momentum: 与 nn.BatchNorm1d 相似的数值稳定项和 running-momentum（用于维护 running_mean/var）
      prefix: 用于构成 BN_EMA 的 key，比如 "client_layers/layers.3"
      name_in_module: 层在 module 中的名字（如果能传入更方便）
      use_ema_for_norm: bool，是否在训练时用 EMA 值做归一化（True -> 符合 paper C.5 顺序）
    注意:  EMA 更新使用 quant_utils.BN_EMA 全局 dict（与你现有导出/折叠逻辑对齐）
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.9, prefix=None, name_in_module=None, use_ema_for_norm=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum  # 用于 running_mean/var 的传统更新（可保留）
        self.prefix = prefix      # e.g. "client_layers"
        self.name_in_module = name_in_module  # e.g. "layers.3"
        self.use_ema_for_norm = use_ema_for_norm

        # affine params (gamma, beta)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # keep running stats for compatibility (not used directly if use_ema_for_norm True)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x: [N, C] or [N, C, ...] (here assume Linear outputs -> [N,C])
        assert x.dim() >= 2
        if x.dim() > 2:
            # collapse spatial dims
            dims = [0] + list(range(2, x.dim()))
            B = x.transpose(1, -1).contiguous().view(-1, self.num_features)
        else:
            B = x

        if self.training:
            # 1) compute batch stats
            batch_mean = B.mean(dim=0)
            # use unbiased estimator? paper uses population-like (we use B.var(unbiased=False))
            batch_var = B.var(dim=0, unbiased=False)

            # 2) update running_mean/var (standard BN behavior) (in-place, no grad)
            with torch.no_grad():
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * batch_var

            # 3) update global BN_EMA (paper: EMA with momentum ~0.99) so FoldLinear can use it later.
            # construct key consistent with quant_utils.update_bn_ema format
            # the quant_utils expects keys like f"{prefix}/{name}_bn" or f"{prefix}.{name}_bn" depending on your convention.
            # Here we follow your quant_utils: key = f"{prefix}/{name}_bn"
            if self.prefix is not None and self.name_in_module is not None:
                key = f"{self.prefix}/{self.name_in_module}_bn"
            else:
                key = None

            if key is not None:
                # BN_EMA is imported from quant_utils (global dict)
                # Use detach->cpu clone to store CPU copies (consistent with update_bn_ema)
                # from quant_utils import BN_EMA, EMA_MOMENTUM #感觉不用啊，不都全局了？
                with torch.no_grad():
                    cur_mean = batch_mean.detach().cpu().clone()
                    cur_var = batch_var.detach().cpu().clone()
                    if key not in BN_EMA:
                        BN_EMA[key] = {'mean': cur_mean, 'var': cur_var}
                    else:#为什么这里还需要计算呢？不能直接用刚刚算的running直接读取吗
                        BN_EMA[key]['mean'] = EMA_MOMENTUM * BN_EMA[key]['mean'] + (1 - EMA_MOMENTUM) * cur_mean
                        BN_EMA[key]['var']  = EMA_MOMENTUM * BN_EMA[key]['var']  + (1 - EMA_MOMENTUM) * cur_var

            # 4) choose stats for normalization:
            if self.use_ema_for_norm and key is not None and key in BN_EMA:
                # use newly updated EMA (move to device)
                mean = BN_EMA[key]['mean'].to(x.device)
                var  = BN_EMA[key]['var'].to(x.device)
            else:
                # fall back to batch stats (standard BN behavior)
                mean = batch_mean.to(x.device)
                var  = batch_var.to(x.device)
        else:
            # eval mode: use running stats
            mean = self.running_mean.to(x.device)
            var  = self.running_var.to(x.device)

        # normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        # reshape to broadcast
        shape = [1, -1] + [1] * (x.dim() - 2)
        gamma = self.weight.view(*shape)
        beta  = self.bias.view(*shape)
        mean_b = mean.view(*shape)
        var_b = var.view(*shape)

        x_norm = (x - mean_b) / torch.sqrt(var_b + self.eps)
        out = x_norm * gamma + beta
        return out

# ==================== Straight-Through Estimator ====================
class DifferentialRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        # Round in forward (simulate quantization)
        i = i.clone()
        i.round_()
        ctx.save_for_backward(i)
        return i

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output

# ==================== Fake Quantization for Activations ====================
def ActQuantization(input, FloatMax=6.0, FloatMin=0.0, num_bits=None):
    """
    Simulate quantization in forward pass with STE for backward.
    (activation fake quant)
    """
    if num_bits is None:
        num_bits = quan_scheme.act_bits
    QuantizeMax = (2 ** num_bits) - 1.0
    Float2QuanScale = QuantizeMax / (FloatMax - FloatMin + 1e-12)
    x = input * Float2QuanScale
    x = DifferentialRound.apply(x)
    x = x / Float2QuanScale
    return x

# ==================== EMA Trackers ====================
def update_activation_ema(name, tensor, momentum=EMA_MOMENTUM):
    batch_min = float(tensor.min().detach().cpu().item())
    batch_max = float(tensor.max().detach().cpu().item())
    if name not in ACT_EMA:
        ACT_EMA[name] = {'min': batch_min, 'max': batch_max}
    else:
        ACT_EMA[name]['min'] = momentum * ACT_EMA[name]['min'] + (1 - momentum) * batch_min
        ACT_EMA[name]['max'] = momentum * ACT_EMA[name]['max'] + (1 - momentum) * batch_max

def get_activation_range(name, default_min=0.0, default_max=6.0):
    if name in ACT_EMA:
        return ACT_EMA[name]['max'], ACT_EMA[name]['min']
    else:
        return default_max, default_min



def update_bn_ema(model: torch.nn.Module, prefix: str = "model"):
    """
    遍历模型中的 BatchNorm1d 层，使用统一的命名规则更新 BN_EMA。
    命名方式：f"{prefix}/{name}_bn"
    示例：
        client_layers/layers.1_bn
        server_layers/layers.3_bn
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            key = f"{prefix}/{name}_bn"
            mean = m.running_mean.detach().cpu().clone()
            var  = m.running_var.detach().cpu().clone()

            if key not in BN_EMA:
                BN_EMA[key] = {'mean': mean, 'var': var}
            else:
                BN_EMA[key]['mean'] = EMA_MOMENTUM * BN_EMA[key]['mean'] + (1 - EMA_MOMENTUM) * mean
                BN_EMA[key]['var']  = EMA_MOMENTUM * BN_EMA[key]['var']  + (1 - EMA_MOMENTUM) * var

# ==================== register hooks for activation EMA ====================
def _activation_hook_factory(name):
    """
    返回一个 forward hook 函数，该 hook 接收 module, input, output，并把 output 的 min/max 更新到 ACT_EMA。
    output 可以是张量或 tuple/list（取第一个张量）。
    """
    def hook(module, inp, outp):
        # outp 可能是 tensor 或 tuple/list
        if isinstance(outp, (tuple, list)):
            t = outp[0]
        else:
            t = outp
        # 只有在是 Tensor 时处理（避免 None 等）
        if isinstance(t, torch.Tensor):
            update_activation_ema(name, t)#直接调用ema函数实现函数复用
    return hook

def register_activation_ema_hooks(model: torch.nn.Module, prefix: str="model"):
    """
    遍历 model 的子模块并对指定类型的模块注册 forward hooks，
    针对每个 hook 存入的 key: f"{prefix}/{name}_out"   （name 使用 module 的命名路径）
    常注册对象：Linear, Conv, GELU, ReLU, Sigmoid, BatchNorm 的输出（即每个操作后的激活）
    使用方式：
        register_activation_ema_hooks(net_glob_client, prefix="client_layers")
        register_activation_ema_hooks(net_glob_server, prefix="server_layers")
    注意：若使用 DataParallel，需要传入 .module（如 net.module）
    """
    for name, module in model.named_modules():
        # 跳过顶层空名字
        if name == "":
            continue
        # 我们通常希望记录以下层的输出（可按需扩展）
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm1d,
                               torch.nn.ReLU, torch.nn.ReLU6, torch.nn.GELU, torch.nn.Sigmoid, torch.nn.Tanh,
                               torch.nn.Dropout, torch.nn.Flatten, CustomBatchNorm1d)):
            hook_name = f"{prefix}/{name}_out"
            module.register_forward_hook(_activation_hook_factory(hook_name))


# ==================== BN Folding for Inference ====================
def FoldLinear(real_linear: nn.Linear, real_bn: nn.BatchNorm1d, prefix, next_name):
    """
    Fold Linear + BN into a single Linear (folded_weight, folded_bias)
    This is useful for deployment quantization: produce the effective weight/bias after folding.
    """
    bn_key = f"{prefix}.{next_name}_bn"
    device = real_bn.weight.device
    if bn_key in BN_EMA:
        mean_ema = BN_EMA[bn_key]['mean'].to(device)
        var_ema = BN_EMA[bn_key]['var'].to(device)
    else:
        mean_ema = torch.zeros_like(real_bn.running_mean)
        var_ema = torch.ones_like(real_bn.running_var)
    # sigma = torch.sqrt(real_bn.running_var + real_bn.eps)
    sigma = torch.sqrt(var_ema + real_bn.eps)
    gamma = real_bn.weight
    beta = real_bn.bias
    # mu = real_bn.running_mean
    mu = mean_ema
    scale = gamma / sigma
    # weight: [out, in] -> scale.unsqueeze(1) broadcast to match out dim
    folded_weight = real_linear.weight * scale.unsqueeze(1)
    b = real_linear.bias if real_linear.bias is not None else torch.zeros(real_linear.out_features, device=folded_weight.device)
    folded_bias = gamma * (b - mu) / sigma + beta
    return folded_weight.detach().clone(), folded_bias.detach().clone()

# ==================== Fake-Quant Linear Module (for training) ====================
class FakeQuantLinear(nn.Linear):
    """
    Linear layer that applies fake-quantization to weights during forward.
    - Bias is initialized to zero and frozen (requires_grad=False) during training as requested.
    - Weight fake-quantization is per-output-channel symmetric quantization with STE.
    """
    def __init__(self, in_features, out_features, bias=True, bits=None):
        # We'll still create bias param but set it to zero and freeze
        super(FakeQuantLinear, self).__init__(in_features, out_features, bias=bias)
        # Initialize bias to zero and freeze it (no gradient)
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()
            self.bias.requires_grad = False
        # bits (if None use global)
        self.bits = bits if bits is not None else quan_scheme.weight_bits

    def weight_fake_quant(self, weight):
        """
        Per-output-channel symmetric fake quantization for weights.
        weight: torch.Tensor of shape (out_features, in_features)
        returns dequantized weight (float) with STE during backward.
        """
        num_bits = self.bits
        qmax = 2 ** (num_bits - 1) - 1  # symmetric signed range -qmax..qmax
        # compute max abs per out-channel
        max_vals = weight.abs().amax(dim=1, keepdim=True)  # shape (out, 1)
        # avoid 0 division
        max_vals = torch.where(max_vals == 0.0, torch.tensor(1.0, device=weight.device), max_vals)
        scale = max_vals / (qmax + 1e-12)  # scale per out-channel
        # scale shape (out,1) -> broadcast
        scaled = weight / scale
        # apply rounding with STE
        q = DifferentialRound.apply(scaled)
        # clip to symmetric range
        q = torch.clamp(q, -qmax, qmax)
        deq = q * scale
        return deq

    def forward(self, input):
        # fake-quant weights
        w_q_deq = self.weight_fake_quant(self.weight)
        # use frozen bias (maybe zero)
        b = self.bias if self.bias is not None else None
        return F.linear(input, w_q_deq, b)

# ==================== Real Quantization (for inference) ====================
def quantize_tensor(tensor, num_bits=8, symmetric=True):
    """
    实际量化函数（推理用）
    tensor: 输入浮点张量
    num_bits: 比特宽度 (通常8)
    symmetric: 是否对称量化 (zero_point=0)
    返回: (int_tensor, scale, zero_point)
    Notes:
      - For symmetric quantization we return int8-like signed tensor in torch.int8 and zero_point=0
      - For asymmetric we return uint8 and a computed zero_point
    """
    qmin = 0
    qmax = 2 ** num_bits - 1
    if symmetric:
        # symmetric signed representation, map to [-2^{b-1} .. 2^{b-1}-1]
        absmax = tensor.abs().max()
        if absmax == 0:
            scale = 1.0
        else:
            scale = absmax / ((2 ** (num_bits - 1)) - 1)
        zero_point = 0
        # quantize to signed range
        q_tensor = torch.clamp((tensor / scale).round(),
                               -((2 ** (num_bits - 1)) - 1),
                               (2 ** (num_bits - 1)) - 1).to(torch.int8)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        if max_val == min_val:
            scale = 1.0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - (min_val / (scale + 1e-12))
        q_tensor = torch.clamp(((tensor / (scale + 1e-12)) + zero_point).round(),
                               qmin, qmax).to(torch.uint8)
    return q_tensor, scale, zero_point


def quantize_linear_layer_from_tensors(w_tensor, b_tensor=None, num_bits=8):
    """
    将给定的权重张量和偏置张量进行实际量化 (推理阶段)
    返回 (w_q, b_q, w_scale, b_scale, w_zero_point, b_zero_point)
    Uses symmetric quantization (signed int).
    """
    w_q, w_scale, w_zp = quantize_tensor(w_tensor, num_bits=num_bits, symmetric=True)
    if b_tensor is not None:
        b_q, b_scale, b_zp = quantize_tensor(b_tensor, num_bits=num_bits, symmetric=True)
    else:
        b_q, b_scale, b_zp = None, None, None
    return w_q, b_q, w_scale, b_scale, w_zp, b_zp


def quantize_activation(a_tensor, act_min, act_max, num_bits=8):
    """
    根据训练阶段统计的激活范围进行实际量化
    返回 (q_tensor, scale, zero_point)
    Asymmetric uint8 is produced here because activations are typically non-centered.
    """
    qmin = 0
    qmax = 2 ** num_bits - 1
    # avoid zero range
    if act_max == act_min:
        scale = 1.0
    else:
        scale = (act_max - act_min) / (qmax - qmin)
    zero_point = qmin - act_min / (scale + 1e-12)
    q_a = torch.clamp((a_tensor / (scale + 1e-12) + zero_point).round(), qmin, qmax).to(torch.uint8)
    return q_a, scale, zero_point

# ==================== Inference Simulation (using quantized params) ====================
import torch.nn.functional as F

def inference_quantized(client_model, server_model, image, act_ema_dict, num_bits=8):
    """
    整个 Split Learning 模型的量化推理流程（client + server）
    模拟真正的 int8 部署。此函数不做真正的 integer-only 优化，而是用量化->反量化的方式模拟部署。
    """
    client_model.eval()
    server_model.eval()

    with torch.no_grad():
        # ============ Client Forward + Quantization ============
        fx = client_model(image)

        act_key = "client_layers/layers.20_out"
        if act_key in ACT_EMA:
            act_min = ACT_EMA[act_key]["min"]
            act_max = ACT_EMA[act_key]["max"]
        else:
            act_min, act_max = -6.0, 6.0
        # 激活量化为 uint8（asymmetric）
        fx_q, fx_scale, fx_zp = quantize_activation(fx, act_min, act_max, num_bits=num_bits)
        
        # # 获取客户端激活范围
        # act_max, act_min = act_ema_dict.get("client_0_out", {"max": 6.0, "min": -6.0}).values()

        # # 激活量化为 uint8（asymmetric）
        # fx_q, fx_scale, fx_zp = quantize_activation(fx, act_min, act_max, num_bits=num_bits)

        # 在传输前可以把 fx_q 视为 uint8 bytes；server 端收到后反量化到 float（此处模拟）
        fx_deq = (fx_q.float() - fx_zp) * fx_scale

        # ============ Server Forward with Quantized Weights ============
        x = fx_deq
        for name, layer in server_model.named_children():
            # try to behave similar to actual structure
            # if layer is a sequential container, iterate inside
            if isinstance(layer, nn.Sequential):
                for sub in layer:
                    if isinstance(sub, nn.Linear):
                        # quantize weight/bias of this linear (simulate)
                        w_q, b_q, w_scale, b_scale, _, _ = quantize_linear_layer_from_tensors(sub.weight.data, sub.bias.data if sub.bias is not None else None, num_bits=num_bits)
                        # dequantize weight to float for simulation
                        w_deq = w_q.float() * w_scale
                        b_deq = (b_q.float() * b_scale) if b_q is not None else None
                        x = F.linear(x, w_deq, b_deq)
                    else:
                        try:
                            x = sub(x)
                        except Exception:
                            pass
            else:
                if isinstance(layer, nn.Linear):
                    w_q, b_q, w_scale, b_scale, _, _ = quantize_linear_layer_from_tensors(layer.weight.data, layer.bias.data if layer.bias is not None else None, num_bits=num_bits)
                    w_deq = w_q.float() * w_scale
                    b_deq = (b_q.float() * b_scale) if b_q is not None else None
                    x = F.linear(x, w_deq, b_deq)
                else:
                    try:
                        x = layer(x)
                    except Exception:
                        pass

        output_float = x
    return output_float

# ==================== Quantized Model Export (improved with BN-folding) ====================
import os

def export_quantized_model(client_model, server_model, export_dir="quantized_export", num_bits=8):
    """
    将客户端和服务器端的模型参数进行量化并导出（用于部署的整数量化参数文件）。
    - 对 Linear + BN 的组合，先折叠 BN 得到 folded_weight, folded_bias，再对 folded 参数做量化。
    - 保存 structure: { 'client': {layer_name: {w_q,b_q, w_scale,b_scale, w_zp,b_zp}}, 'server': {...} }
    """
    os.makedirs(export_dir, exist_ok=True)

    all_layers_quant = {}

    print(f"\n========== 导出量化模型参数 (num_bits={num_bits}) ==========\n")

    # helper to traverse and fold / quantize pairs inside a module (assume sequential-like)
    def process_sequential(mod: nn.Module, prefix: str = "", num_bits: int = 8):
        """
        遍历模块，导出量化参数（含激活统计），支持：
        - Linear + CustomBatchNorm1d 折叠
        - 嵌套子模块递归
        - 与 register_activation_ema_hooks() 注册的命名一致
        """
        #调试
        print(f">>> 进入 process_sequential(): prefix={prefix}")
        children = list(mod.named_children())
        print(">>> 子模块数 =", len(children))

        layers_quant = {}

        for name, m in mod.named_children():
        # 拼接当前层完整路径，与 hook 注册时一致
            full_name = f"{prefix}.{name}" if prefix else name
            # =====================
            # case 1: Linear + BN
            # =====================
            next_modules = list(mod.named_children())
            # 获取该层在当前Sequential中的下标
            idx = [n for n, _ in next_modules].index(name)
            if isinstance(m, nn.Linear):
                # 判断下一层是否是BatchNorm1d
                if idx + 1 < len(next_modules):
                    next_name, next_mod = next_modules[idx + 1]
                    # if isinstance(next_mod, nn.BatchNorm1d):
                    if isinstance(next_mod, CustomBatchNorm1d):
                        folded_w, folded_b = FoldLinear(m, next_mod, prefix, next_name)
                        w_q, b_q, w_scale, b_scale, w_zp, b_zp = quantize_linear_layer_from_tensors(
                            folded_w, folded_b, num_bits=num_bits
                        )

                        layers_quant[f"{full_name}_folded"] = {
                            "w_q": w_q.cpu(),
                            "b_q": b_q.cpu() if b_q is not None else None,
                            "w_scale": float(w_scale),
                            "b_scale": float(b_scale) if b_scale is not None else None,
                            "w_zero_point": int(w_zp.item()) if torch.is_tensor(w_zp) else int(w_zp),
                            "b_zero_point": int(b_zp.item()) if torch.is_tensor(b_zp) else int(b_zp),
                            "orig_type": "Linear+BN_folded"
                        }

                        # === 导出对应激活范围 ===
                        # 修正后
                        bn_key = f"{prefix}.{next_name}_out"
                        if bn_key in ACT_EMA:
                            act_min = float(ACT_EMA[bn_key]['min'])
                            act_max = float(ACT_EMA[bn_key]['max'])
                        else:
                            # 如果 BN 没有记录，就尝试 Linear 层的输出
                            lin_key = f"{full_name}_out"
                            if lin_key in ACT_EMA:
                                act_min = float(ACT_EMA[lin_key]['min'])
                                act_max = float(ACT_EMA[lin_key]['max'])
                            else:
                                act_min, act_max = -6.0, 6.0
                        act_scale = (act_max - act_min) / (2 ** num_bits - 1)
                        act_zp = round(-act_min / act_scale)
                        layers_quant[f"{full_name}_folded"].update({
                            "act_min": act_min,
                            "act_max": act_max,
                            "act_scale": float(act_scale),
                            "act_zero_point": int(act_zp)
                        })
                        continue  # 跳过BN层

            # =====================
            # case 2: Linear（无BN）
            # =====================
            if isinstance(m, nn.Linear):
                w_q, b_q, w_scale, b_scale, w_zp, b_zp = quantize_linear_layer_from_tensors(
                    m.weight.data,
                    m.bias.data if m.bias is not None else None,
                    num_bits=num_bits
                )
                layers_quant[full_name] = {
                    "w_q": w_q.cpu(),
                    "b_q": b_q.cpu() if b_q is not None else None,
                    "w_scale": float(w_scale),
                    "b_scale": float(b_scale) if b_scale is not None else None,
                    "w_zero_point": int(w_zp.item()) if torch.is_tensor(w_zp) else int(w_zp),
                    "b_zero_point": int(b_zp.item()) if torch.is_tensor(b_zp) else int(b_zp),
                    "orig_type": "Linear"
                }

                # === 导出激活范围 ===
                act_key = f"{full_name}_out"
                if act_key in ACT_EMA:
                    act_min = float(ACT_EMA[act_key]['min'])
                    act_max = float(ACT_EMA[act_key]['max'])
                else:
                    act_min, act_max = -6.0, 6.0
                act_scale = (act_max - act_min) / (2 ** num_bits - 1)
                act_zp = round(-act_min / act_scale)
                layers_quant[full_name].update({
                    "act_min": act_min,
                    "act_max": act_max,
                    "act_scale": float(act_scale),
                    "act_zero_point": int(act_zp)
                })

        # =====================
        # case 3: 嵌套模块（递归）
        # =====================
            elif any(True for _ in m.children()):
                sub_layers = process_sequential(m, prefix=full_name, num_bits=num_bits)
                layers_quant.update(sub_layers)

        # =====================
        # case 4: 其他类型层（ReLU, Dropout等）
        # 这里只导出激活范围，不导出权重
        # =====================
            else:
                act_key = f"{full_name}_out"
                if act_key in ACT_EMA:
                    act_min = float(ACT_EMA[act_key]['min'])
                    act_max = float(ACT_EMA[act_key]['max'])
                    act_scale = (act_max - act_min) / (2 ** num_bits - 1)
                    act_zp = round(-act_min / act_scale)
                    layers_quant[full_name] = {
                        "orig_type": m.__class__.__name__,
                        "act_min": act_min,
                        "act_max": act_max,
                        "act_scale": float(act_scale),
                        "act_zero_point": int(act_zp)
                    }

        return layers_quant


    # ============ 导出客户端模型 ============
    print("---- Client Model ----")
    # 如果是 DataParallel，则取内部 module
    if isinstance(client_model, nn.DataParallel):
        client_model = client_model.module
    if isinstance(server_model, nn.DataParallel):
        server_model = server_model.module
    
    client_layers = {}
    target_client = client_model.layers if hasattr(client_model, "layers") else client_model
    client_layers = process_sequential(target_client, prefix="client_layers/layers")
    for k, v in client_layers.items():
        w_scale = v.get("w_scale", None)
        if w_scale is not None:
            print(f"[Client::{k}] type={v.get('orig_type','N/A')} w_scale={w_scale:.6f}")
        else:
            print(f"[Client::{k}] type={v.get('orig_type','N/A')} (no w_scale)")
    all_layers_quant["client"] = client_layers

    # ============ 导出服务器模型 ============
    print("\n---- Server Model ----")
    server_layers = {}
    target_server = server_model.layers if hasattr(server_model, "layers") else server_model
    server_layers = process_sequential(target_server, prefix="server_layers/layers")
    for k, v in server_layers.items():
        w_scale = v.get("w_scale", None)
        if w_scale is not None:
            print(f"[Server::{k}] type={v.get('orig_type','N/A')} w_scale={w_scale:.6f}")
        else:
            print(f"[Server::{k}] type={v.get('orig_type','N/A')} (no w_scale)")
    all_layers_quant["server"] = server_layers

    # ============ 保存量化模型 ============
    export_path = os.path.join(export_dir, f"split_model_quantized_{num_bits}bit.pt")
    torch.save(all_layers_quant, export_path)

    print(f"\n✅ 量化模型参数已导出到: {export_path}")
    print("=========================================================\n")

    return export_path
