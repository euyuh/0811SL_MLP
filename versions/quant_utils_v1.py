import torch
import torch.nn as nn
from torch.nn import functional as F
# ==================== Quantization Scheme Config ====================
class QuantizeScheme(object):
    def __init__(self):
        self.scheme = 'google'
        self.subscheme = 'per_channel'
        self.is_scale_pow = True
        self.weight_bits = 8
        self.act_bits = 8

quan_scheme = QuantizeScheme()

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

# ==================== Fake Quantization Function ====================
def ActQuantization(input, FloatMax=6.0, FloatMin=-6.0, num_bits=None):
    """
    Simulate quantization in forward pass with STE for backward.
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
ACT_EMA = {}   # activation range
BN_EMA = {}    # BN running stats
EMA_MOMENTUM = 0.99

def update_activation_ema(name, tensor, momentum=EMA_MOMENTUM):
    batch_min = float(tensor.min().detach().cpu().item())
    batch_max = float(tensor.max().detach().cpu().item())
    if name not in ACT_EMA:
        ACT_EMA[name] = {'min': batch_min, 'max': batch_max}
    else:
        ACT_EMA[name]['min'] = momentum * ACT_EMA[name]['min'] + (1 - momentum) * batch_min
        ACT_EMA[name]['max'] = momentum * ACT_EMA[name]['max'] + (1 - momentum) * batch_max

def get_activation_range(name, default_min=-6.0, default_max=6.0):
    if name in ACT_EMA:
        return ACT_EMA[name]['max'], ACT_EMA[name]['min']
    else:
        return default_max, default_min

def update_bn_ema(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d):
            key = f"bn_{id(m)}"
            mean = m.running_mean.detach().cpu().clone()
            var = m.running_var.detach().cpu().clone()
            if key not in BN_EMA:
                BN_EMA[key] = {'mean': mean, 'var': var}
            else:
                BN_EMA[key]['mean'] = EMA_MOMENTUM * BN_EMA[key]['mean'] + (1 - EMA_MOMENTUM) * mean
                BN_EMA[key]['var'] = EMA_MOMENTUM * BN_EMA[key]['var'] + (1 - EMA_MOMENTUM) * var

# ==================== BN Folding for Inference ====================
def FoldLinear(real_linear: nn.Linear, real_bn: nn.BatchNorm1d):
    sigma = torch.sqrt(real_bn.running_var + real_bn.eps)
    gamma = real_bn.weight
    beta = real_bn.bias
    mu = real_bn.running_mean
    scale = gamma / sigma
    folded_weight = real_linear.weight * scale.unsqueeze(1)
    b = real_linear.bias if real_linear.bias is not None else torch.zeros(real_linear.out_features, device=folded_weight.device)
    folded_bias = gamma * (b - mu) / sigma + beta
    return folded_weight.detach().clone(), folded_bias.detach().clone()

# ==================== Real Quantization (for inference) ====================

import torch

def quantize_tensor(tensor, num_bits=8, symmetric=True):
    """
    实际量化函数（推理用）
    tensor: 输入浮点张量
    num_bits: 比特宽度 (通常8)
    symmetric: 是否对称量化 (zero_point=0)
    返回: (int_tensor, scale, zero_point)
    """
    qmin = 0
    qmax = 2 ** num_bits - 1
    if symmetric:
        max_val = tensor.abs().max()
        scale = max_val / ((qmax + 1) / 2)
        zero_point = 0
        q_tensor = torch.clamp((tensor / scale).round(),
                               -((qmax + 1) / 2),
                               ((qmax + 1) / 2) - 1).to(torch.int8)
    else:
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - (min_val / scale)
        q_tensor = torch.clamp(((tensor / scale) + zero_point).round(),
                               qmin, qmax).to(torch.uint8)
    return q_tensor, scale, zero_point


def quantize_linear_layer(layer, num_bits=8):
    """
    将 nn.Linear 层的权重、偏置进行实际量化 (推理阶段)
    返回 (w_q, b_q, w_scale, b_scale)
    """
    w_q, w_scale, w_zp = quantize_tensor(layer.weight.data, num_bits=num_bits, symmetric=True)
    if layer.bias is not None:
        b_q, b_scale, b_zp = quantize_tensor(layer.bias.data, num_bits=num_bits, symmetric=True)
    else:
        b_q, b_scale, b_zp = None, None, None
    return w_q, b_q, w_scale, b_scale


def quantize_activation(a_tensor, act_min, act_max, num_bits=8):
    """
    根据训练阶段统计的激活范围进行实际量化
    返回 (q_tensor, scale, zero_point)
    """
    qmin = 0
    qmax = 2 ** num_bits - 1
    scale = (act_max - act_min) / (qmax - qmin)
    zero_point = qmin - act_min / scale
    q_a = torch.clamp((a_tensor / scale + zero_point).round(), qmin, qmax).to(torch.uint8)
    return q_a, scale, zero_point

#量化部署

def inference_quantized(client_model, server_model, image, act_ema_dict, num_bits=8):
    """
    整个 Split Learning 模型的量化推理流程（client + server）
    模拟真正的 int8 部署。
    参数:
        client_model : 已训练好的客户端模型
        server_model : 已训练好的服务器模型
        image        : 单张输入样本 (torch.Tensor)
        act_ema_dict : 激活范围统计字典 (ACT_EMA)
        num_bits     : 量化位宽 (默认8)
    返回:
        output_float : 推理结果 (反量化后的浮点数)
    """

    client_model.eval()
    server_model.eval()

    # ============ Client Forward + Quantization ============
    with torch.no_grad():
        # 前向传播
        fx = client_model(image)

        # 获取客户端激活范围
        act_max, act_min = act_ema_dict.get("client_0_out", {"max": 6.0, "min": -6.0}).values()

        # 激活量化
        fx_q, fx_scale, fx_zp = quantize_activation(fx, act_min, act_max, num_bits=num_bits)

        # 反量化到浮点（server 接收前）
        fx_deq = (fx_q.float() - fx_zp) * fx_scale

        # ============ Server Forward with Quantized Weights ============
        x = fx_deq
        for name, layer in server_model.named_children():
            if isinstance(layer, nn.Linear):
                # 量化权重和偏置
                w_q, b_q, w_scale, b_scale = quantize_linear_layer(layer, num_bits=num_bits)

                # 模拟 int8 GEMM：w_q (int8) @ x(float) ≈ (w_q * w_scale) @ x
                x = F.linear(x, (w_q.float() * w_scale), (b_q.float() * b_scale) if b_q is not None else None)

                # 激活量化（可选，每层都可加）
                act_max, act_min = act_ema_dict.get(f"server_{name}_out", {"max": 6.0, "min": -6.0}).values()
                x_q, x_scale, x_zp = quantize_activation(x, act_min, act_max, num_bits=num_bits)
                x = (x_q.float() - x_zp) * x_scale

            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.GELU):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                try:
                    x = layer(x)
                except Exception:
                    pass

        output_float = x
    return output_float



# ==================== Quantized Model Export ====================

import os

def export_quantized_model(client_model, server_model, export_dir="quantized_export", num_bits=8):
    """
    将客户端和服务器端的模型参数进行量化并导出。
    - 权重、偏置进行 int8 对称量化
    - 输出部分量化结果到终端
    - 保存量化参数到 .pt 文件

    参数：
        client_model : 已训练好的客户端模型
        server_model : 已训练好的服务器模型
        export_dir   : 导出目录
        num_bits     : 量化位宽 (默认8)
    """
    os.makedirs(export_dir, exist_ok=True)

    all_layers_quant = {}

    print(f"\n========== 导出量化模型参数 (num_bits={num_bits}) ==========\n")

    # ============ 导出客户端模型 ============
    print("---- Client Model ----")
    client_layers = {}
    for name, module in client_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w_q, b_q, w_scale, b_scale = quantize_linear_layer(module, num_bits=num_bits)

            print(f"[Client::{name}] weight int8 取值范围: [{int(w_q.min())}, {int(w_q.max())}]  scale={w_scale:.6f}")

            # 打印部分量化参数（整数形式）
            print("  权重量化示例（前10个整数）:", w_q.view(-1)[:10].tolist())

            client_layers[name] = {
                "w_q": w_q.cpu(),
                "b_q": b_q.cpu() if b_q is not None else None,
                "w_scale": w_scale,
                "b_scale": b_scale,
            }

    all_layers_quant["client"] = client_layers

    # ============ 导出服务器模型 ============
    print("\n---- Server Model ----")
    server_layers = {}
    for name, module in server_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w_q, b_q, w_scale, b_scale = quantize_linear_layer(module, num_bits=num_bits)

            print(f"[Server::{name}] weight int8 取值范围: [{int(w_q.min())}, {int(w_q.max())}]  scale={w_scale:.6f}")
            print("  权重量化示例（前10个整数）:", w_q.view(-1)[:10].tolist())

            server_layers[name] = {
                "w_q": w_q.cpu(),
                "b_q": b_q.cpu() if b_q is not None else None,
                "w_scale": w_scale,
                "b_scale": b_scale,
            }

    all_layers_quant["server"] = server_layers

    # ============ 保存量化模型 ============
    export_path = os.path.join(export_dir, f"split_model_quantized_{num_bits}bit.pt")
    torch.save(all_layers_quant, export_path)

    print(f"\n✅ 量化模型参数已导出到: {export_path}")
    print("=========================================================\n")

    return export_path