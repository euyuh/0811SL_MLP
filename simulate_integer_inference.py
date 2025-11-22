# integer_sim_from_export.py
# 说明：使用导出格式（w_q, b_q, w_scale, b_scale, act_min, act_max, act_scale, act_zero_point）
# 做整数仿真推理（整数累加 int32 + requantize 到下层 uint8），参数以 float 存储但值为整数。
# 完整数据读取（使用 test 数据集），逐样本推理并统计准确率。
# 注：本脚本为仿真/验证用途。若想完全 integer-only（无浮点乘法），需把 real_multiplier -> (multiplier,shift)。

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# ============= 配置 =============
EXPORT_PATH = "./quantized_export/split_model_quantized_8bit.pt"  # 你的导出文件
TEST_CSV = '/root/autodl-fs/0811SL_MLP/mnist_test.csv'
TEST_IMG_FOLDER = '/root/autodl-fs/0811SL_MLP/test_images'
BATCH_SIZE = 64
NUM_BITS = 8
QMIN, QMAX = 0, 2**NUM_BITS - 1

# ============= 数据集（与 baseline 一致） =============
class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.iloc[idx]['path']
        img = Image.open(p).convert('L').resize((64,64))
        if self.transform:
            img = self.transform(img)
        label = int(self.df.iloc[idx]['target'])
        return img, label

def load_test_loader():
    df_test = pd.read_csv(TEST_CSV)
    df_test['path'] = [os.path.join(TEST_IMG_FOLDER, f"{i}.png") for i in range(len(df_test))]
    df_test['target'] = df_test['label']

    test_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])

    ds = SkinData(df_test, test_transforms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader

# ============= 模型结构（仅用于层次遍历；运算使用导出的参数进行） =============
class DeployedClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(64*64, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU()
        ])
    def forward(self, x):  # not used for computation in integer sim
        for l in self.layers:
            x = l(x)
        return x

class DeployedServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 10)
        ])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

# ============= 工具函数 =============
def quantize_activation_from_range(x_float_np, act_min, act_max, num_bits=8):
    """用 act_min/act_max 量化 float 激活到 uint8"""
    qmin, qmax = 0, 2**num_bits - 1
    if act_max == act_min:
        scale = 1.0
    else:
        scale = float((act_max - act_min) / (qmax - qmin))
    zero_point = int(round(qmin - act_min / (scale + 1e-12)))
    q = np.round(x_float_np / (scale + 1e-12) + zero_point).astype(np.int32)
    q = np.clip(q, qmin, qmax).astype(np.uint8)
    return q, scale, zero_point

def compute_real_multiplier(w_scale, x_scale, next_act_scale):
    """real multiplier = (w_scale * x_scale) / next_act_scale"""
    # 防止除零
    if next_act_scale == 0:
        return 1.0
    return float((w_scale * x_scale) / next_act_scale)

# ============= 加载导出参数 =============
def load_exported_pt(export_path):
    saved = torch.load(export_path, map_location='cpu')
    client_saved = saved.get('client', {})
    server_saved = saved.get('server', {})
    return client_saved, server_saved

# ============= 将导出 entries 映射到模型 Linear 层索引，并准备 quant params =============
def prepare_quant_params(saved_dict, model_linear_modules):
    """
    saved_dict: 导出字典（client 或 server）
    model_linear_modules: list of nn.Linear modules in model order
    返回：
      params: dict mapping linear_layer_index -> {
            'w_q': np.int8 array [out,in],
            'b_q': np.int32 array [out] (if present),
            'w_scale': float,
            'b_scale': float or None,
            'act_min': float, 'act_max': float, 'act_scale': float, 'act_zp': int
      }
    """
    params = {}
    idx = 0
    keys = list(saved_dict.keys())
    for key in keys:
        entry = saved_dict[key]
        if 'w_q' not in entry:
            continue
        if idx >= len(model_linear_modules):
            print("Warning: more exported linear entries than model linear modules")
            break

        # 抽取字段（兼容 torch tensor 或 numpy）
        w_q_t = entry['w_q']  # torch tensor
        w_q_np = w_q_t.cpu().numpy().astype(np.int32)  # store as int32 for safe matmul
        b_q_np = None
        if 'b_q' in entry and entry['b_q'] is not None:
            b_q_np = entry['b_q'].cpu().numpy().astype(np.int32)

        w_scale = float(entry.get('w_scale', 1.0))
        b_scale = float(entry.get('b_scale', 0.0)) if entry.get('b_scale', None) is not None else None

        # activation stats for this layer's output (act_min/max refer to layer output)
        act_min = float(entry.get('act_min', -6.0))
        act_max = float(entry.get('act_max', 6.0))
        act_scale = float(entry.get('act_scale', (act_max - act_min) / (QMAX - QMIN))) if 'act_scale' in entry else float((act_max - act_min) / (QMAX - QMIN))
        act_zp = int(entry.get('act_zero_point', int(round(QMIN - act_min / (act_scale + 1e-12)))))

        params[idx] = {
            'key': key,
            'w_q': w_q_np,
            'b_q': b_q_np,
            'w_scale': w_scale,
            'b_scale': b_scale,
            'act_min': act_min,
            'act_max': act_max,
            'act_scale': act_scale,
            'act_zp': act_zp
        }
        idx += 1
    return params

# ============= 整数层前向（核心） =============
def int_layer_forward(X_q_np, x_zp, W_q_np, B_q_np, w_scale, b_scale, next_act_scale, next_act_zp):
    """
    X_q_np: uint8 or int32 numpy array [B, in]
    x_zp: int (zero point of current activation)
    W_q_np: int32 numpy array [out, in]
    B_q_np: int32 array [out] or None
    next_act_scale / next_act_zp: 用于 requantize 到下一层激活域
    返回:
      acc_int32: np.int32 [B, out]
      Y_q_next: np.uint8 [B, out] （若提供 next_act_scale/zp）
      Y_float_deq: np.float32 [B, out] (反量化 float，用于最终对照)
    """
    # convert types
    X = X_q_np.astype(np.int32)
    W = W_q_np.astype(np.int32)

    # 中心化 X
    Xc = X - int(x_zp)  # [B, in]

    # int32 accumulator
    acc = Xc.dot(W.T)  # shape [B, out], dtype int32

    # add bias if provided (bias was quantized with some scale b_scale)
    if B_q_np is not None:
        acc = acc + B_q_np.astype(np.int32).reshape(1, -1)

    # 反量化到 float 以便对比（Y_float = acc * (w_scale * x_scale) + b_deq）
    # 注意：此处 b_deq 的计算：如果 b_q 与 b_scale 给出，则 b_deq = b_q * b_scale
    # 如果 b_q 为 None，b_deq = 0
    if B_q_np is not None and b_scale is not None:
        b_deq = B_q_np.astype(np.float32) * b_scale
    else:
        b_deq = np.zeros((acc.shape[1],), dtype=np.float32)

    Y_float = acc.astype(np.float32) * (w_scale * x_scale_global) + b_deq.reshape(1, -1)

    # requantize acc -> next uint8 using real_multiplier
    real_mult = compute_real_multiplier(w_scale, x_scale_global, next_act_scale)
    # Y_q_next = round(acc * real_mult) + next_act_zp
    Y_scaled = np.round(acc.astype(np.float64) * real_mult).astype(np.int64) + int(next_act_zp)
    Y_q_next = np.clip(Y_scaled, QMIN, QMAX).astype(np.uint8)

    return acc, Y_q_next, Y_float

# ============= 主运行函数 =============
def run_integer_simulation(export_path):
    # load exported dicts
    client_saved, server_saved = load_exported_pt(export_path)

    # build models just to get linear layers ordering
    client_model = DeployedClient()
    server_model = DeployedServer()

    # list linear modules (their indices in the ModuleList) for mapping
    client_linears = [i for i, m in enumerate(client_model.layers) if isinstance(m, nn.Linear)]
    server_linears = [i for i, m in enumerate(server_model.layers) if isinstance(m, nn.Linear)]

    # prepare quant params per linear (indexing by sequential 0..)
    client_params = prepare_quant_params(client_saved, client_linears)
    server_params = prepare_quant_params(server_saved, server_linears)

    # 为了执行 requantize，需要为每层知道“下一层 act_scale/zp”
    # 所以我们把 client_params/server_params 转为列表按层序排列，方便取 next
    def to_ordered_list(params_dict):
        ordered = []
        keys = sorted(params_dict.keys())
        for k in keys:
            ordered.append(params_dict[k])
        return ordered

    client_ordered = to_ordered_list(client_params)
    server_ordered = to_ordered_list(server_params)

    # load test data
    loader = load_test_loader()

    total = 0
    correct = 0

    print("开始整数仿真推理（使用导出量化整数参数）...")
    # 全局 x_scale（当前层的输入 scale）在运行时更新；定义全局变量以便 int_layer_forward 使用
    global x_scale_global
    x_scale_global = None

    for batch_idx, (img, label) in enumerate(loader):
        # flatten and convert to numpy float32
        x = img.view(img.size(0), -1).numpy().astype(np.float32)  # shape [B, in]
        B = x.shape[0]

        # ============ Client 部分 ============
        # client first layer's act_min/act_max & z are used to quantize input
        if len(client_ordered) == 0:
            raise RuntimeError("客户端没有导出任何线性层信息")

        first = client_ordered[0]
        in_min = first['act_min']
        in_max = first['act_max']
        # input activation quantization (uint8)
        X_q, x_scale, x_zp = quantize_activation_from_range(x, in_min, in_max)
        # 注意更新全局 x_scale 供 int_layer_forward 使用
        x_scale_global = x_scale

        # iterate client layers
        Xq_curr = X_q  # uint8 numpy
        for li, layer_entry in enumerate(client_ordered):
            W = layer_entry['w_q']
            Bq = layer_entry['b_q']
            w_scale = layer_entry['w_scale']
            b_scale = layer_entry['b_scale']

            # next layer act params (若存在则取 next 的 act_scale/zp，否则复用当前层的 act_scale/zp)
            if li + 1 < len(client_ordered):
                next_act_scale = client_ordered[li+1]['act_scale']
                next_act_zp = client_ordered[li+1]['act_zp']
            else:
                # 如果客户端最后一层的输出直接传 server，尝试用 server 首层的 act 信息
                if len(server_ordered) > 0:
                    next_act_scale = server_ordered[0]['act_scale']
                    next_act_zp = server_ordered[0]['act_zp']
                else:
                    next_act_scale = layer_entry['act_scale']
                    next_act_zp = layer_entry['act_zp']

            acc, Yq_next, Yfloat = int_layer_forward(Xq_curr, x_zp, W, Bq, w_scale, b_scale, next_act_scale, next_act_zp)

            # debug 打印（每批只打印前一两个样本以免日志过大）
            if batch_idx == 0:
                print(f"\nClient layer {li} ({layer_entry['key']})")
                print("  W_q shape:", W.shape, " sample weights:", W.flatten()[:6])
                print("  b_q sample:", (Bq.flatten()[:6] if Bq is not None else None))
                print("  X_q sample (first row) :", Xq_curr[0, :min(8, Xq_curr.shape[1])])
                print("  acc sample (first row)  :", acc[0, :min(6, acc.shape[1])])
                print("  Yfloat sample (first row):", Yfloat[0, :min(6, Yfloat.shape[1])])
                print("  Yq_next sample (first row):", Yq_next[0, :min(6, Yq_next.shape[1])])

            # prepare next input
            Xq_curr = Yq_next
            # update x_scale/x_zp to next layer (用于下一轮 int_layer_forward 中的反量化计算)
            x_scale_global = next_act_scale
            x_zp = next_act_zp

        # ============ 传输：客户端 -> 服务器（这里传 uint8 激活 Xq_curr） ============
        # ============ Server 部分 ============
        Xq_server = Xq_curr
        # iterate server layers
        for si, layer_entry in enumerate(server_ordered):
            W = layer_entry['w_q']
            Bq = layer_entry['b_q']
            w_scale = layer_entry['w_scale']
            b_scale = layer_entry['b_scale']

            if si + 1 < len(server_ordered):
                next_act_scale = server_ordered[si+1]['act_scale']
                next_act_zp = server_ordered[si+1]['act_zp']
            else:
                # 最后一层使用自身 act_scale/act_zp（输出的量化域）
                next_act_scale = layer_entry['act_scale']
                next_act_zp = layer_entry['act_zp']

            acc, Yq_next, Yfloat = int_layer_forward(Xq_server, x_zp, W, Bq, w_scale, b_scale, next_act_scale, next_act_zp)

            if batch_idx == 0:
                print(f"\nServer layer {si} ({layer_entry['key']})")
                print("  W_q shape:", W.shape, " sample weights:", W.flatten()[:6])
                print("  b_q sample:", (Bq.flatten()[:6] if Bq is not None else None))
                print("  acc sample (first row):", acc[0, :min(6, acc.shape[1])])
                print("  Yfloat sample (first row):", Yfloat[0, :min(6, Yfloat.shape[1])])
                print("  Yq_next sample (first row):", Yq_next[0, :min(6, Yq_next.shape[1])])

            Xq_server = Yq_next
            x_scale_global = next_act_scale
            x_zp = next_act_zp

        # 最后 Xq_server 为量化形式的输出；我们也保留 Yfloat 为 float logits（便于比较）
        # 选用反量化的 float logits 做 argmax
        final_logits = Yfloat  # numpy array [B, out]
        preds = np.argmax(final_logits, axis=1)
        correct += int((preds == label.numpy()).sum())
        total += label.size(0)

    acc = correct / total
    print(f"\n===== Integer-simulated Inference Accuracy = {acc:.4f} =====")
    return acc

# ============= 入口 =============
if __name__ == "__main__":
    if not os.path.exists(EXPORT_PATH):
        raise FileNotFoundError(f"导出文件不存在: {EXPORT_PATH}")
    acc = run_integer_simulation(EXPORT_PATH)
