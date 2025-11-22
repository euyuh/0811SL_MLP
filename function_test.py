# verify_export_quant_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import shutil

# 导入你上传的 quant_utils（脚本假设 quant_utils.py 在同目录）
from quant_utils import (
    FakeQuantLinear, register_activation_ema_hooks,
    ACT_EMA, BN_EMA, update_bn_ema,
    export_quantized_model
)

torch.manual_seed(0)

# ----------------------------
# 1) 构造极简 client/server (Sequential 层结构)
# ----------------------------
class TinyClient(nn.Module):
    def __init__(self):
        super().__init__()
        # layers 名称和 SL 示例风格一致（保证 export 的 prefix/path 匹配）
        self.layers = nn.Sequential(
            nn.Flatten(),                      # 0
            FakeQuantLinear(4, 3),             # 1: Linear
            nn.BatchNorm1d(3),                 # 2: BN
            nn.GELU(),                         # 3
            FakeQuantLinear(3, 2),             # 4: Linear (no BN after)
        )

    def forward(self, x):
        return self.layers(x)


class TinyServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            FakeQuantLinear(2, 2),     # 0
            nn.BatchNorm1d(2),         # 1
            nn.GELU(),                 # 2
            FakeQuantLinear(2, 2)      # 3 (final logits)
        )

    def forward(self, x):
        return self.layers(x)


# ----------------------------
# 2) 构建输入（可见具体数值）和 targets
# ----------------------------
# 小 batch=2, 输入是 1x4 的扁平特征（使用 Flatten 保持与模型一致）
inputs = torch.tensor([
    [ 0.20,  0.10, -0.30,  0.50],
    [ 0.00, -0.20,  0.40, -0.10],
], dtype=torch.float32)  # shape (2,4)

targets = torch.tensor([1, 0], dtype=torch.long)  # 分类标签 (batch_size=2)

print("\n=== 输入与标签 (visible) ===")
print("inputs:\n", inputs)
print("targets:\n", targets)


# ----------------------------
# 3) 初始化模型并注册 activation hooks（必须用 prefix 与 export 同步）
# ----------------------------
client_model = TinyClient()
server_model = TinyServer()

# 注册 hook，prefix 要与 export_quantized_model 中一致的前缀格式（export 使用 "client_layers/layers"）
register_activation_ema_hooks(client_model, prefix="client_layers")
register_activation_ema_hooks(server_model, prefix="server_layers")

# 把模型移到 CPU（便于打印）；若你希望跑 GPU，可 .to(device)
device = torch.device("cpu")
client_model.to(device)
server_model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)

# 优化器（分别对 client 和 server）
opt_client = optim.SGD(client_model.parameters(), lr=0.05)
opt_server = optim.SGD(server_model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

# 清空全局 EMA（防止之前运行残留）
ACT_EMA.clear()
BN_EMA.clear()

# ----------------------------
# 4) 进行两次 forward+backward（以填充 ACT_EMA 和 BN_EMA）
#    使用联合训练： client -> server -> loss -> backward -> 两端 update
# ----------------------------
for epoch in range(2):
    print(f"\n\n================ Epoch {epoch+1} ================\n")
    client_model.train()
    server_model.train()
    opt_client.zero_grad()
    opt_server.zero_grad()

    # forward client
    fx = client_model(inputs)  # hooks registered on layers will record ACT_EMA entries
    print("[After client forward] client activation (fx):\n", fx.detach())

    # forward server
    out = server_model(fx)
    print("[After server forward] server output (logits):\n", out.detach())

    # update BN_EMA for both models (prefixes must match what export expects)
    # export expects BN_EMA keys like "client_layers/layers.<idx>_bn" so we use same prefix pattern
    update_bn_ema(client_model, prefix="client_layers")
    update_bn_ema(server_model, prefix="server_layers")

    # loss & backward
    loss = criterion(out, targets)
    print("Loss:", loss.item())
    loss.backward()

    # 打印各层参数与其梯度（完整矩阵）
    print("\n--- PARAMETERS and GRADIENTS (client) ---")
    for name, p in client_model.named_parameters():
        print(f"client.{name} shape={tuple(p.shape)}")
        print("value:\n", p.detach().cpu().numpy())
        print("grad:\n", None if p.grad is None else p.grad.detach().cpu().numpy())
        print("-----")

    print("\n--- PARAMETERS and GRADIENTS (server) ---")
    for name, p in server_model.named_parameters():
        print(f"server.{name} shape={tuple(p.shape)}")
        print("value:\n", p.detach().cpu().numpy())
        print("grad:\n", None if p.grad is None else p.grad.detach().cpu().numpy())
        print("-----")

    # step
    opt_client.step()
    opt_server.step()

    # 打印 ACT_EMA 状态（完整）
    print("\n--- ACT_EMA (after forward) ---")
    for k, v in ACT_EMA.items():
        print(f"{k} -> min={v['min']}, max={v['max']}")

    # 打印 BN_EMA 状态（完整）
    print("\n--- BN_EMA (after update_bn_ema) ---")
    for k, v in BN_EMA.items():
        # print full tensors
        print(f"{k} -> mean={v['mean'].cpu().numpy()}, var={v['var'].cpu().numpy()}")

# ----------------------------
# 5) 调用 export_quantized_model（这将内部调用 FoldLinear 等）
#    导出位置用临时目录，运行后我们 load 并打印导出内容（便于比对）
# ----------------------------
export_dir = "./tmp_quant_export"
# 清理旧目录（如果存在）
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
os.makedirs(export_dir, exist_ok=True)

print("\n\n>>> 调用 export_quantized_model() 开始导出")
export_path = export_quantized_model(client_model, server_model, export_dir=export_dir, num_bits=8)
print("export_path:", export_path)

# Load and inspect saved file
print("\n--- 加载并显示导出文件内容（部分） ---")
saved = torch.load(export_path, map_location="cpu")

def show_layer_info(prefix, d):
    print(f"\n>>> {prefix} 有 {len(d)} 个导出条目")
    for k, v in d.items():
        print(f"\nLayer key: {k}")
        for subk, subv in v.items():
            # 对大张量仅显示形状与前几个元素
            if hasattr(subv, "shape"):
                s = tuple(subv.shape)
                sample = subv.flatten()[:6].tolist() if subv.numel() > 0 else []
                print(f"  {subk}: shape={s}, sample={sample}")
            else:
                print(f"  {subk}: {subv}")

show_layer_info("client", saved.get("client", {}))
show_layer_info("server", saved.get("server", {}))

print("\n=== 验证完成 ===\n")
