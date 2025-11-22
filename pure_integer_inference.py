import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image

#这里做个样子而已，感觉要写好多层
# ============================================================
# 模块 1：纯整数运算所需函数（不依赖 PyTorch 模型结构）
# ============================================================

def load_quant_param(path):
    """
    从导出的 txt 中读取整数权重矩阵
    """
    with open(path, "r") as f:
        rows = f.read().strip().split("\n")
        arr = [[int(x) for x in row.split()] for row in rows]
    return torch.tensor(arr, dtype=torch.int32)


def load_scalar(path):
    """
    从 txt 文件中读取 1 个浮点数
    """
    return float(open(path, "r").read().strip())


def integer_linear(x_int, w_int, b_int, x_zp, w_zp, b_zp, M):
    """
    纯整数线性层（不含激活函数），公式对应：

        y_int = M * Σ( (x_int - x_zp) * (w_int - w_zp) ) + b_int

    其中 M = real_multiplier（折叠后 scale_x * scale_w / scale_out）
    """

    # 减去 zero point
    x_adj = x_int - x_zp
    w_adj = w_int - w_zp

    # GEMM 阶段累加使用 int32
    acc = torch.matmul(x_adj, w_adj.t())     # (N, in) × (in, out)

    # 线性层 bias（也是量化后 int32）
    acc = acc + b_int

    # 乘以 real_multiplier（浮点数）并四舍五入为 int32
    y = torch.round(acc.float() * M).to(torch.int32)

    return y


def quantize_activation(x_float, act_min, act_max, bits=8):
    """
    真实推理输入的量化（uint8）
    """
    qmin, qmax = 0, 2 ** bits - 1
    scale = (act_max - act_min) / (qmax - qmin)
    zp = qmin - act_min / (scale + 1e-12)
    x_q = torch.clamp((x_float / (scale + 1e-12) + zp).round(), qmin, qmax)
    return x_q.to(torch.int32), scale, int(zp)


# ============================================================
# 模块 2：加载 test 数据集（与 SL_MLP_inference_0805 完全一致）
# ============================================================

class SkinData(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.paths = df['path'].tolist()
        self.labels = df['target'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def load_test_dataset(test_csv, test_img_folder):
    df = pd.read_csv(test_csv)
    df["path"] = [os.path.join(test_img_folder, f"{i}.png") for i in range(len(df))]
    df["target"] = df["label"]

    test_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])

    dataset_test = SkinData(df, test_transforms)
    loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    return loader


# ============================================================
# 模块 3：加载导出的量化参数（client + server）
# ============================================================

def load_exported_quant_model(EXPORT_PATH):
    """读取 ./quantized_export 中的整数权重、偏置以及 scale 等"""
    model = {}

    # ---------------------- client fc1 ---------------------
    model["fc1_w"] = load_quant_param(os.path.join(EXPORT_PATH, "client_fc1_weight_int.txt"))
    model["fc1_b"] = load_quant_param(os.path.join(EXPORT_PATH, "client_fc1_bias_int.txt"))
    model["fc1_scale_x"] = load_scalar(os.path.join(EXPORT_PATH, "client_fc1_scale_x.txt"))
    model["fc1_scale_w"] = load_scalar(os.path.join(EXPORT_PATH, "client_fc1_scale_w.txt"))
    model["fc1_scale_out"] = load_scalar(os.path.join(EXPORT_PATH, "client_fc1_scale_out.txt"))
    model["fc1_zp_x"] = load_scalar(os.path.join(EXPORT_PATH, "client_fc1_zp_x.txt"))
    model["fc1_zp_w"] = load_scalar(os.path.join(EXPORT_PATH, "client_fc1_zp_w.txt"))
    model["fc1_zp_out"] = load_scalar(os.path.join(EXPORT_PATH, "client_fc1_zp_out.txt"))

    # ---------------------- server fc2 ---------------------
    model["fc2_w"] = load_quant_param(os.path.join(EXPORT_PATH, "server_fc2_weight_int.txt"))
    model["fc2_b"] = load_quant_param(os.path.join(EXPORT_PATH, "server_fc2_bias_int.txt"))
    model["fc2_scale_x"] = load_scalar(os.path.join(EXPORT_PATH, "server_fc2_scale_x.txt"))
    model["fc2_scale_w"] = load_scalar(os.path.join(EXPORT_PATH, "server_fc2_scale_w.txt"))
    model["fc2_scale_out"] = load_scalar(os.path.join(EXPORT_PATH, "server_fc2_scale_out.txt"))
    model["fc2_zp_x"] = load_scalar(os.path.join(EXPORT_PATH, "server_fc2_zp_x.txt"))
    model["fc2_zp_w"] = load_scalar(os.path.join(EXPORT_PATH, "server_fc2_zp_w.txt"))
    model["fc2_zp_out"] = load_scalar(os.path.join(EXPORT_PATH, "server_fc2_zp_out.txt"))

    # ---------------- Activation 量化参数 ----------------
    model["input_min"] = load_scalar(os.path.join(EXPORT_PATH, "activation_input_min.txt"))
    model["input_max"] = load_scalar(os.path.join(EXPORT_PATH, "activation_input_max.txt"))

    return model


# ============================================================
# 模块 4：纯整数推理 pipeline（client → server）
# ============================================================

def SL_integer_inference(x_float, quant_model):
    """
    输入：预处理后的浮点图像（1, 1, 64, 64）
    输出：整数推理 logits（int32）
    """

    # ---------- (1) 输入量化 ----------
    x_flat = x_float.view(1, -1)  # (1, 4096)
    x_int, scale_x, zp_x = quantize_activation(
        x_flat, quant_model["input_min"], quant_model["input_max"]
    )

    # ============ Client 侧 fc1（整数 GEMM）============
    M1 = (quant_model["fc1_scale_x"] * quant_model["fc1_scale_w"]) / quant_model["fc1_scale_out"]

    fc1_out_int = integer_linear(
        x_int,
        quant_model["fc1_w"],
        quant_model["fc1_b"],
        quant_model["fc1_zp_x"],
        quant_model["fc1_zp_w"],
        quant_model["fc1_zp_out"],
        M1
    )

    # ============ Server 侧 fc2（整数 GEMM）============
    M2 = (quant_model["fc2_scale_x"] * quant_model["fc2_scale_w"]) / quant_model["fc2_scale_out"]

    fc2_out_int = integer_linear(
        fc1_out_int,
        quant_model["fc2_w"],
        quant_model["fc2_b"],
        quant_model["fc2_zp_x"],
        quant_model["fc2_zp_w"],
        quant_model["fc2_zp_out"],
        M2
    )

    return fc2_out_int


# ============================================================
# 模块 5：运行 test 数据集并统计准确率
# ============================================================

def run_integer_test(test_loader, quant_model):
    correct = 0
    total = 0

    for img, label in test_loader:
        # 整数推理
        logits_int = SL_integer_inference(img, quant_model)

        # 输出仍然是 int32 → 取 argmax 即可
        pred = torch.argmax(logits_int, dim=1).item()

        if pred == label:
            correct += 1
        total += 1

    acc = correct / total
    print(f"\n===== 纯整数推理 Accuracy: {acc:.4f} =====\n")
    return acc


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":

    EXPORT_PATH = "./quantized_export/split_model_quantized_8bit.pt"

    print("加载模型参数中...")
    quant_model = load_exported_quant_model(EXPORT_PATH)

    print("加载 test 数据集...")
    test_loader = load_test_dataset(
        test_csv="/root/autodl-fs/0811SL_MLP/mnist_test.csv",
        test_img_folder="/root/autodl-fs/0811SL_MLP/test_images"
    )

    print("开始整数推理测试...")
    run_integer_test(test_loader, quant_model)
