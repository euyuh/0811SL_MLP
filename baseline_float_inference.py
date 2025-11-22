# baseline_float_inference.py
# 功能：
# 1. 加载导出的量化模型参数（包含折叠后的 w_q, scale, zero point）
# 2. 重建 PyTorch float 模型（client + server）
# 3. 将量化权重反量化为 float 并写回模型
# 4. 遍历 test 数据集做浮点推理（baseline）
# 5. 输出 float baseline accuracy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

# =============================
# 1. 重建 PyTorch 模型结构（无BN，因为导出已fold）
# =============================
class DeployedClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
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
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class DeployedServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

# =============================
# 2. 将导出的量化权重反量化写入 PyTorch 模型
# =============================
def load_export_to_pytorch(export_path, client_model, server_model):

    saved = torch.load(export_path, map_location="cpu")
    client_dict = saved["client"]
    server_dict = saved["server"]

    def assign_weights(saved_section, model):
        # 获取 linear 层的顺序
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        idx = 0

        for key, entry in saved_section.items():
            if "w_q" not in entry:
                continue

            w_q = entry["w_q"].float()
            w_scale = float(entry["w_scale"])
            b_q = entry.get("b_q", None)
            b_scale = entry.get("b_scale", None)

            # -----------------------------
            # 反量化：w_deq = w_q * w_scale
            # -----------------------------
            w_deq = w_q * w_scale

            if b_q is not None:
                b_deq = b_q.float() * b_scale
            else:
                b_deq = None

            # 写入线性层
            layer = linear_layers[idx]
            assert layer.weight.shape == w_deq.shape

            with torch.no_grad():
                layer.weight.copy_(w_deq)
                if b_deq is not None:
                    layer.bias.copy_(b_deq)

            print(f"Loaded layer {idx}: {key}")
            idx += 1

    assign_weights(client_dict, client_model)
    assign_weights(server_dict, server_model)

    return client_model, server_model

# =============================
# 3. Test 数据集加载（与你训练时一致）
# =============================
# from SL_MLP_inference_0805 import SkinData  # 你已有的数据集类(直接调用会开始训练)

class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['path']
        X = Image.open(img_path).resize((64,64)).convert('L')
        y = torch.tensor(int(self.df.iloc[index]['target']))
        if self.transform:
            X = self.transform(X)
        return X, y
    

def load_test_dataset():
    test_csv = '/root/autodl-fs/0811SL_MLP/mnist_test.csv'
    test_img_folder = '/root/autodl-fs/0811SL_MLP/test_images'
    df_test = pd.read_csv(test_csv)
    df_test["path"] = [os.path.join(test_img_folder, f"{i}.png") for i in range(len(df_test))]
    df_test["target"] = df_test["label"]

    test_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])

    return SkinData(df_test, test_transforms)

# =============================
# 4. 浮点推理 baseline
# =============================
def run_baseline_inference():

    # 重建模型
    client = DeployedClient()
    server = DeployedServer()

    # 将导出参数反量化写回 float 模型
    export_path = "./quantized_export/split_model_quantized_8bit.pt"
    client, server = load_export_to_pytorch(export_path, client, server)

    client.eval()
    server.eval()

    # 加载 test 数据集
    dataset_test = load_test_dataset()
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

    total = 0
    correct = 0

    with torch.no_grad():
        for img, label in loader_test:
            img = img.view(img.size(0), -1)  # flatten
            out_c = client(img)
            logits = server(out_c)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == label).sum().item()
            total += label.size(0)

    acc = correct / total
    print(f"\n🔥 PyTorch FLOAT Baseline Accuracy = {acc*100:.2f}%")
    return acc


if __name__ == "__main__":
    run_baseline_inference()
