import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. 基础量化运算核心 (通用组件)
# ==========================================
class QuantizedLinear(nn.Module):
    """
    执行: Float输入 -> 量化(Int8) -> 整数矩阵乘 -> 反量化(Float)
    """
    def __init__(self, layer_params, input_stats=None, num_bits=8):
        super().__init__()
        self.w_q = layer_params['w_q'].float()
        self.b_q = layer_params['b_q'].float() if layer_params['b_q'] is not None else None
        self.w_scale = layer_params['w_scale']
        self.b_scale = layer_params['b_scale']
        
        # 设定输入量化范围
        if input_stats:
            self.in_min, self.in_max = input_stats['act_min'], input_stats['act_max']
        else:
            self.in_min, self.in_max = -1.0, 1.0

        # 计算 Input Scale & ZeroPoint
        qmax = 2**num_bits - 1
        if self.in_max == self.in_min:
            self.in_scale = 1.0
        else:
            self.in_scale = (self.in_max - self.in_min) / qmax
        
        self.in_zp = round(-self.in_min / (self.in_scale + 1e-12))

    def forward(self, x_float):
        # 1. Input Quantization (模拟设备端的量化操作)
        x_q = torch.clamp(torch.round(x_float / self.in_scale + self.in_zp), 0, 255)
        
        # 2. Integer MatMul (模拟 NPU/DSP 的整数运算)
        x_shifted = x_q - self.in_zp
        acc_int = F.linear(x_shifted, self.w_q) 
        
        # 3. Dequantize (模拟输出反量化，或者下一层的输入准备)
        out = acc_int * (self.in_scale * self.w_scale)
        if self.b_q is not None:
            out += (self.b_q * self.b_scale)
        return out

# ==========================================
# 2. 客户端模型 (Client Model)
# ==========================================
class ClientInference(nn.Module):
    def __init__(self, client_params):
        super().__init__()
        self.params = client_params
        # 统一使用 ReLU6 输出范围作为层间输入统计
        self.RELU_STATS = {'act_min': 0.0, 'act_max': 6.0}

        # L1: 原始图片输入 (范围约 -0.5 ~ 3.0)
        self.fc1 = QuantizedLinear(self.params['client_layers/layers.1_folded'], 
                                   input_stats={'act_min': -0.5, 'act_max': 3.0})
        
        # L2-L5: 内部层级 (输入均为上一层的 ReLU6 输出)
        self.fc2 = QuantizedLinear(self.params['client_layers/layers.5_folded'], input_stats=self.RELU_STATS)
        self.fc3 = QuantizedLinear(self.params['client_layers/layers.9_folded'], input_stats=self.RELU_STATS)
        self.fc4 = QuantizedLinear(self.params['client_layers/layers.13_folded'], input_stats=self.RELU_STATS)
        self.fc5 = QuantizedLinear(self.params['client_layers/layers.17_folded'], input_stats=self.RELU_STATS)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu6(self.fc1(x))
        x = F.relu6(self.fc2(x))
        x = F.relu6(self.fc3(x))
        x = F.relu6(self.fc4(x))
        # 客户端最后一层输出，也是要传输给服务器的数据
        output = F.relu6(self.fc5(x)) 
        return output

# ==========================================
# 3. 服务器端模型 (Server Model)
# ==========================================
class ServerInference(nn.Module):
    def __init__(self, server_params):
        super().__init__()
        self.params = server_params
        # 同样假设接收的数据和内部数据范围均为 ReLU6 范围
        self.RELU_STATS = {'act_min': 0.0, 'act_max': 6.0}

        # S_L1: 接收来自 Client 的数据 (0~6)
        self.fc1 = QuantizedLinear(self.params['server_layers/layers.0_folded'], input_stats=self.RELU_STATS)
        
        # S_L2-L4: 内部层级
        self.fc2 = QuantizedLinear(self.params['server_layers/layers.4_folded'], input_stats=self.RELU_STATS)
        self.fc3 = QuantizedLinear(self.params['server_layers/layers.8_folded'], input_stats=self.RELU_STATS)
        self.fc4 = QuantizedLinear(self.params['server_layers/layers.12_folded'], input_stats=self.RELU_STATS)
        
        # Classifier: 最后一层输出 Logits
        self.fc5 = QuantizedLinear(self.params['server_layers/layers.15'], input_stats=self.RELU_STATS)

    def forward(self, x):
        # x 是从客户端传输过来的中间特征
        x = F.relu6(self.fc1(x))
        x = F.relu6(self.fc2(x))
        x = F.relu6(self.fc3(x))
        x = F.relu6(self.fc4(x))
        # 分类层
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 4. 主流程：模拟分割推理
# ==========================================
def main():
    device = torch.device("cpu") # 模拟端侧通常为CPU
    BATCH_SIZE = 128
    
    # 1. 准备数据
    transform = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.Grayscale(1), transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
    test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 加载量化参数文件
    export_file = "./quantized_export/split_model_quantized_8bit.pt"
    if not os.path.exists(export_file):
        print("错误：找不到量化参数文件，请先运行训练脚本。")
        return
    
    print(f"Loading parameters from: {export_file}")
    all_params = torch.load(export_file)
    
    # 3. 实例化两个独立的模型
    client_model = ClientInference(all_params['client']).to(device)
    server_model = ServerInference(all_params['server']).to(device)
    
    client_model.eval()
    server_model.eval()
    
    print(f"\n[Split Inference] Starting explicitly split inference...")
    print(f"Client: Handling Layers 1-5")
    print(f"Server: Handling Layers 6-10")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # --- [Step 1] Client Side Computation ---
            # 客户端计算得到中间激活值 (smashed data)
            client_output = client_model(images)
            
            # --- [Step 2] Data Transmission (Simulated) ---
            # 在这里，数据从客户端传输到服务器
            # 实际场景中这里会有: 量化压缩 -> 网络传输 -> 解压
            transmitted_data = client_output.detach().clone() 
            
            # --- [Step 3] Server Side Computation ---
            # 服务器接收数据继续计算
            final_output = server_model(transmitted_data)
            
            # 统计结果
            preds = final_output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 简单验证一下第一批数据
            if total == labels.size(0):
                print(f"First Batch Prediction: {preds[:5].tolist()}")
                print(f"First Batch True Label: {labels[:5].tolist()}")

    acc = 100.0 * correct / total
    print(f"\n========================================")
    print(f"Final Split Quantized Accuracy: {acc:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()