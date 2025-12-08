import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. 基础量化运算工具 (模拟端侧推理)
# ==========================================

class QuantizedLinear(nn.Module):
    """
    模拟量化全连接层推理：
    Input(Float) -> Quantize -> Int8 MatrixMul -> Dequantize -> Output(Float)
    """
    def __init__(self, layer_params, input_stats=None, num_bits=8):
        super().__init__()
        self.layer_params = layer_params
        self.num_bits = num_bits
        
        # 提取权重参数 (已经是 Int8)
        self.w_q = layer_params['w_q'].float() # 转为float容器以便在Python中做矩阵乘法，数值仍是整数
        self.b_q = layer_params['b_q'].float() if layer_params['b_q'] is not None else None
        
        # 提取缩放因子 (Scale)
        self.w_scale = layer_params['w_scale']
        self.b_scale = layer_params['b_scale']
        
        # 提取上一层的输出范围作为本层的输入量化范围
        # 如果没有提供(第一层)，则需要外部传入或动态计算
        if input_stats:
            self.in_min = input_stats['act_min']
            self.in_max = input_stats['act_max']
        else:
            self.in_min = -1.0 # 默认值，第一层通常需特殊处理
            self.in_max = 1.0

        # 计算输入的量化参数 (Scale & ZeroPoint)
        qmin, qmax = 0, 2**num_bits - 1
        if self.in_max == self.in_min:
            self.in_scale = 1.0
        else:
            self.in_scale = (self.in_max - self.in_min) / (qmax - qmin)
        self.in_zp = round(qmin - self.in_min / (self.in_scale + 1e-12))

    def forward(self, x_float):
        # --------------------------------------------
        # Step 1: Input Quantization (Float -> Int)
        # --------------------------------------------
        # 公式: q = round(x / scale + zp)
        x_q = torch.clamp(torch.round(x_float / self.in_scale + self.in_zp), 0, 255)
        
        # --------------------------------------------
        # Step 2: Integer Operation (模拟整数矩阵乘法)
        # --------------------------------------------
        # 实际部署中通常为: (x_q - x_zp) * w_q
        # 这里权重 w_q 是对称量化 (zp=0)，输入 x_q 是非对称 (有 zp)
        x_shifted = x_q - self.in_zp
        
        # 矩阵乘法: [Batch, In] @ [Out, In].T = [Batch, Out]
        # 结果累加器 (Accumulator) 通常是 Int32
        acc_int = F.linear(x_shifted, self.w_q) 
        
        # --------------------------------------------
        # Step 3: Dequantize (Int -> Float)
        # --------------------------------------------
        # 公式: Real = Acc * (S_in * S_w) + Bias_Real
        # 注意: 这里的 b_q 我们根据 quant_utils 是独立量化的，所以单独反量化
        
        # 反量化矩阵乘积部分
        out_float_part = acc_int * (self.in_scale * self.w_scale)
        
        # 加上偏置 (如果存在)
        if self.b_q is not None:
            bias_float = self.b_q * self.b_scale
            out = out_float_part + bias_float
        else:
            out = out_float_part
            
        return out

# ==========================================
# 2. 重构客户端模型 (Inference Only)
# ==========================================

class ClientQuantizedModel(nn.Module):
    def __init__(self, export_path):
        super().__init__()
        print(f"Loading quantized parameters from: {export_path}")
        # 加载导出的字典
        self.all_params = torch.load(export_path)
        self.client_params = self.all_params['client'] # 获取 'client' 部分
        
        # 定义层结构 (必须与训练时的结构一致)
        # 字典的 Key 必须与 export_quantized_model 导出时的命名匹配
        
        # --- Layer 1: Flatten (无参数) ---
        
        # --- Layer 2: Linear + BN (已折叠) ---
        # 第一层的输入是原始图像，范围约为 -0.42 到 2.8 (根据 Normalize 0.1307, 0.3081 计算)
        # 我们手动定义第一层的输入范围
        input_stats_L1 = {'act_min': -0.5, 'act_max': 3.0} 
        self.fc1 = QuantizedLinear(self.client_params['client_layers/layers.1_folded'], input_stats_L1)
        
        # --- Layer 3: ReLU6 (无参数，但影响下一层输入范围) ---
        # 下一层的输入范围 = 上一层的输出范围 (存储在 folded 字典中)
        
        # --- Layer 4: Linear + BN ---
        self.fc2 = QuantizedLinear(self.client_params['client_layers/layers.5_folded'], 
                                   input_stats=self.client_params['client_layers/layers.1_folded'])
        
        # --- Layer 5: Linear + BN ---
        self.fc3 = QuantizedLinear(self.client_params['client_layers/layers.9_folded'], 
                                   input_stats=self.client_params['client_layers/layers.5_folded'])
        
        # --- Layer 6: Linear + BN ---
        self.fc4 = QuantizedLinear(self.client_params['client_layers/layers.13_folded'], 
                                   input_stats=self.client_params['client_layers/layers.9_folded'])
        
        # --- Layer 7: Linear + BN ---
        self.fc5 = QuantizedLinear(self.client_params['client_layers/layers.17_folded'], 
                                   input_stats=self.client_params['client_layers/layers.13_folded'])
        
        # --- Classifier ---
        # 尝试自动查找分类器的 Key
        cls_key = None
        possible_keys = ['classifier', 'client_layers/classifier', 'fc_out', 'output']
        
        # 1. 先尝试常用命名
        for k in possible_keys:
            if k in self.client_params:
                cls_key = k
                break
        
        # 2. 如果没找到，模糊搜索包含 'classifier' 的 key
        if cls_key is None:
            for k in self.client_params.keys():
                if 'classifier' in k:
                    cls_key = k
                    break
        
        if cls_key:
            print(f"Found classifier key: {cls_key}")
            self.classifier = QuantizedLinear(self.client_params[cls_key], 
                                              input_stats=self.client_params['client_layers/layers.17_folded'])
        else:
            print("Warning: Could not find 'classifier' key. Inference might fail or output incorrect dimensions.")
            # 如果真的找不到，为了防止报错，可以赋值为 None，但在 forward 里会出错
            self.classifier = None
    
    def forward(self, x):
        x = x.flatten(1)
        
        # Layer 1
        x = self.fc1(x)
        x = F.relu6(x) # 激活函数在 Float 域进行 (模拟)
        
        # Layer 2
        x = self.fc2(x)
        x = F.relu6(x)
        
        # Layer 3
        x = self.fc3(x)
        x = F.relu6(x)
        
        # Layer 4
        x = self.fc4(x)
        x = F.relu6(x)
        
        # Layer 5
        x = self.fc5(x)
        x = F.relu6(x)
        
        # Classifier
        if self.classifier:
            x = self.classifier(x)
        
        # 物理含义统一：输出 LogSoftmax
        # 即使训练时没改，推理加上这个也不影响 Argmax 的结果
        return F.log_softmax(x, dim=1)

# ==========================================
# 3. 主程序：加载数据并推理
# ==========================================

def main():
    device = torch.device("cpu") # 推理通常在 CPU 验证 (因为涉及模拟 Int 操作)
    batch_size = 128
    
    # 1. 准备数据 (与训练一致)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
    
    data_root = "./data"
    test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. 加载量化模型
    export_file = "./quantized_export/split_model_quantized_8bit.pt"
    if not os.path.exists(export_file):
        print(f"Error: Export file not found at {export_file}")
        return

    try:
        model = ClientQuantizedModel(export_file).to(device)
    except KeyError as e:
        print(f"Error loading model keys: {e}")
        print("Tip: Make sure your export script included all layers.")
        return

    model.eval()
    
    # 3. 执行推理
    print(f"\nStarting Inference on MNIST Test Set ({len(test_dataset)} samples)...")
    print("Mode: Simulated Quantization (Input Quant -> Int Math -> Dequant -> Float Act)")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass (Quantized)
            outputs = model(images)
            
            # 统计准确率
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 打印部分 Log 以验证 (只打印第一个 Batch)
            if total == labels.size(0):
                print(f"Sample Output (LogProbs): {outputs[0].numpy()}")
                print(f"Sample Prediction: {preds[0].item()}, Label: {labels[0].item()}")

    acc = 100.0 * correct / total
    print(f"\n========================================")
    print(f"Final Quantized Inference Accuracy: {acc:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()