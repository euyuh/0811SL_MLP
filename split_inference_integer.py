import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from split_comm_utils import BPSKModem, Int8Codec, Float32Codec
# ==========================================
# 1. 纯整数运算核心 (True Integer Arithmetic)
# ==========================================
class IntegerLinear(nn.Module):
    """
    实现真正的整数推理：
    Weights: int8
    Input: int32 (from quantized input)
    Compute: int32 matmul
    """
    def __init__(self, layer_params, input_stats=None, num_bits=8):
        super().__init__()
        
        # [修改] 1. 权重转换为真正的 int8 存储
        # 这里的 w_q 在加载时虽然是 Tensor，但我们要确保它是 int8 类型
        self.w_q = layer_params['w_q'].to(torch.int8)
        
        # 偏置保持 float (为了保持精度，模拟中通常偏置作为后处理)
        self.b_q = layer_params['b_q']
        self.b_scale = layer_params['b_scale']
        if self.b_q is not None and self.b_scale is not None:
             self.bias_float = self.b_q.float() * self.b_scale
        else:
             self.bias_float = None

        self.w_scale = layer_params['w_scale']
        
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
        # -------------------------------------------------------
        # Step 1: Input Quantization (Float -> Int32)
        # -------------------------------------------------------
        # 先计算出整数值
        x_int_val = torch.round(x_float / self.in_scale + self.in_zp)
        x_q = torch.clamp(x_int_val, 0, 255)
        
        # [关键] 强制转换为 int32 类型，确保后续计算是整数运算
        x_q_int = x_q.int()

        # -------------------------------------------------------
        # Step 2: Integer Matrix Multiplication (纯整数运算)
        # -------------------------------------------------------
        # 移除 ZeroPoint (整数减法)
        x_shifted_int = x_q_int - int(self.in_zp)
        
        # 准备权重：转换为 int32 以便与输入进行乘法 (避免 int8 溢出)
        # 注意：w_q 存储的是 int8，计算时提升为 int32 是硬件标准做法
        w_int = self.w_q.int()
        
        # [关键] 使用 torch.matmul 进行整数矩阵乘法
        # PyTorch 的 F.linear 对 int 输入支持有限，matmul 更通用
        # Input: [Batch, In], Weight.T: [In, Out] -> [Batch, Out]
        acc_int = torch.matmul(x_shifted_int, w_int.t())
             
        # -------------------------------------------------------
        # Step 3: Dequantize (Int32 -> Float)
        # -------------------------------------------------------
        # 将整数累加器转回 float
        acc_float = acc_int.float()
        
        # 乘上总缩放因子 (Input Scale * Weight Scale)
        out = acc_float * (self.in_scale * self.w_scale)
        
        # 加上浮点偏置
        if self.bias_float is not None:
            out += self.bias_float
            
        return out

# ==========================================
# 2. 客户端模型 (Client Model)
# ==========================================
class ClientInference(nn.Module):
    def __init__(self, client_params):
        super().__init__()
        self.params = client_params
        # 辅助函数：从参数字典中提取 input_stats
        def get_stats(layer_key, default_min=0.0, default_max=6.0):
            # 尝试查找对应的激活统计值
            # 导出时的 key 可能是 "client_layers/layers.1_folded"
            # 对应的激活 key 通常是 "client_layers/layers.1_folded" (如果是 folded) 或者加 _out
            # 你的导出代码里写的是：layers_quant[f"{full_name}_folded"].update({...})
            # 所以直接读取 layer_params[key] 里的 act_min/act_max 即可
            
            layer_data = client_params.get(layer_key, {})
            return {
                'act_min': layer_data.get('act_min', default_min),
                'act_max': layer_data.get('act_max', default_max)
            }

        # 注意：这里我们使用上一层的输出统计作为当前层的输入统计
        # 对于第一层，输入是图像，范围是 [-0.5, 3.0] (手动指定)
        self.fc1 = IntegerLinear(
            self.params['client_layers/layers.1_folded'], 
            input_stats={'act_min': -0.5, 'act_max': 3.0}
        )
        
        # 对于后续层，使用参数中保存的统计值
        self.fc2 = IntegerLinear(
            self.params['client_layers/layers.5_folded'], 
            input_stats=get_stats('client_layers/layers.3_out')
        )
        self.fc3 = IntegerLinear(
            self.params['client_layers/layers.9_folded'], 
            input_stats=get_stats('client_layers/layers.7_out')
        )
        self.fc4 = IntegerLinear(
            self.params['client_layers/layers.13_folded'], 
            input_stats=get_stats('client_layers/layers.11_out')
        )
        self.fc5 = IntegerLinear(
            self.params['client_layers/layers.17_folded'], 
            input_stats=get_stats('client_layers/layers.15_out')
        )
        # self.RELU_STATS = {'act_min': 0.0, 'act_max': 6.0}

        # self.fc1 = IntegerLinear(self.params['client_layers/layers.1_folded'], 
        #                          input_stats={'act_min': -0.5, 'act_max': 3.0})
        # self.fc2 = IntegerLinear(self.params['client_layers/layers.5_folded'], input_stats=self.RELU_STATS)
        # self.fc3 = IntegerLinear(self.params['client_layers/layers.9_folded'], input_stats=self.RELU_STATS)
        # self.fc4 = IntegerLinear(self.params['client_layers/layers.13_folded'], input_stats=self.RELU_STATS)
        # self.fc5 = IntegerLinear(self.params['client_layers/layers.17_folded'], input_stats=self.RELU_STATS)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu6(self.fc1(x))
        x = F.relu6(self.fc2(x))
        x = F.relu6(self.fc3(x))
        x = F.relu6(self.fc4(x))
        x = F.relu6(self.fc5(x)) 
        return x

# ==========================================
# 3. 服务器端模型 (Server Model)
# ==========================================
class ServerInference(nn.Module):
    def __init__(self, server_params):
        super().__init__()
        self.params = server_params
        def get_stats(layer_key, default_min=0.0, default_max=6.0):
            layer_data = server_params.get(layer_key, {})
            return {
                'act_min': layer_data.get('act_min', default_min),
                'act_max': layer_data.get('act_max', default_max)
            }

        # Server 第一层输入来自 Client，范围是 ReLU6 [0, 6]
        # 或者使用通信信道的量化范围
        self.fc1 = IntegerLinear(
            self.params['server_layers/layers.0_folded'], 
            input_stats={'act_min': 0.0, 'act_max': 6.0} 
        )
        
        self.fc2 = IntegerLinear(
            self.params['server_layers/layers.4_folded'], 
            input_stats=get_stats('server_layers/layers.2_out')
        )
        self.fc3 = IntegerLinear(
            self.params['server_layers/layers.8_folded'], 
            input_stats=get_stats('server_layers/layers.6_out')
        )
        self.fc4 = IntegerLinear(
            self.params['server_layers/layers.12_folded'], 
            input_stats=get_stats('server_layers/layers.10_out')
        )
        # 最后一层通常没有 ReLU，stats 可能不同
        self.fc5 = IntegerLinear(
            self.params['server_layers/layers.15'], 
            input_stats=get_stats('server_layers/layers.14_out', default_min=-10.0, default_max=10.0)
        )
        # self.RELU_STATS = {'act_min': 0.0, 'act_max': 6.0}

        # self.fc1 = IntegerLinear(self.params['server_layers/layers.0_folded'], input_stats=self.RELU_STATS)
        # self.fc2 = IntegerLinear(self.params['server_layers/layers.4_folded'], input_stats=self.RELU_STATS)
        # self.fc3 = IntegerLinear(self.params['server_layers/layers.8_folded'], input_stats=self.RELU_STATS)
        # self.fc4 = IntegerLinear(self.params['server_layers/layers.12_folded'], input_stats=self.RELU_STATS)
        # self.fc5 = IntegerLinear(self.params['server_layers/layers.15'], input_stats=self.RELU_STATS)

    def forward(self, x):
        x = F.relu6(self.fc1(x))
        x = F.relu6(self.fc2(x))
        x = F.relu6(self.fc3(x))
        x = F.relu6(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    device = torch.device("cpu") # 整数运算通常在 CPU 模拟
    BATCH_SIZE = 128
    transform = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.Grayscale(1), transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
    test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    export_file = "./current_export/split_model_quantized_8bit.pt"
    if not os.path.exists(export_file):
        print("错误：找不到量化参数文件。")
        return
    
    print(f"Loading parameters from: {export_file}")
    # 强制 map_location 到 CPU，因为我们要进行 CPU 整数运算
    all_params = torch.load(export_file, map_location="cpu")
    
    client_model = ClientInference(all_params['client']).to(device)
    server_model = ServerInference(all_params['server']).to(device)

    client_model.eval()
    server_model.eval()
    
    print(f"\n[Parallel Integer Inference] Starting...")
    print("Mode: Int8 Weights, Int32 Accumulation")
    print("Mode: 4-Carrier Parallel Transmission")

    # ================= 配置区域 =================
    NUM_CARRIERS = 4
    # 可调整的功率分配 (例如: 增强高位所在的载波?)
    # 这里的分配顺序对应: Carrier0(HighBits), Carrier1, ..., Carrier3(LowBits)
    # POWER_PROFILE = [1.0, 1.0, 1.0, 1.0] 
    POWER_PROFILE = [2.0, 1.2, 0.5, 0.3] # 示例：不均匀分配
    # ===========================================
    
    modem = BPSKModem(ebno_db=0.0, num_carriers=NUM_CARRIERS, power_profile=POWER_PROFILE)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Client 推理
            client_output = client_model(images)
            val_min, val_max = 0.0, 6.0
            
            # [Step 1: 生成比特流]
            # Data: [Batch, Dim, 8]
            tx_bits_data, scale, zp = Int8Codec.float_to_bits(client_output, val_min, val_max, num_bits=8)
            # Scale: [32]
            scale_tensor = torch.tensor([scale], device=device)
            scale_bits = Float32Codec.float_to_bits(scale_tensor).flatten()
            # ZP: [8]
            zp_tensor = torch.tensor([zp], device=device)
            zp_bits = Int8Codec.int_to_bits(zp_tensor, num_bits=8).flatten()
            
            # [Step 2: 并行流映射 (Reshape & Permute)]
            # 目标形状: [Time_Steps, 4]
            
            # A. 处理数据 (8 bits -> 4 carriers * 2 time_steps)
            # 原始 bits 顺序是 MSB -> LSB (bit0, bit1, ... bit7)
            # view(..., 4, 2) 将其分为: [b0,b1], [b2,b3], [b4,b5], [b6,b7]
            # 对应载波: C0, C1, C2, C3
            # 我们需要转置为 [2, 4] 以便时间维度在前
            data_stream = tx_bits_data.view(-1, 4, 2).permute(0, 2, 1).reshape(-1, 4)
            
            # B. 处理 Scale (32 bits -> 4 carriers * 8 time_steps)
            scale_stream = scale_bits.view(4, 8).t() # [8, 4]
            
            # C. 处理 ZP (8 bits -> 4 carriers * 2 time_steps)
            zp_stream = zp_bits.view(4, 2).t() # [2, 4]
            
            # [Step 3: 拼接成完整的并行传输帧]
            # 帧结构 (时间轴): [Data ... | Scale (8 steps) | ZP (2 steps)]
            tx_frame = torch.cat([data_stream, scale_stream, zp_stream], dim=0)

            # [Step 4: 物理层传输]
            tx_symbols = modem.modulate(tx_frame)
            rx_noisy = modem.add_noise(tx_symbols)
            rx_frame = modem.demodulate(rx_noisy)
            
            # [Step 5: 拆包 (Slicing)]
            len_data = data_stream.size(0)
            len_scale = 8
            len_zp = 2
            
            rx_data_stream = rx_frame[:len_data]
            rx_scale_stream = rx_frame[len_data : len_data + len_scale]
            rx_zp_stream = rx_frame[len_data + len_scale :]
            
            # [Step 6: 恢复数据结构]
            # A. 恢复 Scale
            # [8, 4] -> t() -> [4, 8] -> flatten -> [32]
            rx_scale_bits = rx_scale_stream.t().flatten()
            rx_scale = Float32Codec.bits_to_float(rx_scale_bits).item()
            
            # B. 恢复 ZP
            # [2, 4] -> t() -> [4, 2] -> flatten -> [8]
            rx_zp_bits = rx_zp_stream.t().flatten()
            rx_zp = Int8Codec.bits_to_int(rx_zp_bits).item()
            
            # C. 恢复 Data
            # [Total_Time, 4] -> reshape(N*Dim, 2, 4) -> permute(0, 2, 1) -> [N*Dim, 4, 2] -> flatten -> [N, Dim, 8]
            rx_data_bits_flat = rx_data_stream.view(-1, 2, 4).permute(0, 2, 1).flatten(1)
            rx_data_bits = rx_data_bits_flat.view_as(tx_bits_data)
            
            # 反量化
            server_input = Int8Codec.bits_to_float(rx_data_bits, rx_scale, rx_zp, num_bits=8)

            # 3. Server 推理
            final_output = server_model(server_input)
            
            preds = final_output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if total == labels.size(0):
                 print(f"First Batch Preds: {preds[:5].tolist()}")
                 print(f"DEBUG: PowerProfile={POWER_PROFILE}, Sent Scale={scale:.4f}, Rx Scale={rx_scale:.4f}")

    acc = 100.0 * correct / total
    print(f"\n========================================")
    print(f"Final Accuracy (4-Carrier): {acc:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()