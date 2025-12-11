import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from split_comm_utils import BPSKModem, Int8Codec
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
        # # 辅助函数：从参数字典中提取 input_stats
        # def get_stats(layer_key, default_min=0.0, default_max=6.0):
        #     # 尝试查找对应的激活统计值
        #     # 导出时的 key 可能是 "client_layers/layers.1_folded"
        #     # 对应的激活 key 通常是 "client_layers/layers.1_folded" (如果是 folded) 或者加 _out
        #     # 你的导出代码里写的是：layers_quant[f"{full_name}_folded"].update({...})
        #     # 所以直接读取 layer_params[key] 里的 act_min/act_max 即可
            
        #     layer_data = client_params.get(layer_key, {})
        #     return {
        #         'act_min': layer_data.get('act_min', default_min),
        #         'act_max': layer_data.get('act_max', default_max)
        #     }

        # # 注意：这里我们使用上一层的输出统计作为当前层的输入统计
        # # 对于第一层，输入是图像，范围是 [-0.5, 3.0] (手动指定)
        # self.fc1 = IntegerLinear(
        #     self.params['client_layers/layers.1_folded'], 
        #     input_stats={'act_min': -0.5, 'act_max': 3.0}
        # )
        
        # # 对于后续层，使用参数中保存的统计值
        # self.fc2 = IntegerLinear(
        #     self.params['client_layers/layers.5_folded'], 
        #     input_stats=get_stats('client_layers/layers.3_out')
        # )
        # self.fc3 = IntegerLinear(
        #     self.params['client_layers/layers.9_folded'], 
        #     input_stats=get_stats('client_layers/layers.7_out')
        # )
        # self.fc4 = IntegerLinear(
        #     self.params['client_layers/layers.13_folded'], 
        #     input_stats=get_stats('client_layers/layers.11_out')
        # )
        # self.fc5 = IntegerLinear(
        #     self.params['client_layers/layers.17_folded'], 
        #     input_stats=get_stats('client_layers/layers.15_out')
        # )
        self.RELU_STATS = {'act_min': 0.0, 'act_max': 6.0}

        self.fc1 = IntegerLinear(self.params['client_layers/layers.1_folded'], 
                                 input_stats={'act_min': -0.5, 'act_max': 3.0})
        self.fc2 = IntegerLinear(self.params['client_layers/layers.5_folded'], input_stats=self.RELU_STATS)
        self.fc3 = IntegerLinear(self.params['client_layers/layers.9_folded'], input_stats=self.RELU_STATS)
        self.fc4 = IntegerLinear(self.params['client_layers/layers.13_folded'], input_stats=self.RELU_STATS)
        self.fc5 = IntegerLinear(self.params['client_layers/layers.17_folded'], input_stats=self.RELU_STATS)

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
        # def get_stats(layer_key, default_min=0.0, default_max=6.0):
        #     layer_data = server_params.get(layer_key, {})
        #     return {
        #         'act_min': layer_data.get('act_min', default_min),
        #         'act_max': layer_data.get('act_max', default_max)
        #     }

        # # Server 第一层输入来自 Client，范围是 ReLU6 [0, 6]
        # # 或者使用通信信道的量化范围
        # self.fc1 = IntegerLinear(
        #     self.params['server_layers/layers.0_folded'], 
        #     input_stats={'act_min': 0.0, 'act_max': 6.0} 
        # )
        
        # self.fc2 = IntegerLinear(
        #     self.params['server_layers/layers.4_folded'], 
        #     input_stats=get_stats('server_layers/layers.2_out')
        # )
        # self.fc3 = IntegerLinear(
        #     self.params['server_layers/layers.8_folded'], 
        #     input_stats=get_stats('server_layers/layers.6_out')
        # )
        # self.fc4 = IntegerLinear(
        #     self.params['server_layers/layers.12_folded'], 
        #     input_stats=get_stats('server_layers/layers.10_out')
        # )
        # # 最后一层通常没有 ReLU，stats 可能不同
        # self.fc5 = IntegerLinear(
        #     self.params['server_layers/layers.15'], 
        #     input_stats=get_stats('server_layers/layers.14_out', default_min=-10.0, default_max=10.0)
        # )
        self.RELU_STATS = {'act_min': 0.0, 'act_max': 6.0}

        self.fc1 = IntegerLinear(self.params['server_layers/layers.0_folded'], input_stats=self.RELU_STATS)
        self.fc2 = IntegerLinear(self.params['server_layers/layers.4_folded'], input_stats=self.RELU_STATS)
        self.fc3 = IntegerLinear(self.params['server_layers/layers.8_folded'], input_stats=self.RELU_STATS)
        self.fc4 = IntegerLinear(self.params['server_layers/layers.12_folded'], input_stats=self.RELU_STATS)
        self.fc5 = IntegerLinear(self.params['server_layers/layers.15'], input_stats=self.RELU_STATS)

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
    
    print(f"\n[True Integer Inference] Starting...")
    print("Mode: Int8 Weights, Int32 Accumulation")
    
    correct = 0
    total = 0
    modem = BPSKModem(ebno_db=20.0) # 推理时 SNR 可能不同
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # 1. Client 推理
            client_output = client_model(images)
           
            # 2. 通信过程 (逐步调用)
            # 定义范围 (ReLU6)
            val_min, val_max = 0.0, 6.0
            
            # Step 1: 量化转比特
            # 此时 client_output 是模拟 float，我们将其转为 bit stream
            tx_bits, scale, zp = Int8Codec.float_to_bits(client_output, val_min, val_max, num_bits=8)

            # Step 2: 调制
            tx_symbols = modem.modulate(tx_bits)

            # Step 3: 加噪
            rx_noisy = modem.add_noise(tx_symbols)
            
            # Step 4: 解调
            rx_bits = modem.demodulate(rx_noisy)
            
            # Step 5: 反量化
            # 恢复成浮点数传给 Server (模拟解码后的数据)
            server_input = Int8Codec.bits_to_float(rx_bits, scale, zp, num_bits=8)

            # 3. Server 推理
            final_output = server_model(server_input)
            # # Client Part
            # client_output = client_model(images)
            
            # # Transmission (Simulate)
            # transmitted_data = client_output.detach().clone()
            
            # # Server Part
            # final_output = server_model(transmitted_data)
            
            # Stats
            preds = final_output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if total == labels.size(0):
                 print(f"First Batch Preds: {preds[:5].tolist()}")

    acc = 100.0 * correct / total
    print(f"\n========================================")
    print(f"Final Integer Accuracy: {acc:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()