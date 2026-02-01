import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt  # [新增] 绘图库
import numpy as np               # [新增] 数值处理
import random  # [新增] 用于生成随机索引
import pandas as pd  # [新增] 用于导出 Excel
from split_comm_utils import QAM4Modem, Int8Codec, Float32Codec, MultiStreamManager
# from split_comm_utils import QAM4Modem, Int8Codec, Float32Codec
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
        # acc_int = torch.matmul(x_shifted_int, w_int.t())
        # [修改后] 兼容 GPU 的写法
        if x_shifted_int.is_cuda:
            # GPU: 转为 float 进行乘法 (模拟)，再转回 int
            # 注意：对于 MNIST 这种小规模数据，float32 精度足够覆盖 int32 的累加范围
            acc_int = torch.matmul(x_shifted_int.float(), w_int.float().t()).int()
        else:
            # CPU: 支持真正的 int32 乘法
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
# 4.统计误比特率
# ==========================================
def calculate_channel_bers(tx_bits, rx_bits):
    """
    根据 Bit-Plane Slicing 逻辑计算每个通道的 BER
    tx_bits, rx_bits: [Batch, Dim, 8]
    Return: Tensor [4] 代表 4 个 Carrier 的错误数
    """
    # 展平为 [Total_Pixels, 8]
    t = tx_bits.view(-1, 8)
    r = rx_bits.view(-1, 8)
    
    diff = (t != r).float()
    
    # Carrier 0: Bits 0, 1 (MSB, indices 0-1)
    # Carrier 1: Bits 2, 3
    # Carrier 2: Bits 4, 5
    # Carrier 3: Bits 6, 7 (LSB, indices 6-7)
    errors = torch.zeros(4, device=tx_bits.device)
    for i in range(4):
        # 提取对应通道的 2 个 bit 列
        errors[i] = diff[:, i*2 : (i+1)*2].sum()
        
    return errors, t.size(0) * 2  # 返回错误数和该通道总传输比特数

# ==========================================
# 推理主函数
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    BATCH_SIZE = 128
    
    # === 数据加载 (保持不变) ===
    transform = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.Grayscale(1), transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
    test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # === 加载模型 (保持不变) ===
    export_file = "./current_export/split_model_quantized_8bit.pt"
    if not os.path.exists(export_file):
        print("错误：找不到量化参数文件。")
        return
    try:
        all_params = torch.load(export_file, map_location=device, weights_only=False)
    except TypeError:
        all_params = torch.load(export_file, map_location=device)
    
    client_model = ClientInference(all_params['client']).to(device)
    server_model = ServerInference(all_params['server']).to(device)

    # ================= 实验配置区域 =================
    NUM_RUNS = 1    # 每个配置跑多少次求平均
    SNR_DB = 0.0    # 固定信噪比
    
    # 定义要测试的功率分配方案字典 {名称: 配置}
    # Carrier 0 (MSB) -> Carrier 3 (LSB)
    power_profiles_dict = {
        "Uniform":  [1.0, 1.0, 1.0, 1.0],           # 均等分配
        "1":        [1.6, 1.2, 0.8, 0.4],           # 线性递减
        "2":        [2.0, 1.4, 0.4, 0.2],           # 线性递减
        "3":        [2.3, 1.5, 0.1, 0.1],           # 线性递减
        "4":        [2.6, 1.2, 0.1, 0.1],           # 线性递减
        "5":        [2.9, 0.9, 0.1, 0.1],           # 线性递减
        "6":        [3.2, 0.6, 0.1, 0.1],           # 线性递减
        "Aggressive":    [2.6, 1.3, 0.1, 0.0],           # 你的激进策略
        "Only_MSB":      [3.9, 0.1, 0.0, 0.0],           # 极端保高位
    }

    results_data = [] # 用于存储最终结果
    
    print(f"\n[Experiment Config] SNR: {SNR_DB} dB, Runs per profile: {NUM_RUNS}")

    # ================= 开始遍历各种配置 =================
    for name, profile in power_profiles_dict.items():
        print(f"\n>>> Testing Profile: {name} | Allocation: {profile}")
        
        # 初始化通信模块
        modem = QAM4Modem(snr_db=SNR_DB, num_carriers=4, power_profile=profile)
        stream_manager = MultiStreamManager(device, N=512, K_info=256, crc_len=16)
        
        profile_acc_history = []
        profile_ber_history = torch.zeros(4, device=device) # 累积 BER
        
        for run_i in range(NUM_RUNS):
            client_model.eval()
            server_model.eval()
            correct = 0
            total_samples = 0
            
            # 累积该 Run 的错误比特数和总比特数
            run_bit_errors = torch.zeros(4, device=device)
            run_total_bits = 0
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device), labels.to(device)
                    
                    # 1. Client 推理 & 量化
                    client_out = client_model(images)
                    v_min, v_max = client_out.min().item(), client_out.max().item()
                    tx_bits_raw, scale, zp = Int8Codec.float_to_bits(client_out, v_min, v_max)
                    
                    # 展平准备传输
                    bits_matrix = tx_bits_raw.view(-1, 8)
                    
                    # 2. 通信过程 (编码 -> 调制 -> 信道 -> 解调 -> 解码)
                    tx_frame, info_lens = stream_manager.pack_and_encode(bits_matrix, scale, zp)
                    tx_syms = modem.modulate(tx_frame)
                    rx_noisy = modem.add_noise(tx_syms)
                    rx_bits_matrix, rx_scale, rx_zp = stream_manager.decode_and_unpack(rx_noisy, info_lens, SNR_DB)
                    
                    # 3. Server 推理 & 准确率统计
                    rx_bits_reshaped = rx_bits_matrix.view_as(tx_bits_raw)
                    server_in = Int8Codec.bits_to_float(rx_bits_reshaped, rx_scale, rx_zp)
                    
                    final_out = server_model(server_in)
                    pred = final_out.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total_samples += labels.size(0)
                    
                    # 4. [新增] 统计 BER (对比 tx_bits_raw 和 rx_bits_reshaped)
                    batch_errs, batch_bits = calculate_channel_bers(tx_bits_raw, rx_bits_reshaped)
                    run_bit_errors += batch_errs
                    run_total_bits += batch_bits
                    
                    if batch_idx % 10 == 0:
                        print(f"   Batch {batch_idx} processed...", end='\r')

            # 计算本轮 Run 的指标
            run_acc = 100.0 * correct / total_samples
            run_bers = (run_bit_errors / run_total_bits).cpu().numpy()
            
            profile_acc_history.append(run_acc)
            # 累加 BER 用于最后求平均
            profile_ber_history += (run_bit_errors / run_total_bits)
            
            print(f"   Run {run_i+1}: Acc={run_acc:.2f}%, BERs={run_bers}")

        # === 统计该 Profile 的平均结果 ===
        avg_acc = np.mean(profile_acc_history)
        avg_bers = (profile_ber_history / NUM_RUNS).cpu().numpy()
        
        print(f"   [Result] {name}: Avg Acc={avg_acc:.2f}%")
        
        # 添加到结果列表
        results_data.append({
            "Profile Name": name,
            "Power Allocation": str(profile),
            "Average Accuracy (%)": avg_acc,
            "BER Carrier 0 (MSB)": avg_bers[0],
            "BER Carrier 1": avg_bers[1],
            "BER Carrier 2": avg_bers[2],
            "BER Carrier 3 (LSB)": avg_bers[3]
        })

    # ================= 导出 Excel =================
    df = pd.DataFrame(results_data)
    output_filename = "power_allocation_results.xlsx"
    df.to_excel(output_filename, index=False)
    print(f"\n\n所有测试完成！结果已保存至: {output_filename}")
    print(df)
# def main():
#     # device = torch.device("cpu")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Running on device: {device}")
#     BATCH_SIZE = 128
    
#     # 数据加载
#     transform = transforms.Compose([
#         transforms.Resize(64), transforms.CenterCrop(64),
#         transforms.Grayscale(1), transforms.ToTensor(),
#         transforms.Normalize([0.1307], [0.3081])
#     ])
#     test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
#     # 加载模型
#     export_file = "./current_export/split_model_quantized_8bit.pt"
#     if not os.path.exists(export_file):
#         print("错误：找不到量化参数文件。")
#         return
#     # all_params = torch.load(export_file, map_location="cpu")
#     try:
#         # 尝试加载并直接映射到 GPU (或 CPU)
#         all_params = torch.load(export_file, map_location=device, weights_only=False)
#     except TypeError:
#         all_params = torch.load(export_file, map_location=device)
    
#     # 实例化模型
#     client_model = ClientInference(all_params['client']).to(device)
#     server_model = ServerInference(all_params['server']).to(device)

#     # ================= 配置区域 =================
#     NUM_RUNS = 1    # 跑 5 次求平均
#     SNR_DB = 0.0    # 信噪比
    
#     # 功率分配: Carrier 0 功率最高 (保护 MSB)
#     # 对应: [MSB(7-6), Bits(5-4), Bits(3-2), LSB(1-0)]
#     POWER_PROFILE = [2.6, 1.3, 0.1, 0.0]
#     # POWER_PROFILE = [1.0, 1.0, 1.0, 1.0]

#     # 初始化通信模块
#     # 注意: Polar 码后无需重复编码，QAM4Modem 只做调制
#     modem = QAM4Modem(snr_db=SNR_DB, num_carriers=4, power_profile=POWER_PROFILE)
    
#     # 初始化流管理器 (N=512, K=256 代表 1/2 码率，可根据信道调整)
#     stream_manager = MultiStreamManager(device, N=512, K_info=256, crc_len=16)

#     print(f"\n[System Config]")
#     print(f"  > Scheme: Bit-Plane Slicing + Polar Codes (N=512, K=256)")
#     print(f"  > Power Profile: {POWER_PROFILE} (Carrier 0 protects MSB)")
#     print(f"  > SNR: {SNR_DB} dB")
#     print(f"  > Runs: {NUM_RUNS} for averaging")
#     print("-" * 50)

#     acc_history = []

#     # === 5 次循环主逻辑 ===
#     for run_i in range(NUM_RUNS):
#         client_model.eval()
#         server_model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for batch_idx, (images, labels) in enumerate(test_loader):
#                 images, labels = images.to(device), labels.to(device)
#                 print(batch_idx)
#                 # 1. Client 推理
#                 client_out = client_model(images)
#                 v_min, v_max = client_out.min().item(), client_out.max().item()
                
#                 # 2. 量化 (Float -> Bits)
#                 # bits shape: [Batch, Dim, 8]
#                 tx_bits_raw, scale, zp = Int8Codec.float_to_bits(client_out, v_min, v_max)
                
#                 # 展平为 [Total_Pixels, 8] 以便流管理器切割
#                 bits_matrix = tx_bits_raw.view(-1, 8)

#                 # [新增] 打印数据量
#                 if batch_idx == 0:
#                     print(f"DEBUG: Batch Size: {BATCH_SIZE}")
#                     print(f"DEBUG: Client Output Shape: {client_out.shape}")
#                     print(f"DEBUG: Total Bits to Encode: {bits_matrix.numel()}")
#                     estimated_blocks = bits_matrix.numel() / (256 * 4) # 粗略估算
#                     print(f"DEBUG: Approx Polar Blocks to Decode: {int(estimated_blocks)}")
                
#                 # 3. 封装与编码 (Slice -> Add Scale/ZP -> Encode -> Stack)
#                 # tx_frame: [Time, 4]
#                 tx_frame, info_lens = stream_manager.pack_and_encode(bits_matrix, scale, zp)
                
#                 # 4. 信道传输
#                 tx_syms = modem.modulate(tx_frame)
#                 rx_noisy = modem.add_noise(tx_syms)
                
#                 # 5. 解码与重组 (Unstack -> Decode -> Extract Scale/ZP -> Merge)
#                 rx_bits_matrix, rx_scale, rx_zp = stream_manager.decode_and_unpack(rx_noisy, info_lens, SNR_DB)
                
#                 # 6. 恢复形状与反量化
#                 rx_bits_reshaped = rx_bits_matrix.view_as(tx_bits_raw)
#                 server_in = Int8Codec.bits_to_float(rx_bits_reshaped, rx_scale, rx_zp)
                
#                 # 7. Server 推理
#                 final_out = server_model(server_in)
#                 pred = final_out.argmax(dim=1)
#                 correct += (pred == labels).sum().item()
#                 total += labels.size(0)
                
#         acc = 100.0 * correct / total
#         acc_history.append(acc)
#         print(f"Run {run_i+1}: Accuracy = {acc:.2f}%")

#     # === 结果统计 ===
#     avg_acc = np.mean(acc_history)
#     print("=" * 30)
#     print(f"Final Average Accuracy: {avg_acc:.2f}%")
#     print(f"History: {acc_history}")
#     print("=" * 30)

if __name__ == "__main__":
    main()