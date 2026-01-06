import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 导入你之前的定义好的模块
# 请确保 split_inference_integer.py 和 split_comm_utils.py 在同一目录下
from split_comm_utils import BPSKModem, Int8Codec, Float32Codec
from split_inference_integer import ClientInference, ServerInference

# ==========================================
# 配置区域：功率分配策略列表
# 依据图片 709205336072997f26b893bc1f45a4ec.jpg 转录
# ==========================================
POWER_PROFILES_LIST = [
    [1.0, 1.0, 1.0, 1.0],  # Baseline
    [2.0, 1.8, 0.1, 0.1],
    [2.1, 1.7, 0.1, 0.1],
    [2.2, 1.6, 0.1, 0.1],
    [2.3, 1.5, 0.1, 0.1],
    [2.4, 1.4, 0.1, 0.1],
    [2.5, 1.3, 0.1, 0.1],
    [2.6, 1.2, 0.1, 0.1],
    [2.7, 1.1, 0.1, 0.1],
    [2.8, 1.0, 0.1, 0.1],
    [2.9, 0.9, 0.1, 0.1],
    [3.0, 0.8, 0.1, 0.1],
    [3.1, 0.7, 0.1, 0.1],
    [3.2, 0.6, 0.1, 0.1],
    [3.3, 0.5, 0.1, 0.1],
    [3.4, 0.4, 0.1, 0.1],
    [3.5, 0.3, 0.1, 0.1],
    [3.6, 0.2, 0.1, 0.1],
    [3.7, 0.1, 0.1, 0.1]  # Extreme case
]

def run_inference(client_model, server_model, test_loader, device, power_profile, ebno=0.0):
    """
    对单个功率分布策略执行完整的推理循环
    """
    num_carriers = 4
    # 初始化调制解调器
    modem = BPSKModem(ebno_db=ebno, num_carriers=num_carriers, power_profile=power_profile)
    
    correct = 0
    total = 0
    
    # 误码率统计变量
    total_channel_errors = torch.zeros(num_carriers, device=device)
    total_transmitted_bits = 0
    
    client_model.eval()
    server_model.eval()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # --- 1. Client Forward ---
            client_output = client_model(images)
            val_min = client_output.min().item()
            val_max = client_output.max().item()
            
            # --- 2. Communication Pipeline ---
            # Quantization
            tx_bits_data, scale, zp = Int8Codec.float_to_bits(client_output, val_min, val_max, num_bits=8)
            
            # Serialize Parameters
            scale_tensor = torch.tensor([scale], device=device)
            scale_bits = Float32Codec.float_to_bits(scale_tensor).flatten()
            zp_tensor = torch.tensor([zp], device=device)
            zp_bits = Int8Codec.int_to_bits(zp_tensor, num_bits=8).flatten()
            
            # Reshape & Permute for Parallel Transmission
            # Data: [Batch, Dim, 8] -> [Time, 4]
            data_stream = tx_bits_data.view(-1, 4, 2).permute(0, 2, 1).reshape(-1, 4)
            scale_stream = scale_bits.view(4, 8).t()
            zp_stream = zp_bits.view(4, 2).t()
            
            # Concatenate Frame
            tx_frame = torch.cat([data_stream, scale_stream, zp_stream], dim=0)
            
            # Channel Simulation
            tx_symbols = modem.modulate(tx_frame)
            rx_noisy = modem.add_noise(tx_symbols)
            rx_frame = modem.demodulate(rx_noisy)
            
            # BER Calculation
            bit_errors = (tx_frame != rx_frame).sum(dim=0).float()
            total_channel_errors += bit_errors
            total_transmitted_bits += tx_frame.size(0)
            
            # Unpacking & Dequantization
            len_data = data_stream.size(0); len_scale = 8
            rx_data_stream = rx_frame[:len_data]
            rx_scale_stream = rx_frame[len_data : len_data + len_scale]
            rx_zp_stream = rx_frame[len_data + len_scale :]
            
            rx_scale = Float32Codec.bits_to_float(rx_scale_stream.t().flatten()).item()
            rx_zp = Int8Codec.bits_to_int(rx_zp_stream.t().flatten()).item()
            
            rx_data_bits_flat = rx_data_stream.view(-1, 2, 4).permute(0, 2, 1).flatten(1)
            rx_data_bits = rx_data_bits_flat.view_as(tx_bits_data)
            server_input = Int8Codec.bits_to_float(rx_data_bits, rx_scale, rx_zp, num_bits=8)
            
            # --- 3. Server Forward ---
            final_output = server_model(server_input)
            preds = final_output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    channel_bers = (total_channel_errors / total_transmitted_bits).cpu().numpy()
    
    return accuracy, channel_bers

def main():
    # --- 初始化设置 ---
    device = torch.device("cpu") # 模拟通信通常用CPU
    BATCH_SIZE = 128
    EBNO_DB = 0.0
    EXPORT_FILE = "./current_export/split_model_quantized_8bit.pt"
    EXCEL_FILENAME = "power_allocation_results_0.1_0.1.xlsx"
    PLOT_FILENAME = "power_allocation_plot_0.1_0.1.png"

    print("=== 初始化数据与模型 ===")
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.Grayscale(1), transforms.ToTensor(),
        transforms.Normalize([0.1307], [0.3081])
    ])
    test_ds = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 模型加载
    if not os.path.exists(EXPORT_FILE):
        print(f"Error: {EXPORT_FILE} not found.")
        return
    all_params = torch.load(EXPORT_FILE, map_location="cpu")
    client_model = ClientInference(all_params['client']).to(device)
    server_model = ServerInference(all_params['server']).to(device)

    # 结果存储列表
    results_data = []

    print(f"\n=== 开始批量测试 (EbNo={EBNO_DB} dB) ===")
    print(f"{'Profile':<30} | {'Ratio':<6} | {'Acc(%)':<7} | {'BER C0':<8} | {'BER C1':<8} | {'BER C2':<8} | {'BER C3':<8}")
    print("-" * 115) # 延长分割线

    # --- 循环遍历策略 ---
    for profile in POWER_PROFILES_LIST:
        # 1. 计算 P0/P1 比值 (MSB / Next-MSB)
        # 防止除零 (虽然输入里没有0，加epsilon保险)
        p0 = profile[0]
        p1 = profile[1]
        ratio = p0 / (p1 if p1 > 1e-6 else 1e-6)

        # 2. 执行推理
        acc, bers = run_inference(client_model, server_model, test_loader, device, profile, ebno=EBNO_DB)
        
        # 3. 记录数据
        row = {
            "Power Profile": str(profile),
            "P0 (High Bits)": p0,
            "P1 (Next High)": p1,
            "P2": profile[2],
            "P3 (Low Bits)": profile[3],
            "Ratio (P0/P1)": ratio,
            "Accuracy (%)": acc,
            "BER Carrier 0 (MSB)": bers[0],
            "BER Carrier 1": bers[1],
            "BER Carrier 2": bers[2],
            "BER Carrier 3 (LSB)": bers[3]
        }
        results_data.append(row)
        
        # 实时打印简报
        print(f"{str(profile):<30} | {ratio:<6.2f} | {acc:<7.2f} | {bers[0]:.4f}   | {bers[1]:.4f}   | {bers[2]:.4f}   | {bers[3]:.4f}")

    # --- 保存 Excel ---
    df = pd.DataFrame(results_data)
    df.to_excel(EXCEL_FILENAME, index=False)
    print(f"\n[Success] 结果已保存至: {os.path.abspath(EXCEL_FILENAME)}")

    # --- 绘图 ---
    print("正在生成图表...")
    plt.figure(figsize=(10, 6))
    
    # 提取绘图数据
    x_ratios = df["Ratio (P0/P1)"]
    y_accs = df["Accuracy (%)"]
    
    # 绘制带连线的散点图
    plt.plot(x_ratios, y_accs, linestyle='-', color='gray', alpha=0.5)
    plt.scatter(x_ratios, y_accs, c='blue', s=80, zorder=5)
    
    # 标注每个点的 P0 值
    for i, txt in enumerate(df["P0 (High Bits)"]):
        plt.annotate(f"P0={txt}", (x_ratios[i], y_accs[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.title(f'Accuracy vs. Power Ratio (MSB/Next-MSB) at EbNo={EBNO_DB}dB(p2=p3=0.5)')
    plt.xlabel('Power Ratio (Carrier 0 / Carrier 1)')
    plt.ylabel('Classification Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME)
    print(f"[Success] 图表已保存至: {os.path.abspath(PLOT_FILENAME)}")
    plt.show()

if __name__ == "__main__":
    main()