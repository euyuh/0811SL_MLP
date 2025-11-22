import numpy as np
from scipy.special import erfc

class Modulator:
    def __init__(self, modulation_type, ebno_db, alpha=None):
        """
        调制解调器类
        
        参数:
            modulation_type: 调制类型 ('hqam', '16qam', 'qpsk')
            alpha: HQAM分层调制参数
            ebno_db: Eb/N0值 (dB)
        """
        self.modulation_type = modulation_type
        self.alpha = alpha
        self.ebno_db = ebno_db
        
        # 根据调制类型确定每符号比特数
        if modulation_type == 'qpsk':
            self.bits_per_symbol = 2
        else:  # hqam 或 16qam
            self.bits_per_symbol = 4
        
        # 固定噪声功率谱密度 N0=1
        self.n0 = 1.0
    
    def set_ebno_db(self, ebno_db):
        """设置Eb/N0值 (dB)"""
        self.ebno_db = ebno_db
    
    def set_alpha(self, alpha):
        """设置HQAM的alpha参数"""
        if self.modulation_type == 'hqam':
            self.alpha = alpha
    
    def modulate(self, bit_stream):
        """
        调制函数 - 将二进制比特流调制成复数符号
        
        参数:
            bit_stream: 二进制比特流字符串 (如"00101101")
            
        返回:
            调制后的复数符号流
        """
        # 将比特流转换为整数数组
        bits = np.array([int(bit) for bit in bit_stream])
        
        # 计算线性Eb值 (固定N0=1)
        eb_lin = 10 ** (self.ebno_db / 10)
        
        if self.modulation_type == 'hqam':
            return self._modulate_hqam(bits, eb_lin)
        elif self.modulation_type == '16qam':
            return self._modulate_16qam_std(bits, eb_lin)
        elif self.modulation_type == 'qpsk':
            return self._modulate_qpsk_std(bits, eb_lin)
        else:
            raise ValueError(f"不支持的调制类型: {self.modulation_type}")

    
    def demodulate(self, symbols):
        """
        解调函数 - 将复数符号解调为二进制比特流
        
        参数:
            symbols: 接收到的复数符号流
            
        返回:
            解调后的二进制比特流字符串
        """
        # 计算线性Eb值 (固定N0=1)
        eb_lin = 10 ** (self.ebno_db / 10)
        
        if self.modulation_type == 'hqam':
            bits = self._demodulate_hqam(symbols, eb_lin)
        elif self.modulation_type == '16qam':
            bits = self._demodulate_16qam_std(symbols, eb_lin)
        elif self.modulation_type == 'qpsk':
            bits = self._demodulate_qpsk_std(symbols, eb_lin)
        else:
            raise ValueError(f"不支持的调制类型: {self.modulation_type}")
        
        # 将整数数组转换为二进制比特流字符串
        return ''.join(str(bit) for bit in bits)
    
    def add_noise(self, symbols):
        """
        添加AWGN噪声 (固定N0=1)
        
        参数:
            symbols: 输入复数符号流
            
        返回:
            添加噪声后的复数符号流
        """
        # 固定N0=1，生成复高斯噪声
        noise_std = np.sqrt(self.n0 / 2)  # 复噪声每维的标准差
        noise = noise_std * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
        
        return symbols + noise
    
    # ================================================
    #              原有调制解调算法（保持不变）
    # ================================================
    def _modulate_hqam(self, bits, eb_lin):
        """分层16-HQAM调制"""
        # 计算符号能量 Es = Eb * k
        Es = eb_lin * self.bits_per_symbol
        
        # 计算星座点幅度
        if self.alpha == 0 or self.alpha == float('inf'):
            a = b = np.sqrt(Es / 2)  # QPSK
        else:
            denominator = 1 + (1 + 2 / self.alpha) ** 2
            a = np.sqrt(Es / denominator)
            b = a * (1 + 2 / self.alpha)
        
        # 调制过程
        symbols = bits.reshape(-1, 4)
        tx_symbols = np.zeros(len(symbols), dtype=np.complex128)
        
        for i in range(len(symbols)):
            hp = symbols[i, :2]  # 高层比特
            lp = symbols[i, 2:]  # 低层比特
            
            # 确定象限
            if hp[0] == 0 and hp[1] == 0:    # 第一象限
                I_sign, Q_sign = 1, 1
            elif hp[0] == 0 and hp[1] == 1:  # 第四象限
                I_sign, Q_sign = 1, -1
            elif hp[0] == 1 and hp[1] == 1:  # 第三象限
                I_sign, Q_sign = -1, -1
            else:  # 第二象限 (hp[0]==1 and hp[1]==0)
                I_sign, Q_sign = -1, 1
            
            # 确定幅度
            if lp[0] == 0 and lp[1] == 0:  # (a,a)
                I_amp, Q_amp = a, a
            elif lp[0] == 0 and lp[1] == 1:  # (a,b)
                I_amp, Q_amp = a, b
            elif lp[0] == 1 and lp[1] == 1:  # (b,b)
                I_amp, Q_amp = b, b
            else:  # (b,a)
                I_amp, Q_amp = b, a
            
            tx_symbols[i] = I_sign * I_amp + 1j * Q_sign * Q_amp
        
        return tx_symbols
    
    def _demodulate_hqam(self, symbols, eb_lin):
        """分层16-HQAM解调"""
        # 计算符号能量 Es = Eb * k
        Es = eb_lin * self.bits_per_symbol
        
        # 计算星座点幅度
        if self.alpha == 0 or self.alpha == float('inf'):
            a = b = np.sqrt(Es / 2)  # QPSK
        else:
            denominator = 1 + (1 + 2 / self.alpha) ** 2
            a = np.sqrt(Es / denominator)
            b = a * (1 + 2 / self.alpha)
        
        # 生成所有可能的比特组合
        all_possible_bits = np.array([
            [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
            [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
            [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
            [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
        ])
        
        # 生成对应的理论星座点
        constellation_points = []
        for bits in all_possible_bits:
            hp = bits[:2]  # 高层比特
            lp = bits[2:]  # 低层比特
            
            # 确定象限
            if hp[0] == 0 and hp[1] == 0:
                I_sign, Q_sign = 1, 1
            elif hp[0] == 0 and hp[1] == 1:
                I_sign, Q_sign = 1, -1
            elif hp[0] == 1 and hp[1] == 1:
                I_sign, Q_sign = -1, -1
            else:  # hp[0]==1 and hp[1]==0
                I_sign, Q_sign = -1, 1
            
            # 确定幅度
            if lp[0] == 0 and lp[1] == 0:
                I_amp, Q_amp = a, a
            elif lp[0] == 0 and lp[1] == 1:
                I_amp, Q_amp = a, b
            elif lp[0] == 1 and lp[1] == 1:
                I_amp, Q_amp = b, b
            else:  # lp[0]==1 and lp[1]==0
                I_amp, Q_amp = b, a
            
            constellation_points.append(I_sign * I_amp + 1j * Q_sign * Q_amp)
        
        constellation_points = np.array(constellation_points)
        
        bits = []
        for symbol in symbols:
            # 计算欧氏距离平方（使用平方距离提高效率）
            dist_sq = np.abs(constellation_points - symbol) ** 2
            
            # 找到最小距离对应的星座点索引
            min_index = np.argmin(dist_sq)
            
            # 获取对应的比特序列
            bits.extend(all_possible_bits[min_index])
        
        return np.array(bits, dtype=int)
    
    def _modulate_16qam_std(self, bits, eb_lin):
        """标准16-QAM调制"""
        # 计算符号能量 Es = Eb * k
        Es = eb_lin * self.bits_per_symbol
        norm_factor = np.sqrt(10)  # 16-QAM归一化因子
        
        constellation = np.array([
            -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
            -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
            +3 - 3j, +3 - 1j, +3 + 3j, +3 + 1j,
            +1 - 3j, +1 - 1j, +1 + 3j, +1 + 1j
        ]) / norm_factor * np.sqrt(Es)
        
        symbols = bits.reshape(-1, 4)
        tx_symbols = np.zeros(len(symbols), dtype=np.complex128)
        
        for i in range(len(symbols)):
            symbol_index = symbols[i, 0] * 8 + symbols[i, 1] * 4 + symbols[i, 2] * 2 + symbols[i, 3]
            tx_symbols[i] = constellation[symbol_index]
        
        return tx_symbols
    
    def _demodulate_16qam_std(self, symbols, eb_lin):
        """标准16-QAM解调"""
        # 计算符号能量 Es = Eb * k
        Es = eb_lin * self.bits_per_symbol
        norm_factor = np.sqrt(10)  # 16-QAM归一化因子
        
        constellation = np.array([
            -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
            -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
            +3 - 3j, +3 - 1j, +3 + 3j, +3 + 1j,
            +1 - 3j, +1 - 1j, +1 + 3j, +1 + 1j
        ]) / norm_factor * np.sqrt(Es)
        
        bits = []
        
        for symbol in symbols:
            # 计算欧氏距离平方（使用平方距离提高效率）
            dist_sq = np.abs(constellation - symbol) ** 2
            
            # 找到最小距离对应的星座点索引
            symbol_index = np.argmin(dist_sq)
            
            # 将索引转换为比特
            bits.extend([
                (symbol_index >> 3) & 1,
                (symbol_index >> 2) & 1,
                (symbol_index >> 1) & 1,
                symbol_index & 1
            ])
        
        return np.array(bits, dtype=int)
    
    def _modulate_qpsk_std(self, bits, eb_lin):
        """标准QPSK调制"""
        # 计算符号能量 Es = Eb * k
        Es = eb_lin * self.bits_per_symbol
        
        # 将比特流重塑为每行2比特
        symbols = bits.reshape(-1, 2)
        I_bits = symbols[:, 0]  # I路比特
        Q_bits = symbols[:, 1]  # Q路比特
        
        # 0→+1, 1→-1
        I = 1 - 2 * I_bits.astype(float)
        Q = 1 - 2 * Q_bits.astype(float)
        
        # 能量归一化
        scale = np.sqrt(Es / 2)
        return (I + 1j * Q) * scale
    
    def _demodulate_qpsk_std(self, symbols, eb_lin):
        """标准QPSK解调"""
        # 计算符号能量 Es = Eb * k
        Es = eb_lin * self.bits_per_symbol
        
        # 解除能量归一化
        scale = np.sqrt(Es / 2)
        rx_symbols = symbols / scale
        
        # 提取实部和虚部
        I_rx = np.real(rx_symbols)
        Q_rx = np.imag(rx_symbols)
        
        # I路判决：负值→1，正值→0
        b0 = I_rx < 0
        # Q路判决：负值→1，正值→0
        b1 = Q_rx < 0
        
        # 重组比特流
        bits = np.column_stack([b0, b1]).flatten()
        return bits.astype(int)
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import concurrent.futures
import time
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示图形

def simulate_modulation(mod_type, ebno_db, total_bits):
    """并行执行的调制解调仿真函数"""
    # 转换为线性值
    ebno_lin = 10 ** (ebno_db / 10)
    
    # 创建调制器
    modulator = Modulator(mod_type, ebno_db)
    
    # 生成随机比特流
    bits = np.random.randint(0, 2, total_bits)
    bit_stream = ''.join(str(b) for b in bits)
    
    # 调制
    symbols = modulator.modulate(bit_stream)
    
    # 加噪
    noisy_symbols = modulator.add_noise(symbols)
    
    # 解调
    demod_bits_str = modulator.demodulate(noisy_symbols)
    demod_bits = np.array([int(b) for b in demod_bits_str])
    
    # 计算误码率
    ber_sim = np.mean(bits != demod_bits)
    
    # 计算理论BER
    if mod_type == '16qam':
        ber_theory = (3/8) * erfc(np.sqrt(0.4 * ebno_lin))
    elif mod_type == 'qpsk':
        ber_theory = 0.5 * erfc(np.sqrt(ebno_lin))
    else:
        ber_theory = 0
    
    return ber_sim, ber_theory


def test_ber_performance_parallel():
    # Eb/N0范围 (dB)
    ebno_db_range = np.arange(0, 16, 1)
    total_bits = 1000000
    num_processes = 8
    output_dir = "ber_results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("开始并行BER性能测试...")
    print(f"总比特数: {total_bits}")
    print(f"使用 {num_processes} 个并行进程")
    print(f"测试点数量: {len(ebno_db_range)*2}")
    print(f"结果将保存到: {os.path.abspath(output_dir)}")
    start_time = time.time()
    
    # 初始化结果存储结构
    results = {
        '16qam': {'sim': {}, 'theory': {}},
        'qpsk': {'sim': {}, 'theory': {}}
    }
    
    # 创建进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 准备任务列表
        tasks = []
        for ebno_db in ebno_db_range:
            tasks.append(('16qam', ebno_db, total_bits))
            tasks.append(('qpsk', ebno_db, total_bits))
        
        # 提交所有任务
        futures = [executor.submit(simulate_modulation, mod_type, ebno_db, total_bits)
                  for mod_type, ebno_db, _ in tasks]
        
        # 处理完成的任务
        completed_count = 0
        total_tasks = len(futures)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                # 获取任务参数
                task_index = futures.index(future)
                mod_type, ebno_db, _ = tasks[task_index]
                
                # 获取结果
                ber_sim, ber_theory = future.result()
                
                # 存储结果
                results[mod_type]['sim'][ebno_db] = ber_sim
                results[mod_type]['theory'][ebno_db] = ber_theory
                
                completed_count += 1
                
                # 打印每完成10%任务的进度
                if completed_count % (total_tasks // 10) == 0:
                    print(f"进度: {completed_count}/{total_tasks} ({completed_count/total_tasks:.0%})")
            
            except Exception as e:
                print(f"任务失败: {str(e)}")
                # 记录失败的任务
                task_index = futures.index(future)
                mod_type, ebno_db, _ = tasks[task_index]
                print(f"失败任务: {mod_type} @ {ebno_db}dB")
    
    duration = time.time() - start_time
    print(f"\n测试完成! 总耗时: {duration:.2f}秒")
    
    # ================= 结果整理和绘图 =================
    # 创建排序后的结果列表
    final_results = {
        '16qam': {'sim': [], 'theory': []},
        'qpsk': {'sim': [], 'theory': []}
    }
    
    # 按Eb/N0顺序填充结果
    for ebno_db in ebno_db_range:
        for mod_type in ['16qam', 'qpsk']:
            # 如果该点有结果则添加，否则添加NaN
            if ebno_db in results[mod_type]['sim']:
                final_results[mod_type]['sim'].append(results[mod_type]['sim'][ebno_db])
                final_results[mod_type]['theory'].append(results[mod_type]['theory'][ebno_db])
            else:
                final_results[mod_type]['sim'].append(np.nan)
                final_results[mod_type]['theory'].append(np.nan)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 16QAM结果
    plt.semilogy(ebno_db_range, final_results['16qam']['theory'], 'r--', linewidth=2, label='16QAM Theory')
    plt.semilogy(ebno_db_range, final_results['16qam']['sim'], 'ro-', markersize=6, label='16QAM Simulation')
    
    # QPSK结果
    plt.semilogy(ebno_db_range, final_results['qpsk']['theory'], 'b--', linewidth=2, label='QPSK Theory')
    plt.semilogy(ebno_db_range, final_results['qpsk']['sim'], 'bo-', markersize=6, label='QPSK Simulation')
    
    plt.title('BER Performance Comparison (Parallel Simulation)')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.ylim(1e-6, 1)
    
    # 保存图像
    image_path = os.path.join(output_dir, 'ber_comparison.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"BER曲线已保存为: {image_path}")
    
    # 保存原始数据为CSV文件
    csv_path = os.path.join(output_dir, 'ber_data.csv')
    with open(csv_path, 'w') as f:
        f.write("Eb/N0 (dB),16QAM Theory,16QAM Sim,QPSK Theory,QPSK Sim\n")
        for i, ebno_db in enumerate(ebno_db_range):
            f.write(f"{ebno_db},")
            f.write(f"{final_results['16qam']['theory'][i]},")
            f.write(f"{final_results['16qam']['sim'][i]},")
            f.write(f"{final_results['qpsk']['theory'][i]},")
            f.write(f"{final_results['qpsk']['sim'][i]}\n")
    print(f"原始数据已保存为: {csv_path}")
    
    # 关闭图形
    plt.close()

# ================================================
#  在这里调用测试函数
# ================================================
if __name__ == "__main__":
    test_ber_performance_parallel()
