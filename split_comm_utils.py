import torch
import numpy as np
import math
from MCS_Fun import llrs_computate # 学长的函数
from CAPolarCodec import CAPolarCodec # 学长的类
# ==========================================
# 1. 物理层调制解调器 (Modem) - 支持并行载波
# ==========================================
class BPSKModem:
    def __init__(self, ebno_db=20.0, num_carriers=4, power_profile=None):
        self.ebno_db = ebno_db
        self.num_carriers = num_carriers
        
        # 功率分配 (Power Allocation)
        # power_profile: list of length num_carriers, e.g., [1, 1, 1, 1]
        if power_profile is None:
            self.power_scale = torch.ones(num_carriers)
        else:
            assert len(power_profile) == num_carriers
            # 归一化：保证平均能量为 1 (Total Energy = Num Carriers)
            p_tensor = torch.tensor(power_profile, dtype=torch.float32)
            avg_p = p_tensor.mean()
            self.power_scale = torch.sqrt(p_tensor / avg_p) # 幅度缩放系数

    def modulate(self, bits):
        """
        bits: [Time_Steps, Num_Carriers] (0/1)
        Returns: symbols [Time_Steps, Num_Carriers] (+A/-A)
        """
        # 0->+1, 1->-1
        bpsk_syms = 1.0 - 2.0 * bits.float()
        
        # 应用功率分配 (Broadcast over time dimension)
        # self.power_scale: [4] -> 广播到 [T, 4]
        device = bits.device
        scaled_syms = bpsk_syms * self.power_scale.to(device).unsqueeze(0)
        
        return scaled_syms

    def add_noise(self, symbols):
        """
        AWGN Channel
        注意：噪声功率通常基于平均符号能量 Es=1 计算。
        虽然各载波功率不同，但噪声水平通常假定为一致（白噪声）。
        """
        snr_lin = 10 ** (self.ebno_db / 10.0)
        # Sigma based on standard BPSK (Es=1)
        sigma = torch.sqrt(1.0 / (2.0 * torch.tensor(snr_lin)))
        
        noise = torch.randn_like(symbols) * sigma.to(symbols.device)
        return symbols + noise

    def demodulate(self, noisy_symbols):
        """
        硬判决: >0 -> 0, <=0 -> 1
        功率缩放不改变符号，只改变抗噪能力，所以判决门限依然是 0
        """
        return (noisy_symbols <= 0).float()

# ==========================================
# 2. QAM4 Modem (物理层)
# ==========================================
class QAM4Modem:
    def __init__(self, snr_db=20.0, num_carriers=4, power_profile=None):
        self.snr_db = snr_db
        self.num_carriers = num_carriers
        if power_profile is None:
            self.power_scale = torch.ones(num_carriers)
        else:
            p_tensor = torch.tensor(power_profile, dtype=torch.float32)
            self.power_scale = torch.sqrt(p_tensor / p_tensor.mean())

    def modulate(self, bits):
        """ bits: [Time, 4] -> 4-QAM (2 bits/symbol) -> [Time/2, 4] Complex """
        T, C = bits.shape
        # 补齐 Time 维度以防奇数
        if T % 2 != 0:
            bits = torch.cat([bits, torch.zeros(1, C, device=bits.device)], dim=0)
            self.padded_mod = True
        else:
            self.padded_mod = False
            
        # 每2行合并: 行0->I路, 行1->Q路
        bits_paired = bits.view(-1, 2, C).permute(0, 2, 1) # [Time/2, C, 2]
        
        # 0->+1, 1->-1. 归一化 / sqrt(2)
        syms_I = 1.0 - 2.0 * bits_paired[..., 0].float()
        syms_Q = 1.0 - 2.0 * bits_paired[..., 1].float()
        complex_syms = torch.complex(syms_I, syms_Q) / 1.41421356 
        
        # 功率分配
        return complex_syms * self.power_scale.to(bits.device).unsqueeze(0)

    def add_noise(self, symbols):
        # 物理层信噪比 Es/N0 = Eb/N0 (对于 QPSK/4QAM 且无额外重复时)
        # 这里直接使用输入的 db 值作为符号信噪比
        snr_lin = 10 ** (self.snr_db / 10.0)
        sigma = torch.sqrt(1.0 / (2.0 * torch.tensor(snr_lin)))
        noise_r = torch.randn_like(symbols.real) * sigma.to(symbols.device)
        noise_i = torch.randn_like(symbols.imag) * sigma.to(symbols.device)
        return symbols + torch.complex(noise_r, noise_i)

# ==========================================
# 3. Polar 适配器与流管理器 (核心新增)
# ==========================================
class PolarStreamAdapter:
    """ 将长比特流切分为块，调用学长的 CAPolarCodec 进行处理 """
    def __init__(self, N, K_info, crc_len, mod_ord=4):
        self.N = N
        self.K_info = K_info
        # 实例化学长的类
        # [修改 1] 调试阶段强制将 list_size 设为 1 (SC译码)，速度快 4-10 倍
        # 等跑通了再改回 4
        self.codec = CAPolarCodec(N=N, K_info=K_info, crc_len=crc_len, list_size=1) 
        self.mod_ord = mod_ord
        self.bits_per_sym = int(math.log2(mod_ord))

    def encode_stream(self, bit_stream):
        """ numpy 1D array -> encoded numpy 1D array """
        total_len = len(bit_stream)
        # Padding
        pad_len = (self.K_info - (total_len % self.K_info)) % self.K_info
        if pad_len > 0:
            padded = np.concatenate([bit_stream, np.zeros(pad_len, dtype=int)])
        else:
            padded = bit_stream
            
        # 切块编码
        num_blocks = len(padded) // self.K_info
        reshaped = padded.reshape(num_blocks, self.K_info)
        encoded_chunks = [self.codec.encode(blk) for blk in reshaped]
        
        return np.concatenate(encoded_chunks)

    def decode_stream(self, rx_symbols_complex, snr_db, original_len):
        """ 接收复数符号 -> LLR计算 -> 切块译码 -> 去Padding """
        # 计算 LLR (注意: llrs_computate 需要的是 Eb/N0，我们做转换)
        # Es/N0 = snr_db. Eb/N0 = Es/N0 - 10log10(BitsPerSym * Rate)
        # 这里简单起见，直接传 snr_db 进 llrs_computate 的 ebn0 参数，
        # 并把 code_rate 设为 1.0/bits_per_sym 抵消，使得函数内部计算出正确的 N0
        # 或者直接计算 noise_power
        
        # 1. 计算实际码率 R
        real_rate = self.K_info / self.N
        
        # 2. 计算每个符号的比特数 k (log2(M))
        bits_per_sym = math.log2(self.mod_ord)
        
        # 3. [新增] 显式将 Es/N0 转换为 Eb/N0
        # 公式: Eb/N0(dB) = Es/N0(dB) - 10*log10(Rate * k)
        # 这里的 snr_db 实际上是 Es/N0
        ebno_db = snr_db - 10 * math.log10(real_rate * bits_per_sym)
        
        # 4. 调用学长的函数 (传入正确的 Eb/N0)
        llrs = llrs_computate(rx_symbols_complex, ebno_db, self.mod_ord, real_rate)
        
        # ================= [核心修改] =================
        # 加上负号！翻转 LLR 的极性
        llrs = -llrs 
        # ============================================
        
        # 切块译码
        num_blocks = len(llrs) // self.N
        llrs_reshaped = llrs.reshape(num_blocks, self.N)
        decoded_bits = []
        # [修改 2] 增加 flush=True 确保立即显示进度
        # [修改 3] 增加 Numba 编译提示
        # print(f"  [Polar] Start decoding {num_blocks} blocks (First run may lag due to Numba compile)...")
        for i, blk in enumerate(llrs_reshaped):
            # 每解一个块打印一次，flush=True 强制刷新
            print(f"    > Decoding block {i+1}/{num_blocks}...", end='\r', flush=True)
            bits, crc = self.codec.decode(blk)
            decoded_bits.append(bits)
        print(" " * 50, end='\r') # 清除进度条
        full_stream = np.concatenate(decoded_bits)
        # for blk in llrs_reshaped:
        #     bits, crc = self.codec.decode(blk)
        #     decoded_bits.append(bits)
            
        # full_stream = np.concatenate(decoded_bits)
        # 截取有效长度
        return full_stream[:original_len]

class MultiStreamManager:
    """ 
    负责：
    1. 生成 Scale/ZP 的比特
    2. 将数据切分为 4 个流 (Bit-Plane Slicing)
    3. 将 Scale/ZP 复制 4 份附着在每个流头部
    4. 调用 4 个 Adapter 进行编码
    """
    def __init__(self, device, N=512, K_info=256, crc_len=16):
        self.device = device
        # 4 个独立的编码器对应 4 个信道
        self.adapters = [PolarStreamAdapter(N, K_info, crc_len) for _ in range(4)]

    def pack_and_encode(self, data_bits_matrix, scale, zp):
        """
        data_bits_matrix: [Total_Pixels, 8]
        scale, zp: float / int
        Returns: tx_frame [Time, 4], info_lengths (list)
        """
        # 1. 准备 Metadata (Scale 32bit + ZP 8bit = 40 bits)
        scale_bits = Float32Codec.float_to_bits(torch.tensor([scale], device=self.device)).flatten()
        zp_bits = Int8Codec.int_to_bits(torch.tensor([zp], device=self.device), num_bits=8).flatten()
        meta_bits = torch.cat([scale_bits, zp_bits]).cpu().numpy().astype(int)
        
        # 2. 横向切割数据流 (Bit-Plane Slicing)
        # Stream 0 (Carrier 0): Col 0,1 (MSB)
        # Stream 1 (Carrier 1): Col 2,3
        # ...
        streams = []
        info_lengths = []
        
        for i in range(4):
            # 提取两列 -> 展平
            slice_data = data_bits_matrix[:, i*2 : (i+1)*2].flatten().cpu().numpy().astype(int)
            # 头部附着 Metadata (复制 4 份的体现)
            combined = np.concatenate([meta_bits, slice_data])
            streams.append(combined)
            info_lengths.append(len(combined))

        # 3. 并行 Polar 编码
        encoded_cols = []
        max_len = 0
        for i in range(4):
            enc = self.adapters[i].encode_stream(streams[i])
            encoded_cols.append(enc)
            max_len = max(max_len, len(enc))
            
        # 4. 对齐并堆叠
        # Polar是块编码，如果数据量一样，长度通常一样。防万一做个Padding
        final_tensor_cols = []
        for enc in encoded_cols:
            if len(enc) < max_len:
                enc = np.concatenate([enc, np.zeros(max_len - len(enc))])
            final_tensor_cols.append(torch.from_numpy(enc).to(self.device).float())
            
        # [4, Time] -> [Time, 4]
        tx_frame = torch.stack(final_tensor_cols, dim=0).t()
        return tx_frame, info_lengths

    def decode_and_unpack(self, rx_noisy_frame, info_lengths, snr_db):
        """
        Returns: data_bits_matrix [Total, 8], scale, zp
        """
        # rx_noisy_frame: [Time, 4] (Complex)
        # 这里需要转回 CPU numpy 给 LLR 计算
        rx_np = rx_noisy_frame.cpu().numpy()
        
        decoded_slices = []
        meta_recovered = None
        
        # 1. 并行解码
        for i in range(4):
            # 取出第 i 列 (Carrier i)
            # 注意：Modulation 输出了 Complex 符号，Modem 没有做硬判决
            # 我们需要把 [Time, 4] 里的 Time 展开。
            # QAM Modem 输入 N bits -> 输出 N/2 symbols
            # 所以 rx_np 的长度是 encoded bits 的一半
            
            # 提取列 -> 复数数组
            col_syms = rx_np[:, i] 
            
            # Polar 解码
            # decode_stream 会处理 LLR 和去 Block Padding
            full_bits = self.adapters[i].decode_stream(col_syms, snr_db, info_lengths[i])
            
            # 2. 剥离 Metadata (前40位)
            meta_bits = full_bits[:40]
            data_bits = full_bits[40:]
            
            # 只需要从 Carrier 0 (质量最好) 恢复 Metadata
            if i == 0:
                meta_recovered = torch.from_numpy(meta_bits).to(self.device)
            
            decoded_slices.append(torch.from_numpy(data_bits).to(self.device))
            
        # 3. 恢复 Metadata 数值
        scale_bits = meta_recovered[:32]
        zp_bits = meta_recovered[32:]
        scale = Float32Codec.bits_to_float(scale_bits).item()
        zp = Int8Codec.bits_to_int(zp_bits, num_bits=8).item()
        
        # 4. 恢复数据矩阵 [Total, 8]
        # 每个 slice 是 [Total * 2] -> reshape [Total, 2]
        # 然后横向拼接 [Total, 8]
        num_pixels = decoded_slices[0].numel() // 2
        reshaped_cols = [s.view(num_pixels, 2) for s in decoded_slices]
        data_matrix = torch.cat(reshaped_cols, dim=1)
        
        return data_matrix, scale, zp
    
# ==========================================
# 2. 信源编解码器 (Codec / Quantizer)
# ==========================================
class Int8Codec:
    """
    负责将浮点数据转换为二进制比特流，以及反向转换。
    包含：Float -> Int8 -> Bits 和 Bits -> Int8 -> Float
    """
    @staticmethod
    def get_bit_mask(num_bits, device):
        """
        统一生成 Big-Endian 掩码: [128, 64, 32, 16, 8, 4, 2, 1]
        """
        # 强制生成 float 类型的 mask，避免与 long 运算时的类型报错
        powers = torch.arange(num_bits - 1, -1, -1)
        mask = (2 ** powers).to(device).float() 
        return mask
    @staticmethod
    def float_to_bits(x_float, min_val, max_val, num_bits=8):
        qmax = 2**num_bits - 1
        scale = (max_val - min_val) / (qmax + 1e-12)
        zp = round(-min_val / (scale + 1e-12))
        
        # Float -> Int
        x_int = (x_float / (scale + 1e-12) + zp).round().clamp(0, qmax)
        
        # Int -> Bits
        # 注意：这里 x_int 需要转为 long 才能做位运算，但在 mask 运算前我们利用数学方法或强制转换
        # 更加通用的位提取方法 (支持 float 输入):
        x_int_long = x_int.long()
        mask = Int8Codec.get_bit_mask(num_bits, x_float.device) # [128, 64, ...]
        
        x_expand = x_int_long.unsqueeze(-1)
        mask_expand = mask.long() # 位运算需要 long
        
        # 提取比特: (x & mask) != 0
        bits = ((x_expand & mask_expand) != 0).float()
        
        return bits, scale, zp
    
    @staticmethod
    def bits_to_float(bits, scale, zp, num_bits=8):
        """
        Bits -> Float (严格对应上面的 float_to_bits)
        """
        # 1. 生成同样的 Mask [128, 64, ...]
        mask = Int8Codec.get_bit_mask(num_bits, bits.device)
        
        # 2. Bits (float 0/1) * Mask (float 128/64...) -> Sum
        # 这一步将二进制位加权求和恢复成整数
        x_int = (bits * mask).sum(dim=-1)
        
        # 3. Int -> Float (反量化)
        x_float = (x_int - zp) * scale
        
        return x_float
    # 专门用于将整数 (如 ZeroPoint) 转为比特流
    @staticmethod
    def int_to_bits(x_int, num_bits=8):
        """
        Int -> Bits (不涉及量化，纯数值转二进制)
        x_int: 整数 Tensor 或 Scalar
        """
        # 确保输入是 tensor
        if not torch.is_tensor(x_int):
            x_int = torch.tensor(x_int)
        
        device = x_int.device
        # 转换为 Long 以进行位运算
        x_long = x_int.long()
        
        # 获取掩码 [128, 64, ..., 1]
        mask = Int8Codec.get_bit_mask(num_bits, device).long()
        
        # 处理维度: 如果是标量，增加维度以便广播
        if x_long.dim() == 0:
            x_long = x_long.view(1)
            
        x_expand = x_long.unsqueeze(-1) # [..., 1]
        
        # 提取比特
        bits = ((x_expand & mask) != 0).float()
        
        # 如果输入是标量，返回 [8]；否则返回 [..., 8]
        return bits.squeeze(0) if x_int.dim() == 0 else bits

    # 专门用于将比特流恢复为整数
    @staticmethod
    def bits_to_int(bits, num_bits=8):
        """
        Bits -> Int
        """
        mask = Int8Codec.get_bit_mask(num_bits, bits.device)
        # 加权求和
        x_int = (bits * mask).sum(dim=-1)
        return x_int.long()
    
# ==========================================
# 3. Float32 直接编解码器 (Debug专用)
# ==========================================
class Float32Codec:
    """
    不进行量化，直接将 Float32 的 IEEE 754 内存位模式转换为 32 位比特流。
    优点：无量化误差，无需统计 min/max。
    缺点：通信开销大 (32 bits per pixel vs 8 bits)。
    """
    
    @staticmethod
    def get_bit_mask(device):
        # 生成 32 位的掩码: [2^31, 2^30, ..., 2^0]
        # 注意使用 int64 (long) 以避免溢出
        powers = torch.arange(31, -1, -1, device=device)
        mask = (2 ** powers).long()
        return mask

    @staticmethod
    def float_to_bits(x_float):
        """
        Float32 -> 32 Bits
        Return: bits (形状为 [..., 32])
        """
        # 1. 使用 view 将 float32 视为 int32 (保持位模式不变)
        # 例如: 1.0 (float) -> 0x3f800000 (int)
        # 注意：这里必须用 int32 来匹配 float32 的位宽
        x_int32 = x_float.view(torch.int32)
        
        # 2. 为了方便位运算，转为 int64 (long) 处理，防止符号位干扰
        # 这里的位运算是纯逻辑提取，不涉及数值意义
        x_long = x_int32.long()
        
        # 3. 提取 32 个比特
        mask = Float32Codec.get_bit_mask(x_float.device) # [32]
        
        # 扩展维度以便广播
        x_expand = x_long.unsqueeze(-1) # [..., 1]
        mask_expand = mask.unsqueeze(0) # [1, 32]
        
        # 按位与 -> 转为 0/1 float
        # 注意处理负数存储时的补码/位表示，直接按位与即可
        bits = ((x_expand & mask_expand) != 0).float()
        
        return bits

    @staticmethod
    def bits_to_float(bits):
        """
        32 Bits -> Float32
        """
        # 1. 获取掩码
        mask = Float32Codec.get_bit_mask(bits.device) # [32]
        
        # 2. 加权求和恢复整数位模式
        # bits: [..., 32], mask: [32]
        # 结果可能很大，用 int64 承载
        x_long_recon = (bits.long() * mask).sum(dim=-1)
        
        # 3. 转回 int32 (丢弃 int64 的高位，保留低32位)
        x_int32_recon = x_long_recon.to(torch.int32)
        
        # 4. view 回 float32
        x_float = x_int32_recon.view(torch.float32)
        
        return x_float