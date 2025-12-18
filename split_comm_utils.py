import torch

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