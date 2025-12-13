import torch

# ==========================================
# 1. 物理层调制解调器 (Modem)
# ==========================================
class BPSKModem:
    def __init__(self, ebno_db=20.0):
        self.ebno_db = ebno_db
        # BPSK: 0 -> +1, 1 -> -1
        self.constellation = torch.tensor([1.0, -1.0])

    def set_snr(self, ebno_db):
        self.ebno_db = ebno_db

    def modulate(self, bits):
        """
        [发射] Bits (0/1) -> Symbols (+1/-1)
        bits: [..., num_bits]
        """
        # 0映射为+1, 1映射为-1
        return 1.0 - 2.0 * bits.float()

    def add_noise(self, symbols):
        """
        [信道] AWGN 加噪
        """
        # 计算噪声标准差
        snr_lin = 10 ** (self.ebno_db / 10.0)
        # BPSK 符号能量Es=1, Sigma = sqrt(1 / (2*SNR))
        sigma = torch.sqrt(1.0 / (2.0 * torch.tensor(snr_lin)))
        
        noise = torch.randn_like(symbols) * sigma.to(symbols.device)
        return symbols + noise

    def demodulate(self, noisy_symbols):
        """
        [接收] Symbols -> Bits (硬判决)
        >0 -> 0, <=0 -> 1
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