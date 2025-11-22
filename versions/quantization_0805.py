import torch
import numpy as np

# 偏移系数计算函数
bp = lambda Ep: 2 ** (Ep - 1) - 1
trd_round = lambda num, bias: int(num * 10**bias + 0.5) / (10**bias)

class Float4Quantizer:
    """
    实现符合IEEE 754标准的浮点数量化/反量化方案
    支持FP32到FP4的转换，返回二进制比特流
    """
    # DICT: floating-point format
    DICT = {
        "FP32": [32, 8, 23, bp(8)],
        "FP16": [16, 5, 10, bp(5)],
        "FP8": [8, 4, 3, bp(4)],
        "FP4": [4, 2, 1, bp(2)]
    }

    @staticmethod
    def quantize(tensor):
        """
        将FP32张量量化为FP4格式的二进制比特流
        返回: 二进制字符串，每个FP4值用4位表示 (s0|Ep|Mp)
        """
        flat_tensor = tensor.flatten().detach().cpu().numpy()
        bit_stream = ""
        
        for val in flat_tensor:
            s0, Ep, Mp, _ = Float4Quantizer._quantize_scalar(val)
            # 将分量转为二进制并拼接: s0(1bit) + Ep(2bits) + Mp(1bit)
            fp4_bits = f"{s0:01b}{Ep:02b}{Mp:01b}"
            bit_stream += fp4_bits
            
        return bit_stream

    @staticmethod
    def dequantize(bit_stream, original_shape, device="cpu"):
        """
        将FP4格式的二进制比特流反量化为FP32张量
        参数:
            bit_stream: 二进制字符串，每个FP4值4位
            original_shape: 原始张量形状
            device: 返回张量的设备
        返回:
            反量化后的FP32张量
        """
        values = []
        total_bits = len(bit_stream)
        num_elements = total_bits // 4
        
        for i in range(num_elements):
            # 每4位解析一个FP4值
            start_idx = i * 4
            bits = bit_stream[start_idx:start_idx+4]
            
            # 解析位字段
            s0 = int(bits[0])          # 符号位
            Ep = int(bits[1:3], 2)     # 指数 (2位)
            Mp = int(bits[3])          # 尾数 (1位)
            
            # 推导非规格化标志
            isubnorm = 1 if (Ep == 0 and Mp > 0) else 0
            
            # 计算浮点数值
            val = Float4Quantizer._repr(s0, Ep, Mp, isubnorm)
            values.append(val)
        
        # 重建张量
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        return tensor.reshape(original_shape)

    @staticmethod
    def _quantize_scalar(val, p_target="FP4", p_original="FP32"):
        """量化单个标量值，返回四元组(s0, Ep, Mp, isubnorm)"""
        FP_format = Float4Quantizer.DICT[p_original]
        FP_format_p = Float4Quantizer.DICT[p_target]
        bo = FP_format[3]
        bp_val = FP_format_p[3]
        l_Mp = FP_format_p[2]
        
        # 处理零值
        if val == 0:
            return 0, 0, 0, 0
            
        s0 = 0 if val >= 0 else 1
        abs_val = abs(val)
        
        # 计算指数和尾数
        try:
            E32 = int(np.floor(bo + np.log2(abs_val)))
        except:
            E32 = 0  # 处理log(0)错误
            
        if E32 > 0:  # 规格数
            M32 = abs_val * 2.0**(bo - E32) - 1
        else:        # 非规格数
            E32 = 1
            M32 = abs_val * 2.0**(bo - E32)
        
        # 重新偏置指数
        Ep = E32 - bo + bp_val
        
        # 舍入尾数
        Mp = min(int(trd_round(2 ** l_Mp * M32, 0)), 2 ** l_Mp - 1)
        
        # 处理边缘情况
        if Ep >= 2**FP_format_p[1] - 1:  # 上溢
            Ep = 2**FP_format_p[1] - 1
            isubnorm = 0
        elif Ep > 0:  # 规格数
            isubnorm = 0
        elif Ep == 0 and Mp != 0:  # 非规格数
            isubnorm = 1
        else:  # 下溢
            Ep = 0
            Mp = 0
            isubnorm = 1
            
        return s0, Ep, Mp, isubnorm

    @staticmethod
    def _repr(s0, Ep, Mp, isubnorm, p_target="FP4"):
        """
        计算最终值
        返回: 反量化后的浮点数值
        """
        FP_format_p = Float4Quantizer.DICT[p_target]
        l_Mp = FP_format_p[2]
        bp_val = FP_format_p[3]
        
        # 确保s0是整数
        s0_int = int(s0)
        
        # 计算最终值
        Mp_rebias = Mp * 2**(-l_Mp)
        if isubnorm == 1:  # 非规格数
            return (-1.0)**s0_int * 2.0**(1 - bp_val) * Mp_rebias
        else:  # 规格数
            return (-1.0)**s0_int * 2.0**(Ep - bp_val) * (1.0 + Mp_rebias)

# 定义浮点数格式字典
Float4Quantizer.DICT = {
    "FP32": [32, 8, 23, bp(8)],
    "FP16": [16, 5, 10, bp(5)],
    "FP8": [8, 4, 3, bp(4)],
    "FP4": [4, 2, 1, bp(2)]
}

# =============================== 测试代码 ===============================
import torch

def test_quantization():
    """测试量化与反量化过程"""
    # 创建测试张量 - 使用PyTorch张量而非NumPy数组
    test_tensor = torch.tensor([1.25, -0.75, 0.0, 3.0, 0.1, 0.125], dtype=torch.float32)
    original_shape = test_tensor.shape
    
    print("原始张量:")
    print(test_tensor.numpy())  # 转换为NumPy数组以便打印
    
    # 量化
    bit_stream = Float4Quantizer.quantize(test_tensor)
    print("\n量化后的比特流:")
    print(bit_stream)
    
    # 显示比特流结构
    print("\n比特流结构 (每4位一个FP4值):")
    for i in range(0, len(bit_stream), 4):
        print(f"值{i//4}: {bit_stream[i:i+4]}")
    
    # 反量化
    dequantized_tensor = Float4Quantizer.dequantize(bit_stream, original_shape)
    
    print("\n反量化后的张量:")
    print(dequantized_tensor.numpy())  # 转换为NumPy数组以便打印

if __name__ == "__main__":
    test_quantization()

