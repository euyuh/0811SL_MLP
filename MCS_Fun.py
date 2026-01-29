import numpy as np
import math

# --------------------------------------
# 星座映射表缓存机制
# --------------------------------------
_constellation_cache = {}
# **************************************

# --------------------------------------
# 调制阶数映射机制机制
# --------------------------------------
mcs_ode = {
    'bpsk': 2,
    'qpsk': 4,
    '16qam': 16,
    '64qam': 64,
    '256qam': 256}

ode_mcs ={
    2: 'bpsk',
    4: 'qpsk',
    16: '16qam',
    64: '64qam',
    256: '256qam'
}
# **************************************


# --------------------------------------
# Gray 编码映射 / BLER计算公式
# --------------------------------------
gray_code = lambda n: n ^ (n >> 1)

def get_feasible_MCS_CR(snr_db, df, target_bler=1e-3):
    """
    计算给定 Eb/N0 下的 BLER 值（快速版）
    :param snr_db: 当前SNR值,可向量形式（dB）
    :param target_bler: 目标BLER值
    :param df: MCS调制阶数与码率映射表（DataFrame格式）
    :return: 可行的 MCS 索引和码率表（输入“snr_db”向量-->输出三维矩阵）
    """

    # 调整广播数据形式
    snr_array_db = snr_db[:, None]      # 扩展为列向量，便于广播计算
    L1 = df['Param1'].values[None, :]  # 扩展行向量，便于广播计算
    L2 = df['Param2'].values[None, :]  # 扩展行向量，便于广播计算
    Lm_db = df['Lamb'].values[None, :]  # 扩展行向量，便于广播计算

    # 预处理，dB -> 线性
    snr_array_lin = 10 ** (snr_array_db / 10.0)

    # 计算 BLER (注：BLER=1 if SNR < Lambda)
    bler = L1 * np.exp(-L2 * snr_array_lin)

    # 计算掩码向量，去掉不满足 SNR > Lambda 的项
    mask = snr_array_db > Lm_db
    bler = bler * mask + (1.0 - mask)  # 不满足条件的 BLER 设为 1.0

    # 提取BLER小于目标值的MCS索引和码率
    mask_bler = bler <= target_bler

    # 将DataFrame转换为NumPy数组以提高计算性能，函数.values比访问DataFrame列更快
    cr_values = df['Code Rate'].values
    mod_values = df['Modulation'].values
    # 列表推导式：利用掩码直接切片，比loop + append快速且简洁
    feasible_cr = [cr_values[row_mask] for row_mask in mask_bler]
    feasible_mcs = [mod_values[row_mask] for row_mask in mask_bler]

    return (feasible_mcs, feasible_cr)



# --------------------------------------
# 标准MCS调制阶数与码率映射表
# --------------------------------------
"""
Modulation | Code Rate | Param1 | Param2 | Lambda (dB)
--------------------------------------------------------------"""
Table_Data = [
    # QPSK
    ('qpsk', 78/1024,  1.02e5, 73.22, -8.20),
    ('qpsk', 120/1024, 1.97e5, 67.07, -6.10),
    ('qpsk', 173/1024, 7.02e5, 38.96, -4.61),
    ('qpsk', 308/1024, 3.13e5, 16.27, -1.01),
    ('qpsk', 449/1024, 4.97e4, 9.47,  0.70),
    ('qpsk', 602/1024, 5.22e5, 7.42,  2.48),
    # 16-QAM
    ('16qam', 378/1024, 4.50e4, 3.40, 4.97),
    ('16qam', 490/1024, 4.65e4, 2.19, 6.90),
    ('16qam', 616/1024, 5.34e4, 1.46, 8.71),
    # 64-QAM
    ('64qam', 466/1024, 1.56e4, 0.90, 10.30),
    ('64qam', 567/1024, 8.77e3, 0.54, 12.36),
    ('64qam', 666/1024, 4.09e3, 0.29, 14.44),
    ('64qam', 772/1024, 1.86e3, 0.12, 17.94),
    ('64qam', 873/1024, 91.55,  0.04, 20.06),
    ('64qam', 948/1024, 30.10,  0.02, 21.90),
]

# **************************************


def generate_constellation(M):
    """
    生成并缓存 Gray 码星座图及索映射表
    """
    if M in _constellation_cache:
        return _constellation_cache[M]

    k = int(np.log2(M))

    # --- BPSK ---
    if M == 2:
        mapping_table = {(0,): (1 + 0j),
                         (1,): (-1 + 0j)}
        _constellation_cache[M] = mapping_table
        return mapping_table

    # --- QAM ---
    m_side = int(np.sqrt(M))
    gray = np.arange(m_side) ^ (np.arange(m_side) >> 1)
    I = np.arange(-m_side + 1, m_side, 2)
    Q = np.arange(-m_side + 1, m_side, 2)

    constellation = []
    bit_labels = []
    for i in range(m_side):
        for q in range(m_side):
            symbol = I[i] + 1j * Q[q]
            i_bits = np.array(list(np.binary_repr(gray[i], width=k // 2)), dtype=int)
            q_bits = np.array(list(np.binary_repr(gray[q], width=k // 2)), dtype=int)
            bits = np.concatenate([i_bits, q_bits])
            constellation.append(symbol)
            bit_labels.append(bits)

    # 功率归一化
    conste = np.array(constellation)
    pwNorm_param = np.sqrt(np.mean(np.abs(conste) ** 2))
    conste_norm = conste / pwNorm_param

    mapping_table = {tuple(b): s for b, s in zip(bit_labels, conste_norm)}

    _constellation_cache[M] = mapping_table
    return mapping_table


# ------------ Modulation Method ------------------
# BPSK (deprecated), QPSK, 16QAM, 64QAM, 256QAM
def qam_M_mod(bits: np.ndarray, M: int):
    """
    :param bits: bit stream (0/1)
    :param M: modulation order (2, 4, 16, 64, 256)
    :return: Complex symbol array
    """
    assert M in (2, 4, 16, 64, 256), "仅支持 M = 2, 4, 16, 64, 256！"

    k = int(np.log2(M))
    # effe_len = (len(bits) // k) * k # 有效长度（整除k）
    # bits = bits[:effe_len]

    # Padding 零以满足整除 k
    pad_len = (k - (len(bits) % k)) % k
    if pad_len > 0:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])

    mapping_table = generate_constellation(M)

    if M == 2:
        symbols = np.array([mapping_table[(b,)] for b in bits])
    else:
        bits_reshape = bits.reshape(-1, k)
        symbols = np.array([mapping_table[tuple(b)] for b in bits_reshape])

    return symbols


# --------------------------------------
# 解调函数
# --------------------------------------
def qam_demod(symbols, m):
    mapping_table = generate_constellation(m)

    if m == 2:
        return (np.real(symbols) < 0).astype(int)

    conste_vector = np.array(list(mapping_table.values()))
    bit_labels = np.array(list(mapping_table.keys()))

    # 使用广播计算距离矩阵
    dist = np.abs(symbols[:, None] - conste_vector[None, :]) ** 2  # |y-s|^2
    idx_min = np.argmin(dist, axis=1)

    return bit_labels[idx_min].flatten()

# --------------------------------------
# LLR计算函数·
# --------------------------------------
def llrs_computate(symbols, ebn0, mod_ord, code_rate, method='max-log'):
    """
        Calculates Log-Likelihood Ratios (LLRs) for noisy modulated symbols.

        Args:
            noisy_symbols: The received symbols after the AWGN channel.
            eb_no_db: The Eb/No in dB.
            code_rate: The code rate R = K/N.
            mod_ord: modulation order

        Returns:
            An array of LLRs.
    """
    k = int(math.log2(mod_ord))
    ebn0_lin = 10 ** (ebn0 / 10)
    sig_power = np.mean(np.abs(symbols) ** 2)  # Signal power
    noise_power = sig_power / (code_rate * ebn0_lin * math.log2(mod_ord))

    mapping_table = generate_constellation(mod_ord)
    llrs_per_sym = np.empty((len(symbols), k))

    # 提取星座点，用于距离计算
    conste_vector = np.array(list(mapping_table.values()))
    bit_labels = np.array(list(mapping_table.keys()))

    if mod_ord == 2:
        # LLR for BPSK in AWGN is L(y) = 2*y / sigma^2
        return 2 * symbols.real / noise_power


    for i, y in enumerate(symbols):
        dist = np.abs(y - conste_vector) ** 2  # |y-s|^2
        for bit_idx in range(k):
            mask0 = (bit_labels[:, bit_idx] == 0)
            mask1 = ~mask0
            if method == 'max-log':
                # max-log approximation
                min0 = np.min(dist[mask0])
                min1 = np.min(dist[mask1])

                # LLR = ln( sum_{s in S0} exp(-d2/N0) / sum_{s in S1} exp(-d2/N0) )
                # approx -> -(min0 - min1) / (N0)  where N0 = 2*noise_var
                llr_val = -(min0 - min1) / (2.0 * noise_power)
            else:
                # exact (log-sum-exp) version:
                a0 = -dist[mask0] / (2.0 * noise_power)
                a1 = -dist[mask1] / (2.0 * noise_power)
                # log-sum-exp
                M0 = np.max(a0)
                M1 = np.max(a1)
                s0 = M0 + np.log(np.sum(np.exp(a0 - M0)))
                s1 = M1 + np.log(np.sum(np.exp(a1 - M1)))
                llr_val = s0 - s1

            llrs_per_sym[i, bit_idx] = llr_val

    return llrs_per_sym.reshape(-1)




if __name__ == "__main__":
    np.random.seed(0)
    bits_1 = np.random.randint(0, 2, int(256000))

    mod_ord = 16

    constellation, bit_labels = generate_constellation(mod_ord)
    tx_bits_2, bits_padding = qam_M_mod(bits_1, mod_ord)
    rx_bits_2 = qam_demod(tx_bits_2, mod_ord)
    nerr = np.sum(rx_bits_2 != bits_padding)
    print(f"{mod_ord}-QAM: 比特错误数 = {nerr}, 正确率 = {(1 - nerr / len(bits_padding)) * 100:.4f}%")

    # for M in [2, 4, 16, 64, 256]:
    #     tx, bits_2 = qam_M_mod(bits_1, M)
    #     rx_bits = qam_demod(tx, M)
    #     nerr = np.sum(bits_2 != rx_bits)
    #     print(f"{M}-QAM: 比特错误数 = {nerr}, 正确率 = {(1 - nerr / len(bits_2)) * 100:.4f}%")

