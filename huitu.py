import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_log_scale_results():
    # --- 配置 ---
    EXCEL_FILENAME = "power_allocation_results_0.1_0.1.xlsx"
    PLOT_FILENAME = "power_allocation_log_plot_0.1_0.1.png"
    EBNO_DB = 0.0  # 仅用于图表标题显示

    # --- 1. 读取数据 ---
    if not os.path.exists(EXCEL_FILENAME):
        print(f"Error: 找不到文件 {EXCEL_FILENAME}，请先运行测试脚本生成数据。")
        return

    print(f"正在读取 {EXCEL_FILENAME} ...")
    df = pd.read_excel(EXCEL_FILENAME)

    # 提取数据
    # 确保列名与生成时一致
    x_ratios = df["Ratio (P0/P1)"]
    y_accs = df["Accuracy (%)"]
    p0_values = df["P0 (High Bits)"]

    # --- 2. 绘图 ---
    print("正在生成对数坐标图表...")
    plt.figure(figsize=(12, 7))  # 稍微加宽一点以便显示

    # 绘制连线和散点
    # zorder用于控制绘制层级，保证点在线上方
    plt.plot(x_ratios, y_accs, linestyle='-', color='gray', alpha=0.5, label='Trend')
    plt.scatter(x_ratios, y_accs, c='blue', s=80, zorder=5, label='Data Points')

    # --- 3. 设置对数坐标 ---
    plt.xscale('log') 

    # --- 4. 标注 P0 值 ---
    # 由于是对数坐标，标注位置可能需要微调，这里使用相对偏移
    for i, p0_val in enumerate(p0_values):
        # 格式化 P0 显示，保留一位小数
        label_text = f"P0={p0_val:.1f}"
        
        # 获取当前点的坐标
        x = x_ratios[i]
        y = y_accs[i]
        
        # 简单的避让逻辑：奇数点向上标，偶数点向下标，防止重叠
        # offset 是 (x偏移, y偏移) 单位是 point
        if i % 2 == 0:
            offset = (0, 10) 
            va = 'bottom'
        else:
            offset = (0, -15)
            va = 'top'
            
        plt.annotate(
            label_text, 
            (x, y), 
            textcoords="offset points", 
            xytext=offset, 
            ha='center', 
            va=va,
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color='gray', alpha=0.5) # 加个小箭头指向点
        )

    # --- 5. 图表修饰 ---
    plt.title(f'Accuracy vs. Power Ratio (Log Scale) at EbNo={EBNO_DB}dB', fontsize=14)
    plt.xlabel('Power Ratio (Carrier 0 / Carrier 1) - Log Scale', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    
    # 开启网格 (对数坐标下的网格通常很有用)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # 调整坐标轴范围，稍微留白
    if len(x_ratios) > 0:
        plt.xlim(min(x_ratios) * 0.8, max(x_ratios) * 1.2)
        plt.ylim(min(y_accs) - 5, max(y_accs) + 5)

    plt.legend()
    plt.tight_layout()

    # --- 6. 保存 ---
    plt.savefig(PLOT_FILENAME, dpi=300) # 提高分辨率
    print(f"[Success] 对数坐标图表已保存至: {os.path.abspath(PLOT_FILENAME)}")
    plt.show()

if __name__ == "__main__":
    plot_log_scale_results()