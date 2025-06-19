# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib.font_manager import FontProperties

def plot_smooth_curve(x_original, y_original, save_path=None):
    try:
        # 设置中文字体
        font = FontProperties(family='SimHei')
        
        # 创建样条插值函数
        cs = CubicSpline(x_original, y_original)
        
        # 生成平滑曲线的点
        x_smooth = np.linspace(x_original.min(), x_original.max(), 300)
        y_smooth = cs(x_smooth)
        
        # 创建图形
        plt.figure(figsize=(10, 6), dpi=100)
        
        # 设置样式
        plt.style.use('seaborn')
        
        # 绘制平滑曲线
        plt.plot(x_smooth, y_smooth, '-', color='#2E86C1', linewidth=2, 
                label='平滑拟合曲线 (三次样条)')
        
        # 绘制原始数据点
        plt.plot(x_original, y_original, 'o', color='#E74C3C', 
                markersize=8, label='原始数据点')
        
        # 设置标题和标签
        plt.title("数据平滑曲线图 (Y vs X)", fontproperties=font, fontsize=14)
        plt.xlabel("X轴 数据", fontproperties=font)
        plt.ylabel("Y轴 数据", fontproperties=font)
        
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        plt.legend(prop=font)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片（如果提供保存路径）
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # 显示图形
        plt.show()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        
# 使用示例
x_original = np.array([17.2, 25.8, 31.8, 37.2, 42.6, 49.4, 54.2, 59.8, 66.0, 72.0, 78.6, 83.4])
y_original = np.array([12.9, 10.1, 24.8, 11.4, 38.9, 11.8, 51.2, 15.1, 62.2, 22.3, 70.9, 33.1])

plot_smooth_curve(x_original, y_original, save_path='smooth_curve.png')