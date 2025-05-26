#!/usr/bin/env python3
"""
计算DeepSpeed-MII和llama.cpp框架下预测延迟和真实延迟的线性相关度
Calculate linear correlation between predicted and actual latencies for DeepSpeed-MII and llama.cpp frameworks
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """加载数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"数据加载成功，共 {len(df)} 行数据")
        print(f"列名: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def calculate_correlation(df, framework_name):
    """计算指定框架的相关性"""
    # 过滤指定框架的数据
    framework_data = df[df['Framework'] == framework_name].copy()
    
    if len(framework_data) == 0:
        print(f"未找到框架 {framework_name} 的数据")
        return None
    
    print(f"\n=== {framework_name} 框架分析 ===")
    print(f"数据点数量: {len(framework_data)}")
    
    # 获取真实延迟和预测延迟
    actual_latency = framework_data['Latency'].values
    predicted_latency = framework_data['Predicted_Latency'].values
    
    # 检查是否有缺失值
    valid_mask = ~(np.isnan(actual_latency) | np.isnan(predicted_latency))
    actual_latency = actual_latency[valid_mask]
    predicted_latency = predicted_latency[valid_mask]
    
    print(f"有效数据点数量: {len(actual_latency)}")
    
    if len(actual_latency) < 2:
        print("数据点不足，无法计算相关性")
        return None
    
    # 计算皮尔逊相关系数 (线性相关)
    pearson_corr, pearson_p = pearsonr(actual_latency, predicted_latency)
    
    # 计算斯皮尔曼相关系数 (单调相关)
    spearman_corr, spearman_p = spearmanr(actual_latency, predicted_latency)
    
    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(np.mean((actual_latency - predicted_latency) ** 2))
    
    # 计算平均绝对误差 (MAE)
    mae = np.mean(np.abs(actual_latency - predicted_latency))
    
    # 计算平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((actual_latency - predicted_latency) / actual_latency)) * 100
    
    # 计算决定系数 (R²)
    ss_res = np.sum((actual_latency - predicted_latency) ** 2)
    ss_tot = np.sum((actual_latency - np.mean(actual_latency)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    results = {
        'framework': framework_name,
        'data_points': len(actual_latency),
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r_squared': r_squared,
        'actual_latency': actual_latency,
        'predicted_latency': predicted_latency
    }
    
    # 打印结果
    print(f"皮尔逊相关系数 (线性相关): {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    print(f"斯皮尔曼相关系数 (单调相关): {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"决定系数 (R²): {r_squared:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    
    return results

def plot_correlation(results_list):
    """绘制相关性图表"""
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]
    
    for i, results in enumerate(results_list):
        if results is None:
            continue
            
        ax = axes[i]
        actual = results['actual_latency']
        predicted = results['predicted_latency']
        
        # 散点图
        ax.scatter(actual, predicted, alpha=0.6, s=30)
        
        # 添加完美预测线 (y=x)
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction (y=x)')
        
        # 添加最佳拟合线
        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        ax.plot(actual, p(actual), 'b-', alpha=0.8, label=f'Best Fit (y={z[0]:.3f}x+{z[1]:.3f})')
        
        ax.set_xlabel('实际延迟 (Actual Latency)')
        ax.set_ylabel('预测延迟 (Predicted Latency)')
        ax.set_title(f'{results["framework"]}\nPearson r={results["pearson_correlation"]:.4f}, R²={results["r_squared"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置相等的坐标轴比例
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('framework_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 加载数据
    df = load_data('data/All_results_with_predictions.csv')
    if df is None:
        return
    
    # 显示可用的框架
    frameworks = df['Framework'].unique()
    print(f"可用框架: {frameworks}")
    
    # 分析DeepSpeed-MII框架
    deepspeed_results = calculate_correlation(df, 'Deepspeed-MII')
    
    # 分析llama.cpp框架
    llamacpp_results = calculate_correlation(df, 'llama.cpp')
    
    # 汇总结果
    print("\n" + "="*60)
    print("汇总结果 (Summary Results)")
    print("="*60)
    
    results_list = []
    for results in [deepspeed_results, llamacpp_results]:
        if results is not None:
            results_list.append(results)
            print(f"\n{results['framework']}:")
            print(f"  数据点数量: {results['data_points']}")
            print(f"  皮尔逊相关系数: {results['pearson_correlation']:.4f}")
            print(f"  决定系数 (R²): {results['r_squared']:.4f}")
            print(f"  RMSE: {results['rmse']:.4f}")
            print(f"  MAPE: {results['mape']:.2f}%")
    
    # 绘制图表
    if results_list:
        plot_correlation(results_list)
    
    # 比较两个框架
    if deepspeed_results and llamacpp_results:
        print(f"\n框架比较:")
        print(f"DeepSpeed-MII vs llama.cpp:")
        print(f"  皮尔逊相关系数: {deepspeed_results['pearson_correlation']:.4f} vs {llamacpp_results['pearson_correlation']:.4f}")
        print(f"  R²: {deepspeed_results['r_squared']:.4f} vs {llamacpp_results['r_squared']:.4f}")
        print(f"  RMSE: {deepspeed_results['rmse']:.4f} vs {llamacpp_results['rmse']:.4f}")
        print(f"  MAPE: {deepspeed_results['mape']:.2f}% vs {llamacpp_results['mape']:.2f}%")

if __name__ == "__main__":
    main() 