import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式和字体大小
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# 输入和输出路径
BASE_DATA_PATH = 'data/All_results_with_predictions.csv'
XGBOOST_TRAIN_VAL_PATH = 'trained_models_auto_tuned/train_val_results_with_tuned_xgboost.csv'
XGBOOST_TEST_PATH = 'trained_models_auto_tuned/test_results_with_tuned_xgboost.csv'
OUTPUT_DIR = 'plot_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """加载所有数据文件"""
    print("Loading data files...")
    
    # 加载基础预测数据
    try:
        base_data = pd.read_csv(BASE_DATA_PATH)
        print(f"Loaded {len(base_data)} records from base prediction file")
        
        # 标准化列名
        base_col_mapping = {
            'Hardware': 'hardware',
            'Num of Hardware': 'num_devices',
            'Framework': 'framework',
            'Model': 'model_name',
            'Input Output Length': 'seq_len',
            'Batch Size': 'batch_size',
            'Latency': 'measured_latency',
            'Throughput': 'throughput',
            'Predicted_Latency': 'predicted_latency'
        }
        
        # 重命名列
        for old_col, new_col in base_col_mapping.items():
            if old_col in base_data.columns:
                base_data.rename(columns={old_col: new_col}, inplace=True)
    except Exception as e:
        print(f"Error loading base data: {e}")
        base_data = None
    
    # 加载XGBoost训练和验证数据
    try:
        xgb_train_val = pd.read_csv(XGBOOST_TRAIN_VAL_PATH)
        print(f"Loaded {len(xgb_train_val)} records from XGBoost train/val file")
    except Exception as e:
        print(f"Error loading XGBoost train/val data: {e}")
        xgb_train_val = None
    
    # 加载XGBoost测试数据
    try:
        xgb_test = pd.read_csv(XGBOOST_TEST_PATH)
        print(f"Loaded {len(xgb_test)} records from XGBoost test file")
    except Exception as e:
        print(f"Error loading XGBoost test data: {e}")
        xgb_test = None
    
    # 如果XGBoost数据存在，则尝试合并
    if xgb_train_val is not None and xgb_test is not None:
        try:
            xgb_data = pd.concat([xgb_train_val, xgb_test], ignore_index=True)
            print(f"Combined XGBoost data with {len(xgb_data)} records")
        except Exception as e:
            print(f"Error combining XGBoost data: {e}")
            xgb_data = xgb_train_val  # 如果合并失败，只使用训练/验证数据
    elif xgb_train_val is not None:
        xgb_data = xgb_train_val
    elif xgb_test is not None:
        xgb_data = xgb_test
    else:
        xgb_data = None
    
    return base_data, xgb_data

def calculate_metrics(df, actual_col='measured_latency', pred_col='predicted_latency'):
    """计算预测准确性指标"""
    if actual_col not in df.columns or pred_col not in df.columns:
        print(f"Missing columns for metrics calculation. Required: {actual_col}, {pred_col}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    metrics = {}
    # 均方根误差
    metrics['rmse'] = np.sqrt(np.mean((df[actual_col] - df[pred_col]) ** 2))
    # 平均绝对误差
    metrics['mae'] = np.mean(np.abs(df[actual_col] - df[pred_col]))
    # 平均绝对百分比误差
    metrics['mape'] = np.mean(np.abs((df[actual_col] - df[pred_col]) / df[actual_col])) * 100
    # 中位数绝对百分比误差
    metrics['mdape'] = np.median(np.abs((df[actual_col] - df[pred_col]) / df[actual_col])) * 100
    # 相关系数
    metrics['correlation'] = np.corrcoef(df[actual_col], df[pred_col])[0, 1]
    
    return metrics

def analyze_base_model(base_data):
    """分析基本模型的预测性能"""
    if base_data is None:
        print("No base data available for analysis")
        return
    
    print("\nAnalyzing base model predictions...")
    
    # 1. 计算基本指标
    metrics = calculate_metrics(base_data)
    if metrics:
        print(f"Base Model Metrics:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Median APE: {metrics['mdape']:.2f}%")
        print(f"Correlation: {metrics['correlation']:.4f}")
    
    # 2. 不同批次大小的预测准确性
    plt.figure(figsize=(12, 8))
    
    batch_sizes = sorted(base_data['batch_size'].unique())
    hardware_types = base_data['hardware'].unique()
    
    for hw in hardware_types[:3]:  # 限制只显示前3种硬件类型
        hw_data = base_data[base_data['hardware'] == hw]
        
        bs_results = []
        for bs in batch_sizes:
            bs_data = hw_data[hw_data['batch_size'] == bs]
            if len(bs_data) > 0:
                avg_measured = bs_data['measured_latency'].mean()
                avg_predicted = bs_data['predicted_latency'].mean()
                bs_results.append((bs, avg_measured, avg_predicted))
        
        if bs_results:
            bs_array = np.array(bs_results)
            plt.plot(bs_array[:, 0], bs_array[:, 1], 'o-', label=f'{hw} (Measured)')
            plt.plot(bs_array[:, 0], bs_array[:, 2], 'x--', label=f'{hw} (Predicted)')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title('Predicted vs Measured Latency by Batch Size')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_latency_by_batchsize.png'), dpi=300)
    
    # 3. 不同序列长度的预测准确性
    plt.figure(figsize=(12, 8))
    
    seq_lengths = sorted(base_data['seq_len'].unique())
    hardware_types = base_data['hardware'].unique()
    
    for hw in hardware_types[:3]:  # 限制只显示前3种硬件类型
        hw_data = base_data[base_data['hardware'] == hw]
        
        seq_results = []
        for seq in seq_lengths:
            seq_data = hw_data[hw_data['seq_len'] == seq]
            if len(seq_data) > 0:
                avg_measured = seq_data['measured_latency'].mean()
                avg_predicted = seq_data['predicted_latency'].mean()
                seq_results.append((seq, avg_measured, avg_predicted))
        
        if seq_results:
            seq_array = np.array(seq_results)
            plt.plot(seq_array[:, 0], seq_array[:, 1], 'o-', label=f'{hw} (Measured)')
            plt.plot(seq_array[:, 0], seq_array[:, 2], 'x--', label=f'{hw} (Predicted)')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.title('Predicted vs Measured Latency by Sequence Length')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_latency_by_seqlen.png'), dpi=300)
    
    # 4. 不同硬件平台的预测误差分析
    plt.figure(figsize=(14, 8))
    
    hardware_errors = []
    for hw in hardware_types:
        hw_data = base_data[base_data['hardware'] == hw]
        if len(hw_data) > 0:
            mape = np.mean(np.abs((hw_data['measured_latency'] - hw_data['predicted_latency']) / hw_data['measured_latency'])) * 100
            hardware_errors.append((hw, mape, len(hw_data)))
    
    # 按错误率排序
    hardware_errors.sort(key=lambda x: x[1])
    
    hw_names = [item[0] for item in hardware_errors]
    hw_errors = [item[1] for item in hardware_errors]
    hw_counts = [item[2] for item in hardware_errors]
    
    # 绘制条形图
    bars = plt.bar(hw_names, hw_errors, alpha=0.7)
    
    # 在条形上方添加样本数量
    for i, (bar, count) in enumerate(zip(bars, hw_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Hardware Platform')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.title('Base Model Prediction Error by Hardware Platform')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_error_by_hardware.png'), dpi=300)
    
    # 5. 不同设备数量的预测误差分析
    plt.figure(figsize=(12, 8))
    
    device_counts = sorted(base_data['num_devices'].unique())
    
    device_errors = []
    for dc in device_counts:
        dc_data = base_data[base_data['num_devices'] == dc]
        if len(dc_data) > 0:
            mape = np.mean(np.abs((dc_data['measured_latency'] - dc_data['predicted_latency']) / dc_data['measured_latency'])) * 100
            device_errors.append((dc, mape, len(dc_data)))
    
    dc_values = [item[0] for item in device_errors]
    dc_errors = [item[1] for item in device_errors]
    dc_counts = [item[2] for item in device_errors]
    
    # 绘制折线图
    plt.plot(dc_values, dc_errors, 'o-', linewidth=2, markersize=10)
    
    # 在点上方添加样本数量
    for i, (x, y, count) in enumerate(zip(dc_values, dc_errors, dc_counts)):
        plt.text(x, y + 0.5, f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Number of Devices')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.title('Base Model Prediction Error by Number of Devices')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_error_by_num_devices.png'), dpi=300)
    
    # 6. 不同框架的预测误差分析
    plt.figure(figsize=(12, 8))
    
    framework_types = base_data['framework'].unique()
    
    framework_errors = []
    for fw in framework_types:
        fw_data = base_data[base_data['framework'] == fw]
        if len(fw_data) > 0:
            mape = np.mean(np.abs((fw_data['measured_latency'] - fw_data['predicted_latency']) / fw_data['measured_latency'])) * 100
            framework_errors.append((fw, mape, len(fw_data)))
    
    # 按错误率排序
    framework_errors.sort(key=lambda x: x[1])
    
    fw_names = [item[0] for item in framework_errors]
    fw_errors = [item[1] for item in framework_errors]
    fw_counts = [item[2] for item in framework_errors]
    
    # 绘制条形图
    bars = plt.bar(fw_names, fw_errors, alpha=0.7)
    
    # 在条形上方添加样本数量
    for i, (bar, count) in enumerate(zip(bars, fw_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Framework')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.title('Base Model Prediction Error by Framework')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_error_by_framework.png'), dpi=300)

def analyze_xgboost_model(xgb_data, base_data=None):
    """分析XGBoost模型的预测性能"""
    if xgb_data is None:
        print("No XGBoost data available for analysis")
        return
    
    print("\nAnalyzing XGBoost predictions...")
    print(f"XGBoost data columns: {xgb_data.columns.tolist()}")
    
    # 确认正确的列名
    if 'xgboost_latency_tuned' in xgb_data.columns:
        xgb_pred_col = 'xgboost_latency_tuned'
    elif 'xgboost_latency' in xgb_data.columns:
        xgb_pred_col = 'xgboost_latency'
    else:
        print("Could not find XGBoost prediction column. Using predicted_latency as baseline only.")
        xgb_pred_col = None
    
    # 分析准确性
    if xgb_pred_col:
        # 计算基本指标
        metrics = calculate_metrics(xgb_data, pred_col=xgb_pred_col)
        base_metrics = calculate_metrics(xgb_data)
        
        if metrics and base_metrics:
            print(f"XGBoost Model Metrics:")
            print(f"RMSE: {metrics['rmse']:.4f} (Base: {base_metrics['rmse']:.4f})")
            print(f"MAE: {metrics['mae']:.4f} (Base: {base_metrics['mae']:.4f})")
            print(f"MAPE: {metrics['mape']:.2f}% (Base: {base_metrics['mape']:.2f}%)")
            print(f"Median APE: {metrics['mdape']:.2f}% (Base: {base_metrics['mdape']:.2f}%)")
            print(f"Correlation: {metrics['correlation']:.4f} (Base: {base_metrics['correlation']:.4f})")
        
        # 计算百分比误差
        xgb_data['base_error_pct'] = 100 * np.abs(xgb_data['measured_latency'] - xgb_data['predicted_latency']) / xgb_data['measured_latency']
        xgb_data['xgboost_error_pct'] = 100 * np.abs(xgb_data['measured_latency'] - xgb_data[xgb_pred_col]) / xgb_data['measured_latency']
        
        # 1. 实际值与预测值的散点图
        plt.figure(figsize=(12, 10))
        
        # 基础模型散点图
        plt.scatter(xgb_data['measured_latency'], xgb_data['predicted_latency'], 
                    alpha=0.5, label='Base Model', s=40, color='blue')
        
        # XGBoost模型散点图
        plt.scatter(xgb_data['measured_latency'], xgb_data[xgb_pred_col], 
                    alpha=0.5, label='XGBoost Model', s=40, color='red')
        
        # 完美预测线
        max_val = max(xgb_data['measured_latency'].max(), xgb_data[xgb_pred_col].max())
        min_val = min(xgb_data['measured_latency'].min(), xgb_data[xgb_pred_col].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        plt.xlabel('Measured Latency (ms)')
        plt.ylabel('Predicted Latency (ms)')
        plt.title('Measured vs Predicted Latency (All Data)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_vs_base_all.png'), dpi=300)
        
        # 2. 移除离群值后的散点图 (中间50%)
        plt.figure(figsize=(12, 10))
        
        # 计算测量延迟的25%和75%分位数
        q25 = xgb_data['measured_latency'].quantile(0.25)
        q75 = xgb_data['measured_latency'].quantile(0.75)
        
        # 过滤出中间50%的数据
        filtered_data = xgb_data[(xgb_data['measured_latency'] >= q25) & (xgb_data['measured_latency'] <= q75)]
        
        print(f"Filtered to middle 50% of data: {len(filtered_data)} records (from {len(xgb_data)})")
        
        # 基础模型散点图
        plt.scatter(filtered_data['measured_latency'], filtered_data['predicted_latency'], 
                    alpha=0.5, label='Base Model', s=40, color='blue')
        
        # XGBoost模型散点图
        plt.scatter(filtered_data['measured_latency'], filtered_data[xgb_pred_col], 
                    alpha=0.5, label='XGBoost Model', s=40, color='red')
        
        # 完美预测线
        max_val = max(filtered_data['measured_latency'].max(), filtered_data[xgb_pred_col].max())
        min_val = min(filtered_data['measured_latency'].min(), filtered_data[xgb_pred_col].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        plt.xlabel('Measured Latency (ms)')
        plt.ylabel('Predicted Latency (ms)')
        plt.title('Measured vs Predicted Latency (Middle 50% of Data)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_vs_base_middle50.png'), dpi=300)
        
        # 3. 百分比误差分布直方图
        plt.figure(figsize=(12, 8))
        
        # 设置合理的上限，例如50%
        max_error_to_show = 50
        
        # 过滤出误差在范围内的数据
        base_errors_to_show = xgb_data['base_error_pct'][xgb_data['base_error_pct'] <= max_error_to_show]
        xgb_errors_to_show = xgb_data['xgboost_error_pct'][xgb_data['xgboost_error_pct'] <= max_error_to_show]
        
        plt.hist(base_errors_to_show, bins=30, alpha=0.5, label='Base Model', color='blue')
        plt.hist(xgb_errors_to_show, bins=30, alpha=0.5, label='XGBoost Model', color='red')
        
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution (Errors <= {max_error_to_show}%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_error_distribution.png'), dpi=300)
        
        # 4. 绘制累积误差分布曲线
        plt.figure(figsize=(12, 8))
        
        # 排序误差
        sorted_base_errors = np.sort(xgb_data['base_error_pct'])
        sorted_xgb_errors = np.sort(xgb_data['xgboost_error_pct'])
        
        # 计算累积分布
        y_values = np.arange(1, len(sorted_base_errors) + 1) / len(sorted_base_errors)
        
        # 绘制CDF
        plt.plot(sorted_base_errors, y_values, 'b-', linewidth=2, label='Base Model')
        plt.plot(sorted_xgb_errors, y_values, 'r-', linewidth=2, label='XGBoost Model')
        
        # 添加辅助线
        plt.axvline(x=10, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
        
        # 查找10%误差对应的百分位
        base_10pct_percentile = np.mean(sorted_base_errors <= 10) * 100
        xgb_10pct_percentile = np.mean(sorted_xgb_errors <= 10) * 100
        
        # 添加注释
        plt.text(11, 0.2, f'Base: {base_10pct_percentile:.1f}% of predictions\nwithin 10% error', color='blue')
        plt.text(11, 0.4, f'XGBoost: {xgb_10pct_percentile:.1f}% of predictions\nwithin 10% error', color='red')
        
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'xgboost_error_cdf.png'), dpi=300)

def main():
    """主函数"""
    # 加载数据
    base_data, xgb_data = load_data()
    
    # 分析基本模型
    if base_data is not None:
        analyze_base_model(base_data)
    
    # 分析XGBoost模型
    if xgb_data is not None:
        analyze_xgboost_model(xgb_data, base_data)
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 