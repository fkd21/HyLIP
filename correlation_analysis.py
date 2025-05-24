import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set plot style and font sizes
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

# Input and output paths
BASE_DATA_PATH = 'data/All_results_with_predictions.csv'
XGBOOST_TRAIN_VAL_PATH = 'trained_models_auto_tuned/train_val_results_with_tuned_xgboost.csv'
XGBOOST_TEST_PATH = 'trained_models_auto_tuned/test_results_with_tuned_xgboost.csv'
OUTPUT_DIR = 'plot_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load all data files"""
    print("Loading data files...")
    
    # Load base prediction data
    try:
        base_data = pd.read_csv(BASE_DATA_PATH)
        print(f"Loaded {len(base_data)} records from base prediction file")
        
        # Standardize column names
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
        
        # Rename columns
        for old_col, new_col in base_col_mapping.items():
            if old_col in base_data.columns:
                base_data.rename(columns={old_col: new_col}, inplace=True)
    except Exception as e:
        print(f"Error loading base data: {e}")
        base_data = None
    
    # Load XGBoost training and validation data
    try:
        xgb_train_val = pd.read_csv(XGBOOST_TRAIN_VAL_PATH)
        print(f"Loaded {len(xgb_train_val)} records from XGBoost train/val file")
    except Exception as e:
        print(f"Error loading XGBoost train/val data: {e}")
        xgb_train_val = None
    
    # Load XGBoost test data
    try:
        xgb_test = pd.read_csv(XGBOOST_TEST_PATH)
        print(f"Loaded {len(xgb_test)} records from XGBoost test file")
    except Exception as e:
        print(f"Error loading XGBoost test data: {e}")
        xgb_test = None
    
    # Merge XGBoost data if available
    if xgb_train_val is not None and xgb_test is not None:
        try:
            xgb_data = pd.concat([xgb_train_val, xgb_test], ignore_index=True)
            print(f"Combined XGBoost data with {len(xgb_data)} records")
        except Exception as e:
            print(f"Error combining XGBoost data: {e}")
            xgb_data = xgb_train_val
    elif xgb_train_val is not None:
        xgb_data = xgb_train_val
    elif xgb_test is not None:
        xgb_data = xgb_test
    else:
        xgb_data = None
    
    return base_data, xgb_data

def correlation_analysis(base_data):
    """Analyze correlation between measured and predicted latency"""
    if base_data is None:
        print("No base data available for analysis")
        return
    
    print("\nAnalyzing base model predictions using correlation analysis...")
    
    # 1. Overall correlation scatter plot
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation coefficient
    corr, p_value = pearsonr(base_data['measured_latency'], base_data['predicted_latency'])
    
    # Create scatter plot
    plt.scatter(base_data['measured_latency'], base_data['predicted_latency'], 
                alpha=0.6, s=40, c='blue', edgecolor='k', linewidth=0.5)
    
    # Add perfect prediction line
    max_val = max(base_data['measured_latency'].max(), base_data['predicted_latency'].max())
    min_val = min(base_data['measured_latency'].min(), base_data['predicted_latency'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Add correlation coefficient as text
    plt.annotate(f"Correlation: {corr:.4f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=14, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('Measured Latency (ms)')
    plt.ylabel('Predicted Latency (ms)')
    plt.title('Correlation Between Measured and Predicted Latency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_correlation_overall.png'), dpi=300)
    
    # 2. Correlation by hardware type
    plt.figure(figsize=(14, 10))
    
    hardware_types = base_data['hardware'].unique()
    
    correlation_by_hw = []
    for i, hw in enumerate(hardware_types):
        hw_data = base_data[base_data['hardware'] == hw]
        if len(hw_data) > 10:  # Only consider hardware types with sufficient data
            hw_corr, _ = pearsonr(hw_data['measured_latency'], hw_data['predicted_latency'])
            correlation_by_hw.append((hw, hw_corr, len(hw_data)))
    
    # Sort by correlation value
    correlation_by_hw.sort(key=lambda x: x[1], reverse=True)
    
    hw_names = [item[0] for item in correlation_by_hw]
    hw_corr = [item[1] for item in correlation_by_hw]
    hw_counts = [item[2] for item in correlation_by_hw]
    
    # Create horizontal bar chart
    bars = plt.barh(hw_names, hw_corr, alpha=0.7, color='skyblue')
    
    # Add sample count to bars
    for i, (bar, count) in enumerate(zip(bars, hw_counts)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'n={count}', va='center', fontsize=10)
    
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.xlim(-1, 1)
    plt.xlabel('Correlation Coefficient (r)')
    plt.title('Correlation Between Measured and Predicted Latency by Hardware Platform')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_correlation_by_hardware.png'), dpi=300)
    
    # 3. Correlation by framework
    plt.figure(figsize=(12, 8))
    
    framework_types = base_data['framework'].unique()
    
    correlation_by_fw = []
    for fw in framework_types:
        fw_data = base_data[base_data['framework'] == fw]
        if len(fw_data) > 10:  # Only consider frameworks with sufficient data
            fw_corr, _ = pearsonr(fw_data['measured_latency'], fw_data['predicted_latency'])
            correlation_by_fw.append((fw, fw_corr, len(fw_data)))
    
    # Sort by correlation value
    correlation_by_fw.sort(key=lambda x: x[1], reverse=True)
    
    fw_names = [item[0] for item in correlation_by_fw]
    fw_corr = [item[1] for item in correlation_by_fw]
    fw_counts = [item[2] for item in correlation_by_fw]
    
    # Create horizontal bar chart
    bars = plt.barh(fw_names, fw_corr, alpha=0.7, color='lightgreen')
    
    # Add sample count to bars
    for i, (bar, count) in enumerate(zip(bars, fw_counts)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'n={count}', va='center', fontsize=10)
    
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.xlim(-1, 1)
    plt.xlabel('Correlation Coefficient (r)')
    plt.title('Correlation Between Measured and Predicted Latency by Framework')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_correlation_by_framework.png'), dpi=300)
    
    # 4. Individual scatter plots for top 4 hardware platforms
    plt.figure(figsize=(15, 12))
    
    # Get the top 4 hardware platforms by data count
    top_hw = sorted(correlation_by_hw, key=lambda x: x[2], reverse=True)[:4]
    
    for i, (hw, corr, count) in enumerate(top_hw):
        plt.subplot(2, 2, i+1)
        
        hw_data = base_data[base_data['hardware'] == hw]
        
        plt.scatter(hw_data['measured_latency'], hw_data['predicted_latency'], 
                   alpha=0.6, s=30, c='blue', edgecolor='k', linewidth=0.5)
        
        max_val = max(hw_data['measured_latency'].max(), hw_data['predicted_latency'].max())
        min_val = min(hw_data['measured_latency'].min(), hw_data['predicted_latency'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.annotate(f"Correlation: {corr:.4f}\nn={count}", xy=(0.05, 0.90), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.xlabel('Measured Latency (ms)')
        plt.ylabel('Predicted Latency (ms)')
        plt.title(f'{hw}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_correlation_top_hardware.png'), dpi=300)
    
    # 5. Batch size vs. correlation analysis
    plt.figure(figsize=(12, 8))
    
    batch_sizes = sorted(base_data['batch_size'].unique())
    
    batch_correlations = []
    for bs in batch_sizes:
        bs_data = base_data[base_data['batch_size'] == bs]
        if len(bs_data) > 10:  # Only consider batch sizes with sufficient data
            bs_corr, _ = pearsonr(bs_data['measured_latency'], bs_data['predicted_latency'])
            batch_correlations.append((bs, bs_corr, len(bs_data)))
    
    bs_values = [item[0] for item in batch_correlations]
    bs_corrs = [item[1] for item in batch_correlations]
    bs_counts = [item[2] for item in batch_correlations]
    
    # Create line chart
    plt.plot(bs_values, bs_corrs, 'o-', linewidth=2, markersize=10, color='purple')
    
    # Add sample counts
    for i, (x, y, count) in enumerate(zip(bs_values, bs_corrs, bs_counts)):
        plt.annotate(f"n={count}", (x, y), xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10)
    
    plt.grid(True)
    plt.xlabel('Batch Size')
    plt.ylabel('Correlation Coefficient (r)')
    plt.title('Correlation Between Measured and Predicted Latency by Batch Size')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_correlation_by_batchsize.png'), dpi=300)
    
    # 6. Sequence length vs. correlation analysis
    plt.figure(figsize=(12, 8))
    
    seq_lengths = sorted(base_data['seq_len'].unique())
    
    seq_correlations = []
    for seq in seq_lengths:
        seq_data = base_data[base_data['seq_len'] == seq]
        if len(seq_data) > 10:  # Only consider sequence lengths with sufficient data
            seq_corr, _ = pearsonr(seq_data['measured_latency'], seq_data['predicted_latency'])
            seq_correlations.append((seq, seq_corr, len(seq_data)))
    
    seq_values = [item[0] for item in seq_correlations]
    seq_corrs = [item[1] for item in seq_correlations]
    seq_counts = [item[2] for item in seq_correlations]
    
    # Create line chart
    plt.plot(seq_values, seq_corrs, 'o-', linewidth=2, markersize=10, color='green')
    
    # Add sample counts
    for i, (x, y, count) in enumerate(zip(seq_values, seq_corrs, seq_counts)):
        plt.annotate(f"n={count}", (x, y), xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10)
    
    plt.grid(True)
    plt.xlabel('Sequence Length')
    plt.ylabel('Correlation Coefficient (r)')
    plt.title('Correlation Between Measured and Predicted Latency by Sequence Length')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'base_correlation_by_seqlen.png'), dpi=300)
    
    return

def comparative_correlation_analysis(base_data, xgb_data):
    """Compare correlation between base model and XGBoost model"""
    if base_data is None or xgb_data is None:
        print("Missing data for comparative analysis")
        return
    
    print("\nComparing base model and XGBoost model correlations...")
    
    # Identify XGBoost prediction column
    if 'xgboost_latency_tuned' in xgb_data.columns:
        xgb_pred_col = 'xgboost_latency_tuned'
    elif 'xgboost_latency' in xgb_data.columns:
        xgb_pred_col = 'xgboost_latency'
    else:
        print("Could not find XGBoost prediction column")
        return
    
    # 1. Side-by-side scatter plots
    plt.figure(figsize=(20, 10))
    
    # Base model
    plt.subplot(1, 2, 1)
    
    base_corr, _ = pearsonr(base_data['measured_latency'], base_data['predicted_latency'])
    
    plt.scatter(base_data['measured_latency'], base_data['predicted_latency'], 
               alpha=0.6, s=40, c='blue', edgecolor='k', linewidth=0.5)
    
    max_val = max(base_data['measured_latency'].max(), base_data['predicted_latency'].max())
    min_val = min(base_data['measured_latency'].min(), base_data['predicted_latency'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.annotate(f"Correlation: {base_corr:.4f}", xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('Measured Latency (ms)')
    plt.ylabel('Predicted Latency (ms)')
    plt.title('Base Model')
    plt.grid(True, alpha=0.3)
    
    # XGBoost model
    plt.subplot(1, 2, 2)
    
    xgb_corr, _ = pearsonr(xgb_data['measured_latency'], xgb_data[xgb_pred_col])
    
    plt.scatter(xgb_data['measured_latency'], xgb_data[xgb_pred_col], 
               alpha=0.6, s=40, c='red', edgecolor='k', linewidth=0.5)
    
    max_val = max(xgb_data['measured_latency'].max(), xgb_data[xgb_pred_col].max())
    min_val = min(xgb_data['measured_latency'].min(), xgb_data[xgb_pred_col].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.annotate(f"Correlation: {xgb_corr:.4f}", xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('Measured Latency (ms)')
    plt.ylabel('Predicted Latency (ms)')
    plt.title('XGBoost Model')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_comparison.png'), dpi=300)
    
    # 2. Hardware-specific correlation comparison
    if 'hardware' in xgb_data.columns and 'hardware_encoded' not in xgb_data.columns:
        plt.figure(figsize=(15, 10))
        
        hardware_types = set(base_data['hardware'].unique())
        
        base_hw_corrs = []
        xgb_hw_corrs = []
        hw_labels = []
        hw_counts = []
        
        for hw in hardware_types:
            base_hw_data = base_data[base_data['hardware'] == hw]
            xgb_hw_data = xgb_data[xgb_data['hardware'] == hw]
            
            if len(base_hw_data) > 10 and len(xgb_hw_data) > 10:  # Only consider hardware types with sufficient data
                base_hw_corr, _ = pearsonr(base_hw_data['measured_latency'], base_hw_data['predicted_latency'])
                xgb_hw_corr, _ = pearsonr(xgb_hw_data['measured_latency'], xgb_hw_data[xgb_pred_col])
                
                base_hw_corrs.append(base_hw_corr)
                xgb_hw_corrs.append(xgb_hw_corr)
                hw_labels.append(hw)
                hw_counts.append(len(base_hw_data))
        
        # Sort by base model correlation for clear comparison
        hw_data = sorted(zip(hw_labels, base_hw_corrs, xgb_hw_corrs, hw_counts), key=lambda x: x[1])
        hw_labels = [item[0] for item in hw_data]
        base_hw_corrs = [item[1] for item in hw_data]
        xgb_hw_corrs = [item[2] for item in hw_data]
        hw_counts = [item[3] for item in hw_data]
        
        # Create paired bar chart
        width = 0.35
        x = np.arange(len(hw_labels))
        
        plt.bar(x - width/2, base_hw_corrs, width, label='Base Model', alpha=0.7, color='blue')
        plt.bar(x + width/2, xgb_hw_corrs, width, label='XGBoost Model', alpha=0.7, color='red')
        
        # Add sample counts
        for i, count in enumerate(hw_counts):
            plt.annotate(f"n={count}", (i, min(base_hw_corrs[i], xgb_hw_corrs[i]) - 0.05), 
                       ha='center', fontsize=10)
        
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(-1, 1)
        plt.ylabel('Correlation Coefficient (r)')
        plt.title('Correlation Comparison by Hardware Platform')
        plt.xticks(x, hw_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_comparison_by_hardware.png'), dpi=300)
    
    return

def main():
    """Main function"""
    # Load data
    base_data, xgb_data = load_data()
    
    # Analyze base model correlations
    if base_data is not None:
        correlation_analysis(base_data)
    
    # Compare base model and XGBoost correlations
    if base_data is not None and xgb_data is not None:
        comparative_correlation_analysis(base_data, xgb_data)
    
    print(f"\nCorrelation analysis complete. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 