#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze prediction results and generate plots')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with predictions')
    parser.add_argument('--xgboost', type=str, help='XGBoost predictions CSV file (optional)')
    parser.add_argument('--output_dir', type=str, default='plot', help='Directory to save plots')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved plots')
    parser.add_argument('--no_show', action='store_true', help='Do not show plots, only save')
    return parser.parse_args()

def load_data(input_file, xgboost_file=None):
    """Load and prepare the data for analysis"""
    logger.info(f"Loading data from {input_file}")
    
    try:
        # Load the main data file
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        
        # Check column names and rename if necessary for standardization
        col_mapping = {
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
        
        # Rename columns that exist in the mapping
        for old_col, new_col in col_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # If XGBoost predictions file is provided, merge it
        if xgboost_file and os.path.exists(xgboost_file):
            logger.info(f"Loading XGBoost predictions from {xgboost_file}")
            xgb_df = pd.read_csv(xgboost_file)
            
            # Check if XGBoost predictions column exists
            xgb_col = None
            for col in xgb_df.columns:
                if 'xgboost' in col.lower() or 'predict' in col.lower():
                    if col not in df.columns:
                        xgb_col = col
                        break
            
            if xgb_col:
                # Extract only the prediction column
                xgb_pred = xgb_df[xgb_col]
                
                # Add XGBoost predictions to main dataframe
                if len(xgb_pred) == len(df):
                    df['xgboost_prediction'] = xgb_pred.values
                    logger.info(f"Added XGBoost predictions column '{xgb_col}' to the dataframe")
                else:
                    logger.warning(f"XGBoost predictions length ({len(xgb_pred)}) doesn't match main data ({len(df)})")
            else:
                logger.warning("No XGBoost prediction column found in the XGBoost file")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def calculate_metrics(df):
    """Calculate prediction performance metrics"""
    metrics = {}
    
    if 'measured_latency' in df.columns and 'predicted_latency' in df.columns:
        y_true = df['measured_latency']
        y_pred = df['predicted_latency']
        
        metrics['base_model'] = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Calculate percentage error
        df['base_pct_error'] = 100 * np.abs(y_true - y_pred) / y_true
    
    if 'measured_latency' in df.columns and 'xgboost_prediction' in df.columns:
        y_true = df['measured_latency']
        y_pred = df['xgboost_prediction']
        
        metrics['xgboost_model'] = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Calculate percentage error
        df['xgb_pct_error'] = 100 * np.abs(y_true - y_pred) / y_true
    
    return metrics, df

def plot_prediction_comparison(df, output_dir, dpi=300, show_plots=True):
    """Generate plots comparing actual vs predicted latencies"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Clear any existing plots
    plt.close('all')
    
    # 1. Scatter plot of actual vs predicted latency
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Base model predictions
    ax.scatter(df['measured_latency'], df['predicted_latency'], 
               alpha=0.6, label='Base Model Predictions', s=40)
    
    # XGBoost predictions if available
    if 'xgboost_prediction' in df.columns:
        ax.scatter(df['measured_latency'], df['xgboost_prediction'], 
                   alpha=0.6, label='XGBoost Predictions', s=40)
    
    # Perfect prediction line
    max_val = max(df['measured_latency'].max(), df['predicted_latency'].max())
    min_val = min(df['measured_latency'].min(), df['predicted_latency'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('Measured Latency (ms)', fontsize=12)
    ax.set_ylabel('Predicted Latency (ms)', fontsize=12)
    ax.set_title('Actual vs Predicted Latency', fontsize=14)
    ax.legend(fontsize=12)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    logger.info(f"Saved scatter plot to {plot_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Error distribution histogram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'base_pct_error' in df.columns:
        ax.hist(df['base_pct_error'], bins=30, alpha=0.5, label='Base Model')
    
    if 'xgb_pct_error' in df.columns:
        ax.hist(df['xgb_pct_error'], bins=30, alpha=0.5, label='XGBoost Model')
    
    ax.set_xlabel('Percentage Error (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Error Distribution', fontsize=14)
    ax.legend(fontsize=12)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'error_distribution.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    logger.info(f"Saved error distribution plot to {plot_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 3. Error by hardware type
    if 'hardware' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        hardware_types = df['hardware'].unique()
        
        # Calculate mean percentage error by hardware
        hardware_errors = []
        for hw in hardware_types:
            hw_data = df[df['hardware'] == hw]
            
            base_error = hw_data['base_pct_error'].mean() if 'base_pct_error' in df.columns else None
            xgb_error = hw_data['xgb_pct_error'].mean() if 'xgb_pct_error' in df.columns else None
            
            hardware_errors.append({
                'hardware': hw,
                'base_error': base_error,
                'xgb_error': xgb_error
            })
        
        hw_df = pd.DataFrame(hardware_errors)
        
        # Plot
        x = np.arange(len(hardware_types))
        width = 0.35
        
        if 'base_error' in hw_df.columns:
            ax.bar(x - width/2, hw_df['base_error'], width, label='Base Model')
        
        if 'xgb_error' in hw_df.columns and not hw_df['xgb_error'].isna().all():
            ax.bar(x + width/2, hw_df['xgb_error'], width, label='XGBoost Model')
        
        ax.set_xlabel('Hardware Type', fontsize=12)
        ax.set_ylabel('Mean Percentage Error (%)', fontsize=12)
        ax.set_title('Prediction Error by Hardware Type', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(hardware_types, rotation=45, ha='right')
        ax.legend(fontsize=12)
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'error_by_hardware.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=dpi)
        logger.info(f"Saved hardware error plot to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 4. Error by model type
    if 'model_name' in df.columns:
        # Create a simpler model name for display
        df['model_short'] = df['model_name'].apply(lambda x: x.split('/')[-1] if '/' in str(x) else x)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        model_types = df['model_short'].unique()
        
        # Calculate mean percentage error by model
        model_errors = []
        for model in model_types:
            model_data = df[df['model_short'] == model]
            
            base_error = model_data['base_pct_error'].mean() if 'base_pct_error' in df.columns else None
            xgb_error = model_data['xgb_pct_error'].mean() if 'xgb_pct_error' in df.columns else None
            
            model_errors.append({
                'model': model,
                'base_error': base_error,
                'xgb_error': xgb_error
            })
        
        model_df = pd.DataFrame(model_errors)
        
        # Plot
        x = np.arange(len(model_types))
        width = 0.35
        
        if 'base_error' in model_df.columns:
            ax.bar(x - width/2, model_df['base_error'], width, label='Base Model')
        
        if 'xgb_error' in model_df.columns and not model_df['xgb_error'].isna().all():
            ax.bar(x + width/2, model_df['xgb_error'], width, label='XGBoost Model')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Mean Percentage Error (%)', fontsize=12)
        ax.set_title('Prediction Error by Model Type', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_types, rotation=45, ha='right')
        ax.legend(fontsize=12)
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'error_by_model.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=dpi)
        logger.info(f"Saved model error plot to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 5. Error by sequence length
    if 'seq_len' in df.columns:
        # Group by sequence length
        seq_groups = df.groupby('seq_len')
        
        seq_errors = []
        for seq_len, group in seq_groups:
            base_error = group['base_pct_error'].mean() if 'base_pct_error' in df.columns else None
            xgb_error = group['xgb_pct_error'].mean() if 'xgb_pct_error' in df.columns else None
            
            seq_errors.append({
                'seq_len': seq_len,
                'base_error': base_error,
                'xgb_error': xgb_error,
                'count': len(group)
            })
        
        seq_df = pd.DataFrame(seq_errors).sort_values('seq_len')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if 'base_error' in seq_df.columns:
            ax.plot(seq_df['seq_len'], seq_df['base_error'], 'o-', label='Base Model')
        
        if 'xgb_error' in seq_df.columns and not seq_df['xgb_error'].isna().all():
            ax.plot(seq_df['seq_len'], seq_df['xgb_error'], 'o-', label='XGBoost Model')
        
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Mean Percentage Error (%)', fontsize=12)
        ax.set_title('Prediction Error by Sequence Length', fontsize=14)
        ax.legend(fontsize=12)
        
        # Add sample count as text
        for i, row in seq_df.iterrows():
            ax.annotate(f"n={row['count']}", 
                      (row['seq_len'], max(row['base_error'] if row['base_error'] is not None else 0, 
                                            row['xgb_error'] if row['xgb_error'] is not None else 0)),
                      textcoords="offset points", 
                      xytext=(0,10), 
                      ha='center')
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'error_by_seq_len.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=dpi)
        logger.info(f"Saved sequence length error plot to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 6. Error by framework
    if 'framework' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        framework_types = df['framework'].unique()
        
        # Calculate mean percentage error by framework
        framework_errors = []
        for fw in framework_types:
            fw_data = df[df['framework'] == fw]
            
            base_error = fw_data['base_pct_error'].mean() if 'base_pct_error' in df.columns else None
            xgb_error = fw_data['xgb_pct_error'].mean() if 'xgb_pct_error' in df.columns else None
            
            framework_errors.append({
                'framework': fw,
                'base_error': base_error,
                'xgb_error': xgb_error
            })
        
        fw_df = pd.DataFrame(framework_errors)
        
        # Plot
        x = np.arange(len(framework_types))
        width = 0.35
        
        if 'base_error' in fw_df.columns:
            ax.bar(x - width/2, fw_df['base_error'], width, label='Base Model')
        
        if 'xgb_error' in fw_df.columns and not fw_df['xgb_error'].isna().all():
            ax.bar(x + width/2, fw_df['xgb_error'], width, label='XGBoost Model')
        
        ax.set_xlabel('Framework', fontsize=12)
        ax.set_ylabel('Mean Percentage Error (%)', fontsize=12)
        ax.set_title('Prediction Error by Framework', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(framework_types, rotation=45, ha='right')
        ax.legend(fontsize=12)
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'error_by_framework.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=dpi)
        logger.info(f"Saved framework error plot to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 7. Heatmap of correlations
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove error columns from correlation analysis
    numeric_cols = [col for col in numeric_cols if 'error' not in col]
    
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Variables', fontsize=14)
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=dpi)
        logger.info(f"Saved correlation heatmap to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 8. Effect of num_devices on prediction error
    if 'num_devices' in df.columns:
        # Group by number of devices
        device_groups = df.groupby('num_devices')
        
        device_errors = []
        for num_devices, group in device_groups:
            base_error = group['base_pct_error'].mean() if 'base_pct_error' in df.columns else None
            xgb_error = group['xgb_pct_error'].mean() if 'xgb_pct_error' in df.columns else None
            
            device_errors.append({
                'num_devices': num_devices,
                'base_error': base_error,
                'xgb_error': xgb_error,
                'count': len(group)
            })
        
        device_df = pd.DataFrame(device_errors).sort_values('num_devices')
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'base_error' in device_df.columns:
            ax.plot(device_df['num_devices'], device_df['base_error'], 'o-', label='Base Model')
        
        if 'xgb_error' in device_df.columns and not device_df['xgb_error'].isna().all():
            ax.plot(device_df['num_devices'], device_df['xgb_error'], 'o-', label='XGBoost Model')
        
        ax.set_xlabel('Number of Devices', fontsize=12)
        ax.set_ylabel('Mean Percentage Error (%)', fontsize=12)
        ax.set_title('Prediction Error by Number of Devices', fontsize=14)
        ax.legend(fontsize=12)
        
        # Add sample count as text
        for i, row in device_df.iterrows():
            ax.annotate(f"n={row['count']}", 
                      (row['num_devices'], max(row['base_error'] if row['base_error'] is not None else 0, 
                                               row['xgb_error'] if row['xgb_error'] is not None else 0)),
                      textcoords="offset points", 
                      xytext=(0,10), 
                      ha='center')
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'error_by_num_devices.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=dpi)
        logger.info(f"Saved devices error plot to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

def generate_analysis_report(df, metrics, output_dir):
    """Generate a text report summarizing the analysis"""
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("# Latency Prediction Analysis Report\n\n")
        
        # Dataset summary
        f.write("## Dataset Summary\n")
        f.write(f"- Total number of samples: {len(df)}\n")
        
        if 'hardware' in df.columns:
            f.write(f"- Hardware types: {len(df['hardware'].unique())}\n")
            for hw in df['hardware'].unique():
                f.write(f"  - {hw}: {len(df[df['hardware'] == hw])} samples\n")
        
        if 'model_name' in df.columns:
            f.write(f"- Model types: {len(df['model_name'].unique())}\n")
            
        if 'framework' in df.columns:
            f.write(f"- Frameworks: {df['framework'].unique().tolist()}\n")
            
        if 'seq_len' in df.columns:
            f.write(f"- Sequence length range: {df['seq_len'].min()} to {df['seq_len'].max()}\n")
            
        if 'batch_size' in df.columns:
            f.write(f"- Batch size range: {df['batch_size'].min()} to {df['batch_size'].max()}\n")
            
        if 'num_devices' in df.columns:
            f.write(f"- Number of devices range: {df['num_devices'].min()} to {df['num_devices'].max()}\n")
        
        f.write("\n")
        
        # Prediction metrics
        f.write("## Prediction Performance Metrics\n")
        
        if 'base_model' in metrics:
            base_metrics = metrics['base_model']
            f.write("### Base Model Metrics\n")
            f.write(f"- Mean Squared Error (MSE): {base_metrics['mse']:.4f}\n")
            f.write(f"- Root Mean Squared Error (RMSE): {base_metrics['rmse']:.4f}\n")
            f.write(f"- Mean Absolute Error (MAE): {base_metrics['mae']:.4f}\n")
            f.write(f"- R² Score: {base_metrics['r2']:.4f}\n")
            
            if 'base_pct_error' in df.columns:
                f.write(f"- Mean Percentage Error: {df['base_pct_error'].mean():.2f}%\n")
                f.write(f"- Median Percentage Error: {df['base_pct_error'].median():.2f}%\n")
                f.write(f"- 90th Percentile Error: {df['base_pct_error'].quantile(0.9):.2f}%\n")
            
            f.write("\n")
        
        if 'xgboost_model' in metrics:
            xgb_metrics = metrics['xgboost_model']
            f.write("### XGBoost Model Metrics\n")
            f.write(f"- Mean Squared Error (MSE): {xgb_metrics['mse']:.4f}\n")
            f.write(f"- Root Mean Squared Error (RMSE): {xgb_metrics['rmse']:.4f}\n")
            f.write(f"- Mean Absolute Error (MAE): {xgb_metrics['mae']:.4f}\n")
            f.write(f"- R² Score: {xgb_metrics['r2']:.4f}\n")
            
            if 'xgb_pct_error' in df.columns:
                f.write(f"- Mean Percentage Error: {df['xgb_pct_error'].mean():.2f}%\n")
                f.write(f"- Median Percentage Error: {df['xgb_pct_error'].median():.2f}%\n")
                f.write(f"- 90th Percentile Error: {df['xgb_pct_error'].quantile(0.9):.2f}%\n")
            
            f.write("\n")
        
        # Improvement summary
        if 'base_model' in metrics and 'xgboost_model' in metrics:
            f.write("## Model Comparison\n")
            
            base_rmse = metrics['base_model']['rmse']
            xgb_rmse = metrics['xgboost_model']['rmse']
            rmse_improvement = (base_rmse - xgb_rmse) / base_rmse * 100
            
            base_mae = metrics['base_model']['mae']
            xgb_mae = metrics['xgboost_model']['mae']
            mae_improvement = (base_mae - xgb_mae) / base_mae * 100
            
            f.write(f"- RMSE Improvement: {rmse_improvement:.2f}%\n")
            f.write(f"- MAE Improvement: {mae_improvement:.2f}%\n")
            
            if 'base_pct_error' in df.columns and 'xgb_pct_error' in df.columns:
                base_pct = df['base_pct_error'].mean()
                xgb_pct = df['xgb_pct_error'].mean()
                pct_improvement = (base_pct - xgb_pct) / base_pct * 100
                
                f.write(f"- Percentage Error Improvement: {pct_improvement:.2f}%\n")
            
            f.write("\n")
        
        # Key findings section
        f.write("## Key Findings\n")
        
        # Find cases where error is high
        if 'base_pct_error' in df.columns:
            high_error_threshold = df['base_pct_error'].quantile(0.9)
            high_error_samples = df[df['base_pct_error'] > high_error_threshold]
            
            if len(high_error_samples) > 0:
                f.write(f"### High Error Cases (Base Model)\n")
                f.write(f"- {len(high_error_samples)} samples have errors > {high_error_threshold:.2f}%\n")
                
                # Check if certain models or hardware are overrepresented
                if 'hardware' in high_error_samples.columns:
                    hw_counts = high_error_samples['hardware'].value_counts(normalize=True) * 100
                    for hw, pct in hw_counts.items():
                        overall_pct = len(df[df['hardware'] == hw]) / len(df) * 100
                        if pct > overall_pct * 1.5:  # 50% more than expected
                            f.write(f"  - {hw} is overrepresented in high error cases ({pct:.1f}% vs {overall_pct:.1f}% overall)\n")
                
                if 'model_name' in high_error_samples.columns:
                    model_counts = high_error_samples['model_name'].value_counts(normalize=True) * 100
                    for model, pct in model_counts.items():
                        overall_pct = len(df[df['model_name'] == model]) / len(df) * 100
                        if pct > overall_pct * 1.5:  # 50% more than expected
                            f.write(f"  - {model} is overrepresented in high error cases ({pct:.1f}% vs {overall_pct:.1f}% overall)\n")
                
                f.write("\n")
        
        if 'hardware' in df.columns and 'base_pct_error' in df.columns:
            # Find hardware with highest and lowest errors
            hw_errors = df.groupby('hardware')['base_pct_error'].mean().sort_values()
            
            f.write("### Hardware-specific Findings\n")
            f.write(f"- Best predicted hardware: {hw_errors.index[0]} (mean error: {hw_errors.iloc[0]:.2f}%)\n")
            f.write(f"- Worst predicted hardware: {hw_errors.index[-1]} (mean error: {hw_errors.iloc[-1]:.2f}%)\n")
            f.write("\n")
        
        if 'model_name' in df.columns and 'base_pct_error' in df.columns:
            # Find models with highest and lowest errors
            model_errors = df.groupby('model_name')['base_pct_error'].mean().sort_values()
            
            f.write("### Model-specific Findings\n")
            f.write(f"- Best predicted model: {model_errors.index[0]} (mean error: {model_errors.iloc[0]:.2f}%)\n")
            f.write(f"- Worst predicted model: {model_errors.index[-1]} (mean error: {model_errors.iloc[-1]:.2f}%)\n")
            f.write("\n")
        
        # General conclusions
        f.write("## Conclusions\n")
        
        if 'base_model' in metrics and 'xgboost_model' in metrics:
            if metrics['xgboost_model']['r2'] > metrics['base_model']['r2']:
                f.write("- XGBoost model significantly improves prediction accuracy over the base model\n")
            else:
                f.write("- XGBoost model does not provide significant improvement over the base model\n")
        
        if 'num_devices' in df.columns and 'base_pct_error' in df.columns:
            # Check correlation between num_devices and error
            corr = df[['num_devices', 'base_pct_error']].corr().iloc[0, 1]
            if abs(corr) > 0.3:
                f.write(f"- Number of devices has a {abs(corr):.2f} correlation with prediction error\n")
        
        if 'seq_len' in df.columns and 'base_pct_error' in df.columns:
            # Check correlation between seq_len and error
            corr = df[['seq_len', 'base_pct_error']].corr().iloc[0, 1]
            if abs(corr) > 0.3:
                f.write(f"- Sequence length has a {abs(corr):.2f} correlation with prediction error\n")
    
    logger.info(f"Saved analysis report to {report_path}")
    return report_path

def main():
    args = parse_arguments()
    
    try:
        # Load data
        df = load_data(args.input, args.xgboost)
        
        # Calculate metrics
        metrics, df = calculate_metrics(df)
        
        # Print metrics summary
        logger.info("Prediction Performance Metrics:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in model_metrics.items():
                logger.info(f"- {metric_name}: {value:.4f}")
        
        # Generate plots
        plot_prediction_comparison(df, args.output_dir, args.dpi, not args.no_show)
        
        # Generate analysis report
        report_path = generate_analysis_report(df, metrics, args.output_dir)
        
        logger.info(f"Analysis completed. Results saved to {args.output_dir}")
        logger.info(f"Check {report_path} for detailed analysis")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 