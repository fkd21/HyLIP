import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create directories for saving plots
os.makedirs('framework_hardware_plots', exist_ok=True)
os.makedirs('prediction_analysis_plots', exist_ok=True)

# Read the data
df = pd.read_csv('data/All_results_with_predictions.csv')

# Clean column names
df.columns = ['Hardware', 'Num_Hardware', 'Framework', 'Model', 'Seq_Length', 
              'Batch_Size', 'Latency', 'Throughput', 'Predicted_Latency']

# Merge Deepspeed and Deepspeed-MII
df['Framework'] = df['Framework'].replace('Deepspeed-MII', 'Deepspeed')

# Convert latency from milliseconds to seconds
df['Latency'] = df['Latency'] / 1000
df['Predicted_Latency'] = df['Predicted_Latency'] / 1000

print("Data shape:", df.shape)
print("\nUnique frameworks:", df['Framework'].unique())
print("\nUnique hardware:", df['Hardware'].unique())
print("\nUnique models:", df['Model'].unique())

# Calculate prediction errors
df['Prediction_Error'] = abs(df['Latency'] - df['Predicted_Latency'])
df['Relative_Error'] = df['Prediction_Error'] / df['Latency'] * 100

# 1. Framework and Hardware Scatter Plots (Actual vs Predicted)
print("\n=== Creating Framework and Hardware Scatter Plots ===")

# Get unique combinations
frameworks = df['Framework'].unique()
hardware_types = df['Hardware'].unique()

# Create scatter plots for each framework (Actual vs Predicted Latency)
for framework in frameworks:
    framework_data = df[df['Framework'] == framework]
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different colors for different hardware
    for i, hardware in enumerate(hardware_types):
        hw_data = framework_data[framework_data['Hardware'] == hardware]
        if not hw_data.empty:
            plt.scatter(hw_data['Latency'], hw_data['Predicted_Latency'], 
                       label=f'{hardware}', alpha=0.7, s=50)
    
    # Add perfect prediction line (45-degree line)
    min_val = min(framework_data['Latency'].min(), framework_data['Predicted_Latency'].min())
    max_val = max(framework_data['Latency'].max(), framework_data['Predicted_Latency'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction', alpha=0.8)
    
    plt.xlabel('Actual Latency (s)', fontsize=12)
    plt.ylabel('Predicted Latency (s)', fontsize=12)
    plt.title(f'Prediction Accuracy - {framework}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f'framework_hardware_plots/{framework.replace("/", "_").replace("-", "_")}_prediction_accuracy.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Create scatter plots for each hardware type (Actual vs Predicted Latency)
for hardware in hardware_types:
    hardware_data = df[df['Hardware'] == hardware]
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different colors for different frameworks
    for i, framework in enumerate(frameworks):
        fw_data = hardware_data[hardware_data['Framework'] == framework]
        if not fw_data.empty:
            plt.scatter(fw_data['Latency'], fw_data['Predicted_Latency'], 
                       label=f'{framework}', alpha=0.7, s=50)
    
    # Add perfect prediction line (45-degree line)
    min_val = min(hardware_data['Latency'].min(), hardware_data['Predicted_Latency'].min())
    max_val = max(hardware_data['Latency'].max(), hardware_data['Predicted_Latency'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction', alpha=0.8)
    
    plt.xlabel('Actual Latency (s)', fontsize=12)
    plt.ylabel('Predicted Latency (s)', fontsize=12)
    plt.title(f'Prediction Accuracy - {hardware}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f'framework_hardware_plots/{hardware.replace(" ", "_").replace("/", "_")}_prediction_accuracy.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# 2. Analyze prediction performance by framework and model
print("\n=== Analyzing Prediction Performance ===")

# Calculate metrics for each framework-model combination
performance_metrics = []

for framework in frameworks:
    for model in df['Model'].unique():
        subset = df[(df['Framework'] == framework) & (df['Model'] == model)]
        if len(subset) > 0:
            mae = mean_absolute_error(subset['Latency'], subset['Predicted_Latency'])
            rmse = np.sqrt(mean_squared_error(subset['Latency'], subset['Predicted_Latency']))
            mean_relative_error = subset['Relative_Error'].mean()
            
            performance_metrics.append({
                'Framework': framework,
                'Model': model,
                'MAE': mae,
                'RMSE': rmse,
                'Mean_Relative_Error': mean_relative_error,
                'Data_Points': len(subset)
            })

performance_df = pd.DataFrame(performance_metrics)
performance_df = performance_df.sort_values('Mean_Relative_Error')

print("\nPrediction Performance Summary:")
print(performance_df.to_string(index=False))

# Identify best and worst performing combinations
best_combinations = performance_df.head(3)
worst_combinations = performance_df.tail(3)

print(f"\n=== Best Prediction Performance ===")
for _, row in best_combinations.iterrows():
    print(f"{row['Framework']} + {row['Model']}: {row['Mean_Relative_Error']:.2f}% error")

print(f"\n=== Worst Prediction Performance ===")
for _, row in worst_combinations.iterrows():
    print(f"{row['Framework']} + {row['Model']}: {row['Mean_Relative_Error']:.2f}% error")

# 3. Create detailed comparison plots for best and worst performers
print("\n=== Creating Prediction Performance Comparison Plots ===")

# Function to create comparison plots
def create_comparison_plots(combinations, title_prefix, filename_prefix):
    for i, (_, row) in enumerate(combinations.iterrows()):
        framework = row['Framework']
        model = row['Model']
        subset = df[(df['Framework'] == framework) & (df['Model'] == model)]
        
        if len(subset) == 0:
            continue
            
        # Plot 1: Latency vs Batch Size
        plt.figure(figsize=(12, 8))
        plt.scatter(subset['Batch_Size'], subset['Latency'], 
                   label='Actual Latency', alpha=0.7, s=60, color='blue')
        plt.scatter(subset['Batch_Size'], subset['Predicted_Latency'], 
                   label='Predicted Latency', alpha=0.7, s=60, color='red')
        
        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Latency (s)', fontsize=12)
        plt.title(f'{title_prefix}: Latency vs Batch Size\n{framework} + {model}\n'
                 f'Mean Relative Error: {row["Mean_Relative_Error"]:.2f}%', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'prediction_analysis_plots/{filename_prefix}_batch_size_{i+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        # Plot 2: Latency vs Sequence Length
        plt.figure(figsize=(12, 8))
        plt.scatter(subset['Seq_Length'], subset['Latency'], 
                   label='Actual Latency', alpha=0.7, s=60, color='blue')
        plt.scatter(subset['Seq_Length'], subset['Predicted_Latency'], 
                   label='Predicted Latency', alpha=0.7, s=60, color='red')
        
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Latency (s)', fontsize=12)
        plt.title(f'{title_prefix}: Latency vs Sequence Length\n{framework} + {model}\n'
                 f'Mean Relative Error: {row["Mean_Relative_Error"]:.2f}%', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'prediction_analysis_plots/{filename_prefix}_seq_length_{i+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

# Create plots for best performers
create_comparison_plots(best_combinations, 'Best Prediction Performance', 'best_performance')

# Create plots for worst performers  
create_comparison_plots(worst_combinations, 'Worst Prediction Performance', 'worst_performance')

# 4. Create overall prediction accuracy scatter plot
plt.figure(figsize=(12, 10))
plt.scatter(df['Latency'], df['Predicted_Latency'], alpha=0.6, s=30)

# Add perfect prediction line
min_val = min(df['Latency'].min(), df['Predicted_Latency'].min())
max_val = max(df['Latency'].max(), df['Predicted_Latency'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Latency (s)', fontsize=12)
plt.ylabel('Predicted Latency (s)', fontsize=12)
plt.title('Overall Prediction Accuracy: Predicted vs Actual Latency', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

filename = 'prediction_analysis_plots/overall_prediction_accuracy.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {filename}")

# 5. Create summary statistics plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Error distribution by framework
df.boxplot(column='Relative_Error', by='Framework', ax=axes[0,0])
axes[0,0].set_title('Prediction Error Distribution by Framework')
axes[0,0].set_xlabel('Framework')
axes[0,0].set_ylabel('Relative Error (%)')

# Plot 2: Error distribution by hardware
df.boxplot(column='Relative_Error', by='Hardware', ax=axes[0,1])
axes[0,1].set_title('Prediction Error Distribution by Hardware')
axes[0,1].set_xlabel('Hardware')
axes[0,1].set_ylabel('Relative Error (%)')

# Plot 3: Latency distribution by framework
df.boxplot(column='Latency', by='Framework', ax=axes[1,0])
axes[1,0].set_title('Latency Distribution by Framework')
axes[1,0].set_xlabel('Framework')
axes[1,0].set_ylabel('Latency (s)')

# Plot 4: Latency distribution by hardware
df.boxplot(column='Latency', by='Hardware', ax=axes[1,1])
axes[1,1].set_title('Latency Distribution by Hardware')
axes[1,1].set_xlabel('Hardware')
axes[1,1].set_ylabel('Latency (s)')

plt.tight_layout()
filename = 'prediction_analysis_plots/summary_statistics.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {filename}")

print(f"\n=== Analysis Complete ===")
print(f"Framework and hardware plots saved in: framework_hardware_plots/")
print(f"Prediction analysis plots saved in: prediction_analysis_plots/")
print(f"Total plots created: {len(os.listdir('framework_hardware_plots')) + len(os.listdir('prediction_analysis_plots'))}")