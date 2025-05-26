import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

print("Starting analysis...")

try:
    # Read the data
    df = pd.read_csv('data/All_results_with_predictions.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    df.columns = ['Hardware', 'Num_Hardware', 'Framework', 'Model', 'Seq_Length', 
                  'Batch_Size', 'Latency', 'Throughput', 'Predicted_Latency']

    # Merge Deepspeed and Deepspeed-MII
    df['Framework'] = df['Framework'].replace('Deepspeed-MII', 'Deepspeed')
    
    # Convert latency from milliseconds to seconds
    df['Latency'] = df['Latency'] / 1000
    df['Predicted_Latency'] = df['Predicted_Latency'] / 1000

    # Calculate prediction errors
    df['Prediction_Error'] = abs(df['Latency'] - df['Predicted_Latency'])
    df['Relative_Error'] = df['Prediction_Error'] / df['Latency'] * 100

    # Calculate metrics for each framework-model combination
    performance_metrics = []
    frameworks = df['Framework'].unique()

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

    print('=== PREDICTION PERFORMANCE ANALYSIS ===')
    print(f'Total data points: {len(df)}')
    print(f'Unique frameworks: {len(frameworks)}')
    print(f'Unique models: {len(df["Model"].unique())}')
    print(f'Unique hardware: {len(df["Hardware"].unique())}')

    print('\n=== TOP 5 BEST PREDICTION PERFORMANCE ===')
    best_5 = performance_df.head(5)
    for i, (_, row) in enumerate(best_5.iterrows(), 1):
        model_name = row["Model"].split("/")[-1] if "/" in row["Model"] else row["Model"]
        print(f'{i}. {row["Framework"]} + {model_name}: {row["Mean_Relative_Error"]:.2f}% error ({row["Data_Points"]} points)')

    print('\n=== TOP 5 WORST PREDICTION PERFORMANCE ===')
    worst_5 = performance_df.tail(5)
    for i, (_, row) in enumerate(worst_5.iterrows(), 1):
        model_name = row["Model"].split("/")[-1] if "/" in row["Model"] else row["Model"]
        print(f'{i}. {row["Framework"]} + {model_name}: {row["Mean_Relative_Error"]:.2f}% error ({row["Data_Points"]} points)')

    print('\n=== FRAMEWORK PERFORMANCE SUMMARY ===')
    framework_summary = df.groupby('Framework')['Relative_Error'].agg(['mean', 'std', 'count']).round(2)
    print(framework_summary)

    print('\n=== HARDWARE PERFORMANCE SUMMARY ===')
    hardware_summary = df.groupby('Hardware')['Relative_Error'].agg(['mean', 'std', 'count']).round(2)
    print(hardware_summary)

    print('\n=== FILES GENERATED ===')
    if os.path.exists('framework_hardware_plots'):
        framework_plots = len(os.listdir('framework_hardware_plots'))
    else:
        framework_plots = 0
        
    if os.path.exists('prediction_analysis_plots'):
        prediction_plots = len(os.listdir('prediction_analysis_plots'))
    else:
        prediction_plots = 0
        
    print(f'Framework/Hardware plots: {framework_plots}')
    print(f'Prediction analysis plots: {prediction_plots}')
    print(f'Total plots generated: {framework_plots + prediction_plots}')

    print('\n=== PLOT DESCRIPTIONS ===')
    print('Framework/Hardware plots:')
    print('- Scatter plots showing actual vs predicted latency for each framework')
    print('- Scatter plots showing actual vs predicted latency for each hardware type')
    print('- All plots include 45-degree perfect prediction line')
    print('\nPrediction analysis plots:')
    print('- Best performing framework+model combinations (lowest prediction error)')
    print('- Worst performing framework+model combinations (highest prediction error)')
    print('- Overall prediction accuracy scatter plot')
    print('- Summary statistics with error distributions')
    
    print('\n=== KEY FINDINGS ===')
    print(f'- Best prediction accuracy: {best_5.iloc[0]["Framework"]} + {best_5.iloc[0]["Model"].split("/")[-1]} ({best_5.iloc[0]["Mean_Relative_Error"]:.2f}% error)')
    print(f'- Worst prediction accuracy: {worst_5.iloc[-1]["Framework"]} + {worst_5.iloc[-1]["Model"].split("/")[-1]} ({worst_5.iloc[-1]["Mean_Relative_Error"]:.2f}% error)')
    print(f'- Most accurate framework overall: {framework_summary.sort_values("mean").index[0]} ({framework_summary.sort_values("mean").iloc[0]["mean"]:.2f}% avg error)')
    print(f'- Least accurate framework overall: {framework_summary.sort_values("mean").index[-1]} ({framework_summary.sort_values("mean").iloc[-1]["mean"]:.2f}% avg error)')

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc() 