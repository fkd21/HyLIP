import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径配置
input_file = 'data/All_results_with_predictions.csv'
output_dir = 'trained_models'
plots_dir = 'plot'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

print(f"Loading data from {input_file}")

# 加载数据
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from {input_file}")

# 标准化列名
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

# 重命名列
for old_col, new_col in col_mapping.items():
    if old_col in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)

# 计算校正比率 (measured_latency / predicted_latency)
df['correction_ratio'] = df['measured_latency'] / df['predicted_latency']
print(f"Correction ratio range: {df['correction_ratio'].min():.2f} to {df['correction_ratio'].max():.2f}")

# 处理类别特征
categorical_cols = ['hardware', 'framework', 'model_name']
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

# 准备特征和目标变量
feature_cols = [col for col in df.columns if col.endswith('_encoded')] + ['num_devices', 'seq_len', 'batch_size', 'predicted_latency']

# 目标变量为校正比率
target_col = 'correction_ratio'

# 数据划分
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

print("Training XGBoost model...")
model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=50,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=10
)

# 保存模型
model_path = os.path.join(output_dir, 'xgboost_model.json')
model.save_model(model_path)
print(f"Saved model to {model_path}")

# 评估模型
y_pred_train = model.predict(dtrain)
y_pred_test = model.predict(dtest)

# 计算指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

# 将预测结果添加到原始数据框
df['xgboost_ratio'] = float('nan')
df.loc[X_train.index, 'xgboost_ratio'] = y_pred_train
df.loc[X_test.index, 'xgboost_ratio'] = y_pred_test

# 计算XGBoost预测的延迟时间
df['xgboost_latency'] = df['xgboost_ratio'] * df['predicted_latency']

# 保存结果
df.to_csv(os.path.join(output_dir, 'all_results_with_xgboost.csv'), index=False)
print(f"Saved results to {os.path.join(output_dir, 'all_results_with_xgboost.csv')}")

# 数据分析和可视化
# 1. 实际值与预测值的散点图
plt.figure(figsize=(10, 8))
plt.scatter(df['measured_latency'], df['predicted_latency'], alpha=0.7, label='Base Model')
plt.scatter(df['measured_latency'], df['xgboost_latency'], alpha=0.7, label='XGBoost Model')
plt.plot([df['measured_latency'].min(), df['measured_latency'].max()], 
         [df['measured_latency'].min(), df['measured_latency'].max()], 'r--', label='Perfect Prediction')
plt.xlabel('Measured Latency (ms)')
plt.ylabel('Predicted Latency (ms)')
plt.title('Actual vs Predicted Latency')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), dpi=300)

# 2. 误差分布直方图
plt.figure(figsize=(10, 8))
df['base_error_pct'] = 100 * abs(df['measured_latency'] - df['predicted_latency']) / df['measured_latency']
df['xgboost_error_pct'] = 100 * abs(df['measured_latency'] - df['xgboost_latency']) / df['measured_latency']

plt.hist(df['base_error_pct'], bins=20, alpha=0.5, label='Base Model')
plt.hist(df['xgboost_error_pct'], bins=20, alpha=0.5, label='XGBoost Model')
plt.xlabel('Percentage Error (%)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'error_distribution.png'), dpi=300)

# 3. 特征重要性
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300)

# 4. 相关性热图
plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Variables')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300)

print("XGBoost training and data analysis completed successfully!") 