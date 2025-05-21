import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder # StandardScaler (可选)
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- 1. 配置与初始化 ---
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("脚本开始执行: 配置和初始化...")
# 文件路径配置
input_file = 'data/All_results_with_predictions.csv' # 请确保这个文件存在
output_dir = 'trained_models_auto_tuned'
plots_dir = 'plots_auto_tuned'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs('data', exist_ok=True) # 确保 data 文件夹存在

# 为演示目的，如果输入文件不存在，创建一个虚拟的CSV文件
if not os.path.exists(input_file):
    logging.warning(f"{input_file} not found. Creating a dummy CSV for demonstration.")
    dummy_data = {
        'Hardware': np.random.choice(['GPU_A', 'GPU_B', 'CPU_X'], 5000),
        'Num of Hardware': np.random.randint(1, 5, 5000),
        'Framework': np.random.choice(['Framework1', 'Framework2'], 5000),
        'Model': np.random.choice(['Model_X', 'Model_Y', 'Model_Z'], 5000),
        'Input Output Length': np.random.choice([128, 256, 512], 5000),
        'Batch Size': np.random.choice([16, 32, 64], 5000),
        'Predicted_Latency': np.random.uniform(10, 200, 5000),
        'Latency': np.random.uniform(5, 250, 5000) # Measured Latency
    }
    # 确保 Latency 和 Predicted_Latency 尽可能为正，以避免 correction_ratio 问题
    dummy_data['Predicted_Latency'] = np.abs(dummy_data['Predicted_Latency']) + 1e-6
    dummy_data['Latency'] = np.abs(dummy_data['Latency']) + 1e-6
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(input_file, index=False)
    logging.info(f"Dummy CSV created at {input_file}")

logging.info("--- 数据加载阶段 ---")
logging.info(f"从 {input_file} 加载数据...")
df = pd.read_csv(input_file)
logging.info(f"从 {input_file} 加载了 {len(df)} 行数据")

# 标准化列名
logging.info("标准化列名...")
col_mapping = {
    'Hardware': 'hardware',
    'Num of Hardware': 'num_devices',
    'Framework': 'framework',
    'Model': 'model_name',
    'Input Output Length': 'seq_len',
    'Batch Size': 'batch_size',
    'Latency': 'measured_latency', # 实际测量的延迟
    'Throughput': 'throughput',
    'Predicted_Latency': 'predicted_latency' # 基础模型预测的延迟
}
for old_col, new_col in col_mapping.items():
    if old_col in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)
logging.info("列名标准化完成。")

# --- 2. 特征工程与预处理 ---
logging.info("--- 特征工程与预处理阶段 ---")
# 计算校正比率 (目标变量)
epsilon = 1e-9 # 防止除以零
logging.info("计算校正比率 (correction_ratio)...")
df['predicted_latency'] = df['predicted_latency'].fillna(epsilon).replace(0, epsilon) # 处理 predicted_latency 中的 NaN 和 0
df['correction_ratio'] = df['measured_latency'] / df['predicted_latency']
logging.info(f"校正比率范围: {df['correction_ratio'].min():.4f} 到 {df['correction_ratio'].max():.4f}")
df = df[~df['correction_ratio'].isin([np.inf, -np.inf])] # 移除无限值
df.dropna(subset=['correction_ratio'], inplace=True) # 移除NaN的比率
logging.info("校正比率计算和清理完成。")

# 处理类别特征
logging.info("处理类别特征 (Label Encoding)...")
categorical_cols = ['hardware', 'framework', 'model_name']
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str)) # 确保是字符串类型
        encoders[col] = le
        encoder_path = os.path.join(output_dir, f'{col}_label_encoder.joblib')
        joblib.dump(le, encoder_path)
        logging.info(f"已保存 {col} 编码器到 {encoder_path}")
    else:
        logging.warning(f"类别特征列 {col} 在 DataFrame 中未找到。")
logging.info("类别特征处理完成。")


# 准备特征
logging.info("准备特征集 (X) 和目标变量 (y)...")
feature_cols = [col for col in df.columns if col.endswith('_encoded')] + \
               ['num_devices', 'seq_len', 'batch_size', 'predicted_latency']

# 确保所有特征列都存在
feature_cols = [col for col in feature_cols if col in df.columns]
if not feature_cols:
    raise ValueError("未找到特征列。请检查列名和编码步骤。")

X = df[feature_cols]
y = df['correction_ratio']
logging.info(f"特征集 X 的形状: {X.shape}, 目标变量 y 的形状: {y.shape}")

# --- 3. 数据集划分 (80:10:10) ---
logging.info("--- 数据集划分阶段 (80:10:10) ---")
# 首先，划分出测试集 (10%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# 然后，从剩余数据 (90%) 中划分出训练集 (约80%的原始数据) 和验证集 (约10%的原始数据)
# 验证集占剩余数据的 1/9 (10% / 90%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(1/9.0), random_state=42) # Ensure float division

logging.info(f"训练集大小: {X_train.shape[0]}")
logging.info(f"验证集大小: {X_val.shape[0]}")
logging.info(f"测试集大小: {X_test.shape[0]}")
logging.info("数据集划分完成。")

# --- 4. XGBoost 模型与超参数调优 (RandomizedSearchCV) ---
logging.info("--- XGBoost 模型与超参数调优阶段 ---")
logging.info("使用 RandomizedSearchCV 开始 XGBoost 超参数调优...")
logging.info("这可能需要一些时间，具体取决于 n_iter 和 cv 的设置以及数据量。")

# 定义XGBoost模型
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist',
    device='cpu',
    early_stopping_rounds=50,
    eval_metric='rmse'
)
logging.info(f"XGBoost Regressor initialized with device='cpu'.")

# 定义超参数搜索空间
param_dist = {
    'n_estimators': [100, 200, 300, 500, 700, 1000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1]
}

# 配置 RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

fit_params_for_tuning = {
    'eval_set': [(X_val, y_val)],
    'verbose': False
}

logging.info(f"开始拟合 RandomizedSearchCV (n_iter={random_search.n_iter}, cv={random_search.cv})...")
random_search.fit(X_train, y_train, **fit_params_for_tuning)
logging.info("RandomizedSearchCV 拟合完成。")

logging.info(f"找到的最佳超参数: {random_search.best_params_}")
logging.info(f"最佳交叉验证分数 (负 RMSE): {random_search.best_score_:.4f}")

best_xgb_model = random_search.best_estimator_
logging.info("已获取最佳 XGBoost 模型。")

# --- 5. 模型保存 ---
logging.info("--- 模型保存阶段 ---")
model_path = os.path.join(output_dir, 'best_xgboost_model.json')
best_xgb_model.save_model(model_path)
logging.info(f"已将最佳自动调优模型保存到 {model_path}")

# --- 6. 模型评估 ---
logging.info("--- 模型评估阶段 ---")
logging.info("在训练集上进行预测...")
y_pred_train = best_xgb_model.predict(X_train)
logging.info("在验证集上进行预测...")
y_pred_val = best_xgb_model.predict(X_val)
logging.info("在测试集上进行预测...")
y_pred_test = best_xgb_model.predict(X_test)
logging.info("预测完成。")

logging.info("对预测的 ratio 进行后处理 (确保非负)...")
y_pred_train = np.maximum(y_pred_train, epsilon)
y_pred_val = np.maximum(y_pred_val, epsilon)
y_pred_test = np.maximum(y_pred_test, epsilon)

logging.info("计算评估指标 (RMSE, R²)...")
metrics = {}
for split_name, y_true, y_pred in [('Train', y_train, y_pred_train),
                                   ('Validation', y_val, y_pred_val),
                                   ('Test', y_test, y_pred_test)]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    metrics[split_name] = {'RMSE': rmse, 'R2': r2}
    logging.info(f"{split_name} RMSE: {rmse:.4f}, R²: {r2:.4f}")
logging.info("模型评估完成。")

# --- 7. 结果整合与保存 ---
logging.info("--- 结果整合与保存阶段 ---")

# 为测试集准备和保存结果
df_test_indices = X_test.index
df_test_predictions = pd.DataFrame({
    'xgboost_ratio_tuned': y_pred_test,
    'data_split': 'test' # 添加数据分割标识
}, index=df_test_indices)
df_test_with_preds = df.loc[df_test_indices].join(df_test_predictions)
if 'predicted_latency' in df_test_with_preds.columns:
    df_test_with_preds['xgboost_latency_tuned'] = df_test_with_preds['xgboost_ratio_tuned'] * df_test_with_preds['predicted_latency']
else:
    logging.warning("'predicted_latency' not found in df_test_with_preds for test set, cannot calculate 'xgboost_latency_tuned'.")

output_csv_path_test = os.path.join(output_dir, 'test_results_with_tuned_xgboost.csv')
cols_to_save_test = list(X_test.columns) + ['measured_latency', 'predicted_latency', 'xgboost_ratio_tuned', 'xgboost_latency_tuned', 'data_split']
cols_to_save_existing_test = [col for col in cols_to_save_test if col in df_test_with_preds.columns]
if cols_to_save_existing_test:
    df_test_with_preds[cols_to_save_existing_test].to_csv(output_csv_path_test, index=True)
    logging.info(f"已将带有调优后 XGBoost 预测的测试集结果保存到 {output_csv_path_test}")
else:
    logging.warning("没有有效的列可用于保存测试结果。")


# 为训练集和验证集准备和保存结果
logging.info("为训练集和验证集准备结果...")
# 训练集
df_train_indices = X_train.index
df_train_predictions = pd.DataFrame({
    'xgboost_ratio_tuned': y_pred_train,
    'data_split': 'train' # 添加数据分割标识
}, index=df_train_indices)
df_train_with_preds = df.loc[df_train_indices].join(df_train_predictions)
if 'predicted_latency' in df_train_with_preds.columns:
    df_train_with_preds['xgboost_latency_tuned'] = df_train_with_preds['xgboost_ratio_tuned'] * df_train_with_preds['predicted_latency']
else:
    logging.warning("'predicted_latency' not found in df_train_with_preds for train set, cannot calculate 'xgboost_latency_tuned'.")

# 验证集
df_val_indices = X_val.index
df_val_predictions = pd.DataFrame({
    'xgboost_ratio_tuned': y_pred_val,
    'data_split': 'validation' # 添加数据分割标识
}, index=df_val_indices)
df_val_with_preds = df.loc[df_val_indices].join(df_val_predictions)
if 'predicted_latency' in df_val_with_preds.columns:
    df_val_with_preds['xgboost_latency_tuned'] = df_val_with_preds['xgboost_ratio_tuned'] * df_val_with_preds['predicted_latency']
else:
    logging.warning("'predicted_latency' not found in df_val_with_preds for validation set, cannot calculate 'xgboost_latency_tuned'.")

# 合并训练集和验证集的结果
df_train_val_combined_results = pd.concat([df_train_with_preds, df_val_with_preds])
logging.info(f"合并后的训练集和验证集结果行数: {len(df_train_val_combined_results)}")

output_csv_path_train_val = os.path.join(output_dir, 'train_val_results_with_tuned_xgboost.csv')
# 定义要为训练集和验证集保存的列 (与测试集保持一致)
cols_to_save_train_val = list(X_train.columns) + ['measured_latency', 'predicted_latency', 'xgboost_ratio_tuned', 'xgboost_latency_tuned', 'data_split']
cols_to_save_existing_train_val = [col for col in cols_to_save_train_val if col in df_train_val_combined_results.columns]

if cols_to_save_existing_train_val:
    df_train_val_combined_results[cols_to_save_existing_train_val].to_csv(output_csv_path_train_val, index=True)
    logging.info(f"已将带有调优后 XGBoost 预测的训练集和验证集结果保存到 {output_csv_path_train_val}")
else:
    logging.warning("没有有效的列可用于保存训练集和验证集结果。")


# --- 8. 数据分析和可视化 ---
logging.info("--- 数据分析和可视化阶段 ---")
# 使用 df_test_with_preds 进行测试集绘图，因为它包含了测试集的所有必要列
if 'measured_latency' in df_test_with_preds.columns and \
   'predicted_latency' in df_test_with_preds.columns and \
   'xgboost_latency_tuned' in df_test_with_preds.columns:
    logging.info("生成实际值与预测值散点图 (测试集)...")
    plt.figure(figsize=(10, 8))
    plt.scatter(df_test_with_preds['measured_latency'], df_test_with_preds['predicted_latency'], 
                alpha=0.5, label='Base Model (Initial Predicted Latency)', marker='x', s=50)
    plt.scatter(df_test_with_preds['measured_latency'], df_test_with_preds['xgboost_latency_tuned'], 
                alpha=0.7, label='Tuned XGBoost Model', marker='o', s=50, color='green')

    min_val_plot = min(df_test_with_preds['measured_latency'].min(), 
                       df_test_with_preds['xgboost_latency_tuned'].min(skipna=True), 
                       df_test_with_preds['predicted_latency'].min())
    max_val_plot = max(df_test_with_preds['measured_latency'].max(), 
                       df_test_with_preds['xgboost_latency_tuned'].max(skipna=True), 
                       df_test_with_preds['predicted_latency'].max())

    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', label='Perfect Prediction Line') # MODIFIED
    plt.xlabel('Measured Latency (ms)') # MODIFIED
    plt.ylabel('Predicted Latency (ms)') # MODIFIED
    plt.title('Actual vs. Predicted Latency (Test Set)') # MODIFIED
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted_tuned.png'), dpi=300)
    plt.close()
    logging.info("实际值与预测值散点图 (测试集) 已保存。")
else:
    logging.warning("缺少用于绘制实际值与预测值散点图 (测试集) 的列。")

if 'measured_latency' in df_test_with_preds.columns and \
   'predicted_latency' in df_test_with_preds.columns and \
   'xgboost_latency_tuned' in df_test_with_preds.columns:
    logging.info("生成误差分布直方图 (测试集)...")
    plt.figure(figsize=(10, 8))
    base_error_pct = 100 * abs(df_test_with_preds['measured_latency'] - df_test_with_preds['predicted_latency']) / (df_test_with_preds['measured_latency'] + epsilon)
    xgboost_error_pct = 100 * abs(df_test_with_preds['measured_latency'] - df_test_with_preds['xgboost_latency_tuned']) / (df_test_with_preds['measured_latency'] + epsilon)

    plt.hist(base_error_pct.clip(0, 200), bins=50, alpha=0.6, label='Base Model Error %') # MODIFIED
    plt.hist(xgboost_error_pct.clip(0, 200), bins=50, alpha=0.6, label='Tuned XGBoost Error %') # MODIFIED
    plt.xlabel('Percentage Error (%)') # MODIFIED
    plt.ylabel('Frequency') # MODIFIED
    plt.title('Error Distribution (Test Set)') # MODIFIED
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'error_distribution_tuned.png'), dpi=300)
    plt.close()
    logging.info("误差分布直方图 (测试集) 已保存。")
else:
    logging.warning("缺少用于绘制误差分布直方图 (测试集) 的列。")

logging.info("生成特征重要性图...")
if hasattr(best_xgb_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(12, max(8, len(X_train.columns) * 0.5))) 
    xgb.plot_importance(best_xgb_model, ax=ax, max_num_features=len(X_train.columns))
    plt.title('Feature Importance (Tuned XGBoost Model)') # MODIFIED
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance_tuned.png'), dpi=300)
    plt.close()
    logging.info("特征重要性图已保存。")
else:
    logging.warning("无法绘制特征重要性图。模型可能没有 feature_importances_ 属性。")

logging.info("生成相关性热图...")
plt.figure(figsize=(14, 12))
corr_cols_df = df[X.columns.tolist() + [y.name]].copy()
corr_matrix = corr_cols_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 8})
plt.title('Correlation Heatmap (Features and Target)') # MODIFIED
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap_tuned.png'), dpi=300)
plt.close()
logging.info("相关性热图已保存。")

logging.info("XGBoost 自动调优、训练和分析已成功完成！")
