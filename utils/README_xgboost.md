# XGBoost 模型训练与预测

本目录包含使用XGBoost模型训练和预测LLM延迟时间的工具。

## 功能概述

XGBoost模型用于优化延迟预测的准确性，通过以下两种方式之一工作：

1. **直接预测模式**: 直接预测实际观测到的延迟时间
2. **比率预测模式**: 预测实际延迟与理论预测延迟的比率（校正因子）

## 文件说明

- `train_xgboost_model.py`: XGBoost模型训练脚本
- `predict_with_xgboost.py`: 使用训练好的XGBoost模型进行预测
- `run_xgboost_pipeline.sh`: 一键执行训练和预测的流程脚本

## 使用说明

### 环境依赖

确保安装了以下Python包:
```
pandas
numpy
xgboost
scikit-learn
joblib
matplotlib
```

可以通过pip安装：
```bash
pip install pandas numpy xgboost scikit-learn joblib matplotlib
```

### 训练模型

```bash
python utils/train_xgboost_model.py \
  --input data/all_results_with_predictions.csv \
  --output_dir trained_models \
  --model_name latency_xgboost_model.json \
  --target measured_latency \
  --ratio correction_ratio \
  --verbose
```

参数说明:
- `--input`: 输入CSV文件路径
- `--output_dir`: 输出目录，存放训练好的模型和数据集
- `--model_name`: 模型文件名
- `--target`: 目标列（可选："measured_latency"或"Predicted_Latency"）
- `--ratio`: 比率列名，用于存储实际值与预测值的比率
- `--verbose`: 是否显示详细输出
- `--seed`: 随机种子，用于数据划分（默认：42）

### 使用模型进行预测

```bash
python utils/predict_with_xgboost.py \
  --input data/new_results.csv \
  --output data/new_results_with_xgboost.csv \
  --model trained_models/latency_xgboost_model.json \
  --preprocessing trained_models/latency_xgboost_model_preprocessing.joblib \
  --output_col XGBoost_Predicted_Latency
```

参数说明:
- `--input`: 输入CSV文件路径
- `--output`: 输出CSV文件路径
- `--model`: XGBoost模型文件路径
- `--preprocessing`: 预处理数据文件路径（在训练时自动生成）
- `--output_col`: 预测结果列名

### 一键执行训练和预测流程

```bash
bash utils/run_xgboost_pipeline.sh \
  --input data/all_results_with_predictions.csv \
  --output-dir trained_models \
  --target measured_latency \
  --verbose
```

参数说明:
- `--input`: 输入CSV文件路径
- `--output-dir`: 输出目录
- `--model-name`: 模型文件名（默认：latency_xgboost_model.json）
- `--target`: 目标列（默认：measured_latency）
- `--ratio`: 比率列名（默认：correction_ratio）
- `--prediction-col`: 预测结果列名（默认：XGBoost_Predicted_Latency）
- `--verbose`: 是否显示详细输出

## 输出文件

训练过程会生成以下文件:

1. 训练、验证和测试数据集:
   - `trained_models/train_dataset.csv`
   - `trained_models/validation_dataset.csv`
   - `trained_models/test_dataset.csv`

2. 带预测结果的数据集:
   - `trained_models/train_with_predictions.csv`
   - `trained_models/validation_with_predictions.csv`
   - `trained_models/test_with_predictions.csv`

3. 模型和元数据:
   - `trained_models/latency_xgboost_model.json`: XGBoost模型
   - `trained_models/latency_xgboost_model_preprocessing.joblib`: 预处理对象
   - `trained_models/latency_xgboost_model_metadata.json`: 评估指标和特征重要性
   - `trained_models/latency_xgboost_model_importance.png`: 特征重要性图表

## 比率预测模式的工作原理

在比率预测模式下（默认模式），模型学习预测一个校正因子，该因子定义为:

```
correction_ratio = measured_latency / Predicted_Latency
```

预测时，模型输出的校正因子乘以理论预测延迟，得到最终的预测值:

```
final_prediction = correction_ratio * Predicted_Latency
```

这种方法可以利用理论模型的基本预测，同时根据硬件、框架等上下文因素进行调整，提高预测准确性。 