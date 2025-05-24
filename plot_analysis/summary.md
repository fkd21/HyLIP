# Latency Prediction Model Analysis Summary

## Performance Metrics Comparison

| Metric | Base Model | XGBoost Model | Improvement |
| ------ | ---------- | ------------- | ----------- |
| RMSE | 172.6784 | 14.5287 | 91.6% |
| MAE | 46.6241 | 3.4046 | 92.7% |
| MAPE | 123.33% | 28.34% | 77.0% |
| Median APE | 74.32% | 6.01% | 91.9% |
| Correlation | 0.3509 | 0.9964 | 184.0% |

## Key Findings

1. **XGBoost Significantly Improves Prediction Accuracy**:
   - Root Mean Square Error (RMSE) reduced by over 91%
   - Mean Absolute Error (MAE) reduced by nearly 93%
   - Median percentage error decreased from 74.32% to only 6.01%
   - Correlation coefficient increased from 0.35 to 0.996

2. **Error Distribution**:
   - XGBoost model shows a much tighter error distribution
   - Removes most of the systematic bias present in the base model
   - Higher concentration of errors near zero

3. **Middle 50% Data Analysis**:
   - When focusing on the middle 50% of latency values (removing outliers),
     the XGBoost model still maintains its superiority
   - This indicates robustness across different scenarios

4. **Cumulative Error Analysis**:
   - A significantly higher percentage of XGBoost predictions fall within the 10% error margin
   - The error distribution curve for XGBoost rises much more steeply

## Visualization Key Points

1. **Latency by Batch Size and Sequence Length**:
   - Base model shows systematic patterns of error at different batch sizes and sequence lengths
   - Particularly noticeable gap between predicted and measured values

2. **Error by Hardware Platform**:
   - Different hardware platforms show varying levels of prediction accuracy
   - Base model struggles with newer hardware architectures

3. **Error by Number of Devices**:
   - Base model prediction error varies with the number of devices
   - Multi-device predictions show higher error rates in the base model

4. **Error by Framework**:
   - Different frameworks show varying levels of prediction accuracy
   - Some frameworks are better modeled than others in the base approach

## Conclusions

1. The XGBoost model provides drastically improved prediction accuracy compared to the base model.

2. With a median error of only 6.01%, the XGBoost model is suitable for practical deployment scenarios where accurate latency predictions are required.

3. The high correlation coefficient (0.996) demonstrates that the XGBoost model captures the relationship between input features and latency extremely well.

4. The improvement across all metrics suggests that the XGBoost approach effectively addresses the limitations of the base model by learning complex patterns in the data.

5. Middle 50% analysis shows the model performs well even when excluding outliers that might artificially improve metrics. 