#!/usr/bin/env python3
"""
简洁的相关性分析总结
Concise correlation analysis summary for DeepSpeed and llama.cpp frameworks
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def main():
    # 加载数据
    df = pd.read_csv('data/All_results_with_predictions.csv')
    
    print("="*60)
    print("DeepSpeed与llama.cpp框架预测延迟线性相关度分析")
    print("="*60)
    
    # DeepSpeed (合并 Deepspeed + Deepspeed-MII)
    deepspeed_data = df[df['Framework'].isin(['Deepspeed', 'Deepspeed-MII'])]
    ds_actual = deepspeed_data['Latency'].values
    ds_predicted = deepspeed_data['Predicted_Latency'].values
    ds_corr, ds_p = pearsonr(ds_actual, ds_predicted)
    
    # llama.cpp
    llama_data = df[df['Framework'] == 'llama.cpp']
    llama_actual = llama_data['Latency'].values
    llama_predicted = llama_data['Predicted_Latency'].values
    llama_corr, llama_p = pearsonr(llama_actual, llama_predicted)
    
    print(f"\n1. DeepSpeed框架 (合并Deepspeed + Deepspeed-MII):")
    print(f"   数据点数量: {len(deepspeed_data)}")
    print(f"   皮尔逊线性相关系数: {ds_corr:.4f}")
    print(f"   p-value: {ds_p:.2e}")
    print(f"   相关性强度: {'强' if abs(ds_corr) >= 0.7 else '中等' if abs(ds_corr) >= 0.5 else '弱'}")
    
    print(f"\n2. llama.cpp框架:")
    print(f"   数据点数量: {len(llama_data)}")
    print(f"   皮尔逊线性相关系数: {llama_corr:.4f}")
    print(f"   p-value: {llama_p:.2e}")
    print(f"   相关性强度: {'强' if abs(llama_corr) >= 0.7 else '中等' if abs(llama_corr) >= 0.5 else '弱'}")
    
    print(f"\n3. 对比结果:")
    print(f"   DeepSpeed vs llama.cpp 相关系数: {ds_corr:.4f} vs {llama_corr:.4f}")
    print(f"   DeepSpeed相关性比llama.cpp强 {ds_corr/llama_corr:.1f} 倍")
    
    print(f"\n4. 结论:")
    if ds_corr > 0.7:
        print(f"   ✓ DeepSpeed框架具有强线性相关性，预测效果良好")
    else:
        print(f"   ⚠ DeepSpeed框架线性相关性一般")
        
    if llama_corr > 0.5:
        print(f"   ✓ llama.cpp框架具有中等线性相关性")
    else:
        print(f"   ✗ llama.cpp框架线性相关性较弱，预测效果不佳")
    
    print("="*60)

if __name__ == "__main__":
    main() 