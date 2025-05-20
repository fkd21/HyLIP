import os
import csv
import argparse
from typing import Dict, List, Tuple
import importlib.util
import sys
from collections import defaultdict

from base_predict import predict_latency


def map_model_name_to_config(model_name: str) -> str:
    """Map model names from the CSV to their corresponding config file paths."""
    model_mapping = {
        # Original mappings
        "meta-llama/Llama-2-70b-hf": "configs/model_configs/llama2_70b.py",
        "meta-llama/Llama-2-7b-hf": "configs/model_configs/llama2_7b.py",
        "mistralai/Mistral-7B-v0.1": "configs/model_configs/mistral_7b.py",
        "meta-llama/Meta-Llama-3-8B": "configs/model_configs/llama3_8b.py",
        "Qwen/Qwen2-7B": "configs/model_configs/qwen2_7b.py",
        "Qwen/Qwen2-72B": "configs/model_configs/qwen2_72b.py",
        
        # Additional mappings with case-insensitive variants
        "meta-llama/llama-2-70b-hf": "configs/model_configs/llama2_70b.py",
        "meta-llama/llama-2-7b-hf": "configs/model_configs/llama2_7b.py",
        
        # Additional models
        "meta-llama/Meta-Llama-3-70B": "configs/model_configs/llama3_70b.py",
        "mistralai/Mixtral-8x7B-v0.1": "configs/model_configs/mixtral_8x7b.py",
        "BAAI/Aquila-7B": "configs/model_configs/aquila_7b.py",
        "bigscience/bloom-7b1": "configs/model_configs/bloom_7b1.py",
        "EleutherAI/gpt-j-6b": "configs/model_configs/gpt_j_6b.py",
        "huggyllama/llama-7b": "configs/model_configs/llama2_7b.py",
        "facebook/opt-6.7b": "configs/model_configs/opt_6_7b.py",
        "Qwen/Qwen1.5-7B": "configs/model_configs/qwen1_5_7b.py",
        "Deci/DeciLM-7B": "configs/model_configs/decilm_7b.py",
    }
    return model_mapping.get(model_name)


def map_hardware_name_to_config(hardware_name: str) -> str:
    """Map hardware names from the CSV to their corresponding config file paths."""
    hardware_mapping = {
        "Nvidia A100 GPU": "configs/hardware_configs/A100.json",
        "Nvidia H100 GPU": "configs/hardware_configs/H100.json",
        "Nvidia GH200 GPU": "configs/hardware_configs/GH200.json",
        "AMD MI250 GPU": "configs/hardware_configs/MI250.json",
        "AMD MI300X GPU": "configs/hardware_configs/MI300X.json", 
        "Intel Max 1550": "configs/hardware_configs/max_1550.json",
        "Intel PVC GPU": "configs/hardware_configs/PVC.json",
        "Habana Gaudi2": "configs/hardware_configs/Gaudi2.json",
        "SambaNova SN40L": "configs/hardware_configs/SN40L.json",
    }
    return hardware_mapping.get(hardware_name)


def determine_parallel_mode(framework: str) -> str:
    """Determine the parallel mode based on the framework."""
    # For simplicity, we'll assume TensorRT-LLM uses tensor parallelism,
    # while Deepspeed uses pipeline parallelism when multiple devices are present
    if "TensorRT-LLM" in framework:
        return "tensor"
    elif "Deepspeed" in framework:
        return "pipeline"
    else:
        return "tensor"  # Default to tensor parallelism


def process_csv(input_csv: str, output_csv: str, use_flash_attention: bool = False) -> None:
    """
    Process the CSV file, predict latencies, and write results to a new CSV.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        use_flash_attention: Whether to use flash attention in predictions
    """
    rows = []
    header = None
    
    # Read the input CSV
    with open(input_csv, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read header
        rows = list(reader)
    
    # Add a new column header for theoretical latency
    header.append("Predicted_Latency")
    
    
    # Process each row
    for i, row in enumerate(rows):
        hardware = row[0]
        num_devices = int(row[1])
        framework = row[2]
        model_name = row[3]
        seq_len = int(row[4])
        batch_size = int(row[5])
        measured_latency = float(row[6])
        measured_throughput = float(row[7])
        
        # Get config paths
        model_config_path = map_model_name_to_config(model_name)
        hardware_config_path = map_hardware_name_to_config(hardware)
        
        # Skip if we don't have mappings for this hardware/model
        if not model_config_path or not hardware_config_path:
            print(f"Warning: Skipping row {i+1} due to missing config for {model_name} or {hardware}")
            row.append("N/A")
            continue
        
        # Determine parallel mode
        parallel_mode = determine_parallel_mode(framework)

        # try:
        results = predict_latency(
            model_config_path=model_config_path,
            hardware_config_path=hardware_config_path,
            seq_len=seq_len,
            batch_size=batch_size,
            parallel_mode=parallel_mode,
            num_devices=num_devices,
            dtype="fp16",  # Assuming all models use fp16
            use_flash_attention=use_flash_attention
        )
        
        prefill_latency = results["prefill_latency"]
        decode_latency = results["decode_latency"]
            
        # except Exception as e:
        #     print(f"Error predicting for row {i+1}: {e}")
        #     row.append("Error")
        #     continue
        
        # Calculate end-to-end latency: prefill + decode * (tokens - 1)
        # Assuming seq_len is the total number of tokens to generate
        output_token_num = seq_len
        predicted_latency = prefill_latency + (decode_latency * (output_token_num - 1))
        
        # Add prediction to row
        row.append(str(predicted_latency))
        
        # Print progress
        print(predicted_latency)
        print(f"Processed {i + 1}/{len(rows)} rows")
        if (i+1)%100 == 0:
            #save scv file
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(rows)
    
    # Write the output CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Completed processing {len(rows)} rows. Results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict latencies for models in CSV')
    parser.add_argument('--input', type=str, default='data/All_results.csv',
                      help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/All_results_with_predictions.csv',
                      help='Output CSV file path')
    parser.add_argument('--flash', action='store_true', 
                      help='Use flash attention for predictions')
    
    args = parser.parse_args()

    
    process_csv(args.input, args.output, args.flash) 