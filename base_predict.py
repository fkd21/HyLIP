# latency_predictor.py
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import importlib.util
import sys

# 确保这些导入指向您项目中模块的实际位置
from core.hardware import Hardware
from operators.matmul import get_matmul_latency
from operators.activation import Softmax, SwiGLU, GELU, GLU, GeGLU
from operators.normalization import LayerNorm, RMSNorm
from operators.communication_primitive import AllReduce, AllGather, ReduceScatter
# 假设 fused_attention_estimator.py 位于 'operators' 目录下
from operators.fused_attention_estimator import get_fused_attention_latency


class LatencyPredictor:
    def __init__(
        self,
        model_config_path: str, # 指向 Python 配置文件 (例如 qwen2_config.py)
        hardware_config_path: str,
        parallel_mode: str = "tensor",
        num_devices: int = 1,
        dtype: str = "fp16",
        batch_size: int = 1
    ):
        self.hardware = Hardware(hardware_config_path)
        self.model_config = self._load_model_config(model_config_path)

        self.parallel_mode = parallel_mode
        if self.hardware.has_multi_device():
            self.num_devices = min(num_devices, self.hardware.device_count)
        else:
            self.num_devices = 1
        
        if self.parallel_mode == "pipeline" and self.num_devices == 1:
            self.parallel_mode = "tensor" # 若流水线并行设备数为1，则回退到张量并行（顺序执行）

        self.dtype = dtype
        self.batch_size = batch_size
        self.word_size = {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype, 2) # 每元素的字节数

        # 初始化算子实例
        self.softmax = Softmax(self.hardware)
        self.layer_norm = LayerNorm(self.hardware) # 用于使用 LayerNorm 的模型
        self.rms_norm = RMSNorm(self.hardware)   # 用于使用 RMSNorm 的模型
        self.swiglu = SwiGLU(self.hardware)     # 用于 SwiGLU 激活函数
        self.gelu = GELU(self.hardware)         # 用于 GELU 或类似的 SiLU (如果没有专门的 SiLU 估算器)
        self.glu = GLU(self.hardware)
        self.geglu = GeGLU(self.hardware)
        # 如果您有专门的 SiLU 估算器:
        # from operators.activation import SiLU # 假设存在
        # self.silu = SiLU(self.hardware)


        # 初始化通信原语 (仅当实际使用多于1个设备时)
        if self.num_devices > 1 and self.hardware.has_multi_device():
            self.all_reduce = AllReduce(self.hardware)
            self.all_gather = AllGather(self.hardware)
            self.reduce_scatter = ReduceScatter(self.hardware)
        else:
            self.all_reduce = None
            self.all_gather = None
            self.reduce_scatter = None

    def _load_model_config(self, model_config_path: str) -> Any:
        """
        从给定的 Python 文件加载模型配置类。
        例如: model_config_path = "path/to/qwen2_config.py"
                 期望内部的类名: Qwen2_72B (或类似名称)
        """
        config_dir = os.path.dirname(os.path.abspath(model_config_path))
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
        
        module_name_from_file = os.path.basename(model_config_path).split('.')[0]

        try:
            module = importlib.import_module(module_name_from_file)
            
            # First, try to find any class that inherits from ModelConfig
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue  # Skip private attributes
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, 'get_local_config') and hasattr(attr, 'get_layers') and hasattr(attr, 'get_layer_graph'):
                    # Found a class that has the required methods
                    try:
                        return attr()  # Instantiate it
                    except Exception as e:
                        print(f"Warning: Found class {attr_name} but couldn't instantiate it: {e}")
                        continue
            
            # If no suitable class found by inspection, try heuristic class name derivation
            # 启发式构建类名，例如从 qwen2_72b 构建 Qwen2_72B
            parts = module_name_from_file.split('_')
            class_name_candidate_parts = []
            for part in parts:
                if part.lower() == 'config': continue # 跳过文件名中的 'config' 部分
                if part.isdigit():
                    class_name_candidate_parts.append(part)
                elif part.endswith('b') and part[:-1].isdigit() and len(part)>1: # 例如 7b -> 7B
                    class_name_candidate_parts.append(part[:-1] + part[-1].upper())
                else:
                    class_name_candidate_parts.append(part[0].upper() + part[1:])
            
            # Try different class name formats
            class_name_formats = [
                "_".join(class_name_candidate_parts),  # Llama_2_7B or Llama2_7B
                "".join(class_name_candidate_parts),   # Llama27B
                "".join(class_name_candidate_parts) + "Config", # Llama27BConfig
                "_".join(class_name_candidate_parts) + "Config" # Llama_2_7B_Config
            ]
            
            for class_name in class_name_formats:
                try:
                    model_config_class = getattr(module, class_name)
                    return model_config_class()  # Instantiate the class
                except (AttributeError, TypeError):
                    continue  # Try next format
            
            # Fallback to listing available classes for debugging
            print(f"Error: Could not find a suitable model config class in module '{module_name_from_file}'")
            print(f"Available classes: {[attr for attr in dir(module) if not attr.startswith('_') and isinstance(getattr(module, attr), type)]}")
            raise ImportError(f"Could not find a suitable model config class in module '{module_name_from_file}'")
            
        except ImportError as e:
            print(f"Error: Cannot import module '{module_name_from_file}' from path '{model_config_path}'. Error: {e}")
            sys.exit(1)


    def _get_op_latency(self, op_dict: Dict[str, Any], current_batch_size: int, current_seq_len: int, 
                        is_tp_sharded: bool = False, tp_sharded_dim_factor: int = 1) -> float:
        """ 辅助函数，用于获取各种非 MatMul 算子的延迟，可能针对 TP 进行分片 """
        op_type = op_dict.get("type", op_dict["name"]) 
        in_shape_template = op_dict["in_shape"]
        
        def adjust_b_s(shape_template: Union[Tuple, List], b: int, s: int) -> Union[Tuple, List]:
            # 在形状模板中调整批处理大小 (B) 和序列长度 (S)。
            # S 通常在 (B,S,D) 中的索引为1，或在 (B,H,S,D) 中的索引为2。
            if isinstance(shape_template, tuple):
                if len(shape_template) == 3: # (B_template, S_template, D)
                    return (b, s, shape_template[2])
                if len(shape_template) == 4: # (B_template, H, S_template, D_head) 或 (B,H,Sq,Skv)
                    return (b, shape_template[1], s, shape_template[3]) 
            return shape_template #无法识别或标量形状

        if isinstance(in_shape_template, list): # 针对具有多个输入的算子 (例如 Add, SwiGLU)
            adjusted_in_shape = [adjust_b_s(s_template, current_batch_size, current_seq_len) for s_template in in_shape_template]
        else:
            adjusted_in_shape = adjust_b_s(in_shape_template, current_batch_size, current_seq_len)

        # 如果适用，将 TP 分片应用于相关维度
        if is_tp_sharded and self.num_devices > 1:
            # 确保 tp_sharded_dim_factor 对于 TP 是正确的 self.num_devices
            tp_sharded_dim_factor = self.num_devices 
            
            # 辅助函数，用于对单个形状进行分片
            def shard_shape(current_shape_tuple):
                temp_list = list(current_shape_tuple)
                if op_type in ["SwiGLU", "GELU", "GLU", "GeGLU", "SiLU"]: # 逐元素作用于隐藏/中间特征
                     if len(temp_list) == 3: # (B,S,D_intermediate_sharded)
                         temp_list[2] = temp_list[2] // tp_sharded_dim_factor
                elif op_type == "Softmax" and len(temp_list) == 4 : # 输入 (B,H_sharded,Sq,Skv)
                         temp_list[1] = temp_list[1] // tp_sharded_dim_factor 
                # RMSNorm/LayerNorm 通常作用于完整的隐藏维度，除非明确设计为分片范数。
                # 如果范数的输入是分片的，可能意味着在此之前有一个 AllGather，或者是分片范数。
                # 目前假设在此调用之前范数输入形状已正确处理。
                return tuple(temp_list)

            if isinstance(adjusted_in_shape, list):
                adjusted_in_shape = [shard_shape(s) if isinstance(s, tuple) else s for s in adjusted_in_shape]
            elif isinstance(adjusted_in_shape, tuple):
                adjusted_in_shape = shard_shape(adjusted_in_shape)
        
        lat_val = 0.0
        res: Union[float, Dict[str, float]] = 0.0 # 确保 res 已类型化

        # 根据 op_type 分派到特定的延迟估算器
        if op_type == "RMSNorm":
             res = self.rms_norm.estimate_latency(adjusted_in_shape, self.dtype)
        elif op_type == "LayerNorm":
             res = self.layer_norm.estimate_latency(adjusted_in_shape, self.dtype)
        elif op_type == "Softmax":
             # Softmax 输入形状 (B,H,Sq,Skv) 应由 _adjust_operators_for_phase 正确设置
             # 如果 TP 对 H 进行分片，adjusted_in_shape 将反映这一点。
             res = self.softmax.estimate_latency(adjusted_in_shape, self.dtype)
        elif op_type == "SwiGLU": 
             res = self.swiglu.estimate_latency(adjusted_in_shape, self.dtype) # adjusted_in_shape 是一个列表
        elif op_type == "GELU": # 假设 SiLU 在没有特定 SiLU 估算器的情况下可能使用 GELU 的延迟
             res = self.gelu.estimate_latency(adjusted_in_shape, self.dtype)
        elif op_type == "SiLU": # 如果您有特定的 SiLU 估算器
             # res = self.silu.estimate_latency(adjusted_in_shape, self.dtype) # 如果 self.silu 存在，则取消注释
             res = self.gelu.estimate_latency(adjusted_in_shape, self.dtype) # 目前回退到 GELU
        elif op_type == "GLU":
             res = self.glu.estimate_latency(adjusted_in_shape, self.dtype)
        elif op_type == "GeGLU":
             res = self.geglu.estimate_latency(adjusted_in_shape, self.dtype)
        elif op_type == "Add" or op_dict["name"] in ["attn_add", "mlp_add"]: 
            # 逐元素加法，受内存限制。延迟基于输出张量大小。
            # 来自 op_dict 的输出形状模板:
            out_shape_template = op_dict["out_shape"]
            data_shape = adjust_b_s(out_shape_template, current_batch_size, current_seq_len)
            
            if is_tp_sharded and len(data_shape)==3 and self.num_devices > 1: # 如果 Add 的输出是分片的
                data_shape = (data_shape[0], data_shape[1], data_shape[2] // self.num_devices)
            
            total_elements = np.prod(data_shape)
            total_bytes = total_elements * self.word_size
            # 假设内存流量为2次读取（输入）+ 1次写入（输出），或者更简单：输出大小占主导。
            # 使用输出大小进行带宽计算是一种常见的简化方法。
            res = total_bytes / self.hardware.memory_bandwidth.get("GPU", 1e12) # 使用 .get 以确保安全
        # FusedAttention 现在由 _calculate_..._layer_latency 方法直接处理
        # else if op_type == "FusedAttention": ...
        else: 
            # print(f"警告: 未知 op_type '{op_type}' (名称: {op_dict['name']})。返回0延迟。")
            return 0.0 # 或通过日志/错误处理未知算子

        if isinstance(res, dict): lat_val = res.get("latency", 0.0)
        elif isinstance(res, (float, int)): lat_val = float(res)
        else:
            # print(f"警告: {op_type} 的延迟结果类型意外: {type(res)}。使用 0.0。")
            lat_val = 0.0
            
        return lat_val

    def predict_latency(self, seq_len: int, use_flash_attention: bool = False) -> Dict[str, float]:
        self.model_config.set_seq_len(seq_len) # 使用当前序列长度更新模型配置
        
        # 根据 flash attention 标志获取层图和算子模板
        layer_graph = self.model_config.get_layer_graph(use_flash_attention)
        num_total_layers, operators_template = self.model_config.get_layers(use_flash_attention)

        prefill_latency = self._predict_phase_latency(
            num_total_layers, operators_template, layer_graph, is_prefill=True, current_seq_len=seq_len, use_flash_attention=use_flash_attention
        )
        decode_latency = self._predict_phase_latency(
            num_total_layers, operators_template, layer_graph, is_prefill=False, current_seq_len=1, use_flash_attention=use_flash_attention
        ) # decode 的 current_seq_len 为1 (针对新 token)
        
        return {"prefill_latency": prefill_latency, "decode_latency": decode_latency}

    def _predict_phase_latency(self, num_layers: int, operators_template: List[Dict], 
                               layer_graph: Dict[str, List[str]], is_prefill: bool, 
                               current_seq_len: int, use_flash_attention: bool) -> float: # 添加了 use_flash_attention
        # 针对特定阶段调整算子形状 (prefill vs decode 的 S=1 for Q)
        # total_s_for_kv 是来自 model_config 的完整上下文长度
        current_operators = self._adjust_operators_for_phase(
            operators_template, is_prefill, self.model_config.get_seq_len()
        )
        
        effective_parallel_mode = self.parallel_mode
        if self.parallel_mode == "pipeline" and self.num_devices <= 1:
            effective_parallel_mode = "tensor" # 如果流水线并行不可行，则回退到顺序执行

        if effective_parallel_mode == "tensor":
            return self._predict_tensor_parallel_phase(num_layers, current_operators, layer_graph, is_prefill, current_seq_len)
        elif effective_parallel_mode == "pipeline": # 暗示 self.num_devices > 1
            return self._predict_pipeline_parallel_phase(num_layers, current_operators, layer_graph, is_prefill, current_seq_len)
        else: 
            raise ValueError(f"未知的并行模式: {self.parallel_mode}")


    def _predict_tensor_parallel_phase(self, num_layers: int, operators: List[Dict], 
                                       layer_graph: Dict[str, List[str]], is_prefill: bool, 
                                       current_s_for_batch: int) -> float:
        # 计算一个 Transformer 层在张量并行下的延迟
        layer_latency_tp = self._calculate_tensor_parallel_layer_latency(
            operators, layer_graph, is_prefill, current_s_for_batch, self.model_config.get_seq_len() # total_s_for_kv
        )
        total_latency_layers = layer_latency_tp * num_layers

        # Embedding 查找延迟
        embedding_data_shape = (self.batch_size, current_s_for_batch, self.model_config.get_hidden_size())
        embedding_bytes = np.prod(embedding_data_shape) * self.word_size
        embedding_latency = embedding_bytes / self.hardware.memory_bandwidth.get("GPU", 1e12)

        # 最终层范数 (应用于最后一个 Transformer 层的输出)
        final_norm_shape = (self.batch_size, current_s_for_batch, self.model_config.get_hidden_size())
        # 最终范数的输入假定是完整的 (如果最后一层的输出是行并行 TP，则已 AllReduce)
        final_norm_res = self.rms_norm.estimate_latency(final_norm_shape, self.dtype) # 假设最终范数也使用 RMSNorm
        final_norm_latency = final_norm_res["latency"] if isinstance(final_norm_res, dict) else float(final_norm_res)
        
        # 最终投影到 logits (LM Head)
        M_lm_head = self.batch_size * current_s_for_batch
        # LM head 的词汇表大小针对 TP 列并行进行分片
        N_lm_head_sharded_val = self.model_config.get_vocab_size()
        if self.num_devices > 1: N_lm_head_sharded_val /= self.num_devices
        N_lm_head_sharded = int(N_lm_head_sharded_val)
        K_lm_head = self.model_config.get_hidden_size()
        
        # get_matmul_latency 返回 (latency, tflops_achieved, mem_bw_util)
        matmul_res_tuple = get_matmul_latency(M_lm_head, N_lm_head_sharded, K_lm_head, self.hardware, self.word_size, self.dtype, self.batch_size)
        final_proj_compute_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)
        
        final_proj_comm_latency = 0.0
        if self.num_devices > 1 and self.all_gather: # AllGather 分片的 logits
            all_gather_shape = (self.batch_size, current_s_for_batch, self.model_config.get_vocab_size())
            comm_res = self.all_gather.estimate_latency(all_gather_shape, self.dtype)
            final_proj_comm_latency = comm_res["latency"] if isinstance(comm_res, dict) else float(comm_res)
            
        final_proj_latency = final_proj_compute_latency + final_proj_comm_latency
        total_latency = embedding_latency + total_latency_layers + final_norm_latency + final_proj_latency
        return total_latency

    def _predict_pipeline_parallel_phase(self, num_layers: int, operators: List[Dict], 
                                         layer_graph: Dict[str, List[str]], is_prefill: bool, 
                                         current_s_for_batch: int) -> float:
        # 此方法仅在 self.num_devices > 1 时调用。
        # 单个层顺序执行的延迟 (此简化 PP 模型中阶段内无 TP)
        single_layer_sequential_latency = self._calculate_sequential_layer_latency(
            operators, layer_graph, is_prefill, current_s_for_batch, self.model_config.get_seq_len() # total_s_for_kv
        )

        if num_layers == 0 : return 0.0
        # 假设层在流水线阶段完美平衡
        layers_per_stage = num_layers / self.num_devices 
        compute_time_per_stage = layers_per_stage * single_layer_sequential_latency

        # 流水线阶段之间的激活张量大小
        activation_shape = (self.batch_size, current_s_for_batch, self.model_config.get_hidden_size())
        activation_bytes = np.prod(activation_shape) * self.word_size
        
        pipeline_comm_latency = 0.0
        # 检查硬件是否具有链路带宽和延迟属性
        if self.hardware.has_multi_device() and \
           hasattr(self.hardware, 'link_bandwidth_both_direction') and \
           hasattr(self.hardware, 'link_latency'):
            # 流水线气泡的简单 P2P 通信模型
            pipeline_comm_latency = (activation_bytes / self.hardware.link_bandwidth_both_direction) + self.hardware.link_latency
        
        # 流水线延迟: 气泡时间 + (总计算时间 / 阶段数，如果完美重叠 - 此处未使用)
        # 或: 一个阶段的时间 * 阶段数 (计算) + (阶段数 - 1) * 通信延迟 (气泡)
        # 这是当一个批次流过时的常见推理模型。
        total_transformer_latency_pipelined = (self.num_devices * compute_time_per_stage) + \
                                              ((self.num_devices - 1) * pipeline_comm_latency)

        # Embedding, Final Norm, LM Head 成本 (假设它们在第一/最后阶段运行，PP 阶段内不进行 TP)
        embedding_data_shape = (self.batch_size, current_s_for_batch, self.model_config.get_hidden_size())
        embedding_bytes = np.prod(embedding_data_shape) * self.word_size
        embedding_latency = embedding_bytes / self.hardware.memory_bandwidth.get("GPU", 1e12)

        final_norm_shape = (self.batch_size, current_s_for_batch, self.model_config.get_hidden_size())
        final_norm_res = self.rms_norm.estimate_latency(final_norm_shape, self.dtype) # 假设 RMSNorm
        final_norm_latency = final_norm_res["latency"] if isinstance(final_norm_res, dict) else float(final_norm_res)
        
        # LM Head (在最后阶段顺序执行)
        M_lm_head = self.batch_size * current_s_for_batch
        N_lm_head = self.model_config.get_vocab_size() # 完整词汇表，此处 N 无 TP 分片
        K_lm_head = self.model_config.get_hidden_size()
        matmul_res_tuple = get_matmul_latency(M_lm_head, N_lm_head, K_lm_head, self.hardware, self.word_size, self.dtype, self.batch_size)
        final_proj_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)
                                                                                             
        # 总延迟: 流水线化层的总和 + 第一/最后阶段的非流水线化部分。
        # 此模型假设 embedding/final_proj 对流水线关键路径是累加的。
        total_latency = total_transformer_latency_pipelined + embedding_latency + final_norm_latency + final_proj_latency
        return total_latency


    def _calculate_tensor_parallel_layer_latency(self, operators: List[Dict], layer_graph: Dict[str, List[str]], 
                                                is_prefill: bool, current_s_for_batch: int, total_s_for_kv: int) -> float:
        sorted_ops = self._topological_sort(operators, layer_graph)
        op_latencies: Dict[str, float] = {} 
        # 跟踪算子的 *输出* 是否由于 TP 而在隐藏/特征维度上分片
        op_output_sharded_state: Dict[str, bool] = {} 

        for op_name in sorted_ops:
            op_dict = next((o for o in operators if o["name"] == op_name), None)
            if op_dict is None: continue

            max_dep_latency = 0.0
            # 检查 op_name 是否是图中的一个键 (即，它是否有列出的依赖项)
            # 如果一个算子是根节点 (没有列出其依赖项)，则 max_dep_latency 保持为0。
            if op_name in layer_graph: 
                dependencies = layer_graph.get(op_name, [])
                for dep in dependencies:
                    if dep in op_latencies: max_dep_latency = max(max_dep_latency, op_latencies[dep])
            
            current_op_compute_latency = 0.0
            current_op_comm_latency = 0.0
            op_output_sharded_state[op_name] = False # 此算子输出的默认分片状态
            op_type = op_dict.get("type", op_dict["name"]) # 获取算子类型

            # --- 线性层 (MatMul) 处理 ---
            if op_type == "Linear":
                # 来自 op_dict 的输入形状是模板 (B_template, S_template, D_in)
                # 此 matmul 的有效 S 需要确定 (current_s_for_batch 或 total_s_for_kv)
                # 这由 _adjust_operators_for_phase 处理，它会更新 op_dict["in_shape"]
                effective_in_shape_template = op_dict["in_shape"] # 已由 _adjust_operators_for_phase 调整
                
                # M = Batch * 此 Matmul 的 Effective_Sequence_Length
                M = self.batch_size * effective_in_shape_template[1] # S 在 (B,S,D) 中的索引为1
                logical_K_dim = effective_in_shape_template[2] # D_in
                logical_N_dim = op_dict["out_shape"][2]        # D_out

                actual_N_for_matmul = logical_N_dim
                actual_K_for_matmul = logical_K_dim
                
                parallel_strat = op_dict.get("parallel_strategy", "none")

                if self.num_devices > 1 and parallel_strat != "none":
                    if parallel_strat == "column":
                        actual_N_for_matmul = logical_N_dim / self.num_devices
                        op_output_sharded_state[op_name] = True # 输出是分片的
                    elif parallel_strat == "row":
                        # 输入 K 是分片的 (K_eff = K_logical / TP)
                        actual_K_for_matmul = logical_K_dim / self.num_devices
                        # 输出需要 AllReduce。AllReduce 的输出形状: (B, S_eff, N_logical)
                        if self.all_reduce:
                            allreduce_shape = (self.batch_size, effective_in_shape_template[1], logical_N_dim)
                            comm_res = self.all_reduce.estimate_latency(allreduce_shape, self.dtype)
                            current_op_comm_latency = comm_res["latency"] if isinstance(comm_res, dict) else float(comm_res)
                        op_output_sharded_state[op_name] = False # AllReduce 后变为完整
                    # else: op_output_sharded_state[op_name] 保持 False (此算子不进行 TP 分片)

                matmul_res_tuple = get_matmul_latency(M, int(actual_N_for_matmul), int(actual_K_for_matmul), 
                                                   self.hardware, self.word_size, self.dtype, self.batch_size)
                current_op_compute_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)
            
            # --- Attention QK MatMul 处理 ---
            elif op_name == "qk_matmul": # 如果类型是通用的 "MatMul"，则显式检查名称
                # 来自 op_dict["in_shape"] = [q_shape, k_shape] 的形状已由 _adjust_operators_for_phase 调整
                # q_shape: (B, H_q_total, S_q, D_h), k_shape: (B, H_kv_total, S_kv, D_h)
                q_shape_adj = op_dict["in_shape"][0] # (B, H_q_total_template, S_q, D_h)
                k_shape_adj = op_dict["in_shape"][1] # (B, H_kv_total_template, S_kv, D_h)

                num_q_heads_total = self.model_config.get_num_attention_heads()
                num_kv_heads_total = self.model_config.get_num_key_value_heads()
                d_head = self.model_config.get_head_dim()

                # TP 的每设备有效头数
                heads_q_per_device = num_q_heads_total
                heads_kv_per_device_for_k_tensor = num_kv_heads_total # 这是 K 张量分片的头数

                if self.num_devices > 1:
                    heads_q_per_device = num_q_heads_total / self.num_devices
                    # GQA/MQA 在 TP 下的 KV 头分片策略:
                    if num_kv_heads_total % self.num_devices == 0: # 可均分
                        heads_kv_per_device_for_k_tensor = num_kv_heads_total / self.num_devices
                    elif num_kv_heads_total < self.num_devices: # KV 组少于 TP ranks -> 复制 KV 头
                        heads_kv_per_device_for_k_tensor = num_kv_heads_total 
                    else: # KV 头多于 TP ranks，但不能完美均分 (复杂情况，近似为分片)
                        heads_kv_per_device_for_k_tensor = num_kv_heads_total / self.num_devices
                
    
                M_qk = self.batch_size * int(heads_q_per_device) * q_shape_adj[2] # S_q from q_shape_adj
                N_qk = k_shape_adj[2] # S_kv from k_shape_adj
                K_qk = d_head
                
                if M_qk > 0 and N_qk > 0 and K_qk > 0:
                     matmul_res_tuple = get_matmul_latency(M_qk, N_qk, K_qk, self.hardware, self.word_size, self.dtype, self.batch_size)
                     current_op_compute_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)
                else: current_op_compute_latency = 0.0
                # 如果 TP > 1，QK_matmul 的输出沿 Q-heads 维度分片
                op_output_sharded_state[op_name] = (self.num_devices > 1)

            # --- Attention SV MatMul 处理 ---
            elif op_name == "sv_matmul": # 显式名称检查
                # 来自 op_dict["in_shape"] = [scores_shape, v_shape] 的形状已调整
                scores_shape_adj = op_dict["in_shape"][0] # (B, H_q_total_template, S_q, S_kv)
                # v_shape_adj = op_dict["in_shape"][1]      # (B, H_kv_total_template, S_kv, D_h)

                num_q_heads_total = self.model_config.get_num_attention_heads()
                # num_kv_heads_total = self.model_config.get_num_key_value_heads() # 此处 M 不直接使用
                d_head = self.model_config.get_head_dim()
                
                heads_q_per_device = num_q_heads_total / self.num_devices if self.num_devices > 1 else num_q_heads_total
                
                # Matmul 维度: Scores @ V
                # Scores: (B * heads_q_per_device * S_q, S_kv)
                # V: (S_kv, D_h) (假设 V 是 (B * heads_kv_per_device_for_v_tensor * S_kv, D_h) 并且已广播/分组)
                M_sv = self.batch_size * int(heads_q_per_device) * scores_shape_adj[2] # S_q from scores_shape_adj
                N_sv = d_head
                K_sv = scores_shape_adj[3] # S_kv from scores_shape_adj (或 v_shape_adj[2])
                
                if M_sv > 0 and N_sv > 0 and K_sv > 0:
                    matmul_res_tuple = get_matmul_latency(M_sv, N_sv, K_sv, self.hardware, self.word_size, self.dtype, self.batch_size)
                    current_op_compute_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)
                else: current_op_compute_latency = 0.0
                # 如果 TP > 1，输出沿 Q-heads 维度分片
                op_output_sharded_state[op_name] = (self.num_devices > 1)
            
            # --- FusedAttention 处理 ---
            elif op_type == "FusedAttention":
                # op_dict["in_shape"] 是一个列表: [q_shape_tuple, k_shape_tuple, v_shape_tuple]
                # q_shape_tuple = (current_batch_size, num_total_q_heads_template, seq_len_q_effective, head_dim)
                # k_shape_tuple = (current_batch_size, num_total_kv_heads_template, seq_len_kv_effective, head_dim)
                # v_shape_tuple = (current_batch_size, num_total_kv_heads_template, seq_len_kv_effective, head_dim)
                # 这些形状已由 _adjust_operators_for_phase 调整过 S_q 和 S_kv

                q_actual_shape = op_dict["in_shape"][0]
                k_actual_shape = op_dict["in_shape"][1]
                # v_actual_shape = op_dict["in_shape"][2] # v_actual_shape[2] (S_kv) should be same as k_actual_shape[2]

                num_q_heads_total = self.model_config.get_num_attention_heads()
                num_kv_heads_total = self.model_config.get_num_key_value_heads()
                head_dim_config = self.model_config.get_head_dim()

                seq_len_q_eff = q_actual_shape[2] # S_q 在 (B, H, S, D_h) 中的索引为2
                seq_len_kv_eff = k_actual_shape[2] # S_kv 在 (B, H, S, D_h) 中的索引为2
                
                effective_q_heads = num_q_heads_total
                effective_kv_heads = num_kv_heads_total
                
                if self.num_devices > 1:
                    effective_q_heads = num_q_heads_total / self.num_devices
                    if num_kv_heads_total == 0: # 如果未定义，则避免除以零
                        effective_kv_heads = 0
                    elif num_kv_heads_total % self.num_devices == 0:
                         effective_kv_heads = num_kv_heads_total / self.num_devices
                    elif num_kv_heads_total < self.num_devices: # 如果 KV 头少于 TP 设备，则复制 KV 头
                         effective_kv_heads = num_kv_heads_total
                    else: # KV 头更多，但不能完美均分 (近似为分片)
                         effective_kv_heads = num_kv_heads_total / self.num_devices
                
                current_op_compute_latency = get_fused_attention_latency(
                    batch_size=self.batch_size, # 这是 self.batch_size
                    num_q_heads=int(effective_q_heads),
                    num_kv_heads=int(effective_kv_heads),
                    seq_len_q=seq_len_q_eff,
                    seq_len_kv=seq_len_kv_eff,
                    head_dim=head_dim_config,
                    hardware=self.hardware,
                    word_size_bits=self.word_size * 8, # 将字节转换为位
                    data_type=self.dtype,
                    is_causal=True # LLM 通常是因果的；如果需要，可以配置此项
                )
                # FusedAttention 的输出沿 Q-head 维度分片 (如果 Q-head 被分片)
                op_output_sharded_state[op_name] = (self.num_devices > 1 and effective_q_heads < num_q_heads_total)

            # --- 其他算子 (激活函数, 范数, 逐元素加法) ---
            else: 
                is_input_sharded_for_generic_op = False
                if self.num_devices > 1 and op_name in layer_graph:
                    for dep_op_name in layer_graph.get(op_name, []): # 检查当前算子的依赖项
                        if op_output_sharded_state.get(dep_op_name, False): # 如果任何依赖项的输出是分片的
                            is_input_sharded_for_generic_op = True
                            break
                
                current_op_compute_latency = self._get_op_latency(
                    op_dict, self.batch_size, current_s_for_batch, # current_s_for_batch 用于 _get_op_latency 中的 B,S 调整
                    is_tp_sharded=is_input_sharded_for_generic_op, 
                    tp_sharded_dim_factor=self.num_devices # 如果 is_tp_sharded 为 True，_get_op_latency 将使用此因子
                )
                # 逐元素算子通常保持其主要输入的分片状态
                op_output_sharded_state[op_name] = is_input_sharded_for_generic_op


            op_total_latency = current_op_compute_latency + current_op_comm_latency
            op_latencies[op_name] = max_dep_latency + op_total_latency
            if not op_latencies: return 0.0
        
        # 从此层组件图中汇聚节点的延迟确定层延迟
        all_op_names_in_list = {op["name"] for op in operators}
        all_dependent_nodes = set()
        for op_target, deps_list in layer_graph.items():
            if op_target in all_op_names_in_list: # 仅考虑当前层算子列表中的算子
                for dep in deps_list:
                    if dep in all_op_names_in_list:
                        all_dependent_nodes.add(dep)
        
        # 汇聚节点是算子列表中那些不作为列表中任何其他算子依赖项的节点
        sink_nodes_names = [op_n for op_n in all_op_names_in_list if op_n not in all_dependent_nodes and op_n in op_latencies]
        
        if not sink_nodes_names: 
            # 如果图格式不正确或所有算子都指向此列表之外的内容，
            # 则回退到此层组件中任何已计算算子的最大延迟。
             return max(op_latencies.values()) if op_latencies else 0.0

        final_layer_latency = 0.0
        for sink_op_name in sink_nodes_names:
            final_layer_latency = max(final_layer_latency, op_latencies.get(sink_op_name,0.0))
        return final_layer_latency


    def _calculate_sequential_layer_latency(self, operators: List[Dict], layer_graph: Dict[str, List[str]], 
                                            is_prefill: bool, current_s_for_batch: int, total_s_for_kv: int) -> float:
        # 这类似于 _calculate_tensor_parallel_layer_latency，但 num_devices=1 (无分片，无 TP 通信)
        sorted_ops = self._topological_sort(operators, layer_graph)
        op_latencies: Dict[str, float] = {}

        for op_name in sorted_ops:
            op_dict = next((o for o in operators if o["name"] == op_name), None)
            if op_dict is None: continue

            max_dep_latency = 0.0
            if op_name in layer_graph:
                dependencies = layer_graph.get(op_name, [])
                for dep in dependencies:
                    if dep in op_latencies: max_dep_latency = max(max_dep_latency, op_latencies[dep])
            
            current_op_compute_latency = 0.0
            op_type = op_dict.get("type", op_dict["name"])

            if op_type == "Linear":
                effective_in_shape_template = op_dict["in_shape"] # 已由 _adjust_operators_for_phase 调整
                M = self.batch_size * effective_in_shape_template[1] 
                N = op_dict["out_shape"][2] 
                K = effective_in_shape_template[2]
                matmul_res_tuple = get_matmul_latency(M, N, K, self.hardware, self.word_size, self.dtype, self.batch_size)
                current_op_compute_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)

            elif op_name == "qk_matmul":
                q_shape_adj = op_dict["in_shape"][0] # (B, H_q, S_q, D_h)
                k_shape_adj = op_dict["in_shape"][1] # (B, H_kv, S_kv, D_h)
                # 对于顺序执行，使用完整的 num_q_heads。对于 GQA，H_kv 可能不同。
                # Matmul 是 (B*H_q*S_q, D_h) @ (D_h, S_kv)，如果 GQA，则对 H_kv 进行适当的广播。
                # 或者，有效地，M=B*H_q*S_q, N=S_kv, K=D_h。
                M_qk = self.batch_size * q_shape_adj[1] * q_shape_adj[2] # B * H_q_total * S_q
                N_qk = k_shape_adj[2] # S_kv
                K_qk = q_shape_adj[3] # D_h
                if M_qk > 0 and N_qk > 0 and K_qk > 0:
                    matmul_res_tuple = get_matmul_latency(M_qk, N_qk, K_qk, self.hardware, self.word_size, self.dtype, self.batch_size)
                    current_op_compute_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)
                else: current_op_compute_latency = 0.0

            elif op_name == "sv_matmul":
                scores_shape_adj = op_dict["in_shape"][0] # (B, H_q, S_q, S_kv)
                v_shape_adj = op_dict["in_shape"][1]      # (B, H_kv, S_kv, D_h)
                # M=B*H_q*S_q, N=D_h, K=S_kv
                M_sv = self.batch_size * scores_shape_adj[1] * scores_shape_adj[2] # B * H_q_total * S_q
                N_sv = v_shape_adj[3] # D_h
                K_sv = scores_shape_adj[3] # S_kv
                if M_sv > 0 and N_sv > 0 and K_sv > 0:
                    matmul_res_tuple = get_matmul_latency(M_sv, N_sv, K_sv, self.hardware, self.word_size, self.dtype, self.batch_size)
                    current_op_compute_latency = matmul_res_tuple[0] if isinstance(matmul_res_tuple, tuple) else float(matmul_res_tuple)
                else: current_op_compute_latency = 0.0
            
            elif op_type == "FusedAttention":
                # 输入形状已由 _adjust_operators_for_phase 调整
                q_shape_adj = op_dict["in_shape"][0]
                k_shape_adj = op_dict["in_shape"][1]
                # v_shape_adj = op_dict["in_shape"][2]

                num_q_heads_total = self.model_config.get_num_attention_heads()
                num_kv_heads_total = self.model_config.get_num_key_value_heads()
                head_dim_config = self.model_config.get_head_dim()

                seq_len_q_eff = q_shape_adj[2] # S_q
                seq_len_kv_eff = k_shape_adj[2] # S_kv

                current_op_compute_latency = get_fused_attention_latency(
                    batch_size=self.batch_size,
                    num_q_heads=num_q_heads_total, # 顺序执行，使用总头数
                    num_kv_heads=num_kv_heads_total,
                    seq_len_q=seq_len_q_eff,
                    seq_len_kv=seq_len_kv_eff,
                    head_dim=head_dim_config,
                    hardware=self.hardware,
                    word_size_bits=self.word_size * 8,
                    data_type=self.dtype,
                    is_causal=True # 假设 LLM 是因果的
                )
            else: # 其他算子 (范数, 激活函数, 加法)
                current_op_compute_latency = self._get_op_latency(op_dict, self.batch_size, 
                                                                  current_s_for_batch, # S 用于 B,S 调整
                                                                  is_tp_sharded=False) # 无 TP 分片

            op_latencies[op_name] = max_dep_latency + current_op_compute_latency
            #print(f"op_name: {op_name}, op_compute_latency: {current_op_compute_latency}, max_dep_latency: {max_dep_latency}")
        
        if not op_latencies: return 0.0
        # 从汇聚节点确定层延迟
        all_op_names_in_list = {op["name"] for op in operators}
        all_dependent_nodes = set()
        for op_target, deps_list in layer_graph.items(): # 已修正迭代
            if op_target in all_op_names_in_list:
                 for d in deps_list: # 确保迭代 deps_list 中的每个依赖项
                    if d in all_op_names_in_list:
                        all_dependent_nodes.add(d)

        sink_nodes_names = [op_n for op_n in all_op_names_in_list if op_n not in all_dependent_nodes and op_n in op_latencies]
        if not sink_nodes_names: return max(op_latencies.values()) if op_latencies else 0.0
        final_layer_latency = 0.0
        for sink_op_name in sink_nodes_names:
            final_layer_latency = max(final_layer_latency, op_latencies.get(sink_op_name,0.0))
        return final_layer_latency

    def _adjust_operators_for_phase(self, operators_template: List[Dict], is_prefill: bool, 
                                    total_s_for_kv: int) -> List[Dict]:
        adjusted_operators = []
        s_q = total_s_for_kv if is_prefill else 1
        s_kv = total_s_for_kv 

        for op_template in operators_template:
            adj_op = {key: val for key, val in op_template.items()} # Shallow copy is fine for top-level dict

            op_n = adj_op["name"]
            op_type = adj_op.get("type")

            def _adjust_s_in_single_shape_tuple(shape_template_tuple: Tuple, s_to_use: int, 
                                             is_kv_dim_for_qk_out: bool = False, s_kv_val_for_qk_out: Optional[int] = None) -> Tuple:
                shape = list(shape_template_tuple) 
                if len(shape) == 3: 
                    shape[1] = s_to_use
                elif len(shape) == 4: 
                    if is_kv_dim_for_qk_out and s_kv_val_for_qk_out is not None: 
                        shape[2] = s_to_use 
                        shape[3] = s_kv_val_for_qk_out 
                    else: 
                        shape[2] = s_to_use
                return tuple(shape) 

            current_in_shape_template = op_template.get("in_shape") 
            
            if isinstance(current_in_shape_template, list) and current_in_shape_template and isinstance(current_in_shape_template[0], tuple):
                new_in_shapes = []
                if op_n == "qk_matmul" or (op_type == "MatMul" and len(current_in_shape_template) == 2 and current_in_shape_template[0][1] == current_in_shape_template[1][1]):
                    new_in_shapes.append(_adjust_s_in_single_shape_tuple(current_in_shape_template[0], s_q))
                    new_in_shapes.append(_adjust_s_in_single_shape_tuple(current_in_shape_template[1], s_kv))
                elif op_n == "sv_matmul" or (op_type == "MatMul" and len(current_in_shape_template) == 2 and current_in_shape_template[0][3] == current_in_shape_template[1][2]):
                    new_in_shapes.append(_adjust_s_in_single_shape_tuple(current_in_shape_template[0], s_q, True, s_kv)) 
                    new_in_shapes.append(_adjust_s_in_single_shape_tuple(current_in_shape_template[1], s_kv)) 
                elif op_type == "FusedAttention": 
                    new_in_shapes.append(_adjust_s_in_single_shape_tuple(current_in_shape_template[0], s_q))
                    new_in_shapes.append(_adjust_s_in_single_shape_tuple(current_in_shape_template[1], s_kv))
                    new_in_shapes.append(_adjust_s_in_single_shape_tuple(current_in_shape_template[2], s_kv))
                elif op_type == "SwiGLU" or op_type == "Add" or op_type == "GLU" or op_type == "GeGLU": 
                    for s_tpl in current_in_shape_template:
                        new_in_shapes.append(_adjust_s_in_single_shape_tuple(s_tpl, s_q))
                else: 
                    new_in_shapes = [s_tpl for s_tpl in current_in_shape_template] 
                adj_op["in_shape"] = new_in_shapes 
            elif isinstance(current_in_shape_template, tuple): 
                adj_op["in_shape"] = _adjust_s_in_single_shape_tuple(current_in_shape_template, s_q) 
            
            current_out_shape_template = op_template.get("out_shape")
            if isinstance(current_out_shape_template, tuple):
                if op_n == "qk_matmul": 
                    adj_op["out_shape"] = _adjust_s_in_single_shape_tuple(current_out_shape_template, s_q, True, s_kv)
                else: 
                    adj_op["out_shape"] = _adjust_s_in_single_shape_tuple(current_out_shape_template, s_q)
            
            adjusted_operators.append(adj_op)
        return adjusted_operators

    def _topological_sort(self, operators: List[Dict[str, Any]], graph: Dict[str, List[str]]) -> List[str]:
        """ 执行 Kahn 算法进行拓扑排序。 """
        op_names_in_list = {op["name"] for op in operators}
        
        # 后继节点的邻接表和入度映射
        adj: Dict[str, List[str]] = {op_name: [] for op_name in op_names_in_list}
        in_degree: Dict[str, int] = {op_name: 0 for op_name in op_names_in_list}

        for op_name, dependencies in graph.items():
            if op_name not in op_names_in_list: continue # 图中的算子不在当前算子列表中
            
            for dep_name in dependencies:
                if dep_name not in op_names_in_list: continue # 图中的依赖项不在当前算子列表中
                
                # dep_name 是 op_name 的前驱节点
                adj.setdefault(dep_name, []).append(op_name)
                in_degree[op_name] = in_degree.get(op_name, 0) + 1
        
        # 使用入度为0的节点初始化队列
        queue = [op_name for op_name in op_names_in_list if in_degree.get(op_name, 0) == 0]
        sorted_list = []
        
        while queue:
            u = queue.pop(0)
            sorted_list.append(u)
            
            # 对于 u 的每个后继节点 v
            for v_successor in adj.get(u, []): # 使用 .get 以确保安全，如果 u 没有列出的后继节点
                in_degree[v_successor] -= 1
                if in_degree[v_successor] == 0:
                    queue.append(v_successor)
            
        if len(sorted_list) != len(op_names_in_list):
            # 这表示图中存在循环或从根节点无法到达的算子。
            # print(f"警告: 拓扑排序可能不完整。已排序: {len(sorted_list)}, 总算子数: {len(op_names_in_list)}。"
            #       f" 层图中可能存在循环或断开的组件。")
            # 添加任何剩余的未排序算子，尽管它们的顺序在此处不能由拓扑排序保证。
            for opn in op_names_in_list:
                if opn not in sorted_list:
                    sorted_list.append(opn) 
        return sorted_list


# --- Main 执行示例 (需要虚拟或实际引用的模块) ---
if __name__ == "__main__":
    import argparse



    default_model_config_file = "qwen2_config.py" 



    parser = argparse.ArgumentParser(description='预测 LLM 推理延迟 (V4)')
    parser.add_argument('--model_config', type=str, default=default_model_config_file, help='模型配置 Python 文件路径')
    parser.add_argument('--hardware_config', type=str, default=dummy_hw_file, help='硬件配置 JSON 文件路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--seq_len', type=int, default=128, help='输入序列长度')
    parser.add_argument('--parallel_mode', type=str, default='tensor', choices=['tensor', 'pipeline'], help='并行策略')
    parser.add_argument('--num_devices', type=int, default=1, help='设备数量')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'], help='数据类型')
    parser.add_argument('--flash_attn', action='store_true', help='使用 Flash Attention')
    
    args = parser.parse_args()
    
    config_dir = os.path.dirname(os.path.abspath(args.model_config))
    if config_dir not in sys.path: sys.path.insert(0, config_dir)

    print(f"使用以下参数运行延迟预测: {args}")
    
    predictor = LatencyPredictor( # 直接实例化
        model_config_path=args.model_config,
        hardware_config_path=args.hardware_config,
        parallel_mode=args.parallel_mode,
        num_devices=args.num_devices,
        dtype=args.dtype,
        batch_size=args.batch_size
    )
    latencies = predictor.predict_latency(args.seq_len, args.flash_attn)
    
    print(f"\n--- 结果 ---")
    print(f"Prefill 延迟: {latencies['prefill_latency']:.6f} 秒")
    print(f"Decode 延迟 (每 token): {latencies['decode_latency']:.6f} 秒")
    if latencies['decode_latency'] > 0:
        print(f"Tokens per second (decode): {1.0/latencies['decode_latency']:.2f} tokens/秒")

def predict_latency(
    model_config_path: str,
    hardware_config_path: str,
    seq_len: int,
    batch_size: int = 1,
    parallel_mode: str = "tensor",
    num_devices: int = 1,
    dtype: str = "fp16",
    use_flash_attention: bool = False
) -> Dict[str, float]:
    """
    Predict latency for a given model and hardware configuration.
    
    Args:
        model_config_path: Path to the model configuration file
        hardware_config_path: Path to the hardware configuration file
        seq_len: Input sequence length
        batch_size: Batch size
        parallel_mode: Parallelization mode ("tensor" or "pipeline")
        num_devices: Number of devices to use
        dtype: Data type ("fp16", "bf16", or "fp32")
        use_flash_attention: Whether to use flash attention
        
    Returns:
        Dictionary with prefill_latency and decode_latency
    """
    predictor = LatencyPredictor(
        model_config_path=model_config_path,
        hardware_config_path=hardware_config_path,
        parallel_mode=parallel_mode,
        num_devices=num_devices,
        dtype=dtype,
        batch_size=batch_size
    )
    return predictor.predict_latency(seq_len, use_flash_attention)

