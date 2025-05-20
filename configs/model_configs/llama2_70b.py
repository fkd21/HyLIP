from transformers import AutoConfig
from typing import Dict, List, Tuple, Any
from configs.model_configs.model_config import ModelConfig


class Llama2_70B(ModelConfig):
    def __init__(self, seq_len=2048):
        # 仅使用模型名称和序列长度初始化
        self.model_name = "meta-llama/Llama-2-70B-chat"
        self.model_params = self.get_local_config()
        self.seq_len = seq_len
        
    def get_local_config(self):
        """提供本地模型配置信息，避免从 Hugging Face 加载"""
        return {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 8192,
            "initializer_range": 0.02,
            "intermediate_size": 28672,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 64,
            "num_hidden_layers": 80,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 32000
        }
        
    def get_layer_graph(self, use_flash_attention: bool = False) -> Dict[str, List[str]]:
        """Return the layer graph with dependencies."""
        if use_flash_attention:
            return self.flashattention_transformer_layer_graph
        else:
            return self.transformer_layer_graph
    
    def get_layers(self, use_flash_attention: bool = False) -> Tuple[int, List[Dict[str, Any]]]:
        """Return the total number of layers and operator details for one layer."""
        batch_size = 1
        seq_len = self.get_seq_len()
        hidden_size = self.get_hidden_size()
        num_heads = self.get_num_attention_heads()
        num_kv_heads = self.get_num_key_value_heads()
        head_dim = self.get_head_dim()
        intermediate_size = self.get_intermediate_size()
        total_layers = self.get_num_hidden_layers()
        
        # Operator details for one transformer layer
        operators = [
            {
                "name": "attn_norm",
                "type": "RMSNorm",
                "in_shape": (batch_size, seq_len, hidden_size),
                "out_shape": (batch_size, seq_len, hidden_size),
                "parallel_strategy": "none"
            },
            {
                "name": "q_proj", 
                "type": "Linear",
                "in_shape": (batch_size, seq_len, hidden_size),
                "out_shape": (batch_size, seq_len, num_heads * head_dim),
                "parallel_strategy": "column"
            },
            {
                "name": "k_proj",
                "type": "Linear",
                "in_shape": (batch_size, seq_len, hidden_size),
                "out_shape": (batch_size, seq_len, num_kv_heads * head_dim),
                "parallel_strategy": "column"
            },
            {
                "name": "v_proj",
                "type": "Linear",
                "in_shape": (batch_size, seq_len, hidden_size),
                "out_shape": (batch_size, seq_len, num_kv_heads * head_dim),
                "parallel_strategy": "column"
            },
        ]
        
        if use_flash_attention:
            attention_ops = [
                {"name": "fused_attention", "type": "FusedAttention", 
                 "in_shape": [
                     (batch_size, num_heads, seq_len, head_dim), 
                     (batch_size, num_kv_heads, seq_len, head_dim), 
                     (batch_size, num_kv_heads, seq_len, head_dim)
                 ], 
                 "out_shape": (batch_size, num_heads, seq_len, head_dim)
                },
                {"name": "out_proj", "type": "Linear", "in_shape": (batch_size, seq_len, num_heads * head_dim), 
                 "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            ]
        else:
            attention_ops = [
                {"name": "qk_matmul", "type": "MatMul", 
                 "in_shape": [(batch_size, num_heads, seq_len, head_dim), (batch_size, num_kv_heads, seq_len, head_dim)], 
                 "out_shape": (batch_size, num_heads, seq_len, seq_len)},
                {"name": "softmax", "type": "Softmax", "in_shape": (batch_size, num_heads, seq_len, seq_len), 
                 "out_shape": (batch_size, num_heads, seq_len, seq_len)},
                {"name": "sv_matmul", "type": "MatMul", 
                 "in_shape": [(batch_size, num_heads, seq_len, seq_len), (batch_size, num_kv_heads, seq_len, head_dim)], 
                 "out_shape": (batch_size, num_heads, seq_len, head_dim)},
                {"name": "out_proj", "type": "Linear", "in_shape": (batch_size, seq_len, num_heads * head_dim), 
                 "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            ]
                
        operators.extend(attention_ops)
        operators.extend([
            {"name": "attn_add", "type": "Add", 
             "in_shape": [(batch_size, seq_len, hidden_size), (batch_size, seq_len, hidden_size)], 
             "out_shape": (batch_size, seq_len, hidden_size)},
            
            {"name": "mlp_norm", "type": "RMSNorm", 
             "in_shape": (batch_size, seq_len, hidden_size), 
             "out_shape": (batch_size, seq_len, hidden_size)},
            {"name": "gate_proj", "type": "Linear", 
             "in_shape": (batch_size, seq_len, hidden_size), 
             "out_shape": (batch_size, seq_len, intermediate_size), "parallel_strategy": "column"},
            {"name": "up_proj", "type": "Linear", 
             "in_shape": (batch_size, seq_len, hidden_size), 
             "out_shape": (batch_size, seq_len, intermediate_size), "parallel_strategy": "column"},
            {"name": "mlp_act", "type": "SwiGLU", 
             "in_shape": [(batch_size, seq_len, intermediate_size), (batch_size, seq_len, intermediate_size)], 
             "out_shape": (batch_size, seq_len, intermediate_size)},
            {"name": "down_proj", "type": "Linear", 
             "in_shape": (batch_size, seq_len, intermediate_size), 
             "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            {"name": "mlp_add", "type": "Add", 
             "in_shape": [(batch_size, seq_len, hidden_size), (batch_size, seq_len, hidden_size)], 
             "out_shape": (batch_size, seq_len, hidden_size)},
        ])
        
        return total_layers, operators

    # Layer graph definitions
    transformer_layer_graph = {
        "input":[],
        "attn_norm": ["input"],
        "q_proj":["attn_norm"],
        "k_proj":["attn_norm"],
        "v_proj":["attn_norm"],
        "qk_matmul":["q_proj","k_proj"],
        "softmax":["qk_matmul"],
        "sv_matmul":["softmax","v_proj"],
        "out_proj":["sv_matmul"],
        "attn_add":["input","out_proj"],
        "mlp_norm":["attn_add"],
        "gate_proj":["mlp_norm"],
        "up_proj":["mlp_norm"],
        "mlp_act":["up_proj","gate_proj"],
        "down_proj":["mlp_act"],
        "mlp_add":["attn_add","down_proj"],
        "output":["mlp_add"]
    }

    flashattention_transformer_layer_graph = {
        "input":[],
        "attn_norm": ["input"],
        "q_proj":["attn_norm"],
        "k_proj":["attn_norm"],
        "v_proj":["attn_norm"],
        "fused_attention":["q_proj","k_proj","v_proj"],
        "out_proj":["fused_attention"],
        "attn_add":["input","out_proj"],
        "mlp_norm":["attn_add"],
        "gate_proj":["mlp_norm"],
        "up_proj":["mlp_norm"],
        "mlp_act":["up_proj","gate_proj"],
        "down_proj":["mlp_act"],
        "mlp_add":["attn_add","down_proj"],
        "output":["mlp_add"]
    }
