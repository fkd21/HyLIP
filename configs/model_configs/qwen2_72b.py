from transformers import AutoConfig 
from typing import Dict, List, Tuple, Any
from configs.model_configs.model_config import ModelConfig 





class Qwen2_72B(ModelConfig):
    def __init__(self, seq_len=2048):
        # Initialize the model with name and sequence length
        self.model_name = "Qwen/Qwen2-72B"
        self.model_params = self.get_local_config()
        self.seq_len = seq_len

    def get_local_config(self):
        """Provide local model configuration."""
        return {
            "architectures": ["Qwen2ForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_qwen2.Qwen2Config",
                "AutoModelForCausalLM": "modeling_qwen2.Qwen2ForCausalLM"
            },
            "bias": False,
            "hidden_act": "silu", # This implies SiLU for MLP activation
            "hidden_size": 8192,
            "initializer_range": 0.02,
            "intermediate_size": 22528, # mlp_hidden_size or D_ffn
            "max_position_embeddings": 32768,
            "model_type": "qwen2",
            "num_attention_heads": 64,
            "num_hidden_layers": 80,
            "num_key_value_heads": 8, # GQA
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 2.0,
                "type": "linear"
            },
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 151936
        }

    # Implement abstract methods from ModelConfig
    def get_hidden_size(self) -> int:
        return self.model_params["hidden_size"]

    def get_num_attention_heads(self) -> int:
        return self.model_params["num_attention_heads"]

    def get_num_key_value_heads(self) -> int:
        return self.model_params["num_key_value_heads"]

    def get_intermediate_size(self) -> int:
        return self.model_params["intermediate_size"]

    def get_num_hidden_layers(self) -> int:
        return self.model_params["num_hidden_layers"]
    
    def get_vocab_size(self) -> int:
        return self.model_params["vocab_size"]


    def get_layer_graph(self, use_flash_attention: bool = False) -> Dict[str, List[str]]:
        """Return the layer graph with dependencies."""
        # Ensure 'input' and 'output' nodes used here match names in get_layers()
        transformer_layer_graph = {
            # "input" is a conceptual start, not an op with latency in this context of single layer.
            # Dependencies should point to actual ops.
            "attn_norm": [], # Root of the layer if "input" is just a marker
            "q_proj": ["attn_norm"],
            "k_proj": ["attn_norm"],
            "v_proj": ["attn_norm"],
            "qk_matmul": ["q_proj", "k_proj"], # Assumes RoPE is fused or part of proj
            "softmax": ["qk_matmul"],
            "sv_matmul": ["softmax", "v_proj"],
            "out_proj": ["sv_matmul"],
            "attn_add": ["attn_norm", "out_proj"], # Residual connection: input to attn_norm + out_proj
            "mlp_norm": ["attn_add"],
            "gate_proj": ["mlp_norm"],
            "up_proj": ["mlp_norm"],
            "mlp_act": ["gate_proj", "up_proj"],
            "down_proj": ["mlp_act"],
            "mlp_add": ["attn_add", "down_proj"], # Residual connection: output of attn_add + down_proj
            # "output": ["mlp_add"] # Conceptual end of the layer block
        }

        flashattention_transformer_layer_graph = {
            "attn_norm": [],
            "q_proj": ["attn_norm"],
            "k_proj": ["attn_norm"],
            "v_proj": ["attn_norm"],
            "fused_attention": ["q_proj", "k_proj", "v_proj"], # This op replaces qk_matmul, softmax, sv_matmul
            "out_proj": ["fused_attention"], # Output projection input is now fused_attention
            "attn_add": ["attn_norm", "out_proj"],
            "mlp_norm": ["attn_add"],
            "gate_proj": ["mlp_norm"],
            "up_proj": ["mlp_norm"],
            "mlp_act": ["gate_proj", "up_proj"],
            "down_proj": ["mlp_act"],
            "mlp_add": ["attn_add", "down_proj"],
            # "output": ["mlp_add"]
        }
        if use_flash_attention:
            return flashattention_transformer_layer_graph
        else:
            return transformer_layer_graph

    def get_layers(self, use_flash_attention: bool = False) -> Tuple[int, List[Dict[str, Any]]]:
        """Return the total number of layers and operator details for one layer."""
        batch_size = 1 # Template batch size
        seq_len = self.get_seq_len() # Current sequence length
        hidden_size = self.get_hidden_size()
        num_q_heads = self.get_num_attention_heads()
        num_kv_heads = self.get_num_key_value_heads()
        head_dim = self.get_head_dim()
        intermediate_size = self.get_intermediate_size()
        total_layers = self.get_num_hidden_layers()
        
        # word_size is determined by dtype in predictor, not fixed here.
        # parallel_strategy added for linear layers.
        # type added for specific ops like RMSNorm, SwiGLU.

        common_linear_ops = [
            {"name": "q_proj", "type": "Linear", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, num_q_heads * head_dim), "parallel_strategy": "column"},
            {"name": "k_proj", "type": "Linear", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, num_kv_heads * head_dim), "parallel_strategy": "column"},
            {"name": "v_proj", "type": "Linear", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, num_kv_heads * head_dim), "parallel_strategy": "column"},
        ]

        if use_flash_attention:
            # IMPORTANT: The 'fused_attention' op needs a corresponding latency model in the predictor.
            # Its in_shape might be a list of Q,K,V shapes, or it might be more abstract.
            # Its out_shape is typically (batch_size, seq_len, num_q_heads * head_dim) before out_proj.
            # This is a simplified placeholder.
            attention_ops = [
                *common_linear_ops,
                {"name": "fused_attention", "type": "FusedAttention", 
                 "in_shape": [ # List of inputs Q, K, V after projection and reshape
                              (batch_size, num_q_heads, seq_len, head_dim), 
                              (batch_size, num_kv_heads, seq_len, head_dim), 
                              (batch_size, num_kv_heads, seq_len, head_dim)
                            ], 
                 "out_shape": (batch_size, num_q_heads, seq_len, head_dim), # Output of attention mechanism part
                 # No parallel_strategy here as it's a complex op; sharding is internal or based on inputs.
                },
                 # out_proj's input shape needs to match fused_attention's output after potential reshape
                {"name": "out_proj", "type": "Linear", "in_shape": (batch_size, seq_len, num_q_heads * head_dim), "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            ]
        else:
            attention_ops = [
                *common_linear_ops,
                {"name": "qk_matmul", "type": "MatMul", "in_shape": [(batch_size, num_q_heads, seq_len, head_dim), (batch_size, num_kv_heads, seq_len, head_dim)], "out_shape": (batch_size, num_q_heads, seq_len, seq_len)},
                {"name": "softmax", "type": "Softmax", "in_shape": (batch_size, num_q_heads, seq_len, seq_len), "out_shape": (batch_size, num_q_heads, seq_len, seq_len)},
                {"name": "sv_matmul", "type": "MatMul", "in_shape": [(batch_size, num_q_heads, seq_len, seq_len), (batch_size, num_kv_heads, seq_len, head_dim)], "out_shape": (batch_size, num_q_heads, seq_len, head_dim)},
                {"name": "out_proj", "type": "Linear", "in_shape": (batch_size, seq_len, num_q_heads * head_dim), "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            ]

        operators = [
            # Conceptual input to the layer (residual from previous)
            # {"name": "input_residual_marker", "type": "Marker", "in_shape": None, "out_shape": (batch_size, seq_len, hidden_size)},
            
            {"name": "attn_norm", "type": "RMSNorm", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, hidden_size)},
            *attention_ops,
            {"name": "attn_add", "type": "Add", "in_shape": [(batch_size, seq_len, hidden_size), (batch_size, seq_len, hidden_size)], "out_shape": (batch_size, seq_len, hidden_size)},
            
            {"name": "mlp_norm", "type": "RMSNorm", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, hidden_size)},
            {"name": "gate_proj", "type": "Linear", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, intermediate_size), "parallel_strategy": "column"},
            {"name": "up_proj", "type": "Linear", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, intermediate_size), "parallel_strategy": "column"},
            # Qwen2 uses SiLU activation, typically as part of SwiGLU: SiLU(gate_proj) * up_proj
            {"name": "mlp_act", "type": "SwiGLU", "in_shape": [(batch_size, seq_len, intermediate_size), (batch_size, seq_len, intermediate_size)], "out_shape": (batch_size, seq_len, intermediate_size)},
            {"name": "down_proj", "type": "Linear", "in_shape": (batch_size, seq_len, intermediate_size), "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            {"name": "mlp_add", "type": "Add", "in_shape": [(batch_size, seq_len, hidden_size), (batch_size, seq_len, hidden_size)], "out_shape": (batch_size, seq_len, hidden_size)},
            
            # Conceptual output of the layer
            # {"name": "output_marker", "type": "Marker", "in_shape": (batch_size, seq_len, hidden_size), "out_shape": (batch_size, seq_len, hidden_size)},
        ]
        
        return total_layers, operators

