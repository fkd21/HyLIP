from transformers import AutoConfig
from typing import Dict, List, Tuple, Any
from configs.model_configs.model_config import ModelConfig


class GptJ6b(ModelConfig):
    def __init__(self, seq_len=2048):
        # Initialize the model with name and sequence length
        self.model_name = "EleutherAI/gpt-j-6b"
        self.model_params = self.get_local_config()
        self.seq_len = seq_len
        
    def get_local_config(self):
        """Provide local model configuration to avoid loading from Hugging Face"""
        return {
            "activation_function": "gelu_new",
            "architectures": ["GPTJForCausalLM"],
            "attn_pdrop": 0.0,
            "bos_token_id": 50256,
            "embd_pdrop": 0.0,
            "eos_token_id": 50256,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "model_type": "gptj",
            "n_embd": 4096,
            "n_head": 16,
            "n_layer": 28,
            "n_positions": 2048,
            "resid_pdrop": 0.0,
            "rotary": True,
            "rotary_dim": 64,
            "tie_word_embeddings": False,
            "transformers_version": "4.16.0.dev0",
            "use_cache": True,
            "vocab_size": 50400
        }
        
    def get_hidden_size(self):
        """Return hidden dimension size."""
        return self.model_params["n_embd"]
    
    def get_num_attention_heads(self):
        """Return number of attention heads."""
        return self.model_params["n_head"]
    
    def get_num_hidden_layers(self):
        """Return number of hidden layers."""
        return self.model_params["n_layer"]
    
    def get_intermediate_size(self):
        """Return intermediate size for feed-forward network."""
        # GPT-J uses 4 * hidden_size for the intermediate size
        return 4 * self.get_hidden_size()
        
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
        head_dim = hidden_size // num_heads
        intermediate_size = self.get_intermediate_size()
        total_layers = self.get_num_hidden_layers()
        
        # Operator details for one transformer layer
        operators = [
            {
                "name": "attn_norm",
                "type": "LayerNorm",
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
                "out_shape": (batch_size, seq_len, num_heads * head_dim),
                "parallel_strategy": "column"
            },
            {
                "name": "v_proj",
                "type": "Linear",
                "in_shape": (batch_size, seq_len, hidden_size),
                "out_shape": (batch_size, seq_len, num_heads * head_dim),
                "parallel_strategy": "column"
            },
        ]
        
        if use_flash_attention:
            attention_ops = [
                {"name": "fused_attention", "type": "FusedAttention", 
                 "in_shape": [
                     (batch_size, num_heads, seq_len, head_dim), 
                     (batch_size, num_heads, seq_len, head_dim), 
                     (batch_size, num_heads, seq_len, head_dim)
                 ], 
                 "out_shape": (batch_size, num_heads, seq_len, head_dim)
                },
                {"name": "out_proj", "type": "Linear", "in_shape": (batch_size, seq_len, num_heads * head_dim), 
                 "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            ]
        else:
            attention_ops = [
                {"name": "qk_matmul", "type": "MatMul", 
                 "in_shape": [(batch_size, num_heads, seq_len, head_dim), (batch_size, num_heads, seq_len, head_dim)], 
                 "out_shape": (batch_size, num_heads, seq_len, seq_len)},
                {"name": "softmax", "type": "Softmax", "in_shape": (batch_size, num_heads, seq_len, seq_len), 
                 "out_shape": (batch_size, num_heads, seq_len, seq_len)},
                {"name": "sv_matmul", "type": "MatMul", 
                 "in_shape": [(batch_size, num_heads, seq_len, seq_len), (batch_size, num_heads, seq_len, head_dim)], 
                 "out_shape": (batch_size, num_heads, seq_len, head_dim)},
                {"name": "out_proj", "type": "Linear", "in_shape": (batch_size, seq_len, num_heads * head_dim), 
                 "out_shape": (batch_size, seq_len, hidden_size), "parallel_strategy": "row"},
            ]
                
        operators.extend(attention_ops)
        operators.extend([
            {"name": "attn_add", "type": "Add", 
             "in_shape": [(batch_size, seq_len, hidden_size), (batch_size, seq_len, hidden_size)], 
             "out_shape": (batch_size, seq_len, hidden_size)},
            
            {"name": "mlp_norm", "type": "LayerNorm", 
             "in_shape": (batch_size, seq_len, hidden_size), 
             "out_shape": (batch_size, seq_len, hidden_size)},
            {"name": "gate_proj", "type": "Linear", 
             "in_shape": (batch_size, seq_len, hidden_size), 
             "out_shape": (batch_size, seq_len, intermediate_size), "parallel_strategy": "column"},
            {"name": "mlp_act", "type": "GELU", 
             "in_shape": (batch_size, seq_len, intermediate_size),
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
        "mlp_act":["gate_proj"],
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
        "mlp_act":["gate_proj"],
        "down_proj":["mlp_act"],
        "mlp_add":["attn_add","down_proj"],
        "output":["mlp_add"]
    } 