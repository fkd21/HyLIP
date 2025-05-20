from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class ModelConfig(ABC):
    def __init__(self, model_name, seq_len=2048):
        # 这个初始化不再被使用，子类直接实现自己的初始化方法
        self.model_name = model_name
        self.seq_len = seq_len

    def get_model_params(self):
        return self.model_params
    
    def get_num_attention_heads(self):
        return self.model_params["num_attention_heads"]

    def get_hidden_size(self):
        return self.model_params["hidden_size"]

    def get_num_key_value_heads(self):
        return self.model_params.get("num_key_value_heads", self.get_num_attention_heads())

    def get_head_dim(self):
        return self.get_hidden_size() // self.get_num_attention_heads()

    def get_norm_layers(self):
        return ["attn_norm", "mlp_norm"]

    def get_num_hidden_layers(self):
        return self.model_params["num_hidden_layers"]

    def get_intermediate_size(self):
        return self.model_params["intermediate_size"]

    def get_vocab_size(self):
        return self.model_params["vocab_size"]
    
    def get_seq_len(self):
        return self.seq_len
    
    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
    
    @abstractmethod
    def get_local_config(self) -> Dict:
        """Return the local model configuration as a dictionary."""
        pass
    
    @abstractmethod
    def get_layers(self) -> Tuple[int, List[Dict[str, Any]]]:
        """Return the total number of layers and a list of operator details for one layer.
        
        Returns:
            Tuple containing:
                - Total number of layers
                - List of dictionaries, each containing:
                    - name: Operator name
                    - in_shape: Input shape (tuple or list of tuples for multiple inputs)
                    - out_shape: Output shape
                    - word_size: Word size in bytes
                    - dependencies: List of operator names this operator depends on
        """
        pass
    
    @abstractmethod
    def get_layer_graph(self, use_flash_attention: bool = False) -> Dict[str, List[str]]:
        """Return the layer graph with dependencies.
        
        Args:
            use_flash_attention: Whether to use flash attention optimization
            
        Returns:
            Dictionary mapping operator names to lists of input operator names
        """
        pass