import torch
from abc import ABC, abstractmethod

class BaseLLMModel(ABC):
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text based on the input prompt"""
        pass
    
    @abstractmethod
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for the input text"""
        pass
    
    def to_device(self, inputs):
        """Move inputs to the specified device"""
        if isinstance(inputs, dict):
            return {k: v.to(self.device) for k, v in inputs.items()}
        return inputs.to(self.device) 