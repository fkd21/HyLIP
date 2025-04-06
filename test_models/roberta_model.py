from transformers import RobertaModel, RobertaTokenizer
import torch
from .base_model import BaseLLMModel

class RoBERTaModel(BaseLLMModel):
    def __init__(self, model_name: str = "roberta-base", device: str = None):
        super().__init__(model_name, device)
        self.tokenizer = None
        
    def load_model(self):
        """Load RoBERTa model and tokenizer"""
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """RoBERTa doesn't support text generation, return the input"""
        return prompt
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings from RoBERTa's last hidden state"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state[:, 0, :]  # First token embedding 