from transformers import BertModel, BertTokenizer
import torch
from .base_model import BaseLLMModel

class BERTModel(BaseLLMModel):
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        super().__init__(model_name, device)
        self.tokenizer = None
        
    def load_model(self):
        """Load BERT model and tokenizer"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """BERT doesn't support text generation, return the input"""
        return prompt
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings from BERT's [CLS] token"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding 