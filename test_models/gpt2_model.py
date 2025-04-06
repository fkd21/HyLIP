from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from .base_model import BaseLLMModel

class GPT2Model(BaseLLMModel):
    def __init__(self, model_name: str = "gpt2", device: str = None):
        super().__init__(model_name, device)
        self.tokenizer = None
        
    def load_model(self):
        """Load GPT-2 model and tokenizer"""
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using GPT-2"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings from GPT-2's last hidden state"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling 