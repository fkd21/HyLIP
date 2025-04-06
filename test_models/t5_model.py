from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from .base_model import BaseLLMModel

class T5Model(BaseLLMModel):
    def __init__(self, model_name: str = "t5-base", device: str = None):
        super().__init__(model_name, device)
        self.tokenizer = None
        
    def load_model(self):
        """Load T5 model and tokenizer"""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using T5"""
        if self.model is None:
            self.load_model()
            
        # T5 requires a prefix for text generation
        prefix = "translate English to English: "
        inputs = self.tokenizer(prefix + prompt, return_tensors="pt")
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings from T5's encoder output"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model.encoder(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling 