from transformers import BloomForCausalLM, BloomTokenizer
import torch
from .base_model import BaseLLMModel

class BLOOMModel(BaseLLMModel):
    def __init__(self, model_name: str = "bigscience/bloom-560m", device: str = None):
        super().__init__(model_name, device)
        self.tokenizer = None
        
    def load_model(self):
        """Load BLOOM model and tokenizer"""
        self.tokenizer = BloomTokenizer.from_pretrained(self.model_name)
        self.model = BloomForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.to(self.device)
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using BLOOM"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(prompt, return_tensors="pt")
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
        """Get embeddings from BLOOM's last hidden state"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling 