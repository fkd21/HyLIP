from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from .base_model import BaseLLMModel

class Llama2Model(BaseLLMModel):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", device: str = None):
        super().__init__(model_name, device)
        self.tokenizer = None
        
    def load_model(self):
        """Load Llama2 model and tokenizer"""
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.to(self.device)
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using Llama2"""
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
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings from Llama2's last hidden state"""
        if self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling 