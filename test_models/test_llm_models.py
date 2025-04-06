import torch
from gpt2_model import GPT2Model
from bert_model import BERTModel
from llama2_model import Llama2Model
from t5_model import T5Model
from roberta_model import RoBERTaModel
from bloom_model import BLOOMModel
from opt_model import OPTModel
from deepseek_model import DeepSeekModel

def test_model(model, prompt: str):
    print(f"\nTesting {model.__class__.__name__}")
    print("-" * 50)
    
    # Test text generation
    print("Generating text:")
    generated_text = model.generate(prompt)
    print(f"Input: {prompt}")
    print(f"Output: {generated_text}")
    
    # Test embeddings
    print("\nGetting embeddings:")
    embeddings = model.get_embeddings(prompt)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding mean: {embeddings.mean().item():.4f}")
    print(f"Embedding std: {embeddings.std().item():.4f}")

def main():
    # Test prompt
    prompt = "The quick brown fox jumps over the lazy dog"
    
    # Initialize models
    models = [
        GPT2Model(),
        BERTModel(),
        Llama2Model(),
        T5Model(),
        RoBERTaModel(),
        BLOOMModel(),
        OPTModel(),
        DeepSeekModel()
    ]
    
    # Test each model
    for model in models:
        try:
            test_model(model, prompt)
        except Exception as e:
            print(f"Error testing {model.__class__.__name__}: {str(e)}")

if __name__ == "__main__":
    main() 