import torch
from transformers import AutoTokenizer

def test_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

def test_transformers():
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
        print("Transformers test: Success")
    except Exception as e:
        print(f"Transformers error: {e}")

if __name__ == "__main__":
    test_cuda()
    test_transformers()