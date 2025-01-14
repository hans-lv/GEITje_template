## Intro

This template provides a clean starting point for using the [GEITje-7B-chat-v2](https://huggingface.co/Rijgersberg/GEITje-7B-chat-v2) model with PyTorch, although any other model could probably be substituted. 

**Note:** The test file [`tests/test_GEITje.py`](https://github.com/hans-lv/GEITje_template/blob/main/tests/test_GEITje.py) is currently configured to use 4-bit mode due to VRAM limitations. If your system has sufficient resources (16GB VRAM), you can update it to use 8-bit mode for better performance.

## Requirements

- Windows
- A CUDA-capable GPU
- Python

## Steps to install

1. Clone this repo locally

1. Check Python and CUDA version

   ```
   python --version
   nvidia-smi
   ```

1. Create virtual environment

   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

1. Install PyTorch and dependencies

   ```
   pip3 install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

1. Run `python test_cuda.py` to test the setup

1. Go to [https://huggingface.co/Rijgersberg/GEITje-7B-chat-v2/tree/main](https://huggingface.co/Rijgersberg/GEITje-7B-chat-v2/tree/main)
   Get these files and put them in the right folder: (Warning, large files!).

   ```
   project_root/
   ├── models/
   │   └── GEITje-7B-chat-v2/
   │       ├── config.json
   │       ├── tokenizer.json
   │       ├── tokenizer_config.json
   │       └── model-####-of-#####.safetensors
   ├── test_cuda.py
   ├── venv/
   ```

1. Run `python test_GEITje.py`, you should see a response from the model.
