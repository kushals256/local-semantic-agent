import os
from huggingface_hub import hf_hub_download
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def download_models():
    print("Downloading Qwen2.5-0.5B-Instruct GGUF model...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False
    )
    
    print("Downloading all-MiniLM-L6-v2 ONNX files...")
    files = [
        "onnx/model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt"
    ]
    for file in files:
        hf_hub_download(
            repo_id="Xenova/all-MiniLM-L6-v2",
            filename=file,
            local_dir=EMBEDDINGS_DIR,
            local_dir_use_symlinks=False
        )
        
    # Flatten the onnx directory for easier access
    onnx_path = EMBEDDINGS_DIR / "onnx" / "model.onnx"
    target_path = EMBEDDINGS_DIR / "model.onnx"
    if onnx_path.exists():
        os.replace(onnx_path, target_path)
        try:
            os.rmdir(EMBEDDINGS_DIR / "onnx")
        except OSError:
            pass

    print("✅ All models downloaded successfully!")

if __name__ == "__main__":
    download_models()
