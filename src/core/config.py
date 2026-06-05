import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directories
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
LLM_MODEL_PATH = str(MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
EMBEDDING_MODEL_DIR = str(MODELS_DIR / "embeddings")
EMBEDDING_ONNX_PATH = str(MODELS_DIR / "embeddings" / "model.onnx")
DB_PATH = str(DATA_DIR / "memory.db")

# LLM Config
N_GPU_LAYERS = -1
CTX_SIZE = 4096
SYSTEM_PROMPT = "You are a concise, helpful local assistant. Keep responses brief."
