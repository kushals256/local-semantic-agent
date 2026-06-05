import time
from llama_cpp import Llama
from .config import LLM_MODEL_PATH, N_GPU_LAYERS, CTX_SIZE, SYSTEM_PROMPT

class LLMEngine:
    def __init__(self):
        print(f"Loading LLM from {LLM_MODEL_PATH} with n_gpu_layers={N_GPU_LAYERS}...")
        self.llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=CTX_SIZE,
            verbose=False
        )
        self.system_prompt = SYSTEM_PROMPT
        self.model_path = LLM_MODEL_PATH
        self.n_gpu_layers = N_GPU_LAYERS
        self.ctx_size = CTX_SIZE
        print("LLM loaded.")

    def generate(self, user_prompt: str, context: str = ""):
        """
        Generator that streams tokens back and measures TTFT and tok/s.
        Yields dictionaries with 'token' and 'metrics'.
        """
        # Qwen 2.5 Instruct Format
        prompt = f"<|im_start|>system\n{self.system_prompt}\n"
        if context:
            prompt += f"\nRelevant Memory Context:\n{context}\n"
        prompt += f"<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        start_time = time.time()
        ttft = None
        token_count = 0
        
        stream = self.llm(
            prompt,
            max_tokens=512,
            stream=True,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        
        for output in stream:
            if ttft is None:
                ttft = time.time() - start_time
                
            token = output["choices"][0]["text"]
            token_count += 1
            
            # Current tokens per second
            elapsed = time.time() - start_time
            tok_sec = token_count / elapsed if elapsed > 0 else 0
            
            yield {
                "token": token,
                "metrics": {
                    "ttft": round(ttft, 3),
                    "tok_sec": round(tok_sec, 1)
                }
            }
