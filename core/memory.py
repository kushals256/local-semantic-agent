import sqlite3
import numpy as np
import onnxruntime as ort
import time
from transformers import AutoTokenizer
from .config import DB_PATH, EMBEDDING_ONNX_PATH, EMBEDDING_MODEL_DIR

class MemoryStore:
    def __init__(self, db_path: str = None):
        target_db = db_path if db_path is not None else DB_PATH
        self.conn = sqlite3.connect(target_db, check_same_thread=False)
        self._init_db()
        
        print(f"Loading embedding model from {EMBEDDING_MODEL_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_DIR)
        self.ort_session = ort.InferenceSession(EMBEDDING_ONNX_PATH)
        print("Memory store loaded.")

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                embedding_blob BLOB,
                timestamp REAL,
                source TEXT
            )
        ''')
        self.conn.commit()

    def embed(self, text: str) -> np.ndarray:
        """Generates embeddings using ONNX Runtime with mean pooling."""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="np")
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"]
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # Mean pooling
        last_hidden_state = ort_outs[0]
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.repeat(attention_mask[:, :, np.newaxis], last_hidden_state.shape[2], axis=2)
        
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        
        # L2 Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings[0].astype(np.float32)

    def add(self, text: str, source: str = "user"):
        emb = self.embed(text)
        blob = emb.tobytes()
        c = self.conn.cursor()
        c.execute("INSERT INTO memories (text, embedding_blob, timestamp, source) VALUES (?, ?, ?, ?)",
                  (text, blob, time.time(), source))
        self.conn.commit()

    def search(self, query: str, top_k: int = 3):
        query_emb = self.embed(query)
        
        c = self.conn.cursor()
        c.execute("SELECT id, text, embedding_blob, timestamp FROM memories")
        rows = c.fetchall()
        
        if not rows:
            return []
            
        results = []
        for r in rows:
            mem_id, text, blob, ts = r
            mem_emb = np.frombuffer(blob, dtype=np.float32)
            similarity = np.dot(query_emb, mem_emb)
            results.append((similarity, mem_id, text, ts))
            
        # Sort by similarity descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [{"id": r[1], "text": r[2], "similarity": float(r[0]), "timestamp": r[3]} for r in results[:top_k]]
        
    def get_all(self):
        c = self.conn.cursor()
        c.execute("SELECT id, text, timestamp, source FROM memories ORDER BY timestamp DESC")
        return [{"id": r[0], "text": r[1], "timestamp": r[2], "source": r[3]} for r in c.fetchall()]

    def delete(self, mem_id: int):
        c = self.conn.cursor()
        c.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self.conn.commit()

    def clear_all(self):
        c = self.conn.cursor()
        c.execute("DELETE FROM memories")
        self.conn.commit()

    def count_memories(self) -> int:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM memories")
        return c.fetchone()[0]
        
    def format_context(self, memories: list) -> str:
        if not memories:
            return ""
        return "\n".join([f"- {m['text']}" for m in memories])

