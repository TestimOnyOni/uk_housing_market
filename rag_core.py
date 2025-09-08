# rag_core.py
from __future__ import annotations
import os
import re
import json
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field


# =========================
# Embeddings
# =========================
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

class Embedder:
    def __init__(self, use_openai: bool = USE_OPENAI_EMBEDDINGS):
        self.use_openai = use_openai
        if use_openai:
            from openai import OpenAI
            self.client = OpenAI()
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(LOCAL_EMBED_MODEL)

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.use_openai:
            # Batch to respect token limits
            from openai import OpenAI
            chunks = [texts[i:i+256] for i in range(0, len(texts), 256)]
            vecs: List[np.ndarray] = []
            for ch in chunks:
                resp = self.client.embeddings.create(model=OPENAI_EMBED_MODEL, input=ch)
                arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
                vecs.append(arr)
            out = np.vstack(vecs) if vecs else np.zeros((0, 1536), dtype=np.float32)
            return out