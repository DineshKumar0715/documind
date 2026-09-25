import hashlib
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
from app.core.config import settings
from app.utils.logging import logger

class DeterministicFallbackEmbeddings(Embeddings):
    """
    Fallback deterministic embedding generator used when no Gemini API Key is configured.
    Produces stable 768-dimensional normalized vectors based on hashing & token frequencies.
    """
    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()
        
        for idx, word in enumerate(words):
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            pos = h % self.dimension
            val = ((h >> 8) % 1000) / 1000.0
            vec[pos] += val * (1.0 / (idx + 1) ** 0.5)
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_function() -> Embeddings:
    """
    Instantiate and return the Gemini Embedding model or fallback.
    """
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip() not in ("", "your_gemini_api_key_here"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info(f"Initializing Gemini Embeddings with model: {settings.EMBEDDING_MODEL}")
            return GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GEMINI_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini Embeddings ({e}). Falling back to local deterministic embeddings.")
            return DeterministicFallbackEmbeddings()
    else:
        logger.info("No GEMINI_API_KEY configured. Using local deterministic fallback embeddings.")
        return DeterministicFallbackEmbeddings()
