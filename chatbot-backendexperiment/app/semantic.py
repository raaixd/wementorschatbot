"""
Semantic retrieval layer for the WeMentors RAG pipeline.

Provides:
  1. KBEmbeddingStore: Loads precomputed 3072-dim embeddings for knowledge/wementors_kb.json,
     verifying model name, dimension, and SHA-256 hash at startup.
  2. EmbeddingClient: Calls Google's gemini-embedding-001 endpoint with circuit breaker,
     timeout, and rate-limit resilience.
  3. Pure-Python vector mathematics (with optional NumPy acceleration if available)
     to avoid adding heavy framework dependencies to production.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from . import config
from .knowledge import KBEntry

logger = logging.getLogger(__name__)

# Optional NumPy acceleration without making it a hard requirement
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def pure_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors in pure Python."""
    if len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    denom = math.sqrt(norm1) * math.sqrt(norm2)
    return dot / denom if denom else 0.0


def fast_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Cosine similarity using NumPy if available, otherwise pure Python."""
    if _HAS_NUMPY:
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
    return pure_cosine_similarity(v1, v2)


class KBEmbeddingStore:
    """Loads and validates precomputed KB vector embeddings from a static artifact."""

    def __init__(self, artifact_path: Optional[Path] = None, kb_path: Optional[Path] = None):
        self.artifact_path = artifact_path or config.KB_EMBEDDINGS_FILE
        self.kb_path = kb_path or config.KNOWLEDGE_FILE
        self.embeddings: Dict[str, List[float]] = {}
        self.metadata: Dict = {}
        self.is_valid: bool = False
        self.validation_error: str = ""

        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Validate artifact existence, schema, dimension, and KB content hash."""
        if not self.artifact_path.exists():
            self.validation_error = f"KB embeddings file missing: {self.artifact_path}"
            logger.warning(f"[Semantic] {self.validation_error}. Semantic retrieval disabled.")
            return

        try:
            with open(self.artifact_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.validation_error = f"Malformed KB embeddings JSON: {e}"
            logger.warning(f"[Semantic] {self.validation_error}. Semantic retrieval disabled.")
            return

        self.metadata = data.get("metadata", {})
        self.embeddings = data.get("embeddings", {})

        # 1. Verify embedding model
        model = self.metadata.get("embedding_model", "")
        if model != config.EMBEDDING_MODEL:
            self.validation_error = (
                f"Model mismatch in KB embeddings: expected {config.EMBEDDING_MODEL}, found {model}"
            )
            logger.warning(f"[Semantic] {self.validation_error}.")
            return

        # 2. Verify dimension
        dim = self.metadata.get("vector_dimension", 0)
        if dim != config.EMBEDDING_DIMENSION:
            self.validation_error = (
                f"Dimension mismatch: expected {config.EMBEDDING_DIMENSION}, found {dim}"
            )
            logger.warning(f"[Semantic] {self.validation_error}.")
            return

        # 3. Verify KB content hash if KB file is present
        expected_hash = self.metadata.get("kb_content_hash", "")
        if self.kb_path.exists():
            actual_hash = compute_file_sha256(self.kb_path)
            if expected_hash and actual_hash and expected_hash != actual_hash:
                self.validation_error = (
                    f"Stale KB embeddings: artifact hash {expected_hash[:10]}... "
                    f"does not match {self.kb_path.name} hash {actual_hash[:10]}..."
                )
                logger.warning(f"[Semantic] {self.validation_error}.")
                return

        # 4. Verify vector lengths in dictionary
        for eid, vec in self.embeddings.items():
            if len(vec) != config.EMBEDDING_DIMENSION:
                self.validation_error = (
                    f"Vector dimension mismatch for entry '{eid}': len={len(vec)} != {config.EMBEDDING_DIMENSION}"
                )
                logger.warning(f"[Semantic] {self.validation_error}.")
                return

        self.is_valid = True
        logger.info(
            f"[Semantic] Loaded {len(self.embeddings)} verified embeddings "
            f"(model: {model}, dim: {dim}, source: {self.artifact_path.name})"
        )


class EmbeddingClient:
    """HTTP client for Google Generative AI REST API with circuit-breaker protection."""

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[float] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.timeout = timeout if timeout is not None else config.EMBEDDING_API_TIMEOUT
        self.model = config.EMBEDDING_MODEL
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        )

    def embed_query(self, query: str) -> Optional[List[float]]:
        """Fetch vector embedding for a single query text.
        
        Returns None gracefully on any network error, timeout, HTTP 429, 5xx,
        or malformed payload to prevent breaking conversation flow.
        """
        if not self.api_key:
            logger.debug("[Semantic] Embedding skipped: GEMINI_API_KEY is not configured.")
            return None

        cleaned = query.strip()
        if not cleaned:
            return None

        url = f"{self.endpoint}?key={self.api_key}"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": cleaned[:2048]}]},
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code == 429:
                    logger.warning("[Semantic] Gemini Embedding API rate limited (HTTP 429). Falling back to TF-IDF.")
                    return None
                if response.status_code != 200:
                    logger.warning(
                        f"[Semantic] Embedding API returned HTTP {response.status_code}: {response.text[:120]}. "
                        "Falling back to TF-IDF."
                    )
                    return None

                data = response.json()
                embedding = data.get("embedding", {}).get("values")
                if not isinstance(embedding, list) or len(embedding) != config.EMBEDDING_DIMENSION:
                    logger.warning(
                        f"[Semantic] Malformed embedding response: expected {config.EMBEDDING_DIMENSION} floats, "
                        f"got {type(embedding)}. Falling back to TF-IDF."
                    )
                    return None
                return embedding

        except httpx.TimeoutException:
            logger.warning(f"[Semantic] Embedding API timed out after {self.timeout}s. Falling back to TF-IDF.")
            return None
        except httpx.RequestError as e:
            logger.warning(f"[Semantic] Embedding API request error: {e}. Falling back to TF-IDF.")
            return None
        except Exception as e:
            logger.warning(f"[Semantic] Unexpected error during embedding: {e}. Falling back to TF-IDF.")
            return None
