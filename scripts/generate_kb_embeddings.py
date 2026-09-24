"""
Offline generation utility for WeMentors Knowledge Base embeddings.

Precomputes 3072-dimensional vector embeddings for all entries in
knowledge/wementors_kb.json using Google's gemini-embedding-001 model.

The output static artifact includes cryptographic content hashing
(SHA-256) and schema metadata so that the backend can verify integrity
at startup with zero external network overhead.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge" / "wementors_kb.json"
OUTPUT_FILE = PROJECT_ROOT / "knowledge" / "kb_embeddings.json"

EMBEDDING_MODEL = "gemini-embedding-001"
EXPECTED_DIMENSION = 3072


def compute_kb_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of the knowledge base file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_kb_entries(file_path: Path) -> List[Dict]:
    """Load entries from the knowledge base JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def construct_searchable_text(entry: Dict) -> str:
    """Build searchable representation of KB entry for embedding."""
    parts = [
        entry.get("question", ""),
        " ".join(entry.get("phrasings", [])),
        " ".join(entry.get("keywords", [])),
        entry.get("answer", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def embed_batch_rest(texts: List[str], api_key: str) -> List[List[float]]:
    """Fetch embeddings in a single batch request via Google Generative AI REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:batchEmbedContents?key={api_key}"
    requests = [
        {
            "model": f"models/{EMBEDDING_MODEL}",
            "content": {"parts": [{"text": text[:2048]}]},
        }
        for text in texts
    ]

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json={"requests": requests})
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        embeddings = [item["values"] for item in data.get("embeddings", [])]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Mismatch: expected {len(texts)} embeddings, got {len(embeddings)}")
        return embeddings


def generate_kb_embeddings(
    kb_path: Path = KNOWLEDGE_FILE,
    out_path: Path = OUTPUT_FILE,
    from_cache: Path | None = None,
    api_key: str | None = None,
) -> None:
    """Generate or package KB embeddings artifact with full metadata."""
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge file not found at {kb_path}")

    kb_hash = compute_kb_hash(kb_path)
    entries = load_kb_entries(kb_path)
    print(f"Loaded {len(entries)} KB entries from {kb_path.name}")
    print(f"KB SHA-256: {kb_hash}")

    embeddings_dict: Dict[str, List[float]] = {}

    if from_cache and from_cache.exists():
        print(f"Importing verified precomputed embeddings from {from_cache}")
        with open(from_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        raw_embs = cached.get("embeddings", cached)
        for e in entries:
            eid = e["id"]
            if eid in raw_embs:
                embeddings_dict[eid] = raw_embs[eid]
            else:
                raise ValueError(f"Missing embedding for entry '{eid}' in cache {from_cache}")
    else:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be provided to call the API.")

        print(f"Requesting embeddings from {EMBEDDING_MODEL} via Google Generative AI API...")
        texts = [construct_searchable_text(e) for e in entries]
        vectors = embed_batch_rest(texts, key)
        for e, vec in zip(entries, vectors):
            embeddings_dict[e["id"]] = vec

    # Validate all vectors
    for eid, vec in embeddings_dict.items():
        if len(vec) != EXPECTED_DIMENSION:
            raise ValueError(f"Entry {eid} has dimension {len(vec)}, expected {EXPECTED_DIMENSION}")

    payload = {
        "metadata": {
            "embedding_model": EMBEDDING_MODEL,
            "vector_dimension": EXPECTED_DIMENSION,
            "kb_content_hash": kb_hash,
            "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_entries": len(embeddings_dict),
        },
        "embeddings": embeddings_dict,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully generated static embedding artifact: {out_path} ({len(embeddings_dict)} vectors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate precomputed KB embeddings.")
    parser.add_argument("--cache", type=str, help="Path to cached embeddings to package.")
    parser.add_argument("--api-key", type=str, help="Gemini API Key.")
    args = parser.parse_args()

    cache_path = Path(args.cache) if args.cache else (PROJECT_ROOT / "evaluation" / "cache" / "kb_embeddings.json")
    if not cache_path.exists():
        cache_path = None

    generate_kb_embeddings(from_cache=cache_path, api_key=args.api_key)
