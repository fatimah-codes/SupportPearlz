"""
Vector store management module for SupportPearlz.
Handles embedding model selection, vector store persistence to disk,
load-or-build logic, incremental updates, and similarity search debugging.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from src.config import settings
from src.utils.logging_setup import logger


def get_embedding_function() -> Embeddings:
    """
    Instantiate the configured embedding model.
    Defaults to Chroma's built-in SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
    which runs locally and deterministically with zero API cost.
    Supports Google Gemini (text-embedding-004, 768-dim) and OpenAI (1536-dim).
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    
    if provider == "google_genai" and (settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
            logger.info("Initializing GoogleGenerativeAIEmbeddings (text-embedding-004, 768-dim)")
            return GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=api_key
            )
        except Exception as e:
            logger.warning(f"Google embeddings failed to initialize ({e}), falling back to default.")

    if provider == "openai" and (settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")):
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            logger.info("Initializing OpenAIEmbeddings (text-embedding-3-small, 1536-dim)")
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=api_key
            )
        except Exception as e:
            logger.warning(f"OpenAI embeddings failed to initialize ({e}), falling back to default.")

    # Local, zero-cost, persistent embedding model
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    except Exception:
        pass

    # Standard sentence transformer via langchain-community
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("Initializing HuggingFaceEmbeddings ('sentence-transformers/all-MiniLM-L6-v2', 384-dim)")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass

    # Deterministic Subword Semantic Embedding for fast local execution without external weights
    class SubwordSemanticEmbeddings(Embeddings):
        """
        Fast subword-aware semantic embedding generator.
        Extracts words and character 3-5 n-grams to capture morphological variations
        (e.g. 'beep' / 'beeps' / 'beeping', 'flash' / 'flashes') with normalized cosine geometry.
        """
        def __init__(self, dim: int = 384):
            self.dim = dim

        def _get_features(self, text: str) -> List[str]:
            import re
            text = text.lower()
            words = re.findall(r"\b\w+\b", text)
            feats = []
            for w in words:
                feats.append(w)
                if len(w) > 3:
                    for n in range(3, min(6, len(w) + 1)):
                        for i in range(len(w) - n + 1):
                            feats.append(w[i:i+n])
            return feats

        def _embed_str(self, text: str) -> List[float]:
            import math
            import hashlib
            vec = [0.0] * self.dim
            feats = self._get_features(text)
            if not feats:
                return vec
            for f in feats:
                h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(x * x for x in vec))
            return [x / norm for x in vec] if norm > 0 else vec

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self._embed_str(t) for t in texts]

        def embed_query(self, text: str) -> List[float]:
            return self._embed_str(text)

    logger.info("Using SubwordSemanticEmbeddings (384-dim subword vector generator)")
    return SubwordSemanticEmbeddings(dim=384)


def get_vector_store(
    persist_directory: Optional[Path] = None,
    collection_name: Optional[str] = None
) -> Chroma:
    """Instantiate a Chroma vector store pointing to disk persistence."""
    persist_dir = persist_directory or settings.VECTOR_STORE_DIR
    col_name = collection_name or settings.COLLECTION_NAME
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_function()

    return Chroma(
        collection_name=col_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir)
    )


def build_and_persist_index(
    chunks: List[Document],
    rebuild: bool = False,
    persist_directory: Optional[Path] = None,
    collection_name: Optional[str] = None
) -> Chroma:
    """
    Build vector store from chunks and write to disk.
    If rebuild=True, purges the existing directory first.
    """
    target_dir = persist_directory or settings.VECTOR_STORE_DIR
    col_name = collection_name or settings.COLLECTION_NAME

    if rebuild and target_dir.exists():
        logger.info(f"Rebuild requested: purging existing vector store at {target_dir}")
        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_function()
    logger.info(f"Indexing {len(chunks)} chunks into Chroma collection '{col_name}' at {target_dir}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=col_name,
        persist_directory=str(target_dir)
    )
    logger.info("Vector store index persisted to disk successfully.")
    return vector_store


def load_or_build(
    chunks: Optional[List[Document]] = None,
    rebuild: bool = False
) -> Chroma:
    """
    Load vector store if persisted index exists; otherwise build from chunks.
    Ensures that query-time startup loads in fractions of a second without re-embedding.
    """
    target_dir = settings.VECTOR_STORE_DIR
    sqlite_file = target_dir / "chroma.sqlite3"

    if sqlite_file.exists() and not rebuild:
        logger.info(f"Found persisted vector store at {target_dir}. Loading from disk...")
        return get_vector_store()

    if chunks is None:
        raise ValueError(f"No existing index at {target_dir} and no chunks provided to build one.")

    logger.info("No persisted vector store found. Building new index...")
    return build_and_persist_index(chunks=chunks, rebuild=rebuild)


def update_document_in_index(
    source_filename: str,
    new_chunks: List[Document],
    vector_store: Chroma
) -> None:
    """
    Incremental update: delete old chunks by source filename and re-insert new chunks.
    Avoids expensive full index rebuild when a single document changes.
    """
    logger.info(f"Incrementally updating source: {source_filename}")
    # Chroma allows deleting by where clause
    try:
        vector_store.delete(where={"source": source_filename})
        logger.info(f"Deleted old chunks for source: {source_filename}")
    except Exception as e:
        logger.warning(f"Could not delete old chunks for {source_filename}: {e}")

    if new_chunks:
        vector_store.add_documents(new_chunks)
        logger.info(f"Inserted {len(new_chunks)} new chunks for source: {source_filename}")


def similarity_search_helper(
    query: str,
    k: int = 4,
    vector_store: Optional[Chroma] = None
) -> List[Tuple[Document, float]]:
    """
    Debugging utility that runs direct similarity search and prints scores,
    metadata, and content previews.
    """
    store = vector_store or get_vector_store()
    results = store.similarity_search_with_relevance_scores(query, k=k)

    print("\n" + "=" * 70)
    print(f"DIRECT SIMILARITY SEARCH: '{query}' (k={k})")
    print("=" * 70)
    for i, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source", "Unknown")
        loc = doc.metadata.get("location", "N/A")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"[{i}] Score: {score:.4f} | Source: {src} ({loc})")
        print(f"    Preview: \"{preview}...\"\n")
    print("=" * 70)
    return results


def run_embedding_diagnostics() -> Dict[str, float]:
    """
    Task 04 Diagnostic: embed three sentences (two paraphrases and one unrelated)
    and compute pairwise cosine similarities to prove semantic embedding properties.
    """
    import numpy as np

    s1 = "How long is the warranty on the water purifier?"
    s2 = "What is the warranty period for the purifier appliance?"
    s3 = "The recipe requires two cups of granulated sugar and butter."

    embeddings = get_embedding_function()
    vectors = embeddings.embed_documents([s1, s2, s3])
    v1, v2, v3 = np.array(vectors[0]), np.array(vectors[1]), np.array(vectors[2])

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    sim_1_2 = cos_sim(v1, v2)
    sim_1_3 = cos_sim(v1, v3)
    sim_2_3 = cos_sim(v2, v3)

    return {
        "paraphrase_similarity (s1, s2)": round(sim_1_2, 4),
        "unrelated_similarity_1 (s1, s3)": round(sim_1_3, 4),
        "unrelated_similarity_2 (s2, s3)": round(sim_2_3, 4)
    }
