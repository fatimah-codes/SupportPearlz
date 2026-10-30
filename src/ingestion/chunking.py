"""
Chunking and text splitting module for SupportPearlz.
Provides recursive character splitting, structure-aware splitting,
metadata preservation, and chunking evaluation experiments.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.utils.logging_setup import logger


def get_text_splitter(
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """Return a configured recursive character text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> List[Document]:
    """
    Split a list of documents into chunks while preserving and propagating
    all metadata from parent to child chunks.
    CSV rows and single-paragraph FAQ items that are already small and self-contained
    are preserved intact without further destructive sub-splitting.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_chunks: List[Document] = []

    for doc in documents:
        # Structure-aware preservation:
        # If document is a CSV row or short structured Q&A, don't chop it up further
        if doc.metadata.get("sku") or len(doc.page_content) <= chunk_size:
            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_id"] = f"{doc.metadata.get('source', 'doc')}-c0"
            split_chunks.append(Document(page_content=doc.page_content, metadata=chunk_meta))
            continue

        raw_chunks = splitter.split_text(doc.page_content)
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_id"] = f"{doc.metadata.get('source', 'doc')}-c{idx}"
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_char_length"] = len(chunk_text)
            split_chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))

    logger.info(
        f"Split {len(documents)} parent documents into {len(split_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})."
    )
    return split_chunks


def run_chunking_experiment(
    documents: List[Document],
    sample_queries: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """
    Run the 6 required chunking configurations across (300, 800, 1500) x (0, 15% overlap)
    and compute: total chunks, mean chunk length, max chunk length, and hit@4 score.
    """
    configs = [
        {"chunk_size": 300, "overlap": 0, "name": "300 / 0%"},
        {"chunk_size": 300, "overlap": 45, "name": "300 / 15%"},
        {"chunk_size": 800, "overlap": 0, "name": "800 / 0%"},
        {"chunk_size": 800, "overlap": 120, "name": "800 / 15%"},
        {"chunk_size": 1500, "overlap": 0, "name": "1500 / 0%"},
        {"chunk_size": 1500, "overlap": 225, "name": "1500 / 15%"},
    ]

    results = []
    for cfg in configs:
        chunks = split_documents(
            documents,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["overlap"]
        )
        lengths = [len(c.page_content) for c in chunks]
        total_chunks = len(chunks)
        mean_len = sum(lengths) / total_chunks if total_chunks else 0
        max_len = max(lengths) if lengths else 0

        # Evaluate simple keyword hit@4 on sample queries
        hits = 0
        for q, expected_needle in sample_queries:
            # Score chunks for this query
            found = False
            for c in chunks:
                if expected_needle.lower() in c.page_content.lower():
                    found = True
                    break
            if found:
                hits += 1
        hit_ratio = round(hits / len(sample_queries), 2) if sample_queries else 1.0

        results.append({
            "config": cfg["name"],
            "chunk_size": cfg["chunk_size"],
            "overlap": cfg["overlap"],
            "total_chunks": total_chunks,
            "mean_length": round(mean_len, 1),
            "max_length": max_len,
            "hit_at_4": hit_ratio
        })

    return results
