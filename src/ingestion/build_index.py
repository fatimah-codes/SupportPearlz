"""
CLI entry point for building or rebuilding the SupportPearlz vector index.
Usage:
    python -m src.ingestion.build_index [--rebuild]
"""

import sys
import time
import argparse
from pathlib import Path
from src.config import settings
from src.utils.logging_setup import logger
from src.ingestion.loaders import load_knowledge_base
from src.ingestion.chunking import split_documents
from src.retrieval.vector_store import build_and_persist_index, run_embedding_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportPearlz Vector Index Builder")
    parser.add_argument("--rebuild", action="store_true", help="Force clean rebuild of vector database")
    parser.add_argument("--kb-dir", type=str, default=str(settings.KNOWLEDGE_BASE_DIR), help="Path to knowledge base directory")
    args = parser.parse_args()

    kb_path = Path(args.kb_dir)
    print(f"\n[SupportPearlz] Starting Ingestion Pipeline...")
    print(f"[SupportPearlz] Source Directory: {kb_path.resolve()}")
    print(f"[SupportPearlz] Persistence Target: {settings.VECTOR_STORE_DIR.resolve()}")
    print(f"[SupportPearlz] Force Rebuild: {args.rebuild}\n")

    start_time = time.time()

    # Step 1: Load documents
    docs, report = load_knowledge_base(kb_path)
    print(report.print_summary())

    if not docs:
        print("[ERROR] No valid documents loaded. Ingestion aborted.")
        sys.exit(1)

    # Step 2: Split documents
    print(f"\n[SupportPearlz] Splitting documents into chunks (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
    chunks = split_documents(docs, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    print(f"[SupportPearlz] Produced {len(chunks)} chunks with preserved metadata contracts.")

    # Step 3: Run diagnostic
    print("\n[SupportPearlz] Running Embedding Model Diagnostics...")
    diag = run_embedding_diagnostics()
    for k, v in diag.items():
        print(f"  * {k}: {v}")

    # Step 4: Build and persist vector store
    print(f"\n[SupportPearlz] Generating embeddings and persisting to Chroma database...")
    store = build_and_persist_index(chunks=chunks, rebuild=args.rebuild)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Vector index successfully built and persisted in {elapsed:.2f} seconds.")
    print(f"[SupportPearlz] Ready for instant loading at query time without re-embedding.\n")


if __name__ == "__main__":
    main()
