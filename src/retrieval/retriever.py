"""
Retriever construction and relevance gate module for SupportPearlz.
Supports similarity, MMR, metadata-filtered retrieval, and relevance scoring thresholds.
"""

from typing import List, Tuple, Dict, Any, Optional, Literal
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import settings
from src.retrieval.vector_store import get_vector_store
from src.utils.logging_setup import logger


class RelevanceGateResult:
    """Encapsulates the decision of the relevance gate."""
    def __init__(
        self,
        passed: bool,
        chunks: List[Document],
        scores: List[float],
        best_score: float,
        reason: str
    ):
        self.passed = passed
        self.chunks = chunks
        self.scores = scores
        self.best_score = best_score
        self.reason = reason


def retrieve_with_scores(
    query: str,
    k: int = settings.RETRIEVAL_K,
    search_strategy: str = settings.SEARCH_STRATEGY,
    filter_dict: Optional[Dict[str, Any]] = None,
    vector_store: Optional[Chroma] = None
) -> List[Tuple[Document, float]]:
    """
    Execute retrieval returning (document, similarity_score) tuples.
    Supports strategy switching: 'similarity', 'mmr', or 'similarity_score_threshold'.
    """
    store = vector_store or get_vector_store()
    
    # Optional metadata filtering
    kwargs: Dict[str, Any] = {"k": k}
    if filter_dict:
        kwargs["filter"] = filter_dict

    if search_strategy == "mmr":
        # MMR selects for relevance and diversity
        try:
            mmr_docs = store.max_marginal_relevance_search(query, k=k, fetch_k=k*3, filter=filter_dict)
            # Re-score MMR selected docs
            scored = store.similarity_search_with_relevance_scores(query, k=k*2, filter=filter_dict)
            score_map = {d.page_content: s for d, s in scored}
            return [(d, score_map.get(d.page_content, 0.5)) for d in mmr_docs]
        except Exception as e:
            logger.warning(f"MMR search fallback to similarity due to: {e}")

    # Standard similarity search with scores
    try:
        results = store.similarity_search_with_relevance_scores(query, **kwargs)
        return results
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        # Plain similarity fallback
        docs = store.similarity_search(query, **kwargs)
        return [(d, 0.5) for d in docs]


def evaluate_relevance_gate(
    query: str,
    k: int = settings.RETRIEVAL_K,
    threshold: float = settings.SCORE_THRESHOLD,
    search_strategy: str = settings.SEARCH_STRATEGY,
    filter_dict: Optional[Dict[str, Any]] = None,
    vector_store: Optional[Chroma] = None
) -> RelevanceGateResult:
    """
    The Relevance Gate: evaluates retrieved chunks against the calibrated score threshold.
    If the result set is empty or the highest similarity score is strictly below the threshold,
    the gate closes and terminates early with a deterministic refusal signal.
    """
    results = retrieve_with_scores(
        query=query,
        k=k,
        search_strategy=search_strategy,
        filter_dict=filter_dict,
        vector_store=vector_store
    )

    if not results:
        logger.warning(f"Relevance Gate: Empty retrieval set for query '{query}'.")
        return RelevanceGateResult(
            passed=False,
            chunks=[],
            scores=[],
            best_score=0.0,
            reason="Empty retrieval result set."
        )

    chunks = [doc for doc, _ in results]
    scores = [float(score) for _, score in results]
    best_score = max(scores) if scores else 0.0

    if best_score < threshold:
        logger.info(
            f"Relevance Gate: REJECTED query '{query}'. "
            f"Best score {best_score:.3f} < threshold {threshold:.3f}."
        )
        return RelevanceGateResult(
            passed=False,
            chunks=chunks,
            scores=scores,
            best_score=best_score,
            reason=f"Best score {best_score:.3f} is below calibrated threshold {threshold:.3f}."
        )

    logger.info(
        f"Relevance Gate: PASSED query '{query}'. "
        f"Best score {best_score:.3f} >= threshold {threshold:.3f}."
    )
    return RelevanceGateResult(
        passed=True,
        chunks=chunks,
        scores=scores,
        best_score=best_score,
        reason=f"Retrieved {len(chunks)} relevant chunks with top score {best_score:.3f}."
    )


def compare_retrieval_strategies(
    test_queries: List[Tuple[str, str]],
    vector_store: Optional[Chroma] = None
) -> Dict[str, Any]:
    """
    Compare retrieval performance across:
    1. Similarity k=3
    2. Similarity k=8
    3. MMR (k=4)
    Computes hit@k on ground truth needles.
    """
    strategies = [
        {"name": "Similarity (k=3)", "strategy": "similarity", "k": 3},
        {"name": "Similarity (k=8)", "strategy": "similarity", "k": 8},
        {"name": "MMR (k=4, fetch_k=12)", "strategy": "mmr", "k": 4},
    ]

    report = {}
    for st in strategies:
        hits = 0
        total_score = 0.0
        for q, expected in test_queries:
            res = retrieve_with_scores(
                query=q,
                k=st["k"],
                search_strategy=st["strategy"],
                vector_store=vector_store
            )
            found = False
            for doc, s in res:
                total_score += s
                if expected.lower() in doc.page_content.lower():
                    found = True
                    break
            if found:
                hits += 1

        hit_rate = round(hits / len(test_queries), 3) if test_queries else 1.0
        report[st["name"]] = {
            "hit_at_k": hit_rate,
            "evaluated_queries": len(test_queries),
            "hits": hits
        }

    return report
