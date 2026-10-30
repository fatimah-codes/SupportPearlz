"""
Unit tests for vector store persistence and retrieval relevance gating.
"""

import unittest
from src.config import settings
from src.retrieval.vector_store import load_or_build, similarity_search_helper
from src.retrieval.retriever import evaluate_relevance_gate, retrieve_with_scores


class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store = load_or_build()

    def test_persisted_store_loaded(self):
        self.assertIsNotNone(self.store)

    def test_semantic_search_retrieves_faq(self):
        query = "my purifier beeps three times and light flashes red"
        results = retrieve_with_scores(query, k=4, vector_store=self.store)
        self.assertGreater(len(results), 0)
        found = any(
            any(term in doc.page_content.lower() for term in ["beep", "e-03", "filter", "alert", "symptom", "purifier", "warranty"])
            for doc, _ in results
        )
        self.assertTrue(found)

    def test_relevance_gate_rejection(self):
        # Out-of-domain random query
        query = "The baking recipe requires granulated white sugar and heavy cream"
        gate = evaluate_relevance_gate(query, threshold=settings.SCORE_THRESHOLD, vector_store=self.store)
        # Should be rejected because score is below threshold
        self.assertFalse(gate.passed)

    def test_relevance_gate_acceptance(self):
        query = "How long is the warranty on the AquaPearl 500 Pro?"
        gate = evaluate_relevance_gate(query, threshold=settings.SCORE_THRESHOLD, vector_store=self.store)
        self.assertTrue(gate.passed)
        self.assertGreaterEqual(gate.best_score, settings.SCORE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
