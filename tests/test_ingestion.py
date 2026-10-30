"""
Unit tests for document ingestion and text splitting in SupportPearlz.
"""

import unittest
from pathlib import Path
from src.config import settings
from src.ingestion.loaders import load_knowledge_base, normalize_whitespace, infer_doc_type
from src.ingestion.chunking import split_documents, get_text_splitter


class TestIngestion(unittest.TestCase):

    def setUp(self):
        self.kb_dir = settings.KNOWLEDGE_BASE_DIR

    def test_infer_doc_type(self):
        self.assertEqual(infer_doc_type(Path("warranty_policy.md")), "policy")
        self.assertEqual(infer_doc_type(Path("product_manual_aquapearl500.pdf")), "manual")
        self.assertEqual(infer_doc_type(Path("pricing_guide.csv")), "pricing")
        self.assertEqual(infer_doc_type(Path("faq.md")), "faq")
        self.assertEqual(infer_doc_type(Path("service_agreement.docx")), "agreement")

    def test_whitespace_normalisation(self):
        raw = "Hello   world!\r\n\r\n\r\n\r\nThis   is  a\ttest."
        normalized = normalize_whitespace(raw)
        self.assertEqual(normalized, "Hello world!\n\nThis is a test.")

    def test_load_knowledge_base_and_skip_corrupt(self):
        docs, report = load_knowledge_base(self.kb_dir)
        self.assertGreaterEqual(report.files_loaded, 10)
        self.assertGreater(len(docs), 50)
        # Verify corrupt file was reported and skipped
        skipped_names = [f[0] for f in report.skipped_details]
        self.assertIn("corrupt_sample.pdf", skipped_names)

    def test_metadata_contract_propagation(self):
        docs, _ = load_knowledge_base(self.kb_dir)
        chunks = split_documents(docs, chunk_size=800, chunk_overlap=120)
        self.assertGreater(len(chunks), len(docs))
        for c in chunks[:20]:
            self.assertIn("source", c.metadata)
            self.assertIn("doc_type", c.metadata)
            self.assertIn("location", c.metadata)
            self.assertTrue(len(c.page_content) > 0)


if __name__ == "__main__":
    unittest.main()
