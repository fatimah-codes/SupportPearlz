"""
Document loader layer for SupportPearlz.
Dispatches heterogeneous file formats (.pdf, .docx, .md, .txt, .csv)
normalises text, preserves metadata contracts, and handles file errors gracefully.
"""

import os
import re
import csv
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from src.utils.logging_setup import logger


def compute_file_hash(content: str) -> str:
    """Compute MD5 hash to detect duplicate content."""
    return hashlib.md5(content.strip().encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    """Normalise whitespace and strip extraneous empty lines."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_doc_type(file_path: Path) -> str:
    """Infer document type classification from filename."""
    name = file_path.name.lower()
    if "manual" in name:
        return "manual"
    elif "pricing" in name or "price" in name:
        return "pricing"
    elif "faq" in name:
        return "faq"
    elif "policy" in name or "warranty" in name or "refund" in name or "shipping" in name or "privacy" in name:
        return "policy"
    elif "guide" in name or "installation" in name or "troubleshooting" in name:
        return "guide"
    elif "agreement" in name or "contract" in name:
        return "agreement"
    elif "handbook" in name:
        return "handbook"
    elif "changelog" in name or "release" in name:
        return "changelog"
    return "general"


def infer_product_line(file_path: Path) -> str:
    """Infer applicable product line from document filename."""
    name = file_path.name.lower()
    if "aquapearl500" in name or "aquapearl 500" in name:
        return "AquaPearl 500 Pro"
    elif "aquapearl300" in name:
        return "AquaPearl 300"
    elif "aeropearl" in name:
        return "AeroPearl Mini"
    elif "aquapearl" in name:
        return "AquaPearl Series"
    return "All Products"


def load_pdf(file_path: Path) -> List[Document]:
    """Load PDF using pypdf, extracting page-level metadata."""
    from pypdf import PdfReader
    docs: List[Document] = []
    reader = PdfReader(str(file_path))
    num_pages = len(reader.pages)
    
    doc_type = infer_doc_type(file_path)
    product_line = infer_product_line(file_path)
    
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        clean_text = normalize_whitespace(text)
        if not clean_text:
            continue
        
        metadata: Dict[str, Any] = {
            "source": file_path.name,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "product_line": product_line,
            "version": "2025.1",
            "page": idx + 1,
            "total_pages": num_pages,
            "location": f"Page {idx + 1}"
        }
        docs.append(Document(page_content=clean_text, metadata=metadata))
    return docs


def load_docx(file_path: Path) -> List[Document]:
    """Load DOCX file using python-docx, preserving section/heading structure."""
    from docx import Document as DocxDocument
    doc = DocxDocument(str(file_path))
    docs: List[Document] = []
    
    current_heading = "General"
    section_texts: List[str] = []
    doc_type = infer_doc_type(file_path)
    product_line = infer_product_line(file_path)

    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        if p.style.name.startswith("Heading"):
            if section_texts:
                full_section = normalize_whitespace("\n".join(section_texts))
                if full_section:
                    metadata: Dict[str, Any] = {
                        "source": file_path.name,
                        "file_path": str(file_path),
                        "doc_type": doc_type,
                        "product_line": product_line,
                        "version": "2025.1",
                        "section": current_heading,
                        "location": f"Section: {current_heading}"
                    }
                    docs.append(Document(page_content=full_section, metadata=metadata))
                section_texts = []
            current_heading = txt
        else:
            section_texts.append(txt)

    if section_texts:
        full_section = normalize_whitespace("\n".join(section_texts))
        if full_section:
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "doc_type": doc_type,
                "product_line": product_line,
                "version": "2025.1",
                "section": current_heading,
                "location": f"Section: {current_heading}"
            }
            docs.append(Document(page_content=full_section, metadata=metadata))

    # Also parse tables if any
    for t_idx, table in enumerate(doc.tables):
        rows_data = []
        for row in table.rows:
            row_vals = [cell.text.strip() for cell in row.cells]
            rows_data.append(" | ".join(row_vals))
        table_str = "\n".join(rows_data)
        if table_str:
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "doc_type": doc_type,
                "product_line": product_line,
                "version": "2025.1",
                "section": f"Table {t_idx + 1}",
                "location": f"Table {t_idx + 1}"
            }
            docs.append(Document(page_content=table_str, metadata=metadata))
            
    return docs


def load_markdown_or_text(file_path: Path) -> List[Document]:
    """Load Markdown or text files, slicing into heading-aware sections."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    doc_type = infer_doc_type(file_path)
    product_line = infer_product_line(file_path)
    docs: List[Document] = []

    # Split markdown by top/second-level headers
    sections = re.split(r"\n(?=#{1,3}\s+)", content)
    for sec in sections:
        clean_sec = normalize_whitespace(sec)
        if not clean_sec:
            continue
        
        # Extract header line
        first_line = clean_sec.split("\n")[0].strip("# ").strip()
        location_label = first_line if first_line else "General"

        metadata: Dict[str, Any] = {
            "source": file_path.name,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "product_line": product_line,
            "version": "2025.1",
            "section": location_label,
            "location": f"Section: {location_label}"
        }
        docs.append(Document(page_content=clean_sec, metadata=metadata))

    return docs


def load_csv(file_path: Path) -> List[Document]:
    """
    Load CSV tabular data row-wise, formatting each row into a self-describing sentence.
    This guarantees SKU, product name, list price, and promo price stay bound together,
    preventing the classic 'tabular data ingested as prose' failure mode.
    """
    docs: List[Document] = []
    doc_type = infer_doc_type(file_path)
    product_line = infer_product_line(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sku = row.get("SKU", f"Row-{idx+1}")
            pname = row.get("Product Name", "Product")
            category = row.get("Category", "General")
            desc = row.get("Description", "")
            list_price = row.get("List Price (PKR)", "N/A")
            promo_price = row.get("Promotional Price (PKR)", "N/A")
            status = row.get("Availability Status", "In Stock")
            warranty = row.get("Warranty Period (Months)", "0")

            row_text = (
                f"Pricing and Product Entry for SKU {sku}: {pname} (Category: {category}). "
                f"Description: {desc} "
                f"Standard List Price: PKR {list_price}. "
                f"Current Promotional Price: PKR {promo_price}. "
                f"Availability Status: {status}. "
                f"Standard Warranty Coverage: {warranty} Months."
            )

            metadata: Dict[str, Any] = {
                "source": file_path.name,
                "file_path": str(file_path),
                "doc_type": doc_type,
                "product_line": product_line,
                "version": "2025.1",
                "sku": sku,
                "location": f"SKU {sku}"
            }
            docs.append(Document(page_content=row_text, metadata=metadata))

    return docs


class IngestionReport:
    """Summary record of document ingestion run."""
    def __init__(self):
        self.files_found: int = 0
        self.files_loaded: int = 0
        self.files_skipped: int = 0
        self.skipped_details: List[Tuple[str, str]] = []
        self.loaded_details: List[Dict[str, Any]] = []

    def print_summary(self) -> str:
        lines = [
            "=" * 70,
            "DOCUMENT INGESTION REPORT",
            "=" * 70,
            f"Files Found:   {self.files_found}",
            f"Files Loaded:  {self.files_loaded}",
            f"Files Skipped: {self.files_skipped}",
            "-" * 70,
            "LOADED DOCUMENTS:"
        ]
        for d in self.loaded_details:
            lines.append(
                f"  * {d['name']:<32} | Type: {d['doc_type']:<10} | "
                f"Sections/Pages: {d['units']:<3} | Chars: {d['chars']}"
            )
        if self.skipped_details:
            lines.append("-" * 70)
            lines.append("SKIPPED / CORRUPT FILES:")
            for fname, reason in self.skipped_details:
                lines.append(f"  [SKIPPED] {fname}: {reason}")
        lines.append("=" * 70)
        return "\n".join(lines)


def load_knowledge_base(kb_directory: Path) -> Tuple[List[Document], IngestionReport]:
    """
    Recursively scan knowledge base directory, dispatch appropriate loaders,
    handle corrupt files gracefully, prevent duplicate chunks, and emit report.
    """
    report = IngestionReport()
    all_documents: List[Document] = []
    seen_hashes = set()

    if not kb_directory.exists():
        logger.error(f"Knowledge base directory does not exist: {kb_directory}")
        return all_documents, report

    file_paths = sorted(list(kb_directory.rglob("*")))
    valid_files = [p for p in file_paths if p.is_file()]
    report.files_found = len(valid_files)

    for p in valid_files:
        ext = p.suffix.lower()
        file_docs: List[Document] = []
        try:
            if ext == ".pdf":
                file_docs = load_pdf(p)
            elif ext == ".docx":
                file_docs = load_docx(p)
            elif ext in [".md", ".markdown", ".txt"]:
                file_docs = load_markdown_or_text(p)
            elif ext == ".csv":
                file_docs = load_csv(p)
            else:
                report.files_skipped += 1
                report.skipped_details.append((p.name, f"Unsupported file extension: {ext}"))
                logger.warning(f"Skipping unsupported file format: {p.name}")
                continue

            # Deduplicate and validate
            clean_docs = []
            for d in file_docs:
                h = compute_file_hash(d.page_content)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                clean_docs.append(d)

            if clean_docs:
                report.files_loaded += 1
                total_chars = sum(len(d.page_content) for d in clean_docs)
                report.loaded_details.append({
                    "name": p.name,
                    "doc_type": clean_docs[0].metadata.get("doc_type", "unknown"),
                    "units": len(clean_docs),
                    "chars": total_chars,
                    "metadata_keys": list(clean_docs[0].metadata.keys())
                })
                all_documents.extend(clean_docs)
                logger.info(f"Loaded {p.name}: {len(clean_docs)} document units, {total_chars} chars.")
            else:
                report.files_skipped += 1
                report.skipped_details.append((p.name, "File contained empty or duplicate text."))

        except Exception as e:
            report.files_skipped += 1
            report.skipped_details.append((p.name, str(e)))
            logger.warning(f"Error loading file '{p.name}', skipping: {e}")

    return all_documents, report
