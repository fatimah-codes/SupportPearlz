# SupportPearlz
> **Enterprise Retrieval-Augmented Generation (RAG) Customer Support Agent**  
> **Course:** AAI-407 / PRAC-03 (Practical Assignment &bull; Advanced Level)  
> **Company:** Pearlz Home Systems (Pvt.) Ltd.  
> **Theme:** Facet "Designed in Glass" Hero with Streamlit & Rich Terminal Interfaces

---

## 💎 Project Summary
SupportPearlz is a disciplined, production-grade conversational customer support system built with Python, LangChain, and ChromaDB. It ingests an enterprise knowledge base of appliances manuals, warranty terms, shipping schedules, service agreements, and price catalogs once into a disk-persisted vector store. 

During multi-turn customer dialogues, SupportPearlz resolves pronouns via **Query Condensation**, applies an empirical **Relevance Gate** to block unanswerable queries early, enforces strict **Context Budgeting**, prompts an instruction-tuned LLM (Google Gemini / OpenAI / Local) with low temperature, validates responses against a **Pydantic schema**, and attaches **verifiable, de-duplicated source citations**.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python:** Python 3.10 to 3.13 (Tested on Python 3.13)
- **Git** and terminal environment (PowerShell, Bash, or Zsh)

### 2. Environment Setup
Clone the repository and create an isolated virtual environment:
```bash
git clone <repository_url>
cd supportpearlz

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Configuration & Environment Variables
Copy the documented `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and configure your credentials or tunables:
```ini
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-1.5-flash
TEMPERATURE=0.0
GOOGLE_API_KEY=your_gemini_api_key_here
```
> **Note on Local/Zero-Cost Mode:** You can also run SupportPearlz completely offline with zero API keys by setting `LLM_PROVIDER=mock`.

Verify your configuration with the built-in smoke check:
```bash
python -m src.config
```

---

## 🛠️ Step-by-Step Usage

### 1. Ingest Knowledge Base & Build Vector Index
Run the ingestion pipeline to parse all 12 documents (PDF, DOCX, CSV, MD), generate chunks, and persist the index to `data/vector_store/`:
```bash
python -m src.ingestion.build_index --rebuild
```
*Output:* Generates an Ingestion Report detailing character counts, sections loaded, skipped corrupt test files, and persists 142 chunks to disk in under 3 seconds.

### 2. Launch the Streamlit Web Application (Facet Glass Hero)
To launch the interactive frontend featuring the Facet "Designed in glass" dark smoked-glass hero theme and API key console gateway:
```bash
streamlit run app_streamlit.py
```
- **Gateway Screen:** Features an interactive Three.js 3D WebGL smoked-glass shards hero with cursor tilt, shard lifting, and light reflection.
- **Key Access:** Prompts for your Google Gemini or OpenAI API Key (or click *"Local Demonstration Mode"* for immediate offline access).
- **Console Dashboard:** Opens the interactive customer support console with live chat, query condensation telemetry, citation badges, test suite runner, and parameter tuner!

### 3. Launch the Interactive Terminal CLI
For lightweight terminal usage:
```bash
python -m src.app_cli
```
- Type questions naturally (e.g. *"my aquapearl 500 keeps beeping three times"*).
- Ask follow-ups (e.g. *"is that covered under warranty?"*).
- Built-in commands: `/reset` (clears memory), `/help`, `/exit`.

### 4. Run the 20-Case Automated Evaluation Suite (Task 11)
Execute the benchmark suite evaluating correctness, groundedness, citations, and refusals:
```bash
python -m evaluation.run_eval
```
*Outputs generated:*
- `evaluation/results/eval_results.json`
- `evaluation/results/eval_summary.md`

### 5. Run Unit Tests
Execute the full test suite covering loaders, retrieval, relevance gating, and injection defense:
```bash
python -m unittest discover tests
```

---

## ⚙️ Environment Variables Reference

| Variable Name | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `LLM_PROVIDER` | string | `google_genai` | Provider choice: `google_genai`, `openai`, or `mock` |
| `LLM_MODEL` | string | `gemini-1.5-flash` | Instruction-following chat model identifier |
| `TEMPERATURE` | float | `0.0` | Softmax temperature (set low and explicit for RAG) |
| `EMBEDDING_PROVIDER` | string | `chroma_default` | Embedding backend (`chroma_default`, `google_genai`, `openai`) |
| `EMBEDDING_MODEL` | string | `text-embedding-004` | Embedding model identifier |
| `GOOGLE_API_KEY` | secret | `None` | Google Gemini API key (masked on display) |
| `OPENAI_API_KEY` | secret | `None` | OpenAI API key (masked on display) |
| `VECTOR_STORE_DIR` | path | `data/vector_store` | Target directory for Chroma database persistence |
| `COLLECTION_NAME` | string | `pearlz_kb` | Chroma collection name |
| `CHUNK_SIZE` | int | `800` | Chunk character size (calibrated in Task 03) |
| `CHUNK_OVERLAP` | int | `120` | Chunk character overlap (15% of chunk size) |
| `RETRIEVAL_K` | int | `4` | Number of nearest chunks passed to relevance gate |
| `SCORE_THRESHOLD` | float | `0.08` | Relevance gate cosine threshold (rejects if best score < threshold) |
| `SEARCH_STRATEGY` | string | `similarity` | Retriever policy: `similarity`, `mmr`, or `similarity_score_threshold` |
| `MAX_CONTEXT_CHARS` | int | `6000` | Maximum characters of retrieved context injected into prompt |
| `MEMORY_WINDOW_TURNS`| int | `5` | Dialogue turns retained in conversational sliding window |
| `LOG_LEVEL` | string | `INFO` | Central logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 📂 Project Repository Structure
```
supportpearlz/
├── README.md                      # Complete setup and execution instructions
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Documented template for environment variables
├── .gitignore                     # Excludes .env and data/vector_store/
├── facet_hero.html                # Three.js 0.169.0 WebGL Facet Smoked-Glass Shards Hero
├── app_streamlit.py               # Streamlit Frontend with API Key Gating & Console
├── data/
│   ├── knowledge_base/            # 12 source files (PDF, DOCX, CSV, MD, corrupt test file)
│   └── vector_store/              # Persisted Chroma vector database (git-ignored)
├── src/
│   ├── config.py                  # Pydantic Settings & validation
│   ├── app_cli.py                 # Interactive Rich CLI REPL loop
│   ├── utils/
│   │   └── logging_setup.py       # Central structured logger & turn telemetry
│   ├── ingestion/
│   │   ├── loaders.py             # Extension dispatcher & corrupt file error handling
│   │   ├── chunking.py            # Recursive & structure-aware splitters
│   │   └── build_index.py         # Ingestion CLI builder & diagnostics
│   ├── retrieval/
│   │   ├── vector_store.py        # Disk persistence, load_or_build, & update logic
│   │   └── retriever.py           # Top-k, MMR, & calibrated Relevance Gate
│   └── chains/
│       ├── prompts.py             # Versioned system prompts & injection protection
│       ├── schemas.py             # Pydantic structured output models
│       ├── memory.py              # Bounded history & query condensation
│       └── rag_chain.py           # Composed LCEL RAG execution pipeline
├── evaluation/
│   ├── test_questions.json        # 20 test cases (T-01 to T-20 across 6 categories)
│   ├── run_eval.py                # Automated benchmark execution engine
│   └── results/
│       ├── eval_results.json      # Machine-readable test evaluation data
│       └── eval_summary.md        # Formatted test summary report
├── docs/
│   ├── architecture.svg           # High-resolution vector architecture diagram
│   ├── architecture.png           # Rendered architecture raster diagram
│   ├── technical_report.md        # Comprehensive 7-page engineering report
│   ├── technical_report.pdf       # Formatted PDF engineering report
│   ├── viva_preparation_guide.md  # Model answers to all 20 Viva questions
│   └── planted_gaps_and_seams.md  # Scratch documentation of planted seams & gaps
└── tests/
    ├── test_ingestion.py          # Unit tests for loaders and splitters
    ├── test_retrieval.py          # Unit tests for persistence and relevance gate
    └── test_chain.py              # Unit tests for RAG pipeline and injection defense
```

---

## 🔒 Known Limitations & Production Roadmap
1. **Hybrid Lexical Search (BM25):** While semantic search handles natural language customer queries excellently, alphanumeric part codes (e.g. `E-03` vs `E-08`) benefit from hybrid lexical search (BM25 + Dense vector ensemble).
2. **Dynamic Cross-Encoder Reranking:** Adding a cross-encoder (retrieve 20 candidates, rerank to top 4) would further increase precision on subtle multi-document synthesis queries.
3. **Session State Persistence:** In production, in-memory conversation memory would be backed by an authenticated Redis cluster with per-session TTLs.
