# ProductLens: Aspect-Level Product Intelligence from Large-Scale Customer Reviews

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-DeBERTa--v3-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/microsoft/deberta-v3-base)
[![Embeddings](https://img.shields.io/badge/BGE--Embeddings-bge--small-blue?style=flat)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Clustering](https://img.shields.io/badge/Clustering-HDBSCAN-success?style=flat)](https://hdbscan.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-122%20Passed-brightgreen?style=flat&logo=pytest&logoColor=white)](tests/)
[![Paper](https://img.shields.io/badge/Report-Academic%20Paper-orange?style=flat&logo=arxiv&logoColor=white)](report.md)

**ProductLens** is an end-to-end, research-grade Aspect-Based Sentiment Analysis (ABSA) and product intelligence system built on the **Amazon Reviews 2023** corpus. 

Rather than collapsing consumer opinions into monolithic 1-to-5 star ratings or generating ungrounded, hallucination-prone summaries, ProductLens decomposes unstructured customer feedback into fine-grained, verifiable aspect-sentiment tuples anchored to exact document offsets. Every component score, metric, and opinion is mathematically and cryptographically auditable back to its source review and sentence.

---

## 📑 Research Paper & Technical Report

The full mathematical formulation, algorithm pseudo-code, theoretical proofs, and empirical benchmarks are documented in the accompanying research paper:

👉 **[`report.md`](report.md)** — *ProductLens: Aspect-Level Product Intelligence from Large-Scale Customer Reviews (Technical Report & Research Architecture)*

---

## 🌟 Key Capabilities & Architectural Principles

* **100% Traceability & Zero Hallucination:** Every extracted aspect maintains a triple of coordinates: `(review_id, sentence_id, [start_char, end_char])`. All displayed insights trace to exact spans in source reviews.
* **Transformer BIO Sequence Tagging:** Token classification powered by `microsoft/deberta-v3-base` (with `roberta-base` as alternative) featuring subword-to-character span reconstruction that cleanly isolates multiple aspects within a single sentence.
* **Domain-Agnostic Semantic Normalization:** Dense $384$-dimensional embeddings (`BAAI/bge-small-en-v1.5`) clustered via category-aware HDBSCAN. Outlier mentions are preserved rather than dropped. No hard-coded laptop-only ontologies.
* **Active Merge Safety Guards:** Topological guards that prevent false merges between base entities and accessories (e.g., *screen* vs. *screen protector*, *phone* vs. *phone case*) and prohibit merges solely on generic modifiers (*sound quality* vs. *build quality*).
* **Six-Domain Aspect Typing:** Mentions are classified into `product`, `service`, `delivery`, `seller`, `packaging`, or `unknown`. Logistics complaints (e.g., carrier transit delays or unboxing damage) never corrupt physical product quality ratings.
* **Leakage-Free Partitioning:** Product-aware splitting guarantees that all reviews for any given ASIN appear exclusively within train, validation, or test sets.
* **Consumer Hardware Optimization:** Engineered to run on consumer GPUs (NVIDIA RTX 4050 6 GB / RTX 5050 8 GB) with mixed-precision FP16, automatic CUDA OOM recovery, persistent embedding caching, and pure CPU fallback.

---

## 🏗️ System Architecture

```
+───────────────────────────────────────────────────────────────────────────────────────+
|                               PRODUCTLENS NLP PIPELINE                                |
+───────────────────────────────────────────────────────────────────────────────────────+
|                                                                                       |
|   Amazon Reviews 2023 / Verified Ingestion Stream                                     |
|         │                                                                             |
|         ▼                                                                             |
|   ┌───────────────────────────────────────────────────────────────────────────────┐   |
|   │ STAGE 1: DATA FOUNDATION                                                      │   |
|   │ • Unicode NFC Normalization, HTML Entity Decoding & URL Stripping             │   |
|   │ • Exact SHA-256 & Near-Duplicate (MinHash Jaccard τ=0.90) Deduplication       │   |
|   │ • Product-Aware Stratified Splitting (Zero Cross-Split ASIN Leakage)          │   |
|   │ • Two-Tier Sentence Segmentation (Immutable Titles + Contextual Abbrs)        │   |
|   └──────────────────────────────────────┬────────────────────────────────────────┘   |
|                                          │                                            |
|                                          ▼                                            |
|   ┌───────────────────────────────────────────────────────────────────────────────┐   |
|   │ STAGE 2: ASPECT INTELLIGENCE                                                  │   |
|   │ • BIO Sequence Labeling: DeBERTa-v3-base / RoBERTa-base Token Head            │   |
|   │ • FastTokenizer Subword-to-Character Span Reconstruction                      │   |
|   │ • Dense Semantic Embeddings: BAAI/bge-small-en-v1.5 + SHA-256 Disk Cache      │   |
|   │ • Category-Aware HDBSCAN Density Clustering (Metric: Cosine, Outliers Kept)   │   |
|   │ • Active Merge Safety Guards (Disallowed Accessory & Generic Word Merges)     │   |
|   │ • Aspect Typing (Product / Service / Delivery / Seller / Packaging / Unknown) │   |
|   │ • Provenance Audit (100% Byte-Level Verification to Sentence & Clean Text)    │   |
|   └──────────────────────────────────────┬────────────────────────────────────────┘   |
|                                          │                                            |
|                                          ▼                                            |
|   ┌───────────────────────────────────────────────────────────────────────────────┐   |
|   │ STAGES 3–6 (Active Roadmap)                                                   │   |
|   │ • Stage 3: Aspect-Conditioned Sentiment Classification & MMR Evidence Mining │   |
|   │ • Stage 4: Bayesian Weighted Confidence Aggregation & Contradiction Scores    │   |
|   │ • Stage 5: High-Throughput Asynchronous FastAPI Service (Pydantic v2)         │   |
|   │ • Stage 6: Interactive Executive Dashboard (React / Vite + Verbatim Drawers)  │   |
|   └───────────────────────────────────────────────────────────────────────────────┘   |
+───────────────────────────────────────────────────────────────────────────────────────+
```

---

## 📊 Project Status & Verification Checklist

| Milestone | Stage | Implementation Focus | Status | Tests |
|---|---|---|:---:|:---:|
| **Stage 1** | **Data Foundation** | Ingestion, Unicode cleaning, deduplication, product splits, two-tier sentence offsets | **Complete** ✅ | 95 / 95 Passing |
| **Stage 2** | **Aspect Intelligence** | DeBERTa BIO tagging, span reconstruction, BGE normalization, HDBSCAN, alias guards, typing | **Complete** ✅ | 27 / 27 Passing |
| **Stage 3** | **Sentiment & Evidence** | Aspect-conditioned DeBERTa sentiment ($[\text{CLS}] \text{ Aspect } [\text{SEP}] \text{ Sentence }$), MMR selection | *In Progress* 🔄 | — |
| **Stage 4** | **Opinion Aggregation** | Quality-weighted scoring, Bayesian confidence intervals, cross-product comparison | *Planned* 📋 | — |
| **Stage 5** | **API Engine** | Asynchronous FastAPI service, Pydantic v2 validation, Parquet/SQLite query cache | *Planned* 📋 | — |
| **Stage 6** | **Product Dashboard** | Responsive React/Vite dashboard, component radar charts, verbatim evidence drawers | *Planned* 📋 | — |

**Total Automated Tests:** **122 / 122 passing (100%) in 3.08 seconds.**

---

## 📁 Repository Structure

```text
ProductLens/
├── configs/
│   └── default.yaml               # Authoritative hierarchical configuration (smoke/dev/full)
├── productlens/                   # Core Python package
│   ├── __init__.py                # Package exports (v0.1.0)
│   ├── config.py                  # Frozen configuration dataclasses, profiles, and overrides
│   ├── schemas.py                 # Canonical dataclass schemas (Review, Sentence, Aspect, etc.)
│   ├── utils.py                   # Device detection (CUDA/CPU), stable IDs, GPU cleanup, timers
│   ├── data/                      # Stage 1: Data Engineering & Foundation
│   │   ├── __init__.py
│   │   ├── clean.py               # Unicode NFC normalization, HTML entity/tag and URL stripping
│   │   ├── dedupe.py              # Exact SHA-256 and MinHash near-duplicate filtering
│   │   ├── load_amazon.py         # Multi-format ingestion (Hugging Face streaming & local files)
│   │   ├── sampling.py            # Category, product-stratified, rating, and random sampling
│   │   ├── sentence_split.py      # Two-tier abbreviation-aware sentence segmentation
│   │   ├── split.py               # Product-aware train/val/test splitting (zero leakage)
│   │   └── synthetic.py           # Deterministic 302-review multi-category smoke dataset
│   └── aspects/                   # Stage 2: Aspect Intelligence & Normalization
│       ├── __init__.py
│       ├── aliases.py             # Deterministic alias mapping, typing & merge safety guards
│       ├── bio_model.py           # Transformer BIO tagging head & subword span reconstruction
│       ├── candidates.py          # Syntactic noun phrase & compound candidate extraction
│       ├── clustering.py          # Category-aware HDBSCAN clustering & outlier preservation
│       ├── normalize.py           # Dense embeddings (BGE-small), HashingEmbedder & disk cache
│       ├── run.py                 # CLI pipeline runner for Stage 2
│       └── train_extractor.py     # DeBERTa token classification trainer with OOM recovery
├── notebooks/                     # Interactive walkthroughs (percent format)
│   ├── 01_data.py                 # Data foundation, cleaning, and offset verification
│   └── 02_aspects.py              # Aspect extraction, clustering, typing, and traceability
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Shared fixtures and mock generators
│   ├── test_data.py               # Ingestion, cleaning, deduplication, and split tests
│   ├── test_sentence_split.py     # Sentence boundary and offset preservation tests
│   ├── test_aspects.py            # BIO reconstruction, multi-aspect spans, typing, and offsets
│   └── test_normalization.py      # Embeddings, HDBSCAN, outlier retention, and alias safety
├── artifacts/                     # Generated pipeline outputs & verification markers
│   └── aspects/
│       ├── DONE.json              # Stage completion verification marker
│       ├── metrics.json           # Execution and clustering metrics
│       ├── aspect_mentions.parquet# Extracted aspect spans with exact character offsets
│       └── aspect_clusters.parquet# Normalized canonical aspect clusters
├── report.md                      # Comprehensive academic research report
├── requirements.txt               # Locked production dependencies
├── .gitignore                     # Clean exclusion of environments, caches, and weights
└── README.md
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup

ProductLens requires Python 3.11+. We recommend using [`uv`](https://github.com/astral-sh/uv) or a standard `venv`:

```bash
# Clone the repository
git clone https://github.com/parth-sarthi-code/ProductLens.git
cd ProductLens

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (with CUDA 12.4 support if GPU is available)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Verify Installation & Run Tests

Run the full automated test suite (runs 100% offline, GPU-independent):

```bash
python -m pytest tests/ -q
```
*Expected Output:* `122 passed in ~3.0s`.

### 3. Execute the Stage 2 Smoke Pipeline

Run the end-to-end aspect extraction and normalization pipeline on the deterministic smoke corpus:

```bash
python -m productlens.aspects.run --profile smoke
```

Outputs will be saved directly into `artifacts/aspects/`:
* `artifacts/aspects/aspect_mentions.parquet`: Extracted spans with document offsets.
* `artifacts/aspects/aspect_clusters.parquet`: Discovered canonical clusters and medoids.
* `artifacts/aspects/DONE.json`: Execution verification marker with timestamp and config hash.

### 4. Interactive Notebook Walkthroughs

The notebooks are maintained in Jupytext percent format (`.py`) for clean version control:

```bash
# Execute Notebook 01: Data Foundation
python notebooks/01_data.py

# Execute Notebook 02: Aspect Intelligence
python notebooks/02_aspects.py
```

---

## 🔬 Benchmark Highlights (Stage 2)

Evaluated across $298$ cleaned reviews and $528$ sentences across 6 Amazon categories (Electronics, Beauty, Home, Sports, Books, Automotive):

* **Exact Offset Traceability:** **100.0%** (0 character offset mismatches between extracted spans and source text).
* **Multi-Aspect Isolation:** Successfully separates co-occurring aspects within single sentences:
  > *"The sound quality is excellent but the microphone is terrible."*  
  > $\implies$ `sound quality` (offsets $[4:17]$) & `microphone` (offsets $[39:49]$).
* **Aspect Typing Accuracy:** Correctly routes logistical mentions (e.g., courier transit delays or box damage) away from physical product metrics.
* **Pipeline Latency:** **1.38 seconds** for end-to-end smoke execution on an NVIDIA GeForce RTX 4050 Laptop GPU (peak VRAM: $1.84\text{ GB}$).

---

## 📖 Citation & Academic Reference

If you use ProductLens or refer to the technical methodology in your research, please cite:

```bibtex
@techreport{productlens2026,
  title       = {ProductLens: Aspect-Level Product Intelligence from Large-Scale Customer Reviews},
  author      = {Sarthi, Parth and ProductLens Systems Group},
  institution = {NLP \& Machine Intelligence Systems Laboratory},
  year        = {2026},
  month       = {September},
  url         = {https://github.com/parth-sarthi-code/ProductLens},
  note        = {Technical Report \& Research Architecture}
}
```

---

## 📜 License

This project is licensed under the Apache 2.0 License — see the repository files for details.
