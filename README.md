# ProductLens: Aspect-Level Product Intelligence from Customer Reviews

ProductLens is a machine learning system that analyzes customer reviews and extracts component-level insights instead of relying on a single overall star rating.

When people buy products on platforms like Amazon, reviews cover many different things at once: a laptop might have a gorgeous screen, but terrible battery life and a loud fan. Standard sentiment analysis collapses all of this into a single score like "3.5 out of 5 stars" or "Positive", which hides what is actually good or bad about the product.

ProductLens uses Aspect-Based Sentiment Analysis (ABSA) to break down reviews into specific hardware components (like Display, Battery, Keyboard, and Build Quality), scores each component separately, and links every score back to the exact sentence written by the real customer.

---

## Companion Documentation

This repository contains two in-depth reference documents:

* **what_and_why.md**: An architectural guide explaining every design decision, model selection rationale, and why this system is built this way.
* **report.md**: A formal technical report in academic research paper format with mathematical formulations, span algorithms, and benchmark tables.

---

## The Problem: Why Simple Ratings and LLMs Fall Short

### 1. The Single Star Rating Hides the Real Story
Imagine a customer shopping for a laptop who finds a model rated 3.2 stars. 
* Is the rating low because the processor is slow?
* Is it low because the battery only lasts two hours?
* Or did the buyer simply dislike the colour?

A single number cannot answer these questions. A buyer who always uses their laptop at a desk plugged into power might not care about battery life at all, but they care deeply about screen brightness. ProductLens separates reviews into individual component ratings so buyers and engineers see the full breakdown.

### 2. Delivery and Packaging Issues Pollute Product Ratings
A large number of negative reviews on e-commerce platforms have nothing to do with the product itself:
> "1 star: The delivery driver threw the package over the fence, the cardboard box was crushed, and it arrived three days late."

If this review is mixed into the product score, the product looks flawed even though the hardware inside works perfectly. ProductLens classifies every mention into categories such as Product, Delivery, Packaging, Seller, and Customer Service, ensuring logistical complaints do not drag down product hardware scores.

### 3. The Pitfalls of Simple ChatGPT or LLM Summaries
Many modern projects simply paste reviews into a Large Language Model (like GPT-4) with a prompt asking for a summary. While this sounds easy, it introduces serious real-world engineering problems:
* **Hallucinations:** Large generative models can invent details or mention features that do not exist in the actual reviews.
* **Lack of Proof:** You cannot easily click on a generated sentence to see the exact customer review and sentence it came from.
* **High Cost and Slow Speed:** Sending thousands of customer reviews to external cloud APIs costs significant money per product and takes minutes to run.
* **Inconsistency:** Running the exact same prompt tomorrow can produce a different summary with different scores.

ProductLens solves this with an auditable, local machine learning pipeline: every extracted point has exact character offsets pointing to the verified source sentence, runs locally on consumer hardware in seconds, and costs nothing in external API fees.

---

## How ProductLens Works: Step-by-Step

ProductLens works as an end-to-end pipeline divided into clear stages:

``` text
+-----------------------------------------------------------------------------+
|                         PRODUCTLENS PIPELINE OVERVIEW                       |
+-----------------------------------------------------------------------------+
|                                                                             |
|  1. Review Ingestion & Cleaning                                             |
|     Decode HTML entities, remove tracking links, standardize text.          |
|                                                                             |
|  2. Sentence Splitting with Abbreviation Protection                         |
|     Split reviews into clean sentences without breaking on titles           |
|     like "Dr." or abbreviations like "U.S." and "oz.".                      |
|                                                                             |
|  3. Aspect Span Extraction (DeBERTa-v3)                                     |
|     Use token classification to find exact aspect phrases like              |
|     "battery life" or "screen" using BIO tagging.                           |
|                                                                             |
|  4. Semantic Normalization & Clustering (BGE + HDBSCAN)                     |
|     Convert phrases into semantic vectors and group synonyms               |
|     ("screen", "display", "panel") into one canonical concept.              |
|                                                                             |
|  5. Active Merge Safety Guards                                              |
|     Prevent invalid merges (such as "screen" vs. "screen protector",        |
|     or "sound quality" vs. "build quality").                                |
|                                                                             |
|  6. Aspect Typing & Provenance Verification                                 |
|     Sort mentions into Product, Delivery, Packaging, Service, or Seller.    |
|     Verify that 100% of mentions link to source reviews with zero error.    |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### Step 1: Cleaning and Deduplicating Reviews
Customer reviews on Amazon contain HTML tags, broken characters, and duplicate bot reviews. ProductLens normalizes characters using standard Unicode, strips out web links, and removes both identical copies and near-duplicate reviews using similarity matching.

It also enforces **Product-Aware Splitting**: all reviews for a specific product stay together in either training, validation, or test sets. This ensures the model does not memorize specific product names and must learn genuine aspect language that generalizes to brand new products.

### Step 2: Accurate Sentence Splitting
Before finding aspects, reviews are split into individual sentences. Naive sentence splitters break text whenever they see a period, which corrupts sentences with abbreviations like "Mr. Smith", "5.5 oz.", or "U.S. version". ProductLens uses an abbreviation-aware state engine that protects titles and abbreviations, keeping character offsets completely accurate.

### Step 3: Finding Aspects with DeBERTa-v3 and BIO Tagging
Instead of searching for a fixed list of keywords (which misses unusual phrasing), ProductLens uses **DeBERTa-v3-base**, a modern transformer neural network. 

The model classifies every word using the standard **BIO tagging** scheme:
* **B-ASP (Begin Aspect):** The first word of an aspect phrase.
* **I-ASP (Inside Aspect):** The continuation words of an aspect phrase.
* **O (Outside):** Normal words that are not aspects.

Example:
> "The (O) sound (B-ASP) quality (I-ASP) is (O) great (O) but (O) the (O) microphone (B-ASP) is (O) terrible (O)."

This allows the model to cleanly isolate multiple distinct aspects in a single sentence:
1. `sound quality` (character offset 4 to 17)
2. `microphone` (character offset 39 to 49)

### Step 4: Semantic Normalization (Grouping Synonyms Together)
Different people use different words to describe the same part of a product:
* "screen", "display", "panel", "IPS monitor" all refer to the **Screen**.
* "battery life", "battery backup", "runtime" all refer to the **Battery**.

ProductLens converts each extracted phrase into a dense numerical vector using the **BAAI/bge-small-en-v1.5** embedding model. Phrases with similar meanings end up close together in vector space.

Then, an algorithm called **HDBSCAN** groups these vectors into clusters based on density:
* Unlike older algorithms like k-Means, HDBSCAN does not require you to guess the number of clusters in advance (a blender might have 5 components, while a laptop has 35).
* Outlier mentions (rare or unusual complaints) are never thrown away. Every mention is kept and given its own concept name so engineers never lose rare defect reports.

### Step 5: Merge Safety Guards (Preventing False Merges)
Pure machine learning models can sometimes group words that appear in similar contexts but are actually completely different items. For example:
* The words "screen" and "screen protector" have very high vector similarity because both appear near words like "scratches", "fingerprints", and "glass".
* If an algorithm merges them, a scratch on a cheap plastic protector will be counted as a defect in an expensive OLED screen!

ProductLens implements strict rule guards:
* **Accessory Guard:** Terms with accessory words (like "case", "cover", "protector", "sleeve") are never allowed to merge with base product terms.
* **Generic Word Guard:** Terms that only share generic words (like "quality", "performance", "durability") cannot merge. This prevents "sound quality" from merging with "build quality".

### Step 6: Aspect Typing and Full Audit
Every aspect mention is tagged with its domain type:
* **product:** Physical parts (screen, battery, keyboard, zipper, motor).
* **service:** Customer support, warranties, refund policies.
* **delivery:** Couriers, shipping speeds, late transit.
* **packaging:** Cardboard boxes, bubble wrap, unboxing condition.
* **seller:** Third-party merchants, store responsiveness.

Finally, the system runs an automated audit: it checks that every single extracted aspect can be sliced directly from the original review text using its recorded start and end character positions. If even a single character offset is misaligned, the audit fails.

---

## Current Implementation Status

| Stage | Focus Area | Status | Automated Tests |
|---|---|:---:|:---:|
| **Stage 1** | Data Foundation (Cleaning, Deduplication, Product Splits, Sentence Offsets) | Complete | 95 / 95 Passed |
| **Stage 2** | Aspect Intelligence (DeBERTa BIO Tagging, BGE Embeddings, HDBSCAN, Alias Guards) | Complete | 27 / 27 Passed |
| **Stage 3** | Aspect Sentiment & Evidence Selection (Aspect-Conditioned Sentiment, MMR Quotes) | In Progress | Upcoming |
| **Stage 4** | Opinion Aggregation Engine (Weighted Component Scorecards, Contradiction Scores) | Planned | Upcoming |
| **Stage 5** | REST API Backend (FastAPI, Pydantic v2 validation models, SQLite/Parquet query engine) | Planned | Upcoming |
| **Stage 6** | Web Dashboard (Interactive Component Cards, Radar Charts, Customer Quote Drawers) | Planned | Upcoming |

**Test Suite Health:** 122 automated unit and integration tests passing in ~3 seconds.

---

## Project Structure

``` text
ProductLens/
|-- configs/
|   `-- default.yaml               # Hierarchical pipeline and training settings
|-- productlens/                   # Main Python application package
|   |-- config.py                  # Configuration loader and profile manager
|   |-- schemas.py                 # Structured dataclasses for Reviews, Sentences, Aspects
|   |-- utils.py                   # GPU detection, stable IDs, timing, and cleanup
|   |-- data/                      # Stage 1: Data Engineering
|   |   |-- clean.py               # Text sanitization and HTML decoding
|   |   |-- dedupe.py              # Exact and near-duplicate review filtering
|   |   |-- load_amazon.py         # Amazon review dataset loading
|   |   |-- sentence_split.py      # Abbreviation-aware sentence segmentation
|   |   |-- split.py               # Product-aware train/test splitting
|   |   `-- synthetic.py           # Deterministic multi-category smoke dataset
|   `-- aspects/                   # Stage 2: Aspect Intelligence
|       |-- aliases.py             # Deterministic alias mapping and merge safety guards
|       |-- bio_model.py           # Transformer BIO sequence tagging and span builder
|       |-- candidates.py          # Candidate phrase extractor and stopword filter
|       |-- clustering.py          # Category-aware HDBSCAN clustering
|       |-- normalize.py           # Dense vector embedding generator and disk cache
|       |-- run.py                 # Command-line pipeline runner
|       `-- train_extractor.py     # DeBERTa token classification training pipeline
|-- notebooks/                     # Interactive walkthroughs (percent script format)
|   |-- 01_data.py                 # Data foundation and sentence offset verification
|   `-- 02_aspects.py              # Aspect extraction, clustering, and audit walkthrough
|-- tests/                         # Pytest test suite
|   |-- test_data.py               # Data loading, cleaning, and splitting tests
|   |-- test_sentence_split.py     # Sentence boundary and offset preservation tests
|   |-- test_aspects.py            # BIO reconstruction, multi-aspect spans, typing tests
|   `-- test_normalization.py      # Embeddings, HDBSCAN clustering, and alias safety tests
|-- artifacts/                     # Generated pipeline outputs and verification markers
|   `-- aspects/
|       |-- DONE.json              # Stage completion verification marker
|       |-- metrics.json           # Execution metrics and cluster statistics
|       |-- aspect_mentions.parquet# Extracted aspects with character offsets
|       `-- aspect_clusters.parquet# Grouped canonical aspect clusters
|-- what_and_why.md                # Comprehensive architectural rationale and trade-offs
|-- report.md                      # Academic research paper report with technical details
|-- requirements.txt               # Project Python dependencies
`-- README.md
```

---

## Quickstart Guide

### 1. Requirements and Setup
ProductLens requires Python 3.11+. You can set up an environment using standard Python virtual environments:

```bash
# Clone the repository
git clone https://github.com/parth-sarthi-code/ProductLens.git
cd ProductLens

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (installs PyTorch with CUDA if available)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Run the Full Test Suite
All tests run 100% locally without needing an active internet connection or GPU:

```bash
python -m pytest tests/ -q
```
*Expected Result:* `122 passed in ~3.0s`

### 3. Run the Stage 2 Aspect Pipeline
Execute the complete end-to-end extraction and clustering pipeline on the verified smoke dataset:

```bash
python -m productlens.aspects.run --profile smoke
```

Outputs will be saved in `artifacts/aspects/`:
* `aspect_mentions.parquet`: Every extracted aspect with exact sentence and review offsets.
* `aspect_clusters.parquet`: The grouped canonical clusters and representative terms.
* `DONE.json`: Verification record confirming runtime, hardware used, and 100% offset validity.

### 4. Run the Walkthrough Notebooks
The notebooks are maintained as clean Python scripts using the Jupytext percent format:

```bash
# Notebook 1: Data cleaning, deduplication, and sentence splitting
python notebooks/01_data.py

# Notebook 2: Aspect extraction, clustering, typing, and verification
python notebooks/02_aspects.py
```

---

## Hardware and Performance

ProductLens is engineered to run on standard student and consumer hardware without requiring expensive cloud instances:

* **Tested Machine:** Standard laptop with an NVIDIA GeForce RTX 4050 GPU (6 GB VRAM) running Linux.
* **GPU Memory Usage:** Under 2.0 GB of VRAM during model inference (leaving plenty of memory for other applications).
* **Speed:** The Stage 2 smoke pipeline processes hundreds of reviews, extracts over 1,100 aspect mentions, and clusters them in under 1.5 seconds.
* **CPU Fallback:** If no NVIDIA GPU is detected, the entire pipeline automatically switches to CPU execution without crashing.

---

## Technology Stack

* **Language:** Python 3.11+
* **Deep Learning & Transformers:** PyTorch, Hugging Face Transformers (`microsoft/deberta-v3-base`)
* **Vector Embeddings & Clustering:** Sentence-Transformers (`BAAI/bge-small-en-v1.5`), HDBSCAN, Scikit-learn
* **Data Storage & DataFrames:** Pandas, PyArrow (Parquet)
* **Testing & Configuration:** Pytest, PyYAML
* **Upcoming Serving & UI:** FastAPI, Pydantic v2, React 18, Vite

---

## Citation

If you refer to ProductLens or its technical architecture in your academic project or research, please cite:

```bibtex
@techreport{productlens2026,
  title       = {ProductLens: Aspect-Level Product Intelligence from Large-Scale Customer Reviews},
  author      = {Sarthi, Parth and ProductLens Systems Group},
  institution = {NLP and Machine Intelligence Systems Laboratory},
  year        = {2026},
  month       = {September},
  url         = {https://github.com/parth-sarthi-code/ProductLens},
  note        = {Technical Report and Architecture Guide}
}
```

---

## License

This project is licensed under the Apache 2.0 License. See the repository files for details.
