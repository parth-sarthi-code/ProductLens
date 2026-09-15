# ProductLens — Intelligent Product Review Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

**ProductLens** is an NLP-driven system that transforms thousands of unstructured consumer electronics reviews into structured, component-level product intelligence. 

Instead of assigning a single, generic sentiment score to an entire review, ProductLens implements **Aspect-Based Sentiment Analysis (ABSA)** to evaluate individual hardware components (e.g., *Display, Battery, Keyboard, Performance, Thermals*) with granular sentiment scores, confidence ratings, and representative review evidence.

---

## Documentation Index

The complete project specification is organized into modular documents:

| Document | Purpose & Contents |
| :--- | :--- |
| **[`README.md`](README.md)** | Project overview, core problem, high-level architecture, technology stack, and repo structure. |
| **[`docs/pipeline_and_specs.md`](docs/pipeline_and_specs.md)** | Data strategy, SemEval datasets, Aspect Extraction (BIO), Aspect Normalization, Aspect Sentiment, Joint ABSA, Evidence Selection, and Aggregation Engine. |
| **[`docs/models_and_experiments.md`](docs/models_and_experiments.md)** | Model selection rationale (100M–400M), candidate Transformers (BERT, RoBERTa, DeBERTa-v3, DistilBERT), training strategy, domain adaptation, and ablation studies. |
| **[`docs/team_and_execution.md`](docs/team_and_execution.md)** | Non-goals / scope boundaries, team role divisions (Members 1–4), shared deliverables, design philosophy, and success criteria checklist. |

---

## 1. Project Overview & Motivating Example

When a customer evaluates a laptop review:
> *"The display is beautiful, but battery life is terrible. The keyboard feels very comfortable."*

A standard sentiment analysis system typically predicts:
``` text
Overall Sentiment → Positive (Score: 0.67)
```
This naive aggregation discards critical product signals:
- The customer loved the screen and keyboard.
- The customer found the battery unacceptable.

**ProductLens extracts granular, aspect-conditioned insights:**
``` text
Display  → Positive (Confidence: 0.96)
Battery  → Negative (Confidence: 0.94)
Keyboard → Positive (Confidence: 0.91)
```

By aggregating thousands of reviews, ProductLens produces interpretable component-level report cards backed by verified review citations.

---

## 2. Core Problem & Aspect Normalization

The core machine learning challenge is:
> **Given unstructured user reviews, automatically identify mentions of product aspects, map them to canonical components, and determine the exact sentiment expressed toward each aspect.**

Customers use diverse phrasing to describe identical hardware components:
- `"screen"`, `"display"`, `"IPS panel"`, `"screen quality"` $\longrightarrow$ **DISPLAY**
- `"battery life"`, `"battery backup"`, `"runtime"` $\longrightarrow$ **BATTERY**

ProductLens addresses this through dense semantic phrase embeddings and density-based clustering to normalize synonyms into canonical aspect ontologies.

---

## 3. Project Goals

### Primary Goals
1. **Data Pipeline**: Filter, clean, and segment large-scale electronics reviews into high-quality training subsets.
2. **Aspect Extraction**: Train token-classification models using BIO tagging to isolate aspect mentions.
3. **Aspect-Level Sentiment**: Classify sentiment polarity conditioned on specific aspect spans.
4. **Aspect Normalization**: Map heterogeneous aspect phrases to canonical hardware components via embeddings and clustering.
5. **Intelligent Aggregation**: Compute weighted component scores factoring in model confidence and mention density.
6. **Traceable Evidence**: Select representative, high-diversity review quotes justifying each score.
7. **Interactive Dashboard**: Present insights through a modern, responsive user interface.
8. **Empirical Benchmarking**: Quantitatively evaluate model variants against established NLP baselines.

### Advanced Goals (Extensions)
- Domain adaptation via Masked Language Modeling (MLM) on consumer electronics corpus.
- Joint ABSA modeling (predicting aspect-sentiment tuples simultaneously).
- Head-to-head product comparison view.

---

## 4. End-to-End System Architecture

``` text
                         ┌─────────────────────┐
                         │   Product URL /     │
                         │   Review Dataset    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Ingestion & Clean   │
                         │ Sentence Splitting  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │       Aspect Extraction          │
                   │      DeBERTa/RoBERTa (BIO)       │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼
                          Candidate Aspects
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │ Aspect Normalization & Cluster   │
                   │ Sentence Embeddings + HDBSCAN    │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼
                         Normalized Components
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │ Aspect-Level Sentiment           │
                   │ Transformer Classification       │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼
                      Aspect + Sentiment Pairs
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │ Evidence Selection & MMR Filter  │
                   │ Representative Review Sentences  │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │ Aggregation & Scoring Engine     │
                   │ Weighted Confidence + Frequency  │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │ Product Intelligence Dashboard   │
                   │ (React + Tailwind CSS)           │
                   └──────────────────────────────────┘
```

---

## 5. Technology Stack

### Machine Learning & Data Science
- **Core Languages**: Python 3.10+
- **Deep Learning**: PyTorch, Hugging Face `transformers`, `datasets`, `accelerate`, `peft`
- **Embeddings & Clustering**: `sentence-transformers`, `scikit-learn`, `hdbscan`, `umap-learn`
- **Data Manipulation**: `pandas`, `numpy`, `spacy`

### Backend & API
- **Framework**: FastAPI (Python)
- **Data Store**: SQLite / PostgreSQL
- **Data Validation**: Pydantic v2

### Frontend & Visualizations
- **Framework**: React 18+ (Vite)
- **Styling**: Tailwind CSS
- **Visualization**: Recharts / Lucide Icons

### Tooling & Infrastructure
- Git & GitHub
- Configuration-driven training runners (`PyYAML`)
- Jupyter Lab for exploratory data analysis

---

## 6. Repository Layout

``` text
productlens/
├── docs/                                # Modular Documentation
│   ├── pipeline_and_specs.md           # NLP Pipeline & Feature Specs
│   ├── models_and_experiments.md       # Models, Training & Experiments
│   └── team_and_execution.md           # Team Division & Scope Boundaries
│
├── data/
│   ├── raw/                            # Raw review datasets
│   ├── processed/                      # Cleaned & tokenized splits
│   └── annotations/                    # Gold-standard ABSA annotations
│
├── models/
│   ├── aspect_extraction/              # Token classification checkpoints
│   ├── sentiment/                      # Aspect sentiment checkpoints
│   └── embeddings/                     # Phrase embeddings & cluster models
│
├── notebooks/
│   ├── data_exploration/               # Review distribution & EDA
│   ├── model_experiments/              # Training & fine-tuning logs
│   └── evaluation/                     # Error analysis & confusion matrices
│
├── src/
│   ├── preprocessing/                  # Text cleaning & sentence segmentation
│   ├── aspect_extraction/              # BIO token tagging inference
│   ├── normalization/                  # Embedding similarity & clustering
│   ├── sentiment/                      # Aspect-conditioned sentiment classifier
│   ├── evidence/                       # MMR evidence selection
│   └── aggregation/                    # Scoring algorithms & confidence weighting
│
├── backend/                            # FastAPI REST API
├── frontend/                           # React dashboard
├── tests/                              # Unit & integration tests
├── configs/                            # Model training & pipeline YAML configs
├── requirements.txt
└── README.md
```

---

## 7. Expected Final Deliverables

1. **Ingestion & Preprocessing**: Clean pipeline extracting laptop reviews and splitting into clean sentence units.
2. **Trained NLP Models**: High-performing token extraction and aspect-level sentiment classification models.
3. **Normalization Engine**: Semantic clustering pipeline grouping synonyms to parent hardware components.
4. **Scoring Engine**: Transparent scoring algorithm with confidence intervals and evidence extraction.
5. **Interactive Dashboard**: Modern UI with component cards, sentiment breakdowns, and customer quote drilldowns.
6. **Scientific Benchmark Report**: Quantitative evaluation comparing BERT, RoBERTa, and DeBERTa across token F1 and sentiment Macro F1.

---

## 8. Getting Started

Detailed instructions for environment setup, dataset downloads, and training pipelines will be provided as development milestones are completed.

Refer to [`docs/team_and_execution.md`](docs/team_and_execution.md) for team member deliverables and active milestones.
