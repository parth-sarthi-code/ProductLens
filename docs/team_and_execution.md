# Team Organization, Scope Boundaries & Execution

This document outlines team role divisions, module deliverables, project boundaries (non-goals), shared workflows, and execution philosophies for **ProductLens**.

---

## 1. Explicit Non-Goals (Scope Boundaries)

To maintain focus, scientific rigor, and timely execution within hardware limits, ProductLens explicitly avoids:

- **Training Multi-Billion-Parameter Models From Scratch**: Focusing on fine-tuning and domain adaptation of 100M–400M parameter encoders.
- **End-to-End Training on 100+ GB Raw Datasets**: Ingestion processes filter down to a high-quality laptop/electronics subset rather than boiling the ocean.
- **Mixture-of-Experts (MoE) Infrastructure**: Unnecessary computational overhead for token classification and sentiment tasks.
- **Pure Dependency on Proprietary LLM APIs**: OpenAI/Anthropic APIs will not be used as the core ABSA engine. The project's core contribution is self-hosted, trainable deep learning models.
- **Superficial Chatbot Interfaces**: No generic LLM chat wrappers. The interface is dedicated to structured component intelligence, visual evidence, and metrics.
- **Generic Review-Level Sentiment**: The problem is aspect-based sentiment analysis, not standard binary document classification.

---

## 2. Team Role Division

The team consists of four members with distinct primary ownership areas designed to operate in parallel while integrating through defined contracts:

``` text
┌─────────────────────────────────────────────────────────────┐
│                       Member 1                              │
│             NLP: Aspect Extraction & Sequence               │
└──────────────────────────────┬──────────────────────────────┘
                               │ BIO Spans
┌──────────────────────────────▼──────────────────────────────┐
│                       Member 2                              │
│       NLP: Aspect Sentiment & Normalization Intelligence    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Normalized Aspect-Sentiments
┌──────────────────────────────▼──────────────────────────────┐
│                       Member 3                              │
│            Data Engineering & FastAPI Backend               │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST Endpoints / JSON Schema
┌──────────────────────────────▼──────────────────────────────┐
│                       Member 4                              │
│        Frontend, Data Visualizations & Integration          │
└─────────────────────────────────────────────────────────────┘
```

### Member 1 — NLP / Aspect Extraction

#### Responsibilities
- Review ABSA literature and existing benchmark datasets (e.g., SemEval 2014/2015/2016).
- Prepare and format token-level aspect extraction datasets (BIO tagging scheme).
- Implement subword token alignment with character spans.
- Fine-tune BERT, RoBERTa, and DeBERTa token classifiers.
- Hyperparameter tuning (sequence length, learning rates, weight decay, warmup).
- Evaluate Token & Entity-Level Precision, Recall, and F1 metrics.
- Investigate joint extraction-sentiment architectures as an advanced milestone.

#### Primary Deliverables
- Prepared BIO-tagged training/validation/test datasets.
- Fine-tuned aspect extraction checkpoints.
- Model training & evaluation scripts.
- Benchmark evaluation table and error analysis.

---

### Member 2 — NLP / Sentiment + Aspect Intelligence

#### Responsibilities
- Curate aspect-level sentiment datasets (positive, neutral, negative pairs).
- Fine-tune Transformer classification heads on aspect-conditioned review text.
- Build the aspect normalization pipeline using phrase embeddings (`sentence-transformers`).
- Implement and benchmark aspect clustering methods (cosine thresholding vs. HDBSCAN).
- Formulate model confidence estimation and uncertainty scores.
- Perform detailed error and confusion matrix analysis.
- Collaborate with Member 1 on joint ABSA modeling.

#### Primary Deliverables
- Trained aspect-conditioned sentiment models.
- Embedding and normalization pipeline with canonical taxonomy mappings.
- Clustering experiment notebooks and quantitative metrics.
- Confidence scoring utilities and sentiment error reports.

---

### Member 3 — Data Engineering + Backend

#### Responsibilities
- Build dataset filtering, cleaning, and deduplication scripts for Amazon review dumps.
- Implement streaming/chunked data processing for large JSON/CSV files.
- Design database schemas (SQLite / PostgreSQL) for products, reviews, components, and scores.
- Build high-performance backend endpoints using FastAPI.
- Integrate trained PyTorch/Hugging Face models into batch and online inference pipelines.
- Implement caching mechanisms (e.g., Redis or in-memory caches) for repeated product queries.
- Build clean mock data loaders for frontend development and headless testing.

#### Primary Deliverables
- End-to-end data preparation and ingestion pipeline.
- Database models and migration scripts.
- Production-ready FastAPI backend application.
- Model-serving and inference endpoints with validation schemas.

---

### Member 4 — Frontend + Visualization + System Integration

#### Responsibilities
- Design and build the ProductLens interactive dashboard (React + Tailwind CSS).
- Implement component scorecards, radar charts, and sentiment distribution bars.
- Build interactive aspect drill-down panels displaying supporting review quotes.
- Implement product search, review filtering, and optional side-by-side product comparison.
- Wire frontend components to FastAPI endpoints with robust loading and error states.
- Conduct cross-browser usability testing and responsive design validation.
- Coordinate final system integration between backend, models, and UI.

#### Primary Deliverables
- React-based web dashboard.
- Interactive data visualizations and evidence inspection views.
- Seamless backend API integration.
- Final user interface polish and demonstration workflow.

---

## 3. Shared Responsibilities

All team members actively collaborate on cross-functional activities:
- **Dataset Understanding & Literature Review**: Establishing shared context and domain assumptions.
- **Evaluation & Peer Validation**: Verifying that model metrics are defensible and free from data leakage.
- **Git & GitHub Workflows**: Maintaining clean branch hygiene, readable commit messages, and PR reviews.
- **Testing & Continuous Integration**: Writing unit and integration tests across data, models, and endpoints.
- **Documentation & Reporting**: Authoring comprehensive technical writeups, architecture diagrams, and slide decks.
- **Demo & Presentation Preparation**: Ensuring the end-to-end user experience runs seamlessly for final project demos.

---

## 4. Core Design Philosophy

1. **Model Intelligence Over Model Size**: A well-architected 100M–400M parameter NLP pipeline outperforms brute-force large models in domain specificity, speed, and cost efficiency.
2. **Modular Architecture**: Every subsystem (Extraction, Normalization, Sentiment, Aggregation, Serving) is decoupled with clean interfaces and independent test coverage.
3. **Evidence-Based Interpretability**: Every quantitative component score must be backed by traceable, highlightable user review citations.
4. **Measurable Performance**: Design decisions are validated via rigorous NLP evaluation metrics (Precision, Recall, Macro F1) and ablation studies rather than anecdotal impressions.

---

## 5. Project Success Checklist

- [ ] High-accuracy aspect extraction token classification ($F_1 > \text{baseline}$).
- [ ] Context-aware aspect-level sentiment classification ($F_1 > \text{baseline}$).
- [ ] Reliable normalization of synonymous component expressions.
- [ ] Explainable component scores with representative review quotes.
- [ ] Quantitative benchmarking comparing BERT, RoBERTa, and DeBERTa.
- [ ] Fully reproducible data processing and model fine-tuning pipeline.
- [ ] Responsive, interactive dashboard connecting models, backend, and visualizations.
