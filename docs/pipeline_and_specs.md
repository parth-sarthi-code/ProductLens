# NLP Pipeline & Feature Specifications

This document outlines the end-to-end data and machine learning pipeline for **ProductLens**, detailing aspect extraction, normalization, sentiment analysis, evidence selection, and opinion aggregation.

---

## 1. Data Strategy

The Amazon review ecosystem contains datasets that can be extremely large, potentially reaching hundreds of gigabytes. We will **not attempt to train directly on the entire dataset**. The large corpus will be treated as a source from which a suitable domain-specific dataset is extracted.

### Data Processing Pipeline

``` text
Large Amazon Review Corpus
          ↓
Category filtering
          ↓
Electronics
          ↓
Laptop/computer-related reviews
          ↓
Cleaning & Preprocessing
          ↓
Deduplication
          ↓
Sentence extraction & segmentation
          ↓
Training / validation / test subsets
```

### Core Data Principle

The project does not require all available reviews. A carefully selected, clean, and well-labelled dataset is far more valuable than training blindly on 100+ GB of noisy data.

---

## 2. Training Datasets & ABSA Tasks

The final training dataset contains examples formatted for two major NLP tasks:

### Task A — Aspect Extraction (Token Classification)

Aspect extraction identifies span-level components mentioned in reviews using BIO-tagging format:

``` text
Review:
"The display is beautiful but the battery life is terrible."

Tokens / labels:
The        → O
display    → B-ASPECT
is         → O
beautiful  → O
but        → O
the        → O
battery    → B-ASPECT
life       → I-ASPECT
is         → O
terrible   → O
```

**BIO Tagging Scheme:**
- `B-ASPECT`: Beginning of an aspect mention
- `I-ASPECT`: Continuation of an aspect mention
- `O`: Outside any aspect mention

### Task B — Aspect-Level Sentiment

Determines sentiment orientation conditioned on a specific extracted aspect span:

``` text
Example 1:
Text: "The display is beautiful."
Target Aspect: display
Sentiment: Positive

Example 2:
Text: "The battery life is terrible."
Target Aspect: battery
Sentiment: Negative
```

**Target Classes:**
- `Positive`
- `Neutral`
- `Negative`

*(Fine-grained 5-star sentiment scales may be explored in subsequent iterations).*

---

## 3. Aspect Extraction Architecture

The first NLP stage uses **Transformer-based token classification**:

``` text
Review Text
  ↓
Tokenizer (WordPiece / Byte-Pair Encoding)
  ↓
Encoder Backbone (DeBERTa / RoBERTa)
  ↓
Contextual Token Representations
  ↓
Token Classification Linear Head + Softmax
  ↓
BIO Label Predictions
  ↓
Extracted Aspect Phrases
```

### Example
``` text
"The keyboard has excellent key travel."
→ keyboard: B-ASPECT

"The battery life is surprisingly good."
→ battery: B-ASPECT
→ life:    I-ASPECT
```

Unlike rule-based or regex matching, token classification leverages deep contextual cues to identify novel and unconventional aspect expressions.

---

## 4. Aspect Normalization & Clustering

Reviewers describe the same hardware component using diverse terminology:

``` text
"screen"
"display"
"panel"
"IPS panel"
"screen quality"
       │
       ▼
    DISPLAY
```

``` text
"battery life"
"battery backup"
"runtime"
       │
       ▼
    BATTERY
```

### Normalization Pipeline

``` text
Aspect Phrase
      ↓
Sentence / Phrase Embedding Model
      ↓
Dense Vector Representation
      ↓
Semantic Similarity / Clustering
      ↓
Canonical Normalized Aspect
```

- **Candidate Embedding Models**: Sentence-BERT family, `all-MiniLM-L6-v2`, MPNet-class sentence encoders.
- **Clustering / Matching Approaches**:
  - Exact & fuzzy canonical ontology matching.
  - Cosine similarity thresholding against seed aspect prototypes.
  - Unsupervised clustering: **HDBSCAN** or **K-Means**. HDBSCAN is preferred as it discovers arbitrary cluster counts without fixing $k$ a priori and isolates noise.

---

## 5. Aspect-Level Sentiment Classification

Once an aspect is identified, its specific sentiment polarity is predicted using contextual input:

``` text
[CLS] Aspect Phrase [SEP] Review Sentence Context [SEP]
                        ↓
            Transformer Encoder
                        ↓
            Classification Head
                        ↓
          Positive / Neutral / Negative
```

This isolates conflicting sentiments within complex compound sentences:
> *"The screen is fantastic but battery life is disappointing."*
> - `screen` $\rightarrow$ **Positive**
> - `battery life` $\rightarrow$ **Negative**

---

## 6. Joint ABSA — Advanced Extension

As an advanced optimization beyond pipelined extraction and classification, a **joint aspect-sentiment model** will be explored:

``` text
Review Sentence
       ↓
Transformer Encoder
       ↓
Joint Tagging / Span Classification Head
       ↓
(Aspect, Sentiment) Tuples
```

Example output: `[("display", Positive), ("battery", Negative)]`.

> **Note:** Joint ABSA is an advanced milestone to be benchmarked after the dual-stage baseline is established and verified.

---

## 7. Evidence Extraction & Interpretability

Rather than simply outputting a scalar score, ProductLens provides traceable, explainable review evidence for every component.

### Example Component Output
``` text
BATTERY — 5.4 / 10
Reviews Analyzed: 1,284
Positive: 14% | Neutral: 21% | Negative: 65%

Representative Opinions:
• "Battery barely lasts four hours."
• "Gaming drains the battery quickly."
• "Battery life is decent for office work."
```

### Sentence Selection Algorithm
Representative review snippets are extracted using:
1. **Aspect Cosine Similarity**: Embed review sentences and rank by relevance to the canonical aspect prototype.
2. **Confidence Filtering**: Retain predictions with high model probability.
3. **Maximal Marginal Relevance (MMR) / Diversity Filtering**: Prevent repetitive phrasing and highlight distinct user viewpoints.

---

## 8. Intelligent Aggregation Engine

Individual aspect opinions are aggregated into holistic product reports:

For every normalized aspect:
- Mention volume / frequency
- Proportion of Positive / Neutral / Negative mentions
- Mean model confidence score
- Selected representative evidence quotes

### Component Score Formula
$$\text{Aspect Score} = f(\text{Positive Ratio}, \text{Negative Ratio}, \text{Confidence}, \text{Weights})$$

**Configurable Weighting Parameters:**
- Model classification confidence
- Review helpfulness votes (if present in dataset)
- Review recency / date weighting
- Deduplication penalties for suspected bot reviews

---

## 9. Example Final Output & Product Comparison

### Component Scorecard
``` text
PRODUCT REVIEW INTELLIGENCE REPORT

Display       8.7 / 10  ████████░░
Performance   9.2 / 10  █████████░
Build Quality 8.1 / 10  ████████░░
Keyboard      7.8 / 10  ███████░░░
Thermals      7.1 / 10  ███████░░░
Battery       5.2 / 10  █████░░░░░
Speakers      6.4 / 10  ██████░░░░
```

### Side-by-Side Product Comparison (Extension)
``` text
Component         Laptop A       Laptop B
Display             8.7            8.1
Battery             5.2            8.3
Keyboard            7.8            7.2
Performance         9.2            8.5
Thermals            7.1            8.0
Build Quality       8.1            8.4
```

---

## 10. Amazon URL Integration & Ingestion

ProductLens supports review ingestion via:
1. **Curated Dataset / CSV Ingestion** (Primary research mechanism for reproducible evaluation).
2. **Amazon Product URL Ingestion** (Live acquisition via review scraper/API).

``` text
Amazon Product URL
        ↓
Review Ingestion Service
        ↓
Deduplication & Cleaning
        ↓
Trained NLP Pipeline
        ↓
Product Intelligence Report
```

> **Design Notice**: The core research novelty lies in the ABSA NLP models and aggregation engine, not scraping mechanics. Decoupled CSV loading guarantees reproducibility and safeguards development against third-party UI variations.
