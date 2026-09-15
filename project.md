# ProductLens --- Intelligent Product Review Analysis

## 1. Project Overview

**ProductLens** is an NLP-based intelligent system that analyzes large
collections of product reviews and breaks them down into
**component/aspect-level opinions**.

The initial target domain is **laptops and consumer electronics**.

A user can provide an Amazon product link (or, during development, a
review dataset/CSV). The system obtains the available reviews and
processes them to answer questions such as:

-   What are customers saying about the display?
-   Is the keyboard generally liked?
-   What are the major complaints about battery life?
-   How do customers perceive performance and thermals?
-   Which components/features receive the most positive or negative
    feedback?
-   How strong is the evidence behind each component score?

Instead of assigning one sentiment to an entire review, the system
performs **Aspect-Based Sentiment Analysis (ABSA)**.

### Example

Input review:

> "The display is beautiful, but battery life is terrible. The keyboard
> feels very comfortable."

The system should produce:

``` text
Display  → Positive
Battery  → Negative
Keyboard → Positive
```

This information is then aggregated across hundreds or thousands of
reviews to produce an interpretable component-level product report.

------------------------------------------------------------------------

# 2. Core Problem

A normal sentiment-analysis system might classify the review above
simply as:

``` text
Overall sentiment → Positive
```

That loses important information.

A customer may simultaneously:

-   like the display,
-   dislike the battery,
-   and like the keyboard.

Therefore, our system needs to identify **what aspect is being
discussed** and **what sentiment is associated with that particular
aspect**.

The central problem is:

> Given an unstructured product review, automatically identify product
> aspects/components and determine the sentiment expressed toward each
> aspect.

The project additionally addresses **aspect normalization**, because
customers can refer to the same component using different words.

For example:

``` text
"screen"
"display"
"panel"
"IPS panel"
"screen quality"
```

may all refer to the broader aspect:

``` text
DISPLAY
```

Likewise:

``` text
"battery life"
"battery backup"
"runtime"
```

may map to:

``` text
BATTERY
```

------------------------------------------------------------------------

# 3. Project Goals

## Primary Goals

1.  Collect/process a large product-review corpus.
2.  Clean and preprocess review text.
3.  Extract product aspects from reviews.
4.  Determine sentiment for each extracted aspect.
5.  Normalize semantically similar aspect expressions.
6.  Aggregate opinions across many reviews.
7.  Generate component-level scores and sentiment distributions.
8.  Provide representative review evidence for each component.
9.  Present the results through an interactive dashboard.
10. Evaluate the NLP models quantitatively.

## Secondary/Advanced Goals

If time permits:

-   Domain adaptation toward laptop/electronics reviews.
-   Confidence estimation for predictions.
-   Duplicate/near-duplicate review handling.
-   Comparison between multiple Transformer architectures.
-   Product-to-product comparison.
-   Automatic discovery of previously unseen aspects.

------------------------------------------------------------------------

# 4. Why This Is an Intelligent Model Design Project

The project is not simply:

``` text
Amazon → API → summary
```

The central intelligence is performed by trained NLP models.

The system must learn to:

-   recognize aspect expressions,
-   distinguish relevant from irrelevant words,
-   associate sentiment with the correct aspect,
-   understand multiple opinions in a single sentence,
-   recognize semantically equivalent aspect names,
-   aggregate noisy opinions from many users.

The project therefore combines:

-   Natural Language Processing
-   Transformer-based deep learning
-   Token classification
-   Text classification
-   Sentence embeddings
-   Semantic similarity
-   Clustering
-   Statistical aggregation
-   Model evaluation

------------------------------------------------------------------------

# 5. High-Level Architecture

``` text
                         ┌─────────────────────┐
                         │   Product URL /     │
                         │   Review Dataset    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Review Acquisition  │
                         │ / Dataset Loader    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Preprocessing       │
                         │ Cleaning +          │
                         │ Sentence Splitting │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │       Aspect Extraction          │
                  │      DeBERTa/RoBERTa             │
                  │      Token Classification        │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                         Candidate Aspects
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Aspect Normalization             │
                  │ Sentence Embeddings +           │
                  │ Semantic Similarity / HDBSCAN   │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                       Normalized Components
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Aspect-Level Sentiment           │
                  │ Transformer Classification      │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                    Aspect + Sentiment Pairs
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Evidence Selection               │
                  │ Representative Review Sentences │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Intelligent Aggregation          │
                  │ Scores + Distribution +         │
                  │ Confidence                       │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ Product Intelligence Dashboard  │
                  └──────────────────────────────────┘
```

------------------------------------------------------------------------

# 6. Data Strategy

The Amazon review ecosystem contains datasets that can be extremely
large, potentially reaching hundreds of gigabytes.

We will **not attempt to train directly on the entire dataset**.

The large corpus will be treated as a source from which a suitable
domain-specific dataset is extracted.

## Data pipeline

``` text
Large Amazon Review Corpus
          ↓
Category filtering
          ↓
Electronics
          ↓
Laptop/computer-related reviews
          ↓
Cleaning
          ↓
Deduplication
          ↓
Sentence extraction
          ↓
Training / validation / test subsets
```

The exact dataset size will be decided after inspecting the selected
Amazon dataset.

### Important principle

The project does not require all available reviews.

A carefully selected, clean and well-labelled dataset is more valuable
than blindly training on 100+ GB of noisy data.

------------------------------------------------------------------------

# 7. Training Dataset

The final training dataset should contain examples suitable for two
major NLP tasks.

## Task A --- Aspect Extraction

Example:

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

We use BIO-style sequence labelling:

``` text
B-ASPECT = beginning of an aspect
I-ASPECT = continuation of an aspect
O        = outside an aspect
```

## Task B --- Aspect Sentiment

Example:

``` text
Aspect: display

Text:
"The display is beautiful."

Output:
Positive
```

Another:

``` text
Aspect: battery

Text:
"The battery life is terrible."

Output:
Negative
```

The sentiment classes will initially be:

``` text
Positive
Neutral
Negative
```

A finer-grained sentiment scale may be investigated later.

------------------------------------------------------------------------

# 8. Model Selection

We considered both large language models and smaller Transformer
encoders.

We deliberately decided **not to use a large 1--2B+ model as the primary
model**.

## Why not a large LLM?

A 1--2B parameter model can potentially be adapted using QLoRA on an 8
GB GPU, but it introduces unnecessary complexity for the central ABSA
problem.

Potential disadvantages include:

-   Higher VRAM requirements.
-   Slower experimentation.
-   More complicated training.
-   More complicated deployment.
-   Generative output can be less deterministic.
-   More effort is required to enforce reliable structured outputs.
-   A larger model does not automatically produce better aspect
    extraction.

The goal is to design an effective NLP system, not simply use the
largest available model.

------------------------------------------------------------------------

# 9. Chosen Model Range

The project will focus on **roughly 100--400M parameter Transformer
encoders**.

This range provides a strong balance between:

-   NLP performance
-   training speed
-   memory usage
-   experimentation
-   deployment
-   reproducibility

The available hardware is expected to include an **RTX 4050 Laptop GPU
with 6 GB VRAM** or potentially an **RTX 5050 Laptop GPU with 8 GB
VRAM**.

The architecture should therefore remain comfortably trainable within
this range.

------------------------------------------------------------------------

# 10. Candidate Transformer Models

## BERT-base

Approximately 110M parameters.

Purpose:

-   Classical Transformer baseline.
-   Establish a strong reference point.

Advantages:

-   Extremely well documented.
-   Easy to fine-tune.
-   Large ecosystem.
-   Useful for comparison.

BERT is primarily a **baseline**, rather than necessarily the final
model.

------------------------------------------------------------------------

## RoBERTa-base

Approximately 125M parameters.

Purpose:

-   Stronger BERT-style baseline.
-   Candidate final model.

Advantages:

-   Well-established encoder architecture.
-   Excellent for classification/token-classification tasks.
-   Relatively lightweight.
-   Easily trainable on our target hardware.

------------------------------------------------------------------------

## DeBERTa-v3-base

Approximately 180M parameters.

Purpose:

-   Primary candidate for the final NLP backbone.

Advantages:

-   More modern architecture than BERT.
-   Strong performance on language understanding/classification tasks.
-   Still small enough for practical fine-tuning.
-   Provides a good balance between model capacity and hardware
    requirements.

The current preferred architecture is:

``` text
DeBERTa-v3-base
        ↓
Task-specific heads
        ↓
Aspect extraction / sentiment classification
```

However, the final model will be selected based on actual
validation/test results rather than assuming DeBERTa is automatically
best.

------------------------------------------------------------------------

## DistilBERT

Approximately 66M parameters.

Purpose:

-   Lightweight comparison model.
-   Potential deployment-oriented model.

It can demonstrate the trade-off between:

``` text
Model size
     vs
Accuracy
     vs
Inference speed
```

------------------------------------------------------------------------

# 11. Experimental Model Comparison

Rather than training one model and declaring it successful, we will
benchmark multiple approaches.

Potential experiment:

``` text
TF-IDF + Logistic Regression
          ↓
       Baseline

BERT-base
          ↓
    Transformer baseline

RoBERTa-base
          ↓
      Comparison

DeBERTa-v3-base
          ↓
     Final candidate
```

Evaluation will include:

### Aspect extraction

-   Precision
-   Recall
-   F1-score

### Sentiment classification

-   Accuracy
-   Precision
-   Recall
-   Macro F1
-   Confusion matrix

### System-level considerations

-   Training time
-   GPU memory usage
-   Inference latency
-   Model size

This makes the project experimentally defensible.

------------------------------------------------------------------------

# 12. Aspect Extraction Architecture

The first NLP stage will use **Transformer-based token classification**.

``` text
Review
  ↓
Tokenizer
  ↓
DeBERTa/RoBERTa
  ↓
Contextual token representations
  ↓
Token classification head
  ↓
BIO labels
  ↓
Aspect phrases
```

Example:

``` text
"The keyboard has excellent key travel."

keyboard → B-ASPECT
```

Another:

``` text
"The battery life is surprisingly good."

battery → B-ASPECT
life    → I-ASPECT
```

This approach is preferred over simple keyword matching because it can
learn contextual patterns.

------------------------------------------------------------------------

# 13. Aspect Normalization

Extracted aspect expressions will not always have consistent names.

Examples:

``` text
screen
display
panel
IPS panel
screen quality
```

The normalization layer will use sentence/phrase embeddings.

``` text
Aspect phrase
      ↓
Embedding model
      ↓
Vector representation
      ↓
Semantic similarity
      ↓
Clustering / matching
      ↓
Normalized aspect
```

Potential embedding models:

-   Sentence-BERT family
-   MiniLM
-   MPNet-class sentence encoders

Potential clustering approaches:

-   K-Means
-   DBSCAN
-   HDBSCAN

HDBSCAN is a strong candidate because the number of meaningful aspect
clusters does not have to be fixed beforehand.

------------------------------------------------------------------------

# 14. Aspect-Level Sentiment

After identifying an aspect, the system determines the sentiment
associated with it.

Example:

``` text
Review:
"The screen is fantastic but battery life is disappointing."

Aspect: screen
Sentiment: Positive

Aspect: battery life
Sentiment: Negative
```

The sentiment model will be another Transformer-based classifier.

Conceptually:

``` text
[ASPECT] + review context
          ↓
       Transformer
          ↓
Positive / Neutral / Negative
```

This is fundamentally different from classifying the entire review.

------------------------------------------------------------------------

# 15. Joint ABSA --- Advanced Extension

A possible advanced version is a **joint aspect-sentiment model**.

Instead of independently predicting:

``` text
aspect
   ↓
sentiment
```

the model can directly learn:

``` text
Review
   ↓
Transformer
   ↓
Aspect + sentiment pairs
```

Example:

``` text
"The display is fantastic but the battery is disappointing."

Output:

(display, positive)
(battery, negative)
```

This will be considered after the independent models are working
reliably.

It is an advanced extension rather than a requirement for the initial
MVP.

------------------------------------------------------------------------

# 16. Evidence Extraction

The dashboard should not merely display:

``` text
Battery: 5.4/10
```

It should show why.

For example:

``` text
Battery — 5.4/10

Reviews analyzed: 1,284

Positive: 14%
Neutral: 21%
Negative: 65%

Representative opinions:

• "Battery barely lasts four hours."
• "Gaming drains the battery quickly."
• "Battery life is decent for office work."
```

Representative sentences can be selected using:

-   Sentence embeddings
-   Similarity to the aspect
-   Sentiment confidence
-   Diversity filtering

This creates an interpretable system.

------------------------------------------------------------------------

# 17. Intelligent Aggregation

After processing individual reviews, the system aggregates them.

For every aspect:

``` text
Number of mentions
Positive opinions
Neutral opinions
Negative opinions
Average confidence
Representative evidence
```

A component score can then be calculated.

Conceptually:

``` text
Aspect Score
    =
Weighted sentiment evidence
```

Possible weighting factors:

-   Sentiment confidence
-   Aspect confidence
-   Review quality
-   Duplicate similarity
-   Optional review helpfulness
-   Optional recency

The exact scoring formula will be designed and evaluated experimentally.

------------------------------------------------------------------------

# 18. Example Final Output

For a laptop:

``` text
PRODUCT REVIEW INTELLIGENCE

Display       8.7 / 10
Performance   9.2 / 10
Build         8.1 / 10
Keyboard      7.8 / 10
Thermals      7.1 / 10
Battery       5.2 / 10
Speakers      6.4 / 10
```

Selecting Battery:

``` text
BATTERY — 5.2 / 10

Reviews mentioning battery: 1,284

Positive    14%
Neutral     21%
Negative    65%

Common positive observations:
• Fast charging
• Good idle battery life

Common complaints:
• Short runtime
• High drain during gaming
• Charger required for heavy workloads
```

The UI should also allow the user to inspect supporting reviews.

------------------------------------------------------------------------

# 19. Optional Product Comparison

Once the core system works, products can be compared.

Example:

``` text
                Laptop A    Laptop B

Display           8.7         8.1
Battery           5.2         8.3
Keyboard          7.8         7.2
Performance       9.2         8.5
Thermals          7.1         8.0
Build             8.1         8.4
```

This is an optional extension and should only be implemented after the
core pipeline is stable.

------------------------------------------------------------------------

# 20. Amazon URL Integration

The project can support:

``` text
Amazon Product URL
        ↓
Review acquisition
        ↓
Our trained NLP pipeline
        ↓
Product intelligence report
```

However, **Amazon scraping is not the central research contribution**.

The ML system should also accept a prepared CSV/review collection during
development.

This has several advantages:

-   Development does not depend on live website structure.
-   Model evaluation is reproducible.
-   Training and testing are easier.
-   Changes to Amazon's website do not break the NLP research component.

Any live review acquisition mechanism should comply with the relevant
platform's terms and access restrictions.

------------------------------------------------------------------------

# 21. Training Strategy

The project will use **transfer learning**, not training a Transformer
from scratch.

General process:

``` text
Pretrained Transformer
        ↓
Domain-specific data
        ↓
Fine-tuning
        ↓
Evaluation
        ↓
Best model
```

Hardware target:

``` text
RTX 4050 Laptop GPU — 6 GB VRAM
or
RTX 5050 Laptop GPU — 8 GB VRAM
```

Techniques that may be used:

-   Mixed precision training
-   Gradient accumulation
-   Small per-device batch size
-   Gradient checkpointing if required
-   Sequence length around 128--256 initially
-   Early stopping
-   Learning-rate scheduling

The goal is to maximize experimentation while staying within laptop
hardware limits.

------------------------------------------------------------------------

# 22. Domain Adaptation

A general pretrained model may understand language well but may not be
specialized for consumer electronics.

We therefore want to investigate:

``` text
General pretrained Transformer
              ↓
Electronics review data
              ↓
Laptop-specific adaptation
              ↓
Improved ABSA performance
```

A useful experiment is to compare:

``` text
Generic model
vs
Domain-adapted model
```

This gives the project a meaningful research component.

------------------------------------------------------------------------

# 23. Baselines and Ablation Studies

The final report should ideally include controlled experiments.

Possible comparisons:

### Model comparison

``` text
TF-IDF + Logistic Regression
BERT
RoBERTa
DeBERTa
```

### Data comparison

``` text
General review data
vs
Electronics-only data
vs
Laptop-specific data
```

### Architecture comparison

``` text
Separate aspect + sentiment models
vs
Joint ABSA model
```

### Normalization comparison

``` text
Exact keyword matching
vs
Embedding similarity
vs
Embedding + clustering
```

These experiments help demonstrate which design choices actually improve
the system.

------------------------------------------------------------------------

# 24. Technology Stack

## Machine Learning

-   Python
-   PyTorch
-   Hugging Face Transformers
-   Hugging Face Datasets where appropriate
-   PEFT if parameter-efficient experiments become useful
-   scikit-learn
-   sentence-transformers
-   pandas
-   NumPy

## Backend

-   FastAPI
-   Python
-   PostgreSQL or SQLite

## Frontend

-   React
-   Tailwind CSS
-   Charting library such as Recharts or equivalent

## Development

-   Git
-   GitHub
-   Jupyter notebooks for experiments where useful
-   Configuration-driven training scripts

------------------------------------------------------------------------

# 25. Proposed Repository Structure

``` text
productlens/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── models/
│   ├── aspect_extraction/
│   ├── sentiment/
│   └── embeddings/
│
├── notebooks/
│   ├── data_exploration/
│   ├── model_experiments/
│   └── evaluation/
│
├── src/
│   ├── preprocessing/
│   ├── aspect_extraction/
│   ├── sentiment/
│   ├── normalization/
│   ├── aggregation/
│   └── evidence/
│
├── backend/
│   └── FastAPI application
│
├── frontend/
│   └── React application
│
├── tests/
│
├── configs/
│
├── requirements.txt
│
└── README.md
```

------------------------------------------------------------------------

# 26. Expected Final System

The completed system should be able to:

1.  Accept a product review collection or supported product URL.
2.  Process and clean reviews.
3.  Split reviews into useful textual units.
4.  Extract product aspects.
5.  Normalize synonymous aspect expressions.
6.  Determine aspect-level sentiment.
7.  Calculate confidence.
8.  Aggregate opinions.
9.  Extract representative evidence.
10. Display a component-wise product report.
11. Provide quantitative model evaluation.
12. Optionally compare products.

------------------------------------------------------------------------

# 27. What We Are Explicitly NOT Doing

To keep the project focused, the initial system will not attempt to:

-   Train a multi-billion-parameter model from scratch.
-   Train on the entire 100+ GB review corpus end-to-end.
-   Build a Mixture-of-Experts architecture.
-   Depend entirely on an external LLM API.
-   Build a chatbot just for the sake of adding generative AI.
-   Treat generic review-level sentiment as the primary problem.

The project focuses on **component-level product intelligence**, with
Transformer-based NLP as its core.

------------------------------------------------------------------------

# 28. Team Role Division

There are four team members.

The division is designed so that each member has a substantial technical
responsibility while the modules remain interconnected.

## Member 1 --- NLP / Aspect Extraction

### Responsibilities

-   Study ABSA literature and existing datasets.
-   Prepare aspect-extraction training data.
-   Implement tokenization and BIO labels.
-   Fine-tune BERT/RoBERTa/DeBERTa for aspect extraction.
-   Experiment with sequence length and training parameters.
-   Evaluate precision, recall and F1.
-   Investigate joint aspect extraction as an advanced extension.

### Main deliverables

``` text
Aspect extraction dataset
Aspect extraction model
Training pipeline
Evaluation results
```

------------------------------------------------------------------------

## Member 2 --- NLP / Sentiment + Aspect Intelligence

### Responsibilities

-   Prepare aspect-level sentiment data.
-   Fine-tune Transformer sentiment classifier.
-   Implement aspect normalization.
-   Implement sentence embeddings.
-   Experiment with semantic similarity and clustering.
-   Implement confidence estimation.
-   Work on joint ABSA as an advanced extension.
-   Analyze sentiment errors.

### Main deliverables

``` text
Sentiment model
Embedding/normalization pipeline
Clustering experiments
Evaluation results
```

------------------------------------------------------------------------

## Member 3 --- Data Engineering + Backend

### Responsibilities

-   Investigate and prepare the Amazon review dataset.
-   Build large-dataset filtering pipeline.
-   Handle chunked/streaming processing.
-   Clean and deduplicate reviews.
-   Build database schema.
-   Implement FastAPI backend.
-   Integrate trained models into API endpoints.
-   Handle product/review ingestion.
-   Implement caching where useful.

### Main deliverables

``` text
Processed dataset
Data pipeline
Database
FastAPI backend
Model-serving API
```

------------------------------------------------------------------------

## Member 4 --- Frontend + Visualization + System Integration

### Responsibilities

-   Design the ProductLens dashboard.
-   Build product overview page.
-   Build component score cards.
-   Build sentiment visualizations.
-   Display representative review evidence.
-   Implement aspect drill-down.
-   Implement API integration.
-   Implement optional product comparison.
-   Conduct usability testing.
-   Help with final system integration.

### Main deliverables

``` text
React frontend
Visualization dashboard
Evidence views
API integration
Final user interface
```

------------------------------------------------------------------------

# 29. Shared Responsibilities

All four members should participate in:

-   Dataset understanding
-   Literature review
-   Model evaluation
-   Testing
-   Git/GitHub workflow
-   Documentation
-   Final presentation
-   Project report
-   Demo preparation

The roles describe **primary ownership**, not isolated work.

------------------------------------------------------------------------

# 30. Final Proposed Architecture

The current preferred architecture is:

``` text
                     PRODUCT REVIEWS
                           │
                           ▼
                  DATA PROCESSING
                           │
                           ▼
                 SENTENCE SEGMENTATION
                           │
                           ▼
              ┌────────────────────────┐
              │ DeBERTa-v3-base        │
              │ Aspect Extraction      │
              │ Token Classification   │
              └───────────┬────────────┘
                          │
                          ▼
                  Extracted Aspects
                          │
                          ▼
              ┌────────────────────────┐
              │ Sentence Embeddings    │
              │ Semantic Similarity    │
              │ HDBSCAN / Matching     │
              └───────────┬────────────┘
                          │
                          ▼
                 Normalized Aspects
                          │
                          ▼
              ┌────────────────────────┐
              │ DeBERTa/RoBERTa        │
              │ Aspect Sentiment       │
              │ Classification         │
              └───────────┬────────────┘
                          │
                          ▼
                Aspect-Sentiment Pairs
                          │
                          ▼
              ┌────────────────────────┐
              │ Evidence Selection     │
              │ + Confidence           │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Aggregation Engine     │
              │ Component Scores       │
              └───────────┬────────────┘
                          │
                          ▼
              PRODUCT INTELLIGENCE UI
```

------------------------------------------------------------------------

# 31. Final Design Philosophy

The project follows four principles:

### 1. Model intelligence over model size

A well-designed 100--400M parameter NLP system is preferable to using a
much larger model simply because it has more parameters.

### 2. Modular architecture

Each stage should be independently testable:

``` text
Extraction
Normalization
Sentiment
Aggregation
```

### 3. Evidence-based output

Every component-level conclusion should be traceable to actual review
evidence.

### 4. Measurable performance

The project should be evaluated using proper NLP metrics rather than
only showing a working UI.

------------------------------------------------------------------------

# 32. Project Success Criteria

The project will be considered successful if it can demonstrate:

-   Reliable aspect extraction.
-   Reliable aspect-level sentiment classification.
-   Meaningful normalization of synonymous aspects.
-   Useful component-level aggregation.
-   Evidence-backed product insights.
-   Competitive performance between Transformer architectures.
-   A reproducible training/evaluation pipeline.
-   A functional end-to-end application.

The ultimate objective is:

> **Turn thousands of unstructured product reviews into structured,
> component-level product intelligence that a user can understand at a
> glance.**
