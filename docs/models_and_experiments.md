# ML Models, Training Strategy & Experiments

This document details the model selection rationale, candidate Transformer architectures, experimental methodology, training configurations, and ablation studies for **ProductLens**.

---

## 1. Model Selection Rationale

We deliberately prioritized **compact Transformer encoders (100M–400M parameters)** over multi-billion parameter Large Language Models (LLMs).

### Why Not a Large LLM (1B–7B+)?

While large generative models can be adapted via parameter-efficient fine-tuning (e.g., QLoRA), they introduce substantial drawbacks for core Aspect-Based Sentiment Analysis (ABSA):
- **Excessive VRAM Footprint**: Restricts rapid experimentation on consumer/laptop GPUs.
- **Inference Latency**: Autoregressive decoding is orders of magnitude slower than bidirectional encoder classification over thousands of review sentences.
- **Nondeterministic Generations**: Hallucinations and inconsistent formatting require complex constrained generation wrappers.
- **Inefficient Token Classification**: Encoder-only models (e.g., DeBERTa) naturally excel at dense token-level representations for BIO extraction compared to generative decoders.

The goal is to design an efficient, defensible, and high-accuracy NLP system rather than relying on bloated model scale.

---

## 2. Target Hardware & Parameter Range

- **Model Capacity**: Roughly **100M to 400M parameters**.
- **Target Hardware**:
  - NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)
  - NVIDIA RTX 5050 Laptop GPU (8 GB VRAM)
- **Design Requirement**: Full fine-tuning and inference pipelines must run comfortably within 6–8 GB VRAM constraints.

---

## 3. Candidate Transformer Architectures

| Model | Parameters | Role | Key Strengths |
| :--- | :--- | :--- | :--- |
| **BERT-base** (`bert-base-uncased`) | ~110M | Standard Baseline | Classical Transformer benchmark; highly documented and stable reference point. |
| **RoBERTa-base** (`roberta-base`) | ~125M | Intermediate Baseline | Removed Next Sentence Prediction (NSP), dynamic masking, trained on larger corpus. |
| **DeBERTa-v3-base** (`microsoft/deberta-v3-base`) | ~180M | Primary Candidate | Disentangled attention mechanism, enhanced mask decoder; state-of-the-art NLU benchmark performance. |
| **DistilBERT** (`distilbert-base-uncased`) | ~66M | Lightweight / Edge Benchmark | 40% smaller, 60% faster than BERT; measures accuracy-latency trade-offs for production deployment. |

### Preferred Primary Backbone

``` text
DeBERTa-v3-base
       ↓
Shared Contextual Representations
       ↓
Task-Specific Linear / Dropout Heads
       ↓
[Aspect Extraction BIO Tags]  &  [Aspect-Level Sentiment Class]
```

Final selection will be determined empirically based on test-set metrics across architectures.

---

## 4. Experimental Benchmark Framework

Rather than training a single architecture in isolation, ProductLens evaluates a progression of models:

``` text
1. TF-IDF + Logistic Regression  ──> Classical Baseline
2. BERT-base                     ──> Canonical Transformer Baseline
3. DistilBERT                    ──> Efficiency / Latency Baseline
4. RoBERTa-base                  ──> Optimized Encoder Baseline
5. DeBERTa-v3-base               ──> Primary SOTA NLU Candidate
```

### Evaluation Metrics

#### Task A: Aspect Extraction (Token Classification)
- **Token & Entity-Level Precision**
- **Token & Entity-Level Recall**
- **Strict & Relaxed Micro/Macro F1-Score**

#### Task B: Aspect-Level Sentiment Classification
- **Classification Accuracy**
- **Macro-Averaged Precision & Recall**
- **Macro F1-Score** (critical due to class imbalance between Neutral, Positive, and Negative)
- **Confusion Matrix Analysis** (specifically monitoring Neutral vs. Weak Positive/Negative confusion)

#### System & Production Considerations
- **Peak GPU VRAM Usage (MB)**
- **Training Epoch Duration (s)**
- **Inference Latency per 100 Sentences (ms)**
- **Model Storage Footprint (MB)**

---

## 5. Training Strategy & Optimization

Training relies on **supervised transfer learning** from pretrained Hugging Face checkpoints.

``` text
Pretrained Transformer Checkpoint
                ↓
    Domain-Adapted Weights (Optional)
                ↓
     Supervised Fine-Tuning
     (Aspect Extraction / Sentiment)
                ↓
  Validation Metric Checkpointing & Early Stopping
                ↓
         Final Production Model
```

### Resource Optimization Techniques for 6–8 GB VRAM:
- **Mixed Precision**: Automatic Mixed Precision (AMP `fp16` or `bf16`) reduces activation memory by ~50%.
- **Gradient Accumulation**: Enables effective batch sizes of 32 or 64 while maintaining small per-device physical batches (e.g., 8 or 16).
- **Sequence Length Capping**: Sentences truncated to 128–256 tokens (covers >95% of individual review sentences).
- **Gradient Checkpointing**: Available if peak activation memory approaches VRAM boundaries.
- **Optimization Algorithms**: AdamW with linear or cosine warmup schedule; weight decay of $0.01$.

---

## 6. Domain Adaptation (Electronics / Laptops)

General pretrained checkpoints are trained on Wikipedia and BookCorpus. To bridge domain divergence:

``` text
Generic Pretrained Transformer (e.g., DeBERTa-v3-base)
                       ↓
   Unsupervised Masked Language Modeling (MLM)
        on Laptop & Electronics Reviews Corpus
                       ↓
         Domain-Specialized Encoder
                       ↓
   Downstream ABSA Supervised Fine-Tuning
```

### Scientific Experimentation
Compare performance between:
1. **Generic Pretrained Encoder** + ABSA Head
2. **Domain-Adapted Encoder (MLM)** + ABSA Head

---

## 7. Baselines and Ablation Studies

Controlled ablations validate that each architectural choice contributes meaningfully to performance:

### 1. Model Architecture Ablations
- Classical ML (TF-IDF + LR) vs. DistilBERT vs. BERT vs. RoBERTa vs. DeBERTa-v3.

### 2. Dataset Domain Ablations
- General multi-category Amazon reviews vs. Electronics reviews vs. Laptop-filtered reviews.

### 3. Structural ABSA Ablations
- Two-stage pipeline (Independent Extraction $\rightarrow$ Normalization $\rightarrow$ Sentiment) vs. Single-stage Joint Span Tagging.

### 4. Normalization Strategy Ablations
- Exact keyword matching vs. Embedding Cosine Similarity vs. Unsupervised Density Clustering (HDBSCAN).
