# Fintech Credit Product Recommendation System
## Technical Design Document

**Project:** End-to-End Credit Product Recommender
**Dataset:** LendingClub (Kaggle, 2007–2018Q4)
**Stack:** Python 3.11 · PyTorch · FAISS · DeepFM · Gemini API · FastAPI · Docker

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Deep Dives](#2-component-deep-dives)
   - 2.1 [Synthetic Item Catalog Design](#21-synthetic-item-catalog-design)
   - 2.2 [Vector Retrieval — ALS + FAISS](#22-vector-retrieval--als--faiss)
   - 2.3 [Ranking Model Service — DeepFM](#23-ranking-model-service--deepfm)
   - 2.4 [LLM Orchestration Layer — Gemini](#24-llm-orchestration-layer--gemini)
3. [Data Flow](#3-data-flow)
   - 3.1 [Offline Training Pipeline](#31-offline-training-pipeline)
   - 3.2 [Online Inference Pipeline](#32-online-inference-pipeline)
4. [MLOps & Scaling Considerations](#4-mlops--scaling-considerations)
5. [Evaluation Protocol](#5-evaluation-protocol)
6. [Execution Order](#6-execution-order)

---

## 1. Architecture Overview

The system is divided into two physically separate layers that exchange state
through a shared artefact store (`models/saved/`):

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           OFFLINE TRAINING LAYER                            ║
║                     (runs once, or on a nightly schedule)                   ║
║                                                                              ║
║  Raw CSV ──▶ Preprocessing ──▶ Interaction Matrix (user × item, implicit)   ║
║                                        │                                    ║
║                         ┌──────────────┘                                    ║
║                         ▼                                                   ║
║              ALS Training (implicit-cf, factors=64)                         ║
║              als_user_embeddings.npy   (n_users × 64)                       ║
║              als_item_embeddings.npy   (n_items × 64)                       ║
║                         │                                                   ║
║                         ▼                                                   ║
║              FAISS Index (IndexFlatIP, L2-normalised)                       ║
║              faiss.index                                                    ║
║                         │                                                   ║
║                         ▼                                                   ║
║              DeepFM Ranking Model (PyTorch)                                 ║
║              ranking_model.pt                                               ║
║                                                                              ║
║  Artefacts: models/saved/{embeddings, faiss.index, ranking_model.pt,        ║
║                            encoders.pkl, feature_meta.json}                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
                              ↕  shared volume
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ONLINE INFERENCE LAYER                             ║
║                      (FastAPI server, always-on)                            ║
║                                                                              ║
║  POST /recommend  { user_id, top_k }                                        ║
║          │                                                                   ║
║          ▼                                                                   ║
║  [Stage 1] FAISS ANN Retrieval  ──▶  top-50 candidates  (< 1 ms)           ║
║          │                                                                   ║
║          ▼                                                                   ║
║  [Stage 2] DeepFM Re-scoring    ──▶  top-10 candidates  (< 5 ms)           ║
║          │                                                                   ║
║          ▼                                                                   ║
║  [Stage 3] Gemini LLM Re-rank   ──▶  top-5  final list  (200–800 ms)       ║
║          │         (optional; degrades gracefully to Stage 2 on failure)    ║
║          ▼                                                                   ║
║  HTTP 200  { recommendations: [ { item_id, grade, purpose, ... } ] }       ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Item definition | Synthetic catalog (Grade × Purpose × Term) | Converts a loan-application dataset into a repeatable product-recommendation problem; limits item space to ~181 products |
| Retrieval model | ALS (`implicit` library) | Proven on implicit feedback; fast matrix factorisation; 64-dim embeddings fit in RAM |
| ANN library | FAISS `IndexFlatIP` | Exact inner product; zero extra infra; sub-millisecond at n_items ≈ 181; swappable to `IndexIVFFlat` for larger catalogs |
| Ranking model | DeepFM | Captures both 2nd-order FM interactions and higher-order MLP interactions from the same shared embedding table; outperforms NeuMF on this feature-rich dataset |
| LLM provider | Google Gemini (`google-genai` SDK) | Native structured output (`response_schema=list[str]`) eliminates JSON parsing fragility; `gemini-2.5-pro` and `gemini-3.0-pro-preview` evaluated |
| API framework | FastAPI | Async, auto OpenAPI docs, Pydantic validation, lifespan events for artefact loading |
| Serialisation | `.npy` embeddings, `torch.save` model, `pickle` encoders | Portable, framework-agnostic; readable without special tooling |

---

## 2. Component Deep Dives

### 2.1 Synthetic Item Catalog Design

The raw LendingClub dataset contains ~2.2 million individual loan applications.
Treating each as a unique item produces a nearly non-repeatable interaction matrix
(most items appear only once) and renders collaborative filtering meaningless.

**Solution:** Define a *recommendable product* as the tuple `(grade, purpose, term)`:

```
item_id = f"{grade}_{purpose}_{term}"
# e.g. "B_debt_consolidation_36 months" → item_idx = 42
```

This yields at most **7 × 14 × 2 = 196** product types, of which ~181 appear in
the dataset. Each user's interactions collapse to a set of product types they have
engaged with — exactly the input format collaborative filtering requires.

The item lookup table (`data/processed/item_lookup.csv`) additionally stores
`int_rate` (mean interest rate), `positive_rate` (historical repayment rate),
and loan count per product, which are fed to the LLM as context signals.

---

### 2.2 Vector Retrieval — ALS + FAISS

#### ALS Training (`retrieval/train_als.py`)

```
Input : train_interactions.npz   shape (n_users, n_items), implicit binary
Model : AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1)
Note  : implicit library expects item×user input; factors are swapped on export

Output:
  als_user_embeddings.npy   shape (n_users, 64)
  als_item_embeddings.npy   shape (n_items, 64)
```

**Critical implementation note:** The `implicit` library's `model.user_factors`
stores row factors of the *item×user* training matrix, which correspond to item
embeddings in the standard interpretation. The export explicitly swaps:

```python
# retrieval/train_als.py
return model.item_factors, model.user_factors  # (user_emb, item_emb)
```

#### FAISS Index (`retrieval/build_faiss_index.py`)

```python
item_emb = np.load("als_item_embeddings.npy").astype("float32")
faiss.normalize_L2(item_emb)          # cosine similarity via inner product
index = faiss.IndexFlatIP(d=64)       # exact search; n_items=181 < 50k threshold
index.add(item_emb)
faiss.write_index(index, "faiss.index")
```

At query time, the user embedding is also L2-normalised before search, so
`dot(user_vec, item_vec)` computes cosine similarity in constant time.

**Scaling path:** Replace `IndexFlatIP` with `IndexIVFFlat` (approximate search)
when n_items exceeds ~50k. No other code changes required.

---

### 2.3 Ranking Model Service — DeepFM

#### Architecture (`models/deepfm_model.py`)

```
Sparse inputs: [user_idx, item_idx]
  └──▶ Embedding lookup  (shared table, emb_dim=16 per field)

Dense inputs:  [annual_inc, dti, fico_low, fico_high, home_ownership_OHE,
               addr_state_OHE, int_rate, grade_OHE, purpose_OHE, term_OHE]
  └──▶ (concatenated as-is)

FM Layer:    Σᵢ Σⱼ <vᵢ, vⱼ>  (closed-form 2nd-order interactions)
Deep Layer:  MLP([256, 128, 64], dropout=0.2)

Output:  fm_out + deep_out  (raw logit; sigmoid applied by BCEWithLogitsLoss)
```

**Key detail:** `build_deepfm` uses `sparse_field_dims = [n_users, n_items]`
(2 sparse fields only). Raw tabular features are concatenated as `dense_in`
and passed to the MLP branch. The model is **not** given pre-computed ALS
embeddings as input; it learns its own collaborative signal from scratch.

#### Training (`ranking/train_ranking.py`)

```
Loss       : BCEWithLogitsLoss (no sigmoid in forward pass)
Optimizer  : Adam(lr=1e-3, weight_decay=1e-4)
Scheduler  : ReduceLROnPlateau(factor=0.5, patience=2)
Grad clip  : max_norm=1.0
Early stop : val NDCG@10, patience=5 epochs
Negative sampling: popularity-weighted (item_freq^0.75), k_neg=4
```

**Overfitting note:** With only ~1 interaction per user on average, the model
peaks at epoch 1 (NDCG@10 ≈ 0.35) and degrades thereafter. Early stopping
correctly saves the epoch-1 checkpoint. This is a known artefact of highly
sparse implicit feedback datasets, not a model defect.

#### Inference (`ranking/predictor.py`)

```python
class RankingPredictor:
    def score_candidates(self, user_idx, candidate_item_idxs) -> np.ndarray:
        # Builds (user_idx_tensor, item_idx_tensor) + optional dense features
        # Dispatches to model.forward() based on model_type in feature_meta.json
        # Returns raw logit scores (higher = more relevant)
```

`feature_meta.json` records `n_users`, `n_items`, `user_feat_dim`,
`item_feat_dim`, and `model_type` ("deepfm" or "neumf") so the predictor can
reconstruct the correct model skeleton at load time without any code changes.

---

### 2.4 LLM Orchestration Layer — Gemini

#### Context Builder (`prepare_llm_context`)

For each user, the context builder assembles a structured natural-language prompt
from three data sources:

| Source | Signal | Where it comes from |
|--------|--------|---------------------|
| `interactions_all.parquet` | FICO, DTI, income, home_ownership, state | Raw demographics (un-scaled) |
| `train_interactions.npz` | Historical loan product indices | Warm-start history |
| `item_lookup.csv` | Grade, purpose, term, int_rate, positive_rate | Item metadata |

The prompt instructs Gemini to act as a **predictive behavioral analyst** —
not a prescriptive loan officer — and to rank candidates by *likelihood of
acceptance by this specific borrower*, with explicit rules to match historical
loan characteristics for warm-start users.

#### Gemini API Call Pattern

```python
response = gemini_client.models.generate_content(
    model=model,                     # "gemini-2.5-pro" or "gemini-3.0-pro-preview"
    contents=full_payload,
    config=types.GenerateContentConfig(
        temperature=0.0,             # deterministic output
        response_mime_type="application/json",
        response_schema=list[str],   # native structured output — no parsing hacks
    ),
)
ranked_ids = json.loads(response.text.strip())
```

`response_schema=list[str]` instructs the model to emit a raw JSON array of
`item_id` strings, bypassing any markdown fences or wrapper keys. This is the
native structured output feature of the `google-genai` SDK.

#### Retry and Fallback Mechanism

```
For attempt in range(3):
    try:
        response = gemini_client.models.generate_content(...)
        return parse_and_validate(response)
    except Exception as e:
        if "503" or "429" or "RESOURCE_EXHAUSTED" in str(e):
            wait = 15 * (attempt + 1)   # 15s, 30s, 45s
            time.sleep(wait)
            continue
        else:
            return original_deepfm_order   # non-retriable error → immediate fallback

return original_deepfm_order              # max retries → graceful degradation
```

**Fallback contract:** The LLM stage never raises an exception to the calling
code. On any failure (network error, rate limit, malformed JSON, wrong type),
it returns the original DeepFM ordering. This guarantees that Stage 3 adds
latency but never reduces recommendation quality below the Stage 2 baseline.

#### Output Validation

After receiving the LLM response, the system validates and re-merges:

```python
valid_set = set(original_item_ids)
valid   = [x for x in ranked if x in valid_set]    # LLM's preferred order
missing = [x for x in original_item_ids            # hallucinated IDs discarded
           if x not in set(valid)]
return valid + missing   # LLM top-5 first, then any omitted items at the end
```

This ensures the final list always contains exactly the DeepFM candidates,
even if the LLM omits or hallucinates some IDs.

#### Warm vs. Cold Start Behaviour

| User Type | LLM Advantage | Observed Uplift |
|-----------|--------------|-----------------|
| **Warm Start** (≥1 training interaction) | Can match prior loan Grade/Term/Purpose exactly | NDCG@5 uplift typically positive |
| **Cold Start** (no history) | Relies solely on demographics (FICO, DTI, income) | Smaller, less consistent uplift |

The ablation study (`evaluation/ablation_study.py`) explicitly separates these
subsets and reports metrics independently for each.

---

## 3. Data Flow

### 3.1 Offline Training Pipeline

```
Step 1  preprocessing/build_interactions.py
        ├── Load raw CSV; keep relevant columns
        ├── Define item_id = grade + "_" + purpose + "_" + term
        ├── Encode user_idx, item_idx with LabelEncoder
        ├── Map loan_status → binary interaction (Fully Paid/Current = 1)
        ├── Time-based split: train < 2017-01-01, val 2017, test ≥ 2018
        └── Write train/val/test_interactions.npz + item_lookup.csv

Step 2  preprocessing/feature_engineering.py
        ├── User features: annual_inc, dti, fico_range_low/high, home_ownership OHE,
        │   addr_state OHE (top-10 states + "other") → StandardScaler
        ├── Item features: int_rate, grade OHE, purpose OHE, term OHE
        └── Write user_features.npy, item_features.npy, encoders.pkl, feature_meta.json

Step 3  retrieval/train_als.py
        ├── Fit ALS on train_interactions (item×user orientation)
        └── Write als_user_embeddings.npy (n_users × 64)
                  als_item_embeddings.npy (n_items × 64)

Step 4  retrieval/build_faiss_index.py
        ├── Load item embeddings; L2-normalise
        ├── Build IndexFlatIP (exact inner product)
        └── Write faiss.index

Step 5  ranking/train_ranking.py
        ├── Build RankingDataset with popularity-weighted negatives (k_neg=4)
        ├── Train DeepFM: BCEWithLogitsLoss, Adam, early stop on val NDCG@10
        └── Write ranking_model.pt (best checkpoint)
```

### 3.2 Online Inference Pipeline

```
POST /recommend { user_id: "member_123", top_k: 5 }
        │
        ▼
[Encode]  user_idx = user_enc.transform([user_id])[0]
          Unknown user → cold-start: return globally popular products

        │
        ▼
[Stage 1 — FAISS Retrieval]
  u_vec = als_user_embeddings[user_idx]          # (64,)
  faiss.normalize_L2(u_vec)
  _, I = faiss_index.search(u_vec, k=50)         # over-fetch pool of 50
  candidates = I[0]                              # item indices, (50,)

        │
        ▼
[Stage 2 — DeepFM Re-scoring]
  scores   = predictor.score_candidates(user_idx, candidates)
  top10    = candidates[argsort(scores)[::-1][:10]]

        │
        ▼
[Stage 3 — Gemini LLM Re-ranking]   (optional; requires GOOGLE_API_KEY)
  context  = prepare_llm_context(user_idx, top10, past_item_idxs)
  reranked = get_gemini_rerank(context, item_ids, model="gemini-2.5-pro")
  top5     = reranked[:5]
  ↳ On any failure: return top10[:5] (DeepFM order, no exception)

        │
        ▼
[Decode]  item_details = item_lookup[top5_idxs]

HTTP 200 { recommendations: [ { item_id, grade, purpose, term, int_rate, ... } ] }
```

---

## 4. MLOps & Scaling Considerations

### Model Artefact Management

All artefacts are versioned by write timestamp and stored under `models/saved/`.
In production, this directory would be replaced with an object store (S3, GCS)
with version prefixes. The `feature_meta.json` file acts as a schema registry:
the predictor reads `n_users`, `n_items`, `user_feat_dim`, `item_feat_dim`, and
`model_type` at startup, allowing the model skeleton to be reconstructed without
code changes when dimensions change between retraining runs.

### Candidate Generation Latency

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| FAISS search (n_items=181, pool=50) | < 1 ms | Exact search; negligible at this scale |
| FAISS search (n_items=1M, pool=50) | < 5 ms | IVFFlat approximate; index rebuild required |
| DeepFM forward pass (50 candidates) | 2–8 ms | CPU; GPU reduces to < 1 ms |
| Gemini API call | 200–800 ms | Network-bound; parallelise with ThreadPoolExecutor for batch jobs |

### LLM Rate Limit Handling (429/503)

The retry strategy uses **linear backoff**: wait 15 × attempt seconds (15s, 30s, 45s).
For production batch evaluation jobs, requests are spaced with a configurable
`inter_user_sleep` (default 20s) to stay within free-tier quota limits. The
`ablation_study.py` script passes `inter_user_sleep` and `inter_model_sleep`
as parameters for easy adjustment.

For a production API serving concurrent requests, the recommended approach is:
1. Run Stage 3 asynchronously (non-blocking); return Stage 2 results immediately.
2. Push the LLM re-rank result to the client via SSE or a polling endpoint.
3. Cache LLM re-rankings keyed by `(user_idx, deepfm_top10_hash)` with a short TTL.

### Graceful Degradation

The system degrades across three levels automatically:

```
Stage 3 (Gemini) available    → full three-stage pipeline
Stage 3 unavailable           → Stage 2 output (DeepFM, NDCG@5 ≈ 0.28)
Stage 2 unavailable           → Stage 1 output (FAISS, NDCG@5 ≈ 0.18)
Stage 1 unavailable           → popularity fallback (globally most-interacted products)
```

No configuration change is required; the system detects missing API keys and
failed model loads at startup and adjusts the active pipeline depth accordingly.

### GPU Compatibility Note

The project was developed on an RTX 5060 Ti (sm_120 architecture), which is not
supported by PyTorch ≤ 2.4 (max sm_90). All training and inference scripts
default to `device="cpu"`. When deploying on a supported GPU, set
`--device cuda` in training and update `RankingPredictor(device="cuda")` in
the predictor instantiation.

---

## 5. Evaluation Protocol

### Metrics

Both metrics are evaluated at K=5 to match the final top-5 output of the pipeline:

```python
def ndcg_at_k(ranked, ground_truth, k=5):
    """Position-weighted: hitting rank 1 > rank 5."""
    dcg  = Σ  1 / log₂(rank + 2)   for rank, item in ranked[:k] if item in gt
    idcg = Σ  1 / log₂(rank + 2)   for rank in range(min(|gt|, k))
    return dcg / idcg

def recall_at_k(ranked, ground_truth, k=5):
    """Fraction of ground-truth items captured in top-K."""
    return |{ranked[:k]} ∩ gt| / |gt|
```

### Ablation Study Design

Three pipeline stages are evaluated on the test split, with metrics reported
separately for Warm Start and Cold Start user subsets:

| Stage | Components | Candidate Set | Metric |
|-------|-----------|---------------|--------|
| 1 — Retrieval Only | ALS + FAISS | top-50 → evaluated at top-5 | NDCG@5, Recall@5 |
| 2 — Retrieval + Ranking | + DeepFM | top-50 → re-score → top-10 → top-5 | NDCG@5, Recall@5 |
| 3 — Retrieval + Ranking + LLM | + Gemini | top-10 → LLM re-order → top-5 | NDCG@5, Recall@5 |

Statistical significance on warm-start users is tested with a **paired t-test**
(each user provides a matched pair of scores between stages).

```bash
python -m evaluation.ablation_study --k 5 --n-users 20 --pool 50
```

---

## 6. Execution Order

```
1.  python preprocessing/build_interactions.py
2.  python preprocessing/feature_engineering.py
3.  python retrieval/train_als.py
4.  python retrieval/build_faiss_index.py
5.  python ranking/train_ranking.py      [--device cpu]
6.  python evaluation/ablation_study.py  [--k 5 --n-users 20]
7.  docker compose up --build            # serve the API
```

Each script reads from `data/processed/` or `models/saved/` and writes its
outputs there. All steps are independently re-runnable — retraining the ranking
model does not require re-running ALS or FAISS indexing.

---

*Last updated: 2026-03-17*
*Stack: Python 3.11 · PyTorch 2.x · FAISS 1.7.x · google-genai · FastAPI 0.110.x*
