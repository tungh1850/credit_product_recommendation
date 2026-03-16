# Fintech Credit Product Recommendation System

An end-to-end, multi-stage ML recommendation pipeline for financial credit products,
built on the LendingClub dataset. The system combines collaborative filtering,
deep feature-interaction ranking, and zero-shot LLM re-ranking to recommend
loan products that are both relevant to the borrower and aligned with credit risk principles.

**Stack:** Python · PyTorch · FAISS · DeepFM · Gemini API · FastAPI · Docker

---

## Problem Statement

Recommending financial products is fundamentally different from recommending movies
or e-commerce items. Two challenges make it uniquely hard:

1. **Credit risk constraints.** Offering an unsuitable loan to a high-risk borrower
   (low FICO, high DTI) is not just a relevance failure — it can cause real harm.
   A purely collaborative signal ("users like you also applied for X") ignores
   the borrower's current financial health.

2. **Sparse, implicit feedback.** Borrowers don't rate loans. The only signal is
   behavioural: a repaid or currently active loan is a positive interaction; a
   charge-off or default is a negative one. This makes the recommendation problem
   an implicit feedback problem with a heavily skewed positive/negative ratio.

### The Synthetic Item Catalog

Rather than treating each of the ~2.2 million historical loan applications as a
unique item — which would produce a nearly non-repeatable interaction matrix —
this project defines a **recommendable product** as a unique combination of
**(Grade, Purpose, Term)**. For example:

> `B_debt_consolidation_36 months` → `item_idx = 42`

This maps the LendingClub dataset into a retail-banking scenario: a lender
recommends **types** of credit products to new borrowers, not past individual loans.
The approach constrains the item space to **~181 distinct product configurations**
(7 grades × 14 purposes × 2 terms), making the recommendation problem tractable
while preserving the meaningful financial attributes that drive credit decisions.

---

## Pipeline Architecture

The system implements a classic **retrieval → ranking → re-ranking funnel**,
where each stage narrows the candidate pool and refines the ordering:

```
Raw CSV (LendingClub)
    │
    ▼
Preprocessing            ← binary interactions, time-based split, feature engineering
    │
    ├──▶ ALS Training    ← factorises the implicit interaction matrix (factors=64)
    │         │
    │         ▼
    │    FAISS Index     ← L2-normalised inner product (cosine similarity)
    │                       top-50 candidates in < 1 ms per user
    │
    ├──▶ DeepFM Ranking  ← re-scores FAISS pool using user × item feature interactions
    │         │              (FICO, DTI, grade, purpose, term, income, home_ownership)
    │         ▼             keeps top-10 candidates
    │
    └──▶ Gemini LLM      ← zero-shot re-ranking using natural language reasoning
              │              models: gemini-2.5-pro / gemini-3.0-pro-preview
              ▼
         Final top-5     ← evaluated at NDCG@5 and Recall@5
```

### Stage 1 — Retrieval (ALS + FAISS)

Alternating Least Squares factorises the sparse user×item interaction matrix into
64-dimensional embeddings. User and item vectors are L2-normalised and indexed with
`faiss.IndexFlatIP` for exact inner product search. For this item catalog size
(n_items ≈ 181), exact search is computationally trivial; the FAISS architecture
is retained to demonstrate production-readiness at scale.

### Stage 2 — Ranking (DeepFM)

A DeepFM model re-scores the FAISS candidates by jointly learning second-order
feature interactions (Factorisation Machine layer) and higher-order interactions
(deep MLP). The FM and MLP branches share the same embedding table, and the model
is trained with `BCEWithLogitsLoss` on popularity-weighted negative samples
(item frequency^0.75) to reduce popularity bias.

### Stage 3 — LLM Re-ranking (Gemini)

The top-10 DeepFM candidates are passed to Gemini with a structured prompt that
includes the borrower's FICO score, DTI, income, home ownership, state, and their
full history of previously accepted loans. Gemini reasons in natural language about
credit suitability and returns a re-ordered list of item IDs via
`response_mime_type="application/json"` with `response_schema=list[str]` (native
structured output — no prompt hacking required).

Borrowers are classified as:
- **Warm Start** — ≥ 1 interaction in the training matrix; Gemini can exploit
  historical loan pattern matching.
- **Cold Start** — no training history; Gemini falls back to pure credit risk
  reasoning from demographics.

---

## Key Findings

| Stage | All Users NDCG@5 | All Users Recall@5 | Warm Start NDCG@5 | Cold Start NDCG@5 |
|-------|:-:|:-:|:-:|:-:|
| Stage 1 — Retrieval Only | ~0.18 | ~0.14 | ~0.21 | ~0.12 |
| Stage 2 — Retrieval + Ranking (DeepFM) | ~0.28 | ~0.22 | ~0.32 | ~0.19 |
| Stage 3 — + LLM (best model) | ~0.31 | ~0.25 | ~0.38 | ~0.18 |

> **Note:** Numbers above are indicative of relative improvement trends;
> exact values depend on the random user sample and data version used.
> Run `python -m evaluation.ablation_study` to reproduce on your artefacts.

**Key observations:**

- **DeepFM over Retrieval** is the largest single uplift, driven by its ability
  to capture FICO × grade and DTI × purpose interactions that ALS cannot model.
- **LLM uplift is concentrated on Warm Start users.** When Gemini can match
  a borrower's prior loan characteristics (Grade, Term, Purpose), it consistently
  promotes the most likely accepted products into the top-5. For Cold Start users
  with no history, the LLM has only demographics to reason from and its uplift
  is smaller and less consistent.
- **Graceful degradation:** On any API failure (429 rate limit, 503 service
  unavailable), the system falls back to the DeepFM ranking silently, ensuring
  zero recommendation latency regression.

---

## Quick Start

### Prerequisites

```bash
# Conda (recommended — resolves PyTorch + FAISS native deps)
conda env create -f environment.yml
conda activate credit_recsys

# Or pip
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (never commit secrets):

```bash
GOOGLE_API_KEY=AIza...          # Required for LLM re-ranking (Stage 3)
```

### 1. Download LendingClub Data

```bash
bash scripts/download_data.sh   # requires ~/.kaggle/kaggle.json
```

Place `accepted_2007_to_2018Q4.csv` in `data/raw/`.

### 2. Run the Offline Training Pipeline

```bash
bash scripts/run_pipeline.sh
```

This sequentially executes:

| Step | Script | Output |
|------|--------|--------|
| 1 | `preprocessing/build_interactions.py` | `train/val/test_interactions.npz` |
| 2 | `preprocessing/feature_engineering.py` | `user_features.npy`, `item_features.npy`, `encoders.pkl` |
| 3 | `retrieval/train_als.py` | `als_user_embeddings.npy`, `als_item_embeddings.npy` |
| 4 | `retrieval/build_faiss_index.py` | `faiss.index` |
| 5 | `ranking/train_ranking.py` | `ranking_model.pt` |

### 3. Run the Ablation Study

```bash
# Default: k=5, 20 users (warm-start prioritised), pool=50 FAISS candidates
python -m evaluation.ablation_study

# Custom
python -m evaluation.ablation_study --k 5 --n-users 100 --pool 50
```

Results are printed to stdout and saved to `evaluation/ablation_results.csv`.

### 4. Start the API

```bash
# Docker (recommended)
docker compose up --build

# Or locally
uvicorn api.main:app --reload --port 8000
```

### 5. Make a Recommendation Request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "member_123456", "top_k": 5}'
```

Interactive API docs: http://localhost:8000/docs

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
recommendation_system/
├── preprocessing/        Data loading, interaction matrix, feature engineering
├── models/               ALS (PyTorch), NeuMF, DeepFM model definitions
├── retrieval/            ALS training script + FAISS index builder
├── ranking/              Dataset, training loop (DeepFM), inference predictor
├── api/                  FastAPI app, schemas, recommender logic, LLM reranker
├── evaluation/           Recall@K, NDCG@K, ablation study (3-stage comparison)
├── tests/                pytest unit + integration tests
├── notebooks/            01_full_pipeline.ipynb — end-to-end experiment notebook
├── scripts/              run_pipeline.sh, download_data.sh
├── Dockerfile
├── docker-compose.yml
├── environment.yml
└── requirements.txt
```

See [DESIGN.md](DESIGN.md) for the full system architecture and component deep-dives.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe — confirms all artefacts are loaded |
| `POST` | `/recommend` | Top-K loan recommendations for a user |
| `GET`  | `/items` | List all known loan product types |
| `GET`  | `/users/{user_id}` | Check if a user is in the training set |
