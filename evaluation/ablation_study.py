"""
evaluation/ablation_study.py
-----------------------------
Compares three pipeline configurations side-by-side on the test split,
mirroring the evaluation logic in notebooks/01_full_pipeline.ipynb §13 / §13.5.

  Stage 1 — Retrieval Only        : ALS + FAISS top-K direct
  Stage 2 — Retrieval + Ranking   : ALS + FAISS → DeepFM re-score → top-K
  Stage 3 — Retrieval + Ranking + LLM : Stage 2 → Gemini re-rank → top-K
             (Stage 3 requires GOOGLE_API_KEY in the environment)

Metrics reported per stage, for ALL / Warm-Start / Cold-Start user subsets:
  • NDCG@K   (Normalised Discounted Cumulative Gain)
  • Recall@K (fraction of relevant items in top-K)

Warm Start  = user has ≥ 1 interaction in training matrix
Cold Start  = user has no interaction history (embedding may be out-of-vocabulary)

Usage:
  python -m evaluation.ablation_study [--k 5] [--n-users 20] [--pool 50]
  python -m evaluation.ablation_study --k 5 --n-users 100 --pool 50
"""

import os
import json
import math
import time
import pickle
import argparse
import warnings
import numpy as np
import scipy.sparse as sp
import faiss
import pandas as pd
from typing import List, Set, Tuple, Optional
from google import genai
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

SAVED_DIR     = os.path.join("models", "saved")
PROCESSED_DIR = os.path.join("data", "processed")
OUT_CSV       = os.path.join("evaluation", "ablation_results.csv")

# Gemini models to evaluate in Stage 3
GEMINI_MODEL_A = "gemini-2.5-pro"
GEMINI_MODEL_B = "gemini-3.1-pro-preview"


# =============================================================================
# Metric helpers  (mirrors notebook ndcg_at_k / recall_at_k)
# =============================================================================

def ndcg_at_k(ranked: List[int], gt: Set[int], k: int) -> float:
    if not gt:
        return 0.0
    dcg  = sum(1.0 / math.log2(r + 2)
               for r, item in enumerate(ranked[:k]) if item in gt)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gt), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: List[int], gt: Set[int], k: int) -> float:
    if not gt:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in gt)
    return hits / len(gt)


def _agg(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


# =============================================================================
# Artefact loading
# =============================================================================

def load_artefacts():
    from ranking.predictor import RankingPredictor

    print("[ablation] Loading artefacts …")
    with open(os.path.join(SAVED_DIR, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)

    user_emb    = np.load(os.path.join(SAVED_DIR, "als_user_embeddings.npy"))
    faiss_index = faiss.read_index(os.path.join(SAVED_DIR, "faiss.index"))
    predictor   = RankingPredictor(
        model_path=os.path.join(SAVED_DIR, "ranking_model.pt"),
        meta_path =os.path.join(PROCESSED_DIR, "feature_meta.json"),
    )
    train_mat   = sp.load_npz(os.path.join(PROCESSED_DIR, "train_interactions.npz"))
    test_mat    = sp.load_npz(os.path.join(PROCESSED_DIR, "test_interactions.npz"))
    item_lookup = pd.read_csv(os.path.join(PROCESSED_DIR, "item_lookup.csv"))

    # Raw parquet for user demographics (FICO / income / DTI)
    raw_parquet = os.path.join(PROCESSED_DIR, "interactions_all.parquet")
    if os.path.exists(raw_parquet):
        raw_df = pd.read_parquet(raw_parquet)
        USER_COLS = ["annual_inc", "dti", "fico_range_low", "fico_range_high",
                     "home_ownership", "addr_state"]
        user_profiles_raw = (
            raw_df.sort_values("issue_d")
            .groupby("member_id")
            .last()[USER_COLS]
            .to_dict("index")
        )
    else:
        user_profiles_raw = {}
        print("[ablation] WARNING: interactions_all.parquet not found — "
              "user demographics unavailable for LLM context.")

    print(f"[ablation] Artefacts loaded. "
          f"n_users={user_emb.shape[0]:,}  n_items={item_lookup.shape[0]}")
    return enc, user_emb, faiss_index, predictor, train_mat, test_mat, \
           item_lookup, user_profiles_raw


# =============================================================================
# User sampling: warm / cold split
# =============================================================================

def sample_users(
    test_mat: sp.csr_matrix,
    train_mat: sp.csr_matrix,
    n_users: int,
    seed: int = 42,
) -> Tuple[List[Tuple[int, str]], int, int]:
    """
    Sample test users, tagging each as 'Warm Start' or 'Cold Start'.

    Prioritises warm-start users (those with ≥ 1 training interaction).
    If fewer than n_users warm-start users exist, fills remainder with
    cold-start users.

    Returns
    -------
    sample : list of (user_idx, 'Warm Start' | 'Cold Start')
    n_warm : count of warm-start users in sample
    n_cold : count of cold-start users in sample
    """
    rows, _ = test_mat.nonzero()
    all_test = np.unique(rows)
    rng      = np.random.default_rng(seed)

    warm = [int(u) for u in all_test if train_mat.getrow(u).nnz > 0]
    cold = [int(u) for u in all_test if train_mat.getrow(u).nnz == 0]
    rng.shuffle(warm)
    rng.shuffle(cold)

    n_warm = min(n_users, len(warm))
    n_cold = max(0, n_users - n_warm)
    sample = ([(u, "Warm Start") for u in warm[:n_warm]] +
              [(u, "Cold Start")  for u in cold[:n_cold]])
    return sample, n_warm, n_cold


# =============================================================================
# Stage 1 — Retrieval Only
# =============================================================================

def stage_retrieval_only(
    user_emb: np.ndarray,
    faiss_index,
    test_mat: sp.csr_matrix,
    sample_users: List[Tuple[int, str]],
    k: int,
    pool: int,
) -> List[dict]:
    results = []
    for user_idx, user_type in sample_users:
        gt = set(int(x) for x in test_mat.getrow(user_idx).nonzero()[1])
        if not gt:
            continue
        u_vec = user_emb[user_idx].astype("float32").reshape(1, -1)
        faiss.normalize_L2(u_vec)
        _, I  = faiss_index.search(u_vec, pool)
        ranked = [int(x) for x in I[0] if x >= 0]

        results.append(dict(
            user_idx   = user_idx,
            user_type  = user_type,
            ndcg       = ndcg_at_k(ranked, gt, k),
            recall     = recall_at_k(ranked, gt, k),
        ))
    return results


# =============================================================================
# Stage 2 — Retrieval + Ranking (DeepFM)
# =============================================================================

def stage_retrieval_ranking(
    user_emb: np.ndarray,
    faiss_index,
    predictor,
    test_mat: sp.csr_matrix,
    sample_users: List[Tuple[int, str]],
    k: int,
    pool: int,
) -> List[dict]:
    results = []
    for user_idx, user_type in sample_users:
        gt = set(int(x) for x in test_mat.getrow(user_idx).nonzero()[1])
        if not gt:
            continue
        u_vec = user_emb[user_idx].astype("float32").reshape(1, -1)
        faiss.normalize_L2(u_vec)
        _, I      = faiss_index.search(u_vec, pool)
        cands     = np.array([int(x) for x in I[0] if x >= 0], dtype=np.int64)
        scores    = predictor.score_candidates(user_idx, cands)
        ranked    = [int(x) for x in cands[np.argsort(scores)[::-1]]]

        results.append(dict(
            user_idx   = user_idx,
            user_type  = user_type,
            ndcg       = ndcg_at_k(ranked, gt, k),
            recall     = recall_at_k(ranked, gt, k),
        ))
    return results


# =============================================================================
# Stage 3 — Retrieval + Ranking + Gemini LLM Re-ranking
# =============================================================================

def _build_llm_context(
    user_idx: int,
    candidate_indices: List[int],
    past_item_idxs: np.ndarray,
    item_lookup_idx: pd.DataFrame,
    user_profiles_raw: dict,
    user_enc,
) -> Tuple[str, List[str], dict]:
    """
    Build the Gemini prompt string for one user.
    Mirrors prepare_llm_context() from the notebook (§13.2).

    Returns
    -------
    context_str     : full prompt (system instructions + borrower profile + candidates)
    original_ids    : item_id strings in DeepFM rank order
    id_to_idx       : mapping item_id → item_idx integer
    """
    member_id  = user_enc.classes_[user_idx]
    profile    = user_profiles_raw.get(member_id, {})
    annual_inc = profile.get("annual_inc")
    dti        = profile.get("dti")
    fico_lo    = profile.get("fico_range_low")
    fico_hi    = profile.get("fico_range_high")
    ownership  = profile.get("home_ownership", "N/A")
    state      = profile.get("addr_state",     "N/A")

    fico_str = (f"{fico_lo:.0f}-{fico_hi:.0f}"
                if (fico_lo and fico_hi) else "N/A")
    inc_str  = f"${annual_inc:,.0f}" if annual_inc else "N/A"
    dti_str  = f"{dti:.1f}%"         if dti        else "N/A"

    # Historical loan products
    hist_lines = []
    for idx in past_item_idxs:
        if idx in item_lookup_idx.index:
            row = item_lookup_idx.loc[idx]
            hist_lines.append(
                f"  Grade={row['grade']} "
                f"Purpose={row['purpose']:<18} "
                f"Term={row['term']}"
            )
    history_str = ("\n".join(hist_lines)
                   if hist_lines else "  No prior loan history.")

    # Candidate products
    rows, id_to_idx = [], {}
    for idx in candidate_indices:
        if idx in item_lookup_idx.index:
            r = item_lookup_idx.loc[idx].to_dict()
            r["item_idx"] = int(idx)
            rows.append(r)
            id_to_idx[r["item_id"]] = int(idx)
    cand_df = pd.DataFrame(rows).reset_index(drop=True)

    product_lines = "\n".join(
        f'  {i+1:>2}. item_id="{row["item_id"]}"'
        f'  Grade={row["grade"]}'
        f'  Purpose={row["purpose"]:<22}'
        f'  Term={row["term"]}'
        f'  Rate={row["int_rate"]:.1f}%'
        f'  HistRepayRate={row["positive_rate"]:.0%}'
        for i, (_, row) in enumerate(cand_df.iterrows())
    )

    context_str = (
        "You are a predictive behavioral analyst for LendingClub.\n"
        "A machine-learning model has pre-selected candidate loan products.\n"
        "Your task: reorder them from MOST to LEAST likely to be accepted "
        "by this specific borrower.\n\n"
        "PREDICTION RULES:\n"
        "1. Prioritize candidates that match the borrower's previously accepted "
        "loan characteristics (Grade, Term, Purpose).\n"
        "2. High DTI (>20%) or low FICO (<680) → prefer Grade A/B products.\n"
        "3. Match purpose to profile "
        "(e.g., debt_consolidation for high-DTI; home_improvement for homeowners).\n"
        "4. HistRepayRate > 90% indicates a popular, low-risk product.\n\n"
        f"BORROWER PROFILE:\n"
        f"  FICO Score     : {fico_str}\n"
        f"  Annual Income  : {inc_str}\n"
        f"  Debt-to-Income : {dti_str}\n"
        f"  Home Ownership : {ownership}\n"
        f"  State          : {state}\n\n"
        f"PREVIOUSLY ACCEPTED LOANS:\n"
        f"{history_str}\n\n"
        f"CANDIDATE LOAN PRODUCTS (DeepFM rank order):\n"
        f"{product_lines}\n\n"
        "Return ONLY a JSON array of item_id strings in your predicted order, "
        "most likely first. Include all candidate item_ids."
    )

    original_ids = cand_df["item_id"].tolist()
    return context_str, original_ids, id_to_idx


def _call_gemini(
    gemini_client,
    context_str: str,
    original_item_ids: List[str],
    model: str,
) -> List[str]:
    """
    Call Gemini with structured JSON output and 3-attempt retry on 429/503.
    Falls back to original DeepFM order on persistent failure.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  [ablation] google-genai not installed. Skipping LLM stage.")
        return original_item_ids

    system_prompt = (
        "Role: You are a predictive behavioral analyst for LendingClub.\n"
        "Task: Select the TOP 5 loans the user is most likely to accept, "
        "in order of likelihood.\n"
        "Crucial Rule: Prioritize candidates matching previously accepted loan "
        "characteristics (Grade, Term, Purpose).\n\n"
        "Output: Return ONLY a JSON array of exactly 5 'item_id' strings."
    )
    payload = f"{system_prompt}\n\nData Context:\n{context_str}"

    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=payload,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=list[str],
                ),
            )
            ranked = json.loads(response.text.strip())
            if not isinstance(ranked, list):
                return original_item_ids
            valid_set = set(original_item_ids)
            valid   = [x for x in ranked if x in valid_set]
            missing = [x for x in original_item_ids if x not in set(valid)]
            return valid + missing

        except Exception as e:
            err = str(e)
            if any(code in err for code in ("503", "429", "RESOURCE_EXHAUSTED")):
                wait = 15 * (attempt + 1)
                print(f"    [Retry {attempt+1}/3] {model} busy, "
                      f"waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"    WARNING {model}: {type(e).__name__}: {e}")
                return original_item_ids

    print(f"    WARNING: max retries reached for {model}. "
          "Falling back to DeepFM order.")
    return original_item_ids


def stage_retrieval_ranking_llm(
    user_emb: np.ndarray,
    faiss_index,
    predictor,
    train_mat: sp.csr_matrix,
    test_mat: sp.csr_matrix,
    enc: dict,
    item_lookup: pd.DataFrame,
    user_profiles_raw: dict,
    sample_users: List[Tuple[int, str]],
    k: int,
    pool: int,
    deepfm_top_n: int = 10,
    inter_user_sleep: float = 20.0,
    inter_model_sleep: float = 5.0,
) -> Optional[Tuple[List[dict], List[dict]]]:
    """
    Stage 3: FAISS → DeepFM top-{deepfm_top_n} → Gemini re-rank (both models).

    Returns (results_model_a, results_model_b) or None if no API key found.
    Each result list mirrors the structure of stages 1 & 2.
    """
   
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  [ablation] GOOGLE_API_KEY not set — skipping LLM stage.")
        return None


    gemini_client  = genai.Client(api_key=api_key)
    item_lookup_idx = item_lookup.set_index("item_idx")
    user_enc       = enc["user_enc"]

    results_a, results_b = [], []

    for user_idx, user_type in sample_users:
        gt = set(int(x) for x in test_mat.getrow(user_idx).nonzero()[1])
        if not gt:
            continue

        # FAISS retrieval
        u_vec = user_emb[user_idx].astype("float32").reshape(1, -1)
        faiss.normalize_L2(u_vec)
        _, I   = faiss_index.search(u_vec, pool)
        cands  = np.array([int(x) for x in I[0] if x >= 0], dtype=np.int64)

        # DeepFM scoring
        scores    = predictor.score_candidates(user_idx, cands)
        top_idxs  = [int(x) for x in cands[np.argsort(scores)[::-1][:deepfm_top_n]]]

        # Build LLM context
        past_idxs = train_mat.getrow(user_idx).nonzero()[1]
        ctx, orig_ids, id_to_idx = _build_llm_context(
            user_idx, top_idxs, past_idxs,
            item_lookup_idx, user_profiles_raw, user_enc,
        )

        # Model A
        reranked_a = _call_gemini(gemini_client, ctx, orig_ids, GEMINI_MODEL_A)
        ranked_a   = [id_to_idx[i] for i in reranked_a if i in id_to_idx]
        results_a.append(dict(
            user_idx  = user_idx,
            user_type = user_type,
            ndcg      = ndcg_at_k(ranked_a, gt, k),
            recall    = recall_at_k(ranked_a, gt, k),
        ))
        time.sleep(inter_model_sleep)

        # Model B
        reranked_b = _call_gemini(gemini_client, ctx, orig_ids, GEMINI_MODEL_B)
        ranked_b   = [id_to_idx[i] for i in reranked_b if i in id_to_idx]
        results_b.append(dict(
            user_idx  = user_idx,
            user_type = user_type,
            ndcg      = ndcg_at_k(ranked_b, gt, k),
            recall    = recall_at_k(ranked_b, gt, k),
        ))
        time.sleep(inter_user_sleep)

    return results_a, results_b


# =============================================================================
# Aggregation helpers
# =============================================================================

def _subset_metrics(
    results: List[dict],
    subset: Optional[str],  # None = all users
    k: int,
) -> Optional[dict]:
    """Return mean NDCG@k and Recall@k for a user subset (or all users)."""
    rows = results if subset is None else \
           [r for r in results if r["user_type"] == subset]
    if not rows:
        return None
    return {
        f"NDCG@{k}":   _agg([r["ndcg"]   for r in rows]),
        f"Recall@{k}": _agg([r["recall"] for r in rows]),
        "n":           len(rows),
    }


def _print_section(
    label: str,
    results: List[dict],
    k: int,
) -> None:
    all_m  = _subset_metrics(results, None,          k)
    warm_m = _subset_metrics(results, "Warm Start",  k)
    cold_m = _subset_metrics(results, "Cold Start",  k)

    def fmt(m):
        if m is None:
            return "  N/A"
        return (f"  NDCG@{k}={m[f'NDCG@{k}']:.4f}  "
                f"Recall@{k}={m[f'Recall@{k}']:.4f}  "
                f"(n={m['n']})")

    print(f"  {label:<40} All:{fmt(all_m)}")
    print(f"  {'':40} Warm:{fmt(warm_m)}")
    print(f"  {'':40} Cold:{fmt(cold_m)}")


# =============================================================================
# Main ablation runner
# =============================================================================

def run_ablation(k: int = 5, n_users: int = 20, pool: int = 50) -> pd.DataFrame:
    enc, user_emb, faiss_index, predictor, train_mat, test_mat, \
        item_lookup, user_profiles_raw = load_artefacts()

    sampled, n_warm, n_cold = sample_users(
        test_mat, train_mat, n_users=n_users
    )
    print(f"\n[ablation] {len(sampled)} users  "
          f"({n_warm} warm-start + {n_cold} cold-start)  "
          f"k={k}  pool={pool}\n")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print("Stage 1: Retrieval Only (ALS + FAISS) …")
    r1 = stage_retrieval_only(
        user_emb, faiss_index, test_mat, sampled, k, pool)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print("Stage 2: Retrieval + Ranking (DeepFM) …")
    r2 = stage_retrieval_ranking(
        user_emb, faiss_index, predictor, test_mat, sampled, k, pool)

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    print(f"Stage 3: Retrieval + Ranking + LLM "
          f"({GEMINI_MODEL_A}  /  {GEMINI_MODEL_B}) …")
    llm_out = stage_retrieval_ranking_llm(
        user_emb, faiss_index, predictor, train_mat, test_mat,
        enc, item_lookup, user_profiles_raw, sampled,
        k=k, pool=pool,
    )
    r3a, r3b = (llm_out if llm_out is not None else (None, None))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("ABLATION STUDY RESULTS")
    print("═" * 80)
    _print_section("Stage 1 — Retrieval Only",              r1,  k)
    print()
    _print_section("Stage 2 — Retrieval + Ranking",         r2,  k)
    print()
    if r3a:
        _print_section(f"Stage 3 — + LLM ({GEMINI_MODEL_A})", r3a, k)
        print()
    if r3b:
        _print_section(f"Stage 3 — + LLM ({GEMINI_MODEL_B})", r3b, k)
        print()
    print("═" * 80)

    # ── Build output DataFrame ────────────────────────────────────────────────
    subsets  = [None, "Warm Start", "Cold Start"]
    sub_labels = ["All", "Warm Start", "Cold Start"]

    rows_out = []
    for sub, sub_lbl in zip(subsets, sub_labels):
        row = {"User Subset": sub_lbl}
        for stage_lbl, stage_results in [
            ("Stage1_Retrieval",              r1),
            ("Stage2_Ranking",                r2),
        ]:
            m = _subset_metrics(stage_results, sub, k)
            row[f"{stage_lbl}_NDCG@{k}"]   = m[f"NDCG@{k}"]   if m else float("nan")
            row[f"{stage_lbl}_Recall@{k}"]  = m[f"Recall@{k}"] if m else float("nan")

        for stage_lbl, stage_results in [
            (f"Stage3_{GEMINI_MODEL_A.replace('.', '_')}", r3a),
            (f"Stage3_{GEMINI_MODEL_B.replace('.', '_')}", r3b),
        ]:
            if stage_results is not None:
                m = _subset_metrics(stage_results, sub, k)
                row[f"{stage_lbl}_NDCG@{k}"]  = m[f"NDCG@{k}"]   if m else float("nan")
                row[f"{stage_lbl}_Recall@{k}"] = m[f"Recall@{k}"] if m else float("nan")

        rows_out.append(row)

    df_out = pd.DataFrame(rows_out).set_index("User Subset")

    print("\nSummary table:\n")
    print(df_out.to_string(float_format=lambda v: f"{v:.4f}"))
    print()

    os.makedirs("evaluation", exist_ok=True)
    df_out.to_csv(OUT_CSV)
    print(f"[ablation] Results saved → {OUT_CSV}")

    return df_out


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ablation study: Retrieval / Ranking / LLM "
                    "with NDCG@K and Recall@K split by Warm/Cold Start users."
    )
    parser.add_argument(
        "--k",       type=int, default=5,
        help="Rank cut-off for metrics (default: 5)"
    )
    parser.add_argument(
        "--n-users", type=int, default=20,
        help="Number of test users to evaluate (default: 20, "
             "warm-start users are prioritised)"
    )
    parser.add_argument(
        "--pool",    type=int, default=50,
        help="FAISS candidate pool size per user (default: 50)"
    )
    args = parser.parse_args()
    run_ablation(k=args.k, n_users=args.n_users, pool=args.pool)
