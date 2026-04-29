"""
End-to-end routing evaluation on MedHop validation split.

Compares three conditions on retrieval quality (Recall@K):
  always_single  — one BM25+rerank pass per question (vanilla RAG)
  always_multi   — up to 2 BM25+rerank hops with early stopping (IRCoT-style)
  routed         — router decides single vs. multi per question

Writes results to top-level results.json under the "router_e2e" key.
Mirrors the structure and flags of scripts/evaluate_reranker.py.

Run from repo root:
    python scripts/evaluate_router_e2e.py
    python scripts/evaluate_router_e2e.py --model reranker/checkpoints/finetuned
    python scripts/evaluate_router_e2e.py --run_name lr_6feat_v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from reranker.reranker import Reranker
from router.router import predict_hop
from eval_harness.runner import evaluate_rankings, append_to_leaderboard

DATA_DIR   = REPO_ROOT / "retriever" / "data"
RESULTS_FILE = REPO_ROOT / "results.json"

# IRCoT early-stopping hyperparameters
MAX_HOPS      = 2
CONF_PLATEAU  = 0.05  # stop if top reranker score improves by less than this


# ---------------------------------------------------------------------------
# Lightweight BM25 helper (avoids the noisy tqdm inside BM25Retriever.__init__)
# ---------------------------------------------------------------------------

def _bm25_rank_texts(query: str, texts: list[str], k: int) -> list[str]:
    """Rank texts by BM25 and return top-k texts. Uses rank_bm25 directly."""
    if not texts:
        return []
    tokenized = [t.lower().split() for t in texts]
    bm25   = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][: min(k, len(texts))]
    return [texts[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Single-hop and multi-hop retrieval paths
# ---------------------------------------------------------------------------

def run_single_hop(
    query: str,
    support_texts: list[str],
    reranker: Reranker,
) -> list[str]:
    """
    Vanilla RAG: BM25 over support set then rerank.
    Returns all passages ranked best-first (no truncation) so that
    evaluate_rankings() can compute R@K for any K.
    """
    candidates = _bm25_rank_texts(query, support_texts, k=30)
    ranked = reranker.rerank(query, candidates, top_k=None)  # return all, ranked
    return [p for p, _ in ranked]


def run_multi_hop(
    query: str,
    support_texts: list[str],
    reranker: Reranker,
    max_hops: int = MAX_HOPS,
    conf_plateau: float = CONF_PLATEAU,
) -> list[str]:
    """
    IRCoT-style multi-hop: iterative BM25+rerank with early stopping.
    Returns accumulated unique passages in order of discovery (hop 1 best-first,
    then new passages from hop 2 best-first), suitable for R@K evaluation.

    Early stopping: if top reranker score improves by < conf_plateau between hops,
    the additional hop is unlikely to surface new relevant evidence.
    """
    augmented_q    = query
    prev_top_score = -float("inf")
    seen: set[str] = set()
    accumulated: list[str] = []

    for _ in range(max_hops):
        candidates = _bm25_rank_texts(augmented_q, support_texts, k=30)
        # Rank all candidates (top_k=None) so new passages can surface in later slots
        ranked = reranker.rerank(augmented_q, candidates, top_k=None)
        if not ranked:
            break

        top_score = ranked[0][1]
        if top_score - prev_top_score < conf_plateau:
            break
        prev_top_score = top_score

        for passage, _ in ranked:
            if passage not in seen:
                accumulated.append(passage)
                seen.add(passage)

        # Augment query with top passage excerpt for next hop (Baleen / IRCoT style)
        augmented_q = query + " Context: " + ranked[0][0][:200]

    return accumulated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="Reranker model path or HF name. Auto-detects fine-tuned checkpoint if present.",
    )
    parser.add_argument(
        "--run_name", default="lr_6feat_v1",
        help="Suffix for results.json key: 'routed_<run_name>'",
    )
    args = parser.parse_args()

    # ---- Load retriever data -----------------------------------------------
    print("Loading data...")
    for fname in ("corpus.json", "val_triples.json"):
        if not (DATA_DIR / fname).exists():
            print(f"ERROR: retriever/data/{fname} not found.")
            print("Run from repo root:  cd retriever && python data_prep.py")
            sys.exit(1)

    with open(DATA_DIR / "corpus.json") as f:
        corpus: dict[str, str] = json.load(f)
    with open(DATA_DIR / "val_triples.json") as f:
        val_triples: list[dict] = json.load(f)

    # Load MedHop val split to get the candidate lists for predict_hop()
    print("Loading MedHop val split for candidate lists (cached after first run)...")
    ds = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    id_to_candidates: dict[str, list[str]] = {
        ex["id"]: ex["candidates"] for ex in ds["validation"]
    }

    print(f"  corpus : {len(corpus)} passages")
    print(f"  val set: {len(val_triples)} examples")

    # ---- Load reranker ------------------------------------------------------
    finetuned_ckpt = REPO_ROOT / "reranker" / "checkpoints" / "finetuned"
    if args.model:
        model_name = args.model
        print(f"Reranker: {args.model}")
    elif finetuned_ckpt.exists():
        model_name = str(finetuned_ckpt)
        print(f"Reranker: fine-tuned checkpoint ({finetuned_ckpt.name})")
    else:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print("Reranker: zero-shot ms-marco (fine-tuned checkpoint not found — "
              "run reranker/train_reranker.py --epochs 1 to build it)")

    reranker = Reranker(model_name=model_name)

    # ---- Evaluate all three conditions in one pass -------------------------
    cond_keys = ("always_single", "always_multi", "routed")
    rankings: dict[str, dict[str, list[str]]] = {c: {} for c in cond_keys}
    relevants: dict[str, dict[str, set[str]]] = {c: {} for c in cond_keys}
    routing_counts = {"single": 0, "multi": 0}
    skipped_no_support = 0
    skipped_no_answer  = 0

    print("\nEvaluating...")
    for triple in tqdm(val_triples, desc="Val examples"):
        support_pids = [p for p in triple["supports"] if p in corpus]
        if not support_pids:
            skipped_no_support += 1
            continue

        relevant_pids = {p for p in support_pids if triple["answer"] in corpus[p]}
        if not relevant_pids:
            skipped_no_answer += 1
            continue

        support_texts = [corpus[pid] for pid in support_pids]
        # Reverse map: passage text -> PID (for converting ranked texts back to PIDs)
        text_to_pid = {corpus[pid]: pid for pid in support_pids}

        query      = triple["query"]
        candidates = id_to_candidates.get(triple["id"], [])
        qid        = triple["id"]

        def to_pids(texts: list[str]) -> list[str]:
            return [text_to_pid.get(t, t) for t in texts]

        # Compute single-hop and multi-hop results once; reuse for all three conditions.
        # Both return the full ranked passage list (no top-k truncation) so that
        # evaluate_rankings() can correctly compute R@5 and R@10 as distinct numbers.
        single_pids = to_pids(run_single_hop(query, support_texts, reranker))
        multi_pids  = to_pids(run_multi_hop(query, support_texts, reranker))

        rankings["always_single"][qid] = single_pids
        rankings["always_multi"][qid]  = multi_pids

        decision = predict_hop(query, candidates, support_texts)
        routing_counts[decision] += 1
        rankings["routed"][qid] = single_pids if decision == "single" else multi_pids

        for c in cond_keys:
            relevants[c][qid] = relevant_pids

    evaluated = len(rankings["routed"])
    print(f"\nEvaluated : {evaluated} examples")
    print(f"Skipped   : {skipped_no_support} (no support passages in corpus), "
          f"{skipped_no_answer} (answer not in any support)")
    print(f"Routing   : {routing_counts['single']} → single-hop, "
          f"{routing_counts['multi']} → multi-hop")

    # ---- Compute and report metrics ----------------------------------------
    print("\n=== End-to-End Results (closed-domain, MedHop val) ===")
    header = f"{'Condition':<22}  {'R@1':>6}  {'R@5':>6}  {'R@10':>7}"
    print(header)
    print("-" * len(header))

    for cond in cond_keys:
        m = evaluate_rankings(rankings[cond], relevants[cond], k_values=(1, 5, 10))
        r1, r5, r10 = m.get("recall@1", 0.0), m.get("recall@5", 0.0), m.get("recall@10", 0.0)
        print(f"{cond:<22}  {r1:>6.3f}  {r5:>6.3f}  {r10:>7.3f}")

        lb_key = f"routed_{args.run_name}" if cond == "routed" else cond
        payload: dict = {
            "recall@1":  round(r1,  4),
            "recall@5":  round(r5,  4),
            "recall@10": round(r10, 4),
            "n_examples": evaluated,
            "reranker_model": model_name,
        }
        if cond == "routed":
            payload["n_single"] = routing_counts["single"]
            payload["n_multi"]  = routing_counts["multi"]
        append_to_leaderboard("router_e2e", lb_key, payload, RESULTS_FILE)

    print(f"\nResults appended to {RESULTS_FILE} under 'router_e2e'")


if __name__ == "__main__":
    main()
