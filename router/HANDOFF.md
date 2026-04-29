# Router — Handoff

**Owner:** Person C (Selina / Qian Jingyun)
**Branch:** `person-c/router`
**Status:** v1 shipped (binary classifier, 6 features, 68.4% val accuracy). v2 in progress — see bottleneck section.
**Version:** 1.1

For Person D (Ken, agent + integration) and for team discussion. The interfaces below are working and importable today.

---

## Headline Results

### Router binary accuracy (MedHop val, 342 examples)

| Model | Accuracy | vs. always-multi baseline |
|-------|--------:|-------------------------:|
| Always-multi (baseline) | 62.6% | — |
| Always-single (baseline) | 37.4% | — |
| **Logistic Regression, 6 features (v1)** | **68.4%** | **+5.8 pp** |

Routes 84 / 342 val examples (24.6%) to single-hop; 258 / 342 (75.4%) to multi-hop.

### End-to-end retrieval (MedHop val, zero-shot reranker, closed-domain)

| Condition | R@1 | R@5 | R@10 |
|-----------|----:|----:|-----:|
| always_single | 0.035 | 0.354 | 0.602 |
| always_multi  | 0.035 | 0.354 | 0.605 |
| **routed_lr_6feat_v1** | 0.035 | 0.354 | 0.605 |

**These numbers are misleading — read the bottleneck section before drawing conclusions.**
The router routes correctly but shows no e2e gain because the zero-shot reranker and BM25 retrieval are both too weak to make multi-hop superior to single-hop. Expected to improve substantially once the fine-tuned reranker checkpoint is available.

---

## What's in this branch

```
router/
  router.py              # public interface — import predict_hop() from here
  features.py            # 6-feature extraction (BM25 + structural)
  label.py               # BM25 weak-supervision labeling (generates labels_train.json)
  train.py               # LogisticRegression + StandardScaler + 5-fold CV
  evaluate.py            # val accuracy, confusion matrix, feature importances
  explore.py             # one-time dataset stats (not needed for pipeline)
  model.pkl              # trained LR pipeline (committed — no retraining needed)
  labels_train.json      # BM25-derived binary labels for train split (committed)
  results.json           # local router accuracy results
  results/               # confusion matrix and feature importance plots
  HANDOFF.md             # this file
  __init__.py            # package init — exposes predict_hop, predict_multi
scripts/
  evaluate_router_e2e.py # end-to-end eval: routed vs always_single vs always_multi
```

---

## For Person D (agent loop + integration)

### Calling the router from your pipeline

```python
from router import predict_hop

decision = predict_hop(
    question   = "interacts_with DB00773?",
    candidates = ["DB00072", "DB00294", "DB00338"],  # from MedHop example
    supports   = [                                    # from first retrieval pass
        "DB00773 has been shown to interact with DB00072 in clinical trials.",
        "Studies show DB00294 affects the same pathway as DB00338.",
    ],
)
# decision is 'single' or 'multi'
```

**Return values:** `'single'` → run vanilla RAG (one retrieve+rerank pass). `'multi'` → run IRCoT-style retrieve–reason loop with early stopping.

**Fallback behaviour:** If `model.pkl` is missing, `predict_hop()` returns `'multi'` unconditionally (the safer default — it never skips evidence). No exception is raised.

### When to call the router

The router needs `supports` (passage texts) to extract features. **This means it runs after a first-pass retrieval, not before.** In your agent loop the correct sequence is:

```
question
  → first retrieval pass (BM25 or MedCPT, k=30)
  → predict_hop(question, candidates, supports=first_pass_results)
  → 'single': rerank first_pass_results → answer
  → 'multi':  IRCoT loop (retrieve → rerank → augment query → repeat)
              stop when reranker top-1 score stops improving
```

If the loop has no first-pass results available yet, call `predict_multi()` to force multi-hop (safe fallback):

```python
from router import predict_multi
decision = predict_multi()  # always returns 'multi'
```

### IRCoT branching reference (from scripts/evaluate_router_e2e.py)

```python
from reranker import rerank

MAX_HOPS     = 2
CONF_PLATEAU = 0.01   # NOTE: 0.05 is too aggressive — see bottleneck C below

def run_single_hop(query, support_texts, k=5):
    """Vanilla RAG — one rerank pass."""
    ranked = rerank(query, support_texts, top_k=k)
    return [p for p, _ in ranked]

def run_multi_hop(query, support_texts, k=5):
    """IRCoT — iterative rerank with condensed memory."""
    augmented_q    = query
    prev_top_score = -float("inf")
    seen, accumulated = set(), []

    for _ in range(MAX_HOPS):
        ranked = rerank(augmented_q, support_texts, top_k=None)
        if not ranked or ranked[0][1] - prev_top_score < CONF_PLATEAU:
            break
        prev_top_score = ranked[0][1]
        for p, _ in ranked:
            if p not in seen:
                accumulated.append(p); seen.add(p)
        augmented_q = query + " Context: " + ranked[0][0][:200]  # condensed memory

    return accumulated[:k]
```

**Use `reranker/checkpoints/finetuned` for the reranker** (R@5 = 0.567) rather than the zero-shot default (R@5 = 0.336). See HANDOFF.md in the reranker/ directory.

---

## For teammates (reproducing and extending)

### Reproducing router accuracy

```bash
# Install router deps (all already in cs572 env)
conda activate cs572
pip install -r router/requirements.txt

# Labels are committed — skip unless relabeling
# cd router && python label.py

# model.pkl is committed — skip unless retraining
# cd router && python train.py

# Evaluate on val split + write to shared leaderboard
cd router && python evaluate.py && cd ..
```

### Reproducing e2e eval

```bash
# Prerequisite: retriever corpus data
ls retriever/data/corpus.json || (cd retriever && python data_prep.py)

# Run e2e eval (zero-shot reranker, ~4 min on CPU)
conda activate cs572
python scripts/evaluate_router_e2e.py

# Run e2e eval with fine-tuned reranker (recommended — build checkpoint first)
python reranker/train_reranker.py --epochs 1        # ~10 min GPU / 90 min CPU
python scripts/evaluate_router_e2e.py \
    --model reranker/checkpoints/finetuned \
    --run_name lr_6feat_v1_finetuned_rr

# View full leaderboard
python -c "from eval_harness.runner import print_leaderboard; print_leaderboard()"
```

---

## Known Bottlenecks (v1 → v2 work items)

These are the reasons the current e2e numbers show no routing benefit. See `plans/can-you-identify-the-toasty-honey.md` for the full analysis.

### 🔴 Critical

**A. Fine-tuned reranker checkpoint is missing.**
`reranker/checkpoints/finetuned` is gitignored and doesn't exist unless locally built. The e2e eval falls back to zero-shot ms-marco (R@5=0.336) instead of the fine-tuned model (R@5=0.567, +23 pp). The routing decision can only produce visible e2e gains when the reranker is strong enough for multi-hop to improve on single-hop.
→ *Action:* Behjat (B) to regenerate checkpoint, or share directly. No code change needed.

**B. E2E eval uses BM25 retrieval, not fine-tuned MedCPT.**
`scripts/evaluate_router_e2e.py` uses a local `_bm25_rank_texts()` helper (BM25 R@5=0.199) and never calls `from retriever.retriever import retrieve` (MedCPT R@5=0.412, +21 pp). The router was trained on BM25-based features; evaluating with BM25 retrieval tests the same weak signal twice.
→ *Action (v2):* Selina (C) to refactor `run_single_hop()` / `run_multi_hop()` to call `retriever.retrieve()`. Requires Carol (A) to confirm retriever checkpoint path.

**C. Early stopping threshold (conf_plateau=0.05) disables multi-hop.**
Cross-encoder score deltas between hop 1 and hop 2 are typically ±0.01–0.03. The 0.05 threshold fires immediately on almost every example, so `always_multi == always_single` at R@5. The current code already uses `conf_plateau=0.01` in the reference snippet above; the e2e script still uses 0.05.
→ *Action (immediate):* Lower `conf_plateau` to 0.01 in `scripts/evaluate_router_e2e.py` line 41, or replace the raw-score stopping criterion with rank displacement (did the top-1 passage change between hops?).

### 🟡 Medium (v2 router improvements)

**D. Router labels are a difficulty proxy, not true hop-count labels.**
`router/label.py` uses a BM25 median split on answer-surfacing scores. MedHop is entirely multi-hop by design, so there are no true single-hop questions. The labels measure relative BM25 retrieval difficulty, not genuine hop-count necessity. This misaligns the training objective with the routing decision.
→ *Better approach (Adaptive-RAG style):* For each training example, run 1-hop retrieval → check if answer is in top-k. If yes → label 0. Else run 2-hop → if recall improves → label 1. Labels grounded in actual retrieval outcomes, not BM25 score distribution. Requires the retriever.

**E. `n_supports` feature dominates and is likely a confound.**
`n_supports` coefficient = -0.96 (≈10× the next feature). This is a dataset-structure artifact (MedHop assigns different passage counts to different questions) rather than a signal about multi-hop necessity. The model is essentially predicting labels that were derived from BM25 score variance using a feature that reflects dataset curation.
→ *Better features to add (no new deps):* `top_bm25_answer_score` (already computed in `label.py`, just not exposed as a feature), `candidate_coverage` (fraction of candidate drug IDs appearing in support texts), `bm25_dense_score_gap` (requires dense retriever scores from Carol (A)).

### 🟢 Minor

**F. Text→PID collision risk in e2e script.**
`text_to_pid = {corpus[pid]: pid for pid in support_pids}` silently overwrites if two passages share identical text. The correct pattern (from `scripts/evaluate_reranker.py` lines 97–106) scores PIDs directly:
```python
scores     = reranker.score(query, [corpus[pid] for pid in support_pids])
ranked_pids = [pid for pid, _ in sorted(zip(support_pids, scores), key=lambda x: x[1], reverse=True)]
```
→ *Action:* Fix in `scripts/evaluate_router_e2e.py` before final results are locked in.

---

## Shared `results.json` keys

This component writes under two keys:

```json
{
  "router": {
    "lr_6feat_v1": {
      "accuracy": 0.684,
      "accuracy_always_multi_baseline": 0.626,
      "improvement_pp": 5.85,
      "n_val": 342,
      "feature_importances": { ... }
    }
  },
  "router_e2e": {
    "always_single":        { "recall@5": 0.354, "recall@10": 0.602, ... },
    "always_multi":         { "recall@5": 0.354, "recall@10": 0.605, ... },
    "routed_lr_6feat_v1":   { "recall@5": 0.354, "recall@10": 0.605,
                               "n_single": 84, "n_multi": 258, ... }
  }
}
```

Print everything with:
```bash
python -c "from eval_harness.runner import print_leaderboard; print_leaderboard()"
```

---

## Method Summary

**Labels.** No ground-truth hop-count labels exist in MedHop. Weak supervision via BM25: for each training example, score how well BM25 surfaces the answer drug from the support passages; split at the training-set median to produce a 50/50 binary label (0 = easier, 1 = harder). Label quality is limited — see bottleneck D.

**Features (6).** All computed from each example's support passages and candidate list — no neural model required at feature extraction time.

| # | Feature | Coefficient | Description |
|---|---------|------------:|-------------|
| 0 | `n_supports` | −0.962 | Number of support passages (dataset-structure artifact) |
| 1 | `n_candidates` | −0.003 | Number of candidate answer drug IDs |
| 2 | `top_bm25_score` | +0.124 | BM25 top-1 score when querying by the drug ID |
| 3 | `bm25_score_gap` | +0.107 | Score gap between top-1 and top-2 BM25 passages |
| 4 | `candidates_in_supports` | 0.000 | Fraction of candidates appearing in any support |
| 5 | `top5_bm25_entropy` | +0.186 | Entropy of top-5 BM25 scores (evidence spread) |

**Classifier.** `sklearn.linear_model.LogisticRegression` inside a `Pipeline` with `StandardScaler`. 5-fold CV = 68.3% ± 9.3%. Val accuracy = 68.4%. The high CV variance suggests features have low discriminative power — consistent with bottleneck E.

---

## Open Questions for Team Meeting

1. **Carol (A):** Is the fine-tuned MedCPT checkpoint available? What is its path relative to the repo root? (Need to call `make_medcpt(..., checkpoint_dir=...)` in the e2e eval script.)

2. **Behjat (B):** Can you share or regenerate the fine-tuned reranker checkpoint at `reranker/checkpoints/finetuned`? Re-running `train_reranker.py --epochs 1` takes 90 min CPU / ~10 min GPU.

3. **Ken (D):** In your agent loop, at which point does the first retrieval pass happen? The router needs passage texts before it can route — are those available before your loop starts, or does your code need to call the router differently?

---

## Contact

Open an issue tagged `Selina-Qian` or ping on the branch.
