# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedHop-Agent: agentic multi-hop retrieval for biomedical QA (CS 572 Final Project, Emory, Spring 2026). Four-stage pipeline evaluated on the **MedHop** benchmark (QAngaroo, Welbl et al. 2018):

```
Question → Router → Retriever → Reranker → Agent → Answer
```

Each stage is a separate subdirectory with its own `requirements.txt`, owned by a different team member.

## Setup

```bash
conda activate 572_retrieval   # Python 3.10, all deps pre-installed

# Or create from scratch:
conda create -n 572_retrieval python=3.10 -y && conda activate 572_retrieval
pip install -r retriever/requirements.txt
pip install -r reranker/requirements.txt
pip install -r router/requirements.txt
pip install -r agent/requirements.txt
```

For the LLM-based agent, fill in `.env` at repo root (gitignored):
```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

## Dataset

**MedHop** from QAngaroo (Welbl et al. 2018). A multi-hop reading comprehension benchmark.
- 1,620 train / 342 validation examples
- Each example: query (`interacts_with DBXXXXX?`), ~22 support passages (median), 9 candidate DrugBank IDs, 1 correct answer
- Most support passages are **distractors** — typically only 1 out of ~22 contains the answer
- The official task: given the question + support passages + candidates → select the correct answer
- Do NOT use MedQA or MedHopQA — different benchmarks

## Two Evaluation Modes

**Closed-domain** (`--closed_domain`): Each question's ~22 bundled support passages are reranked directly by the reranker. The retriever is not used. This matches how teammates evaluated their components.

**Open-domain** (`--open_domain`): The retriever searches a global corpus of ~50K passages (all MedHop passages pooled), returns top-30, then the reranker narrows to top-5. Each hop can discover new passages. Requires `retriever/data/corpus.json` and retriever checkpoint.

## Running Each Component

**Router** (Person C — run from `router/`):
```bash
cd router
python label.py       # BM25 weak-supervision labels → labels_train.json
python train.py       # LogisticRegression → model.pkl
python evaluate.py    # Val accuracy + plots → results.json, results/
```

**Retriever** (Person A — run from `retriever/`):
```bash
cd retriever
python data_prep.py            # Reformat MedHop → corpus.json, triples (~3 min)
python hard_negatives.py       # Mine BM25 hard negatives (~5 min)
python train.py --epochs 5     # Fine-tune MedCPT (~30 min GPU)
python evaluate.py --checkpoint checkpoints/epoch_5
```

**Reranker** (Person B — run from repo root):
```bash
python reranker/train_reranker.py --epochs 1                        # fine-tune (~90 min CPU)
python scripts/evaluate_reranker.py --model reranker/checkpoints/finetuned \
    --run_name ms_marco_finetuned --compare_to_retriever
```

**Agent** (Person D — run from repo root):
```bash
# Closed-domain (reranker checkpoint only)
python agent/evaluate.py --closed_domain --generator heuristic --max_examples 5
python agent/evaluate.py --closed_domain --generator azure

# Open-domain (needs retriever/data/ + retriever checkpoint)
python agent/evaluate.py --open_domain --generator azure

# Full ablation suite
python agent/evaluate.py --ablation

# View consolidated leaderboard
python -c "from eval_harness.runner import print_leaderboard; print_leaderboard()"
```

## Architecture

### Component interfaces

- **Router** (`router/router.py`): `predict_hop(question, candidates, supports)` → `'single'` or `'multi'`. Loads `model.pkl` at import; falls back to always-multi if missing.
- **Retriever** (`retriever/retriever.py`): `retrieve(question, corpus=None, k=5)` → list of passage strings. Lazy-loads fine-tuned MedCPT; falls back to BM25.
- **Reranker** (`reranker/reranker.py`): `Reranker.rerank(query, passages, top_k)` → `[(text, score), ...]` best-first. For fine-tuned: `Reranker(model_name="reranker/checkpoints/finetuned")`.
- **Agent** (`agent/agent.py`): `run_agent(query, candidates, supports, config, hop_decision, reranker, retriever_fn)` → `AgentResult` with answer, hops used, traces.
- **Pipeline** (`agent/pipeline.py`): `run_pipeline(example, config, use_router, reranker, retriever_fn)` → dict with prediction, gold, correctness, diagnostics.

### Agent loop (IRCoT-style)

1. **Router** decides single-hop (1 hop) or multi-hop (up to 3 hops)
2. **Each hop**: rerank passages with current query → top-5 → generator reasons over them → extracts intermediate DrugBank IDs → updates condensed memory
3. **Query refinement**: next hop appends intermediate entities to the query, causing different passages to surface
4. **Early stopping**: fires when reranker score gap is large AND generator confidence ≥ 0.8 AND answer is produced
5. **Final answer**: selected from the candidates list; validated and fallback logic if generator hallucinates

Two generator backends:
- `heuristic`: scores candidates by weighted mention count in reranked passages (no API needed)
- `azure`: Azure OpenAI chat completions for chain-of-thought reasoning (requires `.env`)

### Shared evaluation harness (`eval_harness/`)

- `metrics.py`: `recall_at_k`, `ndcg_at_k`, `map_score`, `reciprocal_rank`, `exact_match`
- `runner.py`: `evaluate_rankings()`, `append_to_leaderboard(component, run_name, metrics)`, `print_leaderboard()`
- All results merge into shared `results.json` at repo root

### Data dependencies

| What | Path | Generated by | Gitignored |
|------|------|-------------|------------|
| Router model | `router/model.pkl` | `router/train.py` | No (1.2 KB) |
| Router labels | `router/labels_train.json` | `router/label.py` | No |
| Retriever corpus | `retriever/data/corpus.json` | `retriever/data_prep.py` | Yes |
| Retriever checkpoint | `retriever/checkpoints/epoch_N/` | `retriever/train.py` | Yes |
| Reranker checkpoint | `reranker/checkpoints/finetuned/` | `reranker/train_reranker.py` | Yes |
| Azure credentials | `.env` | Manual | Yes |

### What each eval mode requires

| Mode | Reranker ckpt | Retriever data | Retriever ckpt |
|------|--------------|----------------|----------------|
| Closed-domain | Needed | No | No |
| Open-domain | Needed | Yes (`corpus.json`) | Yes |

## Ablation Suite

`python agent/evaluate.py --ablation` runs up to 8 configurations:

| Run | Generator | Router | Hops | Mode | Requires |
|-----|-----------|--------|------|------|----------|
| `closed_heuristic_single` | heuristic | off | 1 | closed | reranker |
| `closed_heuristic_multi` | heuristic | off | 3 | closed | reranker |
| `closed_heuristic_router` | heuristic | on | 1-3 | closed | reranker |
| `open_heuristic_single` | heuristic | off | 1 | open | + retriever |
| `open_heuristic_multi` | heuristic | off | 3 | open | + retriever |
| `open_heuristic_router` | heuristic | on | 1-3 | open | + retriever |
| `llm_no_router` | Azure OpenAI | off | 3 | best available | + API key |
| `llm_with_router` | Azure OpenAI | on | 1-3 | best available | + API key |

Primary metric: **Exact Match** (does predicted DrugBank ID match gold?). Also reports avg hops, early stopping rate, breakdown by router decision.

## Experiment Plan

### Phase 1: Closed-domain (ready now)
Requires only: reranker checkpoint (`reranker/checkpoints/finetuned/`) + `.env` for LLM runs.
Falls back to zero-shot reranker if checkpoint is missing.

```bash
# Step 1: Quick sanity check (should finish in seconds)
python agent/evaluate.py --closed_domain --generator heuristic --max_examples 5

# Step 2: Full closed-domain heuristic ablations (~6 min total)
python agent/evaluate.py --closed_domain --generator heuristic --no_router --max_hops 1 --run_name closed_heuristic_single
python agent/evaluate.py --closed_domain --generator heuristic --no_router --run_name closed_heuristic_multi
python agent/evaluate.py --closed_domain --generator heuristic --run_name closed_heuristic_router

# Step 3: LLM ablations (~30 min each, requires .env)
python agent/evaluate.py --closed_domain --generator azure --no_router --run_name closed_llm_no_router
python agent/evaluate.py --closed_domain --generator azure --run_name closed_llm_with_router
```

### Phase 2: Open-domain (requires retriever setup)
Requires: `retriever/data/corpus.json` + retriever checkpoint + reranker checkpoint + `.env`.

```bash
# Step 1: Generate retriever data (if not already present, ~3 min, no GPU)
cd retriever && python data_prep.py && cd ..

# Step 2: Open-domain ablations (slower — retriever search adds time per hop)
python agent/evaluate.py --open_domain --generator heuristic --no_router --max_hops 1 --run_name open_heuristic_single
python agent/evaluate.py --open_domain --generator heuristic --no_router --run_name open_heuristic_multi
python agent/evaluate.py --open_domain --generator heuristic --run_name open_heuristic_router
python agent/evaluate.py --open_domain --generator azure --no_router --run_name open_llm_no_router
python agent/evaluate.py --open_domain --generator azure --run_name open_llm_with_router
```

Or run everything at once: `python agent/evaluate.py --ablation` (auto-skips unavailable modes).

### Time estimates (342 validation examples, CPU)
- Heuristic, single-hop: ~1 min
- Heuristic, multi-hop (3 hops): ~3 min
- LLM, multi-hop (3 hops): ~30 min (dominated by API latency)
- Full closed-domain ablation (3 heuristic + 2 LLM): ~1 hour
- Full open-domain adds retriever overhead per hop

### New machine setup
```bash
conda create -n 572_retrieval python=3.10 -y && conda activate 572_retrieval
pip install -r retriever/requirements.txt
pip install -r reranker/requirements.txt
pip install -r router/requirements.txt
pip install -r agent/requirements.txt
```
Then copy gitignored artifacts (reranker/retriever checkpoints, retriever/data/) and fill in `.env`.

### Understanding results
- **Exact Match (EM)** is the primary metric — does the predicted DrugBank ID match the gold answer?
- The heuristic generator uses regex entity extraction + weighted mention counting (not real reasoning — it's a retrieval-only baseline)
- The LLM generator (Azure OpenAI) does actual chain-of-thought reasoning via structured prompts
- Comparing heuristic vs LLM shows the value of reasoning beyond retrieval
- Comparing router on vs off shows whether adaptive hop-count selection helps
- Comparing single-hop vs multi-hop shows whether iterative query refinement helps
- All results append to `results.json` under `"end_to_end"` — view with `print_leaderboard()`

## Team

| Component | Owner |
|-----------|-------|
| Retriever (fine-tuned MedCPT bi-encoder) | Carol (Person A) |
| Reranker (fine-tuned cross-encoder) + Eval Harness | Behjat (Person B) |
| Router (logistic regression, BM25 weak supervision) | Selina (Person C) |
| Agent loop (IRCoT) + Integration + End-to-end eval | Ken (Person D) |
