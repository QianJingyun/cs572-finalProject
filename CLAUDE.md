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

Three generator backends:
- `heuristic`: scores candidates by weighted mention count in reranked passages (no API needed)
- `azure`: Azure OpenAI chat completions for chain-of-thought reasoning (requires `.env`)
- `graph_llm`: graph-guided agent with LLM adjudication (requires `.env`) — see below

### Graph-guided agent (`graph_llm` generator)

The `azure` and `heuristic` generators both follow the IRCoT loop above, where the reranker selects top-5 passages per hop and the generator reasons over them. Analysis of MedHop revealed this is suboptimal because:

1. **Reranker surfaces wrong passages.** It ranks by similarity to `"interacts_with DB01171?"`. Bridge passages (e.g., about proteins P21397 ↔ Q05940 linking two drugs) don't mention the query drug, so they rank low and are cut at top-5.
2. **LLM does chain discovery over opaque tokens.** Tracing `DB → P → Q → DB` co-occurrence is pure pattern matching — regex beats GPT-4 at this. Drug names are replaced with DrugBank IDs and proteins with UniProt IDs, so the LLM can't use world knowledge.
3. **LLM can't judge evidence in the old setup.** Even if it sees the right passages, it can't distinguish "these proteins interact functionally" from "they're mentioned in the same gene survey" when everything is opaque codes and it has no context about which passages matter.

**Dataset structure** driving this design:
- 99.4% of examples have NO passage containing both the query drug and the answer
- The chain goes through intermediate protein IDs: `query_drug → protein_A → [bridge passages] → protein_B → answer_drug`
- Chain depth: 0.6% direct, 12.6% 1-hop protein bridge, 63.5% 2-hop, 23.4% deeper
- Entity graph BFS puts the answer in top-3 candidates 42–47% of the time (vs 11% random baseline from 9 candidates)

**Graph-guided agent loop** (`run_graph_agent` in `agent/agent.py`):

1. **Hop 1 — Discover** (regex + BFS, no API): Extract all `DB\d{5}` / `[PQ]\d{4,5}` entities from every passage. Build entity co-occurrence graph (edges = same-passage co-occurrence). BFS from query drug to each candidate. Combine graph score (40%) with reranker heuristic score (60%) — the two signals are complementary. Early-stop if one candidate dominates.

2. **Hop 2 — Verify** (reranker, no API in closed-domain): For each top-3 candidate's best chain, identify the weakest link (fewest supporting passages). Build targeted queries for those gaps. In open-domain: retrieve new passages and rebuild graph. In closed-domain: rerank existing passages with gap-targeted queries.

3. **Hop 3 — Judge** (LLM): Present top-3 candidates with their chain passages to the LLM. Each candidate shows the entity path and the specific passages forming each link. The LLM evaluates which chain describes a genuine functional drug interaction vs. incidental co-mention (gene surveys, genotyping studies). This is reading comprehension — the LLM's actual strength.

Key files: `agent/graph.py` (entity graph), `agent/prompts.py` (`CHAIN_EVAL_PROMPT`), `agent/generator.py` (`evaluate_chains` method).

```bash
# Graph-guided LLM (requires .env)
python agent/evaluate.py --closed_domain --generator graph_llm --run_name graph_llm_no_router --no_router
python agent/evaluate.py --closed_domain --generator graph_llm --run_name graph_llm_with_router
```

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

`python agent/evaluate.py --ablation` runs up to 12 configurations:

| Run | Generator | Router | Hops | Mode | Requires |
|-----|-----------|--------|------|------|----------|
| `closed_heuristic_single` | heuristic | off | 1 | closed | reranker |
| `closed_heuristic_multi` | heuristic | off | 3 | closed | reranker |
| `closed_heuristic_router` | heuristic | on | 1-3 | closed | reranker |
| `open_heuristic_single` | heuristic | off | 1 | open | + retriever |
| `open_heuristic_multi` | heuristic | off | 3 | open | + retriever |
| `open_heuristic_router` | heuristic | on | 1-3 | open | + retriever |
| `graph_llm_with_router` | graph_llm | on | 1-3 | best available | + API key |
| `graph_llm_no_router` | graph_llm | off | 3 | best available | + API key |
| `graph_llm_single` | graph_llm | off | 1 | best available | + API key |
| `llm_single` | Azure OpenAI | off | 1 | best available | + API key |
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

### Phase 3: Graph-guided LLM (requires .env)
```bash
python agent/evaluate.py --closed_domain --generator graph_llm --no_router --run_name graph_llm_no_router
python agent/evaluate.py --closed_domain --generator graph_llm --run_name graph_llm_with_router
python agent/evaluate.py --closed_domain --generator graph_llm --no_router --max_hops 1 --run_name graph_llm_single
```

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

## Analysis: Why the IRCoT LLM Underperforms the Heuristic

Empirical analysis of the MedHop dataset (342 validation examples) revealed structural properties that explain why the `azure` (IRCoT) generator matches or underperforms the heuristic:

### Dataset structure

- Each query is `interacts_with DBXXXXX?` — all entity names are **anonymized** (DrugBank IDs for drugs, UniProt IDs like P/Q codes for proteins)
- **99.4%** of examples have NO passage containing both the query drug and the answer drug
- The answer is always reachable through intermediate protein IDs: `query_drug → protein → [bridge] → protein → answer_drug`
- Chain depth distribution: 0.6% direct, 12.6% 1-hop protein bridge, 63.5% 2-hop, 23.4% 3+ hops

### Why the old LLM approach fails

The IRCoT agent asks the LLM to reason over reranker top-5 passages. Three problems:

1. **Reranker surfaces wrong passages.** Ranked by similarity to `"interacts_with DB01171?"`, bridge passages about `P21397 ↔ Q05940` (that don't mention the query drug) rank low and get cut at top-5.
2. **Chain discovery over opaque tokens.** Tracing `DB → P → Q → DB` co-occurrence is pattern matching — regex beats LLMs. With anonymized IDs, the LLM can't use biomedical world knowledge.
3. **No evidence quality judgment.** Even if the right passages are surfaced, the LLM can't distinguish functional protein interactions from incidental co-mentions in gene surveys — it's never shown the chain structure.

### Candidate ranking recall (342 validation examples)

These measure how often the gold answer appears in the top-k candidates **before** the LLM makes a final pick. The end-to-end pipeline outputs a single prediction, so recall@3/5 only applies to the intermediate candidate ranking — not to final end-to-end results (which are always recall@1 = exact match).

| Method | Recall@1 (EM) | Recall@3 | Recall@5 |
|--------|--------------|----------|----------|
| Heuristic (reranker-weighted counts) | 0.2135 | 0.4620 | 0.6930 |
| Graph (BFS shortest path, no reranker) | 0.1608 | 0.4181 | 0.6287 |
| **Combined (40% graph + 60% heuristic)** | **0.1930** | **0.4883** | **0.6871** |
| Random baseline (9 candidates) | 0.1111 | 0.3333 | 0.5556 |

The combined scoring has the **best recall@3** (48.8%) — meaning the gold answer lands in the top-3 pool that the LLM adjudicates over almost half the time. But heuristic alone has the best recall@1 (21.4%), which is why the heuristic generator still wins on exact match when the LLM doesn't reliably reorder the top-3.

The `combined_40_60` recall reflects **hop 1 scoring only** (graph + heuristic, no LLM). The `graph_llm_with_router` end-to-end EM (0.2018) is the result after all 3 hops — early stopping, gap filling, and LLM adjudication over the top-3. The LLM adds a small boost (+0.0088 over combined recall@1 of 0.1930), but early stopping hurts in some cases.

### End-to-end results (342 validation examples, closed-domain)

| Run | Generator | Router | EM | Hops | Early Stop |
|-----|-----------|--------|----|------|------------|
| `closed_heuristic_single` | heuristic | off | 0.2018 | 1.0 | 3.2% |
| `closed_heuristic_multi` | heuristic | off | 0.1930 | 2.9 | 6.4% |
| `closed_heuristic_router` | heuristic | on | 0.1842 | 2.5 | 4.1% |
| `llm_single_v1` | azure | off | 0.1784 | 1.0 | 14.3% |
| `llm_no_router_v1` | azure | off | 0.1930 | 2.6 | 25.2% |
| `llm_with_router_v1` | azure | on | 0.1959 | 2.3 | 21.9% |
| `graph_llm_with_router` | graph_llm | on | 0.2018 | 2.2 | 19.6% |

### Baseline numbers motivating graph_llm

| Method | EM | Notes |
|--------|-----|-------|
| Heuristic (reranker-weighted counts) | 0.2018 | Best previous result |
| IRCoT LLM (azure, best config) | 0.1959 | LLM adds no value over heuristic |
| Entity graph top-1 (BFS shortest path) | 0.1608 | Graph alone, no reranker |
| Entity graph top-3 recall | 0.4181 | Answer in narrowed set 42% of time |
| Graph + heuristic combined top-3 recall | 0.4883 | Signals are complementary |
| graph_llm with router (actual, v1) | 0.2018 | LLM adjudication is a wash in v1 |
| Estimated graph_llm EM (LLM at 60% on top-3) | ~0.30 | Theoretical ceiling if LLM discriminates well |

### Why graph_llm hasn't reached its ceiling

The graph_llm matches the heuristic (0.2018 EM) but doesn't beat it. Analysis of the 342 examples:
- Answer in combined top-3: 167/342 (48.8%) — this is the pool the LLM works with
- Answer already at position 1 when in top-3: 65/167 (38.9%) — LLM can't help here, just needs to not break it
- LLM could only help on 102 examples where the answer is at position 2 or 3
- Early stopping (score gap > 0.35) fires on 69 examples but is only 21.7% accurate — this actively hurts performance
- For the LLM to reach ~0.30 EM, it would need ~60% accuracy when the answer is in its top-3, but it currently doesn't discriminate well enough between functional interactions and incidental co-mentions

### Why graph_llm should work (design rationale)

The graph_llm generator splits work by what each tool is best at:
- **Graph (regex + BFS)**: exhaustive chain discovery over opaque tokens — finds ALL paths from query drug to candidates
- **Reranker heuristic**: textual relevance signal — which candidates appear in relevant passages
- **LLM**: evidence quality judgment over pre-identified chains — "does this chain describe a functional drug interaction or an incidental gene survey co-mention?"

The LLM's new task is reading comprehension (its strength), not opaque token pattern matching (its weakness). It sees the specific passages forming each candidate's chain, organized by link, and judges which chain is strongest.
