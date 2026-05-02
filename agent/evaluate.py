"""
End-to-end evaluation on MedHop validation.

Usage (from repo root):
  python agent/evaluate.py --closed_domain
  python agent/evaluate.py --open_domain
  python agent/evaluate.py --generator azure --azure_deployment gpt-4.1
  python agent/evaluate.py --ablation
  python agent/evaluate.py --generator heuristic --max_examples 5
"""
import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

from agent.config import AgentConfig, setup_paths
from agent.pipeline import run_pipeline

setup_paths()


def _load_reranker():
    from reranker.reranker import Reranker

    ckpt = _REPO_ROOT / "reranker" / "checkpoints" / "finetuned"
    model = str(ckpt) if ckpt.exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Loading reranker: {model}")
    return Reranker(model_name=model)


def _load_retriever():
    """Load Person A's retriever for open-domain mode."""
    from retriever import retrieve
    # Trigger lazy loading so the model is ready before the eval loop
    print("Loading retriever...")
    retrieve("test", k=1)
    return retrieve


def _load_val():
    from datasets import load_dataset

    ds = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    return ds["validation"]


def run_evaluation(
    val_split,
    config: AgentConfig,
    use_router: bool = True,
    max_examples: int | None = None,
    reranker=None,
    retriever_fn=None,
) -> dict:
    from tqdm import tqdm
    from eval_harness.metrics import exact_match

    n = min(len(val_split), max_examples) if max_examples else len(val_split)
    results = []

    for i in tqdm(range(n), desc="Evaluating"):
        out = run_pipeline(
            val_split[i], config=config, use_router=use_router,
            reranker=reranker, retriever_fn=retriever_fn,
        )
        results.append(out)

    em_scores = [exact_match(r["prediction"], r["gold"]) for r in results]
    em = sum(em_scores) / len(em_scores)

    single = [r for r in results if r["router_decision"] == "single"]
    multi = [r for r in results if r["router_decision"] == "multi"]

    single_em = (
        sum(exact_match(r["prediction"], r["gold"]) for r in single) / len(single)
        if single
        else 0.0
    )
    multi_em = (
        sum(exact_match(r["prediction"], r["gold"]) for r in multi) / len(multi)
        if multi
        else 0.0
    )

    return {
        "exact_match": round(em, 4),
        "n_examples": len(results),
        "n_correct": int(sum(em_scores)),
        "avg_hops": round(sum(r["hops_used"] for r in results) / len(results), 2),
        "early_stop_rate": round(sum(r["early_stopped"] for r in results) / len(results), 4),
        "single_hop_em": round(single_em, 4),
        "multi_hop_em": round(multi_em, 4),
        "n_single": len(single),
        "n_multi": len(multi),
        "generator": config.generator,
        "eval_mode": config.eval_mode,
        "use_router": use_router,
    }


def run_ablations(val_split, reranker, retriever_fn, max_examples: int | None = None) -> dict:
    from eval_harness.runner import append_to_leaderboard

    ablations = {}
    has_retriever = retriever_fn is not None

    # --- Closed-domain ablations (always available) ---

    print("\n=== Closed-domain: Heuristic, single-hop ===")
    cfg = AgentConfig(generator="heuristic", eval_mode="closed", max_hops=1)
    ablations["closed_heuristic_single"] = run_evaluation(
        val_split, cfg, use_router=False, max_examples=max_examples, reranker=reranker,
    )
    append_to_leaderboard("end_to_end", "closed_heuristic_single", ablations["closed_heuristic_single"])

    print("\n=== Closed-domain: Heuristic, multi-hop (no router) ===")
    cfg = AgentConfig(generator="heuristic", eval_mode="closed", max_hops=3)
    ablations["closed_heuristic_multi"] = run_evaluation(
        val_split, cfg, use_router=False, max_examples=max_examples, reranker=reranker,
    )
    append_to_leaderboard("end_to_end", "closed_heuristic_multi", ablations["closed_heuristic_multi"])

    print("\n=== Closed-domain: Heuristic, with router ===")
    ablations["closed_heuristic_router"] = run_evaluation(
        val_split, cfg, use_router=True, max_examples=max_examples, reranker=reranker,
    )
    append_to_leaderboard("end_to_end", "closed_heuristic_router", ablations["closed_heuristic_router"])

    # --- Open-domain ablations (require retriever data) ---

    if has_retriever:
        print("\n=== Open-domain: Heuristic, single-hop ===")
        cfg = AgentConfig(generator="heuristic", eval_mode="open", max_hops=1)
        ablations["open_heuristic_single"] = run_evaluation(
            val_split, cfg, use_router=False, max_examples=max_examples,
            reranker=reranker, retriever_fn=retriever_fn,
        )
        append_to_leaderboard("end_to_end", "open_heuristic_single", ablations["open_heuristic_single"])

        print("\n=== Open-domain: Heuristic, multi-hop (no router) ===")
        cfg = AgentConfig(generator="heuristic", eval_mode="open", max_hops=3)
        ablations["open_heuristic_multi"] = run_evaluation(
            val_split, cfg, use_router=False, max_examples=max_examples,
            reranker=reranker, retriever_fn=retriever_fn,
        )
        append_to_leaderboard("end_to_end", "open_heuristic_multi", ablations["open_heuristic_multi"])

        print("\n=== Open-domain: Heuristic, with router ===")
        ablations["open_heuristic_router"] = run_evaluation(
            val_split, cfg, use_router=True, max_examples=max_examples,
            reranker=reranker, retriever_fn=retriever_fn,
        )
        append_to_leaderboard("end_to_end", "open_heuristic_router", ablations["open_heuristic_router"])
    else:
        print("\nSkipping open-domain ablations (retriever/data/ not found — run retriever/data_prep.py first)")

    # --- LLM ablations (require API key) ---

    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        mode = "open" if has_retriever else "closed"
        rtr = retriever_fn if has_retriever else None

        print(f"\n=== {mode.title()}-domain: Azure OpenAI, no router ===")
        cfg_llm = AgentConfig(generator="azure", eval_mode=mode, max_hops=3)
        ablations["llm_no_router"] = run_evaluation(
            val_split, cfg_llm, use_router=False, max_examples=max_examples,
            reranker=reranker, retriever_fn=rtr,
        )
        append_to_leaderboard("end_to_end", "llm_no_router", ablations["llm_no_router"])

        print(f"\n=== {mode.title()}-domain: Azure OpenAI, with router ===")
        ablations["llm_with_router"] = run_evaluation(
            val_split, cfg_llm, use_router=True, max_examples=max_examples,
            reranker=reranker, retriever_fn=rtr,
        )
        append_to_leaderboard("end_to_end", "llm_with_router", ablations["llm_with_router"])
    else:
        print("\nSkipping Azure OpenAI ablations (AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT not set)")

    return ablations


def main():
    parser = argparse.ArgumentParser(description="End-to-end MedHop evaluation")
    parser.add_argument("--generator", choices=["heuristic", "azure"], default="heuristic")
    parser.add_argument("--no_router", action="store_true")
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--azure_deployment", default="gpt-4.1")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--open_domain", action="store_const", const="open", dest="eval_mode",
        help="Retrieve from the full MedHop corpus (requires retriever/data/)",
    )
    mode_group.add_argument(
        "--closed_domain", action="store_const", const="closed", dest="eval_mode",
        help="Rank within each question's support passages only",
    )
    parser.set_defaults(eval_mode="open")

    args = parser.parse_args()

    print("Loading MedHop validation split...")
    val_split = _load_val()
    print(f"  {len(val_split)} examples")

    reranker = _load_reranker()

    # Load retriever for open-domain mode
    retriever_fn = None
    if args.eval_mode == "open" or args.ablation:
        corpus_path = _REPO_ROOT / "retriever" / "data" / "corpus.json"
        if corpus_path.exists():
            retriever_fn = _load_retriever()
        elif args.eval_mode == "open" and not args.ablation:
            print(
                f"\nERROR: Open-domain mode requires {corpus_path}.\n"
                "Run `cd retriever && python data_prep.py` first, or use --closed_domain."
            )
            sys.exit(1)
        else:
            print(f"\nNote: {corpus_path} not found — open-domain ablations will be skipped.")

    if args.ablation:
        ablations = run_ablations(val_split, reranker, retriever_fn, max_examples=args.max_examples)
        print("\n=== Ablation Summary ===")
        for name, metrics in ablations.items():
            print(
                f"  {name:<30} EM={metrics['exact_match']:.4f}  "
                f"hops={metrics['avg_hops']:.1f}  "
                f"early_stop={metrics['early_stop_rate']:.2%}  "
                f"mode={metrics['eval_mode']}"
            )
        return

    config = AgentConfig(
        generator=args.generator,
        max_hops=args.max_hops,
        eval_mode=args.eval_mode,
        azure_deployment=args.azure_deployment,
    )
    use_router = not args.no_router
    run_name = args.run_name or f"{args.eval_mode}_{args.generator}_{'router' if use_router else 'no_router'}"

    print(
        f"\nConfig: generator={config.generator}, max_hops={config.max_hops}, "
        f"eval_mode={config.eval_mode}, router={'on' if use_router else 'off'}"
    )

    metrics = run_evaluation(
        val_split, config, use_router=use_router, max_examples=args.max_examples,
        reranker=reranker, retriever_fn=retriever_fn,
    )

    print(f"\n=== Results: {run_name} ===")
    print(f"  Exact Match:     {metrics['exact_match']:.4f}")
    print(f"  Correct:         {metrics['n_correct']}/{metrics['n_examples']}")
    print(f"  Avg hops:        {metrics['avg_hops']:.2f}")
    print(f"  Early stop rate: {metrics['early_stop_rate']:.2%}")
    print(f"  Single-hop EM:   {metrics['single_hop_em']:.4f} ({metrics['n_single']} examples)")
    print(f"  Multi-hop EM:    {metrics['multi_hop_em']:.4f} ({metrics['n_multi']} examples)")

    from eval_harness.runner import append_to_leaderboard

    append_to_leaderboard("end_to_end", run_name, metrics)
    print(f"\nResults appended to results.json under 'end_to_end' > '{run_name}'")


if __name__ == "__main__":
    main()
