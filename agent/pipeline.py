"""
End-to-end pipeline: Router -> Retriever -> Reranker -> Agent -> Answer
"""
from .config import AgentConfig, setup_paths

setup_paths()


def run_pipeline(
    example: dict,
    config: AgentConfig | None = None,
    use_router: bool = True,
    reranker=None,
    retriever_fn=None,
    generator=None,
) -> dict:
    """
    Run the full pipeline on one MedHop example.

    Args:
        retriever_fn: Callable(query, k=int) -> list[str]. Passed through to the
                      agent for open-domain mode. Ignored in closed-domain mode.
        generator:    Pre-built generator instance. Passed through to the agent.

    Returns dict with prediction, gold, correctness, and diagnostics.
    """
    if config is None:
        config = AgentConfig()

    query = example["query"]
    candidates = example["candidates"]
    supports = example["supports"]
    gold = example["answer"]

    if use_router:
        from router import predict_hop

        hop_decision = predict_hop(query, candidates, supports)
    else:
        hop_decision = "multi"

    if config.generator == "graph_llm":
        from .agent import run_graph_agent

        result = run_graph_agent(
            query=query,
            candidates=candidates,
            supports=supports,
            config=config,
            hop_decision=hop_decision,
            reranker=reranker,
            retriever_fn=retriever_fn,
            generator=generator,
        )
    else:
        from .agent import run_agent

        result = run_agent(
            query=query,
            candidates=candidates,
            supports=supports,
            config=config,
            hop_decision=hop_decision,
            reranker=reranker,
            retriever_fn=retriever_fn,
            generator=generator,
        )

    return {
        "id": example["id"],
        "prediction": result.answer,
        "gold": gold,
        "correct": result.answer.strip().upper() == gold.strip().upper(),
        "hops_used": result.hops_used,
        "early_stopped": result.early_stopped,
        "method": result.method,
        "router_decision": hop_decision,
        "eval_mode": config.eval_mode,
    }
