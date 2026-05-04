"""
IRCoT-style agent loop for multi-hop biomedical QA.

Each hop: retrieve/rerank passages, feed top-k to the generator for
reasoning, accumulate findings in condensed memory, optionally stop early.

Two eval modes:
  closed: rank within each question's pre-bundled support passages (reranker only)
  open:   retrieve from the full MedHop corpus first, then rerank (full pipeline)
"""
import re
import logging
from dataclasses import dataclass, field

from .config import AgentConfig, REPO_ROOT, setup_paths
from .generator import ReasoningResult, HeuristicGenerator, AzureOpenAIGenerator
from .graph import (
    build_graph,
    score_candidates as graph_score_candidates,
    find_weak_links,
    build_gap_query,
    get_chain_passages,
)

setup_paths()

DRUG_ID_RE = re.compile(r"DB\d{5}")
logger = logging.getLogger(__name__)


@dataclass
class HopTrace:
    hop: int
    query_used: str
    top_passages: list[tuple[str, float]]
    result: ReasoningResult
    score_gap: float
    stopped_early: bool


@dataclass
class AgentResult:
    answer: str
    hops_used: int
    traces: list[HopTrace]
    early_stopped: bool
    method: str


@dataclass
class Memory:
    entities: list[str] = field(default_factory=list)
    chain: str = ""
    evidence: str = ""

    def update(self, result: ReasoningResult):
        seen = set(self.entities)
        for e in result.entities:
            if e not in seen:
                self.entities.append(e)
                seen.add(e)
        if result.chain:
            self.chain = result.chain
        if result.evidence:
            parts = self.evidence.split(" | ") if self.evidence else []
            parts.append(result.evidence)
            self.evidence = " | ".join(parts[-2:])

    def to_dict(self) -> dict:
        return {"entities": self.entities, "chain": self.chain, "evidence": self.evidence}


def _build_refined_query(original_query: str, query_drug: str, memory: Memory) -> str:
    if not memory.entities:
        return original_query
    intermediates = [e for e in memory.entities if e != query_drug and e not in original_query]
    if not intermediates:
        return original_query
    return f"{original_query} {' '.join(intermediates[:3])}"


def _extract_query_drug(query: str) -> str:
    match = DRUG_ID_RE.search(query)
    return match.group(0) if match else ""


def _get_passages_closed(supports, current_query, reranker, top_k):
    """Closed-domain: rerank the pre-bundled support passages directly.

    Returns (full_ranked, top_k_passages). The generator only sees top_k,
    so a refined query on a later hop can surface a different top-k set.
    """
    ranked = reranker.rerank(current_query, supports, top_k=None)
    return ranked, ranked[:top_k]


def _get_passages_open(current_query, retriever_fn, reranker, retrieval_k, top_k):
    """Open-domain: retrieve from global corpus, then rerank."""
    retrieved = retriever_fn(current_query, k=retrieval_k)
    if not retrieved:
        return [], []
    ranked = reranker.rerank(current_query, retrieved, top_k=None)
    return ranked, ranked[:top_k]


def run_agent(
    query: str,
    candidates: list[str],
    supports: list[str],
    config: AgentConfig | None = None,
    hop_decision: str = "multi",
    reranker=None,
    retriever_fn=None,
    generator=None,
) -> AgentResult:
    """
    Run the IRCoT agent loop on a single MedHop example.

    Args:
        reranker:     Pre-loaded Reranker instance (avoids reload per call).
        retriever_fn: Callable(query, k=int) -> list[str]. Required for open-domain.
                      In closed-domain mode this is ignored.
        generator:    Pre-built generator instance. If None, one is created from config.
    """
    if config is None:
        config = AgentConfig()

    query_drug = _extract_query_drug(query)

    if generator is None:
        if config.generator == "azure":
            generator = AzureOpenAIGenerator(
                deployment=config.azure_deployment,
                api_version=config.azure_api_version,
            )
        else:
            generator = HeuristicGenerator()

    if reranker is None:
        from reranker.reranker import Reranker

        ckpt = REPO_ROOT / "reranker" / "checkpoints" / "finetuned"
        reranker = Reranker(model_name=str(ckpt) if ckpt.exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2")

    open_domain = config.eval_mode == "open"
    if open_domain and retriever_fn is None:
        from retriever import retrieve
        retriever_fn = retrieve

    max_hops = 1 if hop_decision == "single" else config.max_hops
    memory = Memory()
    traces: list[HopTrace] = []
    final_answer = None
    early_stopped = False

    for hop in range(1, max_hops + 1):
        current_query = _build_refined_query(query, query_drug, memory)

        if open_domain:
            ranked, top_k = _get_passages_open(
                current_query, retriever_fn, reranker, config.retrieval_k, config.top_k_per_hop,
            )
        else:
            ranked, top_k = _get_passages_closed(
                supports, current_query, reranker, config.top_k_per_hop,
            )

        if not ranked:
            break

        score_gap = (ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 0.0

        result = generator.reason(
            query=query,
            query_drug=query_drug,
            candidates=candidates,
            ranked_passages=top_k,
            memory=memory.to_dict() if hop > 1 else None,
        )

        memory.update(result)

        trace = HopTrace(
            hop=hop,
            query_used=current_query,
            top_passages=top_k,
            result=result,
            score_gap=score_gap,
            stopped_early=False,
        )
        traces.append(trace)

        if result.answer is not None and hop >= config.min_hops_before_stop:
            should_stop = (
                score_gap >= config.score_gap_threshold
                or result.confidence >= config.confidence_threshold
            )
            if should_stop:
                trace.stopped_early = True
                final_answer = result.answer
                early_stopped = True
                logger.info(
                    "Early stop at hop %d: gap=%.2f, conf=%.2f, answer=%s",
                    hop, score_gap, result.confidence, result.answer,
                )
                break

    if final_answer is None:
        if traces and traces[-1].result.answer is not None:
            final_answer = traces[-1].result.answer
        else:
            final_answer = generator.extract_answer(candidates, memory.evidence, query=query)

    if final_answer not in candidates:
        for c in candidates:
            if c.lower() == final_answer.lower():
                final_answer = c
                break
        else:
            counts = {c: memory.evidence.count(c) for c in candidates}
            final_answer = max(counts, key=counts.get) if any(counts.values()) else candidates[0]

    return AgentResult(
        answer=final_answer,
        hops_used=len(traces),
        traces=traces,
        early_stopped=early_stopped,
        method=config.generator,
    )


# ---------------------------------------------------------------------------
# Graph-guided agent: discover → verify → judge
# ---------------------------------------------------------------------------


def _heuristic_scores(
    query: str, candidates: list[str], supports: list[str], reranker
) -> dict[str, float]:
    """Reranker-weighted mention counts (same signal as HeuristicGenerator)."""
    ranked = reranker.rerank(query, supports, top_k=None)
    scores: dict[str, float] = {c: 0.0 for c in candidates}
    for passage, score in ranked:
        for c in candidates:
            if c in passage:
                scores[c] += score
    return scores


def run_graph_agent(
    query: str,
    candidates: list[str],
    supports: list[str],
    config: AgentConfig | None = None,
    hop_decision: str = "multi",
    reranker=None,
    retriever_fn=None,
    generator=None,
) -> AgentResult:
    """Graph-guided agent: discover chains → fill gaps → LLM judges."""
    if config is None:
        config = AgentConfig()

    query_drug = _extract_query_drug(query)

    if reranker is None:
        from reranker.reranker import Reranker
        ckpt = REPO_ROOT / "reranker" / "checkpoints" / "finetuned"
        reranker = Reranker(
            model_name=str(ckpt) if ckpt.exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    is_azure = config.generator == "graph_llm"
    if is_azure and generator is None:
        generator = AzureOpenAIGenerator(
            deployment=config.azure_deployment,
            api_version=config.azure_api_version,
        )

    open_domain = config.eval_mode == "open"
    if open_domain and retriever_fn is None:
        from retriever import retrieve
        retriever_fn = retrieve

    max_hops = 1 if hop_decision == "single" else config.max_hops
    traces: list[HopTrace] = []
    all_passages = list(supports)

    # ------------------------------------------------------------------
    # Hop 1: Graph discovery + heuristic scoring
    # ------------------------------------------------------------------
    edge_weight, passage_entities = build_graph(all_passages)
    graph_scores = graph_score_candidates(
        query_drug, candidates, edge_weight, passage_entities
    )
    heur_scores = _heuristic_scores(query, candidates, all_passages, reranker)

    g_max = max((s.graph_score for s in graph_scores), default=1.0) or 1.0
    h_max = max(heur_scores.values()) or 1.0
    combined: dict[str, float] = {}
    graph_lookup = {s.candidate: s for s in graph_scores}
    for c in candidates:
        gs = graph_lookup[c].graph_score / g_max if c in graph_lookup else 0.0
        hs = heur_scores[c] / h_max
        combined[c] = 0.4 * gs + 0.6 * hs

    ranked_cands = sorted(combined, key=combined.get, reverse=True)
    top_cand = ranked_cands[0]
    second_score = combined[ranked_cands[1]] if len(ranked_cands) > 1 else 0.0
    score_gap = combined[top_cand] - second_score

    hop1_result = ReasoningResult(
        entities=[query_drug],
        chain=graph_lookup[top_cand].best_chain.path
        if graph_lookup[top_cand].best_chain
        else [query_drug],
        evidence=f"graph+heuristic top={top_cand} score={combined[top_cand]:.3f}",
        confidence=combined[top_cand],
        answer=top_cand,
    )
    traces.append(
        HopTrace(1, query, [], hop1_result, score_gap, stopped_early=False)
    )

    # Early stop: dominant candidate with no ambiguity
    if score_gap > 0.35 and max_hops >= 1:
        logger.info("Graph early stop: %s with gap=%.2f", top_cand, score_gap)
        traces[-1].stopped_early = True
        return AgentResult(top_cand, 1, traces, True, "graph_llm")

    # ------------------------------------------------------------------
    # Hop 2: Gap filling — strengthen/weaken top candidates' chains
    # ------------------------------------------------------------------
    top3 = ranked_cands[:3]
    if max_hops >= 2:
        new_passages_added = False
        for cand in top3:
            cs = graph_lookup.get(cand)
            if cs is None or cs.best_chain is None:
                continue
            weak = find_weak_links(cs.best_chain, edge_weight, all_passages)
            if not weak:
                continue
            weakest = weak[0]
            gap_q = build_gap_query(weakest, query_drug)

            if open_domain and retriever_fn is not None:
                new_psgs = retriever_fn(gap_q, k=config.retrieval_k)
                if new_psgs:
                    reranked = reranker.rerank(gap_q, new_psgs, top_k=5)
                    for text, _sc in reranked:
                        if text not in all_passages:
                            all_passages.append(text)
                            new_passages_added = True
            else:
                reranked = reranker.rerank(gap_q, all_passages, top_k=5)

        if new_passages_added:
            edge_weight, passage_entities = build_graph(all_passages)
            graph_scores = graph_score_candidates(
                query_drug, candidates, edge_weight, passage_entities
            )
            graph_lookup = {s.candidate: s for s in graph_scores}
            heur_scores = _heuristic_scores(query, candidates, all_passages, reranker)
            g_max = max((s.graph_score for s in graph_scores), default=1.0) or 1.0
            h_max = max(heur_scores.values()) or 1.0
            for c in candidates:
                gs = graph_lookup[c].graph_score / g_max if c in graph_lookup else 0.0
                hs = heur_scores[c] / h_max
                combined[c] = 0.4 * gs + 0.6 * hs
            ranked_cands = sorted(combined, key=combined.get, reverse=True)
            top3 = ranked_cands[:3]

        hop2_result = ReasoningResult(
            entities=[query_drug],
            chain=f"gap-fill for {top3}",
            evidence=f"top3={top3} after gap filling",
            confidence=combined[ranked_cands[0]],
            answer=ranked_cands[0],
        )
        traces.append(
            HopTrace(2, f"gap-fill queries for {top3}", [], hop2_result, 0.0, False)
        )

    # ------------------------------------------------------------------
    # Hop 3: LLM adjudication over top candidates' evidence chains
    # ------------------------------------------------------------------
    if is_azure and generator is not None and max_hops >= 3:
        candidate_chains: list[tuple[str, list[str], list[str]]] = []
        for cand in top3:
            cs = graph_lookup.get(cand)
            if cs and cs.best_chain:
                chain_psgs = get_chain_passages(cs.best_chain, all_passages)
                candidate_chains.append((
                    cand,
                    cs.best_chain.path,
                    [text for _, text in chain_psgs],
                ))
            else:
                candidate_chains.append((cand, [query_drug, cand], []))

        if candidate_chains:
            llm_answer = generator.evaluate_chains(
                query, query_drug, candidates, candidate_chains
            )
        else:
            llm_answer = ranked_cands[0]

        hop3_result = ReasoningResult(
            entities=[query_drug],
            chain=f"LLM picked {llm_answer}",
            evidence=f"LLM adjudication over {[c[0] for c in candidate_chains]}",
            confidence=1.0,
            answer=llm_answer,
        )
        traces.append(
            HopTrace(3, "chain evaluation", [], hop3_result, 0.0, False)
        )
        final_answer = llm_answer
    else:
        final_answer = ranked_cands[0]

    if final_answer not in candidates:
        final_answer = ranked_cands[0]

    return AgentResult(
        answer=final_answer,
        hops_used=len(traces),
        traces=traces,
        early_stopped=False,
        method="graph_llm",
    )
