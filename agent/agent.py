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
    """Closed-domain: rerank the pre-bundled support passages directly."""
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
) -> AgentResult:
    """
    Run the IRCoT agent loop on a single MedHop example.

    Args:
        reranker:     Pre-loaded Reranker instance (avoids reload per call).
        retriever_fn: Callable(query, k=int) -> list[str]. Required for open-domain.
                      In closed-domain mode this is ignored.
    """
    if config is None:
        config = AgentConfig()

    query_drug = _extract_query_drug(query)

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

        if hop >= config.min_hops_before_stop:
            should_stop = (
                score_gap >= config.score_gap_threshold
                and result.confidence >= config.confidence_threshold
                and result.answer is not None
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

        if result.answer is not None and result.confidence >= 0.95:
            final_answer = result.answer
            early_stopped = True
            break

    if final_answer is None:
        if traces and traces[-1].result.answer is not None:
            final_answer = traces[-1].result.answer
        else:
            final_answer = generator.extract_answer(candidates, memory.evidence)

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
