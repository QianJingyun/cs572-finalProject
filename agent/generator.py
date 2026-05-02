"""
Generator backends for the agent loop.

HeuristicGenerator:  no API, counts candidate mentions weighted by reranker score.
AzureOpenAIGenerator: Azure OpenAI chat completions for reasoning + answer extraction.
"""
import os
import re
from dataclasses import dataclass

from .prompts import (
    REASONING_PROMPT,
    ANSWER_PROMPT,
    MEMORY_TEMPLATE,
    format_passages,
    format_candidates,
)

DRUG_ID_RE = re.compile(r"DB\d{5}")


@dataclass
class ReasoningResult:
    entities: list[str]
    chain: str
    evidence: str
    confidence: float
    answer: str | None


class HeuristicGenerator:
    """Scores candidates by weighted mention count in reranked passages."""

    def reason(
        self,
        query: str,
        query_drug: str,
        candidates: list[str],
        ranked_passages: list[tuple[str, float]],
        memory: dict | None = None,
    ) -> ReasoningResult:
        combined = " ".join(p for p, _ in ranked_passages)
        found_ids = sorted(set(DRUG_ID_RE.findall(combined)))

        candidate_scores: dict[str, float] = {c: 0.0 for c in candidates}
        for passage, score in ranked_passages:
            for cand in candidates:
                if cand in passage:
                    candidate_scores[cand] += score

        if memory and memory.get("entities"):
            for cand in candidates:
                for entity in memory["entities"]:
                    for passage, score in ranked_passages:
                        if cand in passage and entity in passage:
                            candidate_scores[cand] += score * 0.5

        best_cand = max(candidate_scores, key=candidate_scores.get)
        best_score = candidate_scores[best_cand]
        total = sum(candidate_scores.values())
        confidence = best_score / total if total > 0 else 0.0

        intermediates = [e for e in found_ids if e != query_drug and e not in candidates]
        chain_parts = [query_drug] + intermediates[:3]
        if best_cand not in chain_parts:
            chain_parts.append(best_cand)

        return ReasoningResult(
            entities=found_ids,
            chain=" -> ".join(chain_parts),
            evidence=(
                f"{best_cand} appears in "
                f"{sum(1 for p, _ in ranked_passages if best_cand in p)}"
                f"/{len(ranked_passages)} passages"
            ),
            confidence=confidence,
            answer=best_cand if confidence > 0.3 else None,
        )

    def extract_answer(self, candidates: list[str], evidence_summary: str) -> str:
        counts = {c: evidence_summary.count(c) for c in candidates}
        return max(counts, key=counts.get) if any(counts.values()) else candidates[0]


class AzureOpenAIGenerator:
    """Azure OpenAI chat completions for reasoning and answer extraction."""

    def __init__(self, deployment: str = "gpt-4.1", api_version: str = "2024-12-01-preview"):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=api_version,
        )
        self.deployment = deployment

    def _chat(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def reason(
        self,
        query: str,
        query_drug: str,
        candidates: list[str],
        ranked_passages: list[tuple[str, float]],
        memory: dict | None = None,
    ) -> ReasoningResult:
        memory_section = ""
        if memory and memory.get("evidence"):
            memory_section = MEMORY_TEMPLATE.format(
                entities=", ".join(memory.get("entities", [])),
                chain=memory.get("chain", "none yet"),
                evidence=memory.get("evidence", "none yet"),
            )

        prompt = REASONING_PROMPT.format(
            query=query,
            candidates=format_candidates(candidates),
            memory_section=memory_section,
            passages=format_passages(ranked_passages),
            query_drug=query_drug,
        )

        text = self._chat(prompt, max_tokens=512)
        return self._parse_reasoning(text, candidates)

    def extract_answer(self, candidates: list[str], evidence_summary: str) -> str:
        prompt = ANSWER_PROMPT.format(
            query="(see evidence below)",
            candidates=format_candidates(candidates),
            evidence_summary=evidence_summary,
        )
        text = self._chat(prompt, max_tokens=32)
        match = DRUG_ID_RE.search(text)
        if match and match.group(0) in candidates:
            return match.group(0)
        for c in candidates:
            if c in text:
                return c
        return candidates[0]

    def _parse_reasoning(self, text: str, candidates: list[str]) -> ReasoningResult:
        entities: list[str] = []
        chain = ""
        evidence = ""
        confidence = 0.0
        answer = None

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("ENTITIES:"):
                raw = line[len("ENTITIES:"):].strip().strip("[]")
                entities = [e.strip() for e in raw.split(",") if DRUG_ID_RE.match(e.strip())]
            elif line.startswith("CHAIN:"):
                chain = line[len("CHAIN:"):].strip()
            elif line.startswith("EVIDENCE:"):
                evidence = line[len("EVIDENCE:"):].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[len("CONFIDENCE:"):].strip())
                except ValueError:
                    confidence = 0.0
            elif line.startswith("ANSWER:"):
                raw_answer = line[len("ANSWER:"):].strip()
                if raw_answer != "UNCERTAIN":
                    match = DRUG_ID_RE.search(raw_answer)
                    if match and match.group(0) in candidates:
                        answer = match.group(0)

        return ReasoningResult(
            entities=entities,
            chain=chain,
            evidence=evidence,
            confidence=min(max(confidence, 0.0), 1.0),
            answer=answer,
        )
