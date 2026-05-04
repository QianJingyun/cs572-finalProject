"""
Generator backends for the agent loop.

HeuristicGenerator:  no API, counts candidate mentions weighted by reranker score.
AzureOpenAIGenerator: Azure OpenAI chat completions for reasoning + answer extraction.
"""
import logging
import os
import re
from dataclasses import dataclass

from .prompts import (
    REASONING_PROMPT,
    ANSWER_PROMPT,
    MEMORY_TEMPLATE,
    CHAIN_EVAL_PROMPT,
    format_passages,
    format_candidates,
    format_chain_evidence,
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

    def extract_answer(self, candidates: list[str], evidence_summary: str, query: str = "") -> str:
        counts = {c: evidence_summary.count(c) for c in candidates}
        return max(counts, key=counts.get) if any(counts.values()) else candidates[0]


class AzureOpenAIGenerator:
    """Azure OpenAI chat completions for reasoning and answer extraction."""

    def __init__(
        self,
        deployment: str = "gpt-4.1",
        api_version: str = "2024-12-01-preview",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
    ):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=azure_endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=api_version,
        )
        self.deployment = deployment

    @classmethod
    def from_endpoint(cls, endpoint):
        """Create from an AzureEndpoint dataclass."""
        return cls(
            deployment=endpoint.deployment,
            api_version=endpoint.api_version,
            api_key=endpoint.api_key,
            azure_endpoint=endpoint.endpoint,
        )

    def _chat(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.getLogger(__name__).warning("Azure API error: %s", e)
            return "ENTITIES: []\nCHAIN:\nEVIDENCE:\nCONFIDENCE: 0.0\nANSWER: UNCERTAIN"

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

    def extract_answer(self, candidates: list[str], evidence_summary: str, query: str = "") -> str:
        prompt = ANSWER_PROMPT.format(
            query=query or "(see evidence below)",
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

    def evaluate_chains(
        self,
        query: str,
        query_drug: str,
        candidates: list[str],
        candidate_chains: list[tuple[str, list[str], list[str]]],
    ) -> str:
        """Pick the best candidate from pre-computed evidence chains."""
        prompt = CHAIN_EVAL_PROMPT.format(
            query=query,
            query_drug=query_drug,
            n_candidates=len(candidate_chains),
            chain_evidence=format_chain_evidence(candidate_chains),
        )
        text = self._chat(prompt, max_tokens=64)
        match = DRUG_ID_RE.search(text)
        if match and match.group(0) in candidates:
            return match.group(0)
        for c in candidates:
            if c in text:
                return c
        return candidate_chains[0][0] if candidate_chains else candidates[0]

    def _parse_reasoning(self, text: str, candidates: list[str]) -> ReasoningResult:
        entities: list[str] = []
        chain = ""
        evidence = ""
        confidence = 0.0
        answer = None

        for line in text.strip().split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("ENTITIES"):
                raw = stripped.split(":", 1)[-1].strip().strip("[]")
                entities = [e.strip() for e in raw.split(",") if DRUG_ID_RE.match(e.strip())]
            elif upper.startswith("CHAIN"):
                chain = stripped.split(":", 1)[-1].strip()
            elif upper.startswith("EVIDENCE"):
                evidence = stripped.split(":", 1)[-1].strip()
            elif upper.startswith("CONFIDENCE"):
                try:
                    confidence = float(stripped.split(":", 1)[-1].strip())
                except ValueError:
                    confidence = 0.0
            elif upper.startswith("ANSWER"):
                raw_answer = stripped.split(":", 1)[-1].strip()
                if raw_answer.upper() != "UNCERTAIN":
                    match = DRUG_ID_RE.search(raw_answer)
                    if match and match.group(0) in candidates:
                        answer = match.group(0)

        # Fix #2: accept any valid candidate found in the text even if
        # the structured ANSWER field wasn't parsed correctly.
        if answer is None:
            for match in DRUG_ID_RE.finditer(text):
                if match.group(0) in candidates:
                    answer = match.group(0)
                    break

        return ReasoningResult(
            entities=entities,
            chain=chain,
            evidence=evidence,
            confidence=min(max(confidence, 0.0), 1.0),
            answer=answer,
        )
