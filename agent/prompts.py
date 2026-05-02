"""
Prompt templates for the IRCoT agent on MedHop.
"""

REASONING_PROMPT = """\
You are a biomedical reasoning agent answering drug interaction questions.

QUESTION: {query}

CANDIDATE ANSWERS (you must eventually choose one of these): {candidates}

{memory_section}

PASSAGES (ranked by relevance):
{passages}

INSTRUCTIONS:
1. Read the passages carefully.
2. Identify all DrugBank IDs (format: DBXXXXX) mentioned in these passages.
3. Trace interaction chains: which drugs interact with {query_drug}? Which drugs interact with those?
4. Note any direct or indirect evidence linking {query_drug} to any candidate answer.

Respond in this exact format:
ENTITIES: [comma-separated DrugBank IDs found in passages]
CHAIN: [describe the interaction chain you've found so far, e.g., "DB00773 -> DB00123 -> DB00456"]
EVIDENCE: [1-2 sentence summary of key evidence]
CONFIDENCE: [0.0 to 1.0, how confident you are that you can identify the answer]
ANSWER: [if confidence >= 0.8, state the DrugBank ID from candidates; otherwise write UNCERTAIN]
"""

ANSWER_PROMPT = """\
You are a biomedical reasoning agent. Based on all the evidence gathered, select the final answer.

QUESTION: {query}
CANDIDATE ANSWERS: {candidates}

ACCUMULATED EVIDENCE:
{evidence_summary}

Select exactly one DrugBank ID from the candidates list that best answers the question.
Respond with ONLY the DrugBank ID (e.g., DB00072). Nothing else.
"""

MEMORY_TEMPLATE = """\
PREVIOUS REASONING (from earlier hops):
Entities found: {entities}
Interaction chain: {chain}
Evidence so far: {evidence}
"""


def format_passages(ranked_passages: list[tuple[str, float]], max_chars: int = 8000) -> str:
    lines = []
    total = 0
    for i, (text, score) in enumerate(ranked_passages, 1):
        entry = f"[Passage {i}, relevance={score:.2f}]\n{text}\n"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


def format_candidates(candidates: list[str]) -> str:
    return ", ".join(candidates)
