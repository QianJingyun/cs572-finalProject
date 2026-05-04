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


CHAIN_EVAL_PROMPT = """\
You are a biomedical expert evaluating evidence for drug interactions.

QUESTION: {query}

Below are {n_candidates} candidate answers. For each, an evidence chain shows \
how the query drug ({query_drug}) connects to that candidate through shared \
biological entities (proteins, enzymes, receptors). Each link in the chain is \
supported by a passage from the biomedical literature.

{chain_evidence}

EVALUATION CRITERIA:
- STRONG evidence: passages describe functional relationships (enzyme inhibition, \
receptor binding, metabolic pathways, pharmacological effects, shared mechanism of action)
- WEAK evidence: incidental co-mention (gene surveys, genotyping studies, literature \
reviews listing many unrelated entities in a table or list)
- Shorter chains with direct functional evidence are stronger than longer chains \
through gene catalogues
- Consider whether the connecting entities (proteins/enzymes) mediate a plausible \
drug-drug interaction mechanism

Select the candidate with the strongest evidence for a genuine interaction with {query_drug}.
Respond with ONLY the DrugBank ID (e.g., DB00123). Nothing else.
"""


def format_chain_evidence(
    candidate_chains: list[tuple[str, list[str], list[str]]],
) -> str:
    """Format candidate chains for the CHAIN_EVAL_PROMPT.

    Each element is (candidate_id, path_as_strings, passage_texts).
    """
    parts: list[str] = []
    for rank, (cand, path, passages) in enumerate(candidate_chains, 1):
        header = (
            f"--- CANDIDATE {rank}: {cand} (chain depth {len(path) - 1}) ---\n"
            f"Path: {' -> '.join(path)}"
        )
        psg_lines: list[str] = []
        for i, text in enumerate(passages):
            trimmed = text[:600] if len(text) > 600 else text
            psg_lines.append(f"  [Link {i + 1}] {trimmed}")
        parts.append(header + "\n" + "\n".join(psg_lines))
    return "\n\n".join(parts)


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
