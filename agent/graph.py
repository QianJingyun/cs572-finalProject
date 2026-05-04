"""
Entity co-occurrence graph for chain discovery in MedHop.

Builds a bipartite graph of entity co-occurrences from passages, then
uses BFS to find evidence chains from the query drug to each candidate.
"""
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

ENTITY_RE = re.compile(r"(?:DB\d{5}|[PQ]\d{4,5})")


@dataclass
class Chain:
    candidate: str
    path: list[str]
    passage_indices: list[int]
    depth: int


@dataclass
class CandidateScore:
    candidate: str
    graph_score: float
    best_chain: Chain | None
    n_shortest_paths: int


@dataclass
class WeakLink:
    entity_a: str
    entity_b: str
    passage_idx: int
    edge_weight: int
    passage_text: str


def extract_entities(text: str) -> set[str]:
    return set(ENTITY_RE.findall(text))


def build_graph(passages: list[str]):
    """Returns (edge_weight dict, per-passage entity sets)."""
    passage_entities = [extract_entities(s) for s in passages]

    edge_weight: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entities in passage_entities:
        elist = list(entities)
        for i, a in enumerate(elist):
            for b in elist[i + 1 :]:
                edge_weight[a][b] += 1
                edge_weight[b][a] += 1

    return edge_weight, passage_entities


def _trace_passage_indices(
    path: list[str], passage_entities: list[set[str]]
) -> list[int]:
    """Find one passage per consecutive edge in the path."""
    indices: list[int] = []
    used: set[int] = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for j, entities in enumerate(passage_entities):
            if a in entities and b in entities and j not in used:
                indices.append(j)
                used.add(j)
                break
        else:
            indices.append(-1)
    return indices


def find_chains(
    query_drug: str,
    candidates: list[str],
    edge_weight: dict,
    passage_entities: list[set[str]],
    max_depth: int = 4,
) -> dict[str, list[Chain]]:
    """BFS to find all shortest chains from query_drug to each candidate."""
    cand_set = set(candidates)
    result: dict[str, list[Chain]] = defaultdict(list)
    best_depth: dict[str, int] = {}

    visited = {query_drug}
    queue: deque[tuple[str, list[str]]] = deque([(query_drug, [query_drug])])

    while queue:
        current, path = queue.popleft()
        depth = len(path) - 1
        if depth >= max_depth:
            continue

        for neighbor in edge_weight.get(current, {}):
            new_depth = depth + 1
            if neighbor in cand_set:
                prev = best_depth.get(neighbor, max_depth + 1)
                if new_depth <= prev:
                    best_depth[neighbor] = new_depth
                    new_path = path + [neighbor]
                    psg_idx = _trace_passage_indices(new_path, passage_entities)
                    result[neighbor].append(
                        Chain(neighbor, new_path, psg_idx, new_depth)
                    )

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return dict(result)


def score_candidates(
    query_drug: str,
    candidates: list[str],
    edge_weight: dict,
    passage_entities: list[set[str]],
    max_depth: int = 4,
) -> list[CandidateScore]:
    """Rank candidates by graph connectivity to query_drug."""
    chains = find_chains(
        query_drug, candidates, edge_weight, passage_entities, max_depth
    )

    scores: list[CandidateScore] = []
    for cand in candidates:
        cand_chains = chains.get(cand, [])
        if not cand_chains:
            scores.append(CandidateScore(cand, 0.0, None, 0))
            continue

        min_depth = min(c.depth for c in cand_chains)
        shortest = [c for c in cand_chains if c.depth == min_depth]

        total_weight = 0.0
        best_weight = 0.0
        best_chain = shortest[0]
        for chain in shortest:
            w = 1.0
            for i in range(len(chain.path) - 1):
                a, b = chain.path[i], chain.path[i + 1]
                w *= edge_weight.get(a, {}).get(b, 1)
            total_weight += w
            if w > best_weight:
                best_weight = w
                best_chain = chain

        score = total_weight / (min_depth ** 2)
        scores.append(CandidateScore(cand, score, best_chain, len(shortest)))

    scores.sort(key=lambda s: -s.graph_score)
    return scores


def find_weak_links(
    chain: Chain,
    edge_weight: dict,
    passages: list[str],
) -> list[WeakLink]:
    """Return edges in a chain sorted by weakness (fewest supporting passages)."""
    links: list[WeakLink] = []
    for i in range(len(chain.path) - 1):
        a, b = chain.path[i], chain.path[i + 1]
        w = edge_weight.get(a, {}).get(b, 0)
        psg_idx = chain.passage_indices[i] if i < len(chain.passage_indices) else -1
        psg_text = passages[psg_idx] if 0 <= psg_idx < len(passages) else ""
        links.append(WeakLink(a, b, psg_idx, w, psg_text))
    links.sort(key=lambda l: l.edge_weight)
    return links


def build_gap_query(weak_link: WeakLink, query_drug: str) -> str:
    return f"{weak_link.entity_a} {weak_link.entity_b} {query_drug}"


def get_chain_passages(chain: Chain, passages: list[str]) -> list[tuple[int, str]]:
    """Return (index, text) for each passage in the chain."""
    out: list[tuple[int, str]] = []
    for idx in chain.passage_indices:
        if 0 <= idx < len(passages):
            out.append((idx, passages[idx]))
    return out
