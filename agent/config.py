"""
Configuration for the agent loop and pipeline.
"""
import sys
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).parent.parent


def setup_paths():
    """Add component directories to sys.path so cross-component imports work."""
    for p in [
        str(REPO_ROOT),
        str(REPO_ROOT / "router"),
        str(REPO_ROOT / "retriever"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class AgentConfig:
    generator: str = "heuristic"  # "heuristic" | "azure"
    azure_deployment: str = "gpt-4.1"
    azure_api_version: str = "2024-12-01-preview"

    # "closed" = rank within each question's support passages (no retriever)
    # "open"   = retrieve from the full MedHop corpus, then rerank
    eval_mode: str = "open"
    retrieval_k: int = 30  # passages to retrieve in open-domain mode

    max_hops: int = 3
    top_k_per_hop: int = 5

    score_gap_threshold: float = 2.0
    confidence_threshold: float = 0.8
    min_hops_before_stop: int = 1
