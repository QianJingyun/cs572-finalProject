"""
Run end-to-end pipeline evaluation from repo root.

Usage:
  python scripts/evaluate_pipeline.py
  python scripts/evaluate_pipeline.py --generator azure --max_examples 50
  python scripts/evaluate_pipeline.py --ablation
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.evaluate import main

if __name__ == "__main__":
    main()
