#!/usr/bin/env python3
"""
Run the full product matching pipeline (ARCHITECTURE.md).

  1. build_features.py  — cache/store_*_features.jsonl
  2. match_products.py  — matches.csv (+ optional LLM judge)

Examples:
  python3 run_matching.py                    # features + match + inline LLM judge
  python3 run_matching.py --skip-judge         # local matching only (faster)
  python3 run_matching.py --limit 500
  python3 run_matching.py --limit 200 --judge-max 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature build + matching pipeline.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-max", type=int, default=None)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    bf = [sys.executable, str(SCRIPTS / "build_features.py")]
    mp = [sys.executable, str(SCRIPTS / "match_products.py")]
    if args.limit is not None:
        bf.extend(["--limit", str(args.limit)])
        mp.extend(["--limit", str(args.limit)])
    if args.rebuild_features:
        mp.append("--rebuild-features")
    if args.skip_judge:
        mp.append("--skip-judge")
    if args.judge_max is not None:
        mp.extend(["--judge-max", str(args.judge_max)])
    if args.no_progress:
        mp.append("--no-progress")

    subprocess.check_call(bf, cwd=str(ROOT))
    subprocess.check_call(mp, cwd=str(ROOT))


if __name__ == "__main__":
    main()
