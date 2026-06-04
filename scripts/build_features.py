#!/usr/bin/env python3
"""
Step 1: Build normalized feature JSONL files for Store A and Store B.

Reads grocery_store_*_items_final.csv and writes:
  cache/store_a_features.jsonl
  cache/store_b_features.jsonl

Store B uses Python-only parsing; Store A uses brand vocabulary from B plus
Walmart PL heuristics. See ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tqdm import tqdm

from matching_lib import build_brand_vocab_from_b, extract_features

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_A = REPO_ROOT / "grocery_store_a_items_final.csv"
STORE_B = REPO_ROOT / "grocery_store_b_items_final.csv"
CACHE_DIR = REPO_ROOT / "cache"
OUT_A = CACHE_DIR / "store_a_features.jsonl"
OUT_B = CACHE_DIR / "store_b_features.jsonl"


def load_csv_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    return rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature JSONL caches for A and B.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per store (for testing)")
    args = parser.parse_args()

    if not STORE_A.exists() or not STORE_B.exists():
        raise SystemExit(f"Missing CSV files under {REPO_ROOT}")

    print("Loading Store B...")
    rows_b = load_csv_rows(STORE_B, args.limit)
    print(f"  {len(rows_b):,} rows")

    feats_b = [extract_features(r, "B") for r in tqdm(rows_b, desc="Features Store B", unit="row")]
    write_jsonl(OUT_B, feats_b)
    print(f"Wrote {OUT_B}")

    brand_vocab = build_brand_vocab_from_b(rows_b)
    print(f"Brand vocabulary size: {len(brand_vocab):,}")

    print("Loading Store A...")
    rows_a = load_csv_rows(STORE_A, args.limit)
    print(f"  {len(rows_a):,} rows")

    feats_a = [
        extract_features(r, "A", brand_vocab=brand_vocab)
        for r in tqdm(rows_a, desc="Features Store A", unit="row")
    ]
    write_jsonl(OUT_A, feats_a)
    print(f"Wrote {OUT_A}")

    with_brand_a = sum(1 for f in feats_a if f.get("brand_normalized"))
    with_brand_b = sum(1 for f in feats_b if f.get("brand_normalized"))
    pl_a = sum(1 for f in feats_a if f.get("is_private_label"))
    pl_b = sum(1 for f in feats_b if f.get("is_private_label"))
    print(f"Store A: brand populated {with_brand_a:,} | private label {pl_a:,}")
    print(f"Store B: brand populated {with_brand_b:,} | private label {pl_b:,}")


if __name__ == "__main__":
    main()
