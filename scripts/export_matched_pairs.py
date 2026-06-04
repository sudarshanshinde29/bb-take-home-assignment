#!/usr/bin/env python3
"""
Export rows from matches.csv that have a non-empty item_id_B (actual A→B pairs).

matches.csv is one row per Store A product (233,199 when complete). Rows with no
confident substitute keep item_id_B blank — those are non-matches, not missing data.

Usage (from repo root):
  python3 scripts/export_matched_pairs.py
  python3 scripts/export_matched_pairs.py -o matches_only.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "matches.csv"
DEFAULT_OUTPUT = REPO_ROOT / "matches_only.csv"


def export_matched_pairs(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Write rows with non-empty item_id_B. Returns (total_rows, matched_rows)."""
    with input_path.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames or "item_id_A" not in reader.fieldnames:
            raise ValueError(f"{input_path}: expected columns item_id_A, item_id_B")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    matched = [
        r
        for r in rows
        if (r.get("item_id_B") or "").strip()
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matched)

    return len(rows), len(matched)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter matches.csv to rows with a Store B match (non-empty item_id_B).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT.name})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    total, matched = export_matched_pairs(args.input, args.output)
    empty = total - matched
    pct = (100.0 * matched / total) if total else 0.0
    print(f"Read {total:,} rows from {args.input}")
    print(f"  with item_id_B (matched):   {matched:,} ({pct:.1f}%)")
    print(f"  empty item_id_B (no match): {empty:,}")
    print(f"Wrote {matched:,} rows to {args.output}")


if __name__ == "__main__":
    main()
