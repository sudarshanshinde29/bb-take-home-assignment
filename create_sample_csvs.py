#!/usr/bin/env python3
"""Create small sample CSVs from the full grocery store item files."""

import csv
from pathlib import Path

# Config
NUM_ROWS = 10
BASE_DIR = Path(__file__).resolve().parent

SOURCES = [
    ("grocery_store_a_items_final.csv", "grocery_store_a_items_sample.csv"),
    ("grocery_store_b_items_final.csv", "grocery_store_b_items_sample.csv"),
]


def write_sample(source: Path, dest: Path, num_rows: int) -> None:
    with source.open(newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        rows = []
        for _ in range(num_rows):
            try:
                rows.append(next(reader))
            except StopIteration:
                break

    with dest.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote {dest.name}: header + {len(rows)} rows (from {source.name})")


def main() -> None:
    for source_name, dest_name in SOURCES:
        source = BASE_DIR / source_name
        dest = BASE_DIR / dest_name

        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")

        write_sample(source, dest, NUM_ROWS)


if __name__ == "__main__":
    main()
