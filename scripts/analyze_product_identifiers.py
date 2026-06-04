#!/usr/bin/env python3
"""
Product identifier & metadata analysis for BetterBasket take-home data.

WHAT THIS SCRIPT DOES
---------------------
Runs seven read-only analyses on grocery_store_a_items_final.csv (Walmart) and
grocery_store_b_items_final.csv (Wegmans):

  1. Full item_info JSON key inventory (all nesting levels)
  2. UPC/GTIN-like values inside item_info (key-based + filtered regex)
  3. Filtered digit scan on name, url, sizing_comp (+ description sample on A)
  4. Cross-store overlap of extracted product codes (A vs B)
  5. URL internal ID extraction + within-store cross-check vs datapoint_id,
     raw_data_id, item_id
  6. Regex false-positive guards (timestamps, date-like 12–14 digit runs)
  7. Store B tags column: parse lists and scan for barcodes / supplier tokens

WHAT IT HELPS WITH
------------------
Before building the matcher, answers:
  - Can we use a UPC-first exact-match stage? (Check 4)
  - Where do identifiers live if anywhere? (Checks 1–3, 7)
  - Are datapoint_id / raw_data_id the same as URL product tokens? (Check 5)
  - How noisy is a naive 12–14 digit regex? (Check 6)

OUTPUT
------
Prints a summary to stdout and writes:
  analysis_output/identifier_analysis_report.txt

USAGE
-----
From repo root:
  python3 scripts/analyze_product_identifiers.py

Optional:
  python3 scripts/analyze_product_identifiers.py --description-sample 5000
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STORE_A = REPO_ROOT / "grocery_store_a_items_final.csv"
STORE_B = REPO_ROOT / "grocery_store_b_items_final.csv"
OUTPUT_DIR = REPO_ROOT / "analysis_output"
REPORT_PATH = OUTPUT_DIR / "identifier_analysis_report.txt"

# ---------------------------------------------------------------------------
# Patterns & heuristics
# ---------------------------------------------------------------------------

UPC_REGEX = re.compile(r"\b\d{12,14}\b")

IDENTIFIER_KEY_RE = re.compile(
    r"(upc|gtin|barcode|ean|product_?id|sku|item_?id|ext_?id|ic_)",
    re.I,
)

WALMART_URL_ID_RE = re.compile(r"/(\d+)/?$")
WEGMANS_URL_ID_RE = re.compile(r"/product/(\d+)-")

TAG_DIGIT_ONLY_RE = re.compile(r"^\d{12,14}$")


def is_likely_timestamp_or_date(num: str) -> bool:
    """Filter compressed timestamps / date-like false positives for GTIN regex."""
    n = len(num)
    if n not in (12, 13, 14):
        return False

    # 14-digit YYYYMMDDHHMMSS-style (common in scraped payloads)
    if n == 14 and num.startswith(("19", "20")):
        try:
            year, month, day = int(num[:4]), int(num[4:6]), int(num[6:8])
            if 1990 <= year <= 2035 and 1 <= month <= 12 and 1 <= day <= 31:
                return True
        except ValueError:
            pass

    # 12-digit YYYYMMDDxxxx
    if n == 12 and num.startswith(("19", "20")):
        try:
            year, month, day = int(num[:4]), int(num[4:6]), int(num[6:8])
            if 1990 <= year <= 2035 and 1 <= month <= 12 and 1 <= day <= 31:
                return True
        except ValueError:
            pass

    # All same digit or mostly zeros — often placeholders
    if len(set(num)) <= 2:
        return True

    return False


def extract_url_id(url: str, store: str) -> str | None:
    url = (url or "").strip()
    if not url:
        return None
    if store == "A":
        m = WALMART_URL_ID_RE.search(url)
        return m.group(1) if m else None
    m = WEGMANS_URL_ID_RE.search(url)
    return m.group(1) if m else None


def parse_json_field(raw: str) -> Any | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def walk_json(obj: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    """Yield (dot.path, value) for all leaves in nested JSON."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            child = f"{path}.{key}" if path else key
            yield child, val
            yield from walk_json(val, child)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            child = f"{path}[{i}]"
            yield child, val
            yield from walk_json(val, child)


def extract_codes_from_text(text: str, *, apply_filters: bool) -> set[str]:
    if not text:
        return set()
    found = set(UPC_REGEX.findall(text))
    if apply_filters:
        found = {c for c in found if not is_likely_timestamp_or_date(c)}
    return found


def parse_tags(raw: str) -> list[str]:
    """Parse Wegmans tags field (Python-list-like string or JSON array)."""
    raw = (raw or "").strip()
    if not raw or raw == "{}":
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
        if isinstance(val, dict):
            return [str(k) for k in val.keys()]
    except (SyntaxError, ValueError):
        pass
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x) for x in val]
    except json.JSONDecodeError:
        pass
    return [raw]


# ---------------------------------------------------------------------------
# Per-store accumulators
# ---------------------------------------------------------------------------


@dataclass
class StoreStats:
    label: str
    total_rows: int = 0
    item_info_parse_ok: int = 0
    item_info_parse_fail: int = 0
    item_info_keys: Counter = field(default_factory=Counter)
    item_info_identifier_keys: Counter = field(default_factory=Counter)
    rows_with_code_in_item_info: int = 0
    codes_from_item_info_keys: Counter = field(default_factory=Counter)
    codes_from_item_info_regex: Counter = field(default_factory=Counter)
    rows_with_code_in_name: int = 0
    rows_with_code_in_url: int = 0
    rows_with_code_in_sizing: int = 0
    rows_with_code_in_description: int = 0
    all_codes: set = field(default_factory=set)
    url_ids: set = field(default_factory=set)
    url_id_eq_datapoint: int = 0
    url_id_eq_raw_data: int = 0
    url_id_eq_item_id: int = 0
    url_id_rows_with_url: int = 0
    tags_rows: int = 0
    tags_parsed_rows: int = 0
    tag_token_counter: Counter = field(default_factory=Counter)
    rows_with_digit_tag: int = 0
    codes_in_tags: Counter = field(default_factory=Counter)
    regex_raw_hits: int = 0
    regex_filtered_out: int = 0


def analyze_store(
    path: Path,
    store: str,
    *,
    description_sample: int | None,
) -> StoreStats:
    stats = StoreStats(label=f"Store {store} ({path.name})")
    desc_limit = description_sample

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats.total_rows += 1
            row_codes: set[str] = set()

            # --- Check 1 & 2: item_info ---
            info_raw = row.get("item_info") or ""
            info = parse_json_field(info_raw)
            if info_raw.strip() and info is None:
                stats.item_info_parse_fail += 1
            elif info is not None:
                stats.item_info_parse_ok += 1
                for key_path, val in walk_json(info):
                    leaf_key = key_path.split(".")[-1].split("[")[0]
                    stats.item_info_keys[leaf_key] += 1
                    if IDENTIFIER_KEY_RE.search(leaf_key):
                        stats.item_info_identifier_keys[key_path] += 1
                        if val is not None and not isinstance(val, (dict, list)):
                            s = str(val).strip()
                            codes = extract_codes_from_text(s, apply_filters=False)
                            if codes or (s.isdigit() and 12 <= len(s) <= 14):
                                code = next(iter(codes), s if s.isdigit() else None)
                                if code and not is_likely_timestamp_or_date(code):
                                    row_codes.add(code)
                                    stats.codes_from_item_info_keys[code] += 1

                # Regex on full JSON string (filtered)
                for raw_match in UPC_REGEX.findall(info_raw):
                    stats.regex_raw_hits += 1
                    if is_likely_timestamp_or_date(raw_match):
                        stats.regex_filtered_out += 1
                    else:
                        row_codes.add(raw_match)
                        stats.codes_from_item_info_regex[raw_match] += 1

            if row_codes:
                stats.rows_with_code_in_item_info += 1

            # --- Check 3: other columns ---
            name = row.get("name") or ""
            url = row.get("url") or ""
            sizing = row.get("sizing_comp") or ""
            desc = row.get("description") or ""

            name_codes = extract_codes_from_text(name, apply_filters=True)
            if name_codes:
                stats.rows_with_code_in_name += 1
                row_codes |= name_codes

            url_codes = extract_codes_from_text(url, apply_filters=True)
            if url_codes:
                stats.rows_with_code_in_url += 1
                row_codes |= url_codes

            sizing_codes = extract_codes_from_text(sizing, apply_filters=True)
            if sizing_codes:
                stats.rows_with_code_in_sizing += 1
                row_codes |= sizing_codes

            scan_desc = desc_limit is None or stats.total_rows <= desc_limit
            if scan_desc and desc:
                desc_codes = extract_codes_from_text(desc, apply_filters=True)
                if desc_codes:
                    stats.rows_with_code_in_description += 1
                    row_codes |= desc_codes

            stats.all_codes.update(row_codes)

            # --- Check 5: URL ID cross-check ---
            url_id = extract_url_id(url, store)
            if url_id:
                stats.url_id_rows_with_url += 1
                stats.url_ids.add(url_id)
                dp = (row.get("datapoint_id") or "").strip()
                rd = (row.get("raw_data_id") or "").strip()
                iid = (row.get("item_id") or "").strip()
                if url_id == dp:
                    stats.url_id_eq_datapoint += 1
                if url_id == rd:
                    stats.url_id_eq_raw_data += 1
                if url_id == iid:
                    stats.url_id_eq_item_id += 1

            # --- Check 7: Store B tags ---
            if store == "B":
                tags_raw = row.get("tags") or ""
                if tags_raw.strip() and tags_raw.strip() != "{}":
                    stats.tags_rows += 1
                    tokens = parse_tags(tags_raw)
                    if tokens:
                        stats.tags_parsed_rows += 1
                    for tok in tokens:
                        stats.tag_token_counter[tok] += 1
                        if TAG_DIGIT_ONLY_RE.match(tok):
                            stats.rows_with_digit_tag += 1
                            if not is_likely_timestamp_or_date(tok):
                                stats.codes_in_tags[tok] += 1
                                stats.all_codes.add(tok)
                        else:
                            for c in extract_codes_from_text(tok, apply_filters=True):
                                stats.codes_in_tags[c] += 1
                                stats.all_codes.add(c)

    return stats


def format_counter(counter: Counter, top_n: int = 25) -> str:
    lines = []
    for key, count in counter.most_common(top_n):
        lines.append(f"    {key}: {count:,}")
    if not lines:
        lines.append("    (none)")
    return "\n".join(lines)


def build_report(
    stats_a: StoreStats,
    stats_b: StoreStats,
    *,
    description_sample: int | None,
) -> str:
    overlap = stats_a.all_codes & stats_b.all_codes
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 72)
        lines.append(title)
        lines.append("=" * 72)

    lines.append("PRODUCT IDENTIFIER ANALYSIS REPORT")
    lines.append(f"Generated by: scripts/analyze_product_identifiers.py")
    lines.append(f"Store A: {STORE_A.name}")
    lines.append(f"Store B: {STORE_B.name}")

    for stats in (stats_a, stats_b):
        store = "A" if "a_items" in stats.label else "B"
        section(stats.label)
        lines.append(f"Total rows: {stats.total_rows:,}")
        lines.append("")
        lines.append("CHECK 1 — item_info JSON keys (top 25 leaf key names)")
        lines.append(format_counter(stats.item_info_keys))
        lines.append("")
        lines.append("CHECK 1b — Keys matching identifier-like pattern (upc/gtin/...)")
        lines.append(format_counter(stats.item_info_identifier_keys, top_n=15))
        lines.append("")
        lines.append("CHECK 2 — item_info product codes")
        lines.append(f"  Rows with ≥1 code in item_info: {stats.rows_with_code_in_item_info:,}")
        lines.append(f"  item_info parse OK / fail: {stats.item_info_parse_ok:,} / {stats.item_info_parse_fail:,}")
        lines.append(f"  Unique codes from identifier keys: {len(stats.codes_from_item_info_keys):,}")
        lines.append(f"  Unique codes from filtered JSON regex: {len(stats.codes_from_item_info_regex):,}")
        lines.append("  Top codes from identifier keys:")
        lines.append(format_counter(stats.codes_from_item_info_keys, top_n=10))
        lines.append("")
        lines.append("CHECK 3 — Filtered 12–14 digit codes in other columns")
        lines.append(f"  name: {stats.rows_with_code_in_name:,}")
        lines.append(f"  url (text scan, not url_id token): {stats.rows_with_code_in_url:,}")
        lines.append(f"  sizing_comp: {stats.rows_with_code_in_sizing:,}")
        if store == "A" and description_sample:
            lines.append(
                f"  description (first {description_sample:,} rows only): "
                f"{stats.rows_with_code_in_description:,}"
            )
        else:
            lines.append(f"  description: {stats.rows_with_code_in_description:,}")
        lines.append(f"  Unique codes union (all sources): {len(stats.all_codes):,}")
        lines.append("")
        lines.append("CHECK 5 — URL internal ID vs other ID columns")
        lines.append(f"  Rows with extractable url_id: {stats.url_id_rows_with_url:,}")
        lines.append(f"  Unique url_id values: {len(stats.url_ids):,}")
        if stats.url_id_rows_with_url:
            n = stats.url_id_rows_with_url
            lines.append(
                f"  url_id == datapoint_id: {stats.url_id_eq_datapoint:,} "
                f"({100 * stats.url_id_eq_datapoint / n:.2f}%)"
            )
            lines.append(
                f"  url_id == raw_data_id: {stats.url_id_eq_raw_data:,} "
                f"({100 * stats.url_id_eq_raw_data / n:.2f}%)"
            )
            lines.append(
                f"  url_id == item_id: {stats.url_id_eq_item_id:,} "
                f"({100 * stats.url_id_eq_item_id / n:.2f}%)"
            )
        lines.append("")
        lines.append("CHECK 6 — Regex noise in item_info (raw vs filtered)")
        lines.append(f"  Raw \\b\\d{{12,14}}\\b matches in item_info strings: {stats.regex_raw_hits:,}")
        lines.append(f"  Filtered out as timestamp/date-like: {stats.regex_filtered_out:,}")
        lines.append(f"  Kept after filter: {stats.regex_raw_hits - stats.regex_filtered_out:,}")

        if store == "B":
            lines.append("")
            lines.append("CHECK 7 — Store B tags column")
            lines.append(f"  Rows with non-empty tags: {stats.tags_rows:,}")
            lines.append(f"  Rows where tags parsed to a list: {stats.tags_parsed_rows:,}")
            lines.append(f"  Rows with 12–14 digit tag token: {stats.rows_with_digit_tag:,}")
            lines.append(f"  Unique GTIN-like codes in tags: {len(stats.codes_in_tags):,}")
            lines.append("  Top 20 tag tokens:")
            lines.append(format_counter(stats.tag_token_counter, top_n=20))
            lines.append("  Top GTIN-like codes in tags:")
            lines.append(format_counter(stats.codes_in_tags, top_n=10))

    section("CHECK 4 — Cross-store code overlap (A ∩ B)")
    lines.append(f"  Unique codes in A: {len(stats_a.all_codes):,}")
    lines.append(f"  Unique codes in B: {len(stats_b.all_codes):,}")
    lines.append(f"  Codes in both stores: {len(overlap):,}")
    if overlap:
        sample = sorted(overlap)[:15]
        lines.append(f"  Sample shared codes: {', '.join(sample)}")
    lines.append("")
    lines.append("RECOMMENDATION (auto-generated, review manually)")
    if len(overlap) >= 100:
        lines.append(
            "  → Meaningful cross-store code overlap: consider a UPC/GTIN-first "
            "exact-match stage, then attribute matching."
        )
    elif len(overlap) > 0:
        lines.append(
            "  → Small cross-store overlap: use code matching when present; "
            "primary matcher should be attribute-based."
        )
    else:
        lines.append(
            "  → No shared codes across stores after filters: skip UPC-first stage; "
            "rely on brand + name + size + category/tags."
        )

    section("NOTES")
    lines.append("  • url_id (Walmart/Wegmans internal product id) is NOT comparable across stores.")
    lines.append("  • Use column names when loading CSVs, never positional index alignment.")
    lines.append("  • Description scan on A may be sampled for speed; see --description-sample.")
    lines.append("  • GS1 checksum validation is not applied; codes are heuristic matches only.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze product identifiers in grocery store A/B CSVs."
    )
    parser.add_argument(
        "--description-sample",
        type=int,
        default=10000,
        help="Scan description on store A only for first N rows (0 = full file). "
        "Default 10000 for speed; store B scans all descriptions (smaller file).",
    )
    args = parser.parse_args()
    desc_sample = args.description_sample if args.description_sample > 0 else None

    if not STORE_A.exists() or not STORE_B.exists():
        raise SystemExit(
            f"Missing CSV files. Expected:\n  {STORE_A}\n  {STORE_B}\n"
            "Run from repo root or place files next to this script's parent."
        )

    print("Analyzing Store A...")
    stats_a = analyze_store(STORE_A, "A", description_sample=desc_sample)
    print("Analyzing Store B...")
    stats_b = analyze_store(STORE_B, "B", description_sample=None)

    report = build_report(stats_a, stats_b, description_sample=desc_sample)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
