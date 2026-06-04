# Scripts

**Start here for the submission overview:** **[README.md](../README.md)** (algorithm, analysis, current run status).

See **[ARCHITECTURE.md](../ARCHITECTURE.md)** for the full matching pipeline (feature build → retrieval → hybrid routing → LLM judge → `matches.csv`).

## Matching pipeline (implementation)

| Script | Step |
|--------|------|
| `build_features.py` | Step 1 — `cache/store_*_features.jsonl` |
| `match_products.py` | Steps 2–4 — `matches.csv` (all A rows; empty B = non-match) |
| `export_matched_pairs.py` | Filter to `matches_only.csv` (rows with `item_id_B` set) |
| `llm_judge.py` | Step 3 — used by `match_products.py` |
| `matching_lib.py` | Shared parsers / rules |

**From repo root:**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

python3 run_matching.py                       # full run + inline LLM judge + progress bar
python3 run_matching.py --skip-judge          # local only (faster, no API)
python3 run_matching.py --limit 500           # smoke test with LLM on ambiguous rows
python3 run_matching.py --judge-max 100       # cap new LLM API calls
```

Progress: tqdm bar shows `fast` / `local` / `llm` / `cache` counts. Judge runs inside `match_products.py` (not a separate step).

## `analyze_product_identifiers.py`

**Purpose:** Read-only exploration of whether the Walmart (A) and Wegmans (B) CSVs contain structured product identifiers (UPC/GTIN/barcode) or other ID fields useful for exact matching.

**Helps with:**

| Check | Question answered |
|-------|-------------------|
| 1 | What keys exist inside `item_info` JSON (including nested)? |
| 2 | Are there UPC-like values under identifier keys or in JSON text? |
| 3 | Do `name`, `url`, `sizing_comp`, or `description` contain 12–14 digit codes? |
| 4 | How many codes appear in **both** stores (UPC-first matching viable)? |
| 5 | What is the Walmart/Wegmans URL product id, and does it equal `datapoint_id` / `raw_data_id`? |
| 6 | How many regex hits are false positives (timestamps, dates)? |
| 7 | Does Wegmans `tags` hide barcodes or supplier IDs? |

**Output:** `analysis_output/identifier_analysis_report.txt` (and printed summary)

**Run from repo root:**

```bash
python3 scripts/analyze_product_identifiers.py
```

**Options:**

```bash
# Scan all of store A descriptions (slower)
python3 scripts/analyze_product_identifiers.py --description-sample 0

# Smaller description sample on A (faster)
python3 scripts/analyze_product_identifiers.py --description-sample 5000
```

**Requirements:** Python 3.9+ standard library only (`csv`, `json`, `re`, `ast`).
