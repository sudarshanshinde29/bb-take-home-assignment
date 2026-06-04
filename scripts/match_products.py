#!/usr/bin/env python3
"""
Steps 2, 2b, 3, 4: Block, retrieve, fast-path match, inline LLM judge, write matches.csv.

Usage (from repo root):
  python3 scripts/build_features.py
  python3 scripts/match_products.py              # local + LLM judge (default)
  python3 scripts/match_products.py --skip-judge # local only, no API

See ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from tqdm import tqdm

from llm_judge import (
    JUDGE_CACHE,
    get_openai_client,
    judge_one,
    load_creds,
    load_judge_cache,
)
from matching_lib import (
    CONFIDENCE_LOW_REJECT,
    attribute_pool_conflict,
    can_fast_accept,
    needs_llm_judge,
    product_search_tokens,
    unit_families_compatible,
    unit_family_from_size,
)

REPO_ROOT = _SCRIPT_DIR.parent
CACHE_A = REPO_ROOT / "cache" / "store_a_features.jsonl"
CACHE_B = REPO_ROOT / "cache" / "store_b_features.jsonl"
MATCHES_CSV = REPO_ROOT / "matches.csv"
META_JSONL = REPO_ROOT / "cache" / "match_meta.jsonl"

TOP_K = 20
JUDGE_TOP = 5
MIN_MATCH_SCORE = 0.72  # minimum to write item_id_B (local or LLM); hybrid floor
TOKEN_POOL_CAP = 4000
GLOBAL_FALLBACK_LIMIT = 40


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_features(limit: int | None) -> None:
    if CACHE_A.exists() and CACHE_B.exists() and limit is None:
        return
    cmd = [sys.executable, str(_SCRIPT_DIR / "build_features.py")]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    print("Building feature caches...")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def build_b_indices(b_list: list[dict]) -> dict[str, Any]:
    brand_index: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, list[int]] = defaultdict(list)
    b_by_id: dict[str, dict] = {}

    for i, feat in enumerate(b_list):
        b_by_id[feat["item_id"]] = feat
        if feat.get("brand_normalized") and not feat.get("is_private_label"):
            brand_index[feat["brand_normalized"]].append(i)
        for tok in feat.get("search_tokens") or product_search_tokens(feat):
            token_index[tok].append(i)

    return {
        "b_list": b_list,
        "b_by_id": b_by_id,
        "brand_index": brand_index,
        "token_index": token_index,
    }


def candidate_indices(a_feat: dict, idx: dict[str, Any]) -> list[int]:
    """Token-index retrieval + brand block; no wide PL grocery buckets."""
    brand_index = idx["brand_index"]
    token_index = idx["token_index"]
    b_list = idx["b_list"]

    cands: set[int] = set()
    brand = a_feat.get("brand_normalized") or ""

    # National brand: same brand in B only
    if brand and not a_feat.get("is_private_label"):
        cands.update(brand_index.get(brand, []))

    # Lexical retrieval on product nouns (corn, cob, frame, ...)
    tokens = a_feat.get("search_tokens") or product_search_tokens(a_feat)
    for tok in tokens:
        cands.update(token_index.get(tok, []))

    if len(cands) > TOKEN_POOL_CAP:
        # Keep items that hit the most query tokens
        hit_counts: dict[int, int] = defaultdict(int)
        for tok in tokens:
            for i in token_index.get(tok, []):
                hit_counts[i] += 1
        ranked = sorted(hit_counts.items(), key=lambda x: (-x[1], x[0]))
        cands = {i for i, _ in ranked[:TOKEN_POOL_CAP]}

    if not cands:
        # Global fuzzy fallback (never b_list[0])
        query = a_feat.get("match_text") or ""
        choices = {str(i): b_list[i].get("match_text") or "" for i in range(len(b_list))}
        hits = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=GLOBAL_FALLBACK_LIMIT)
        cands = {int(key) for _, _, key in hits}

    return list(cands)


def _bm25_scores(query_tokens: list[str], pool: list[int], b_list: list[dict]) -> dict[int, float]:
    if not query_tokens or not pool:
        return {}
    docs = [
        (b_list[i].get("search_tokens") or product_search_tokens(b_list[i]))
        for i in pool
    ]
    bm25 = BM25Okapi(docs)
    raw = bm25.get_scores(query_tokens)
    if len(raw) == 0:
        return {}
    max_s = max(raw) if max(raw) > 0 else 1.0
    return {pool[i]: float(raw[i]) / max_s for i in range(len(pool))}


def score_candidates(
    a_feat: dict,
    cand_indices: list[int],
    b_list: list[dict],
) -> list[tuple[float, dict]]:
    query = a_feat.get("match_text") or ""
    query_tokens = a_feat.get("search_tokens") or product_search_tokens(a_feat)
    pool = list(cand_indices)

    if not pool:
        return []

    bm25_map = _bm25_scores(query_tokens, pool, b_list)

    scored: list[tuple[float, dict]] = []
    for i in pool:
        bf = b_list[i]
        # Hard gates: unit family, numeric size, size labels (NB vs 3), variants
        if not unit_families_compatible(a_feat.get("size"), bf.get("size")):
            continue
        if attribute_pool_conflict(a_feat, bf):
            continue

        fuzzy = fuzz.token_set_ratio(query, bf.get("match_text") or "") / 100.0
        bm25_s = bm25_map.get(i, 0.0)

        # Require shared product token when we have query tokens
        if query_tokens:
            b_tokens = set(bf.get("search_tokens") or [])
            if not (set(query_tokens) & b_tokens):
                continue

        combined = 0.45 * fuzzy + 0.55 * bm25_s
        scored.append((combined, bf))

    scored.sort(key=lambda x: (-x[0], x[1]["item_id"]))
    return scored[:TOP_K]


def pick_match(
    a_feat: dict,
    scored: list[tuple[float, dict]],
    *,
    skip_judge: bool,
    judge_cache: dict[str, dict],
    client: Any | None,
    creds: dict | None,
    stats: dict[str, int],
    min_score: float = MIN_MATCH_SCORE,
) -> tuple[str, str, str]:
    """Returns item_id_B (may be empty), match_type, source."""

    def bump(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1

    def accept_local(
        top_score: float,
        item_id_b: str,
        source: str,
        mtype: str = "local",
    ) -> tuple[str, str, str]:
        if item_id_b:
            bf = next((bf for _, bf in scored if bf["item_id"] == item_id_b), top_feat)
            if attribute_pool_conflict(a_feat, bf):
                bump("no_match")
                return "", "no_match", "attribute_conflict"
        if top_score >= min_score and item_id_b:
            return item_id_b, mtype, source
        bump("no_match")
        return "", "no_match", "low_confidence"

    if not scored:
        bump("no_match")
        return "", "no_match", "empty"

    top_score, top_feat = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None
    top_wrap = {"feat": top_feat, "score": top_score}
    fallback_id = top_feat["item_id"]

    if can_fast_accept(a_feat, top_wrap, top_score, second_score):
        bump("fast_path")
        return top_feat["item_id"], "exact", "fast_path"

    if top_score < CONFIDENCE_LOW_REJECT:
        bump("no_match")
        return "", "no_match", "below_low_band"

    # Earlier-style: local top-1 when score high enough and attrs OK (incl. --skip-judge)
    if skip_judge or not needs_llm_judge(a_feat, top_wrap, top_score, second_score):
        bump("local_top")
        return accept_local(top_score, fallback_id, "local" if skip_judge else "local_high")

    item_id_a = a_feat["item_id"]
    if item_id_a in judge_cache:
        rec = judge_cache[item_id_a]
        bump("judge_cache")
        chosen = rec.get("chosen_item_id_B") or ""
        if rec.get("match_found") is False or not chosen:
            bump("llm_reject")
            return "", "no_match", "judge_cache_reject"
        if chosen:
            bf = next((bf for _, bf in scored if bf["item_id"] == chosen), None)
            if bf and attribute_pool_conflict(a_feat, bf):
                bump("no_match")
                return "", "no_match", "judge_cache_attr"
        return chosen, rec.get("match_type", "none"), "judge_cache"

    candidates = [{"feat": bf, "score": sc} for sc, bf in scored[:JUDGE_TOP]]
    rec = judge_one(
        a_feat,
        candidates,
        use_cache=True,
        client=client,
        creds=creds,
    )
    judge_cache[item_id_a] = rec
    src = rec.get("source", "llm")
    if src == "llm":
        bump("judge_api")
    elif src == "local_content_filter":
        bump("judge_content_filter")

    chosen = rec.get("chosen_item_id_B") or ""
    if rec.get("match_found") is False or not chosen:
        bump("llm_reject")
        return "", "no_match", "llm_reject"

    bf = next((bf for _, bf in scored if bf["item_id"] == chosen), None)
    if bf and attribute_pool_conflict(a_feat, bf):
        bump("no_match")
        return "", "no_match", "llm_choice_attr"
    return chosen, rec.get("match_type", "exact"), src


def flush_meta_batch(path: Path, batch: list[dict]) -> None:
    if not batch:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in batch:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Match Store A items to Store B.")
    parser.add_argument("--limit", type=int, default=None, help="Limit A rows (testing)")
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Disable inline LLM judge (local top-1 only)",
    )
    parser.add_argument("--judge-max", type=int, default=None, help="Cap new LLM API calls (testing)")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip cache/match_meta.jsonl (faster; matches.csv still written)",
    )
    parser.add_argument(
        "--meta-flush-every",
        type=int,
        default=2000,
        help="Flush match_meta.jsonl every N rows (default 2000)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_MATCH_SCORE,
        help=f"Minimum combined score to accept a match (default {MIN_MATCH_SCORE})",
    )
    args = parser.parse_args()

    if args.rebuild_features or not CACHE_A.exists():
        ensure_features(args.limit)

    print("Loading features...")
    a_list = load_jsonl(CACHE_A)
    b_list = load_jsonl(CACHE_B)
    if args.limit:
        a_list = a_list[: args.limit]
    print(f"  A: {len(a_list):,} | B: {len(b_list):,}")

    idx = build_b_indices(b_list)
    judge_cache = load_judge_cache(JUDGE_CACHE)

    client = None
    creds = None
    if not args.skip_judge:
        try:
            creds = load_creds(REPO_ROOT / "openai_creds.yaml")
            client = get_openai_client(creds)
            cached = len(judge_cache)
            print(f"Inline LLM judge: {creds['model']} (cached decisions: {cached:,})")
            print("Tip: delete cache/judge_results.jsonl if old wrong LLM fallbacks are cached.")
        except FileNotFoundError:
            print("No openai_creds.yaml — LLM judge disabled; use local matching only")
            args.skip_judge = True
    else:
        print("LLM judge disabled (--skip-judge)")

    if not args.skip_meta and META_JSONL.exists():
        META_JSONL.unlink()

    stats: dict[str, int] = defaultdict(int)
    new_judge_calls = 0
    t0 = time.time()

    meta_batch: list[dict] = []
    rows_written = 0

    MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered + flush per row so matches.csv is visible while the run is in progress
    matches_file = MATCHES_CSV.open("w", newline="", encoding="utf-8", buffering=1)
    matches_writer = csv.writer(matches_file)
    matches_writer.writerow(["item_id_A", "item_id_B"])
    matches_file.flush()

    iterator: Any = a_list
    if not args.no_progress:
        iterator = tqdm(
            a_list,
            total=len(a_list),
            desc="Match A→B (+ LLM when needed)",
            unit="item",
            dynamic_ncols=True,
        )

    for a_feat in iterator:
        if args.judge_max is not None and new_judge_calls >= args.judge_max:
            args.skip_judge = True

        cands = candidate_indices(a_feat, idx)
        scored = score_candidates(a_feat, cands, idx["b_list"])
        top_score = scored[0][0] if scored else 0.0

        before_api = stats["judge_api"]
        item_id_b, match_type, source = pick_match(
            a_feat,
            scored,
            skip_judge=args.skip_judge,
            judge_cache=judge_cache,
            client=client,
            creds=creds,
            stats=stats,
            min_score=args.min_score,
        )
        if stats["judge_api"] > before_api:
            new_judge_calls += 1

        matches_writer.writerow([a_feat["item_id"], item_id_b])
        matches_file.flush()
        rows_written += 1

        if not args.skip_meta:
            meta_batch.append(
                {
                    "item_id_A": a_feat["item_id"],
                    "item_id_B": item_id_b,
                    "match_type": match_type,
                    "source": source,
                    "score": round(top_score, 4),
                    "unit_family_a": unit_family_from_size(a_feat.get("size")),
                }
            )
            if len(meta_batch) >= args.meta_flush_every:
                flush_meta_batch(META_JSONL, meta_batch)
                meta_batch.clear()

        if not args.no_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                fast=stats["fast_path"],
                local=stats.get("local_top", 0),
                nomatch=stats["no_match"],
                llm=stats["judge_api"],
                cache=stats.get("judge_cache", 0),
                refresh=False,
            )

    flush_meta_batch(META_JSONL, meta_batch)
    matches_file.close()

    elapsed = time.time() - t0
    matched = rows_written - stats["no_match"]
    print(f"\nWrote {MATCHES_CSV} ({rows_written:,} rows, {matched:,} with item_id_B) in {elapsed:.1f}s")
    if not args.skip_meta:
        print(f"Meta: {META_JSONL}")
    print("Stats:", dict(stats))
    if not args.skip_judge:
        print(f"New LLM API calls this run: {new_judge_calls:,}")


if __name__ == "__main__":
    main()
