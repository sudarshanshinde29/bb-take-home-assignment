#!/usr/bin/env python3
"""
Step 3: Azure OpenAI judge module (called inline from match_products.py).

Reads openai_creds.yaml, caches results in cache/judge_results.jsonl (resumable).
You do not need to run this file separately — use match_products.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from matching_lib import feature_summary_line

REPO_ROOT = Path(__file__).resolve().parent.parent


class ContentFilterError(Exception):
    """Azure OpenAI blocked the prompt (Responsible AI / content filter)."""


def is_content_filter_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "content_filter" in text
        or "content management policy" in text
        or "responsibleaipolicyviolation" in text
    )


def judge_product_line(feat: dict) -> str:
    """Structured product line (avoids sending very long/raw titles to the API)."""
    variants = ", ".join(feat.get("variant_tokens") or []) or "none"
    size_s = ""
    if feat.get("size"):
        size_s = feat["size"].get("raw") or f"{feat['size']['value']} {feat['size']['unit']}"
    product = (feat.get("product_text") or feat.get("name") or "")[:100]
    return (
        f"brand={feat.get('brand_normalized') or 'unknown'} | product={product} | "
        f"size={size_s or 'unknown'} | PL={feat.get('is_private_label')} | "
        f"variants={variants} | dept={feat.get('department')}"
    )
CREDS_PATH = REPO_ROOT / "openai_creds.yaml"
JUDGE_CACHE = REPO_ROOT / "cache" / "judge_results.jsonl"

SYSTEM_PROMPT = """You are a precise grocery product matching engine for competitive price indexing.

Given one target product from Store A (Walmart) and up to 5 candidates from Store B (Wegmans),
choose the single best functional substitute for pricing comparison, or none.

Rules:
- EXACT: same national brand, same product type, same variant/flavor, compatible size and form.
  Both must NOT be private label.
- NON-EXACT: both private label, essentially the same product for a shopper (same type + size),
  brands may differ (e.g. Great Value vs Wegmans).
- If same brand but clearly different variant (e.g. honey vs vanilla yogurt), do NOT call exact.
- If no candidate is a reasonable substitute, set match_found to false.

Respond with JSON only:
{
  "match_found": boolean,
  "chosen_item_id_B": string or null,
  "match_type": "exact" | "non-exact" | "none",
  "reasoning": string
}
"""


def load_creds(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    oai = data.get("openai") or data
    return {
        "api_key": oai["api_key"],
        "base_url": oai.get("endpoint", "").rstrip("/"),
        "model": oai.get("deployment_name") or oai.get("model", "gpt-4o-mini"),
    }


def load_judge_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[str(rec["item_id_A"])] = rec
    return cache


def append_judge_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_user_prompt(
    target: dict,
    candidates: list[dict],
    *,
    sanitized: bool = False,
) -> str:
    line_fn = judge_product_line if sanitized else feature_summary_line
    lines = [
        "TARGET (Store A):",
        line_fn(target),
        f"item_id_A: {target['item_id']}",
        "",
        "CANDIDATES (Store B):",
    ]
    for i, c in enumerate(candidates, 1):
        feat = c.get("feat") or c
        lines.append(f"[{i}] item_id_B={feat['item_id']}")
        lines.append(line_fn(feat))
        if feat.get("tags"):
            tags = [t for t in feat["tags"][:5] if "internal" not in t]
            if tags:
                lines.append(f"    tags={tags}")
    return "\n".join(lines)


def call_judge(
    client: Any,
    model: str,
    target: dict,
    candidates: list[dict],
    *,
    sanitized: bool = False,
    max_retries: int = 4,
) -> dict:
    user_content = build_user_prompt(target, candidates, sanitized=sanitized)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as e:
            if is_content_filter_error(e):
                raise ContentFilterError(str(e)) from e
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge API failed after retries: {last_err}")


def resolve_judge_result(
    result: dict,
    candidates: list[dict],
    fallback_id: str,
) -> tuple[str, str]:
    """Return (item_id_B, match_type). Empty id when judge rejects the shortlist."""
    if result.get("match_found") and result.get("chosen_item_id_B"):
        chosen = str(result["chosen_item_id_B"])
        valid = {str((c.get("feat") or c)["item_id"]) for c in candidates}
        if chosen in valid:
            mt = result.get("match_type") or "exact"
            if mt not in ("exact", "non-exact", "none"):
                mt = "exact"
            return chosen, mt
    return "", "none"


def get_openai_client(creds: dict[str, str]) -> Any:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install openai: pip install openai") from e
    return OpenAI(api_key=creds["api_key"], base_url=creds["base_url"])


def judge_one(
    target: dict,
    candidates: list[dict],
    *,
    creds_path: Path = CREDS_PATH,
    cache_path: Path = JUDGE_CACHE,
    use_cache: bool = True,
    client: Any | None = None,
    creds: dict | None = None,
) -> dict:
    item_id_a = str(target["item_id"])
    cache = load_judge_cache(cache_path) if use_cache else {}
    if item_id_a in cache:
        return cache[item_id_a]

    fallback = str((candidates[0].get("feat") or candidates[0])["item_id"]) if candidates else ""

    if not candidates:
        rec = {
            "item_id_A": item_id_a,
            "chosen_item_id_B": fallback,
            "match_type": "none",
            "match_found": False,
            "reasoning": "no candidates",
            "source": "local",
        }
        if use_cache:
            append_judge_cache(cache_path, rec)
        return rec

    if client is None:
        creds = creds or load_creds(creds_path)
        client = get_openai_client(creds)
        model = creds["model"]
    else:
        model = creds["model"] if creds else load_creds(creds_path)["model"]

    shortlist = candidates[:5]
    api_result: dict | None = None
    try:
        api_result = call_judge(client, model, target, shortlist, sanitized=False)
    except ContentFilterError:
        try:
            api_result = call_judge(
                client, model, target, shortlist, sanitized=True, max_retries=1
            )
        except (ContentFilterError, RuntimeError):
            rec = {
                "item_id_A": item_id_a,
                "chosen_item_id_B": "",
                "match_type": "none",
                "match_found": False,
                "reasoning": "Azure content filter blocked prompt; no LLM match assigned.",
                "source": "local_content_filter",
            }
            if use_cache:
                append_judge_cache(cache_path, rec)
            return rec

    chosen, mt = resolve_judge_result(api_result or {}, candidates, fallback)
    rec = {
        "item_id_A": item_id_a,
        "chosen_item_id_B": chosen,
        "match_type": mt,
        "match_found": (api_result or {}).get("match_found", False),
        "reasoning": (api_result or {}).get("reasoning", ""),
        "source": "llm",
    }
    if use_cache:
        append_judge_cache(cache_path, rec)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM judge on a single example.")
    parser.add_argument("--item-id-a", required=True)
    args = parser.parse_args()
    # Minimal CLI test hook — full batch driven from match_products.py
    print("Use match_products.py to run batch judging. item_id test:", args.item_id_a)


if __name__ == "__main__":
    main()
