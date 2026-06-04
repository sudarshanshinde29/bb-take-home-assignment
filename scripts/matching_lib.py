"""
Shared feature extraction and matching helpers for Store A / Store B product matching.
See ARCHITECTURE.md for pipeline context.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# Walmart private-label prefixes (Store A)
WALMART_PL_BRANDS = {
    "great value",
    "mainstays",
    "better homes & gardens",
    "better homes and gardens",
    "equate",
    "ol' roy",
    "ol roy",
    "parent's choice",
    "parents choice",
    "sam's choice",
    "sams choice",
    "marketside",
    "freshness guaranteed",
    "backyard grill",
    "hyper tough",
    "pen+gear",
    "pen gear",
    "expert grill",
    "auto drive",
    "hometrends",
    "your zone",
    "time and tru",
    "no boundaries",
    "george",
    "athletic works",
    "secret treasures",
    "swiss tech",
    "world table",
    "golden rewards",
}

# Flavor / variant keywords (lowercase)
VARIANT_KEYWORDS = {
    "vanilla",
    "chocolate",
    "strawberry",
    "blueberry",
    "honey",
    "original",
    "classic",
    "lemon",
    "lime",
    "orange",
    "grape",
    "cherry",
    "raspberry",
    "peach",
    "mango",
    "pineapple",
    "coconut",
    "almond",
    "cinnamon",
    "maple",
    "garlic",
    "onion",
    "ranch",
    "italian",
    "basil",
    "parmesan",
    "cheddar",
    "mozzarella",
    "unsalted",
    "salted",
    "low sodium",
    "no salt added",
    "diet",
    "zero",
    "caffeine free",
    "decaf",
    "light",
    "dark",
    "mild",
    "medium",
    "hot",
    "spicy",
    "bold",
    "regular",
    "plain",
    "smoke",
    "hickory",
    "mesquite",
    "lavender",
    "rose",
    "cocoa",
    "mocha",
    "espresso",
    "caramel",
    "birthday cake",
    "cotton candy",
}

FORM_KEYWORDS = {
    "frozen": "frozen",
    "fresh": "fresh",
    "refrigerated": "refrigerated",
    "shelf stable": "shelf_stable",
    "canned": "canned",
}

SIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(fl\.?\s*oz|fluid\s*ounce|fluid\s*ounces|oz|ounce|ounces|lb|lbs|pound|pounds|"
    r"gal|gallon|gallons|ml|l|liter|liters|kg|g|gram|grams|count|ct|pack|pk|each|bags|pieces)\b",
    re.I,
)

# Diaper/apparel size labels (word-bounded)
SIZE_LABEL_RE = re.compile(
    r"\b(?:size\s*)?(nb|newborn|nwt|size\s*[1-9][a-z]?|#[1-9])\b",
    re.I,
)

# Hybrid routing thresholds (combined fuzzy+BM25 score)
CONFIDENCE_LOW_REJECT = 0.50  # below → empty (obvious garbage only)
FAST_PATH_MIN_SCORE = 0.88  # national-brand auto accept
NATIONAL_LLM_MAX_SCORE = 0.88  # skip LLM at/above if not a tie
PL_LLM_MAX_SCORE = 0.84  # private-label ↔ PL: skip LLM at/above if not a tie
SCORE_MARGIN = 0.05

DIMENSION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)(?:\s*[xX]\s*(\d+(?:\.\d+)?))?\s*(?:in|inch|inches|\")?",
    re.I,
)

# Unit dimension families (non-overlapping gates for matching)
UNIT_FAMILY_COUNT = "COUNT"
UNIT_FAMILY_VOLUME = "VOLUME"
UNIT_FAMILY_WEIGHT = "WEIGHT"
UNIT_FAMILY_DIMENSION = "DIMENSION"
UNIT_FAMILY_UNKNOWN = "UNKNOWN"

COUNT_UNITS = {"ct", "count", "pack", "pk", "each", "bags", "pieces"}
VOLUME_UNITS = {"fl oz", "ml", "l", "gal", "pt", "qt", "gallon", "liter", "liters"}
WEIGHT_UNITS = {"oz", "lb", "g", "kg", "lbs", "pound", "pounds", "gram", "grams", "ounce", "ounces"}

STOPWORDS = {
    "the", "and", "for", "with", "from", "new", "free", "size", "pack", "per", "each",
    "oz", "ounce", "count", "fl", "value", "great", "wegmans", "walmart", "brand",
    "shop", "all", "more", "your", "our", "can", "are", "has", "use", "made", "style",
}

# Precompiled variant patterns (word boundaries — avoids "hot" in "photo")
_VARIANT_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
for _kw in VARIANT_KEYWORDS:
    if " " in _kw:
        _VARIANT_PATTERNS.append((_kw, re.compile(re.escape(_kw), re.I)))
    else:
        _VARIANT_PATTERNS.append((_kw, re.compile(rf"\b{re.escape(_kw)}\b", re.I)))

COARSE_MAP = {
    "food": "grocery",
    "grocery": "grocery",
    "frozen": "grocery",
    "pets": "pets",
    "pet": "pets",
    "home": "home",
    "household": "home",
    "health": "health",
    "beauty": "health",
    "personal care": "health",
    "more departments": "other",
    "electronics": "other",
    "sports": "other",
    "auto": "other",
    "baby": "baby",
    "apparel": "other",
}


def parse_json_field(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


def parse_tags(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw or raw == "{}":
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(x).lower() for x in val]
        if isinstance(val, dict):
            return [str(k).lower() for k in val.keys()]
    except (SyntaxError, ValueError):
        pass
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x).lower() for x in val]
    except json.JSONDecodeError:
        pass
    return [raw.lower()]


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_brand(brand: str) -> str:
    b = normalize_text(brand)
    b = re.sub(r"\s+", " ", b)
    return b


def unit_family_from_size(size: dict[str, Any] | None) -> str:
    if not size:
        return UNIT_FAMILY_UNKNOWN
    if size.get("unit_family"):
        return size["unit_family"]
    unit = (size.get("unit") or "").lower().strip()
    if unit in COUNT_UNITS or unit in ("ct", "count"):
        return UNIT_FAMILY_COUNT
    if "fl oz" in unit or unit in VOLUME_UNITS:
        return UNIT_FAMILY_VOLUME
    if unit in WEIGHT_UNITS:
        return UNIT_FAMILY_WEIGHT
    return UNIT_FAMILY_UNKNOWN


def unit_families_compatible(
    size_a: dict[str, Any] | None,
    size_b: dict[str, Any] | None,
) -> bool:
    """Hard gate: different known families cannot match."""
    fa = unit_family_from_size(size_a)
    fb = unit_family_from_size(size_b)
    if fa == UNIT_FAMILY_UNKNOWN or fb == UNIT_FAMILY_UNKNOWN:
        return True
    return fa == fb


def parse_size_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    dim = DIMENSION_RE.search(text)
    if dim:
        parts = [g for g in dim.groups() if g]
        return {
            "value": float(parts[0]),
            "unit": "dimension",
            "raw": dim.group(0),
            "unit_family": UNIT_FAMILY_DIMENSION,
            "dimension_parts": parts,
        }
    best = None
    for m in SIZE_RE.finditer(text):
        val = float(m.group(1))
        unit = m.group(2).lower().replace(".", "").replace("  ", " ")
        unit = unit.replace("fluid ounce", "fl oz").replace("fluid ounces", "fl oz")
        unit = unit.replace("ounces", "oz").replace("ounce", "oz")
        unit = unit.replace("pounds", "lb").replace("pound", "lb").replace("lbs", "lb")
        unit = unit.replace("gallons", "gal").replace("gallon", "gal")
        unit = unit.replace("grams", "g").replace("gram", "g")
        unit = unit.replace("liters", "l").replace("liter", "l")
        unit = unit.replace("count", "ct").replace("pack", "ct").replace("pk", "ct")
        unit = unit.replace("bags", "ct").replace("pieces", "ct")
        uf = unit_family_from_size({"unit": unit})
        best = {"value": val, "unit": unit, "raw": m.group(0), "unit_family": uf}
    return best


def size_to_bucket(size: dict[str, Any] | None) -> str:
    if not size:
        return "unknown"
    if size.get("unit_family") == UNIT_FAMILY_DIMENSION:
        parts = size.get("dimension_parts") or [size["value"]]
        return "dimension:" + "x".join(str(round(float(p), 1)) for p in parts)
    val, unit = size["value"], size["unit"]
    if unit in ("fl oz", "oz"):
        base_val, base_unit = val, "oz"
        if unit == "fl oz":
            base_val, base_unit = val, "fl_oz"
    elif unit == "lb":
        base_val, base_unit = val, "lb"
    elif unit in ("g", "kg"):
        base_val = val * 1000 if unit == "kg" else val
        base_unit = "g"
    elif unit in ("ml", "l"):
        base_val = val * 1000 if unit == "l" else val
        base_unit = "ml"
    elif unit in ("ct", "each"):
        base_val, base_unit = val, unit
    elif unit == "gal":
        base_val, base_unit = val * 128, "fl_oz"
    else:
        base_val, base_unit = val, unit
    rounded = round(base_val, 1)
    return f"{base_unit}:{rounded}"


def extract_size_labels(text: str) -> set[str]:
    t = normalize_text(text)
    labels: set[str] = set()
    for m in SIZE_LABEL_RE.finditer(t):
        lab = m.group(1).lower().replace(" ", "")
        if lab.startswith("size"):
            lab = lab.replace("size", "size_")
        labels.add(lab)
    return labels


def size_labels_conflict(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    if a & b:
        return False
    return True


def sizes_compatible(a: dict[str, Any] | None, b: dict[str, Any] | None, rel_tol: float = 0.2) -> bool:
    """Both sides must have parsed size (fast-path / strict checks)."""
    if not unit_families_compatible(a, b):
        return False
    if not a or not b:
        return False
    ba, bb = size_to_bucket(a), size_to_bucket(b)
    if ba == "unknown" or bb == "unknown":
        return False
    if ba == bb:
        return True
    ua, ub = ba.split(":", 1)[0], bb.split(":", 1)[0]
    if ua != ub:
        return False
    try:
        va, vb = float(ba.split(":", 1)[1]), float(bb.split(":", 1)[1])
    except ValueError:
        return True
    if va == 0 or vb == 0:
        return va == vb
    # COUNT family: require exact integer match (no 24ct vs 20ct)
    if unit_family_from_size(a) == UNIT_FAMILY_COUNT:
        return int(round(va)) == int(round(vb))
    return abs(va - vb) / max(va, vb) <= rel_tol


def extract_variant_tokens(text: str) -> set[str]:
    t = normalize_text(text)
    found: set[str] = set()
    for kw, pat in _VARIANT_PATTERNS:
        if pat.search(t):
            found.add(kw)
    return found


def product_search_tokens(feat: dict[str, Any]) -> list[str]:
    """Core noun-like tokens for BM25 / inverted index retrieval."""
    text = feat.get("product_text") or feat.get("match_text") or ""
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    out: list[str] = []
    brand = feat.get("brand_normalized") or ""
    for tok in tokens:
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        if brand and tok in brand.split():
            continue
        if tok.isdigit():
            continue
        out.append(tok)
    # preserve order, unique
    seen: set[str] = set()
    unique: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:12]


def variant_tokens_conflict(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    if a & b:
        return False
    return True


def extract_form(name: str, item_info: dict | None, tags: list[str]) -> str | None:
    t = normalize_text(name)
    for kw, form in FORM_KEYWORDS.items():
        if kw in t:
            return form
    if item_info:
        st = (item_info.get("storage_type") or "")
        if st:
            st_l = str(st).lower()
            if "frozen" in st_l:
                return "frozen"
            if "refrigerat" in st_l:
                return "refrigerated"
    for tag in tags:
        if "frozen" in tag:
            return "frozen"
    return None


def coarse_department(category_0: str, category_1: str) -> str:
    for part in (category_0, category_1):
        p = normalize_text(part)
        for key, dept in COARSE_MAP.items():
            if key in p:
                return dept
    return "other"


def detect_private_label_a(name: str, brand: str) -> bool:
    b = normalize_brand(brand)
    if b in WALMART_PL_BRANDS:
        return True
    t = normalize_text(name)
    for pl in WALMART_PL_BRANDS:
        if t.startswith(pl + " ") or t == pl:
            return True
    return False


def detect_private_label_b(brand: str, tags: list[str]) -> bool:
    b = normalize_brand(brand)
    if b == "wegmans":
        return True
    tag_str = " ".join(tags)
    if "wegmans_brand" in tag_str or "wegmans brand" in tag_str or "private label" in tag_str:
        return True
    return False


def extract_brand_from_name(name: str, brand_vocab: list[str]) -> str:
    """Longest matching brand from vocabulary at start of name or as substring."""
    t = normalize_text(name)
    best = ""
    for brand in sorted(brand_vocab, key=len, reverse=True):
        if not brand or len(brand) < 2:
            continue
        if t.startswith(brand + " ") or t == brand:
            if len(brand) > len(best):
                best = brand
    return best


def build_brand_vocab_from_b(rows: list[dict]) -> list[str]:
    brands = set()
    for r in rows:
        b = normalize_brand(r.get("brand_raw") or "")
        if b and len(b) > 1:
            brands.add(b)
    brands.update(WALMART_PL_BRANDS)
    return sorted(brands, key=len, reverse=True)


def extract_features(
    row: dict[str, str],
    store: str,
    brand_vocab: list[str] | None = None,
) -> dict[str, Any]:
    name = row.get("name") or ""
    brand_raw = row.get("brand_raw") or ""
    item_info = parse_json_field(row.get("item_info") or "")
    sizing = parse_json_field(row.get("sizing_comp") or "")
    tags = parse_tags(row.get("tags") or "") if store == "B" else []

    cat0 = (item_info or {}).get("category_0") or ""
    cat1 = (item_info or {}).get("category_1") or ""
    cat2 = (item_info or {}).get("category_2") or ""
    category_coarse = f"{coarse_department(str(cat0), str(cat1))}|{normalize_text(str(cat1))[:40]}"

    size = None
    if sizing and sizing.get("size_user_friendly"):
        size = parse_size_from_text(str(sizing["size_user_friendly"]))
    if not size:
        size = parse_size_from_text(name)

    brand = normalize_brand(brand_raw)
    if not brand and store == "A" and brand_vocab:
        brand = extract_brand_from_name(name, brand_vocab)
        if not brand:
            for pl in WALMART_PL_BRANDS:
                if normalize_text(name).startswith(pl):
                    brand = pl
                    break

    if store == "A":
        is_pl = detect_private_label_a(name, brand)
    else:
        is_pl = detect_private_label_b(brand, tags)

    is_organic = "organic" in normalize_text(name) or any("organic" in t for t in tags)

    variant_tokens = sorted(extract_variant_tokens(name))
    size_labels = sorted(extract_size_labels(name))
    form = extract_form(name, item_info, tags)

    product_text = normalize_text(name)
    if brand and product_text.startswith(brand):
        product_text = product_text[len(brand) :].strip()

    match_text = f"{brand} {product_text}".strip() if brand else product_text

    unit_family = unit_family_from_size(size)

    return {
        "item_id": str(row.get("item_id") or ""),
        "name": name,
        "brand_normalized": brand,
        "size": size,
        "unit_family": unit_family,
        "size_bucket": size_to_bucket(size),
        "search_tokens": product_search_tokens(
            {"product_text": product_text, "brand_normalized": brand, "match_text": match_text}
        ),
        "is_private_label": is_pl,
        "is_organic": is_organic,
        "category_coarse": category_coarse,
        "department": coarse_department(str(cat0), str(cat1)),
        "category_0": str(cat0),
        "category_1": str(cat1),
        "form": form,
        "variant_tokens": variant_tokens,
        "size_labels": size_labels,
        "product_text": product_text,
        "match_text": match_text or normalize_text(name),
        "tags": tags[:10] if store == "B" else [],
    }


def feature_summary_line(feat: dict[str, Any]) -> str:
    size_s = ""
    if feat.get("size"):
        size_s = feat["size"].get("raw") or f"{feat['size']['value']} {feat['size']['unit']}"
    variants = ", ".join(feat.get("variant_tokens") or []) or "none"
    return (
        f"name={feat.get('name', '')[:120]} | brand={feat.get('brand_normalized') or 'unknown'} | "
        f"size={size_s or 'unknown'} | PL={feat.get('is_private_label')} | "
        f"form={feat.get('form') or 'unknown'} | variants={variants} | "
        f"dept={feat.get('department')} | cat={feat.get('category_coarse', '')[:50]}"
    )


def feature_size_labels(feat: dict[str, Any]) -> set[str]:
    if feat.get("size_labels"):
        return set(feat["size_labels"])
    return extract_size_labels(feat.get("name") or "")


def sizes_definite_conflict(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    rel_tol: float = 0.08,
) -> bool:
    """True only when both sides have size and they disagree (e.g. 72 ct vs 62 ct)."""
    if not a or not b:
        return False
    return not sizes_compatible(a, b, rel_tol=rel_tol)


def attribute_pool_conflict(target: dict[str, Any], cand: dict[str, Any]) -> bool:
    """
    Definite mismatches for retrieval pool / local write.
    Unknown size on one side does not block (sparse A metadata).
    """
    if variant_tokens_conflict(
        set(target.get("variant_tokens") or []),
        set(cand.get("variant_tokens") or []),
    ):
        return True
    if size_labels_conflict(
        feature_size_labels(target),
        feature_size_labels(cand),
    ):
        return True
    if sizes_definite_conflict(target.get("size"), cand.get("size")):
        return True
    return False


# Alias used by scoring and accept_local
attribute_conflicts = attribute_pool_conflict


def attribute_conflicts_strict(target: dict[str, Any], cand: dict[str, Any]) -> bool:
    """Fast-path: definite conflict or missing size on either side."""
    if attribute_pool_conflict(target, cand):
        return True
    if not target.get("size") or not cand.get("size"):
        return True
    return False


def can_fast_accept(
    target: dict[str, Any],
    top: dict[str, Any],
    top_score: float,
    second_score: float | None,
    *,
    min_score: float = FAST_PATH_MIN_SCORE,
    min_margin: float = SCORE_MARGIN,
) -> bool:
    cand_feat = top.get("feat") or top
    if target.get("is_private_label") or cand_feat.get("is_private_label"):
        return False
    if not target.get("brand_normalized") or target["brand_normalized"] != cand_feat.get("brand_normalized"):
        return False
    if attribute_conflicts_strict(target, cand_feat):
        return False
    if top_score < min_score:
        return False
    if second_score is not None and (top_score - second_score) < min_margin:
        return False
    return True


def needs_llm_judge(
    target: dict[str, Any],
    top: dict[str, Any] | None,
    top_score: float,
    second_score: float | None,
    *,
    min_margin: float = SCORE_MARGIN,
    national_llm_max: float = NATIONAL_LLM_MAX_SCORE,
    pl_llm_max: float = PL_LLM_MAX_SCORE,
) -> bool:
    """
    Earlier-style LLM gating: call judge only for borderline scores or ties.
    Skips LLM when fast-path would apply or score is clearly high with margin.
    """
    if top is None:
        return False
    if can_fast_accept(target, top, top_score, second_score, min_margin=min_margin):
        return False
    cand_feat = top.get("feat") or top
    tight_race = second_score is not None and (top_score - second_score) < min_margin
    if tight_race:
        return True
    if target.get("is_private_label") and cand_feat.get("is_private_label"):
        return top_score < pl_llm_max
    return top_score < national_llm_max
