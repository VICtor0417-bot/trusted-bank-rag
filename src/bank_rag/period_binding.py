"""Deterministic spelling normalization for evidence-period comparisons only."""
import re
from typing import Any

_QUARTERS = {"一": "1", "二": "2", "三": "3", "四": "4"}
_QUARTER = re.compile(r"(?<![0-9一二三四五六七八九十百])第?([一二三四1-4])季度")


def period_comparison_key(text: str) -> str:
    """Collapse layout spaces and equivalent quarter spellings; retain year."""
    compact = re.sub(r"\s+", "", text)
    return _QUARTER.sub(lambda m: _QUARTERS.get(m[1], m[1]) + "季度", compact)


def document_covers_period(period: str, document: Any) -> bool:
    """Match a full period, or a year-bearing title with its quarter in the row.

    Do not infer an absent year/quarter or equate months with quarter snapshots.
    This is a comparison key, not a rewrite of evidence or retrieval queries.
    """
    title = period_comparison_key(str(getattr(document, "meta", {}).get("source_title", "")))
    content = period_comparison_key(str(getattr(document, "content", "")))
    key = period_comparison_key(period)
    if not key:
        return False
    if key in title or key in content:
        return True
    match = re.fullmatch(r"(20\d{2})年(.+)", key)
    return bool(match and match[1] in title and match[2] in content)
