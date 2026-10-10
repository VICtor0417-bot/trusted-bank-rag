"""Deterministic canonical values used before evidence planning."""

from __future__ import annotations

import re
import json
from pathlib import Path


_PROVINCE_CANONICALIZATION: tuple[dict[str, str], ...] = (
    {"raw_value": "北京", "canonical_value": "北京市"},
    {"raw_value": "天津", "canonical_value": "天津市"},
    {"raw_value": "上海", "canonical_value": "上海市"},
    {"raw_value": "重庆", "canonical_value": "重庆市"},
    {"raw_value": "河北", "canonical_value": "河北省"},
    {"raw_value": "山西", "canonical_value": "山西省"},
    {"raw_value": "辽宁", "canonical_value": "辽宁省"},
    {"raw_value": "吉林", "canonical_value": "吉林省"},
    {"raw_value": "黑龙江", "canonical_value": "黑龙江省"},
    {"raw_value": "江苏", "canonical_value": "江苏省"},
    {"raw_value": "浙江", "canonical_value": "浙江省"},
    {"raw_value": "安徽", "canonical_value": "安徽省"},
    {"raw_value": "福建", "canonical_value": "福建省"},
    {"raw_value": "江西", "canonical_value": "江西省"},
    {"raw_value": "山东", "canonical_value": "山东省"},
    {"raw_value": "河南", "canonical_value": "河南省"},
    {"raw_value": "湖北", "canonical_value": "湖北省"},
    {"raw_value": "湖南", "canonical_value": "湖南省"},
    {"raw_value": "广东", "canonical_value": "广东省"},
    {"raw_value": "海南", "canonical_value": "海南省"},
    {"raw_value": "四川", "canonical_value": "四川省"},
    {"raw_value": "贵州", "canonical_value": "贵州省"},
    {"raw_value": "云南", "canonical_value": "云南省"},
    {"raw_value": "陕西", "canonical_value": "陕西省"},
    {"raw_value": "甘肃", "canonical_value": "甘肃省"},
    {"raw_value": "青海", "canonical_value": "青海省"},
    {"raw_value": "台湾", "canonical_value": "台湾省"},
    {"raw_value": "内蒙古", "canonical_value": "内蒙古自治区"},
    {"raw_value": "广西", "canonical_value": "广西壮族自治区"},
    {"raw_value": "西藏", "canonical_value": "西藏自治区"},
    {"raw_value": "宁夏", "canonical_value": "宁夏回族自治区"},
    {"raw_value": "新疆", "canonical_value": "新疆维吾尔自治区"},
    {"raw_value": "香港", "canonical_value": "香港特别行政区"},
    {"raw_value": "澳门", "canonical_value": "澳门特别行政区"},
)


def _load_prefecture_canonicalization() -> tuple[dict[str, str], ...]:
    """Load the checked-in nationwide prefecture-level replacement library."""
    path = Path(__file__).with_name("area_canonicalization.json")
    return tuple(json.loads(path.read_text(encoding="utf-8")))


AREA_CANONICALIZATION: tuple[dict[str, str], ...] = (
    _PROVINCE_CANONICALIZATION + _load_prefecture_canonicalization()
)

_AREA_LOOKUP = {
    item["raw_value"]: item["canonical_value"]
    for item in AREA_CANONICALIZATION
}
_CANONICAL_VALUES = set(_AREA_LOOKUP.values())


def canonicalize_area(value: str) -> str:
    """Return one stable administrative-area value without using an LLM."""
    compact = re.sub(r"\s+", "", str(value or "").strip())
    if compact in _CANONICAL_VALUES:
        return compact
    return _AREA_LOOKUP.get(compact, compact)


def is_known_area(value: str) -> bool:
    """Return whether a parsed value is a real supported area or nationwide."""
    canonical = canonicalize_area(value)
    return canonical == "全国" or canonical in _CANONICAL_VALUES


def canonicalize_area_mentions(text: str) -> tuple[str, list[dict[str, str]]]:
    """Replace known raw area mentions and retain an auditable mapping."""
    matches: list[tuple[int, int, str, str]] = []
    for raw, canonical in sorted(_AREA_LOOKUP.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = rf"{re.escape(raw)}(?!市|省|自治区|特别行政区)"
        matches.extend((match.start(), match.end(), raw, canonical) for match in re.finditer(pattern, text))
    matches.sort(key=lambda item: item[0])
    normalized = text
    for start, end, _raw, canonical in reversed(matches):
        normalized = normalized[:start] + canonical + normalized[end:]
    changes: list[dict[str, str]] = []
    for _start, _end, raw, canonical in matches:
        item = {"raw_value": raw, "canonical_value": canonical}
        if item not in changes:
            changes.append(item)
    return normalized, changes


def area_match_key(value: str) -> str:
    """Canonical comparison key for source labels such as ``北 京``."""
    canonical = canonicalize_area(value)
    return re.sub(r"(?:特别行政区|自治区|省|市)$", "", canonical)


def area_present_in_text(value: str, text: str) -> bool:
    """Match known spelling variants without equating different scopes.

    Explicit suffixes take precedence: 吉林市 cannot satisfy 吉林省. Bare
    labels use the existing canonicalization library (北京 -> 北京市).
    """
    canonical = canonicalize_area(value)
    if not canonical:
        return True
    compact = re.sub(r"\s+", "", str(text or ""))
    aliases = {canonical, *(raw for raw, target in _AREA_LOOKUP.items() if target == canonical)}
    return any(
        re.search(re.escape(alias) + r"(?!省|市|自治区|特别行政区)", compact)
        for alias in aliases
    )
