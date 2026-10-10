"""Strict source/structure matching for OFFLINE scoring only. No RAG imports."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

SCORING_LOCATOR_VERSION = "strict-locator-v2"


def source_id(value: Any) -> str:
    match = re.search(r"(?:^|[\\/])(\d{1,4})_", str(value or ""))
    return match.group(1).zfill(3) if match else ""


def _equal(left: Any, right: Any) -> bool:
    if right is None:
        return False
    return re.sub(r"\s+", "", str(left)) == re.sub(r"\s+", "", str(right))


def locator_hit(locator: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> bool:
    """Missing structural metadata is NOT proof of an exact Gold match."""
    for record in evidence:
        item = {**record, **(record.get("meta") or {})}
        expected_id = str(locator.get("source_id") or "").zfill(3)
        if not locator.get("source_id") or source_id(item.get("source_title") or item.get("source_path")) != expected_id:
            continue
        if any(locator.get(field) is not None and not _equal(locator[field], item.get(field))
               for field in ("sheet", "row", "page", "section", "table_id")):
            continue
        table = locator.get("table", locator.get("table_index"))
        if table is not None:
            actual = item.get("table_index", item.get("table"))
            if actual is None:
                match = re.fullmatch(r"Word表格(\d+)", str(item.get("sheet") or ""))
                actual = match.group(1) if match else None
            if not _equal(table, actual):
                continue
        cell = locator.get("cell", locator.get("source_cell"))
        if cell:
            cells = [item.get("cell"), item.get("source_cell"), *(item.get("value_cells") or [])]
            # Existing row representation explicitly lists coordinates as A1=... .
            cells += re.findall(r"(?<![A-Za-z0-9])([A-Z]+\d+)\s*=", str(item.get("content") or ""))
            if not any(_equal(cell, candidate) for candidate in cells):
                continue
        return True
    return False
