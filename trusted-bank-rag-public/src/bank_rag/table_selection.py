"""Explicit row/column contracts; no question IDs, Gold or numeric-neighbor inference."""
from __future__ import annotations
import re
from decimal import Decimal, InvalidOperation
from typing import Any

PREMIUM_COLUMNS = {
    '财产险': ('财产险', '财险', '财产保险'),
    '寿险': ('寿险', '人寿险', '人寿保险'),
    '意外险': ('意外险', '意外伤害保险'),
    '健康险': ('健康险', '健康保险'),
}


def explicit_premium_selection(question: str, areas: list[str], metric: str | None) -> dict:
    if metric != '原保险保费收入':
        return {}
    columns = [label for label, aliases in PREMIUM_COLUMNS.items() if any(alias in question for alias in aliases)]
    scopes = list(dict.fromkeys(re.findall(r'公司本级|全国合计', question)))
    quoted_rows = re.findall(r'(?:行|统计口径)[“"]([^”"]+)[”"]', question)
    rows = scopes or quoted_rows or areas
    if not columns or not rows:
        return {}
    operation = 'sum' if len(columns) > 1 and re.search(r'合计|求和|相加|加总|总和', question) else 'lookup'
    return {'row_dimension': 'reporting_scope' if scopes or quoted_rows else 'region',
            'row_values': rows, 'columns': columns, 'base_metric': metric,
            'operation': operation, 'origin': 'explicit_user_conditions'}


def row_key(value: str) -> str:
    return re.sub(r'(?:特别行政区|自治区|省|市)$', '', re.sub(r'\s+', '', value))


def bound_fact(document: Any, row_value: str, column_label: str) -> dict | None:
    """Require header+row+numeric coordinate in one stored evidence group."""
    meta = document.meta
    if meta.get('evidence_type') != 'table_evidence_group' or row_key(str(meta.get('row_label', ''))) != row_key(row_value):
        return None
    content = str(document.content or '')
    if '表头：' not in content or '指标行：' not in content:
        return None
    header, body = content.split('指标行：', 1)
    header = header.split('表头：', 1)[1]
    body = body.split('数值单元格：', 1)[0]
    cells = lambda text: re.findall(r'\b([A-Z]+)(\d+)=([^；;\n]+)', text)
    columns = {col for col, _, label in cells(header) if re.sub(r'\s+', '', label) in PREMIUM_COLUMNS.get(column_label, (column_label,))}
    if len(columns) != 1:
        return None
    col = next(iter(columns))
    matches = [(f'{c}{r}', value.strip()) for c, r, value in cells(body) if c == col and int(r) == meta.get('row')]
    if len(matches) != 1:
        return None
    coordinate, raw = matches[0]
    if coordinate not in meta.get('value_cells', []):
        return None
    try:
        value = Decimal(raw.replace(',', ''))
    except InvalidOperation:
        return None
    units = re.findall(r'单位\s*[:：]\s*([^；;\n]+)', header)
    if not value.is_finite() or len(set(units)) != 1:
        return None
    return {'row_value': row_value, 'column': column_label, 'cell': coordinate, 'value': value, 'unit': units[0].strip()}


def selection_slot_matches(document: Any, slot: dict) -> bool:
    period = re.sub(r'\s+', '', str(slot.get('period', '')))
    text = re.sub(r'\s+', '', str(document.meta.get('source_title', ''))+' '+str(document.content or ''))
    return (not period or period in text) and bound_fact(document, slot['row_value'], slot['column']) is not None
