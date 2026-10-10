"""Structural catalog blocks, frozen from the prior shadow experiment.

Construction never receives questions or Gold. Blocks are discovery records;
original row/cell evidence remains unchanged in the source-local corpus.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Any
from haystack import Document
from sentence_transformers import SentenceTransformer


def token_count(model, text):
    return len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()

def row_key(meta: dict[str, Any]) -> str:
    table = meta.get("table") or meta.get("table_index")
    if table is None:
        table_id = str(meta.get("table_id") or "")
        table = table_id.rsplit(":", 1)[-1] if table_id else ""
    row = meta.get("row") or meta.get("row_start") or ""
    return f"{meta.get('sheet') or ''}|{table}|{row}"

def normalized_row(document: Document) -> str:
    text = compact(document.content or "")
    match = re.search(r"行：?(\d+)\s*(.*)$", text)
    if match:
        return f"行{match.group(1)}：{match.group(2)}".strip()
    return text

def group_key(document: Document) -> tuple[str, str]:
    meta = document.meta
    sheet = str(meta.get("sheet") or "")
    table = meta.get("table") or meta.get("table_index")
    if table is None:
        table_id = str(meta.get("table_id") or "")
        table = table_id.rsplit(":", 1)[-1] if table_id else ""
    return sheet, str(table or "")

def row_order(document: Document) -> tuple[int, int, str]:
    meta = document.meta
    return (
        int(meta.get("row") or meta.get("row_start") or 10**9),
        int(meta.get("row_end") or meta.get("row") or 10**9), str(document.id),
    )

def split_long_line(
    model: SentenceTransformer, base: str, line: str, key: str, meta_prefix: str, max_tokens: int
) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    remaining = line
    part = 1
    while remaining:
        low, high, accepted = 1, len(remaining), 0
        while low <= high:
            middle = (low + high) // 2
            trial = f"{meta_prefix}\n{base}\n行片段{part}：{remaining[:middle]}".strip()
            if token_count(model, trial) <= max_tokens:
                accepted = middle
                low = middle + 1
            else:
                high = middle - 1
        if accepted == 0:
            raise RuntimeError("Source metadata alone exhausts embedding context")
        parts.append((f"行片段{part}：{remaining[:accepted]}", key))
        remaining = remaining[accepted:]
        part += 1
    return parts

def build_blocks(
    model: SentenceTransformer, baseline: Document, source_documents: list[Document]
) -> list[Document]:
    row_documents = [item for item in source_documents if item.meta.get("evidence_type") == "table_row"]
    if not row_documents:
        row_documents = [item for item in source_documents if item.meta.get("evidence_type") == "table_row_group"]
    grouped: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for document in row_documents:
        grouped[group_key(document)].append(document)
    blocks: list[Document] = []
    max_tokens = int(getattr(model, "max_seq_length", 512) or 512)
    for sheet_table in sorted(grouped):
        sheet, table = sheet_table
        rows = sorted(grouped[sheet_table], key=row_order)
        base = (
            f"表格来源：{baseline.meta.get('source_title') or ''}\n"
            f"来源文件：{baseline.meta.get('source_path') or ''}\n"
            f"工作表：{sheet}\n表格：{table}"
        ).strip()
        meta = dict(baseline.meta) | {
            "sheet": sheet, "table": table, "evidence_type": "table_source_catalog_block",
        }
        meta_prefix = " ".join(str(meta.get(key, "")) for key in ("source_title", "sheet", "cell", "section"))
        prepared: list[tuple[str, str]] = []
        for row in rows:
            line, key = normalized_row(row), row_key(row.meta)
            if token_count(model, f"{meta_prefix}\n{base}\n{line}".strip()) <= max_tokens:
                prepared.append((line, key))
            else:
                prepared.extend(split_long_line(model, base, line, key, meta_prefix, max_tokens))
        current_lines: list[str] = []
        current_keys: list[str] = []
        for line, key in prepared:
            trial_lines = current_lines + [line]
            trial = f"{meta_prefix}\n{base}\n" + "\n".join(trial_lines)
            if current_lines and token_count(model, trial) > max_tokens:
                block_meta = dict(meta) | {"block_index": len(blocks) + 1, "row_keys": current_keys}
                blocks.append(Document(content=base + "\n" + "\n".join(current_lines), meta=block_meta))
                current_lines, current_keys = [line], [key]
            else:
                current_lines, current_keys = trial_lines, current_keys + [key]
        if current_lines or not rows:
            block_meta = dict(meta) | {"block_index": len(blocks) + 1, "row_keys": current_keys}
            blocks.append(Document(content=base + ("\n" + "\n".join(current_lines) if current_lines else ""), meta=block_meta))
    if not blocks:
        blocks.append(Document(content=baseline.content, meta=dict(baseline.meta) | {
            "evidence_type": "table_source_catalog_block", "block_index": 1, "row_keys": [],
        }))
    return blocks
