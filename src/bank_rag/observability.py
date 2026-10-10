"""Read-only Evidence Decision Trace helpers.

The helpers in this module only describe decisions that have already happened.
They do not rank, filter, mutate, or otherwise select evidence.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


REFUSAL_PATTERNS = (
    "依据不足", "无法确认", "无法回答", "不能回答", "未检索到", "没有可用于核验",
)
CLARIFICATION_PATTERNS = ("需要澄清", "请补充", "请明确")


def evidence_id(value: Any) -> str:
    """Return a stable, non-semantic identifier for a document/debug item."""
    if isinstance(value, dict):
        existing = value.get("evidence_id") or value.get("document_id") or value.get("id")
        meta = value
        content = str(value.get("content") or "")
    else:
        existing = getattr(value, "id", None)
        meta = getattr(value, "meta", {}) or {}
        content = str(getattr(value, "content", "") or "")
    if existing:
        return str(existing)
    identity = "|".join(str(meta.get(key) or "") for key in (
        "source_path", "source_title", "evidence_type", "page", "sheet", "row", "cell", "section",
    ))
    digest = hashlib.sha256(f"{identity}|{content}".encode("utf-8")).hexdigest()[:16]
    return f"ev-{digest}"


def evidence_snapshot(document: Any) -> dict[str, Any]:
    """Describe one evidence record, including structure-preservation signals."""
    meta = getattr(document, "meta", {}) or {}
    content = str(getattr(document, "content", "") or "")
    evidence_type = str(meta.get("evidence_type") or "")
    structured = evidence_type.startswith("table_") or evidence_type == "docx_table_row"
    value_cells = list(meta.get("value_cells") or [])
    cell_mentions = set(re.findall(r"\b[A-Z]{1,3}\d+\b", content))
    header_present = bool(meta.get("header_rows")) or "表头：" in content
    row_label_present = bool(meta.get("row_label")) or "指标行：" in content or "行标题" in content
    unit_present = bool(re.search(r"(?:单位[:：]|亿元|万元|万件|百分比|百分点|%)", content))
    value_present = bool(value_cells) or "数值单元格：" in content or bool(re.search(r"\d", content))
    adjacent_cells_present = len(value_cells) >= 2 or len(cell_mentions) >= 2
    return {
        "evidence_id": evidence_id(document),
        "source_title": meta.get("source_title") or meta.get("file_name") or "",
        "source_path": meta.get("source_path") or "",
        "evidence_type": evidence_type,
        "page": meta.get("page"),
        "sheet": meta.get("sheet"),
        "row": meta.get("row"),
        "cell": meta.get("cell"),
        "section": meta.get("section"),
        "structured_evidence": structured,
        "structure": {
            "header_present": header_present,
            "row_label_present": row_label_present,
            "adjacent_cells_present": adjacent_cells_present,
            "unit_present": unit_present,
            "value_present": value_present,
            "relation_recoverable": bool(
                not structured or (header_present and row_label_present and value_present)
            ),
        },
    }


def debug_evidence_ids(items: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(evidence_id(item) for item in items))


def final_behavior(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "").strip()
    if any(pattern in answer for pattern in CLARIFICATION_PATTERNS):
        return "clarify"
    if result.get("insufficient_evidence") or any(pattern in answer for pattern in REFUSAL_PATTERNS):
        return "refuse"
    if not answer:
        return "unknown"
    if any(pattern in answer for pattern in ("部分", "仅能确认", "只能确认")):
        return "partial"
    return "answer"


def build_evidence_decision_trace(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative runtime trace from the final LangGraph state."""
    attempts = list((state.get("retrieval_debug") or {}).get("attempts") or [])
    retrieval_items = [
        item for attempt in attempts
        for stage in ("bm25_top_k", "embedding_top_k")
        for item in (attempt.get(stage) or [])
    ]
    cross_items = [
        item for attempt in attempts for item in (attempt.get("cross_encoder_top_k") or [])
    ]
    documents = list(state.get("documents") or [])
    assessment = dict(state.get("evidence_assessment") or {})
    slots = []
    for slot in assessment.get("evidence_slots") or state.get("evidence_slots") or []:
        filled = bool(slot.get("filled"))
        slots.append({
            "slot_id": slot.get("id") or "",
            "required_condition": {
                key: slot.get(key) for key in ("period", "area", "entity", "metric") if slot.get(key)
            },
            "required_evidence": slot.get("required_evidence") or "",
            "filled": filled,
            "evidence_ids": list(slot.get("evidence_ids") or []),
            "reason": slot.get("match_reason") or (
                "matched_by_deterministic_slot_rules" if filled else "no_document_matched_all_slot_conditions"
            ),
        })
    behavior = final_behavior(result)
    complete = assessment.get("complete")
    graph_trace = list(state.get("graph_trace") or result.get("graph_trace") or [])
    explicitly_rejected = any(
        node in graph_trace for node in ("reject_incomplete", "reject_unsupported")
    )
    gate_decision = "refuse" if explicitly_rejected else ("answer" if complete else "refuse")
    if behavior == "clarify" and not complete:
        gate_decision = "clarify"
    if behavior == "partial" and complete:
        gate_decision = "partial"
    missing_slots = [slot["slot_id"] for slot in slots if not slot["filled"]]
    reasons = list(assessment.get("reasons") or [])
    assembly_ids = [evidence_id(document) for document in documents]
    generated = result.get("answer_method") == "deepseek_evidence_constrained"
    final_context_documents = documents if generated else []

    # Runtime has no Gold/Expected Source. Only label a stage when the state
    # proves it; otherwise retain unknown for the offline scorer/human audit.
    if behavior == "answer" and complete:
        death_stage = "no_failure"
        death_reason = "gate passed and the system produced an answer"
    elif not retrieval_items and not cross_items and not documents:
        death_stage = "retrieval"
        death_reason = "no candidate evidence was recorded and assembly is empty"
    elif complete and behavior in {"refuse", "clarify"} and "generate" in graph_trace:
        death_stage = "generation"
        death_reason = "evidence gate passed before the final refusal or clarification"
    else:
        death_stage = "unknown"
        death_reason = "runtime trace has no Gold/Expected Source proof for an earlier stage"

    return {
        "question_id": "",
        "question": state.get("question") or result.get("question") or "",
        "question_type": str((state.get("question_plan") or {}).get("intent") or ""),
        "retrieval": {
            "candidate_evidence_ids": debug_evidence_ids(retrieval_items),
            "candidate_count": len(retrieval_items),
        },
        "cross_encoder": {
            "evidence_ids": debug_evidence_ids(cross_items),
            "candidate_count": len(cross_items),
        },
        "required_slots": slots,
        "assembly": {
            "evidence_ids": assembly_ids,
            "evidence": [evidence_snapshot(document) for document in documents],
            "complete": complete,
            "reason": "；".join(reasons),
        },
        "sufficiency_decision": {
            "decision": gate_decision,
            "reason": "；".join(reasons) or ("evidence_complete" if complete else "evidence_incomplete"),
            "missing_slots": missing_slots,
        },
        "final_context": {
            "evidence_ids": [evidence_id(document) for document in final_context_documents],
            "evidence": [evidence_snapshot(document) for document in final_context_documents],
        },
        "final_behavior": behavior,
        "death_stage": death_stage,
        "death_reason": death_reason,
    }
