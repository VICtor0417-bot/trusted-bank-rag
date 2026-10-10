"""LangGraph orchestration for the trusted bank RAG workflow.

The graph keeps question planning, retrieval, evidence gating, deterministic
calculation, generation and output validation as separately observable steps.
Retrieval and answer implementation remain owned by ``TrustedRAG``.
"""

from __future__ import annotations

import os
import re
import inspect
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .canonicalization import area_match_key, area_present_in_text
from .financial_lexicon import metric_variants
from .period_binding import document_covers_period
from .question_parser import QuestionPlan, insurance_operating_overview_metrics, metric_binding_for, parse_question
from .table_selection import selection_slot_matches
from .observability import build_evidence_decision_trace, evidence_id
from .evidence_gate_policy import numeric_slot_fact, statistical_lookup, temporal_semantics, unsupported_requested_operation


class RAGState(TypedDict, total=False):
    question: str
    question_plan: dict[str, Any]
    source_routing: list[dict[str, Any]]
    llm_question_plan: dict[str, Any]
    catalog_promotions: list[dict[str, Any]]
    evidence_slots: list[dict[str, Any]]
    soft_evidence_hints: list[dict[str, Any]]
    documents: list[Any]
    retrieval_attempts: int
    retrieval_stages: list[str]
    retrieval_debug: dict[str, Any]
    evidence_assessment: dict[str, Any]
    missing_evidence_items: list[dict[str, Any]]
    supplemental_queries: list[str]
    query_rewrite: dict[str, Any]
    result: dict[str, Any]
    answer_validation: dict[str, Any]
    graph_trace: list[str]


def _plan(state: RAGState) -> QuestionPlan:
    return QuestionPlan(**state["question_plan"])


def _llm_planner_enabled() -> bool:
    """Allow evaluations to isolate the parallel LLM planner contribution."""
    return os.getenv("RAG_ENABLE_LLM_PLANNER", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _document_key(document: Any) -> str:
    document_id = getattr(document, "id", None)
    if document_id:
        return str(document_id)
    return f"{getattr(document, 'content', '')}\n{getattr(document, 'meta', {})!r}"


def _merge_documents(*groups: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for group in groups:
        for document in group:
            key = _document_key(document)
            if key not in seen:
                seen.add(key)
                merged.append(document)
    return merged


def _normalise_title(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value).lower()


def classify_planner_requirements(
    hard_slots: list[dict[str, Any]], planner_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep unverified LLM requirements as retrieval hints, never hard gates."""
    hard_terms = {
        _normalise_title(str(slot.get("required_evidence") or ""))
        for slot in hard_slots
        if slot.get("required_evidence")
    }
    hints: list[dict[str, Any]] = []
    for slot in planner_slots:
        required = str(slot.get("required_evidence") or "").strip()
        normalized = _normalise_title(required)
        confirmed = bool(normalized and normalized in hard_terms)
        hints.append({
            **slot,
            "requirement_level": "hard_confirmed" if confirmed else "soft",
            "can_block_answer": False,
            "validation": "matched_deterministic_slot" if confirmed else "planner_only",
        })
    return hints


def build_evidence_slots(plan: QuestionPlan) -> list[dict[str, Any]]:
    """Plan only the raw evidence required before a derived calculation.

    Difference, growth, ranking and aggregation are operations over source
    values.  They must not become retrieval requirements themselves.
    """
    question = plan.original_question
    selection = plan.table_selection
    if selection:
        return [
            {'id': f'slot_{i+1}', 'requirement_type': 'raw_evidence',
             'period': period, 'row_dimension': selection['row_dimension'],
             'row_value': row, 'column': column, 'metric': selection['base_metric'],
             'required_evidence': f'{period} {row} {column} {selection["base_metric"]}',
             'retrieval_terms': [f'{period} {row} {column} {selection["base_metric"]}'],
             'derived_operations': [selection['operation']] if selection['operation'] != 'lookup' else []}
            for i, (period, row, column) in enumerate(
                (p, r, c) for p in (plan.periods or [''])
                for r in selection['row_values'] for c in selection['columns'])
        ]
    overview_metrics = insurance_operating_overview_metrics(question)
    # A request for an insurance operating overview is a deterministic domain
    # expansion.  The user may not name any individual financial indicator,
    # but every standard row is still required to answer the comparison.  Do
    # not let the absence of a literal financial alias disable slot-aware
    # preservation for this class of full-table questions.
    if not plan.financial_aliases and not overview_metrics:
        return []
    quantitative = bool(
        overview_metrics
        or
        plan.intent == "table_calculation"
        or plan.operations
        or re.search(r"(?:多少|总额|总资产|总负债|资产总额|负债总额|收入|金额|数值|比率|率)", question)
    )
    if not quantitative:
        return []
    implicit_operations: list[str] = []
    if re.search(r"(?:哪类|哪个|哪一个|谁).{0,12}(?:最高|最低|更多|更少|较高|较低)", question):
        implicit_operations.append("比较排序")
    entities = list(dict.fromkeys(re.findall(
        r"大型商业银行|股份制商业银行|城市商业银行|农村商业银行|外资银行|"
        r"人身险公司|财产保险公司|财产险公司|再保险公司|保险公司", question
    )))
    periods = list(plan.periods)
    derived_concepts = {
        "合计", "比较", "增加额", "减少额", "增长率", "环比增长率", "同比增长率",
        "百分点变化", "差额", "平均值", "最大值", "最小值", "求和",
    }
    selector = (plan.table_program or {}).get("selector") or {}
    selected_metric = str(selector.get("metric") or plan.metric or "")
    binding = plan.metric_binding or metric_binding_for(question, selected_metric)
    metrics = list(overview_metrics)
    if not metrics:
        metrics = [selected_metric] if selected_metric and binding["hard"] and selected_metric not in derived_concepts else []
    if not metrics:
        metrics = list(dict.fromkeys(
            str(item.get("canonical") or "") for item in plan.financial_aliases
            if item.get("canonical") and str(item.get("canonical")) not in derived_concepts
            and not any(str(item.get("matched_alias") or "") in entity for entity in entities if entity)
        ))
    if not metrics:
        return []
    expanded_periods: list[str] = []
    for period in periods:
        match = re.fullmatch(r"(20\d{2})年第?([一二三四1-4])季度", period)
        # Monthly premium files are cumulative snapshots; a quarter request
        # needs each constituent month as an explicit evidence slot before the
        # system decides whether to use the quarter-end value or an increment.
        if match and plan.metric == "原保险保费收入":
            quarter_map = {"一": 1, "二": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}
            start = (quarter_map[match.group(2)] - 1) * 3 + 1
            expanded_periods.extend(f"{match.group(1)}年{month}月" for month in range(start, start + 3))
        else:
            expanded_periods.append(period)
    periods = expanded_periods or [""]
    entities = entities or [plan.entity or ""]
    areas = list(dict.fromkeys(plan.areas or ([plan.area] if plan.area else [""])))
    effective_operations = list(dict.fromkeys([*plan.operations, *implicit_operations]))
    annual_terminal_snapshot = bool(
        len(periods) == 1
        and re.fullmatch(r"20\d{2}年", periods[0])
        and any(re.search(r"20\d{2}年12月(?:保险业经营情况表|全国各地区原保险保费收入情况表)", title)
                for title in plan.source_titles)
    )
    if (
        len(periods) == 1 and len(areas) == 1 and len(entities) == 1
        and len(metrics) == 1 and not effective_operations
        and not annual_terminal_snapshot
    ):
        return []
    slots: list[dict[str, Any]] = []
    for period in periods:
        for area in areas:
            for entity in entities:
                for metric in metrics:
                    terms = [term for term in (period, area, entity, metric) if term]
                    slots.append({
                        "id": f"slot_{len(slots)+1}",
                        "requirement_type": "raw_evidence",
                        "period": period, "area": area, "entity": entity, "metric": metric,
                        "required_evidence": " ".join(terms),
                        "retrieval_terms": [" ".join(terms)],
                        "derived_operations": effective_operations,
                    })
    return slots


def _slot_is_filled(slot: dict[str, Any], documents: list[Any]) -> bool:
    if slot.get('row_value') and slot.get('column'):
        return any(numeric_slot_fact(document, slot) is not None for document in documents)
    for document in documents:
        meta = getattr(document, "meta", {})
        if str(meta.get("evidence_type", "")) == "table_source_catalog":
            continue
        content = str(getattr(document, "content", ""))
        title = str(meta.get("source_title", ""))
        text = f"{title} {content}"
        period = str(slot.get("period") or "")
        if period and not document_covers_period(period, document):
            continue
        area = str(slot.get("area") or "")
        compact_area = area_match_key(area)
        compact_text = re.sub(r"\s+", "", text)
        if compact_area and not area_present_in_text(area, text):
            continue
        entity = str(slot.get("entity") or "")
        entity_variants = {entity, entity.replace("财产保险公司", "财产险公司"), entity.replace("财产险公司", "财产保险公司")}
        if entity and not any(value and value in text for value in entity_variants):
            continue
        metric = str(slot.get("metric") or "")
        # A broad file title is not proof that this particular evidence record
        # contains the requested metric and its value.
        if metric and not any(value in re.sub(r"\s+", "", content) for value in metric_variants(metric)):
            continue
        return True
    return False


def assess_evidence(
    plan: QuestionPlan,
    documents: list[Any],
    source_routing: list[dict[str, Any]],
    evidence_slots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply a conservative, deterministic evidence-completeness gate."""
    usable = [
        document
        for document in documents
        if len(re.sub(r"\s+", "", str(getattr(document, "content", "")))) >= 8
        and not str(getattr(document, "content", "")).startswith("解析失败")
    ]
    expected_titles = plan.required_source_titles
    # Routing proposes places to look. It cannot create a required source.
    # Only explicit sources are required; inferred families remain hints.
    actual_titles = [
        str(getattr(document, "meta", {}).get("source_title", ""))
        for document in usable
    ]
    missing_sources = [
        expected
        for expected in expected_titles
        if expected
        and not any(
            _normalise_title(expected) in _normalise_title(actual)
            or _normalise_title(actual) in _normalise_title(expected)
            for actual in actual_titles
            if actual
        )
    ]
    slot_periods = list(dict.fromkeys(
        str(slot.get("period") or "") for slot in (evidence_slots or []) if slot.get("period")
    ))
    temporal = temporal_semantics(plan)
    expected_periods = (slot_periods or temporal['statistical_periods']) if temporal['statistical_lookup'] else []
    missing_periods = [
        period for period in expected_periods
        if not any(document_covers_period(period, document) for document in usable)
    ]
    table_evidence = [
        document for document in usable
        if str(getattr(document, "meta", {}).get("evidence_type", "")) != "table_source_catalog"
    ]
    binding = plan.metric_binding or metric_binding_for(plan.original_question, plan.metric)
    table_context = bool(
        statistical_lookup(plan)
        or evidence_slots
        or (binding["hard"] and re.search(
            r"多少|总额|金额|数值|余额|收入|比率|充足率|最高|最低", plan.original_question
        ))
    )
    missing_areas: list[str] = []
    if table_context:
        for area in plan.areas or ([plan.area] if plan.area else []):
            if area and not any(
                area_present_in_text(area, str(getattr(document, "content", "")))
                for document in table_evidence
            ):
                missing_areas.append(area)
    definition_request = bool(re.search(r"(?:怎么算|如何计算|计算公式|公式是什么|定义)", plan.original_question))
    requested_metrics = [plan.metric] if plan.metric and binding["hard"] else []
    if definition_request:
        requested_metrics = list(dict.fromkeys(
            str(item.get("canonical")) for item in plan.financial_aliases if item.get("canonical")
        )) or requested_metrics
    missing_metrics: list[str] = []
    if table_context or definition_request:
        for requested_metric in requested_metrics:
            variants = metric_variants(requested_metric)
            if not any(
                any(variant in str(getattr(document, "content", "")) for variant in variants)
                for document in usable
            ):
                missing_metrics.append(requested_metric)
    missing_metric = "、".join(missing_metrics) or None
    overview_metrics = insurance_operating_overview_metrics(plan.original_question)
    missing_overview_slots: list[tuple[str, str]] = []
    if overview_metrics and len(plan.source_titles) > 1:
        for expected_title in plan.source_titles:
            inferred_period = next((p for p in plan.periods if p in expected_title), None)
            source_documents = [
                document for document in table_evidence
                if (
                    plan.source_title_provenance == "deterministic_inferred"
                    and inferred_period
                    and document_covers_period(inferred_period, document)
                ) or (plan.source_title_provenance != "deterministic_inferred" and (
                    _normalise_title(expected_title) in _normalise_title(
                        str(getattr(document, "meta", {}).get("source_title", ""))
                    )
                    or _normalise_title(
                        str(getattr(document, "meta", {}).get("source_title", ""))
                    ) in _normalise_title(expected_title)
                ))
            ]
            for metric in overview_metrics:
                if not any(
                    metric in re.sub(r"\s+", "", str(getattr(document, "content", "")))
                    for document in source_documents
                ):
                    missing_overview_slots.append((expected_title, metric))
    # A catalog is a routing card, never answer evidence, even for a
    # regulatory query that did not request a numeric table lookup.
    missing_table_detail = bool(usable and not table_evidence)
    missing_items: list[dict[str, Any]] = []
    missing_items.extend({"kind": "source", "value": value, "description": f"指定来源 {value}"} for value in missing_sources)
    missing_items.extend({"kind": "period", "value": value, "description": f"统计期间 {value}"} for value in missing_periods)
    missing_items.extend(
        {
            "kind": "area",
            "value": value,
            "description": " ".join(part for part in [*plan.periods, value, plan.metric or ""] if part),
        }
        for value in missing_areas
    )
    for value in missing_metrics:
        missing_items.append({
            "kind": "metric", "value": value,
            "description": " ".join(part for part in [*plan.periods, *(plan.areas or []), value] if part),
        })
    if missing_table_detail:
        missing_items.append({
            "kind": "table_detail", "value": plan.metric or "表格明细",
            "description": " ".join(part for part in [*plan.periods, *(plan.areas or []), plan.metric or "", "具体数值 行 表头"] if part),
        })
    missing_items.extend(
        {
            "kind": "metric_slot",
            "value": metric,
            "description": f"{source_title} {metric} 指标行和数值",
        }
        for source_title, metric in missing_overview_slots
    )
    if not usable:
        missing_items.append({
            "kind": "general", "value": plan.metric or "可用证据",
            "description": " ".join(part for part in [*plan.periods, *(plan.areas or []), plan.metric or "", "监管文件或统计表证据"] if part),
        })
    slot_status = []
    for slot in (evidence_slots or []):
        matches = [document for document in usable if _slot_is_filled(slot, [document])]
        slot_status.append({
            **slot,
            "filled": bool(matches),
            "evidence_ids": [evidence_id(document) for document in matches],
            "accepted_numeric_groups": [fact for document in matches
                if slot.get('row_value') and slot.get('column')
                and (fact := numeric_slot_fact(document, slot)) is not None],
            "match_reason": (
                "matched_by_deterministic_slot_rules"
                if matches else "no_document_matched_all_slot_conditions"
            ),
        })
    for slot in slot_status:
        if not slot["filled"]:
            missing_items.append({
                "kind": "evidence_slot", "value": slot["id"],
                "description": slot["required_evidence"],
            })
    reasons: list[str] = []
    if not usable:
        reasons.append("没有召回可用正文或表格证据")
    if missing_sources:
        reasons.append("缺少指定来源：" + "、".join(missing_sources))
    if missing_periods:
        reasons.append("缺少统计期间：" + "、".join(missing_periods))
    if missing_areas:
        reasons.append("缺少地区明细：" + "、".join(missing_areas))
    if missing_metric:
        reasons.append("缺少指标证据：" + missing_metric)
    if missing_table_detail:
        reasons.append("仅召回表格来源目录，缺少可回答的具体行或单元格")
    if missing_overview_slots:
        reasons.append(
            "缺少月份与指标组合："
            + "、".join(f"{title} / {metric}" for title, metric in missing_overview_slots)
        )
    missing_slots = [slot for slot in slot_status if not slot["filled"]]
    if missing_slots:
        reasons.append("缺少必要证据槽：" + "、".join(slot["required_evidence"] for slot in missing_slots))
    return {
        "complete": bool(usable) and not missing_sources and not missing_periods and not missing_areas and not missing_metric and not missing_table_detail and not missing_overview_slots and not missing_slots,
        "temporal_semantics": temporal,
        "usable_document_count": len(usable),
        "expected_sources": expected_titles,
        "found_sources": list(dict.fromkeys(title for title in actual_titles if title)),
        "missing_sources": missing_sources,
        "expected_periods": expected_periods,
        "missing_periods": missing_periods,
        "missing_areas": missing_areas,
        "missing_metric": missing_metric,
        "missing_table_detail": missing_table_detail,
        "missing_overview_slots": [
            {"source_title": title, "metric": metric}
            for title, metric in missing_overview_slots
        ],
        "missing_evidence_items": missing_items,
        "evidence_slots": slot_status,
        "required_evidence": plan.required_evidence,
        "metric_binding": binding,
        "source_routing_is_hint": True,
        "reasons": reasons,
    }


def validate_answer(result: dict[str, Any], *, generated: bool) -> dict[str, Any]:
    """Validate output without another LLM call or automatic answer rewrite."""
    answer = str(result.get("answer") or "").strip()
    citations = list(result.get("citations") or [])
    inline_references = re.findall(r"\[E\d+\]", answer)
    issues: list[str] = []
    if not answer:
        issues.append("答案为空")
    if not citations and not result.get("insufficient_evidence"):
        issues.append("没有结构化来源引用")
    if generated and answer and not inline_references and not result.get("insufficient_evidence"):
        issues.append("生成答案没有 [E编号] 行内引用")
    if result.get("verified") is False and re.search(r"\d", answer):
        issues.append("答案中的数字未全部通过证据匹配")
    return {
        "status": "pass" if not issues else "warning",
        "non_empty": bool(answer),
        "citation_count": len(citations),
        "inline_evidence_references": inline_references,
        "number_verified": result.get("verified"),
        "issues": issues,
        "llm_review_used": False,
    }


def build_rag_graph(rag: Any):
    """Compile one graph bound to an initialized ``TrustedRAG`` instance."""

    def retrieve_with_required_slots(state: RAGState, **kwargs: Any) -> list[Any]:
        """Pass hard slots when supported while retaining adapter compatibility."""
        parameters = inspect.signature(rag.retrieve).parameters
        if "evidence_slots" in parameters:
            kwargs["evidence_slots"] = state.get("evidence_slots", [])
        return rag.retrieve(state["question"], **kwargs)

    def parse_node(state: RAGState) -> RAGState:
        plan = parse_question(state["question"])
        return {
            "question_plan": plan.to_dict(),
            "graph_trace": [*state.get("graph_trace", []), "parse_question"],
        }

    def plan_evidence_node(state: RAGState) -> RAGState:
        plan = _plan(state)
        slots = build_evidence_slots(plan)
        routing = rag._route_sources(state["question"], plan)
        # Each evidence slot gets an independent source-routing chance. This
        # prevents one entity or month from occupying the whole candidate set.
        for slot in slots:
            slot_query = str(slot.get("required_evidence") or "")
            try:
                slot_routes = rag._route_sources(slot_query, parse_question(slot_query), limit=1)
            except TypeError:  # lightweight test doubles and legacy adapters
                slot_routes = rag._route_sources(slot_query, parse_question(slot_query))[:1]
            for candidate in slot_routes:
                if candidate.get("source_title") not in {item.get("source_title") for item in routing}:
                    routing.append(candidate)
        return {
            "source_routing": routing,
            "evidence_slots": slots,
            "retrieval_attempts": 0,
            "graph_trace": [*state.get("graph_trace", []), "plan_required_evidence"],
        }

    def after_parallel_planning(state: RAGState) -> str:
        # Unsupported calculations still need a normal retrieval pass.  The
        # evidence result tells the user whether the source data is absent in
        # addition to explaining which operation cannot be performed.
        return "check_evidence_completeness"

    def reject_unsupported_node(state: RAGState) -> RAGState:
        plan = _plan(state)
        assessment = state.get("evidence_assessment", {})
        result = {
            "question": state["question"],
            "question_plan": state["question_plan"],
            "source_routing": state.get("source_routing", []),
            "citations": rag.citations(state.get("documents", [])),
            "verified": False,
            "insufficient_evidence": True,
            "answer": "已找到相关证据，但该问题包含当前本地计算器尚不支持的“"
            + "、".join(plan.unsupported_operations)
            + "”运算，因此无法完成该部分。",
            "evidence_assessment": assessment,
        }
        return {"result": result, "graph_trace": [*state.get("graph_trace", []), "reject_unsupported"]}

    def retrieval_node(state: RAGState) -> RAGState:
        documents = retrieve_with_required_slots(
            state, plan=_plan(state), routed_sources=state.get("source_routing", []),
            queries=[state["question"]],
        )
        mode = str(getattr(rag, "retrieval_mode", "hybrid"))
        stages = ["bm25"] if mode == "bm25" else ["bm25", "embedding", "rrf_fusion", "cross_encoder_rerank"]
        return {
            "documents": documents,
            "retrieval_attempts": 1,
            "retrieval_stages": stages,
            "retrieval_debug": {"attempts": [dict(getattr(rag, "last_retrieval_debug", {}))]},
        }

    def llm_question_understanding_node(state: RAGState) -> RAGState:
        if not _llm_planner_enabled():
            return {
                "llm_question_plan": {
                    "task_type": "",
                    "conditions": {
                        "time": [], "region": [], "institution": [], "metric": [], "other": [],
                    },
                    "source_routing": [],
                    "evidence_slots": [],
                    "method": "disabled_by_RAG_ENABLE_LLM_PLANNER",
                }
            }
        return {"llm_question_plan": rag.understand_question(state["question"])}

    def merge_parallel_node(state: RAGState) -> RAGState:
        planner_slots = list(state.get("llm_question_plan", {}).get("evidence_slots", []))
        planner_method = str(state.get("llm_question_plan", {}).get("method", ""))
        planner_trace = (
            "llm_question_understanding_disabled"
            if planner_method == "disabled_by_RAG_ENABLE_LLM_PLANNER"
            else "llm_question_understanding"
        )
        return {
            "soft_evidence_hints": classify_planner_requirements(
                state.get("evidence_slots", []), planner_slots
            ),
            "graph_trace": [
                *state.get("graph_trace", []),
                "parallel_original_retrieval_and_llm_planning",
                "hybrid_retrieval",
                planner_trace,
                "merge_question_understanding",
            ]
        }

    def promote_catalog_node(state: RAGState) -> RAGState:
        attempts = state.get("retrieval_debug", {}).get("attempts", [])
        candidates = rag.promote_table_catalogs(
            state["question"], _plan(state), attempts[0] if attempts else {}
        ) if hasattr(rag, "promote_table_catalogs") else []
        existing = list(state.get("source_routing", []))
        new = [item for item in candidates if item["source_title"] not in {
            str(route.get("source_title")) for route in existing
        }]
        if not new:
            return {"catalog_promotions": candidates, "graph_trace": [*state.get("graph_trace", []), "promote_table_catalogs"]}
        routing = [*existing, *new]
        additional = retrieve_with_required_slots(
            state, plan=_plan(state), routed_sources=routing,
            queries=[state["question"]],
        )
        debug = dict(state.get("retrieval_debug", {}))
        debug["catalog_promotion"] = {
            "candidates": new,
            "file_internal_top_k": rag._debug_documents(additional, len(additional)) if hasattr(rag, "_debug_documents") else [],
        }
        return {
            "source_routing": routing,
            "documents": _merge_documents(additional, state.get("documents", [])),
            "retrieval_debug": debug,
            "catalog_promotions": candidates,
            "graph_trace": [*state.get("graph_trace", []), "promote_table_catalogs", "file_internal_retrieval"],
        }

    def evidence_check_node(state: RAGState) -> RAGState:
        assessment = assess_evidence(
            _plan(state), state.get("documents", []), state.get("source_routing", []),
            state.get("evidence_slots", []),
        )
        return {
            "evidence_assessment": assessment,
            "graph_trace": [*state.get("graph_trace", []), "check_evidence_completeness"],
        }

    def after_evidence_check(state: RAGState) -> str:
        if state["evidence_assessment"]["complete"]:
            if unsupported_requested_operation(_plan(state)):
                return "reject_unsupported"
            return "classify_problem"
        # One normal retrieval pass is enough to establish whether prerequisite
        # data exists. A rewrite retry cannot make an unsupported operation
        # executable and only delays a response that must still be rejected.
        if unsupported_requested_operation(_plan(state)):
            return "reject_incomplete_evidence"
        if state.get("retrieval_attempts", 0) < 2:
            return "rewrite_missing_evidence"
        return "reject_incomplete_evidence"

    def rewrite_missing_evidence_node(state: RAGState) -> RAGState:
        plan = _plan(state)
        assessment = state.get("evidence_assessment", {})
        missing_items = list(assessment.get("missing_evidence_items") or [])
        context_parts = []
        if assessment.get("found_sources"):
            context_parts.append("已召回来源：" + "；".join(assessment["found_sources"][:8]))
        if state.get("source_routing"):
            context_parts.append("计划数据源：" + "；".join(str(item.get("source_title", "")) for item in state["source_routing"][:8]))
        rewrite = rag.rewrite_missing_evidence(
            state["question"], plan, missing_items, "\n".join(context_parts)
        )
        planner_terms = [
            str(term).strip()
            for slot in state.get("soft_evidence_hints", [])
            for term in slot.get("retrieval_terms", [])
            if str(term).strip()
        ]
        supplemental_queries = list(dict.fromkeys([
            *planner_terms, *list(rewrite.get("queries") or [])
        ]))[:8]
        return {
            "missing_evidence_items": missing_items,
            "supplemental_queries": supplemental_queries,
            "query_rewrite": rewrite,
            "graph_trace": [*state.get("graph_trace", []), "rewrite_missing_evidence"],
        }

    def after_rewrite(state: RAGState) -> str:
        return "supplemental_retrieval" if state.get("supplemental_queries") else "reject_incomplete_evidence"

    def supplemental_retrieval_node(state: RAGState) -> RAGState:
        plan = _plan(state)
        additional = retrieve_with_required_slots(
            state, plan=plan, routed_sources=state.get("source_routing", []),
            queries=state.get("supplemental_queries", []),
        )
        retrieval_debug = dict(state.get("retrieval_debug", {}))
        retrieval_debug["attempts"] = [
            *retrieval_debug.get("attempts", []),
            dict(getattr(rag, "last_retrieval_debug", {})),
        ]
        return {
            "documents": _merge_documents(state.get("documents", []), additional),
            "retrieval_attempts": state.get("retrieval_attempts", 1) + 1,
            "retrieval_debug": retrieval_debug,
            "graph_trace": [*state.get("graph_trace", []), "supplemental_retrieval"],
        }

    def reject_incomplete_evidence_node(state: RAGState) -> RAGState:
        plan = _plan(state)
        assessment = state["evidence_assessment"]
        reason = "；".join(assessment.get("reasons", [])) or "证据不完整"
        unsupported = ""
        if plan.unsupported_operations:
            planner_requirements = [
                str(slot.get("required_evidence", "")).strip()
                for slot in state.get("llm_question_plan", {}).get("evidence_slots", [])
                if str(slot.get("required_evidence", "")).strip()
            ]
            if planner_requirements:
                reason += "；未完整找到问题所需数据：" + "、".join(planner_requirements[:3])
            unsupported = (
                "；同时，当前本地计算器暂不支持“"
                + "、".join(plan.unsupported_operations)
                + "”运算，无法完成该部分"
            )
        result = {
            "question": state["question"],
            "question_plan": state["question_plan"],
            "source_routing": state.get("source_routing", []),
            "citations": rag.citations(state.get("documents", [])),
            "verified": False,
            "insufficient_evidence": True,
            "answer": f"依据不足，无法确认。{reason}{unsupported}。",
        }
        return {"result": result, "graph_trace": [*state.get("graph_trace", []), "reject_incomplete_evidence"]}

    def classify_node(state: RAGState) -> RAGState:
        return {"graph_trace": [*state.get("graph_trace", []), "classify_problem_type"]}

    def after_classification(state: RAGState) -> str:
        return "table_calculator" if _plan(state).intent == "table_calculation" else "generate"

    def table_node(state: RAGState) -> RAGState:
        table_result = rag._deterministic_table_answer(state["question"])
        trace = [*state.get("graph_trace", []), "deterministic_table_calculator"]
        if not table_result:
            return {"graph_trace": trace}
        answer, evidence = table_result
        return {
            "result": {
                "question": state["question"],
                "question_plan": state["question_plan"],
                "source_routing": state.get("source_routing", []),
                "answer": answer,
                "evidence": [document.content for document in evidence],
                "citations": rag.citations(evidence),
                "verified": True,
                "answer_method": "deterministic_python",
            },
            "graph_trace": trace,
        }

    def after_table(state: RAGState) -> str:
        return "validate_answer" if state.get("result") else "generate"

    def generate_node(state: RAGState) -> RAGState:
        generation_kwargs = {}
        action = state.get("evidence_assessment", {}).get("shadow_jev_action")
        if action == "needs_clarification":
            generation_kwargs["generation_advisory"] = {
                "judgment_source": "existing_jev", "action": action,
            }
        result = rag._ask_pipeline(
            state["question"], plan=_plan(state), source_routing=state.get("source_routing", []),
            docs=state.get("documents", []), try_table=False,
            **generation_kwargs,
        )
        result.setdefault("answer_method", "deepseek_evidence_constrained")
        return {"result": result, "graph_trace": [*state.get("graph_trace", []), "generate"]}

    def validate_node(state: RAGState) -> RAGState:
        generated = state.get("result", {}).get("answer_method") == "deepseek_evidence_constrained"
        validation = validate_answer(state["result"], generated=generated)
        result = dict(state["result"])
        result["answer_validation"] = validation
        return {
            "result": result,
            "answer_validation": validation,
            "graph_trace": [*state.get("graph_trace", []), "validate_answer_without_llm"],
        }

    def finalize_node(state: RAGState) -> RAGState:
        result = dict(state["result"])
        trace = [*state.get("graph_trace", []), "finalize"]
        result["orchestrator"] = "langgraph"
        result["framework_version"] = "evidence-gated-v4-parallel-question-planning"
        result["graph_trace"] = trace
        if state.get("retrieval_stages"):
            result["retrieval_stages"] = state["retrieval_stages"]
            result["retrieval_attempts"] = state.get("retrieval_attempts", 0)
        if state.get("retrieval_debug"):
            result["retrieval_debug"] = state["retrieval_debug"]
        if state.get("evidence_assessment"):
            result["evidence_assessment"] = state["evidence_assessment"]
        if state.get("missing_evidence_items"):
            result["missing_evidence_items"] = state["missing_evidence_items"]
        if state.get("supplemental_queries"):
            result["supplemental_queries"] = state["supplemental_queries"]
        if state.get("query_rewrite"):
            result["query_rewrite"] = state["query_rewrite"]
        if state.get("llm_question_plan"):
            result["llm_question_plan"] = state["llm_question_plan"]
        if state.get("catalog_promotions"):
            result["catalog_promotions"] = state["catalog_promotions"]
        if state.get("evidence_slots"):
            result["evidence_slots"] = state["evidence_slots"]
        if state.get("soft_evidence_hints"):
            result["soft_evidence_hints"] = state["soft_evidence_hints"]
        result["evidence_decision_trace"] = build_evidence_decision_trace(state, result)
        return {"result": result, "graph_trace": trace}

    builder = StateGraph(RAGState)
    builder.add_node("parse_question", parse_node)
    builder.add_node("plan_required_evidence", plan_evidence_node)
    builder.add_node("reject_unsupported", reject_unsupported_node)
    builder.add_node("hybrid_retrieval", retrieval_node)
    builder.add_node("llm_question_understanding", llm_question_understanding_node)
    builder.add_node("merge_question_understanding", merge_parallel_node)
    builder.add_node("promote_table_catalogs", promote_catalog_node)
    builder.add_node("check_evidence_completeness", evidence_check_node)
    builder.add_node("rewrite_missing_evidence", rewrite_missing_evidence_node)
    builder.add_node("supplemental_retrieval", supplemental_retrieval_node)
    builder.add_node("reject_incomplete_evidence", reject_incomplete_evidence_node)
    builder.add_node("classify_problem", classify_node)
    builder.add_node("table_calculator", table_node)
    builder.add_node("generate", generate_node)
    builder.add_node("validate_answer", validate_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "parse_question")
    builder.add_edge("parse_question", "plan_required_evidence")
    builder.add_edge("plan_required_evidence", "hybrid_retrieval")
    builder.add_edge("plan_required_evidence", "llm_question_understanding")
    builder.add_edge(["hybrid_retrieval", "llm_question_understanding"], "merge_question_understanding")
    builder.add_conditional_edges("merge_question_understanding", after_parallel_planning, {
        "reject_unsupported": "reject_unsupported",
        "check_evidence_completeness": "promote_table_catalogs",
    })
    builder.add_edge("promote_table_catalogs", "check_evidence_completeness")
    builder.add_edge("reject_unsupported", "finalize")
    builder.add_conditional_edges("check_evidence_completeness", after_evidence_check, {
        "classify_problem": "classify_problem", "rewrite_missing_evidence": "rewrite_missing_evidence",
        "reject_unsupported": "reject_unsupported",
        "reject_incomplete_evidence": "reject_incomplete_evidence",
    })
    builder.add_conditional_edges("rewrite_missing_evidence", after_rewrite, {
        "supplemental_retrieval": "supplemental_retrieval",
        "reject_incomplete_evidence": "reject_incomplete_evidence",
    })
    builder.add_edge("supplemental_retrieval", "check_evidence_completeness")
    builder.add_edge("reject_incomplete_evidence", "finalize")
    builder.add_conditional_edges("classify_problem", after_classification, {
        "table_calculator": "table_calculator", "generate": "generate",
    })
    builder.add_conditional_edges("table_calculator", after_table, {
        "validate_answer": "validate_answer", "generate": "generate",
    })
    builder.add_edge("generate", "validate_answer")
    builder.add_edge("validate_answer", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def invoke_rag_graph(rag: Any, question: str) -> dict[str, Any]:
    """Invoke a cached compiled graph and return the public result payload."""
    graph = getattr(rag, "_compiled_rag_graph", None)
    if graph is None:
        graph = build_rag_graph(rag)
        rag._compiled_rag_graph = graph
    try:
        state = graph.invoke({"question": question, "graph_trace": []})
        return state["result"]
    except Exception as exc:
        logging.getLogger(__name__).exception('RAG workflow failed')
        return {"question": question, "answer": "系统在处理本次请求时发生异常，未能完成可靠回答；请稍后重试。",
                "answer_action": "DELIVERY_ERROR", "runtime_error": type(exc).__name__,
                "runtime_failure_stage": "langgraph_invocation", "verified": False,
                "insufficient_evidence": False, "citations": [], "orchestrator": "langgraph"}
