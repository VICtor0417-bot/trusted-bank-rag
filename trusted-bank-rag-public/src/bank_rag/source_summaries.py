"""Deterministic file-level summaries used by planning and catalog queries."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REGIONS = [
    "全国", "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃",
    "青海", "宁夏", "新疆", "深圳", "大连", "宁波", "厦门", "青岛", "广州", "无锡",
]
INSTITUTIONS = [
    "大型商业银行", "股份制商业银行", "城市商业银行", "农村商业银行", "外资银行",
    "商业银行", "消费金融公司", "财产保险公司", "人身险公司", "保险公司", "银行保险机构",
]
METRICS = [
    "总资产", "总负债", "资产总额", "保险金额", "原保险保费收入", "原保险赔付支出",
    "不良贷款率", "不良贷款余额", "拨备覆盖率", "贷款拨备率", "净利润", "净资产",
    "核心偿付能力充足率", "综合偿付能力充足率", "最低资本", "保险风险最低资本",
    "长寿风险", "财产保险", "人寿保险", "健康保险", "意外伤害保险",
]


def _meta(record: Any) -> dict[str, Any]:
    return record.get("meta", {}) if isinstance(record, dict) else dict(getattr(record, "meta", {}) or {})


def _content(record: Any) -> str:
    return str(record.get("content", "") if isinstance(record, dict) else getattr(record, "content", "") or "")


def _ordered(values: set[str]) -> list[str]:
    return sorted((value for value in values if value), key=lambda value: (len(value), value))


def build_file_summaries(knowledge: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with knowledge.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                groups[str(record.get("meta", {}).get("source_title", ""))].append(record)

    summaries: list[dict[str, Any]] = []
    for title, records in groups.items():
        first = _meta(records[0])
        texts = [_content(record) for record in records]
        searchable = title + "\n" + "\n".join(texts)
        evidence_types = {str(_meta(record).get("evidence_type", "")) for record in records}
        document_type = "统计表" if any(value.startswith("table_") for value in evidence_types) else "监管制度"
        title_years = re.findall(r"(?:19|20)\d{2}", title)
        years = sorted(set(re.findall(r"(?:19|20)\d{2}", searchable)))
        months = sorted(set(re.findall(r"(?:19|20)\d{2}年(?:1[0-2]|[1-9])月", searchable)))
        quarters = sorted(set(re.findall(r"(?:19|20)\d{2}年第?[一二三四1-4]季度", searchable)))
        time_scope = list(dict.fromkeys([*months, *quarters, *[f"{year}年" for year in years]]))[:36]
        regions = [value for value in REGIONS if value in searchable]
        institutions = [value for value in INSTITUTIONS if value in searchable]
        metrics = [value for value in METRICS if value in searchable]
        sheets = _ordered({str(_meta(record).get("sheet", "")) for record in records})
        sections = _ordered({str(_meta(record).get("section", "")) for record in records})
        pages = [int(_meta(record)["page"]) for record in records if str(_meta(record).get("page", "")).isdigit()]
        source_id_match = re.match(r"\s*(\d{1,4})_", title)
        domain = "保险业" if re.search(r"保险|偿付能力|寿险", searchable) else "银行业" if re.search(r"银行|贷款|资本", searchable) else "金融监管"
        capabilities = []
        if any(re.search(r"\d", text) for text in texts): capabilities.append("数值查询")
        if re.search(r"(?:定义|是指|计算公式|=|＝)", searchable): capabilities.append("定义或公式")
        if document_type == "监管制度": capabilities.append("制度条款查询")
        if len(time_scope) > 1: capabilities.append("多期比较")
        topic = "、".join(metrics[:8] or institutions[:6] or sheets[:4] or sections[:4]) or "文件正文"
        summary = f"{document_type}，属于{domain}，主要覆盖{topic}。"
        if time_scope: summary += "时间范围包括" + "、".join(time_scope[:8]) + "。"
        if regions: summary += "地区范围包括" + "、".join(regions[:8]) + "。"
        retrieval_terms = list(dict.fromkeys([
            re.sub(r"^\s*\d{1,4}_", "", title),
            *[" ".join(part for part in [time_scope[0] if time_scope else "", institution, metric] if part) for institution in institutions[:3] for metric in metrics[:4]],
            *metrics[:8], *sections[:4], *sheets[:4],
        ]))[:24]
        summaries.append({
            "source_id": source_id_match.group(1) if source_id_match else str(first.get("doc_id", "")),
            "source_title": title,
            "source_path": first.get("source_path", ""),
            "primary_year": int(title_years[0]) if title_years else None,
            "document_type": document_type,
            "domain": domain,
            "time_scope": time_scope,
            "region_scope": regions,
            "institution_scope": institutions,
            "metrics": metrics,
            "contains_definition": bool(re.search(r"(?:定义|是指)", searchable)),
            "contains_formula": bool(re.search(r"(?:计算公式|[=＝]|÷|/).{0,40}(?:100%|余额|资本)", searchable)),
            "contains_numeric_data": any(re.search(r"\d", text) for text in texts),
            "sheets_or_sections": [*sheets[:12], *sections[:12]],
            "content_capabilities": capabilities,
            "location": {"page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None, "record_count": len(records)},
            "summary": summary,
            "retrieval_terms": [term for term in retrieval_terms if term],
        })
    return sorted(summaries, key=lambda item: (item["source_id"], item["source_title"]))


def write_file_summaries(knowledge: Path, output: Path) -> int:
    summaries = build_file_summaries(knowledge)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in summaries:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(summaries)


def load_file_summaries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary_packets(knowledge: Path) -> list[dict[str, Any]]:
    """Build bounded source packets. DeepSeek writes the semantic summaries."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with knowledge.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                groups[str(record.get("meta", {}).get("source_title", ""))].append(record)
    packets: list[dict[str, Any]] = []
    for title, records in groups.items():
        first = _meta(records[0])
        source_id_match = re.match(r"\s*(\d{1,4})_", title)
        title_years = re.findall(r"(?:19|20)\d{2}", title)
        sheets = list(dict.fromkeys(str(_meta(row).get("sheet", "")) for row in records if _meta(row).get("sheet")))[:15]
        sections = list(dict.fromkeys(str(_meta(row).get("section", "")) for row in records if _meta(row).get("section")))[:15]
        pages = [int(_meta(row)["page"]) for row in records if str(_meta(row).get("page", "")).isdigit()]
        # Evenly sample the file so long regulations are not represented only by their cover page.
        indexes = sorted(set(round(i * (len(records) - 1) / 7) for i in range(min(8, len(records)))))
        excerpts = []
        for index in indexes:
            row = records[index]
            text = re.sub(r"\s+", " ", _content(row)).strip()[:700]
            if text:
                excerpts.append({"location": {k: _meta(row).get(k) for k in ("page", "sheet", "section") if _meta(row).get(k) is not None}, "text": text})
        packets.append({
            "source_id": source_id_match.group(1) if source_id_match else str(first.get("doc_id", "")),
            "source_title": title,
            "source_path": first.get("source_path", ""),
            "primary_year": int(title_years[0]) if title_years else None,
            "location": {"page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None, "record_count": len(records)},
            "sheets": sheets, "sections": sections, "excerpts": excerpts,
        })
    return sorted(packets, key=lambda item: (item["source_id"], item["source_title"]))


def write_llm_file_summaries(knowledge: Path, output: Path, *, batch_size: int = 6) -> int:
    """Ask DeepSeek to create every semantic file summary, with resumable output."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for LLM summaries")
    packets = _summary_packets(knowledge)
    existing = {item.get("source_title"): item for item in load_file_summaries(output)} if output.exists() else {}
    pending = [item for item in packets if item["source_title"] not in existing or existing[item["source_title"]].get("summary_generator") != "deepseek"]
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=float(os.getenv("DEEPSEEK_SUMMARY_TIMEOUT", "90")),
        max_retries=1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt = """你是银行保险监管知识库的文件梗概生成模块。你不能回答用户问题，只能根据给定文件材料制作检索用梗概。
对每个输入文件严格输出一个 JSON 对象，并以 JSON 数组返回。不得使用材料外知识，不得把示例数字误认为年份或范围。
每个对象必须包含：
source_id；document_type（统计表/监管制度/其他）；domain（银行业/保险业/金融监管/其他）；time_scope；region_scope；institution_scope；metrics；contains_definition；contains_formula；contains_numeric_data；sheets_or_sections；content_capabilities；summary；retrieval_terms。
其中 time_scope/region_scope/institution_scope/metrics/sheets_or_sections/content_capabilities/retrieval_terms 必须是字符串数组；summary 用2-4句说明文件是什么、包含哪些内容、适合回答什么问题；retrieval_terms 给出8-20个更可能出现在用户问题、文件标题、表头或正文中的正式检索表达。只保留材料明确支持的内容。

输入文件：
""" + json.dumps(batch, ensure_ascii=False)
        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=7000,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        parsed = json.loads(raw)
        generated = parsed if isinstance(parsed, list) else parsed.get("files") or parsed.get("summaries") or parsed.get("items")
        if not isinstance(generated, list):
            raise ValueError("DeepSeek summary response did not contain a JSON array")
        by_id = {str(item.get("source_id")): item for item in generated if isinstance(item, dict)}
        for packet in batch:
            semantic = by_id.get(str(packet["source_id"]))
            if not semantic:
                raise ValueError(f"DeepSeek omitted source_id={packet['source_id']}")
            immutable = {key: packet[key] for key in ("source_id", "source_title", "source_path", "primary_year", "location")}
            semantic.update(immutable)
            semantic["summary_generator"] = "deepseek"
            existing[packet["source_title"]] = semantic
        ordered = [existing[item["source_title"]] for item in packets if item["source_title"] in existing]
        with output.open("w", encoding="utf-8") as handle:
            for item in ordered:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"deepseek_summaries={len(existing)}/{len(packets)}", flush=True)
    return len(existing)


def earliest_file_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every file whose title carries the earliest explicit year."""
    dated = [item for item in summaries if isinstance(item.get("primary_year"), int)]
    if not dated:
        return []
    earliest = min(item["primary_year"] for item in dated)
    return [item for item in dated if item["primary_year"] == earliest]
