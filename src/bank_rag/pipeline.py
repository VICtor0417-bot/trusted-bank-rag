import json
import os
import re
from difflib import SequenceMatcher
from dataclasses import replace
from pathlib import Path
from typing import Any

from haystack import Document, Pipeline, component
from haystack.dataclasses import ChatMessage
from haystack.components.builders import PromptBuilder
try:
    from haystack.components.generators.chat import OpenAIChatGenerator
except ImportError:  # Haystack 2.18 lazy-export compatibility on Windows
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

from .canonicalization import area_match_key, area_present_in_text
from .financial_lexicon import canonical_financial_term, metric_variants
from .generation_policy import clarification_advisory, GenerationReply, generate_with_one_incomplete_retry, sdk_length_reply
from .evidence_gate_policy import CASE_CONTRACT, decode_case_contract, needs_case_contract, temporal_semantics, unsupported_requested_operation
from .question_parser import QuestionPlan, insurance_operating_overview_metrics, parse_question
from .observability import evidence_id
from .table_selection import bound_fact, selection_slot_matches
from .initial_global_cache import InitialGlobalCache


EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# The supplied evaluation set omits the literal multiple-choice options.  In
# that format, a question such as "which statement is correct" is a request to
# recover the supported statement, not a request to reject the question.
IMPLICIT_OPTION_INSTRUCTION = (
    "\n\n\u9644\u52a0\u89c4\u5219\uff1a\u82e5\u95ee\u9898\u4f7f\u7528\u201c\u4e0b\u5217\u54ea\u9879\u6b63\u786e\u201d\u3001"
    "\u201c\u4ee5\u4e0b\u54ea\u4e00\u9879\u4e00\u81f4\u201d\u6216\u201c\u54ea\u4e00\u7ec4\u8868\u8ff0\u201d\u4f46\u672a\u63d0\u4f9b\u9009\u9879\uff0c"
    "\u8fd9\u662f\u8981\u6c42\u76f4\u63a5\u4ece\u8bc1\u636e\u4e2d\u7ed9\u51fa\u6700\u8d34\u5207\u7684\u5b8c\u6574\u89c4\u5b9a\u8868\u8ff0\uff1b"
    "\u4e0d\u5f97\u4ec5\u56e0\u7f3a\u5c11\u9009\u9879\u800c\u62d2\u7b54\u3002\u4ec5\u5f53\u8bc1\u636e\u4e0d\u542b\u8be5\u89c4\u5b9a\u65f6\uff0c\u624d\u56de\u7b54\u4f9d\u636e\u4e0d\u8db3\u3002"
)

_IMPLICIT_CHOICE_PATTERN = re.compile(
    r"(?:下列哪[一项项].{0,12}(?:正确|符合|一致)|以下哪[一项项].{0,12}(?:正确|符合|一致)|"
    r"哪[一项项]表述正确|哪一组表述)"
)
_MULTI_FACT_CHOICE_PATTERN = re.compile(r"(?:哪一组|两项表述|两项).{0,24}(?:两项|均属于|均符合)")

def _has_explicit_options(question: str) -> bool:
    """Return True when the caller supplied separate A--D choices."""
    return bool(
        re.search(r"(?:^|\n)\s*A[.、:：]\s*\S+", question, flags=re.IGNORECASE)
        and re.search(r"(?:^|\n)\s*B[.、:：]\s*\S+", question, flags=re.IGNORECASE)
    )


def _has_selected_option(answer: str | None) -> bool:
    """Require a visible A--D decision when choices were supplied."""
    return bool(
        re.search(
            r"(?:\u6b63\u786e\u9009\u9879|\u552f\u4e00\u6b63\u786e\u9009\u9879|\u7b54\u6848|\u9009\u62e9)\s*(?:\u4e3a|\u662f)?\s*[:\uff1a]?\s*[A-D]",
            answer or "",
            flags=re.IGNORECASE,
        )
    )


def _explicit_options(question: str) -> dict[str, str]:
    """Extract A--D choices from the evaluation-format question."""
    return {
        letter.upper(): re.sub(r"\s+", " ", text).strip()
        for letter, text in re.findall(
            r"(?ms)^\s*([A-D])[.]\s*(.*?)(?=^\s*[A-D][.]|\Z)", question
        )
        if text.strip()
    }


def _evidence_extract_fallback(
    documents: list[Document], require_two: bool, question: str = ""
) -> str:
    """Return auditable evidence when the upstream model repeatedly returns no text."""
    # With supplied choices, an evidence-only dump is a poor fallback: it
    # hides the decision the user asked for.  First try exact normalized
    # matching of an original choice against the retrieved evidence.  This is
    # deterministic and works especially well for regulatory clauses, whose
    # wording is preserved in the knowledge base.
    options = _explicit_options(question)
    if options:
        # Distractors in the supplied questions often quote a real rule from
        # another attachment.  Score choices against the attachment named in
        # the stem whenever it is available, rather than against every broad
        # retrieval result.
        title_documents = [
            (index, document)
            for index, document in enumerate(documents, start=1)
            if _title_matches(question, str(document.meta.get("source_title", "")))
        ]
        scoring_documents = title_documents or list(enumerate(documents, start=1))
        evidence = "\n".join(document.content or "" for _, document in scoring_documents)
        normalized_evidence = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", evidence).lower()
        matches: list[tuple[float, str, str, int]] = []
        for letter, option in options.items():
            normalized_option = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", option).lower()
            if len(normalized_option) >= 8 and normalized_option in normalized_evidence:
                source_index = next(
                    (
                        index
                        for index, document in scoring_documents
                        if normalized_option
                        in re.sub(
                            r"[^0-9A-Za-z\u4e00-\u9fff]", "", document.content or ""
                        ).lower()
                    ),
                    1,
                )
                matches.append((1.0, letter, f"{option} [E{source_index}]", source_index))
                continue

            # A multi-fact choice can span two retrieved chunks, or differ
            # slightly because of PDF line breaks/OCR. Score every semicolon-
            # separated fact against every evidence item instead of requiring
            # literal equality of the whole option.
            parts = [
                re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", part).lower()
                for part in re.split(r"[\u3002\uff1b;]", option)
            ]
            parts = [part for part in parts if len(part) >= 8]
            if not parts:
                continue
            part_scores: list[tuple[float, int]] = []
            for part in parts:
                best_score, best_index = 0.0, 1
                for index, document in scoring_documents:
                    normalized_document = re.sub(
                        r"[^0-9A-Za-z\u4e00-\u9fff]", "", document.content or ""
                    ).lower()
                    if not normalized_document:
                        continue
                    match = SequenceMatcher(None, part, normalized_document, autojunk=False).find_longest_match()
                    sequence_score = match.size / len(part)
                    if len(part) >= 12:
                        grams = {part[offset : offset + 4] for offset in range(len(part) - 3)}
                        gram_score = sum(gram in normalized_document for gram in grams) / len(grams)
                    else:
                        gram_score = sequence_score
                    score = max(sequence_score, gram_score)
                    if score > best_score:
                        best_score, best_index = score, index
                part_scores.append((best_score, best_index))
            score = sum(item[0] for item in part_scores) / len(part_scores)
            best_index = max(part_scores)[1]
            matches.append((score, letter, f"{option} [E{best_index}]", best_index))
        if matches:
            matches.sort(reverse=True)
            best_score, letter, option_with_evidence, _ = matches[0]
            second_score = matches[1][0] if len(matches) > 1 else 0.0
            # Do not fabricate a choice if no option is substantively present
            # in the retrieved evidence. A clear best match is safe to expose.
            if best_score >= 0.55 and best_score - second_score >= 0.08:
                return f"\u6b63\u786e\u9009\u9879\uff1a{letter}\n{option_with_evidence}"

    selected: list[tuple[int, str]] = []
    seen: set[str] = set()
    needed = 2 if require_two else 1
    for index, document in enumerate(documents, start=1):
        content = re.sub(r"\s+", " ", document.content or "").strip()
        if len(content) < 20 or content.startswith("\u89e3\u6790\u5931\u8d25"):
            continue
        fingerprint = re.sub(r"\W+", "", content)[:240]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        excerpt = content[:1000]
        sentence_end = max(excerpt.rfind("\u3002"), excerpt.rfind("\uff1b"), excerpt.rfind("\n"))
        if sentence_end >= 80:
            excerpt = excerpt[: sentence_end + 1]
        selected.append((index, excerpt))
        if len(selected) >= needed:
            break
    if not selected:
        return ""
    if require_two and len(selected) == 2:
        return "\n".join(
            f"{position}. {content} [E{index}]"
            for position, (index, content) in enumerate(selected, 1)
        )
    return f"{selected[0][1]} [E{selected[0][0]}]"


def _is_substantive_evidence(document: Document) -> bool:
    """Reject headings, parse-error placeholders, and near-empty snippets."""
    content = document.content.strip()
    if document.meta.get("parse_error") or content.startswith("解析失败"):
        return False
    return len(content) >= 100 and len(re.findall(r"[\u4e00-\u9fff]", content)) >= 40

ANSWER_TEMPLATE = """你是银行监管知识助手。仅可依据【证据】回答。
规则：
1. 先在心中逐项核对问题要求，再一次性输出最终答案；有多个子问时必须逐项回答，不能只摘录原文或只回答其中一项。
2. 证据足以回答时，必须直接给出结论，不得仅因证据没有复述题干原话而拒答。只有确实缺少关键事实、期间或口径时，才回答“依据不足，无法确认”，并说明缺少什么。
3. 对“比较、变化、原因、是否、分别、同时”等问题，须明确写出比较对象/结论；对可由证据直接计算的简单差额，写出算式和单位。
4. 不得把证据整段照抄为答案；应将相关事实组织为简洁、完整的自然语言答案。
5. 不得编造或补全金额、比例、日期、机构名称、文号、条款内容。
6. 结论后用 [E1]、[E2] 标注支撑它的证据；不可引用未提供证据。
7. 场景判断必须区分“应当、可以、不得、原则上”等原文规范强度。
8. 答案控制在 800 个中文字符以内，但必须先完成全部结论、必要条件和引用再结束；不要输出思考过程、前言或重复证据。

【问题】{{ question }}
【证据卡片】
{{ evidence_cards }}

【回答】"""


def _evidence_cards(documents: list[Document], *, max_chars: int = 1000) -> str:
    """Make provenance-preserving, generation-friendly evidence cards.

    max_chars limits each fragment, never the complete evidence. All fragments
    retain their original citation index and provenance; no suffix is discarded.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    cards: list[str] = []
    for index, document in enumerate(documents, start=1):
        meta = document.meta
        location = " / ".join(
            str(meta[key])
            for key in ("sheet", "row", "row_start", "row_end", "cell", "page", "section")
            if meta.get(key) not in (None, "")
        )
        source = str(meta.get("source_title") or meta.get("file_name") or "未标明来源")
        content = re.sub(r"\s+", " ", document.content or "").strip()
        fragments: list[str] = []
        remaining = content
        while len(remaining) > max_chars:
            boundary = max(remaining.rfind("。", 0, max_chars), remaining.rfind("；", 0, max_chars))
            end = boundary + 1 if boundary >= max_chars // 2 else max_chars
            fragments.append(remaining[:end])
            remaining = remaining[end:]
        fragments.append(remaining)
        label = f"[E{index}] 来源：{source}"
        if location:
            label += f"；位置：{location}"
        for part, fragment in enumerate(fragments, start=1):
            fragment_label = label
            if len(fragments) > 1:
                fragment_label += f"；片段：{part}/{len(fragments)}"
            cards.append(f"{fragment_label}\n可用事实：{fragment}")
    return "\n\n".join(cards)


@component
class LocalDocumentEmbedder:
    """Haystack 3.0-compatible wrapper around a local Sentence Transformer."""

    def __init__(self, model: str, meta_fields_to_embed: list[str] | None = None):
        self.model_name = model
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.model: Any = None

    def warm_up(self) -> None:
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        self.warm_up()
        texts = []
        for document in documents:
            metadata = " ".join(str(document.meta.get(key, "")) for key in self.meta_fields_to_embed)
            texts.append(f"{metadata}\n{document.content}".strip())
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embedded_documents = [
            replace(document, embedding=vector.tolist())
            for document, vector in zip(documents, vectors)
        ]
        return {"documents": embedded_documents}


@component
class LocalTextEmbedder:
    """Query-side counterpart of LocalDocumentEmbedder."""

    def __init__(self, model: str):
        self.model_name = model
        self.model: Any = None

    def warm_up(self) -> None:
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict:
        self.warm_up()
        return {"embedding": self.model.encode(text, normalize_embeddings=True).tolist()}


@component
class LocalCrossEncoderRanker:
    """Cross-encoder reranker that runs after hybrid candidate fusion."""

    def __init__(self, model: str, top_k: int = 6):
        self.model_name = model
        self.top_k = top_k
        self.model: Any = None

    def warm_up(self) -> None:
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]) -> dict:
        self.warm_up()
        # The title and section make short regulatory clauses distinguishable
        # to the reranker without changing the stored evidence text.
        def rank_text(document: Document) -> str:
            context = " ".join(
                str(document.meta.get(field, ""))
                for field in ("source_title", "section", "sheet")
            )
            return f"{context}\n{document.content}".strip()

        scores = self.model.predict([(query, rank_text(document)) for document in documents])
        # Do not overwrite BM25, embedding or RRF scores on the candidate
        # objects. Stage-level diagnostics need to preserve each ranker's own
        # score after the full pipeline has completed.
        scored_documents = [
            replace(document, score=float(score))
            for document, score in zip(documents, scores)
        ]
        return {"documents": sorted(scored_documents, key=lambda item: item.score or 0, reverse=True)[: self.top_k]}


def load_documents(knowledge: Path) -> list[Document]:
    documents: list[Document] = []
    for line in knowledge.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if not item["meta"].get("parse_error"):
            documents.append(Document(content=item["content"], meta=item["meta"]))
    return documents


def _question_title(question: str) -> str | None:
    """Extract the attachment title used by the competition QA format."""
    # Unicode escapes keep this source portable when edited from a Windows
    # terminal whose active code page is not UTF-8.
    titles = _question_titles(question)
    return titles[0] if titles else None


def _question_titles(question: str) -> list[str]:
    """Return every attachment title named in a multi-file question."""
    return list(
        dict.fromkeys(
            title.strip()
            for title in re.findall(r"\u300a\s*([^\u300b]+?)\s*\u300b", question)
            if title.strip()
        )
    )


def _title_matches(question_title: str, source_title: str) -> bool:
    """Match attachment titles despite numbering, spaces, and bracket variants."""
    def normalize(value: str) -> str:
        value = value.replace("\uff08", "(").replace("\uff09", ")")
        value = re.sub(r"^\d{1,4}[_-]", "", value)
        value = re.sub(r"(?:pdf|docx?|xlsx?|xls|\u9644\u4ef6|\u901a\u77e5|\u7248)", "", value, flags=re.IGNORECASE)
        return re.sub(r"[\s_\-()\uff08\uff09]", "", value)

    title_key = normalize(question_title)
    source_key = normalize(source_title)
    if title_key in source_key or source_key in title_key:
        return True
    # QA stems sometimes omit the generic “表” before a period marker.
    shortened_title = title_key.replace("\u8868", "")
    shortened_source = source_key.replace("\u8868", "")
    return len(shortened_title) >= 8 and (
        shortened_title in shortened_source or shortened_source in shortened_title
    )


def _is_spreadsheet_question(question: str) -> bool:
    """Only spreadsheet questions may bypass global hybrid retrieval by title."""
    return bool(re.search(r"(?:excel|\u5de5\u4f5c\u8868)", question, flags=re.IGNORECASE))


def _question_anchors(question: str) -> list[str]:
    """Return meaningful quoted labels, including the parts of slash-separated headers."""
    anchors: list[str] = []
    for value in re.findall(r"[\u201c\"]\s*([^\u201d\"]+?)\s*[\u201d\"]", question):
        for part in value.split("/"):
            part = part.strip()
            if len(part) >= 2 and part not in anchors:
                anchors.append(part)
    return anchors


def _compact_label(value: str) -> str:
    return re.sub(r"^(?:其中：|其中:)", "", re.sub(r"\s+", "", value)).strip()


def _question_focus_terms(question: str) -> list[str]:
    """Extract discriminative CJK phrases from a document-scoped question."""
    question = re.sub(r"\u300a[^\u300b]+\u300b", "", question)
    question = re.sub(
        r"(?:根据|关于|下列|以下|哪一项|哪一组|两项|表述|属于|材料|内容|规定|请|问题|正确|均|是否|的)",
        " ",
        question,
    )
    terms: set[str] = set()
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        # Long n-grams retain the object/action relation (for example,
        # “注册资本最低限额”), while shorter ones tolerate wording variation.
        upper = min(8, len(phrase))
        for size in range(upper, 1, -1):
            for start in range(0, len(phrase) - size + 1):
                terms.add(phrase[start : start + size])
    return sorted(terms, key=len, reverse=True)[:32]


def _prompt_document_with_context(document: Document) -> Document:
    """Add stable provenance context to generation only, preserving stored evidence."""
    title = str(document.meta.get("source_title", "")).strip()
    section = str(document.meta.get("section", "")).strip()
    prefix_parts = []
    if title:
        prefix_parts.append(f"文件：{title}")
    if section and section not in {"正文", title}:
        prefix_parts.append(f"章节：{section}")
    if not prefix_parts:
        return document
    return replace(document, content=f"{'；'.join(prefix_parts)}\n{document.content}")


def _row_cells(document: Document) -> dict[str, str]:
    """Read ``B5=value`` pairs from one spreadsheet row evidence record."""
    return {
        column: value.strip()
        for column, value in re.findall(r"([A-Z]+)\d+=([^\uFF1B;\n]+)", document.content)
    }


def _as_number(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else None


def _canonical_area(value: str) -> str:
    """Match spreadsheet labels such as ``浙 江`` against ``浙江省``."""
    value = re.sub(r"(?:合计|总计)$", "", _compact_label(value))
    return area_match_key(value)


def _canonical_metric(value: str) -> str:
    """Keep literal table semantics while tolerating an ``收入`` suffix."""
    value = _compact_label(value)
    value = _compact_label(canonical_financial_term(value))
    value = re.sub(r"[（(]\s*%\s*[)）]$", "", value)
    return re.sub(r"(?:收入|金额|数)$", "", value)


def _format_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _is_table_metric_label(value: str) -> bool:
    """Exclude labels from date/header rows when comparing table indicators."""
    label = _compact_label(value)
    if len(label) < 2:
        return False
    return not bool(re.search(r"(?:20\d{2}|季度|月份|时间|项目|单位|同比|环比)", label))


def _column_number(column: str) -> int:
    """Convert an Excel column label to a sortable ordinal."""
    number = 0
    for character in column:
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def _row_number(document: Document) -> int:
    """Return the original worksheet row for stable table ordering."""
    row = document.meta.get("row")
    if isinstance(row, int):
        return row
    match = re.search(r"\u884c：(\d+)", document.content)
    return int(match.group(1)) if match else 0


def _first_value_column(parsed: list[tuple[Document, dict[str, str]]]) -> str | None:
    """Choose the leftmost data column, never a label column.

    Competition questions sometimes describe a coordinate as ``年-季度`` even
    though the worksheet stores the actual headers as ``一季度`` ... ``四季度``.
    The first data column is the only faithful interpretation in that case.
    """
    # ``_as_number`` deliberately accepts embedded numbers for natural-language
    # answers, but table coordinates must reject title cells such as
    # ``2023年银行业...`` in column A.  Otherwise A is selected before the real
    # first value column B and deterministic lookup silently falls through.
    numeric_columns = {
        column
        for _document, cells in parsed
        for column, value in cells.items()
        if re.fullmatch(r"\s*-?\d+(?:\.\d+)?%?\s*", value.replace(",", ""))
    }
    return min(numeric_columns, key=_column_number) if numeric_columns else None


def _resolve_table_column(
    label: str, headers: dict[str, str], parsed: list[tuple[Document, dict[str, str]]]
) -> str | None:
    """Resolve a literal header or a flattened period coordinate."""
    column = headers.get(_compact_label(label))
    if column:
        return column
    # ``年-季度`` is a QA coordinate, not a literal Excel cell value.  Do not
    # accidentally use a later supplementary numeric series (for example G:J
    # growth rates); select the leftmost principal value series.
    if re.search(r"(?:\u5e74.?\u5b63\u5ea6|\u5b63\u5ea6)", _compact_label(label)):
        return _first_value_column(parsed)
    return None


def _period_block_rows(parsed: list[tuple[Document, dict[str, str]]], period_label: str) -> list[tuple[Document, dict[str, str]]]:
    """Keep one quarter block for matrix-style tables.

    In a matrix such as ``机构 × 指标`` the first column contains a quarter only
    on the first row of each block.  A question using the generic QA coordinate
    ``年-季度`` refers to that first block, rather than mixing cumulative values
    from all four quarterly blocks.
    """
    if not re.search(r"(?:\u5e74.?\u5b63\u5ea6|\u5b63\u5ea6)", _compact_label(period_label)):
        return parsed
    starts = [
        index
        for index, (_document, cells) in enumerate(parsed)
        if any(re.fullmatch(r"[一二三四1-4]\u5b63\u5ea6", _compact_label(value)) for value in cells.values())
    ]
    if len(starts) < 2:
        return parsed
    return parsed[starts[0] : starts[1]]


def _metric_label_for_column(cells: dict[str, str], value_column: str) -> str | None:
    """Find the nearest non-numeric label to the left of a value cell."""
    value_position = _column_number(value_column)
    labels = [
        (column, value)
        for column, value in cells.items()
        if _column_number(column) < value_position
        and _as_number(value) is None
        and _is_table_metric_label(value)
    ]
    if not labels:
        return None
    # B (指标) should beat A (季度) in an institution-by-metric matrix.
    return max(labels, key=lambda item: _column_number(item[0]))[1].strip()


def _fallback_period_values(cells: dict[str, str], use_last_series: bool) -> tuple[float, float] | None:
    """Resolve flattened multi-row spreadsheet headers without guessing labels."""
    numeric_columns = sorted(
        (column for column, value in cells.items() if _as_number(value) is not None),
        key=_column_number,
    )
    if len(numeric_columns) < 2:
        return None
    series: list[list[str]] = []
    for column in numeric_columns:
        if not series or _column_number(column) != _column_number(series[-1][-1]) + 1:
            series.append([column])
        else:
            series[-1].append(column)
    first_series = series[0]
    end_series = series[-1] if use_last_series else first_series
    start = _as_number(cells[first_series[0]])
    end = _as_number(cells[end_series[-1]])
    if start is None or end is None:
        return None
    return start, end


class TrustedRAG:
    def __init__(self, knowledge: Path, embedding_model: str = EMBEDDING_MODEL, reranker_model: str = RERANKER_MODEL):
        self._initial_global_cache = InitialGlobalCache()
        self.store = InMemoryDocumentStore()
        self.last_retrieval_debug: dict[str, Any] = {}
        self.retrieval_mode = os.getenv("RAG_RETRIEVAL_MODE", "bm25").lower()
        documents = load_documents(knowledge)
        # Table questions name the original attachment explicitly.  Keep this
        # lightweight source index so BM25 does not lose a data row to generic
        # table headers or identically named indicators from another month.
        self.documents_by_title: dict[str, list[Document]] = {}
        for document in documents:
            title = str(document.meta.get("source_title", ""))
            self.documents_by_title.setdefault(title, []).append(document)
        from .source_summaries import load_file_summaries
        summary_path = knowledge.with_name("file_summaries.jsonl")
        use_file_summaries = os.getenv("RAG_USE_FILE_SUMMARIES", "1").strip().lower() not in {"0", "false", "no"}
        detailed_summaries = load_file_summaries(summary_path) if use_file_summaries else []
        self._source_summaries = detailed_summaries or self._build_source_summaries()
        if not documents:
            raise ValueError("知识库为空；请先运行 index")
        if self.retrieval_mode not in {"bm25", "hybrid"}:
            raise ValueError("RAG_RETRIEVAL_MODE 仅支持 bm25 或 hybrid")
        if self.retrieval_mode == "bm25":
            self.store.write_documents(documents)
            self.retrieval = Pipeline()
            self.retrieval.add_component("bm25", InMemoryBM25Retriever(self.store, top_k=20))
            return
        # Index one compact catalog record per table source.  Indexing all
        # table cells would overwhelm narrative retrieval (there are far more
        # cells than clauses), while excluding tables entirely prevents BM25
        # and embedding retrieval from recovering a missed source route.
        table_source_catalog: list[Document] = []
        for source_title, source_documents in self.documents_by_title.items():
            table_documents = [
                document for document in source_documents
                if str(document.meta.get("evidence_type", "")).startswith("table_")
            ]
            if not table_documents:
                continue
            sheets = sorted({str(document.meta.get("sheet", "")) for document in table_documents if document.meta.get("sheet")})
            header_samples = [
                re.sub(r"\s+", " ", document.content or "").strip()[:500]
                for document in table_documents
                if document.meta.get("evidence_type") in {"table_row", "table_row_group"}
            ][:3]
            table_source_catalog.append(Document(
                content=(
                    f"表格来源：{source_title}\n工作表：{'；'.join(sheets)}\n"
                    + "\n".join(header_samples)
                ).strip(),
                meta={
                    "source_title": source_title,
                    "source_path": table_documents[0].meta.get("source_path", ""),
                    "evidence_type": "table_source_catalog",
                    "sheet": "；".join(sheets),
                },
            ))
        self.table_source_catalog = table_source_catalog
        # Opt-in controlled integration of the previously frozen structural
        # representation. Catalog blocks remain discovery hints, never facts.
        if os.getenv("RAG_MULTIVECTOR_CATALOG", "0").strip().lower() in {"1", "true", "yes"}:
            from bank_rag.catalog_blocks import build_blocks
            from sentence_transformers import SentenceTransformer
            catalog_model = SentenceTransformer(embedding_model)
            blocks = []
            for catalog in table_source_catalog:
                for block in build_blocks(catalog_model, catalog, self.documents_by_title[catalog.meta["source_title"]]):
                    # Preserve the established catalog interface used by
                    # promotion, protection and evidence eligibility checks.
                    meta = dict(block.meta, evidence_type="table_source_catalog",
                                catalog_representation="structural_block")
                    blocks.append(Document(content=block.content, meta=meta))
            table_source_catalog = blocks
            self.table_source_catalog = blocks
        # LlamaIndex-style linked nodes: search representations reference
        # unchanged original evidence IDs. References are not embedding text.
        if os.getenv("RAG_CATALOG_ROW_LINK", "1").strip().lower() in {"1", "true", "yes"}:
            from bank_rag.catalog_row_link import CatalogRowResolver
            self._catalog_row_resolver = CatalogRowResolver(self.documents_by_title)
            self._catalog_row_resolver.bind_catalogs(table_source_catalog)
        # Keep individual table rows/cells out of the global semantic store;
        # the compact source records above make the original files retrievable,
        # after which title-aware retrieval selects auditable rows and cells.
        semantic_documents = [
            document
            for document in documents
            if not str(document.meta.get("evidence_type", "")).startswith("table_")
        ]
        semantic_documents.extend(table_source_catalog)
        if not semantic_documents:
            raise ValueError("No narrative documents are available for hybrid retrieval")
        index = Pipeline()
        index.add_component("embedder", LocalDocumentEmbedder(model=embedding_model, meta_fields_to_embed=["source_title", "sheet", "cell", "section"]))
        index.add_component("writer", DocumentWriter(self.store))
        index.connect("embedder.documents", "writer.documents")
        index.run({"embedder": {"documents": semantic_documents}})

        self.retrieval = Pipeline()
        self.retrieval.add_component("bm25", InMemoryBM25Retriever(self.store, top_k=20))
        self.retrieval.add_component("query_embedder", LocalTextEmbedder(model=embedding_model))
        self.retrieval.add_component("dense", InMemoryEmbeddingRetriever(self.store, top_k=20))
        self.retrieval.add_component("fuse", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=20))
        self.retrieval.add_component("rerank", LocalCrossEncoderRanker(model=reranker_model, top_k=6))
        self.retrieval.connect("query_embedder.embedding", "dense.query_embedding")
        self.retrieval.connect("bm25.documents", "fuse.documents")
        self.retrieval.connect("dense.documents", "fuse.documents")
        self.retrieval.connect("fuse.documents", "rerank.documents")

    @staticmethod
    def _merge_retrieval_candidates(*candidate_lists: list[Document]) -> list[Document]:
        """Keep original-query ranking first while adding rewrite-only recalls."""
        merged: list[Document] = []
        seen: set[str] = set()
        for candidates in candidate_lists:
            for document in candidates:
                document_id = str(document.id)
                if document_id not in seen:
                    seen.add(document_id)
                    merged.append(document)
        return merged

    def _build_source_summaries(self) -> list[dict[str, str]]:
        """Build bounded, non-generative source sketches for question planning."""
        summaries: list[dict[str, str]] = []
        for source_title, source_documents in self.documents_by_title.items():
            sheets = list(dict.fromkeys(
                str(document.meta.get("sheet", "")).strip()
                for document in source_documents if document.meta.get("sheet")
            ))[:4]
            evidence_types = {
                str(document.meta.get("evidence_type", "")) for document in source_documents
            }
            source_type = "统计表" if any(value.startswith("table_") for value in evidence_types) else "监管文件"
            samples: list[str] = []
            for document in source_documents:
                text = re.sub(r"\s+", " ", document.content or "").strip()
                if text and not text.startswith("解析失败"):
                    samples.append(text[:240])
                if len(samples) == 2:
                    break
            summaries.append({
                "source_title": source_title,
                "source_type": source_type,
                "sheets": "；".join(sheets),
                "summary": "；".join(samples),
            })
        return summaries

    def file_summaries_for_question(self, question: str, limit: int = 40) -> list[dict[str, Any]]:
        """Select concise source sketches with deterministic lexical scoring."""
        compact_question = _compact_label(question)
        focus_terms = [term for term in _question_focus_terms(question) if len(term) >= 2]
        date_terms = re.findall(r"20\d{2}年(?:[1-9]|1[0-2])月|20\d{2}年(?:第?[一二三四1-4]季度)?", question)
        ranked: list[tuple[int, dict[str, Any]]] = []
        for item in self._source_summaries:
            searchable = _compact_label(json.dumps(item, ensure_ascii=False, default=str))
            score = sum(len(term) ** 2 for term in focus_terms if _compact_label(term) in searchable)
            score += sum(80 for term in date_terms if term and term in searchable)
            title_key = _compact_label(item["source_title"])
            if title_key and title_key in compact_question:
                score += 1000
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["source_title"]))
        selected = [item for score, item in ranked if score > 0][:limit]
        if len(selected) < min(12, limit):
            selected_titles = {item["source_title"] for item in selected}
            selected.extend(
                item for _, item in ranked
                if item["source_title"] not in selected_titles
            )
        return selected[:limit]

    def _answer_file_catalog_question(self, question: str) -> dict[str, Any] | None:
        """Handle corpus-level file inventory questions from the summary catalog."""
        if not re.search(r"(?:知识库|文件).*(?:时间最早|最早).*(?:文件)?|(?:时间最早|最早)的文件", question):
            return None
        from .source_summaries import earliest_file_summaries

        matches = earliest_file_summaries(self._source_summaries)
        if not matches:
            # Catalog queries can use immutable source titles even when semantic
            # file summaries are deliberately disabled for an evaluation.
            title_rows = []
            for source_title, source_docs in self.documents_by_title.items():
                title_years = re.findall(r"(?:19|20)\d{2}", source_title)
                if title_years:
                    title_rows.append({
                        "source_title": source_title,
                        "source_path": source_docs[0].meta.get("source_path", "") if source_docs else "",
                        "primary_year": int(title_years[0]),
                    })
            matches = earliest_file_summaries(title_rows)
        if not matches:
            return None
        citations: list[dict[str, Any]] = []
        for item in matches:
            source_title = str(item.get("source_title", ""))
            source_docs = self.documents_by_title.get(source_title, [])
            if source_docs:
                citations.extend(self.citations(source_docs[:1]))
            else:
                citations.append({
                    "source_title": source_title,
                    "source_path": item.get("source_path", ""),
                    "evidence_type": "file_summary_catalog",
                })
        year = matches[0]["primary_year"]
        titles = "；".join(str(item.get("source_title", "")) for item in matches)
        return {
            "question": question,
            "answer": f"知识库中标题时间最早的是{year}年，共{len(matches)}个文件：{titles}。",
            "citations": citations,
            "verified": True,
            "insufficient_evidence": False,
            "answer_method": "deterministic_file_catalog",
            "source_routing": [{"source_title": item.get("source_title", ""), "reason": "标题年份为知识库最早年份"} for item in matches],
            "retrieval_debug": {"catalog_matches": [item.get("source_title", "") for item in matches]},
            "graph_trace": ["file_catalog_query", "finalize"],
            "orchestrator": "catalog",
        }

    def understand_question(self, question: str) -> dict[str, Any]:
        """Ask DeepSeek for a retrieval plan; never use it as a fact answer."""
        allowed_types = {
            "table_lookup", "multi_condition_lookup", "comparison", "metric_definition",
            "regulation_lookup", "unsupported_or_unclear",
        }
        summaries = self.file_summaries_for_question(question)
        fallback = {
            "task_type": "unsupported_or_unclear",
            "conditions": {"time": [], "region": [], "institution": [], "metric": [], "other": []},
            "source_routing": [], "evidence_slots": [],
            "method": "deterministic_fallback", "file_summaries_used": len(summaries),
        }
        if not os.getenv("DEEPSEEK_API_KEY"):
            return fallback
        file_summaries = json.dumps(summaries, ensure_ascii=False)
        prompt = f'''你是一个银行监管与统计问答系统中的“问题理解与证据规划模块”。

你的任务不是回答问题，也不是生成事实答案。
你的任务是：
1. 判断用户问题属于哪一类任务；
2. 提取问题中的时间、地区、机构、指标等条件；
3. 根据提供的文件梗概，判断哪些类型的文件更可能包含所需证据；
4. 将问题拆解成若干个必须分别找到的证据项；
5. 为每个证据项生成适合检索的关键词或正式表达。

【用户问题】
{question}

【当前知识库文件梗概】
{file_summaries}

【任务类型】
仅允许从以下类型中选择：
- table_lookup：单个或少量统计数值查询
- multi_condition_lookup：多个时间/地区/指标组合查询
- comparison：多期、多地区或多指标比较
- metric_definition：指标定义、计算公式、口径说明
- regulation_lookup：监管制度、规定、要求、适用范围
- unsupported_or_unclear：知识库可能不足，或问题缺少必要条件

【输出要求】
请严格输出 JSON：
{{
  "task_type": "",
  "conditions": {{"time": [], "region": [], "institution": [], "metric": [], "other": []}},
  "source_routing": [{{"source_type": "", "reason": ""}}],
  "evidence_slots": [{{"id": "", "required_evidence": "", "retrieval_terms": []}}]
}}

【重要规则】
- 不要回答用户问题。
- 不要生成具体数值、公式或监管结论。
- 不要因为文件梗概中出现某个文件名，就断定答案一定在该文件中。
- source_routing 只能表示“优先检索方向”，不能排除其他候选文件。
- 如果问题包含多个时间、地区或指标，应拆成多个 evidence_slots。
- 如果用户使用口语、简称或非正式表达，可以在 retrieval_terms 中补充更正式的同义表达。
- 不要改变原问题中的时间、地区、指标和比较关系。
- 如果某个条件无法确定，应保留为空或标记 unclear，不要自行猜测。'''
        generator = OpenAIChatGenerator(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=Secret.from_env_var("DEEPSEEK_API_KEY"),
            api_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            generation_kwargs={"temperature": 0.0, "max_tokens": 1600},
        )
        try:
            raw = generator.run(messages=[ChatMessage.from_user(prompt)])["replies"][0].text.strip()
            raw_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            parsed = json.loads(raw_json)
            if not isinstance(parsed, dict) or parsed.get("task_type") not in allowed_types:
                raise ValueError("invalid task_type")
            conditions = parsed.get("conditions") if isinstance(parsed.get("conditions"), dict) else {}
            clean_conditions = {
                key: [str(value).strip() for value in conditions.get(key, []) if str(value).strip()][:12]
                if isinstance(conditions.get(key, []), list) else []
                for key in ("time", "region", "institution", "metric", "other")
            }
            source_routing = []
            for item in parsed.get("source_routing", []) if isinstance(parsed.get("source_routing"), list) else []:
                if isinstance(item, dict) and str(item.get("source_type", "")).strip():
                    source_routing.append({
                        "source_type": str(item.get("source_type", "")).strip()[:160],
                        "reason": str(item.get("reason", "")).strip()[:300],
                    })
            evidence_slots = []
            for index, item in enumerate(parsed.get("evidence_slots", []) if isinstance(parsed.get("evidence_slots"), list) else []):
                if not isinstance(item, dict) or not str(item.get("required_evidence", "")).strip():
                    continue
                terms = item.get("retrieval_terms", [])
                clean_terms = list(dict.fromkeys(
                    str(term).strip()[:160] for term in terms
                    if isinstance(term, str) and term.strip()
                ))[:5] if isinstance(terms, list) else []
                evidence_slots.append({
                    "id": str(item.get("id") or f"E{index + 1}")[:40],
                    "required_evidence": str(item["required_evidence"]).strip()[:300],
                    "retrieval_terms": clean_terms,
                })
                if len(evidence_slots) == 24:
                    break
            return {
                "task_type": parsed["task_type"], "conditions": clean_conditions,
                "source_routing": source_routing[:8], "evidence_slots": evidence_slots,
                "method": "deepseek_question_understanding", "file_summaries_used": len(summaries),
            }
        except Exception as exc:
            return {**fallback, "method": "deterministic_fallback_after_llm_error", "error": type(exc).__name__}

    @staticmethod
    def _debug_documents(documents: list[Document], top_k: int) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for rank, document in enumerate(documents[:top_k], 1):
            meta = document.meta
            values.append({
                "evidence_id": evidence_id(document),
                "rank": rank,
                "score": float(document.score) if document.score is not None else None,
                "content": re.sub(r"\s+", " ", document.content or "").strip()[:600],
                "source_title": meta.get("source_title") or meta.get("file_name") or "",
                "source_path": meta.get("source_path") or "",
                "evidence_type": meta.get("evidence_type") or "",
                "page": meta.get("page"),
                "sheet": meta.get("sheet"),
                "row": meta.get("row"),
                "cell": meta.get("cell"),
                "section": meta.get("section"),
                **({"row_keys": meta.get("row_keys", []),
                    "table": meta.get("table"),
                    "catalog_representation": meta.get("catalog_representation")}
                   if meta.get("catalog_representation") == "structural_block" else {}),
            })
        return values

    @staticmethod
    def _merge_debug_documents(*groups: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for group in groups:
            for item in group:
                key = (
                    item.get("source_title"), item.get("evidence_type"), item.get("page"),
                    item.get("sheet"), item.get("row"), item.get("cell"), item.get("section"),
                    item.get("content"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(dict(item))
                if len(merged) >= top_k:
                    break
            if len(merged) >= top_k:
                break
        for rank, item in enumerate(merged, 1):
            item["rank"] = rank
        return merged

    def _hybrid_query(self, query: str) -> tuple[list[Document], dict[str, list[dict[str, Any]]]]:
        output = self.retrieval.run(
            {"bm25": {"query": query}, "query_embedder": {"text": query}, "rerank": {"query": query}},
            include_outputs_from={"bm25", "dense", "fuse", "rerank"},
        )
        stages = {
            "bm25_top_k": self._debug_documents(output.get("bm25", {}).get("documents", []), 20),
            "embedding_top_k": self._debug_documents(output.get("dense", {}).get("documents", []), 20),
            "rrf_top_k": self._debug_documents(output.get("fuse", {}).get("documents", []), 20),
            "cross_encoder_top_k": self._debug_documents(output["rerank"]["documents"], 6),
        }
        documents = output["rerank"]["documents"]
        if os.getenv("RAG_CATALOG_ROW_LINK", "1").strip().lower() in {"1", "true", "yes"}:
            from bank_rag.catalog_row_link import CatalogRowResolver
            if not hasattr(self, "_catalog_row_resolver"):
                self._catalog_row_resolver = CatalogRowResolver(self.documents_by_title)
            documents, trace = self._catalog_row_resolver.expand(documents, query)
            stages["catalog_row_link"] = trace
        return documents, stages

    def promote_table_catalogs(
        self, question: str, plan: QuestionPlan, retrieval_debug: dict[str, Any],
        *, top_n: int = 6,
    ) -> list[dict[str, Any]]:
        """Route only reranked table catalogs with two independent matches."""
        promoted: list[dict[str, Any]] = []
        for candidate in (retrieval_debug.get("cross_encoder_top_k") or [])[:top_n]:
            if candidate.get("evidence_type") != "table_source_catalog":
                continue
            title = str(candidate.get("source_title") or "")
            records = self.documents_by_title.get(title, [])
            internal = [record for record in records if record.meta.get("evidence_type") != "table_source_catalog"]
            if not internal:
                continue
            catalog_text = f"{title} {candidate.get('content') or ''}"
            matches: list[str] = []
            if plan.periods and any(
                period in catalog_text or (re.match(r"20\d{2}", period) and period[:4] in title)
                for period in plan.periods
            ):
                matches.append("time")
            if (plan.entity and plan.entity in catalog_text) or (
                re.search(r"(?:哪类银行|各类银行|分机构类|银行类型|机构类别)", question)
                and ("分机构类" in catalog_text or "机构；" in catalog_text)
            ):
                matches.append("institution")
            if plan.metric and any(
                variant in catalog_text or any(variant in record.content for record in internal)
                for variant in metric_variants(plan.metric)
            ):
                matches.append("metric")
            if len(matches) < 2:
                continue
            source_id = re.match(r"^(\d{1,4})_", title)
            promoted.append({
                "source_id": source_id.group(1) if source_id else str(internal[0].meta.get("doc_id") or ""),
                "source_title": title,
                "score": candidate.get("score"),
                "reranker_rank": candidate.get("rank"),
                "matched_conditions": matches,
                "reasons": ["Cross-Encoder目录卡升级", "匹配条件：" + "、".join(matches)],
            })
        return promoted[:3]

    def retrieve(
        self, question: str, plan: QuestionPlan | None = None,
        routed_sources: list[dict[str, Any]] | None = None,
        queries: list[str] | None = None,
        evidence_slots: list[dict[str, Any]] | None = None,
    ) -> list[Document]:
        # The official evaluation supplies A--D choices.  Comparing choices
        # reliably sometimes needs a second clause from the same attachment;
        # use a wider evidence set only for that offline/evaluation format.
        top_k = 10 if _has_explicit_options(question) else 6
        plan = plan or parse_question(question)
        evidence_slots = evidence_slots or []
        routed_slot_candidates: list[Document] = []
        if evidence_slots and routed_sources:
            routed_titles = [str(item.get("source_title") or "") for item in routed_sources]
            routed_slot_candidates = [
                document
                for source_title, source_documents in self.documents_by_title.items()
                if any(title and _title_matches(title, source_title) for title in routed_titles)
                for document in source_documents
            ]
        # Retrieval starts with exactly the user's wording.  Supplemental
        # queries are supplied explicitly by the evidence-gated LangGraph only
        # after it has identified which required evidence is missing.
        queries = list(dict.fromkeys(queries or [question]))
        reuse_initial_global = self._initial_global_cache.begin_retrieve(question, routed_sources, queries)
        self.last_retrieval_debug = {
            "mode": self.retrieval_mode,
            "queries": queries,
            "bm25_top_k": [],
            "embedding_top_k": [],
            "rrf_top_k": [],
            "cross_encoder_top_k": [],
        }
        if self.retrieval_mode == "bm25":
            bm25_groups = [
                self.retrieval.run({"bm25": {"query": query}})["bm25"]["documents"]
                for query in queries
            ]
            bm25_documents = self._merge_retrieval_candidates(*bm25_groups)
            self.last_retrieval_debug["bm25_top_k"] = self._debug_documents(bm25_documents, 20)
            selected = self._title_aware_bm25(
                question, bm25_documents, top_k=top_k, plan=plan,
                routed_sources=routed_sources, retrieval_queries=queries,
            )
            return self._preserve_required_slots(
                selected, self._merge_retrieval_candidates(routed_slot_candidates, bm25_documents),
                evidence_slots, top_k,
            )
        # Exact attachment selection is essential for spreadsheets.  Applying
        # it to every regulation title, however, can turn a harmless title
        # variation such as “(PDF)” into an empty retrieval result.
        if (
            (plan.intent == "table_calculation" and routed_sources)
            or (_is_spreadsheet_question(question) and _question_title(question))
        ):
            documents = self._title_aware_bm25(
                question, [], top_k=top_k, plan=plan,
                routed_sources=routed_sources, retrieval_queries=queries,
            )
            self.last_retrieval_debug["mode"] = "structured_source_route"
            self.last_retrieval_debug["structured_route_top_k"] = self._debug_documents(documents, top_k)
            return self._preserve_required_slots(
                documents, self._merge_retrieval_candidates(routed_slot_candidates, documents),
                evidence_slots, top_k,
            )
        query_outputs = [self._initial_global_cache.call(
            query, self._hybrid_query, allow_reuse=reuse_initial_global,
        ) for query in queries]
        global_documents = self._merge_retrieval_candidates(*(item[0] for item in query_outputs))
        if any(item[1].get("catalog_row_link") for item in query_outputs):
            self.last_retrieval_debug["catalog_row_link"] = [
                trace for item in query_outputs for trace in item[1].get("catalog_row_link", [])
            ]
        for stage, limit in (
            ("bm25_top_k", 20), ("embedding_top_k", 20),
            ("rrf_top_k", 20), ("cross_encoder_top_k", 6),
        ):
            self.last_retrieval_debug[stage] = self._merge_debug_documents(
                *(item[1][stage] for item in query_outputs), top_k=limit
            )
        # Preserve high-confidence reranked narrative evidence. Source routing
        # and table-specific selection may add better records, but must not
        # silently discard a strong clause already found by Cross-Encoder.
        protected = [
            document for document in global_documents[:3]
            if str(document.meta.get("evidence_type", "")) != "table_source_catalog"
            and float(document.score or 0.0) >= 0.80
        ]
        self.last_retrieval_debug["protected_evidence"] = self._debug_documents(protected, len(protected))
        selected = self._title_aware_bm25(
            question, global_documents, top_k=top_k, plan=plan,
            routed_sources=routed_sources, retrieval_queries=queries,
        )
        candidates = self._merge_retrieval_candidates(protected, selected, global_documents)
        return self._preserve_selected_evidence(
            candidates, self._merge_retrieval_candidates(routed_slot_candidates, global_documents),
            evidence_slots, top_k, protected, selected,
        )

    def _preserve_selected_evidence(
        self,
        ranked_documents: list[Document],
        additional_candidates: list[Document],
        evidence_slots: list[dict[str, Any]],
        top_k: int,
        ce_protected: list[Document],
        local_selected: list[Document],
    ) -> list[Document]:
        """Keep completed local selection alongside existing CE/slot protection.

        Slot matching and choice remain owned by the existing slot selector.
        Expand only to accommodate these already selected prerequisites; other
        global candidates fill the original budget rather than enlarging it.
        """
        baseline = self._preserve_required_slots(
            ranked_documents, additional_candidates, evidence_slots, top_k,
        )
        slot_ids = list(dict.fromkeys(
            slot["document_id"]
            for slot in self.last_retrieval_debug.get(
                "slot_aware_evidence_preservation", {},
            ).get("slots", [])
            if slot.get("verified")
        ))
        pool = self._merge_retrieval_candidates(ranked_documents, additional_candidates)
        lookup = {str(document.id): document for document in pool}
        representatives = [lookup[identity] for identity in slot_ids]
        # A confident discovery block/explicit table reference has its own
        # locally ranked window. Do not let another routed source consume the
        # whole context before these unchanged original rows can be inspected.
        # This is bounded candidate preservation, not a new required slot or a
        # claim that every linked row answers the question. Parent CE scores are
        # never assigned to rows. Reuse the existing 3-parent/0.80 confidence
        # boundary and the caller's existing local top-k.
        parents = {
            str(item.get('evidence_id')): item
            for item in self.last_retrieval_debug.get('cross_encoder_top_k', [])[:3]
            if float(item.get('score') or 0) >= 0.80
        }
        linked_window, linked_groups, seen_linked_parents = [], [], set()
        for group in self.last_retrieval_debug.get('catalog_row_link', []):
            parent = parents.get(str(group.get('catalog_id')))
            if parent is None or group['catalog_id'] in seen_linked_parents:
                continue
            records = [
                lookup[identity] for identity in group.get('resolved_ids', [])[:top_k]
                if identity in lookup
                and lookup[identity].meta.get('source_title') == parent.get('source_title')
                and lookup[identity].meta.get('source_path') == parent.get('source_path')
            ]
            if not records:
                continue
            seen_linked_parents.add(group['catalog_id'])
            linked_window.extend(records)
            linked_groups.append({
                'parent_id': group['catalog_id'], 'parent_score': parent['score'],
                'resolution_method': group.get('resolution_method'),
                'window_ids': [str(d.id) for d in records],
            })
        # Overview/multi-metric retrieval may expand its returned list beyond
        # the caller's existing top-k. Protect that window, not every expansion;
        # verified hard slots above remain independently protected in full.
        local_window = local_selected[:top_k]
        seeds = self._merge_retrieval_candidates(ce_protected, local_window)
        required = self._merge_retrieval_candidates(representatives, seeds, linked_window)
        limit = max(top_k, len(required))
        selected = self._merge_retrieval_candidates(required, ranked_documents)[:limit]
        self.last_retrieval_debug["selected_evidence_preservation"] = {
            "baseline_ids": [str(document.id) for document in baseline],
            "selected_local_ids": [str(document.id) for document in local_window],
            "local_returned_count": len(local_selected),
            "ce_protected_ids": [str(document.id) for document in ce_protected],
            "required_slot_ids": slot_ids,
            "linked_candidate_windows": linked_groups,
            "linked_preserved_ids": [str(d.id) for d in self._merge_retrieval_candidates(linked_window)],
            "final_ids": [str(document.id) for document in selected],
            "final_limit": limit,
        }
        return selected

    @staticmethod
    def _document_matches_slot(document: Document, slot: dict[str, Any]) -> bool:
        """Deterministically verify that one evidence record satisfies a slot."""
        if slot.get('row_value') and slot.get('column'):
            return selection_slot_matches(document, slot)
        meta = document.meta or {}
        if str(meta.get("evidence_type", "")) == "table_source_catalog":
            return False
        content = str(document.content or "")
        if len(re.sub(r"\s+", "", content)) < 8 or content.startswith("解析失败"):
            return False
        title = str(meta.get("source_title", ""))
        compact_content = re.sub(r"\s+", "", content)
        compact_text = re.sub(r"\s+", "", f"{title} {content}")

        period = str(slot.get("period") or "")
        if period and re.sub(r"\s+", "", period) not in compact_text:
            match = re.match(r"(20\d{2})年(.+)", period)
            if not match:
                return False
            year, tail = match.groups()
            tail_variants = {tail, re.sub(r"^第", "", tail)}
            if year not in re.sub(r"\s+", "", title) or not any(
                value and re.sub(r"\s+", "", value) in compact_content
                for value in tail_variants
            ):
                return False

        area = area_match_key(str(slot.get("area") or ""))
        if area and not area_present_in_text(str(slot.get("area") or ""), f"{title} {content}"):
            return False
        entity = str(slot.get("entity") or "")
        entity_variants = {
            entity,
            entity.replace("财产保险公司", "财产险公司"),
            entity.replace("财产险公司", "财产保险公司"),
        }
        if entity and not any(value and value in compact_text for value in entity_variants):
            return False
        metric = str(slot.get("metric") or "")
        if metric and not any(
            re.sub(r"\s+", "", variant) in compact_content
            for variant in metric_variants(metric)
        ):
            return False
        return True

    @staticmethod
    def _slot_match_quality(document: Document, slot: dict[str, Any]) -> int:
        """Prefer an exact structured row over notes that mention a metric."""
        meta = document.meta or {}
        evidence_type = str(meta.get("evidence_type", ""))
        row_label = re.sub(
            r"^\s*\d+[、.]\s*", "", re.sub(r"\s+", "", str(meta.get("row_label", "")))
        )
        quality = 2 if evidence_type == "table_evidence_group" else 1 if evidence_type == "table_row" else 0
        metric = str(slot.get("metric") or "")
        if metric and any(row_label == re.sub(r"\s+", "", variant) for variant in metric_variants(metric)):
            quality += 20
        area = area_match_key(str(slot.get("area") or ""))
        if area and area_match_key(row_label) == area:
            quality += 20
        entity = re.sub(r"\s+", "", str(slot.get("entity") or ""))
        if entity and entity in row_label:
            quality += 10
        return quality

    def _preserve_required_slots(
        self,
        ranked_documents: list[Document],
        additional_candidates: list[Document],
        evidence_slots: list[dict[str, Any]],
        top_k: int,
    ) -> list[Document]:
        """Protect one verified best evidence group per hard Required Slot.

        Slot representatives are selected before the remaining globally ranked
        documents.  When required slots exceed ``top_k``, the evidence window
        expands instead of silently dropping a prerequisite.
        """
        candidates = self._merge_retrieval_candidates(ranked_documents, additional_candidates)
        if not evidence_slots:
            return candidates[:top_k]

        protected: list[Document] = []
        protected_keys: set[str] = set()
        slot_debug: list[dict[str, Any]] = []
        for slot in evidence_slots:
            matches = [document for document in candidates if self._document_matches_slot(document, slot)]
            if not matches:
                slot_debug.append({
                    "slot_id": slot.get("id"),
                    "required_evidence": slot.get("required_evidence"),
                    "verified": False,
                })
                continue
            rank = {str(document.id): index for index, document in enumerate(candidates)}
            best = max(
                matches,
                key=lambda document: (
                    self._slot_match_quality(document, slot),
                    float(document.score or 0.0),
                    -rank[str(document.id)],
                ),
            )
            key = str(best.id)
            if key not in protected_keys:
                protected_keys.add(key)
                protected.append(best)
            slot_debug.append({
                "slot_id": slot.get("id"),
                "required_evidence": slot.get("required_evidence"),
                "verified": True,
                "source_title": best.meta.get("source_title"),
                "evidence_type": best.meta.get("evidence_type"),
                "score": best.score,
                "document_id": key,
            })

        limit = max(top_k, len(protected))
        selected = self._merge_retrieval_candidates(protected, candidates)[:limit]
        self.last_retrieval_debug["slot_aware_evidence_preservation"] = {
            "required_slot_count": len(evidence_slots),
            "protected_document_count": len(protected),
            "final_limit": limit,
            "slots": slot_debug,
        }
        return selected

    def rewrite_missing_evidence(
        self,
        question: str,
        plan: QuestionPlan,
        missing_items: list[dict[str, Any]],
        existing_context: str,
    ) -> dict[str, Any]:
        """Use one constrained LLM call to rewrite only missing evidence.

        The model may propose retrieval phrases; it cannot alter the parsed
        conditions or answer the question.  If no API key is configured, the
        missing descriptions themselves remain usable deterministic queries.
        """
        descriptions = [str(item.get("description") or item.get("value") or "").strip() for item in missing_items]
        descriptions = [item for item in descriptions if item]
        fallback = list(dict.fromkeys(descriptions))[:5]
        if len(fallback) == 1:
            fallback.extend([
                f"{fallback[0]} 监管统计表",
                f"{fallback[0]} 表头 行 单元格",
            ])
        if not descriptions:
            return {"queries": [], "method": "none", "raw_output": ""}
        if not os.getenv("DEEPSEEK_API_KEY"):
            return {"queries": fallback, "method": "deterministic_missing_items", "raw_output": ""}

        prompt = (
            "你是银行监管资料检索改写器。将用户表达改写为更可能出现在监管文件、统计表标题、"
            "表头或正文中的正式检索表达。只改写当前缺失证据对应的部分，不改变原问题的时间、"
            "地区、指标和任务含义。输出2-5个候选检索短语，每行一个；不回答问题，不解释，"
            "不补充未知事实，不输出编号或代码块。\n\n"
            f"用户原问题：\n{question}\n\n"
            "缺失证据：\n- " + "\n- ".join(descriptions) + "\n\n"
            f"已有上下文：\n{existing_context or '无'}"
        )
        generator = OpenAIChatGenerator(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=Secret.from_env_var("DEEPSEEK_API_KEY"),
            api_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            generation_kwargs={"temperature": 0.0, "max_tokens": 768},
        )
        try:
            raw = generator.run(messages=[ChatMessage.from_user(prompt)])["replies"][0].text.strip()
        except Exception as exc:
            return {
                "queries": fallback,
                "method": "deterministic_fallback_after_llm_error",
                "raw_output": "",
                "error": type(exc).__name__,
            }
        candidates: list[str] = []
        for line in raw.splitlines():
            candidate = re.sub(r"^\s*(?:[-*]|\d+[.、)]|候选\s*\d*[:：])\s*", "", line).strip()
            candidate = candidate.strip("`'\"")
            if not candidate or len(candidate) > 160 or candidate in candidates:
                continue
            if re.search(r"(?:答案|因此|所以|结论)[：:]", candidate):
                continue
            candidates.append(candidate)
            if len(candidates) == 5:
                break
        if len(candidates) < 2:
            candidates = list(dict.fromkeys([*candidates, *fallback]))[:5]
        return {"queries": candidates, "method": "deepseek_missing_evidence_rewrite", "raw_output": raw}

    def _route_sources(self, question: str, plan: QuestionPlan, limit: int | None = None) -> list[dict[str, Any]]:
        """Route a natural question to high-confidence source files."""
        explicit_titles = plan.required_source_titles or _question_titles(question)
        route_limit = limit if limit is not None else max(3, len(plan.source_titles), len(plan.periods))
        if explicit_titles:
            routed: list[dict[str, Any]] = []
            for requested in explicit_titles:
                matches = [title for title in self.documents_by_title if _title_matches(requested, title)]
                for title in matches[:1]:
                    routed.append({"source_title": title, "score": 1000, "reasons": ["题面/解析计划明确指定来源"]})
            if routed:
                return routed[:route_limit]

        # Hints supply formal report wording to the existing scorer. They do
        # not short-circuit routing or exclude other candidate sources.
        routing_question = question
        if plan.source_title_provenance == "deterministic_inferred":
            routing_question += " " + " ".join(plan.source_titles)
        question_key = _compact_label(routing_question)
        terms = [term for term in _question_focus_terms(routing_question) if len(term) >= 4]
        date_terms = re.findall(r"20\d{2}年(?:[1-9]|1[0-2])月|20\d{2}年(?:[一二三四1-4]|第[一二三四1-4])季度|20\d{2}年", routing_question)
        candidates: list[dict[str, Any]] = []
        for source_title, documents in self.documents_by_title.items():
            title_key = _compact_label(source_title)
            reasons: list[str] = []
            score = 0
            title_without_id = re.sub(r"^\d{1,4}_", "", source_title)
            topic_parts = [part for part in title_without_id.split("_") if len(_compact_label(part)) >= 6]
            matching_parts = [part for part in topic_parts if _compact_label(part) in question_key]
            if matching_parts:
                best = max(matching_parts, key=lambda part: len(_compact_label(part)))
                score += 700 + len(_compact_label(best))
                reasons.append(f"命中明确主题“{best}”")
            matching_dates = [term for term in date_terms if term in source_title]
            if matching_dates:
                score += 90 * len(matching_dates)
                reasons.append("命中统计期间：" + "、".join(matching_dates))
            matched_terms = [term for term in terms if term in title_key]
            if matched_terms:
                longest = max(matched_terms, key=len)
                score += len(longest) * len(longest) + sum(len(term) for term in matched_terms[:3])
                reasons.append(f"标题关键词命中“{longest}”")
            sheets = {str(document.meta.get("sheet", "")) for document in documents}
            if plan.metric and any(_compact_label(plan.metric) in _compact_label(sheet) for sheet in sheets):
                score += 36
                reasons.append("工作表名称匹配指标")
            if plan.metric and any(
                any(variant in document.content for variant in metric_variants(plan.metric))
                for document in documents
            ):
                score += 80
                reasons.append("文件内容包含目标指标")
            if score >= 100 or any(reason.startswith("命中明确主题") for reason in reasons):
                candidates.append({"source_title": source_title, "score": score, "reasons": reasons})
        candidates.sort(key=lambda item: (int(item["score"]), str(item["source_title"])), reverse=True)
        return candidates[:route_limit]

    def _implicit_source_titles(self, question: str, plan: QuestionPlan) -> list[str]:
        return [str(item["source_title"]) for item in self._route_sources(question, plan)]

    def _title_aware_bm25(
        self, question: str, bm25_documents: list[Document], top_k: int = 6,
        plan: QuestionPlan | None = None, routed_sources: list[dict[str, Any]] | None = None,
        retrieval_queries: list[str] | None = None,
    ) -> list[Document]:
        """Prefer rows from the attachment cited in a table question.

        This is retrieval-only grounding: no QA answer or expected value is
        consulted.  It makes month-specific spreadsheet questions deterministic
        and still falls back to ordinary BM25 for regulation/document questions.
        """
        plan = plan or parse_question(question)
        routed_sources = routed_sources if routed_sources is not None else self._route_sources(question, plan)
        titles = [str(item["source_title"]) for item in routed_sources]
        if not titles:
            return bm25_documents[:6]

        source_groups: list[list[Document]] = []
        for title in titles:
            group = [
                document
                for source_title, documents in self.documents_by_title.items()
                if _title_matches(title, source_title)
                for document in documents
            ]
            if group:
                # In repeated quarterly blocks, a metric label occurs once
                # per quarter. Bind every record to its nearest preceding
                # quarter marker instead of ranking same-named rows together.
                if len(plan.periods) == 1:
                    quarter = re.search(r"第?([一二三四1-4])季度", plan.periods[0])
                    if quarter:
                        markers = sorted(
                            (
                                int(document.meta["row"]), match.group(1)
                            )
                            for document in group
                            if document.meta.get("evidence_type") == "table_row"
                            and document.meta.get("row") is not None
                            and (match := re.search(r"A\d+=([一二三四1-4])季度(?:[；;\n]|$)", document.content))
                        )
                        if len(markers) >= 2 and any(value == quarter.group(1) for _, value in markers):
                            target_start = next(row for row, value in markers if value == quarter.group(1))
                            target_end = next((row for row, _ in markers if row > target_start), 10**9)
                            scoped = [
                                document for document in group
                                if document.meta.get("row") is not None
                                and target_start <= int(document.meta["row"]) < target_end
                            ]
                            if scoped:
                                group = scoped
                source_groups.append(group)
        if not source_groups:
            return bm25_documents[:6]

        anchors = _question_anchors(question)
        query_texts = list(dict.fromkeys(retrieval_queries or [question]))
        focus_terms = list(dict.fromkeys(
            term for query in query_texts for term in _question_focus_terms(query)
        ))

        def deterministic_variants(term: str) -> list[str]:
            """Return conservative forms that may occur in a spreadsheet.

            Administrative suffixes and whitespace vary across source tables.
            These are spelling variants only; cross-level mappings such as
            Guangzhou -> Guangdong must never satisfy evidence requirements.
            """
            value = str(term or "").strip()
            if not value:
                return []
            variants = [value]
            compact = re.sub(r"\s+", "", value)
            if compact and compact != value:
                variants.append(compact)
            without_suffix = re.sub(r"(?:特别行政区|自治区|省|市)$", "", compact)
            if len(without_suffix) >= 2:
                variants.append(without_suffix)
            if compact.endswith("市") and len(without_suffix) >= 2:
                variants.append(without_suffix + "市")
            elif compact.endswith("省") and len(without_suffix) >= 2:
                variants.append(without_suffix + "省")
            return list(dict.fromkeys(item for item in variants if item))

        constraint_terms: list[str] = []
        if plan.table_selection:
            constraint_terms.extend(plan.table_selection['row_values'])
            constraint_terms.extend(plan.table_selection['columns'])
        if plan.entity:
            constraint_terms.extend(deterministic_variants(plan.entity))
        for area in [*plan.areas, *([plan.area] if plan.area else [])]:
            constraint_terms.extend(deterministic_variants(area))
        if plan.metric:
            for metric in metric_variants(plan.metric):
                constraint_terms.extend(deterministic_variants(metric))
        for period in plan.periods:
            constraint_terms.extend(deterministic_variants(period))
            match = re.match(r"(20\d{2})年第?([一二三四1-4])季度", period)
            if match:
                constraint_terms.extend([f"{match.group(1)}年{match.group(2)}季度", f"{match.group(2)}季度"])
        # On the second pass these terms include the LLM's missing-evidence
        # rewrites.  They affect only retrieval ranking and never alter the
        # parsed time, area, metric, or answer task.
        for query in query_texts[1:]:
            constraint_terms.extend(_question_focus_terms(query))
        constraint_terms = list(dict.fromkeys(constraint_terms))
        if not hasattr(self, "last_retrieval_debug"):
            self.last_retrieval_debug = {}
        self.last_retrieval_debug["file_internal_query_terms"] = constraint_terms
        self.last_retrieval_debug["file_internal_query_method"] = (
            "deterministic_aliases_plus_missing_evidence_rewrite"
            if len(query_texts) > 1 else "deterministic_aliases"
        )

        def score(document: Document) -> tuple[int, int, int]:
            content = document.content
            compact_content = re.sub(r"\s+", "", content)
            anchor_score = sum(200 for anchor in anchors if re.sub(r"\s+", "", anchor) in compact_content)
            constraint_score = sum(
                300 for term in constraint_terms
                if re.sub(r"\s+", "", term) in compact_content
            )
            # Preserve row-level records: they usually contain both indicator
            # and value, whereas individual cell documents contain only one.
            evidence_type = str(document.meta.get("evidence_type", ""))
            structure_score = 50 if evidence_type in {
                "row", "table_row", "row_group", "table_evidence_group"
            } else 0
            row_label = re.sub(r"\s+", "", str(document.meta.get("row_label", "")))
            requested_areas = plan.areas or ([plan.area] if plan.area else [])
            for requested_area in requested_areas:
                compact_area = re.sub(r"(?:省|市|自治区|特别行政区)$", "", requested_area)
                if compact_area and (
                    row_label == compact_area
                    or (compact_area == "全国" and row_label in {"全国", "全国合计"})
                ):
                    structure_score += 1000
            # Apply the same question focus signal to clauses and table rows.
            # Definitions and notes in tables need object/action matching too;
            # their structure, anchors and parsed constraints still contribute.
            focus_score = 0
            for term in focus_terms:
                if re.sub(r"\s+", "", term) in compact_content:
                    focus_score += min(len(term) * len(term), 64)
            # For a regulation title query, the former ordering accidentally
            # preferred very short section headings (because -len was sorted in
            # reverse). Prefer substantive clauses, while capping the length so
            # one unusually large document cannot occupy every result.
            substantive_score = min(len(content), 1400) // 20
            if len(content.strip()) < 80:
                substantive_score -= 100
            return (anchor_score + constraint_score + focus_score + structure_score + substantive_score, constraint_score + focus_score, structure_score, len(content))

        # Controlled migration: the existing scorer remains the default until
        # full-chain checks confirm ranking gains without answer regressions.
        import os
        use_lexical = os.getenv("RAG_SOURCE_LOCAL_LEXICAL", "0").strip().lower() in {"1", "true", "yes"}
        ordered_groups = []
        local_rankings = []
        for group in source_groups:
            if use_lexical and any(str(d.meta.get("evidence_type", "")).startswith("table_") for d in group):
                from bank_rag.source_local_lexical import rank_source_local
                ranked, detail = rank_source_local(group, question, score, plan.areas or ([plan.area] if plan.area else []))
                ordered_groups.append(ranked)
                local_rankings.append(detail)
            else:
                ordered_groups.append(sorted(group, key=score, reverse=True))
        if use_lexical:
            self.last_retrieval_debug["source_local_lexical"] = local_rankings
        overview_metrics = insurance_operating_overview_metrics(question)
        overview_evidence: list[Document] = []
        if overview_metrics and len(source_groups) > 1:
            # Keep one auditable evidence group for every requested metric and
            # period. Otherwise repeated notes or early rows can fill top-k.
            for group in source_groups:
                group_records = [
                    document for document in group
                    if str(document.meta.get("evidence_type", "")) == "table_evidence_group"
                ]
                for metric in overview_metrics:
                    matches = [
                        document for document in group_records
                        if re.sub(r"^\s*\d+[、.]\s*", "", str(document.meta.get("row_label", "")).strip()) == metric
                    ]
                    if matches:
                        # 财产险/人身险 appear below both income and claims;
                        # the overview asks for the first pair under income.
                        matches.sort(key=lambda document: int(document.meta.get("row") or 10**9))
                        overview_evidence.append(matches[0])
            top_k = max(top_k, len(source_groups) * len(overview_metrics))
            self.last_retrieval_debug["file_internal_evidence_slots"] = [
                {
                    "source_title": document.meta.get("source_title"),
                    "row": document.meta.get("row"),
                    "row_label": document.meta.get("row_label"),
                }
                for document in overview_evidence
            ]
        requested_metrics = list(dict.fromkeys(
            str(item.get("canonical") or "") for item in plan.financial_aliases
            if item.get("canonical") and str(item.get("canonical")) not in {
                "合计", "比较", "增加额", "减少额", "增长率", "百分点变化", "差额",
                "平均值", "最大值", "最小值", "求和", "财产保险", "人身保险",
            }
        ))
        multi_metric_evidence: list[Document] = []
        if len(requested_metrics) > 1:
            for group in source_groups:
                records = [document for document in group if document.meta.get("evidence_type") == "table_evidence_group"]
                for metric in requested_metrics:
                    matches = [
                        document for document in records
                        if any(variant == re.sub(r"^\s*\d+[、.]\s*", "", str(document.meta.get("row_label", "")).strip()) for variant in metric_variants(metric))
                    ]
                    if matches:
                        multi_metric_evidence.append(max(matches, key=score))
            if multi_metric_evidence:
                top_k = max(top_k, len(multi_metric_evidence))
                self.last_retrieval_debug["file_internal_multi_metric_slots"] = self._debug_documents(
                    multi_metric_evidence, len(multi_metric_evidence)
                )
        # A derived calculation over several regions needs one raw row for
        # every source-period × region combination. Global ranking alone often
        # kept both regions from the newer month but only one from the older
        # month, even though all four rows existed in the selected files.
        multi_area_evidence: list[Document] = []
        requested_areas = list(dict.fromkeys(plan.areas or ([plan.area] if plan.area else [])))
        if len(requested_areas) > 1 and plan.metric:
            for group in source_groups:
                records = [
                    document for document in group
                    if document.meta.get("evidence_type") == "table_evidence_group"
                ]
                for requested_area in requested_areas:
                    compact_area = re.sub(r"(?:特别行政区|自治区|省|市)$", "", requested_area)
                    matches = [
                        document for document in records
                        if re.sub(r"(?:特别行政区|自治区|省|市)$", "", re.sub(
                            r"\s+", "", re.sub(
                                r"^\s*\d+[、.]\s*", "", str(document.meta.get("row_label", "")).strip()
                            )
                        )) == compact_area
                    ]
                    if matches:
                        multi_area_evidence.append(max(matches, key=score))
            if multi_area_evidence:
                top_k = max(top_k, len(multi_area_evidence))
                self.last_retrieval_debug["file_internal_multi_area_slots"] = self._debug_documents(
                    multi_area_evidence, len(multi_area_evidence)
                )
        # Multi-file questions must retain evidence from every explicitly named
        # attachment. Seed one highest-scoring row from each file before filling
        # the rest of the evidence window.
        explicit_rows = []
        if use_lexical and os.getenv("RAG_CATALOG_ROW_LINK", "1").strip().lower() in {"1", "true", "yes"}:
            from bank_rag.explicit_row_selection import explicit_row_candidates
            explicit_rows, explicit_audit = explicit_row_candidates(ordered_groups, question, plan.periods)
            self.last_retrieval_debug["explicit_row_selection"] = explicit_audit
        ordered = [
            *overview_evidence,
            *multi_metric_evidence,
            *multi_area_evidence,
            *explicit_rows,
            *[group[0] for group in ordered_groups if group],
        ]
        ordered.extend(document for group in ordered_groups for document in group[1:])
        combined: list[Document] = []
        seen: set[str] = set()
        for document in [*ordered, *bm25_documents]:
            # Haystack IDs are stable per content/metadata pair; guard against
            # a source-selected row occurring again in the BM25 result.
            document_id = str(document.id)
            if document_id not in seen:
                seen.add(document_id)
                combined.append(document)
            if len(combined) == top_k:
                break
        return combined

    def _source_documents_for_question(self, question: str) -> list[Document]:
        titles = _question_titles(question)
        if not titles:
            return []
        return [
            document
            for source_title, documents in self.documents_by_title.items()
            if any(_title_matches(title, source_title) for title in titles)
            for document in documents
        ]

    def _resolve_regional_table_value(
        self, requested_title: str, area: str, requested_column: str, metric: str
    ) -> tuple[float, str, str, list[Document]] | None:
        """Resolve one area × column value from the best matching attachment."""
        candidates: list[tuple[int, float, str, str, list[Document]]] = []
        for source_title, documents in self.documents_by_title.items():
            if not _title_matches(requested_title, source_title):
                continue
            parsed = sorted(
                ((document, _row_cells(document)) for document in documents if document.meta.get("evidence_type") == "table_row"),
                key=lambda item: _row_number(item[0]),
            )
            header_document: Document | None = None
            column: str | None = None
            for document, cells in parsed:
                for candidate_column, label in cells.items():
                    if _canonical_metric(label) == _canonical_metric(requested_column):
                        header_document, column = document, candidate_column
                        break
                if column:
                    break
            if not column or not header_document:
                continue
            for document, cells in parsed:
                labels = [label for label in cells.values() if _as_number(label) is None]
                if not any(_canonical_area(label) == _canonical_area(area) for label in labels):
                    continue
                value = _as_number(cells.get(column, ""))
                if value is None:
                    continue
                unit_match = next(
                    (
                        re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", item.content)
                        for item, _item_cells in parsed
                        if re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", item.content)
                    ),
                    None,
                )
                source_score = 1000 if _compact_label(metric) in _compact_label(source_title) else 0
                candidates.append((source_score, value, unit_match.group(1).strip() if unit_match else "", source_title, [header_document, document]))
                break
        if not candidates:
            return None
        _score, value, unit, source_title, evidence = max(candidates, key=lambda item: item[0])
        return value, unit, source_title, evidence

    def _execute_explicit_table_selection(self, plan: QuestionPlan) -> tuple[str, list[Document]] | None:
        """Sum explicit columns of one verified source row; never use its total column."""
        selection = plan.table_selection
        if (selection.get('operation') != 'sum' or len(plan.periods) != 1
                or len(selection.get('row_values', [])) != 1 or len(plan.source_titles) != 1):
            return None
        row = selection['row_values'][0]
        matches = []
        for title, documents in self.documents_by_title.items():
            if not _title_matches(plan.source_titles[0], title):
                continue
            for document in documents:
                facts = [bound_fact(document, row, col) for col in selection['columns']]
                if all(facts) and len({fact['unit'] for fact in facts}) == 1:
                    matches.append((document, facts))
        # Multiple table/sheet/source coordinates cannot be arbitrarily chosen.
        identities = {(doc.meta.get('source_title'), doc.meta.get('sheet'), doc.meta.get('row')) for doc, _ in matches}
        if len(identities) != 1 or not matches:
            return None
        document, facts = matches[0]
        signatures = {tuple((f['cell'], str(f['value']), f['unit']) for f in fs) for _, fs in matches}
        if len(signatures) != 1:
            return None
        total = sum(fact['value'] for fact in facts)
        unit = facts[0]['unit']
        details = '，'.join(f"{f['column']} {_format_number(f['value'])}{unit}（{f['cell']}）" for f in facts)
        expression = ' + '.join(_format_number(f['value']) for f in facts)
        return f"{plan.periods[0]}，{row}：{details}；{expression} = {_format_number(total)}{unit}。[E1]", [document]

    def _execute_table_program(self, plan: QuestionPlan) -> tuple[str, list[Document]] | None:
        """Execute the compiled sum/difference/rate/mean program for regional tables."""
        if plan.table_selection:
            return self._execute_explicit_table_selection(plan)
        program = plan.table_program or {}
        selector = program.get("selector", {})
        operations = list(program.get("final_operations", []))
        if (
            selector.get("dimension") != "area"
            or not selector.get("members")
            or not operations
        ):
            return None
        # Two or more regions means "which region is higher" and is handled
        # by the regional ranking executor.  One region across two periods is
        # instead a period comparison and must stay in this executor so it can
        # report both source values and their difference.
        if (
            "rank" in operations
            and "sequential_difference" not in operations
            and len(selector.get("members", [])) >= 2
        ):
            return self._execute_ranked_regional_program(plan)
        requested_column = "合计" if selector.get("column_policy") == "explicit_total" else str(selector.get("metric", ""))
        per_source: list[tuple[str, list[tuple[str, float]], float, str]] = []
        evidence: list[Document] = []
        for requested_title in selector.get("sources", []):
            values: list[tuple[str, float]] = []
            source_label = requested_title
            unit = ""
            for area in selector["members"]:
                resolved = self._resolve_regional_table_value(
                    requested_title, area, requested_column, str(selector.get("metric", ""))
                )
                if not resolved:
                    return None
                value, found_unit, source_label, source_evidence = resolved
                values.append((area, value))
                unit = unit or found_unit
                evidence.extend(source_evidence)
            aggregate = sum(value for _area, value in values) if program.get("per_source_aggregate") == "sum" else values[0][1]
            per_source.append((source_label, values, aggregate, unit))

        unit = next((item[3] for item in per_source if item[3]), "")
        suffix = f" {unit}" if unit else ""
        trace = []
        for source_label, values, aggregate, _found_unit in per_source:
            raw_values = " + ".join(f"{area}({_format_number(value)})" for area, value in values)
            if program.get("per_source_aggregate") == "sum":
                trace.append(f"{source_label}：{raw_values} = {_format_number(aggregate)}{suffix}。")
            else:
                trace.append(f"{source_label}：{raw_values}{suffix}。")

        aggregates = [item[2] for item in per_source]
        final_lines: list[str] = []
        if "sequential_difference" in operations:
            if len(aggregates) < 3:
                return None
            changes = [
                aggregates[index] - aggregates[index - 1]
                for index in range(1, len(aggregates))
            ]
            for index, value in enumerate(changes, start=1):
                final_lines.append(
                    f"第{index}个连续期间增加额={_format_number(aggregates[index])}-{_format_number(aggregates[index - 1])}={_format_number(value)}{suffix}。"
                )
            if "difference_of_differences" in operations and len(changes) >= 2:
                gap = changes[1] - changes[0]
                direction = "多出" if gap >= 0 else "少于"
                final_lines.append(
                    f"第二个增加额较第一个{direction}{_format_number(abs(gap))}{suffix}。"
                )
            if "max" in operations:
                index, value = max(enumerate(changes, start=1), key=lambda item: item[1])
                final_lines.append(f"最大增加额为第{index}个相邻期间的{_format_number(value)}{suffix}。")
            if "min" in operations:
                index, value = min(enumerate(changes, start=1), key=lambda item: item[1])
                final_lines.append(f"最小增加额为第{index}个相邻期间的{_format_number(value)}{suffix}。")
            return "\n".join([*trace, *final_lines]), self._merge_retrieval_candidates(evidence)
        if "mean" in operations:
            final_lines.append(f"算术平均值=({_format_number(aggregates[0])}{''.join('+' + _format_number(value) for value in aggregates[1:])})/{len(aggregates)}={_format_number(sum(aggregates) / len(aggregates))}{suffix}。")
        if "max" in operations:
            index, value = max(enumerate(aggregates), key=lambda item: item[1])
            final_lines.append(f"最大值={_format_number(value)}{suffix}，来自{per_source[index][0]}。")
        if "min" in operations:
            index, value = min(enumerate(aggregates), key=lambda item: item[1])
            final_lines.append(f"最小值={_format_number(value)}{suffix}，来自{per_source[index][0]}。")
        if any(operation in operations for operation in ("difference", "reverse_difference", "growth_rate", "percentage_point_difference")) and len(aggregates) != 2:
            return None
        if len(aggregates) == 2:
            current, previous = aggregates
            difference = current - previous
            if "difference" in operations:
                final_lines.append(f"增加额={_format_number(current)}-{_format_number(previous)}={_format_number(difference)}{suffix}。")
            if "reverse_difference" in operations:
                final_lines.append(f"减少额={_format_number(previous)}-{_format_number(current)}={_format_number(-difference)}{suffix}。")
            if "growth_rate" in operations:
                final_lines.append("上期值为 0，增长率无法计算。" if previous == 0 else f"增长率=({_format_number(current)}-{_format_number(previous)})/{_format_number(previous)}×100%={_format_number(difference / previous * 100)}%。")
            if "rank" in operations and len(selector.get("members", [])) == 1:
                current_label, previous_label = per_source[0][0], per_source[1][0]
                if difference > 0:
                    final_lines.append(f"{current_label}更高，比{previous_label}高{_format_number(difference)}{suffix}。")
                elif difference < 0:
                    final_lines.append(f"{previous_label}更高，比{current_label}高{_format_number(-difference)}{suffix}。")
                else:
                    final_lines.append(f"两期数值相同，均为{_format_number(current)}{suffix}。")
        return "\n".join([*trace, *final_lines]), self._merge_retrieval_candidates(evidence)

    def _execute_ranked_regional_program(self, plan: QuestionPlan) -> tuple[str, list[Document]] | None:
        """Apply a calculation to each region, then rank the calculated results."""
        program = plan.table_program or {}
        selector = program.get("selector", {})
        operations = list(program.get("final_operations", []))
        sources = list(selector.get("sources", []))
        members = list(selector.get("members", []))
        if selector.get("dimension") != "area" or len(members) < 2 or not sources:
            return None
        requested_column = "合计" if selector.get("column_policy") == "explicit_total" else str(selector.get("metric", ""))
        evidence: list[Document] = []
        results: list[tuple[str, float, str, str]] = []
        details: list[str] = []
        for area in members:
            samples: list[tuple[str, float, str]] = []
            for requested_title in sources:
                resolved = self._resolve_regional_table_value(
                    requested_title, area, requested_column, str(selector.get("metric", ""))
                )
                if not resolved:
                    return None
                value, unit, source_title, source_evidence = resolved
                samples.append((source_title, value, unit))
                evidence.extend(source_evidence)
            unit = next((sample[2] for sample in samples if sample[2]), "")
            suffix = f" {unit}" if unit else ""
            if any(operation in operations for operation in ("difference", "reverse_difference", "growth_rate", "percentage_point_difference")):
                if len(samples) != 2:
                    return None
                current, previous = samples[0][1], samples[1][1]
                if "growth_rate" in operations:
                    if previous == 0:
                        return None
                    score = (current - previous) / previous * 100
                    score_unit, label = "%", "增长率"
                    details.append(f"{area}：({_format_number(current)}-{_format_number(previous)})/{_format_number(previous)}×100%={_format_number(score)}%。")
                elif "percentage_point_difference" in operations:
                    score = current - previous
                    score_unit, label = "个百分点", "百分点变化"
                    details.append(f"{area}：{_format_number(current)}-{_format_number(previous)}={_format_number(score)}{score_unit}。")
                elif "reverse_difference" in operations:
                    score = previous - current
                    score_unit, label = unit, "减少额"
                    details.append(f"{area}：{_format_number(previous)}-{_format_number(current)}={_format_number(score)}{suffix}。")
                else:
                    score = current - previous
                    score_unit, label = unit, "增加额"
                    details.append(f"{area}：{_format_number(current)}-{_format_number(previous)}={_format_number(score)}{suffix}。")
            else:
                # Ranking a direct table value uses the first named source.
                score, score_unit, label = samples[0][1], unit, "数值"
                details.append(f"{area}：{samples[0][0]}为{_format_number(score)}{suffix}。")
            results.append((area, score, score_unit, label))

        ascending = program.get("rank_order") == "ascending" or "min" in operations
        ranked = sorted(results, key=lambda item: item[1], reverse=not ascending)
        winner, runner_up = ranked[0], ranked[1]
        gap = abs(winner[1] - runner_up[1])
        relation = "更少/更低" if ascending else "更多/更高"
        gap_suffix = "个百分点" if winner[2] == "%" and winner[3] == "增长率" else (f" {winner[2]}" if winner[2] else "")
        conclusion = f"结论：{winner[0]}的{winner[3]}{relation}，为{_format_number(winner[1])}{gap_suffix}；较{runner_up[0]}多/高{_format_number(gap)}{gap_suffix}。"
        return "\n".join([*details, conclusion]), self._merge_retrieval_candidates(evidence)

    def _planned_multi_source_table_answer(
        self, plan: QuestionPlan
    ) -> tuple[str, list[Document]] | None:
        """Execute a multi-period regional-table calculation from row evidence.

        This path is intentionally narrow: it requires two attachments named in
        the question, an area, an indicator, and a supported arithmetic request.
        Unlike a model completion it resolves the header and region row first,
        then calculates from the two retrieved values locally.
        """
        if (
            len(plan.source_titles) < 2
            or not plan.area
            or not plan.metric
            or not ({"增加额", "减少额", "增长率", "差额", "平均值"} & set(plan.operations))
        ):
            return None

        requested_column = "合计" if plan.aggregation_policy == "prefer_explicit_total" else plan.metric
        evidence: list[Document] = []
        values: list[float] = []
        source_labels: list[str] = []
        unit = ""
        for requested_title in plan.source_titles:
            candidates: list[tuple[int, str, Document, Document, float, str]] = []
            for source_title, documents in self.documents_by_title.items():
                if not _title_matches(requested_title, source_title):
                    continue
                parsed = sorted(
                    ((document, _row_cells(document)) for document in documents if document.meta.get("evidence_type") == "table_row"),
                    key=lambda item: _row_number(item[0]),
                )
                header_document: Document | None = None
                column: str | None = None
                desired_metric = _canonical_metric(requested_column)
                for document, cells in parsed:
                    for candidate_column, label in cells.items():
                        if _canonical_metric(label) == desired_metric:
                            header_document, column = document, candidate_column
                            break
                    if column:
                        break
                if not column or not header_document:
                    continue
                for document, cells in parsed:
                    labels = [label for label in cells.values() if _as_number(label) is None]
                    if any(_canonical_area(label) == _canonical_area(plan.area) for label in labels):
                        value = _as_number(cells.get(column, ""))
                        if value is not None:
                            unit_match = next(
                                (
                                    re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", item.content)
                                    for item, _item_cells in parsed
                                    if re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", item.content)
                                ),
                                None,
                            )
                            source_score = 1000 if _compact_label(plan.metric) in _compact_label(source_title) else 0
                            candidates.append((source_score, source_title, header_document, document, value, unit_match.group(1).strip() if unit_match else ""))
                        break
            if not candidates:
                return None
            _score, source_title, header_document, area_document, value, found_unit = max(candidates, key=lambda item: item[0])
            values.append(value)
            source_labels.append(source_title)
            if not unit and found_unit:
                unit = found_unit
            evidence.extend([header_document, area_document])

        unit_suffix = f" {unit}" if unit else ""
        if "平均值" in plan.operations:
            expression = "+".join(_format_number(value) for value in values)
            average = sum(values) / len(values)
            answer = (
                f"{plan.area}{requested_column}：" + "；".join(
                    f"{label}为{_format_number(value)}{unit_suffix}" for label, value in zip(source_labels, values)
                ) + f"。\n算术平均值=({expression})/{len(values)}={_format_number(average)}{unit_suffix}。"
            )
            return answer, self._merge_retrieval_candidates(evidence)

        if len(values) != 2:
            return None
        current, previous = values
        increase = current - previous
        parts = [
            f"{plan.area}{plan.metric}：{source_labels[0]}为{_format_number(current)}{unit_suffix}，"
            f"{source_labels[1]}为{_format_number(previous)}{unit_suffix}。",
        ]
        if "增加额" in plan.operations or "差额" in plan.operations:
            parts.append(f"增加额={_format_number(increase)}{unit_suffix}。")
        if "减少额" in plan.operations:
            parts.append(f"减少额={_format_number(-increase)}{unit_suffix}。")
        if "增长率" in plan.operations:
            if previous == 0:
                parts.append("上期值为 0，增长率无法计算。")
            else:
                parts.append(f"增长率=({_format_number(current)}-{_format_number(previous)})/{_format_number(previous)}×100%={_format_number(increase / previous * 100)}%。")
        return "\n".join(parts), self._merge_retrieval_candidates(evidence)

    def _planned_matrix_table_answer(
        self, plan: QuestionPlan
    ) -> tuple[str, list[Document]] | None:
        """Calculate a metric for one institution category across two tables."""
        if (
            len(plan.source_titles) != 2
            or not plan.entity
            or not plan.metric
            or len(plan.periods) != 2
            or not ({"差额", "增加额", "减少额", "百分点变化"} & set(plan.operations))
        ):
            return None

        def period_header(period: str) -> str:
            match = re.search(r"([一二三四1-4])季度", period)
            if not match:
                return _compact_label(period)
            quarter = {"1": "一", "2": "二", "3": "三", "4": "四"}.get(match.group(1), match.group(1))
            return f"{quarter}季度末"

        values: list[float] = []
        evidence: list[Document] = []
        units: list[str] = []
        for requested_title, requested_period in zip(plan.source_titles, plan.periods):
            rows = [
                document
                for source_title, documents in self.documents_by_title.items()
                if _title_matches(requested_title, source_title)
                for document in documents
                if document.meta.get("evidence_type") == "table_row"
            ]
            parsed = sorted(((document, _row_cells(document)) for document in rows), key=lambda item: _row_number(item[0]))
            wanted_header = _compact_label(period_header(requested_period))
            header_document: Document | None = None
            column: str | None = None
            for document, cells in parsed:
                for candidate_column, label in cells.items():
                    if _compact_label(label) == wanted_header:
                        header_document, column = document, candidate_column
                        break
                if column:
                    break
            if not column or not header_document:
                return None

            entity_index: int | None = None
            entity_document: Document | None = None
            value: float | None = None
            for index, (document, cells) in enumerate(parsed):
                labels = [label for label in cells.values() if _as_number(label) is None]
                if any(_compact_label(label) == _compact_label(plan.entity) for label in labels):
                    candidate = _as_number(cells.get(column, ""))
                    if candidate is not None:
                        entity_index, entity_document, value = index, document, candidate
                        break
            if entity_index is None or entity_document is None or value is None:
                return None

            # Matrix tables write the metric only at the first row of its
            # institution block.  Walk upward from “人身险公司” to confirm the
            # requested metric rather than treating a nearby core-capital row
            # as the same measure.
            metric_document: Document | None = None
            desired_metric = _canonical_metric(plan.metric)
            for document, cells in reversed(parsed[: entity_index + 1]):
                if any(_canonical_metric(label) == desired_metric for label in cells.values()):
                    metric_document = document
                    break
            if metric_document is None:
                return None

            unit_match = next(
                (
                    re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", document.content)
                    for document, _cells in parsed
                    if re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", document.content)
                ),
                None,
            )
            if unit_match:
                units.append(unit_match.group(1).strip())
            values.append(value)
            evidence.extend([header_document, metric_document, entity_document])

        current, previous = values
        percent_like = any("%" in unit for unit in units)
        display_scale = 100 if percent_like and max(abs(current), abs(previous)) <= 10 else 1
        current_display, previous_display = current * display_scale, previous * display_scale
        delta = current_display - previous_display
        unit = "%" if percent_like else ""
        direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
        answer = (
            f"{plan.entity}“{plan.metric}”：{plan.periods[0]}末为{_format_number(current_display)}{unit}，"
            f"{plan.periods[1]}末为{_format_number(previous_display)}{unit}。\n"
            f"变化={_format_number(current_display)}{unit}-{_format_number(previous_display)}{unit}="
            f"{_format_number(delta)}{unit}，即{direction}{_format_number(abs(delta))}个百分点。"
        )
        return answer, self._merge_retrieval_candidates(evidence)

    def _planned_regional_growth_comparison(
        self, plan: QuestionPlan
    ) -> tuple[str, list[Document]] | None:
        """Compare growth rates for two named regions across two monthly tables."""
        if (
            len(plan.source_titles) != 2
            or len(plan.areas) != 2
            or not plan.metric
            or "增长率" not in plan.operations
            or "比较" not in plan.operations
        ):
            return None
        requested_column = "合计" if plan.aggregation_policy == "prefer_explicit_total" else plan.metric

        def resolve(requested_title: str, area: str) -> tuple[float, str, list[Document]] | None:
            candidates: list[tuple[int, float, str, list[Document]]] = []
            for source_title, documents in self.documents_by_title.items():
                if not _title_matches(requested_title, source_title):
                    continue
                parsed = sorted(
                    ((document, _row_cells(document)) for document in documents if document.meta.get("evidence_type") == "table_row"),
                    key=lambda item: _row_number(item[0]),
                )
                header_document: Document | None = None
                column: str | None = None
                for document, cells in parsed:
                    for candidate_column, label in cells.items():
                        if _canonical_metric(label) == _canonical_metric(requested_column):
                            header_document, column = document, candidate_column
                            break
                    if column:
                        break
                if not column or not header_document:
                    continue
                for document, cells in parsed:
                    labels = [label for label in cells.values() if _as_number(label) is None]
                    if any(_canonical_area(label) == _canonical_area(area) for label in labels):
                        value = _as_number(cells.get(column, ""))
                        if value is not None:
                            unit_match = next(
                                (
                                    re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", item.content)
                                    for item, _item_cells in parsed
                                    if re.search(r"\u5355\u4f4d[:：]\s*([^\uff1b;\n]+)", item.content)
                                ),
                                None,
                            )
                            score = 1000 if _compact_label(plan.metric) in _compact_label(source_title) else 0
                            candidates.append((score, value, unit_match.group(1).strip() if unit_match else "", [header_document, document]))
                        break
            if not candidates:
                return None
            _score, value, unit, documents = max(candidates, key=lambda item: item[0])
            return value, unit, documents

        results: list[tuple[str, float, float, float, str, list[Document]]] = []
        for area in plan.areas:
            current = resolve(plan.source_titles[0], area)
            previous = resolve(plan.source_titles[1], area)
            if not current or not previous or previous[0] == 0:
                return None
            current_value, current_unit, current_docs = current
            previous_value, previous_unit, previous_docs = previous
            rate = (current_value - previous_value) / previous_value * 100
            results.append((area, current_value, previous_value, rate, current_unit or previous_unit, [*current_docs, *previous_docs]))

        higher, lower = sorted(results, key=lambda item: item[3], reverse=True)
        details = []
        evidence: list[Document] = []
        for area, current, previous, rate, unit, documents in results:
            suffix = f" {unit}" if unit else ""
            details.append(
                f"{area}：{plan.source_titles[0]}为{_format_number(current)}{suffix}，{plan.source_titles[1]}为{_format_number(previous)}{suffix}，"
                f"增长率=({_format_number(current)}-{_format_number(previous)})/{_format_number(previous)}×100%={_format_number(rate)}%。"
            )
            evidence.extend(documents)
        comparison = f"因此{higher[0]}更高（{_format_number(higher[3])}%），高于{lower[0]}的{_format_number(lower[3])}%。"
        return "\n".join([*details, comparison]), self._merge_retrieval_candidates(evidence)

    def _deterministic_table_answer(self, question: str) -> tuple[str, list[Document]] | None:
        """Keep calculator references aligned with its returned operand evidence.

        Both LangGraph and the legacy pipeline use this exit. References cover
        the complete calculation evidence, not individually inferred claims.
        """
        result = self._deterministic_table_answer_uncited(question)
        if result is None:
            return None
        answer, evidence = result
        referenced = {int(index) for index in re.findall(r"\[E(\d+)\]", answer)}
        required = set(range(1, len(evidence) + 1))
        if required and not required.issubset(referenced):
            references = "".join(f"[E{index}]" for index in range(1, len(evidence) + 1))
            answer = f"{answer}\n计算/查询依据：{references}"
        return answer, evidence

    def _deterministic_table_answer_uncited(self, question: str) -> tuple[str, list[Document]] | None:
        """Perform the two spreadsheet operations used in the provided QA set.

        The calculation is made only from indexed row evidence.  This avoids an
        LLM being asked to compare a whole column after a six-document retrieval.
        """
        plan = parse_question(question)
        program_answer = self._execute_table_program(plan)
        if program_answer:
            return program_answer
        comparison_answer = self._planned_regional_growth_comparison(plan)
        if comparison_answer:
            return comparison_answer
        matrix_answer = self._planned_matrix_table_answer(plan)
        if matrix_answer:
            return matrix_answer
        planned_answer = self._planned_multi_source_table_answer(plan)
        if planned_answer:
            return planned_answer
        source_documents = self._source_documents_for_question(question)
        rows = [d for d in source_documents if d.meta.get("evidence_type") == "table_row"]
        anchors = _question_anchors(question)
        if not rows or not anchors:
            return None

        parsed = [(document, _row_cells(document)) for document in rows]
        parsed.sort(key=lambda item: _row_number(item[0]))
        headers: dict[str, str] = {}
        for _document, cells in parsed:
            for column, value in cells.items():
                label = _compact_label(value)
                if label:
                    headers.setdefault(label, column)

        # "Indicator at coordinate X: what is its value?"  This is kept
        # deterministic for normal spreadsheet lookup, including flattened
        # coordinates such as ``年-季度`` that are represented by a quarter
        # header row in the source workbook.
        if len(anchors) == 2 and "\u6570\u503c" in question and "\u6700\u9ad8" not in question:
            target, coordinate = map(_compact_label, anchors[:2])
            column = _resolve_table_column(coordinate, headers, parsed)
            if column:
                for document, cells in parsed:
                    labels = {_compact_label(value) for value in cells.values() if _as_number(value) is None}
                    number = _as_number(cells.get(column, ""))
                    if target in labels and number is not None:
                        answer = f"{number:.8f}".rstrip("0").rstrip(".")
                        return answer, [document]

        # "A from B to C changes by how much": result is C - B.
        if len(anchors) >= 3 and "\u4ece" in question and "\u5230" in question and "\u53d8\u5316" in question:
            target, start, end = map(_compact_label, anchors[:3])
            start_column, end_column = headers.get(start), headers.get(end)
            for document, cells in parsed:
                values = {_compact_label(value): column for column, value in cells.items()}
                if target not in values:
                    continue
                start_value = _as_number(cells.get(start_column, "")) if start_column else None
                end_value = _as_number(cells.get(end_column, "")) if end_column else None
                if start_value is None or end_value is None:
                    # Some supplied worksheets use merged multi-row headers.
                    # Their QA stem may retain only “year-quarter” / “quarter”,
                    # which cannot be looked up as a literal row cell. Use the
                    # first contiguous value series (for example B:E) instead.
                    use_last_series = bool(
                        re.search(r"\u672c\u5e74\u7d2f\u8ba1|\u622a\u81f3\u5f53\u671f", end)
                    )
                    fallback_values = _fallback_period_values(cells, use_last_series)
                    if fallback_values:
                        start_value, end_value = fallback_values
                if start_value is not None and end_value is not None:
                    answer = f"{end_value - start_value:.2f}".rstrip("0").rstrip(".")
                    return answer, [document]

        # "At column X, which item is highest?"
        if "\u6700\u9ad8" in question and anchors:
            column = _resolve_table_column(_compact_label(anchors[0]), headers, parsed)
            if not column:
                return None
            candidates: list[tuple[float, str, Document]] = []
            # The generic period coordinate is normally the second quoted
            # anchor.  For matrix tables it selects the first quarter block;
            # without this boundary, a cumulative Q4 net-profit figure can be
            # compared against a Q1 loan balance.
            # If an institution-by-metric matrix omits a quarter, report the
            # first displayed period rather than silently mixing Q1 stock data
            # with cumulative Q4 profit.  Simple one-row-per-institution tables
            # have no repeated period blocks and are left unchanged.
            period_anchor = anchors[1] if len(anchors) >= 2 else "\u5b63\u5ea6"
            scope = _period_block_rows(parsed, period_anchor)
            for document, cells in scope:
                number = _as_number(cells.get(column, ""))
                if number is None:
                    continue
                label = _metric_label_for_column(cells, column)
                if label:
                    candidates.append((number, label, document))
            if candidates:
                _number, label, document = max(candidates, key=lambda item: item[0])
                return label.strip(), [document]
        return None

    @staticmethod
    def citations(documents: list[Document]) -> list[dict[str, Any]]:
        fields = ("source_title", "source_path", "sheet", "row", "row_start", "row_end", "cell", "page", "section", "evidence_type")
        return [{key: document.meta[key] for key in fields if key in document.meta} for document in documents]

    @staticmethod
    def verify_numbers(answer: str, documents: list[Document]) -> bool:
        """A conservative guardrail: every non-year numeric token must be in retrieved evidence."""
        evidence = "\n".join(document.content for document in documents)
        numbers = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:%|亿元|万元|天|年)?", answer)
        return all(token in evidence or re.fullmatch(r"20\d{2}年?", token) for token in numbers)

    def _ask_pipeline(
        self,
        question: str,
        *,
        plan: QuestionPlan | None = None,
        source_routing: list[dict[str, Any]] | None = None,
        docs: list[Document] | None = None,
        try_table: bool = True,
        generation_advisory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the existing answer logic with optionally prepared context.

        LangGraph supplies the parsed plan, routed sources and retrieved
        documents so this method does not repeat expensive retrieval.  The
        legacy orchestrator calls the same method without prepared context.
        """
        plan = plan or parse_question(question)
        if unsupported_requested_operation(plan):
            return {
                "question": question,
                "question_plan": plan.to_dict(),
                "source_routing": [],
                "citations": [],
                "verified": False,
                "insufficient_evidence": True,
                "answer": "需要澄清：该问题包含当前本地计算器尚不支持的“"
                + "、".join(plan.unsupported_operations)
                + "”运算。请明确分母、权重、样本范围或统计口径后再计算。",
            }
        # A table calculation needs a source, or sufficiently distinctive
        # natural-language context that can resolve one.  Returning an
        # unrelated retrieved paragraph is worse than a clear clarification.
        if source_routing is None:
            source_routing = self._route_sources(question, plan)
        if plan.intent == "table_calculation" and not source_routing:
            return {
                "question": question,
                "question_plan": plan.to_dict(),
                "source_routing": [],
                "citations": [],
                "verified": False,
                "insufficient_evidence": True,
                "answer": "依据不足，无法确认。该表格计算未提供可定位的统计表、具体期间或足够明确的业务范围；请补充年份/月份、机构或地区、指标及对应统计表线索。",
            }
        if docs is None:
            docs = self.retrieve(question, plan=plan, routed_sources=source_routing)
        result: dict[str, Any] = {
            "question": question,
            "question_plan": plan.to_dict(),
            "source_routing": source_routing,
            "citations": self.citations(docs),
            "verified": False,
        }
        table_result = self._deterministic_table_answer(question) if try_table else None
        if table_result:
            answer, evidence = table_result
            result["answer"] = answer
            result["evidence"] = [document.content for document in evidence]
            result["citations"] = self.citations(evidence)
            result["verified"] = True
            return result
        substantive_docs = docs
        # Matching a file title alone is not enough: the retrieved excerpts
        # must contain usable regulatory text before the model may answer.
        if _question_title(question) and not substantive_docs:
            result["answer"] = "依据不足：已检索到相关文件名，但没有可用于核验的正文条款。"
            result["insufficient_evidence"] = True
            return result
        evidence_docs = docs
        result["citations"] = self.citations(evidence_docs)
        if not os.getenv("DEEPSEEK_API_KEY"):
            result["answer"] = "未配置 DEEPSEEK_API_KEY；以下为已检索证据，请据此核验。"
            result["evidence"] = [document.content for document in docs]
            result["answer_method"] = "retrieved_evidence_only"
            return result
        generator = OpenAIChatGenerator(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=Secret.from_env_var("DEEPSEEK_API_KEY"),
            api_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            generation_kwargs={
                "temperature": 0.0,
                "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192")),
            },
        )
        # Keep citations tied to the original records, while giving the model
        # title/section context that is otherwise only stored in metadata.
        prompt_docs = [_prompt_document_with_context(document) for document in docs]
        # Keep the complete retrieved evidence list in the API response and
        # evaluation report. Presentation fragments share these original
        # citations and retain the complete content supplied to generation.
        result["citations"] = self.citations(docs)

        generation_question = question
        resolved = plan.resolved_conditions or {}
        locked_context = ""
        if resolved.get("locked"):
            locked_context = (
                "\n【不可覆盖的确定条件】\n"
                + json.dumps(resolved, ensure_ascii=False)
                + "\n以上条件由确定性解析器给出。原问题只作为语言背景；不得重新解释相对时间、地区、机构或指标。"
                "取数、计算、引用和结论必须以这些条件及证据为准。\n"
            )
        has_explicit_options = _has_explicit_options(question)
        implicit_choice_without_options = _IMPLICIT_CHOICE_PATTERN.search(question) and not has_explicit_options
        if implicit_choice_without_options:
            # The competition spreadsheet stores the question stem but omits the
            # original A/B/C/D choices. Reframe it as an extractive regulation
            # query so the model gives the supported rule instead of declining.
            answer_shape = "两条相互独立、完整且可核验的规定" if _MULTI_FACT_CHOICE_PATTERN.search(question) else "一条完整、可核验的规定"
            generation_question = (
                f"请直接陈述材料中与下述题干最相关的{answer_shape}。"
                "原题的选择项未提供，不要提及选项缺失，也不要拒答；"
                "只需给出有证据支持的规定并标注证据。\n原题：" + question
            )
        elif has_explicit_options:
            generation_question = (
                question
                + "\n\n\u8bf7\u9010\u4e00\u6838\u5bf9 A-D \u9009\u9879\u4e0e\u8bc1\u636e\uff0c\u9009\u51fa\u552f\u4e00\u6b63\u786e\u9879\u3002"
                "\u56de\u7b54\u5fc5\u987b\u5148\u5199\u9009\u9879\u5b57\u6bcd\uff0c\u518d\u5199\u51fa\u5b8c\u6574\u9009\u9879\u5185\u5bb9\u5e76\u6807\u6ce8\u8bc1\u636e\u3002"
            )
        evidence_cards = _evidence_cards(prompt_docs)
        prompt = PromptBuilder(template=ANSWER_TEMPLATE).run(
            question=generation_question,
            evidence_cards=evidence_cards,
        )["prompt"]
        prompt += locked_context
        temporal_context = "\n【时间角色，仅用于证据与回答条件判断】\n" + json.dumps(temporal_semantics(plan), ensure_ascii=False)
        prompt += temporal_context
        case_contract = needs_case_contract(question, docs)
        if case_contract:
            prompt += CASE_CONTRACT
        advisory_context = clarification_advisory(plan.to_dict(), generation_advisory)
        if advisory_context:
            prompt += advisory_context
            result["generation_advisory"] = dict(generation_advisory)
        if (
            plan.metric == "原保险保费收入"
            and re.search(r"20\d{2}年第?[一二三四1-4]季度", question)
        ):
            prompt += (
                "\n\n季度累计口径规则：月度原保险保费收入表通常为本年累计数。"
                "如果用户询问季度总额但未明确说“单季新增”或“季度内新增”，应列出该季度各月累计值，"
                "并将季末月份的累计值表述为“截至该季度末的累计总额”；不得仅因缺少上季度末数据而拒答。"
                "只有用户明确要求单季新增额时，才需要用季末累计值减去上季度末累计值。"
            )
        if (
            any(re.fullmatch(r"20\d{2}年", period) for period in plan.periods)
            and {str(item.get("canonical")) for item in plan.financial_aliases}
            & {"资产总额", "负债总额"}
        ):
            prompt += (
                "\n\n年度时点指标规则：总资产、总负债属于时点余额，不存在把四个季度相加得到“全年合计”的口径。"
                "当用户只指定年份、证据按季度列示时，应列出各季度余额，并明确四季度数是年末余额；"
                "不得仅因证据没有另列“全年合计”而拒答。"
            )
        if implicit_choice_without_options:
            prompt += IMPLICIT_OPTION_INSTRUCTION
        # Empty-answer / option / locked-condition retries must see the same
        # complete facts and citation mapping as the first generation.
        short_evidence = evidence_cards
        retry_prompt = (
            "上一次没有返回文本。请直接给出完整、简洁的最终答案，不要输出思考过程或前言。"
            "只能使用下列证据；先覆盖问题的全部要求，再结束回答。"
            "题干即使包含“哪项正确”“哪一组”，也没有提供选项；绝不能以缺少选项为由拒答，"
            "也不要要求用户补充选项。只在六条证据均不含任何相关规定时才回答“依据不足”。"
            + ("题干要求两项时，必须列出两条独立规定。" if _MULTI_FACT_CHOICE_PATTERN.search(question) else "")
            + "答案控制在 800 个中文字符以内，并用 [E编号] 标注出处。\n\n"
            f"用户问题：{question}\n\n证据：\n{short_evidence}"
        )
        retry_prompt += locked_context
        if has_explicit_options:
            retry_prompt = (
                "\u4f60\u662f\u91d1\u878d\u76d1\u7ba1\u9009\u62e9\u9898\u56de\u7b54\u52a9\u624b\u3002\u4e0a\u4e00\u6b21\u6ca1\u6709\u8fd4\u56de\u6587\u672c\uff0c\u8bf7\u6839\u636e\u8bc1\u636e\u9010\u9879\u6838\u5bf9 A-D \u9009\u9879\u3002"
                "\u5fc5\u987b\u9009\u51fa\u4e00\u4e2a\u9009\u9879\uff0c\u7b54\u6848\u7b2c\u4e00\u884c\u56fa\u5b9a\u5199\uff1a\u6b63\u786e\u9009\u9879\uff1aX\u3002"
                "\u7b2c\u4e8c\u884c\u5199\u51fa\u8be5\u9009\u9879\u7684\u5b8c\u6574\u539f\u6587\uff0c\u5e76\u4ee5 [E\u7f16\u53f7] \u6807\u6ce8\u8bc1\u636e\u3002"
                "\u4e0d\u80fd\u53ea\u6444\u5f15\u8bc1\u636e\u3001\u4e0d\u80fd\u56de\u7b54\u201c\u65e0\u6b63\u786e\u9009\u9879\u201d\u3002\n\n"
                f"\u9898\u76ee\u4e0e\u9009\u9879\uff1a\n{question}\n\n\u8bc1\u636e\uff1a\n{short_evidence}"
            )
        retry_prompt += advisory_context
        retry_prompt += temporal_context
        if case_contract:
            retry_prompt += CASE_CONTRACT

        generation_attempts = []
        def generate_text(current_prompt: str, *, delivery_only: bool = False) -> GenerationReply:
            overrides = {}
            if case_contract:
                overrides['response_format'] = {'type': 'json_object'}
            if delivery_only:
                overrides['extra_body'] = {'thinking': {'type': 'disabled'}}
            try:
                reply = generator.run(
                    messages=[ChatMessage.from_user(current_prompt)],
                    **({'generation_kwargs': overrides} if overrides else {}),
                )["replies"][0]
            except Exception as exc:
                recovered = sdk_length_reply(exc)
                if recovered is None:
                    raise
                truncated, usage = recovered
                generation_attempts.append({'finish_reason': 'length', 'usage': usage,
                    'strategy': 'direct_delivery_no_thinking' if delivery_only else 'default',
                    'structured_condition_contract': case_contract,
                    'sdk_length_exception_recovered': True})
                return truncated
            meta = reply.meta or {}
            generation_attempts.append({"finish_reason": meta.get("finish_reason"), "usage": meta.get("usage", {}),
                                        "strategy": "direct_delivery_no_thinking" if delivery_only else "default",
                                        "structured_condition_contract": case_contract})
            text = reply.text or ""
            if case_contract and text.strip() and meta.get('finish_reason') != 'length':
                try:
                    contract = decode_case_contract(text, question, docs)
                    result['condition_assessment'] = contract
                    result['answer_action'] = contract['action']
                    text = contract['answer']
                except (ValueError, TypeError, KeyError, IndexError):
                    generation_attempts[-1]['contract_error'] = 'invalid_structured_condition_response'
                    text = ''
            return GenerationReply(text, meta.get("finish_reason"))

        generated, retry_reason = generate_with_one_incomplete_retry(
            generate_text,
            prompt,
            retry_prompt,
            truncation_generate=lambda text: generate_text(text, delivery_only=True),
        )
        answer = generated.text.strip()
        if retry_reason == "empty":
            result["retried_after_empty"] = True
        elif retry_reason == "length":
            result["retried_after_truncation"] = True
        result["generation_attempts"] = generation_attempts
        if generated.finish_reason == "length":
            result["generation_error"] = "truncated_after_one_retry"
            result["generation_incomplete"] = True
            result["answer_action"] = "DELIVERY_ERROR"
            result["answer"] = "本次未能在限定额度内交付可靠结论。已保留检索证据及来源，但不能将截断内容作为完整答案；请稍后重试。"
            result["verified"] = False
            return result
        locked_year = resolved.get("year") if resolved.get("locked") and resolved.get("relative_time_original") else None
        def conflicting_years(text: str) -> list[int]:
            if not locked_year:
                return []
            return sorted({int(value) for value in re.findall(r"(?<!\d)(20\d{2})年", text) if int(value) != locked_year})
        conflicts = conflicting_years(answer)
        if conflicts:
            result["locked_condition_conflict"] = {"field": "year", "expected": locked_year, "observed": conflicts}
            correction_prompt = (
                "请修正上一版答案。确定性解析已锁定年份为"
                f"{locked_year}年（原表达：{resolved['relative_time_original']}）。"
                "上一版重新解释了相对时间，与锁定条件冲突。不得提及或检索其他年份；"
                "不得因为错误年份缺少证据而拒答。只根据下列证据回答锁定年份的问题，"
                "如证据足够则直接给出数值，并用 [E编号] 引用；不得编造。\n\n"
                f"原问题（仅供语言理解）：{question}\n\n证据：\n{short_evidence}"
            )
            correction_prompt += advisory_context
            try:
                correction_reply = generate_text(correction_prompt) if len(generation_attempts) < 2 else None
                corrected = correction_reply.text.strip() if correction_reply and correction_reply.finish_reason != "length" else ""
                if correction_reply and correction_reply.finish_reason == "length":
                    result["generation_error"] = "truncated_during_locked_condition_correction"
                    result["generation_incomplete"] = True
            except Exception as exc:
                result["locked_condition_correction_error"] = type(exc).__name__
                corrected = ""
            if corrected and not conflicting_years(corrected):
                answer = corrected
                result["locked_condition_corrected"] = True
            else:
                result["locked_condition_corrected"] = False
                # A wrong-year answer must never escape as a valid conclusion.
                if result.get("generation_incomplete"):
                    answer = "本次答案生成未完成，请重试。"
                else:
                    answer = f"依据不足，无法确认{locked_year}年的所问数据；生成结果与已锁定时间条件冲突。"
                    result["insufficient_evidence"] = True
        if not answer:
            answer = _evidence_extract_fallback(
                prompt_docs,
                require_two=bool(_MULTI_FACT_CHOICE_PATTERN.search(question)),
                question=question,
            )
            if answer:
                result["generation_fallback"] = "evidence_extract_after_one_empty_retry"
            else:
                answer = "模型未返回文本，且未检索到可直接引用的证据；请重试。"
                result["generation_error"] = "empty_after_one_retry_without_evidence"
        result["answer"] = answer
        result["verified"] = self.verify_numbers(answer, docs)
        if not result["verified"]:
            result["verification_note"] = "回答中存在未在证据中逐字出现的数值，需人工复核。"
        return result

    def ask(self, question: str) -> dict[str, Any]:
        enabled = os.getenv("RAG_REUSE_INITIAL_GLOBAL", "0").strip().lower() in {"1", "true", "yes", "on"}
        with self._initial_global_cache.request(question, enabled=enabled) as request:
            result = self._ask_without_initial_global_cache(question)
            if enabled:
                result["initial_global_reuse"] = {"enabled": True, "events": request.events}
            return result

    def _ask_without_initial_global_cache(self, question: str) -> dict[str, Any]:
        """Answer through LangGraph by default, with a legacy rollback switch."""
        catalog_result = self._answer_file_catalog_question(question)
        if catalog_result is not None:
            return catalog_result
        orchestrator = os.getenv("RAG_ORCHESTRATOR", "langgraph").strip().lower()
        if orchestrator in {"legacy", "haystack"}:
            result = self._ask_pipeline(question)
            result["orchestrator"] = "legacy"
            return result
        if orchestrator != "langgraph":
            raise ValueError("RAG_ORCHESTRATOR 仅支持 langgraph 或 legacy")
        from .graph import invoke_rag_graph

        return invoke_rag_graph(self, question)
