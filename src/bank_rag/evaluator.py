"""Deterministic, evidence-aware evaluation for generated RAG answers.

The evaluator deliberately separates three questions:

* whether business numbers are supported by evidence or a valid calculation;
* whether source/time/structural numbers are valid metadata rather than facts;
* whether citations point to the expected and resolvable sources.

It contains no question IDs, corpus-specific source IDs, gold-answer exceptions,
or model calls.  The same rules therefore apply to future unseen evaluations.
"""
from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence


EVALUATION_RULES_VERSION = "evidence-aware-numeric-v1"
CITATION_RULES_VERSION = "strict-citation-v2"

ANSWER_FACT = "ANSWER_FACT"
SOURCE_METADATA = "SOURCE_METADATA"
TEMPORAL_METADATA = "TEMPORAL_METADATA"
STRUCTURAL_NUMBER = "STRUCTURAL_NUMBER"

_NUMBER_RE = re.compile(
    r"(?<![\d.])[-+]?(?:(?:\d{1,3}(?:[,\u00a0\u202f ]\d{3})+)|\d+)(?:\.\d+)?"
)
_UNIT_RE = re.compile(r"^(个百分点|万亿元|亿元|万元|%|倍|天|年|月|季度|页|段|条|项|号|部分|个)")
_REFUSAL_RE = re.compile(
    r"依据不足|无法(?:确认|回答|判断|计算|提供)|资料(?:中)?(?:没有|未提供)|"
    r"不能(?:确认|推断)|不能唯一确定"
)
_SOURCE_CONTEXT_RE = re.compile(
    r"文件|文号|编号|来源|页码|位置|段落|工作表|sheet|row|行号|单元格|附件",
    re.IGNORECASE,
)
_TIME_CONTEXT_RE = re.compile(r"年|月|季度|上半年|下半年|期末|年末|月末|季末")
_STRUCTURAL_CONTEXT_RE = re.compile(r"部分|章节|条款|第\s*$|序号|步骤|方面|组成|包括|如下")
_INLINE_REF_RE = re.compile(r"\[E(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceRecord:
    content: str
    metadata: Mapping[str, Any]


@dataclass
class NumericClaim:
    raw_generated_value: str
    normalized_value: str
    unit: str
    claim_type: str
    context: str
    comparison_result: bool = False
    evidence_value: str = ""
    matched_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NumericEvaluation:
    passed: bool
    claims: list[NumericClaim]
    answer_fact_count: int
    failed_claims: list[NumericClaim]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "answer_fact_count": self.answer_fact_count,
            "claims": [claim.to_dict() for claim in self.claims],
            "failed_claims": [claim.to_dict() for claim in self.failed_claims],
        }


@dataclass
class CitationEvaluation:
    passed: bool
    expected_sources_complete: bool
    inline_references_valid: bool
    resolvable_citation_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def source_matches(expected: object, actual: object) -> bool:
    left, right = normalize_text(expected), normalize_text(actual)
    return bool(left and right and (left in right or right in left))


def canonical_decimal(raw: object) -> Decimal | None:
    """Canonicalise thousands separators without changing numeric scale."""
    text = unicodedata.normalize("NFKC", str(raw or "")).strip()
    text = re.sub(r"(?<=\d)[,\u00a0\u202f ](?=\d{3}(?:\D|$))", "", text)
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def decimal_text(value: Decimal | None) -> str:
    if value is None:
        return ""
    rendered = format(value.normalize(), "f")
    return "0" if rendered in {"-0", ""} else rendered


def numeric_equal(left: Decimal, right: Decimal) -> bool:
    """Use a small tolerance only for serialisation-level floating tails."""
    difference = abs(left - right)
    tolerance = max(Decimal("1e-9"), max(abs(left), abs(right)) * Decimal("1e-12"))
    return difference <= tolerance


def _claim_unit(text: str, end: int) -> str:
    match = _UNIT_RE.match(text[end:])
    return match.group(1) if match else ""


def _line_prefix(text: str, start: int) -> str:
    return text[text.rfind("\n", 0, start) + 1:start]


def classify_numeric_claim(text: str, start: int, end: int, raw: str, unit: str) -> str:
    left = text[max(0, start - 14):start]
    right = text[end:min(len(text), end + 14)]
    context = left + raw + right
    value = canonical_decimal(raw)

    if re.fullmatch(r"\s*", _line_prefix(text, start)) and re.match(r"\s*[.、)）]", right):
        return STRUCTURAL_NUMBER

    # A source locator is metadata only when its local wording says so.  A
    # business fact equal to a file/page number is never accepted merely
    # because that number happens to occur in metadata.
    if _SOURCE_CONTEXT_RE.search(left[-10:]):
        return SOURCE_METADATA
    if unit == "个" and re.match(r"\s*个?\s*文件", right):
        return SOURCE_METADATA
    if left.endswith("第") and unit in {"页", "段", "号"}:
        return SOURCE_METADATA

    # Explicit business units always describe answer facts.  Their nearby
    # month/quarter labels must not reclassify the amount as time metadata.
    if unit in {"个百分点", "万亿元", "亿元", "万元", "%", "倍", "天"}:
        return ANSWER_FACT

    if unit in {"年", "月", "季度"}:
        return TEMPORAL_METADATA
    if value is not None and Decimal(1900) <= value <= Decimal(2100) and _TIME_CONTEXT_RE.search(context):
        return TEMPORAL_METADATA
    # Handles compact lists such as "7、8、9月" and "第1、3季度".
    if re.search(r"[、,，至到~-]\s*\d+(?:\s*[、,，至到~-]\s*\d+)*\s*(?:月|季度)", right):
        return TEMPORAL_METADATA
    if re.search(r"(?:月|季度)", right[:6]) and re.search(r"[、,，至到~-]", context):
        return TEMPORAL_METADATA

    if left.endswith("第") and unit in {"条", "项", "部分"}:
        return STRUCTURAL_NUMBER
    if unit in {"部分", "项"} and _STRUCTURAL_CONTEXT_RE.search(context):
        return STRUCTURAL_NUMBER

    return ANSWER_FACT


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    claims: list[NumericClaim] = []
    normalized = unicodedata.normalize("NFKC", text)
    citation_spans = [match.span() for match in _INLINE_REF_RE.finditer(normalized)]
    for match in _NUMBER_RE.finditer(normalized):
        if any(start <= match.start() and match.end() <= end for start, end in citation_spans):
            continue
        raw = match.group(0)
        value = canonical_decimal(raw)
        if value is None:
            continue
        unit = _claim_unit(normalized, match.end())
        claim_type = classify_numeric_claim(normalized, match.start(), match.end(), raw, unit)
        context = normalized[max(0, match.start() - 18):min(len(normalized), match.end() + len(unit) + 18)]
        claims.append(NumericClaim(
            raw_generated_value=raw,
            normalized_value=decimal_text(value),
            unit=unit,
            claim_type=claim_type,
            context=context,
        ))
    return claims


def _unit_compatible(answer_unit: str, evidence_text: str, evidence_unit: str) -> bool:
    if not answer_unit:
        return True
    if answer_unit == "%":
        return evidence_unit == "%" or "%" in evidence_text
    if answer_unit in {"亿元", "万元", "万亿元", "个百分点"}:
        return answer_unit == evidence_unit or answer_unit in evidence_text
    return True


def _evidence_numbers(evidence: Sequence[EvidenceRecord]) -> list[tuple[Decimal, str, str]]:
    values: list[tuple[Decimal, str, str]] = []
    for record in evidence:
        for claim in extract_numeric_claims(record.content):
            value = canonical_decimal(claim.normalized_value)
            if value is not None:
                values.append((value, claim.unit, record.content))
    return values


def _metadata_text(citations: Sequence[Mapping[str, Any]], evidence: Sequence[EvidenceRecord]) -> str:
    parts = [json.dumps(dict(citation), ensure_ascii=False, sort_keys=True) for citation in citations]
    parts.extend(json.dumps(dict(record.metadata), ensure_ascii=False, sort_keys=True) for record in evidence)
    return "\n".join(parts)


def _all_numeric_values(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for match in _NUMBER_RE.finditer(unicodedata.normalize("NFKC", text)):
        value = canonical_decimal(match.group(0))
        if value is not None:
            values.append(value)
    return values


def _safe_arithmetic(expression: str) -> Decimal | None:
    expression = (
        expression.replace(",", "").replace(" ", "")
        .replace("－", "-").replace("−", "-")
        .replace("×", "*").replace("÷", "/")
    )
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def evaluate(item: ast.AST) -> Decimal:
        if isinstance(item, ast.Expression):
            return evaluate(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return Decimal(str(item.value))
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.USub, ast.UAdd)):
            value = evaluate(item.operand)
            return -value if isinstance(item.op, ast.USub) else value
        if isinstance(item, ast.BinOp) and isinstance(item.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(item.left), evaluate(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            if isinstance(item.op, ast.Mult):
                return left * right
            if right == 0:
                raise ArithmeticError("division by zero")
            return left / right
        raise ValueError("unsupported arithmetic expression")

    try:
        return evaluate(node)
    except (ArithmeticError, InvalidOperation, ValueError):
        return None


def validate_derived_calculation(calculation: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate a supplied deterministic calculation without using raw gold."""
    if not calculation.get("inputs_verified"):
        return False, "inputs_not_verified"
    expression = str(calculation.get("expression") or "")
    stated = canonical_decimal(calculation.get("result"))
    computed = _safe_arithmetic(expression)
    if computed is None or stated is None:
        return False, "invalid_expression_or_result"
    return numeric_equal(computed, stated), decimal_text(computed)


def _inline_derived_values(answer: str, evidence_values: Sequence[tuple[Decimal, str, str]]) -> list[Decimal]:
    number = r"[-+]?(?:(?:\d{1,3}(?:[, ]\d{3})+)|\d+)(?:\.\d+)?"
    pattern = re.compile(
        rf"(?P<left>{number})\s*(?P<op>[+\-－−×*/÷])\s*(?P<right>{number})\s*[=＝]\s*(?:\*\*)?(?P<result>{number})(?:\*\*)?"
    )
    supported: list[Decimal] = []
    available = [value for value, _unit, _content in evidence_values]
    for match in pattern.finditer(unicodedata.normalize("NFKC", answer)):
        left = canonical_decimal(match.group("left"))
        right = canonical_decimal(match.group("right"))
        stated = canonical_decimal(match.group("result"))
        if left is None or right is None or stated is None:
            continue
        if not all(any(numeric_equal(value, candidate) for candidate in available) for value in (left, right)):
            continue
        computed = _safe_arithmetic(f"{left}{match.group('op')}{right}")
        if computed is not None and numeric_equal(computed, stated):
            supported.append(stated)
    return supported


def evaluate_numeric_claims(
    answer: str,
    *,
    question: str = "",
    evidence: Sequence[EvidenceRecord] = (),
    citations: Sequence[Mapping[str, Any]] = (),
    derived_calculations: Sequence[Mapping[str, Any]] = (),
) -> NumericEvaluation:
    claims = extract_numeric_claims(answer)
    evidence_numbers = _evidence_numbers(evidence)
    metadata_text = _metadata_text(citations, evidence)
    metadata_values = _all_numeric_values(metadata_text)
    temporal_values = _all_numeric_values(question + "\n" + metadata_text + "\n" + "\n".join(r.content for r in evidence))
    derived_values = _inline_derived_values(answer, evidence_numbers)
    for calculation in derived_calculations:
        valid, _reason = validate_derived_calculation(calculation)
        value = canonical_decimal(calculation.get("result"))
        if valid and value is not None:
            derived_values.append(value)

    for claim in claims:
        value = canonical_decimal(claim.normalized_value)
        if value is None:
            continue
        if claim.claim_type == STRUCTURAL_NUMBER:
            claim.comparison_result = True
            claim.matched_by = "structural_context"
            continue
        if claim.claim_type == SOURCE_METADATA:
            if claim.unit == "个" and re.search(r"个\s*文件", claim.context):
                distinct_sources = {
                    normalize_text(citation.get("source_path") or citation.get("source_title"))
                    for citation in citations
                    if normalize_text(citation.get("source_path") or citation.get("source_title"))
                }
                count = Decimal(len(distinct_sources))
                claim.comparison_result = numeric_equal(value, count)
                claim.evidence_value = decimal_text(count)
                claim.matched_by = "source_metadata_count" if claim.comparison_result else ""
                continue
            match = next((candidate for candidate in metadata_values if numeric_equal(value, candidate)), None)
            claim.comparison_result = match is not None
            claim.evidence_value = decimal_text(match)
            claim.matched_by = "source_metadata" if match is not None else ""
            continue
        if claim.claim_type == TEMPORAL_METADATA:
            match = next((candidate for candidate in temporal_values if numeric_equal(value, candidate)), None)
            claim.comparison_result = match is not None
            claim.evidence_value = decimal_text(match)
            claim.matched_by = "temporal_question_or_metadata" if match is not None else ""
            continue

        evidence_match = next(
            (
                (candidate, evidence_unit)
                for candidate, evidence_unit, content in evidence_numbers
                if numeric_equal(value, candidate) and _unit_compatible(claim.unit, content, evidence_unit)
            ),
            None,
        )
        if evidence_match is not None:
            claim.comparison_result = True
            claim.evidence_value = decimal_text(evidence_match[0])
            claim.matched_by = "evidence_content"
            continue
        derived_match = next((candidate for candidate in derived_values if numeric_equal(value, candidate)), None)
        if derived_match is not None:
            claim.comparison_result = True
            claim.evidence_value = decimal_text(derived_match)
            claim.matched_by = "deterministic_calculation"

    failed = [claim for claim in claims if not claim.comparison_result]
    return NumericEvaluation(
        passed=not failed,
        claims=claims,
        answer_fact_count=sum(claim.claim_type == ANSWER_FACT for claim in claims),
        failed_claims=failed,
    )


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text))


def is_full_refusal(text: str) -> bool:
    """Distinguish a whole-answer refusal from a later scoped caveat."""
    match = _REFUSAL_RE.search(text)
    if not match:
        return False
    first_sentence_end = min(
        [position for marker in ("。", "！", "？", "\n") if (position := text.find(marker)) >= 0]
        or [len(text)]
    )
    return match.start() <= first_sentence_end


def looks_truncated(text: str) -> bool:
    stripped = re.sub(r"\[E\d+\]\s*$", "", text.strip(), flags=re.IGNORECASE).rstrip()
    return bool(stripped) and stripped[-1] in {"；", ";", "：", ":", "，", ","}


def refusal_matches(expected: str, answer: str) -> bool:
    if not (is_full_refusal(expected) and is_full_refusal(answer)):
        return False
    expected_numbers = _all_numeric_values(expected)
    answer_numbers = _all_numeric_values(answer)
    if not all(any(numeric_equal(value, candidate) for candidate in answer_numbers) for value in expected_numbers):
        return False
    left, right = normalize_text(expected), normalize_text(answer)
    if len(left) < 2:
        return True
    bigrams = {left[index:index + 2] for index in range(len(left) - 1)}
    coverage = sum(bigram in right for bigram in bigrams) / len(bigrams)
    return coverage >= 0.30


def evaluate_citations(
    answer: str,
    citations: Sequence[Mapping[str, Any]],
    *,
    expected_sources: Sequence[str] = (),
    resolved_citations: Sequence[bool] | None = None,
    safe_refusal: bool = False,
) -> CitationEvaluation:
    if safe_refusal:
        return CitationEvaluation(True, True, True, 0, "safe_refusal_does_not_require_citation")
    expected = [source for source in expected_sources if normalize_text(source)]
    expected_complete = all(
        any(source_matches(source, citation.get("source_title") or citation.get("source_path")) for citation in citations)
        for source in expected
    )
    references = [int(value) for value in _INLINE_REF_RE.findall(answer)]
    inline_valid = bool(references) and all(1 <= index <= len(citations) for index in references)
    if resolved_citations is None:
        resolvable = len(citations)
    else:
        resolvable = sum(bool(value) for value in resolved_citations)
    referenced_resolve = resolved_citations is None or all(
        index <= len(resolved_citations) and bool(resolved_citations[index - 1])
        for index in references if 1 <= index <= len(citations)
    )
    passed = bool(citations) and expected_complete and inline_valid and resolvable > 0 and referenced_resolve
    if not citations:
        reason = "no_citations"
    elif not expected_complete:
        reason = "expected_source_missing"
    elif not inline_valid:
        reason = "invalid_inline_reference"
    elif resolvable == 0 or not referenced_resolve:
        reason = "citations_not_resolvable"
    else:
        reason = "citations_resolve_and_expected_sources_complete"
    return CitationEvaluation(passed, expected_complete, inline_valid, resolvable, reason)
