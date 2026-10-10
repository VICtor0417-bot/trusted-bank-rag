"""Deterministic decomposition for Chinese regulatory/table questions.

The parser deliberately separates *what the user said* from *how a table is
queried*.  In particular, ``合计`` is never silently treated as "sum every
numeric field".  It becomes an explicit aggregation policy which the table
executor must verify against an actual total row or total column.
"""

from __future__ import annotations

import re
from datetime import date
from dataclasses import asdict, dataclass, field

from .canonicalization import canonicalize_area, canonicalize_area_mentions, is_known_area
from .financial_lexicon import canonical_financial_term, matched_financial_terms
from .operation_lexicon import SUPPORTED_OPERATION_PATTERNS, UNSUPPORTED_OPERATION_PATTERNS
from .table_selection import explicit_premium_selection


_OPERATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("比较", r"(?:比较|判断哪.+(?:更高|更低|较高|较低|更多|更少)|哪.+(?:更高|更低|较高|较低|更多|更少)|多出多少|少(?:了|于)多少|相差多少)"),
    ("百分点变化", r"(?:变化(?:约)?多少个百分点|(?:提高|下降|上升|降低).{0,8}个百分点|百分点)"),
    ("增长率", r"(?:增长率|增幅|增长百分比|同比(?:增长)?|环比(?:增长)?)"),
    ("增加额", r"(?:增加额|增加金额|增量|增加了多少|新增(?:金额|数额)?)"),
    ("减少额", r"(?:减少额|减少金额|减量|减少了多少|下降(?:金额|数额)?)"),
    ("差额", r"(?:差额|相差多少|差值)"),
    ("平均值", r"(?:平均值|平均数|均值)"),
    ("最大值", r"(?:最大值|最高值|最多|最大)"),
    ("最小值", r"(?:最小值|最低值|最少|最小)"),
    ("求和", r"(?:求和|相加|加总|合并计算|汇总|合并|总和)"),
)

# Kept as a private alias so existing parser code remains compact; the source
# of truth lives in operation_lexicon.py.
_OPERATION_PATTERNS = SUPPORTED_OPERATION_PATTERNS

_AGGREGATION_PATTERN = re.compile(r"(?:合计|总计|总额|合计数|总数)")
_SOURCE_PATTERN = re.compile(r"《\s*([^》]+?)\s*》")
_PERIOD_PATTERN = re.compile(
    r"(?:20\d{2}年(?:[1-9]|1[0-2])月|20\d{2}年(?:第?[一二三四1-4]季度|上半年|下半年|全年)|"
    r"(?:[一二三四1-4])季度|(?:本期|上期|同期|当期|上一期))"
)
_AREA_PATTERN = re.compile(
    r"(?:[\u4e00-\u9fff]{2,20}?(?:特别行政区|自治区|自治州|地区|盟|省|市)|全国)"
)
_ENTITY_PATTERN = re.compile(
    r"(?:大型商业银行|股份制商业银行|城市商业银行|农村商业银行|外资银行|"
    r"人身险公司|财产保险公司|财产险公司|再保险公司|保险公司)"
)

_CHINESE_MONTHS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}

INSURANCE_OPERATING_OVERVIEW_METRICS: tuple[str, ...] = (
    "原保险保费收入", "财产险", "人身险", "原保险赔付支出", "总资产", "净资产",
)


def insurance_operating_overview_metrics(question: str) -> list[str]:
    """Expand a request for the operating overview into its standard rows."""
    if "保险业经营情况" not in question:
        return []
    return list(INSURANCE_OPERATING_OVERVIEW_METRICS)


def _normalise_time_language(text: str, *, today: date | None = None) -> str:
    """Normalise Chinese and relative months for deterministic planning only."""
    reference = today or date.today()
    text = text.replace("今年", f"{reference.year}年").replace("去年", f"{reference.year - 1}年")
    for chinese, number in sorted(_CHINESE_MONTHS.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"{chinese}月", f"{number}月", text)
    return text


@dataclass(frozen=True)
class QuestionPlan:
    original_question: str
    intent: str
    area: str | None
    areas: list[str]
    entity: str | None
    metric: str | None
    aggregation: str | None
    aggregation_policy: str | None
    operations: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    rewritten_question: str = ""
    table_program: dict | None = None
    unsupported_operations: list[str] = field(default_factory=list)
    financial_aliases: list[dict] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    clarification_needed: list[str] = field(default_factory=list)
    resolved_conditions: dict = field(default_factory=dict)
    area_canonicalizations: list[dict[str, str]] = field(default_factory=list)
    table_selection: dict = field(default_factory=dict)
    metric_binding: dict = field(default_factory=dict)
    source_title_provenance: str = "legacy"

    @property
    def required_source_titles(self) -> list[str]:
        """Inferred report families are retrieval hints, never requirements.

        Older serialized plans must prove explicit provenance from the question;
        absence of the new field must not promote an inference to a requirement.
        """
        if self.source_title_provenance == "legacy":
            return _source_titles_and_periods(self.original_question)[0]
        return list(self.source_titles) if self.source_title_provenance == "user_explicit" else []

    def to_dict(self) -> dict:
        return asdict(self)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _expand_compact_month_title(title: str) -> list[str]:
    """Expand ``2023年10月、11月、12月某表`` into three real title hints."""
    match = re.match(r"(20\d{2})年", title)
    if not match:
        return [title]
    year = match.group(1)
    month_matches = list(re.finditer(r"(\d{1,2})月", title))
    if len(month_matches) < 2:
        return [title]
    suffix = title[month_matches[-1].end() :]
    # Only expand a compact month list, not a title that happens to mention
    # multiple unrelated dates in its body.
    between = title[month_matches[0].end() : month_matches[-1].start()]
    if not re.fullmatch(r"[\s、,，和及与0-9月]+", between):
        return [title]
    return [f"{year}年{int(item.group(1))}月{suffix}" for item in month_matches]


def _extract_month_periods(text: str) -> list[str]:
    """Expand compact month lists/ranges while preserving their stated order.

    Examples include ``2023年12月和11月`` and ``2023年9月至12月``.  The old
    period regex only retained the first fully-qualified month, which made the
    source planner route to the wrong files before retrieval began.
    """
    periods: list[str] = []
    pattern = re.compile(
        r"(?P<year>20\d{2})年(?P<body>\d{1,2}月(?:\s*(?:、|,|，|和|及|与|至|到|相对于|较|比|[-—])\s*(?:(?:20\d{2})年)?\d{1,2}月)*)"
    )
    for match in pattern.finditer(text):
        current_year = int(match.group("year"))
        tokens = list(re.finditer(r"(?:(20\d{2})年)?(\d{1,2})月", match.group("body")))
        if not tokens:
            continue
        values: list[tuple[int, int]] = []
        for token in tokens:
            if token.group(1):
                current_year = int(token.group(1))
            month = int(token.group(2))
            if 1 <= month <= 12:
                values.append((current_year, month))
        is_range = bool(re.search(r"(?:至|到|[-—])", match.group("body")))
        if is_range and len(values) == 2:
            start = values[0][0] * 12 + values[0][1] - 1
            end = values[1][0] * 12 + values[1][1] - 1
            step = 1 if end >= start else -1
            values = [(value // 12, value % 12 + 1) for value in range(start, end + step, step)]
        periods.extend(f"{year}年{month}月" for year, month in values)
    return _unique(periods)


def _source_titles_and_periods(question: str) -> tuple[list[str], list[str]]:
    raw_titles = [item.strip() for item in _SOURCE_PATTERN.findall(question)]
    source_titles = _unique([expanded for title in raw_titles for expanded in _expand_compact_month_title(title)])
    periods = _extract_month_periods(question)
    # In a connected period expression, omitted quarter years inherit the
    # preceding explicit year, just as compact month lists already do.
    # Never use today's year when the question provides no year.
    current_year = None
    for match in _PERIOD_PATTERN.finditer(question):
        period = match.group(0)
        year = re.match(r"(20\d{2})年", period)
        if year:
            current_year = year.group(1)
        elif current_year and re.fullmatch(r"[一二三四1-4]季度", period):
            period = f"{current_year}年{period}"
        periods.append(period)
    if not periods:
        periods.extend(f"{year}年" for year in re.findall(r"(?<!\d)((?:19|20)\d{2})年", question))
    for title in source_titles:
        periods.extend(re.findall(r"20\d{2}年(?:[1-9]|1[0-2])月", title))
    return source_titles, _unique(periods)


def _infer_monthly_source_titles(
    question: str, periods: list[str], areas: list[str], metric: str | None,
    entity: str | None = None,
) -> list[str]:
    """Infer only unambiguous monthly report families omitted by the user.

    A city/province plus ``原保险保费收入`` has one clear family in this
    dataset: the monthly national regional premium tables.  We intentionally
    do not infer a source for generic terms such as ``资产`` or ``收入``.
    """
    canonical_metric = canonical_financial_term(metric or "")
    company = "财产保险公司" if entity == "财产险公司" else entity
    month_periods = [period for period in periods if re.fullmatch(r"20\d{2}年(?:[1-9]|1[0-2])月", period)]
    if not month_periods:
        # Monthly insurance tables report premium income on a year-to-date
        # basis.  A year-only question therefore has one deterministic annual
        # endpoint: the December cumulative record.  Route there before a
        # semantic reranker can spread equal-scoring candidates across months.
        year_periods = [period for period in periods if re.fullmatch(r"20\d{2}年", period)]
        if len(year_periods) == 1 and canonical_metric == "原保险保费收入":
            year = year_periods[0][:-1]
            if areas:
                return [f"{year}年12月全国各地区原保险保费收入情况表"]
            if company in {"人身险公司", "财产保险公司", "再保险公司"}:
                return [f"{year}年12月{company}经营情况表"]
            return [f"{year}年12月保险业经营情况表"]
        return []
    # The nationwide monthly operating table is the unambiguous source for
    # month-specific "保险业经营情况" questions and for nationwide premium
    # totals with no province/city filter.
    if not areas and canonical_metric == "原保险保费收入" and company in {
        "人身险公司", "财产保险公司", "再保险公司",
    }:
        return [f"{period}{company}经营情况表" for period in month_periods]
    if "保险业经营情况" in question or (
        not areas and canonical_metric == "原保险保费收入"
    ):
        return [f"{period}保险业经营情况表" for period in month_periods]
    if not areas:
        return []
    monthly_premium_metrics = {
        "原保险保费收入", "财产保险", "人寿保险", "健康保险", "意外伤害保险",
    }
    if canonical_metric not in monthly_premium_metrics:
        return []
    titles = [f"{period}全国各地区原保险保费收入情况表" for period in month_periods]
    # Difference/increase is conventionally later period minus earlier period.
    # The executor consumes a two-source calculation as [current, previous].
    if len(titles) == 2 and not re.search(r"(?:连续|相邻月份|相邻期间)", question):
        def period_key(title: str) -> tuple[int, int]:
            match = re.search(r"(20\d{2})年(\d{1,2})月", title)
            return (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        return sorted(titles, key=period_key, reverse=True)
    return titles


def _extract_metric(question: str, areas: list[str], operations: list[str]) -> str | None:
    """Return the requested indicator without inventing an equivalent term."""
    quoted = re.findall(r"[\u201c\"]\s*([^\u201d\"]+?)\s*[\u201d\"]", question)
    for item in quoted:
        candidate = item.strip()
        # Users often put an arithmetic instruction in quotes (for example
        # “后期－前期”).  It is not an indicator and must not override the
        # actual financial term elsewhere in the question.
        if re.fullmatch(r"(?:后期|前期|本期|上期|当期|上一期)[+\-－−×*/÷].*", candidate):
            continue
        if candidate in {"合计", "总计", "算式", "过程"}:
            continue
        return candidate

    # Prefer a recognised financial/table indicator in the original wording.
    # This must happen before instruction fragments are stripped below.  For
    # example, in "把 10 月和 11 月的全国合计原保险保费收入算一下增加额，
    # 写后期－前期的算式", the trailing "后期－前期" is an operation
    # explanation, not the metric.  Picking it previously prevented monthly
    # table-family inference and sent the calculation back to BM25.
    non_metric_terms = {
        "增加额", "减少额", "增长率", "百分点变化", "差额", "平均值", "最大值", "最小值", "求和", "合计",
    }
    # A source title names the broad table family (for example
    # ``原保险保费收入情况表``), while the question body may request a more
    # specific column such as 寿险 or 健康险.  Prefer terms outside quoted
    # source titles so the family name cannot overwrite the requested metric.
    metric_search_text = _SOURCE_PATTERN.sub("", question)
    # Institution names are filters, not insurance-business indicators.
    # Mask the complete entity before matching substring aliases such as 人身险.
    metric_search_text = _ENTITY_PATTERN.sub(" ", metric_search_text)
    # Some users mention the monthly table family without title brackets, for
    # example "2023年12月和10月全国各地区原保险保费收入情况表中的全国合计寿险收入".
    # Remove that broad family label before choosing the requested indicator;
    # otherwise 原保险保费收入 wins over the actual 寿险/财产险/健康险 column.
    metric_search_text = re.sub(
        r"(?:20\d{2}年(?:[1-9]|1[0-2])月(?:和|及|与|、|至|-)?)*"
        r"全国各地区原保险保费收入情况表",
        "",
        metric_search_text,
    )
    recognised_metrics = [
        item["canonical"]
        for item in matched_financial_terms(metric_search_text)
        if item["canonical"] not in non_metric_terms
    ]
    if not recognised_metrics:
        recognised_metrics = [
            item["canonical"]
            for item in matched_financial_terms(_ENTITY_PATTERN.sub(" ", question))
            if item["canonical"] not in non_metric_terms
        ]
    if recognised_metrics:
        # ``matched_financial_terms`` is already ordered by the most specific
        # matched wording.  Re-sorting by canonical-label length made
        # “不良贷款率” lose to the broader substring “不良贷款”.
        return recognised_metrics[0]

    text = question
    text = _SOURCE_PATTERN.sub("", text)
    text = _PERIOD_PATTERN.sub("", text)
    text = re.sub(
        r"(?:连续[一二三四五六七八九十0-9两]+个?月|第[一二三四五六七八九十0-9两]+个?月|"
        r"列出[一二三四五六七八九十0-9两]+个?相邻月份区间|相邻月份区间|相邻期间)",
        "",
        text,
    )
    text = _AREA_PATTERN.sub("", text)
    for area in areas:
        text = text.replace(area, "")
    # These are task instructions rather than table indicators.  Keeping them
    # here would let a short trailing phrase such as "判断最大值" become the
    # extracted metric after the real indicator has been removed from a title.
    text = re.sub(
        r"(?:请|帮我|需要|如何|计算|查询|检索|给出|列出|分别|其中|根据|对|将|的|"
        r"是多少|为多少|情况|数据|数值|判断|哪个|哪一|更高|更低|更多|更少|最大|最小)",
        " ",
        text,
    )
    text = _AGGREGATION_PATTERN.sub(" ", text)
    for _name, pattern in _OPERATION_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"(?:和|及|与|并|再|然后|从|到|相比|同比|环比|后期|前期|本期|上期|算式|过程)", " ", text)
    candidates = [item.strip() for item in re.split(r"[，。；、,：:？?\s]+", text)]
    candidates = [item for item in candidates if len(item) >= 2 and not re.fullmatch(r"[0-9.%]+", item)]
    if candidates:
        return max(candidates, key=len)

    # In compact multi-month wording, the source title may be the only place
    # that names the metric: 《2023年9月、10月…全国各地区原保险保费收入情况表》.
    # Fall back to that title-shaped phrase only after normal extraction fails,
    # so a user-specified metric outside the title always takes precedence.
    title_metric = re.search(r"全国各地区(?P<metric>[\u4e00-\u9fffA-Za-z0-9（）()]+?)(?:情况表|统计表)", question)
    if title_metric:
        candidate = title_metric.group("metric").strip()
        if len(candidate) >= 2:
            return candidate
    return None


def metric_binding_for(question: str, metric: str | None) -> dict:
    """Distinguish supported labels from lossy sentence-stripping hints.

    This provenance controls hard constraints; unknown wording is still kept
    for retrieval, but must not be locked as a deterministically known metric.
    """
    if not metric:
        return {"value": None, "provenance": "unresolved", "hard": False}
    terms = matched_financial_terms(_ENTITY_PATTERN.sub(" ", question), max_terms=100)
    if canonical_financial_term(metric) in {t["canonical"] for t in terms}:
        return {"value": metric, "provenance": "lexicon_match", "hard": True}
    quoted = re.findall(r'[“"]\s*([^”"]+?)\s*[”"]', question)
    if metric in quoted and not re.search(r'怎么|如何|是否|应该|多少|后期|前期', metric):
        return {"value": metric, "provenance": "explicit_quoted_label", "hard": True}
    return {"value": metric, "provenance": "unverified_retrieval_hint", "hard": False}


def _rewrite_question(
    question: str,
    intent: str,
    area: str | None,
    areas: list[str],
    entity: str | None,
    metric: str | None,
    aggregation: str | None,
    operations: list[str],
    periods: list[str],
    source_titles: list[str],
) -> str:
    """Create a lossless, schema-oriented version of the user's request.

    This is not a translation and not an LLM-authored synonym substitution. It
    only makes the already extracted slots explicit so another RAG system can
    use it as a second retrieval query without changing the requested meaning.
    """
    if intent != "table_calculation" or not metric:
        return question
    scope = "、".join(f"《{title}》" for title in source_titles) or "相关原始统计表"
    filters = [f"地区为{'、'.join(areas)}" if areas else (f"地区为{area}" if area else "按原表地区口径")]
    if entity:
        filters[0] = f"机构类别为“{entity}”"
    filters.append(f"指标为“{metric}”")
    if aggregation:
        filters.append("统计口径优先使用原表已有的“合计”或“总计”行、列或字段；不存在时不得擅自汇总")
    period_text = "、".join(periods) if periods else "同一统计口径下的本期和上期"
    operation_text = "、".join(operations) if operations else "取数"
    return f"根据{scope}，{ '；'.join(filters) }；以{period_text}为比较期间，计算{operation_text}，并引用参与计算的原始单元格。"


def _compile_table_program(
    question: str, intent: str, areas: list[str], entity: str | None, metric: str | None,
    aggregation: str | None, operations: list[str], source_titles: list[str], periods: list[str],
) -> dict | None:
    """Compile supported table language into a small, auditable operation tree."""
    if intent != "table_calculation" or not metric or not source_titles:
        return None
    selector = {
        "sources": source_titles,
        "periods": periods,
        "dimension": "entity" if entity else "area",
        "members": [entity] if entity else areas,
        "metric": metric,
        # ``全国合计寿险收入`` uses the 全国合计 row and the 寿险 column.
        # A generic 合计 column is only appropriate when the requested metric
        # itself is total original-premium income.
        "column_policy": (
            "explicit_total"
            if aggregation and canonical_financial_term(metric) == "原保险保费收入"
            else "metric_column"
        ),
    }
    aggregate = "sum" if "求和" in operations else "identity"
    final_operations: list[str] = []
    if "平均值" in operations:
        final_operations.append("mean")
    if "增加额" in operations or "差额" in operations:
        final_operations.append("difference")
    if "减少额" in operations:
        final_operations.append("reverse_difference")
    if "增长率" in operations:
        final_operations.append("growth_rate")
    if "百分点变化" in operations:
        final_operations.append("percentage_point_difference")
    if "最大值" in operations:
        final_operations.append("max")
    if "最小值" in operations:
        final_operations.append("min")
    if "比较" in operations:
        final_operations.append("rank")
    if len(source_titles) >= 3 and re.search(r"(?:连续|相邻月份|相邻期间)", question) and "difference" in final_operations:
        final_operations.append("sequential_difference")
        if re.search(r"第二个?月.{0,12}第一个?月", question):
            final_operations.append("difference_of_differences")
    rank_order = "ascending" if re.search(r"(?:更低|较低|更少|最少|最低)", question) else "descending"
    return {"selector": selector, "per_source_aggregate": aggregate, "final_operations": final_operations, "rank_order": rank_order}


def _is_table_calculation_request(question: str, operations: list[str]) -> bool:
    """Distinguish arithmetic table questions from document comparisons.

    Regulatory questions commonly say “对比两份附件” or “合并的报告事项”.
    Those are retrieval/synthesis questions, not instructions to run the table
    calculator.  Requiring a numeric/calculation cue prevents the calculator
    from returning its generic period clarification before retrieval begins.
    """
    arithmetic = {"百分点变化", "增长率", "增加额", "减少额", "差额", "平均值", "最大值", "最小值", "求和"}
    if any(operation in arithmetic for operation in operations):
        return True
    if "比较" not in operations:
        return False
    return bool(re.search(
        r"(?:计算|数值|金额|数额|比例|比率|百分点|增长|增幅|同比|环比|收入|余额|保费|最高|最低|多少)",
        question,
    ))


def parse_question(question: str) -> QuestionPlan:
    """Split a natural-language question into an auditable query plan.

    This is intentionally rule-based at this stage: it is fast, deterministic
    and safe to expose in an API.  A future LLM synonym step can add *candidate*
    terms after this plan, but must not overwrite ``metric`` or the policy.
    """
    original_question = " ".join(question.split())
    question = _normalise_time_language(original_question)
    # Strip leading instruction verbs before matching an administrative area;
    # otherwise the greedy CJK pattern can read ``计算深圳市`` as one city name.
    # Attachment names often contain an administrative scope (for example
    # “全国各地区…”).  That describes the table, not necessarily the requested
    # region, so remove quoted file titles before extracting the row filter.
    area_search_text = _SOURCE_PATTERN.sub("", question)
    # Common geographic shorthand must be expanded before the normal area
    # matcher runs.  Preserve the user's city scope even when a source only
    # publishes provincial statistics; Guangzhou must not become Guangdong.
    area_search_text = re.sub(r"北上广(?:三地)?", "北京市、上海市、广州市", area_search_text)
    area_search_text, area_canonicalizations = canonicalize_area_mentions(area_search_text)
    # Remove dates before looking for a province/city.  Otherwise the greedy
    # CJK matcher can consume ``月的北京市`` as a fictitious city.
    area_search_text = _PERIOD_PATTERN.sub(" ", area_search_text)
    area_search_text = re.sub(r"^[\s，,、和及与]+", "", area_search_text)
    area_search_text = re.sub(r"^(?:请|帮我|需要|如何|计算|查询|检索|给出|分别|根据|对|将|能否|能不能|能解释|解释|为什么)+", "", area_search_text)
    area_search_text = re.sub(r"^[\s，,、和及与]+", "", area_search_text)
    area_search_text = re.sub(r"^(?:请|帮我|需要|如何|计算|查询|检索|给出|分别|根据|对|将|能否|能不能|能解释|解释|为什么)+", "", area_search_text)
    raw_areas = _AREA_PATTERN.findall(area_search_text)
    areas = _unique([
        canonicalize_area(re.sub(r"^(?:比较|和|及|与|、|对|的|月|年|期|将|把|在)+", "", value).strip())
        for value in raw_areas
    ])
    # The regex intentionally finds long CJK area-shaped phrases.  Only allow
    # values backed by the nationwide canonicalization library (plus 全国) to
    # become hard filters.  This prevents source descriptions such as
    # “三份全国各地区…情况表” from becoming a fictitious administrative area.
    areas = [
        value for value in areas
        if is_known_area(value)
        and not re.fullmatch(r"(?:判断)?哪(?:一|个)?省", value)
    ]
    area = areas[0] if len(areas) == 1 else None
    entity_match = _ENTITY_PATTERN.search(area_search_text)
    entity = entity_match.group(0) if entity_match else None
    operations = [name for name, pattern in _OPERATION_PATTERNS if re.search(pattern, question)]
    unsupported_operations = [
        name for name, pattern in UNSUPPORTED_OPERATION_PATTERNS
        if re.search(pattern, question, flags=re.IGNORECASE)
    ]
    aggregation = "合计" if _AGGREGATION_PATTERN.search(question) else None
    metric_question = re.sub(r'(?:行|统计口径)[“"][^”"]+[”"]', '', question)
    metric = _extract_metric(metric_question, areas, operations)
    metric_binding = metric_binding_for(metric_question, metric)
    table_selection = explicit_premium_selection(question, areas, metric)
    source_titles, periods = _source_titles_and_periods(question)
    source_title_provenance = "user_explicit" if source_titles else "deterministic_inferred"
    if not source_titles:
        if table_selection and periods and all(re.fullmatch(r'20\d{2}年\d{1,2}月', p) for p in periods):
            source_titles = [f'{p}全国各地区原保险保费收入情况表' for p in periods]
        else:
            source_titles = _infer_monthly_source_titles(question, periods, areas, metric, entity)
    if not source_titles:
        source_title_provenance = "none"
    intent = "table_calculation" if _is_table_calculation_request(question, operations) else "retrieval"
    policy = "prefer_explicit_total" if aggregation else None

    rewritten_question = _rewrite_question(
        question, intent, area, areas, entity, metric, aggregation, operations, periods, source_titles
    )
    table_program = _compile_table_program(
        question, intent, areas, entity, metric, aggregation, operations, source_titles, periods
    )
    if table_selection:
        table_program = {
            'selector': {'sources': source_titles, 'periods': periods,
                         'dimension': table_selection['row_dimension'],
                         'members': table_selection['row_values'],
                         'metric': metric, 'columns': table_selection['columns'],
                         'column_policy': 'explicit_columns'},
            'per_source_aggregate': table_selection['operation'],
            'final_operations': ['sum_columns'] if table_selection['operation'] == 'sum' else [],
        }
        if table_selection['operation'] == 'sum':
            policy = 'explicit_selected_columns'
        rewritten_question = question
    if unsupported_operations:
        table_program = None

    queries = [original_question]
    if rewritten_question != question:
        queries.append(rewritten_question)
    if metric:
        queries.extend(" ".join(part for part in (scope, metric) if part) for scope in (areas or ([area] if area else [])))
    if metric and aggregation:
        queries.extend(" ".join(part for part in (scope, metric, "合计", "总计") if part) for scope in (areas or ([area] if area else [])))
    if source_titles and metric:
        queries.extend(f"《{title}》 {metric}" for title in source_titles)

    evidence: list[str] = []
    clarifications: list[str] = []
    if intent == "table_calculation":
        evidence.extend(["可定位到原始表格的行、列、工作表", "每个参与计算的原始数值"])
    if aggregation and not table_selection:
        evidence.append("原表中真实存在的“合计”或“总计”行、列、字段；若不存在，须明确可相加的子项范围")
    if table_selection:
        evidence.extend(f'{row} / {column}：同一证据组中的表头、原始单元格和单位'
                        for row in table_selection['row_values'] for column in table_selection['columns'])
    if "增长率" in operations:
        evidence.append("同一统计口径下的本期值和上期值")
    overview_metrics = insurance_operating_overview_metrics(question)
    if overview_metrics and len(periods) > 1:
        evidence.extend(
            f"{period}：{metric}的指标行和数值"
            for period in periods
            for metric in overview_metrics
        )
    if operations and not periods and len(source_titles) < 2:
        clarifications.append("未给出两个可比较的统计期间；请指定本期和上期，或指定两份对应统计表。")
    if operations and not metric:
        clarifications.append("未能识别统计指标；请补充原表中的指标名称。")
    if unsupported_operations:
        clarifications.append(
            f"识别到当前本地计算器尚不支持的运算：{'、'.join(unsupported_operations)}；请明确计算口径或启用受限规划器。"
        )

    return QuestionPlan(
        original_question=original_question,
        intent=intent,
        area=area,
        areas=areas,
        entity=entity,
        metric=metric,
        aggregation=aggregation,
        aggregation_policy=policy,
        operations=operations,
        periods=periods,
        source_titles=source_titles,
        source_title_provenance=source_title_provenance,
        rewritten_question=rewritten_question,
        table_program=table_program,
        unsupported_operations=unsupported_operations,
        financial_aliases=matched_financial_terms(question),
        retrieval_queries=_unique(queries),
        required_evidence=_unique(evidence),
        clarification_needed=clarifications,
        resolved_conditions={
            "relative_time_original": next((value for value in ("今年", "去年") if value in original_question), ""),
            "year": int(re.match(r"(20\d{2})年", periods[0]).group(1)) if len(periods) == 1 and re.match(r"(20\d{2})年", periods[0]) else None,
            "periods": periods,
            "region": areas or ([area] if area else []),
            "institution": [entity] if entity else [],
            "metric": [metric] if metric and metric_binding["hard"] else [],
            "resolution_source": "deterministic_parser",
            "locked": True,
            **({'table_selection': table_selection} if table_selection else {}),
        },
        area_canonicalizations=area_canonicalizations,
        table_selection=table_selection,
        metric_binding=metric_binding,
    )
