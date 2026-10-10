"""Controlled vocabulary for auditable table-calculation requests."""

from __future__ import annotations


# Rules here map to operations implemented by the local executor. Ordering is
# deliberate: comparison and percentage-point phrases precede shorter terms.
SUPPORTED_OPERATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("比较", r"(?:比较|对比|判断哪.+(?:更高|更低|较高|较低|更多|更少)|哪.+(?:更高|更低|较高|较低|更多|更少)|多出(?:多少|的?(?:金额|数额))|少(?:了|于)多少|相差多少|排名|排序)"),
    ("百分点变化", r"(?:变化(?:约)?多少个百分点|(?:提高|下降|上升|降低).{0,8}个百分点|百分点)"),
    ("增长率", r"(?:增长率|增幅|增长百分比|同比(?:增长)?|环比(?:增长)?|增速)"),
    # Bare ``新增`` often means "newly added requirements/items" in a
    # regulation question.  Treat it as arithmetic only when the wording asks
    # for an amount or number.
    ("增加额", r"(?:增加额|增加金额|增量|增加(?:了)?多少|新增(?:金额|数额|多少)|增长(?:金额|数额))"),
    ("减少额", r"(?:减少额|减少金额|减量|减少了多少|下降(?:金额|数额)?|缩减(?:金额|数额)?)"),
    ("差额", r"(?:差额|差值|相差(?:金额|数额)?|净增|净减)"),
    ("平均值", r"(?:算术平均值|平均值|平均数|均值|月均|季均)"),
    ("最大值", r"(?:最大值|最高值|最多|最大|峰值)"),
    ("最小值", r"(?:最小值|最低值|最少|最小|谷值)"),
    # ``汇总表`` is a document type, not an instruction to calculate a sum.
    # Bare “合并” often describes a regulatory-document treatment (for
    # example “合并的监管报告事项”), not arithmetic.  It is a sum only when
    # paired with an explicit calculation verb or a numeric-object phrase.
    ("求和", r"(?:求和|相加|加总|合并计算|汇总(?!表)|合并(?:汇总|相加|加总|后(?:计算|求和|相减)|的?(?=(?:金额|数值|保费|收入)))|总和|合计为多少|合起来)"),
)

# Recognized but deliberately withheld from the simple executor: each needs a
# denominator, weight, sample range, or statistical method made explicit.
UNSUPPORTED_OPERATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("加权平均", r"(?:加权平均|权重平均)"),
    ("中位数", r"(?:中位数|中位)"),
    ("方差标准差", r"(?:方差|标准差|波动率)"),
    ("占比", r"(?:占比|比重|市场份额|构成比)"),
    ("贡献率", r"(?:贡献率|贡献度)"),
    ("复合增长率", r"(?:复合增长率|CAGR|年复合增长)"),
    ("预测", r"(?:预测|预计|估算未来|趋势外推)"),
)


OPERATION_CATALOG: dict[str, str] = {
    "求和": "对明确成员的同口径数值求和",
    "增加额": "本期减上期",
    "减少额": "上期减本期",
    "差额": "两个明确数值相减",
    "增长率": "(本期-上期)/上期×100%",
    "百分点变化": "两个百分比直接相减，结果单位为百分点",
    "平均值": "明确数值的算术平均",
    "比较": "先计算每个对象的目标值，再排序并计算差距",
    "最大值": "对明确候选值取最大值",
    "最小值": "对明确候选值取最小值",
}
