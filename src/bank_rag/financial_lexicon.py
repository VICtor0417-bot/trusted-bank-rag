"""Conservative Chinese finance terminology aliases for retrieval expansion.

Entries are deliberately *aliases*, not merely related concepts.  For
example, ``原保险保费收入`` is not equated with ``签单保费``: doing so would
increase recall at the cost of a financially incorrect table lookup.
"""

from __future__ import annotations

import re


# Canonical term -> equivalent names, abbreviations and common regulator/table
# spellings.  The groups cover banking, insurance, trust, leasing, securities,
# payments, risk, accounting and regulatory reporting.  Additions should keep
# the same strict-equivalence standard.
FINANCIAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    # Banking balance sheet, deposits and lending
    "资产总额": ("总资产", "资产合计", "资产总计", "资产规模"),
    "负债总额": ("总负债", "负债合计", "负债总计", "负债规模"),
    "所有者权益": ("股东权益", "权益总额", "净资产", "所有者权益合计"),
    "存款余额": ("各项存款余额", "存款总额", "存款规模"),
    "贷款余额": ("各项贷款余额", "贷款总额", "贷款规模"),
    "人民币贷款余额": ("本外币贷款中的人民币贷款", "人民币各项贷款余额"),
    "人民币存款余额": ("本外币存款中的人民币存款", "人民币各项存款余额"),
    "个人贷款": ("个人贷款余额", "零售贷款", "个人信贷"),
    "公司贷款": ("企业贷款", "对公贷款", "法人贷款"),
    "普惠型小微企业贷款": ("普惠小微贷款", "普惠型小微贷款", "小微企业普惠贷款"),
    "涉农贷款": ("农户贷款和农村企业贷款", "农业相关贷款"),
    "绿色贷款": ("绿色信贷", "绿色信贷余额"),
    "房地产贷款": ("房贷", "不动产贷款", "房地产信贷"),
    "个人住房贷款": ("住房按揭贷款", "个人按揭贷款", "住房贷款"),
    "消费贷款": ("个人消费贷款", "消费信贷", "消费类贷款"),
    "信用卡贷款": ("信用卡透支余额", "信用卡应收账款"),
    "票据融资": ("票据贴现", "票据业务融资"),
    "贷款核销": ("呆账核销", "不良贷款核销"),
    "贷款损失准备": ("贷款拨备", "信贷减值准备", "贷款减值准备"),
    "贷款拨备率": ("拨备率", "贷款损失准备率"),
    "拨备覆盖率": ("贷款损失准备覆盖率", "不良贷款拨备覆盖率"),
    "不良贷款余额": ("不良贷款", "不良信贷余额"),
    "不良贷款率": ("不良率", "不良贷款占比"),
    "关注类贷款": ("关注贷款",),
    "正常类贷款": ("正常贷款",),
    "次级类贷款": ("次级贷款",),
    "可疑类贷款": ("可疑贷款",),
    "损失类贷款": ("损失贷款",),
    "流动性比例": ("流动性比率",),
    "流动性覆盖率": ("LCR",),
    "净稳定资金比例": ("NSFR",),
    "存贷比": ("贷款存款比", "贷存比"),
    "贷款集中度": ("授信集中度", "信贷集中度"),
    "单一客户贷款集中度": ("单一客户授信集中度",),
    "最大十家客户贷款集中度": ("最大十户贷款集中度", "前十大客户贷款集中度"),
    # Bank capital, risk and prudential regulation
    "资本充足率": ("资本充足率指标", "CAR"),
    "核心一级资本充足率": ("核心一级资本率", "CET1资本充足率", "CET1比率"),
    "一级资本充足率": ("一级资本率", "Tier 1资本充足率"),
    "核心一级资本": ("CET1资本",),
    "一级资本": ("Tier 1资本",),
    "二级资本": ("Tier 2资本",),
    "资本净额": ("监管资本净额",),
    "风险加权资产": ("加权风险资产", "RWA"),
    "信用风险加权资产": ("信用风险RWA",),
    "市场风险加权资产": ("市场风险RWA",),
    "操作风险加权资产": ("操作风险RWA",),
    "杠杆率": ("监管杠杆率",),
    "杠杆率分母": ("调整后的表内外资产余额", "杠杆率计量总资产"),
    "逆周期资本缓冲": ("逆周期资本缓冲要求",),
    "系统重要性银行附加资本": ("系统重要性附加资本", "SIB附加资本"),
    "内部资本充足评估程序": ("ICAAP",),
    "内部流动性充足评估程序": ("ILAAP",),
    "压力测试": ("情景压力测试",),
    "信用风险": ("信贷风险",),
    "市场风险": ("市场价格风险",),
    "操作风险": ("运营风险",),
    "流动性风险": ("资金流动性风险",),
    "利率风险": ("银行账簿利率风险", "IRRBB"),
    "声誉风险": ("信誉风险",),
    "集中度风险": ("风险集中度",),
    "国别风险": ("国家风险",),
    "合规风险": ("监管合规风险",),
    "洗钱风险": ("反洗钱风险",),
    # Insurance business and solvency
    "原保险保费收入": ("原保险保费", "原保费收入", "原保费"),
    "原保险赔付支出": ("原保险赔款支出", "原保险赔款", "赔付支出"),
    "赔款支出": (),
    "人身保险": ("人身险",),
    "财产保险": ("财产险", "财险"),
    "人寿保险": ("寿险", "人寿险", "寿险收入"),
    "健康保险": ("健康险",),
    "意外伤害保险": ("意外险", "意外伤害险"),
    "机动车辆保险": ("车险", "机动车保险"),
    "责任保险": ("责任险",),
    "保证保险": ("保证险",),
    "农业保险": ("农险",),
    "再保险": ("分保",),
    "再保险公司": ("再保公司",),
    "保险资产管理公司": ("保险资管公司", "保险资产管理机构"),
    "保险金额": ("保额",),
    "签单数量": ("保单件数", "签单件数"),
    "保户投资款新增交费": ("保户投资款",),
    "投连险独立账户新增交费": ("投连险新增交费", "投资连结保险新增交费"),
    "综合偿付能力充足率": ("综合偿付能力率", "综合偿付能力比率"),
    "核心偿付能力充足率": ("核心偿付能力率", "核心偿付能力比率"),
    "实际资本": ("认可资本",),
    "资金运用余额": (),
    "最低资本": ("最低资本要求",),
    "偿付能力报告": ("偿二代报告", "偿付能力监管报告"),
    "偿付能力风险管理": ("SARMRA", "偿付能力风险管理能力"),
    "保险风险": ("承保风险",),
    "巨灾风险": ("巨灾损失风险",),
    "准备金": ("保险准备金", "责任准备金"),
    "未到期责任准备金": ("未赚保费准备金",),
    "未决赔款准备金": ("未决赔款",),
    "寿险责任准备金": ("人身险责任准备金",),
    "退保率": ("解约率",),
    "赔付率": ("赔款率",),
    "综合成本率": ("综合费用率和赔付率",),
    # Trust, leasing and consumer finance
    "信托公司": ("信托机构",),
    "信托业务": ("信托服务",),
    "资产服务信托": ("资产服务类信托",),
    "资产管理信托": ("资管信托",),
    "公益慈善信托": ("公益信托", "慈善信托"),
    "财富管理服务信托": ("财富管理信托",),
    "家族信托": ("家庭信托",),
    "财产权信托": ("财产信托",),
    "融资租赁公司": ("租赁公司", "金融租赁公司"),
    "融资租赁资产": ("租赁资产",),
    "应收融资租赁款": ("融资租赁应收款",),
    "消费金融公司": ("消金公司",),
    "汽车金融公司": ("汽金公司",),
    "金融资产管理公司": ("AMC", "不良资产管理公司"),
    # Markets, investments, payments and foreign exchange
    "债券投资": ("债券持仓", "债券资产"),
    "证券投资基金": ("基金投资", "公募基金投资"),
    "股票投资": ("权益投资", "股票资产"),
    "长期股权投资": ("长期权益投资",),
    "衍生品": ("金融衍生品", "衍生金融工具"),
    "资产证券化": ("ABS", "证券化业务"),
    "同业拆借": ("拆借", "银行间拆借"),
    "同业存单": ("NCD",),
    "回购": ("债券回购", "质押式回购"),
    "货币经纪公司": ("货币经纪机构",),
    "支付机构": ("非银行支付机构", "第三方支付机构"),
    "银行卡收单": ("收单业务",),
    "跨境支付": ("跨境资金收付",),
    "外汇风险": ("汇率风险",),
    "外汇敞口": ("外汇风险敞口",),
    # Accounting, reporting and common analytical language
    "监管报表": ("监管统计报表", "监管统计表"),
    "非现场监管": ("非现场检查", "非现场监测"),
    "现场检查": ("现场监管检查",),
    "年度审计报告": ("年审报告", "年度审计"),
    "关联交易": ("关联方交易",),
    "重大风险和损失": ("重大风险损失",),
    "资产负债表": ("资产负债情况表",),
    "利润表": ("损益表",),
    "现金流量表": ("现金流表",),
    "营业收入": ("经营收入",),
    "营业支出": ("经营支出",),
    "净利润": ("税后利润", "净收益"),
    "营业利润": ("经营利润",),
    "同比增长率": ("同比增幅", "同比增速", "同比增长"),
    "环比增长率": ("环比增幅", "环比增速", "环比增长"),
    "增加额": ("增量", "增加金额", "新增额"),
    "减少额": ("减量", "减少金额", "下降额"),
    "平均值": ("均值", "平均数"),
    "合计": ("总计", "总额", "合计数"),
}


def _compact(value: str) -> str:
    return re.sub(r"[\s_\-—（）()【】\[\]《》'\"，,。.:：/]+", "", value).lower()


_ALIASES: dict[str, str] = {}
for _canonical, _values in FINANCIAL_SYNONYMS.items():
    for _term in (_canonical, *_values):
        _ALIASES[_compact(_term)] = _canonical


def canonical_financial_term(value: str) -> str:
    """Return a canonical label only for an exact known alias."""
    return _ALIASES.get(_compact(value), value)


def matched_financial_terms(text: str, max_terms: int = 6) -> list[dict[str, object]]:
    """Find the most specific alias groups mentioned in free-form text."""
    compact_text = _compact(text)
    found: list[tuple[int, str, str]] = []
    for alias, canonical in _ALIASES.items():
        if len(alias) >= 2 and alias in compact_text:
            found.append((len(alias), canonical, alias))
    found.sort(key=lambda item: item[0], reverse=True)
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for _length, canonical, alias in found:
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append({"canonical": canonical, "matched_alias": alias, "aliases": list(FINANCIAL_SYNONYMS[canonical])})
        if len(result) >= max_terms:
            break
    return result


def expand_financial_query(text: str, max_terms: int = 4, aliases_per_term: int = 5) -> str | None:
    """Create an additive query; never replace the user's original wording."""
    terms = matched_financial_terms(text, max_terms=max_terms)
    if not terms:
        return None
    groups = []
    for item in terms:
        canonical = str(item["canonical"])
        aliases = list(item["aliases"])[:aliases_per_term]
        groups.append("/".join([canonical, *aliases]))
    return f"{text}\n金融同义字段扩展：{'；'.join(groups)}"


def metric_variants(value: str) -> tuple[str, ...]:
    """Return exact equivalent spellings for table-column comparison."""
    canonical = canonical_financial_term(value)
    if canonical not in FINANCIAL_SYNONYMS:
        return (value,)
    return (canonical, *FINANCIAL_SYNONYMS[canonical])
