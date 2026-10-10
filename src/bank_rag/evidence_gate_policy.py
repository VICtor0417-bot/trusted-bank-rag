"""Gate-only temporal, bound-fact and quote-grounded case-condition contracts.

No retrieval queries, source IDs, ranking parameters or Gold enter this module.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .canonicalization import area_match_key
from .period_binding import document_covers_period
from .table_selection import bound_fact


def statistical_lookup(plan: Any) -> bool:
    question = plan.original_question
    titles = " ".join(plan.source_titles)
    if plan.table_selection:
        return True
    statistical_title = bool(re.search(r"情况表|统计表|资产.*负债.*(?:月度|季度)|经营数据", titles))
    regulatory_title = bool(re.search(r"办法|规则|条例|要求|标准|目录|通知|指引", titles))
    if regulatory_title and not statistical_title:
        return False
    # Formula definitions and policy conditions are not observations over time.
    if re.search(r"定义|公式|怎么算|如何计算|计算口径|计量方式", question):
        return False
    return bool(statistical_title or (
        (plan.metric_binding or {}).get('hard')
        and re.search(r"是多少|有多少|为多少|分别多少|变化|增长|差额|总额|余额|本年累计|季度末|月末", question)
    ))


def temporal_semantics(plan: Any) -> dict:
    statistical = statistical_lookup(plan)
    roles = []
    for period in plan.periods:
        year = re.match(r"(20\d{2})年", period)
        if year and re.search(re.escape(year.group(1)) + r"年(?:版|修订|发布|颁布)", plan.original_question):
            role = 'REGULATION_VERSION'
        elif year and re.search(re.escape(year.group(1)) + r"年.{0,12}(?:生效|施行|实施之日)", plan.original_question):
            role = 'EFFECTIVE_DATE'
        elif statistical:
            role = 'STATISTICAL_PERIOD'
        else:
            role = 'SCENARIO_OR_REPORT_DATE'
        roles.append({'value': period, 'role': role})
    return {'statistical_lookup': statistical, 'roles': roles,
            'statistical_periods': [r['value'] for r in roles if r['role'] == 'STATISTICAL_PERIOD']}


def unsupported_requested_operation(plan: Any) -> bool:
    if not plan.unsupported_operations:
        return False
    if plan.intent == 'table_calculation':
        return True
    if re.search(r'(?:请|帮我)(?:直接)?计算', plan.original_question):
        return True
    # Descriptions such as "预计未来能正常还款" are premises, not instructions
    # to run a forecasting calculator. Explicit requests remain unsupported.
    return bool(re.search(r"(?:请|帮我|能否|能不能).{0,12}(?:预测|估算未来|趋势外推)|(?:预测|预计).{0,12}(?:是多少|会是多少|增长多少)", plan.original_question))


def numeric_slot_fact(document: Any, slot: dict) -> dict | None:
    """Bind a value to its actual header coordinate, row, scope, period and unit."""
    if not document_covers_period(str(slot.get('period') or ''), document) and slot.get('period'):
        return None
    stored = str(document.meta.get('row_label') or '')
    requested = str(slot.get('row_value') or '')
    if slot.get('row_dimension') == 'region':
        scope_key = lambda value: area_match_key(re.sub(r'(?:合计|总计)$', '', re.sub(r'\s+', '', value)))
        if scope_key(stored) != scope_key(requested):
            return None
    elif stored != requested:
        return None
    fact = bound_fact(document, stored, str(slot.get('column') or ''))
    if fact is None:
        return None
    return {**fact, 'value': str(fact['value']), 'period': slot.get('period'),
            'requested_scope': requested, 'actual_row_label': stored,
            'evidence_id': str(document.id) if getattr(document, 'id', None) is not None else None,
            'table_id': document.meta.get('table_id'),
            'row': document.meta.get('row'), 'source_title': document.meta.get('source_title')}


def needs_case_contract(question: str, documents: list[Any]) -> bool:
    regulatory = any(str(d.meta.get('file_type')) in {'doc', 'docx', 'pdf'}
                     and not d.meta.get('parse_error') for d in documents)
    return bool(regulatory and re.search(r"是否|能否|能确认|满足|符合|归入|最低门槛|现在可以|应如何|应当.*吗", question))


CASE_CONTRACT = """
【个案条件门控：必须输出 JSON，不输出分析过程】
结构：{"action":"ANSWER|CLARIFY|CONDITIONAL_ANSWER|REFUSE", "conditions":[
{"requirement":"完成用户本次判断所必需的条件", "status":"SATISFIED|UNSATISFIED|UNKNOWN",
"user_quote":"原问题中逐字支持该个案事实的短引文；未知则为空",
"rule_quote":"证据中逐字规定该条件的短引文", "evidence_ref":"E1"}],
"answer":"完整最终答案，结论先行，保留条件分支、单位和[E编号]引用"}
只有本次用户要求真正依赖的条件才进入conditions；不能扩展成对机构全部资质的审查。
SATISFIED：用户明确给出满足该必要条件的事实；UNSATISFIED：明确给出相反事实。
UNKNOWN：用户没有说明，或只说提交材料/提出申请/应当满足/如合同约定/拟采用。
没有说已满足不等于未满足，也不等于已满足。法规要求成立不证明本行实际上已成立。
情景给出的数值或已完成时长，应与证据阈值比较；不要把这种比较当作预测。
题面特征同时适合多个产品/类别，不能因为某一类更像就擅自选定；若类别改变门槛，类别为UNKNOWN。
条件未知但规则已找到时，优先CONDITIONAL_ANSWER，列出会改变结论的分支或询问具体条件。
仅凭规范无法确认个案实际结果时，明确无法确认该事实并说明所需材料；不要宣称全库无答案。
若用户只询问规则、公式或条件清单，而非个案结果，可ANSWER且conditions为空。
法规版本、生效时间、报告期限不是统计期间；不要求规则证据重复版本年份。
引用只用提供的E编号。最多800中文字符。不要先输出无条件结论，再以条件未知作尾注。
"""


def decode_case_contract(text: str, question: str, documents: list[Any]) -> dict:
    payload = json.loads(text)
    if payload.get('action') not in {'ANSWER', 'CLARIFY', 'CONDITIONAL_ANSWER', 'REFUSE'}:
        raise ValueError('Invalid case action')
    if not isinstance(payload.get('conditions'), list) or not isinstance(payload.get('answer'), str) or not payload['answer'].strip():
        raise ValueError('Invalid case condition contract')
    norm = lambda s: re.sub(r'\s+', '', unicodedata.normalize('NFKC', s))
    ledger = []
    for item in payload['conditions']:
        if not isinstance(item, dict) or not item.get('requirement') or item.get('status') not in {'SATISFIED', 'UNSATISFIED', 'UNKNOWN'}:
            raise ValueError('Invalid condition state')
        entry = dict(item)
        ref = re.fullmatch(r'\[?E(\d+)\]?', str(item.get('evidence_ref') or ''))
        doc = documents[int(ref.group(1)) - 1] if ref and 0 < int(ref.group(1)) <= len(documents) else None
        rule = str(item.get('rule_quote') or '')
        user = str(item.get('user_quote') or '')
        entry['rule_quote_verified'] = bool(doc and len(norm(rule)) >= 4 and norm(rule) in norm(str(doc.content or '')))
        entry['user_quote_verified'] = bool(len(norm(user)) >= 3 and norm(user) in norm(question))
        entry['user_assertion_verified'] = user_quote_is_assertion(question, user)
        entry['proposed_status'] = entry['status']
        if not entry['rule_quote_verified'] or (entry['status'] != 'UNKNOWN' and not entry['user_assertion_verified']):
            entry['status'] = 'UNKNOWN'
            entry['contract_warning'] = 'Condition cannot be asserted without rule and actual user fact'
        ledger.append(entry)
    payload['conditions'] = ledger
    unknown = [c for c in ledger if c['status'] == 'UNKNOWN']
    invalid_conditional = payload['action'] == 'CONDITIONAL_ANSWER' and any(c.get('contract_warning') for c in ledger)
    if unknown and (payload['action'] == 'ANSWER' or invalid_conditional):
        # A structured action cannot overrule unknown factual prerequisites.
        payload['action'] = 'CLARIFY'
        # Prefer genuinely unspecified premises over an already-stated fact
        # whose rule quotation could not be verified. Neither permits trusting
        # an unrestricted answer that has failed its own condition contract.
        missing_facts = [c for c in unknown if not c['user_assertion_verified']]
        needed = missing_facts or unknown
        payload['answer'] = '不能仅凭当前信息作出确定结论。需要确认：' + '；'.join(c['requirement'] for c in needed) + '。现有规则要求不能替代该个案事实，未说明不代表已满足或未满足。'
        verified_rules = [c for c in needed if c['rule_quote_verified']]
        if verified_rules:
            payload['answer'] += '\n已核验的相关要求：' + '；'.join('“' + c['rule_quote'] + '”[' + str(c['evidence_ref']).strip('[]') + ']' for c in verified_rules)
    payload['unknown_count'] = len(unknown)
    return payload


def user_quote_is_assertion(question: str, quote: str) -> bool:
    """Literal overlap is insufficient when the quote is inside an if/query."""
    norm = lambda s: re.sub(r'\s+', '', unicodedata.normalize('NFKC', s))
    q, value = norm(question), norm(quote)
    if len(value) < 3:
        return False
    for match in re.finditer(re.escape(value), q):
        before = q[:match.start()]
        boundary = max((before.rfind(mark) for mark in '。；;！？!?\n'), default=-1)
        prefix = before[boundary+1:]
        # Preserve the grammatical scope before the selected short quotation;
        # the model cannot remove "if" and thereby manufacture a fact.
        if re.search(r'如果|假如|倘若|假设|若(?!干)|(?:^|[，,：:（(])如(?!何|实|期|数|约)|是否|能否|能确认|可否', prefix):
            continue
        if re.match(r'如果|假如|倘若|假设|若(?!干)|如(?!何|实|期|数|约)', value):
            continue
        if re.search(r'(?:应当|必须|需要)$', prefix) or re.match(r'应当|必须|需要', value):
            continue
        return True
    return False
