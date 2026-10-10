"""Prefer original-question row matches before source-local window truncation.

This selects existing evidence only. It does not infer facts, create hard slots,
or reinterpret parser conditions. No labels or source IDs are injected.
"""
import re


def _compact(text):
    return re.sub(r'\s+', '', str(text or ''))


def _label(text):
    return re.sub(r'^(?:其中[：:]?|\d+[.、．])', '', _compact(text))


def _periods(text):
    text = _compact(text)
    quarters = {'一': '1', '二': '2', '三': '3', '四': '4'}
    return {
        'year': set(re.findall(r'(20\d{2})年', text)),
        'quarter': {quarters.get(x, x) for x in re.findall(r'第?([一二三四1-4])季度', text)},
        'month': {str(int(x)) for x in re.findall(r'(\d{1,2})月', text)},
    }


def explicit_row_candidates(groups, question, periods):
    """Keep literal row labels; disambiguate only with recorded header cells.

    A period conflict disqualifies this *priority*, never removes a candidate
    from the ordinary ranking. Unknown period fields remain unknown. Among
    repeated metric rows prefer headers explicitly present in the question;
    unresolved ties are retained in existing lexical order, not guessed.
    """
    query = _compact(question)
    requested = _periods(' '.join(periods))
    selected, audit, seen = [], [], set()
    for group in groups:
        matches = {}
        for document in group:
            meta = document.meta
            if meta.get('evidence_type') != 'table_evidence_group':
                continue
            label = _label(meta.get('row_label'))
            if len(label) < 2 or not re.search(r'[\u4e00-\u9fff]', label) or label not in query:
                continue
            # Only recorded source metadata supplies file-level dates.
            source_period = _periods(meta.get('source_title', ''))
            if any(requested[k] and source_period[k] and not (requested[k] & source_period[k]) for k in requested):
                continue
            content = document.content or ''
            headers = content.split('表头：', 1)[-1].split('指标行：', 1)[0] if '表头：' in content else ''
            explicit_headers = sorted({
                value for raw in re.findall(r'[A-Z]+\d+=([^；;\n]*)', headers)
                if len(value := _label(raw)) >= 2
                and re.search(r'[\u4e00-\u9fff]', value)
                and value in query
                and not re.search(r'\d', value)
            })
            matches.setdefault(label, []).append((len(explicit_headers), document, explicit_headers))
        for label, options in matches.items():
            best = max(item[0] for item in options)
            for strength, document, headers in options:
                if strength == best and str(document.id) not in seen:
                    seen.add(str(document.id)); selected.append(document)
                    audit.append({'evidence_id': str(document.id), 'row_label': label,
                                  'explicit_header_matches': headers,
                                  'source_title': document.meta.get('source_title')})
    return selected, audit
