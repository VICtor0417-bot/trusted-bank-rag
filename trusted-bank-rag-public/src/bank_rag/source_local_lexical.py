"""Source-local lexical ranking of complete evidence; no answer/Gold inputs."""
from collections import Counter
from functools import lru_cache
from math import log
import re


def _row_key(document):
    meta = document.meta
    if meta.get('row') is None or not meta.get('table_id') or meta.get('evidence_type') not in {'table_row', 'table_cell', 'table_evidence_group'}:
        return ('record', str(document.id))
    return (meta.get('doc_id') or meta.get('source_path'), meta['table_id'], meta.get('sheet'), meta['row'])


def _cells(document):
    row = document.meta.get('row')
    pairs = {coordinate: value.strip() for coordinate, value in re.findall(r'([A-Z]+\d+)=([^；;\n]*)', document.content or '')
             if re.search(r'\d+$', coordinate)[0] == str(row)}
    if not pairs and document.meta.get('evidence_type') == 'table_cell':
        coordinate = document.meta.get('cell') or ''
        if re.fullmatch(r'[A-Z]+\d+', coordinate) and '数值：' in (document.content or ''):
            pairs[coordinate] = document.content.rsplit('数值：', 1)[1].strip()
    return pairs


def complete_candidates(documents):
    """Remove duplicates only when a retained whole row contains identical cells.

    Headers/notes/clauses remain independent. Conflicting or additional cell
    values remain visible, even when they have the same physical row locator.
    """
    representatives = {}
    for d in documents:
        if d.meta.get('evidence_type') in {'table_evidence_group', 'table_row'} and _cells(d):
            key = _row_key(d)
            priority = (d.meta.get('evidence_type') == 'table_evidence_group', len(_cells(d)), len(d.content or ''))
            previous = representatives.get(key)
            if previous is None or priority > previous[0]:
                representatives[key] = (priority, d)
    kept, seen, removed = [], set(), []
    for d in documents:
        if str(d.id) in seen:
            continue
        seen.add(str(d.id))
        pair = representatives.get(_row_key(d))
        representative = pair[1] if pair else None
        own = _cells(d)
        if representative is not None and str(d.id) != str(representative.id) and own and all(_cells(representative).get(k) == v for k,v in own.items()):
            removed.append({'removed_id':str(d.id), 'retained_id':str(representative.id), 'proof':'same locator and identical coordinate/value coverage'})
        else:
            kept.append(d)
    return kept, removed


@lru_cache(maxsize=65536)
def _line_tokens(text):
    import jieba
    # Split script boundaries consistently in queries and evidence. Native
    # Jieba may keep a mixed identifier such as "X组" whole while tables write
    # "(X)". Preserve the identifier itself; do not invent aliases or facts.
    text=re.sub(r'([A-Za-z0-9])([\u4e00-\u9fff])',r'\1 \2',text)
    text=re.sub(r'([\u4e00-\u9fff])([A-Za-z0-9])',r'\1 \2',text)
    # Native search mode. No custom dictionary, synonym expansion or Gold terms.
    return tuple(t.lower() for t in jieba.cut_for_search(text) if re.search(r'[\w\u4e00-\u9fff]', t))


@lru_cache(maxsize=8192)
def _tokens(text):
    # Repeated headers and metadata are tokenized once, not once per cell.
    # Newlines are token boundaries already; token multiplicities are unchanged.
    return tuple(token for line in text.splitlines() for token in _line_tokens(line) if line.strip())


def prepare_source(documents):
    """Warm the immutable source index before serving queries."""
    candidates,_=complete_candidates(documents)
    _index(tuple(d.content or '' for d in candidates))


@lru_cache(maxsize=32)
def _index(texts):
    counts = [Counter(_tokens(t)) for t in texts]
    frequencies = Counter(t for c in counts for t in c)
    lengths = [sum(c.values()) for c in counts]
    return counts, frequencies, lengths, sum(lengths) / max(1,len(lengths))


def rank_source_local(documents, question, fallback_score, areas=()):
    candidates, removed = complete_candidates(documents)
    if not candidates:
        return [], {'candidate_count':0, 'removed_duplicates':removed}
    texts = tuple(d.content or '' for d in candidates)
    counts, frequencies, lengths, average = _index(texts)
    query = set(_tokens(question))
    scores = []
    for counts_i, length in zip(counts, lengths):
        total = 0.0
        for token in query:
            tf = counts_i.get(token,0)
            if tf:
                idf = log(1 + (len(candidates)-frequencies[token]+0.5)/(frequencies[token]+0.5))
                total += idf * tf * 2.5 / (tf + 1.5*(0.25+0.75*length/max(average,1)))
        scores.append(total)
    def area_key(value):
        return re.sub(r'(?:特别行政区|自治区|省|市)$','',re.sub(r'\s+','', value))
    requested = {area_key(a) for a in areas if a}
    identifiers={m.upper() for m in re.findall(r'(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*)\s*(?:组|类|档|级)',question)}
    def order(index):
        d = candidates[index]
        label = area_key(str(d.meta.get('row_label') or ''))
        area_match = bool(requested and (label in requested or ('全国' in requested and label=='全国合计')))
        explicit_values={v.strip().strip('()（）').upper() for v in _cells(d).values()}
        identifier_match=bool(identifiers & explicit_values)
        return (area_match, identifier_match, scores[index], fallback_score(d))
    # Stable content/metadata identity breaks equal-score ties independently of
    # process hash randomization. The old score only breaks lexical ties.
    indices = sorted(range(len(candidates)),key=lambda i:str(candidates[i].id))
    indices.sort(key=order,reverse=True)
    return [candidates[i] for i in indices], {
        'method':'native_jieba_search_bm25_okapi', 'k1':1.5, 'b':0.75,
        'candidate_count':len(candidates), 'removed_duplicates':removed,
        'ranking':[{'evidence_id':str(candidates[i].id),'rank':rank,'lexical_score':scores[i]} for rank,i in enumerate(indices,1)],
    }
