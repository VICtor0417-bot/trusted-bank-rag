"""Resolve discovery block lineage to unchanged original Corpus records."""
from collections import defaultdict
import re
import unicodedata

from .source_local_lexical import rank_source_local


def row_key(meta):
    table = meta.get('table') or meta.get('table_index')
    if table is None:
        identity = str(meta.get('table_id') or '')
        table = identity.rsplit(':', 1)[-1] if identity else ''
    row = meta.get('row') or meta.get('row_start') or ''
    return f"{meta.get('sheet') or ''}|{table}|{row}"


class CatalogRowResolver:
    def __init__(self, documents_by_title):
        self.documents_by_title = documents_by_title
        self.indexes = {}
        self.nodes = {}
        for records in documents_by_title.values():
            for document in records:
                identity = str(document.id)
                if identity in self.nodes and self.nodes[identity] is not document:
                    previous = self.nodes[identity]
                    if previous.content != document.content or previous.meta != document.meta:
                        raise ValueError(f"Conflicting evidence node ID: {identity}")
                self.nodes[identity] = document

    def bind_catalogs(self, catalogs):
        """Build explicit references from document lineage, never from queries.

        Attach references after catalog ID construction: retrieval text, vector
        input fields and established IDs remain unchanged. Original Corpus
        documents are not modified. Full records, including their headers and
        qualifications, are the referenced answer evidence.
        """
        for catalog in catalogs:
            meta = catalog.meta
            title = str(meta.get('source_title') or '')
            index = self._index(title)
            if meta.get('catalog_representation') == 'structural_block':
                keys = list(dict.fromkeys(meta.get('row_keys') or []))
            else:
                keys = sorted(index)
            records = [d for key in keys for d in index.get(key, [])
                       if meta.get('source_path') and d.meta.get('source_path') == meta['source_path']]
            meta['reference_ids'] = list(dict.fromkeys(str(d.id) for d in records))
            meta['reference_schema'] = 'original_evidence_ids_v1'
            self.nodes[str(catalog.id)] = catalog

    def _resolve_references(self, root):
        """Follow explicit node links with source checks and cycle detection."""
        found, issues, edges, seen = [], [], [], set()
        source = (root.meta.get('source_title'), root.meta.get('source_path'))

        def visit(identity, path):
            identity = str(identity)
            if identity in path:
                issues.append({'id': identity, 'reason': 'reference_cycle'})
                return
            if identity in seen:
                return
            node = self.nodes.get(identity)
            if node is None:
                issues.append({'id': identity, 'reason': 'missing_reference'})
                return
            if not all(source) or (node.meta.get('source_title'), node.meta.get('source_path')) != source:
                issues.append({'id': identity, 'reason': 'source_mismatch'})
                return
            if node.meta.get('parse_error'):
                issues.append({'id': identity, 'reason': 'parse_error'})
                return
            edges.append({'from': path[-1], 'to': identity})
            seen.add(identity)
            if 'reference_ids' in node.meta:
                if len(path) >= 32:
                    issues.append({'id': identity, 'reason': 'reference_depth_limit'})
                    return
                for child in node.meta['reference_ids']:
                    visit(child, path + [identity])
            elif node.meta.get('evidence_type') == 'table_source_catalog':
                issues.append({'id': identity, 'reason': 'unresolved_catalog'})
            else:
                found.append(node)

        for identity in root.meta['reference_ids']:
            visit(identity, [str(root.id)])
        return found, issues, edges

    def _index(self, title):
        if title not in self.indexes:
            index = defaultdict(list)
            for document in self.documents_by_title.get(title, []):
                if document.meta.get('evidence_type') in {
                    'table_row', 'table_row_group', 'table_evidence_group', 'table_cell'
                } and not document.meta.get('parse_error'):
                    index[row_key(document.meta)].append(document)
            self.indexes[title] = index
        return self.indexes[title]

    def _referenced_table_records(self, document):
        """Resolve explicit table numbers in a clause to its own DOCX tables.

        A narrative source hit is not a table hit. Only an actual table reference
        in the retrieved text, an existing table_index and identical source
        lineage authorize expansion. Never infer a table from the question.
        """
        numbers = list(dict.fromkeys(int(n) for n in re.findall(
            r"(?:表格|表)\s*([1-9]\d*)(?![\d.．\-－—])",
            unicodedata.normalize('NFKC', document.content or ''),
        )))
        meta = document.meta
        if not numbers or not meta.get('source_title') or not meta.get('source_path'):
            return [], []
        records = [
            d for group in self._index(meta['source_title']).values() for d in group
            if d.meta.get('source_path') == meta['source_path']
            and d.meta.get('file_type') == 'docx'
            and d.meta.get('table_index') in numbers
        ]
        return records, numbers

    def expand(self, documents, question):
        """One ranked seed per CE candidate, then tails; no Gold or guessed IDs.

        Original objects/IDs/content are returned. A block score is not a row
        score and is deliberately never copied onto the resolved records.
        Unresolvable structural catalogs are excluded from answer candidates.
        """
        groups, trace, narrative_groups = [], [], []
        for parent_rank, document in enumerate(documents, 1):
            meta = document.meta
            explicit = 'reference_ids' in meta
            if not explicit and (meta.get('evidence_type') != 'table_source_catalog' or meta.get('catalog_representation') != 'structural_block'):
                groups.append([document])
                if (parent_rank <= 3 and float(document.score or 0) >= 0.80
                        and meta.get('evidence_type') in {'clause', 'page', 'legacy_word_clause'}):
                    found, numbers = self._referenced_table_records(document)
                    if found:
                        ranked, detail = rank_source_local(found, question, lambda d: (0,))
                        # Expansion adds candidates without moving the original
                        # CE seeds out of the pre-existing selection window.
                        narrative_groups.append(ranked)
                        trace.append({
                            'catalog_id': str(document.id),
                            'source_title': meta['source_title'],
                            'source_path': meta['source_path'],
                            'row_keys': list(dict.fromkeys(row_key(d.meta) for d in found)),
                            'missing_row_keys': [],
                            'resolved_ids': [str(d.id) for d in ranked],
                            'reference_ids': [str(d.id) for d in found],
                            'reference_issues': [], 'reference_edges': [],
                            'resolution_method': 'same_source_explicit_table_reference',
                            'referenced_tables': numbers,
                            'ranking': detail, 'row_score_inherited': False,
                        })
                continue
            title = str(meta.get('source_title') or '')
            keys = list(dict.fromkeys(meta.get('row_keys') or []))
            index = self._index(title)
            found, missing = [], []
            for key in keys if not explicit else []:
                # Match the exact producer locator AND source path. No fuzzy
                # title, neighboring row, cell-text inference or row-only join.
                records = [d for d in index.get(key, [])
                           if d.meta.get('source_path') == meta.get('source_path')]
                if records:
                    found.extend(records)
                else:
                    missing.append(key)
            issues, edges = [], []
            if explicit:
                found, issues, edges = self._resolve_references(document)
            ranked, detail = rank_source_local(found, question, lambda d: (0,))
            groups.append(ranked)
            trace.append({'catalog_id': str(document.id), 'source_title': title,
                          'source_path': meta.get('source_path'),
                          'row_keys': keys, 'missing_row_keys': missing,
                          'resolved_ids': [str(d.id) for d in ranked],
                          'reference_ids': meta.get('reference_ids'),
                          'reference_issues': issues, 'reference_edges': edges,
                          'resolution_method': 'explicit_node_reference' if explicit else 'legacy_row_locator',
                          'ranking': detail, 'row_score_inherited': False})
        groups.extend(narrative_groups)
        ordered = [g[0] for g in groups if g]
        ordered += [d for g in groups for d in g[1:]]
        result, seen = [], set()
        for document in ordered:
            if str(document.id) not in seen:
                seen.add(str(document.id))
                result.append(document)
        return result, trace
