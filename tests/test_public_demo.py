import json
from pathlib import Path
from bank_rag.graph import build_rag_graph
ROOT=Path(__file__).resolve().parents[1]
def test_workflow_compiles():
    assert build_rag_graph(object())
def test_demo_config_and_queries():
    cfg=json.loads((ROOT/'configs/demo.json').read_text(encoding='utf-8'))
    assert cfg['retrieval_mode']=='bm25'
    assert len(json.loads((ROOT/'data/demo/queries.json').read_text(encoding='utf-8')))==4
