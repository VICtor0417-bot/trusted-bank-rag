"""Synthetic, offline tests: no real credentials, API calls or private fixtures."""
import json
import socket
import sys
from types import SimpleNamespace

import pytest

from bank_rag import graph, jev_gate


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setenv("RAG_ENABLE_JEV", "1")
    monkeypatch.setenv("RAG_ENABLE_LLM_PLANNER", "0")
    monkeypatch.setenv("RAG_ORCHESTRATOR", "langgraph")
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("HAYSTACK_TELEMETRY_ENABLED", "false")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    original_connect = socket.socket.connect
    def no_network(sock, address):
        # Windows asyncio implements its internal socketpair with loopback TCP.
        # Permit only that stdlib caller, not HTTP clients or arbitrary sockets.
        if (sys._getframe(1).f_code is getattr(socket.socketpair, "__code__", None)
                and isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}):
            return original_connect(sock, address)
        raise AssertionError("Offline tests must not connect to a network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)


def document(identity="e1", row=2, content="Synthetic archive evidence."):
    return SimpleNamespace(id=identity, content=content, meta={
        "source_title": "Fictional rules", "source_path": "demo/rules.docx",
        "evidence_type": "clause", "row": row, "section": "Archive",
    })


class Client:
    def __init__(self, action="answer", error=None):
        self.action, self.error, self.calls = action, error, []

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"answers": {"action": {"choice": self.action}}, "usage": {"input_tokens": 10}}


@pytest.mark.parametrize("rule", [False, True])
@pytest.mark.parametrize("action", ["answer", "needs_evidence", "needs_clarification", "unsupported"])
def test_or_policy(rule, action):
    decision = jev_gate.admission(rule, {"status": "ok", "action": action})
    assert decision["allow_to_answer"] == (rule or action == "answer")
    assert decision["rule_allow"] is rule


def test_disabled_and_missing_key_do_not_call(monkeypatch):
    client = Client()
    monkeypatch.setenv("RAG_ENABLE_JEV", "0")
    assert jev_gate.judge_evidence("question", {}, [document()], client=client)["status"] == "disabled"
    assert not client.calls
    monkeypatch.setenv("RAG_ENABLE_JEV", "1")
    assert jev_gate.judge_evidence("question", {}, [document()])["status"] == "missing_api_key"
    assert jev_gate.judge_evidence("question", {}, [], client=client)["status"] == "empty_evidence"
    assert not client.calls


def test_judgment_input_is_only_question_locked_conditions_and_evidence():
    plan = {"resolved_conditions": {"locked": True, "year": 2041}, "gold": "must not be sent"}
    client = Client()
    result = jev_gate.judge_evidence("synthetic question", plan, [document()], client=client)
    assert result["action"] == "answer"
    sent = client.calls[0]
    assert set(sent["state"]) == {"user_question", "resolved_conditions", "evidence"}
    assert "gold" not in sent["state"]
    assert sent["state"]["evidence"][0]["source"]["section"] == "Archive"
    assert set(sent["questions"]) == {"action", "evidence_sufficient", "missing_user_conditions"}
    sent["state"]["resolved_conditions"]["year"] = 0
    assert plan["resolved_conditions"]["year"] == 2041


@pytest.mark.parametrize("client", [Client(error=TimeoutError("sensitive body")), Client("invalid")])
def test_errors_never_add_permission_or_expose_exception_message(client):
    result = jev_gate.judge_evidence("question", {}, [document()], client=client)
    assert result["status"] == "error" and result["action"] is None
    assert "sensitive body" not in json.dumps(result)
    assert not jev_gate.admission(False, result)["allow_to_answer"]
    assert jev_gate.admission(True, result)["allow_to_answer"]


def test_dedup_preserves_coordinates_conflicts_and_original_objects():
    first = document()
    other_coordinate = document("e2", row=3)
    conflict = document(content="Changed synthetic representation")
    retained, audit = jev_gate.deduplicate_documents([first, document(), other_coordinate, conflict])
    assert retained == [first, other_coordinate, conflict]
    assert retained[0] is first
    assert audit["dropped_exact_duplicate_ids"] == ["e1"]
    assert audit["identity_conflicts"] == ["e1"]
    assert graph._merge_documents([first], [document(), other_coordinate, conflict]) == retained


class GraphRAG:
    retrieval_mode = "bm25"
    last_retrieval_debug = {}

    def __init__(self):
        self.retrieve_count = 0
        self.generate_count = 0
        self.advisory = None

    def _route_sources(self, *args, **kwargs):
        return []

    def retrieve(self, *args, **kwargs):
        self.retrieve_count += 1
        return [document()]

    def citations(self, docs):
        return [d.meta for d in docs]

    def rewrite_missing_evidence(self, *args):
        return {"queries": ["synthetic missing fact"], "method": "offline"}

    def _ask_pipeline(self, question, **kwargs):
        self.generate_count += 1
        self.advisory = kwargs.get("generation_advisory")
        return {"question": question, "answer": "Synthetic answer [E1]",
                "citations": self.citations(kwargs["docs"]), "verified": True}

    def _deterministic_table_answer(self, question):
        raise AssertionError("Jev-only admission must not bypass calculator prerequisites")


def install_judgment(monkeypatch, action="answer", status="ok"):
    monkeypatch.setattr(graph, "judge_evidence", lambda *args: {
        "enabled": True, "status": status, "action": action if status == "ok" else None,
    })


@pytest.mark.parametrize("question", [
    "2025年1月北京市原保险保费收入是多少？",
    "计算北京市2025年1月和2月原保险保费收入增长率",
])
def test_jev_only_admission_generates_and_trace_keeps_rule_failure(monkeypatch, question):
    install_judgment(monkeypatch)
    rag = GraphRAG()
    result = graph.invoke_rag_graph(rag, question)
    assessment = result["evidence_assessment"]
    assert not assessment["complete"] and assessment["jev_allow"]
    assert assessment["admission_sources"] == ["jev"]
    assert rag.generate_count == 1 and rag.retrieve_count == 1
    assert result["evidence_decision_trace"]["sufficiency_decision"]["decision"] == "answer"
    assert not result["evidence_decision_trace"]["assembly"]["complete"]


@pytest.mark.parametrize("action,status", [("needs_evidence", "ok"), ("needs_clarification", "ok"), (None, "error")])
def test_both_blocked_supplements_once_then_refuses(monkeypatch, action, status):
    install_judgment(monkeypatch, action, status)
    rag = GraphRAG()
    result = graph.invoke_rag_graph(rag, "2025年1月北京市原保险保费收入是多少？")
    assert result["insufficient_evidence"]
    assert rag.retrieve_count == 2 and rag.generate_count == 0


def test_rule_allow_survives_jev_warning_and_passes_advisory(monkeypatch):
    install_judgment(monkeypatch, "needs_clarification")
    rag = GraphRAG()
    result = graph.invoke_rag_graph(rag, "演示归档需要谁确认？")
    assert result["evidence_assessment"]["admission_sources"] == ["rule_gate"]
    assert rag.advisory["action"] == "needs_clarification"


def test_jev_cannot_enable_unsupported_operation(monkeypatch):
    install_judgment(monkeypatch)
    rag = GraphRAG()
    result = graph.invoke_rag_graph(rag, "请预测2025年北京市原保险保费收入会是多少？")
    assert result["insufficient_evidence"]
    assert rag.generate_count == 0
    assert "reject_unsupported" in result["graph_trace"]


def test_real_bm25_ask_with_mock_jev(tmp_path, monkeypatch):
    from bank_rag.pipeline import TrustedRAG
    knowledge = tmp_path / "knowledge.jsonl"
    knowledge.write_text(json.dumps({"content": "Synthetic archive evidence for the public demo.",
        "meta": document().meta}) + "\n", encoding="utf-8")
    install_judgment(monkeypatch)
    result = TrustedRAG(knowledge).ask("2025年1月北京市原保险保费收入是多少？ archive")
    assert "runtime_error" not in result
    assert result["evidence_assessment"]["jev_allow"]
    assert result["answer_method"] == "retrieved_evidence_only"
    assert result["evidence"] and result["citations"]


def test_sdk_request_and_response_with_offline_transport(monkeypatch):
    httpx2 = pytest.importorskip("httpx2")
    typesafe_sdk = pytest.importorskip("typesafe_sdk")
    actual_client = typesafe_sdk.TypeSafeClient
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {
                "action": {"type": "choice", "choice": "answer", "confidence": 1.0,
                           "probabilities": {"answer": 1.0, "needs_evidence": 0.0,
                                             "needs_clarification": 0.0, "unsupported": 0.0}},
                "evidence_sufficient": {"type": "noul", "noul": 0.99},
                "missing_user_conditions": {"type": "noul", "noul": 0.01},
            },
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    def factory(**kwargs):
        assert kwargs["retry"].max_retries == 0
        return actual_client(api_key="offline-test-key", transport=httpx2.MockTransport(respond), **kwargs)

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test-key")
    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", factory)
    result = jev_gate.judge_evidence("synthetic question", {}, [document()])
    assert result["status"] == "ok" and result["action"] == "answer"
    assert len(requests) == 1
    assert requests[0]["model"] == "jev-1.13.0"
    assert requests[0]["questions"]["action"]["type"] == "choice"
    assert set(requests[0]["state"]) == {"user_question", "resolved_conditions", "evidence"}


@pytest.mark.parametrize("enabled,status", [("0", "disabled"), ("1", "missing_api_key")])
@pytest.mark.parametrize("question,allowed", [
    ("演示归档需要谁确认？", True),
    ("2025年1月北京市原保险保费收入是多少？", False),
])
def test_disabled_or_unconfigured_graph_keeps_rule_routing(monkeypatch, enabled, status, question, allowed):
    monkeypatch.setenv("RAG_ENABLE_JEV", enabled)
    rag = GraphRAG()
    result = graph.invoke_rag_graph(rag, question)
    assert result["evidence_assessment"]["jev"]["status"] == status
    assert result["evidence_assessment"]["allow_to_answer"] is allowed
    assert rag.generate_count == int(allowed)
    assert rag.retrieve_count == (1 if allowed else 2)


def test_api_ask_runs_jev_admission_with_mock_generation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from bank_rag import api, pipeline
    knowledge = tmp_path / "knowledge.jsonl"
    knowledge.write_text(json.dumps({"content": "Synthetic archive evidence for the public demo.",
        "meta": document().meta}) + "\n", encoding="utf-8")
    monkeypatch.setenv("KNOWLEDGE_PATH", str(knowledge))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    install_judgment(monkeypatch)
    generator = SimpleNamespace(run=lambda **kwargs: {"replies": [
        SimpleNamespace(text="Synthetic archive answer [E1]", meta={"finish_reason": "stop"})]})
    monkeypatch.setattr(pipeline, "OpenAIChatGenerator", lambda **kwargs: generator)
    api.get_rag.cache_clear()
    try:
        with TestClient(api.app) as client:
            response = client.post("/ask", json={
                "question": "2025年1月北京市原保险保费收入是多少？ archive"})
        assert response.status_code == 200
        result = response.json()
        assert "runtime_error" not in result
        assert result["evidence_assessment"]["admission_sources"] == ["jev"]
        assert result["answer"] == "Synthetic archive answer [E1]"
        assert result["answer_method"] == "deepseek_evidence_constrained"
    finally:
        api.get_rag.cache_clear()
