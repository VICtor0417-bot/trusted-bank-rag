from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import TrustedRAG
from .question_parser import parse_question


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@lru_cache
def get_rag() -> TrustedRAG:
    # Public distribution includes synthetic demo data only.
    knowledge = Path(os.getenv("KNOWLEDGE_PATH", "data/demo/generated/knowledge.jsonl"))
    if not knowledge.exists():
        raise FileNotFoundError(f"未找到知识库：{knowledge}")
    return TrustedRAG(knowledge)


app = FastAPI(title="可信银行监管 RAG", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(payload: Question) -> dict:
    try:
        rag = get_rag()
        documents = rag.retrieve(payload.question)
        return {"citations": rag.citations(documents), "evidence": [document.content for document in documents]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/parse-question")
def parse_question_endpoint(payload: Question) -> dict:
    """Explain how a table/regulation question will be interpreted before asking."""
    return parse_question(payload.question).to_dict()


@app.post("/ask")
def ask(payload: Question) -> dict:
    try:
        return get_rag().ask(payload.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
