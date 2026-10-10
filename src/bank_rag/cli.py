from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_answers, evaluate_retrieval
from .batch_test import run_batch_test
from .ingest import write_hierarchical_knowledge, write_knowledge
from .pipeline import TrustedRAG
from .source_summaries import write_file_summaries, write_llm_file_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="银行监管可信 RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="解析资料并生成可审计 JSONL 知识库")
    index.add_argument("--source", type=Path, required=True)
    index.add_argument("--output", type=Path, default=Path("data/knowledge.jsonl"))
    index.add_argument("--strategy", choices=("baseline", "hierarchical"), default="baseline")
    summarize = sub.add_parser("summarize", help="从知识库生成结构化文件级梗概")
    summarize.add_argument("--knowledge", type=Path, required=True)
    summarize.add_argument("--output", type=Path, default=Path("data/file_summaries.jsonl"))
    summarize.add_argument("--llm", action="store_true", help="调用 DeepSeek 生成语义梗概")
    summarize.add_argument("--batch-size", type=int, default=6)
    for name in ("retrieve", "ask"):
        command = sub.add_parser(name)
        command.add_argument("--knowledge", type=Path, required=True)
        command.add_argument("--question", required=True)
    evaluation = sub.add_parser("evaluate", help="根据 QA Excel 或 JSONL 测量来源命中率")
    evaluation.add_argument("--knowledge", type=Path, required=True)
    evaluation.add_argument("--qa", type=Path, required=True)
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--offset", type=int, default=0, help="跳过前 N 题，用于分批评测")
    evaluation.add_argument("--judge", action="store_true", help="额外调用 DeepSeek 对回答与标准答案做语义评分（需配合 --answer）")
    evaluation.add_argument("--review", action="store_true", help="额外调用 DeepSeek 按问题和证据复核回答质量，不使用标准答案")
    evaluation.add_argument("--answer", action="store_true", help="调用 DeepSeek 逐题作答，并写入正确率报告")
    evaluation.add_argument("--report-dir", type=Path, default=Path("reports/qa_eval"))
    batch = sub.add_parser("batch-test", help="批量作答无标准答案的 CSV 压力测试集")
    batch.add_argument("--knowledge", type=Path, required=True)
    batch.add_argument("--questions", type=Path, required=True, help="包含 question 列的 CSV 文件")
    batch.add_argument("--limit", type=int)
    batch.add_argument("--offset", type=int, default=0)
    batch.add_argument("--report-dir", type=Path, default=Path("reports/batch_test"))
    args = parser.parse_args()
    if args.command == "index":
        writer = write_hierarchical_knowledge if args.strategy == "hierarchical" else write_knowledge
        print(json.dumps({"indexed_records": writer(args.source, args.output), "knowledge": str(args.output), "strategy": args.strategy}, ensure_ascii=False))
        return
    if args.command == "summarize":
        writer = write_llm_file_summaries if args.llm else write_file_summaries
        kwargs = {"batch_size": args.batch_size} if args.llm else {}
        print(json.dumps({"file_summaries": writer(args.knowledge, args.output, **kwargs), "output": str(args.output), "generator": "deepseek" if args.llm else "deterministic"}, ensure_ascii=False))
        return
    rag = TrustedRAG(args.knowledge)
    if args.command == "retrieve":
        docs = rag.retrieve(args.question)
        print(json.dumps({"citations": rag.citations(docs), "evidence": [doc.content for doc in docs]}, ensure_ascii=False, indent=2))
    elif args.command == "ask":
        print(json.dumps(rag.ask(args.question), ensure_ascii=False, indent=2))
    elif args.command == "batch-test":
        print(json.dumps(run_batch_test(args.questions, rag, args.report_dir, args.limit, args.offset), ensure_ascii=False, indent=2))
    else:
        if args.answer:
            print(json.dumps(evaluate_answers(args.qa, rag, args.report_dir, args.limit, args.offset, args.judge, args.review), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(evaluate_retrieval(args.qa, rag, args.limit, args.offset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
