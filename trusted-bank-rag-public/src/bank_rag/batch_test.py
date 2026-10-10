"""Batch execution for question sets without reference answers.

This is intentionally separate from ``evaluate``: it records a reproducible
system run but does not pretend to calculate accuracy when no gold answer is
available.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .pipeline import TrustedRAG


OUTPUT_FIELDS = (
    "id",
    "qa_type",
    "difficulty",
    "question",
    "source_title",
    "file_label",
    "answer",
    "verified",
    "citations_json",
    "question_plan_json",
    "source_routing_json",
    "error",
)


def run_batch_test(
    questions_path: Path,
    rag: TrustedRAG,
    report_dir: Path,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Ask each CSV question and write an auditable answer log.

    Input must contain a ``question`` column. Existing extra columns (such as
    source_title and file_label) are carried into the result for review.
    """
    with questions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "question" not in rows[0]:
        raise ValueError("题库 CSV 必须包含 question 列")

    selected = rows[offset : offset + limit if limit is not None else None]
    report_dir.mkdir(parents=True, exist_ok=True)
    details_path = report_dir / "details.csv"
    jsonl_path = report_dir / "details.jsonl"

    nonempty = verified = errors = 0
    with details_path.open("w", encoding="utf-8-sig", newline="") as csv_handle, jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            record: dict[str, Any] = {key: row.get(key, "") for key in OUTPUT_FIELDS}
            question = (row.get("question") or "").strip()
            try:
                result = rag.ask(question)
                answer = str(result.get("answer") or "")
                record.update(
                    answer=answer,
                    verified=bool(result.get("verified")),
                    citations_json=json.dumps(result.get("citations", []), ensure_ascii=False),
                    question_plan_json=json.dumps(result.get("question_plan", {}), ensure_ascii=False),
                    source_routing_json=json.dumps(result.get("source_routing", []), ensure_ascii=False),
                    error="",
                )
                nonempty += bool(answer.strip())
                verified += bool(result.get("verified"))
            except Exception as exc:  # preserve remaining test results
                errors += 1
                record.update(
                    answer="",
                    verified=False,
                    citations_json="[]",
                    question_plan_json="{}",
                    source_routing_json="[]",
                    error=f"{type(exc).__name__}: {exc}",
                )
            writer.writerow(record)
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"已完成 {index}/{len(selected)} 题", flush=True)

    count = len(selected)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "batch_answer_run_without_gold_answers",
        "questions": count,
        "nonempty_answers": nonempty,
        "nonempty_rate": round(nonempty / count, 4) if count else None,
        "verified_answers": verified,
        "verified_rate": round(verified / count, 4) if count else None,
        "errors": errors,
        "details_csv": str(details_path),
        "details_jsonl": str(jsonl_path),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
