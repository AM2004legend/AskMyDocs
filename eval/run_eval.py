"""Run the eval set and print a results table per question type.

    python eval/run_eval.py                 # route + retrieval + refusal only: no LLM calls, free
    python eval/run_eval.py --answers       # also generate answers (uses LLM quota)
    python eval/run_eval.py --answers --grade   # also LLM-grade answers against `expected`

Documents must already be uploaded (UI or API) and ready. Every run is saved to
eval/results/<timestamp>.json so before/after numbers can be compared.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import get_pool  # noqa: E402
from app.docmap.summaries import call_json  # noqa: E402
from app.llm.answer import answer_question, prepare  # noqa: E402
from app.llm.client import LLMError, get_llm  # noqa: E402

EXPECTED_ROUTE = {"specific": "specific", "whole_document": "whole_document", "entity": "entity"}
TYPES = ["specific", "whole_document", "entity", "unanswerable"]
GRADE_SYSTEM = (
    "You grade answers from a document Q&A system against a reference answer. Respond with JSON only: "
    '{"verdict": "correct" | "partial" | "wrong", "reason": "<one sentence>"}. '
    "correct = contains the key facts of the reference and nothing contradicting it; "
    "partial = some key facts missing; wrong = incorrect, contradicting, or a refusal when an answer exists."
)


def find_documents(names: set[str]) -> dict[str, int]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ON (filename) filename, id, status FROM documents WHERE filename = ANY(%s)"
            " ORDER BY filename, created_at DESC",
            [list(names)],
        ).fetchall()
    found = {r["filename"]: r["id"] for r in rows if r["status"] == "ready"}
    missing = names - found.keys()
    if missing:
        print(f"Skipping questions for documents not uploaded/ready yet: {', '.join(sorted(missing))}\n")
    if not found:
        sys.exit("None of the eval documents are uploaded yet.")
    return found


def retrieval_hit(q: dict, excerpts) -> bool | None:
    locs, evidence = q.get("expected_locs") or [], [e.lower() for e in q.get("evidence") or []]
    if not locs and not evidence:
        return None  # nothing to check (e.g. "summarise the story")
    return any(e.loc in locs or any(ev in e.text.lower() for ev in evidence) for e in excerpts)


def grade(q: dict, answer: str) -> float:
    messages = [
        {"role": "system", "content": GRADE_SYSTEM},
        {"role": "user", "content": f"Question: {q['question']}\nReference: {q['expected']}\nAnswer: {answer}"},
    ]
    verdict = call_json(get_llm(), messages).get("verdict", "wrong")
    return {"correct": 1.0, "partial": 0.5}.get(verdict, 0.0)


def mean(values) -> str:
    values = [v for v in values if v is not None]
    return f"{sum(values) / len(values):.0%}" if values else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(Path(__file__).parent / "questions.yaml"))
    parser.add_argument("--answers", action="store_true", help="generate answers with the LLM")
    parser.add_argument("--grade", action="store_true", help="LLM-grade answers (implies --answers)")
    parser.add_argument("--label", default="", help="name for this run, e.g. 'baseline' or 'docmap'")
    parser.add_argument("--document", default="", help="only questions for this filename")
    args = parser.parse_args()
    args.answers = args.answers or args.grade

    spec = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    questions = [q for q in spec["questions"] if not args.document or q["document"] == args.document]
    doc_ids = find_documents({q["document"] for q in questions})
    questions = [q for q in questions if q["document"] in doc_ids]

    results = []
    for q in questions:
        started = time.perf_counter()
        prepared = prepare(doc_ids[q["document"]], q["question"])
        row = {
            "id": q["id"],
            "type": q["type"],
            "route": prepared.route,
            "route_ok": None if q["type"] == "unanswerable" else prepared.route == EXPECTED_ROUTE[q["type"]],
            "hit": None if q["type"] == "unanswerable" else retrieval_hit(q, prepared.excerpts),
            "refused": prepared.refuse,
            "best_similarity": prepared.best_similarity,
            "locs": [e.loc for e in prepared.excerpts],
        }
        if args.answers:
            try:
                answer = answer_question(doc_ids[q["document"]], q["question"])
                row.update(answer=answer.answer, refused=answer.refused)
                if args.grade and q["type"] != "unanswerable":
                    row["correct"] = grade(q, answer.answer)
            except LLMError as exc:
                row["error"] = str(exc)
        row["refusal_ok"] = row["refused"] == (q["type"] == "unanswerable")
        row["latency_ms"] = int((time.perf_counter() - started) * 1000)
        results.append(row)
        flag = "OK " if row["refusal_ok"] and row["route_ok"] is not False and row["hit"] is not False else "!! "
        print(
            f"{flag}{q['id']:<7} route={row['route']:<15} hit={row['hit']!s:<5} "
            f"refused={row['refused']!s:<5} {q['question']}"
        )

    print("\n| type | n | route ok | retrieval hit | refusal ok | correct | avg latency |")
    print("|---|---|---|---|---|---|---|")
    for t in TYPES + ["all"]:
        rows = [r for r in results if t in (r["type"], "all")]
        if not rows:
            continue
        latency = sum(r["latency_ms"] for r in rows) / len(rows)
        print(
            f"| {t} | {len(rows)} | {mean(r['route_ok'] for r in rows)} | {mean(r['hit'] for r in rows)} | "
            f"{mean(r['refusal_ok'] for r in rows)} | {mean(r.get('correct') for r in rows)} | {latency:.0f} ms |"
        )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{stamp}{'-' + args.label if args.label else ''}.json"
    out.write_text(json.dumps({"label": args.label, "answers": args.answers, "results": results}, indent=2), "utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
