"""Evaluate dense retrieval against the JSONL golden question set."""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunking import chunk_pages
from embeddings import (
    MODEL_NAME,
    RankedChunk,
    embed_passages,
    embed_query,
    load_embedding_model,
    rank_chunks,
)
from loader import load_documents

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.jsonl"


class EvalQuestion(TypedDict):
    id: str
    question: str
    expected_answer: str
    source: str | None
    company: str
    quarter: str
    page: int | None
    type: str


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    """Read and validate the JSONL evaluation set."""
    questions: list[EvalQuestion] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path} on line {line_number}") from error
        if not isinstance(question, dict):
            raise ValueError(f"Expected a JSON object in {path} on line {line_number}")
        required_fields = {
            "id",
            "question",
            "expected_answer",
            "source",
            "company",
            "quarter",
            "page",
            "type",
        }
        missing = required_fields - question.keys()
        if missing:
            raise ValueError(
                f"Missing fields in {path} on line {line_number}: "
                f"{', '.join(sorted(missing))}"
            )
        string_fields = ("id", "question", "expected_answer", "company", "quarter", "type")
        if any(not isinstance(question[field], str) for field in string_fields):
            raise ValueError(f"Expected string fields in {path} on line {line_number}")
        if question["source"] is not None and not isinstance(question["source"], str):
            raise ValueError(f"Expected source to be a string or null in {path} on line {line_number}")
        page = question["page"]
        if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page <= 0):
            raise ValueError(f"Expected page to be a positive integer or null in {path} on line {line_number}")
        questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "source": question["source"],
                "company": question["company"],
                "quarter": question["quarter"],
                "page": page,
                "type": question["type"],
            }
        )
    if not questions:
        raise ValueError(f"No evaluation questions found in {path}")
    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Question IDs must be unique in {path}")
    return questions


def calculate_metrics(
    questions: list[EvalQuestion],
    retrieved: dict[str, list[RankedChunk]],
) -> dict[str, float | int]:
    """Calculate hit-rate@k and MRR for questions with known source pages."""
    answerable = [
        question
        for question in questions
        if question["source"] is not None and question["page"] is not None
    ]
    if not answerable:
        raise ValueError("No answerable questions with expected source and page")

    hits = 0
    reciprocal_ranks = 0.0
    for question in answerable:
        results = retrieved[question["id"]]
        expected = (question["source"], question["page"])
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if (result["source"], result["page"]) == expected
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1 / rank

    total = len(answerable)
    return {
        "answerable_questions": total,
        "hit_rate_at_k": hits / total,
        "mrr_at_k": reciprocal_ranks / total,
    }


def run_evaluation(
    questions: list[EvalQuestion],
    *,
    k: int = 5,
    model_name: str = MODEL_NAME,
    chunk_size: int = 300,
    overlap: int = 50,
) -> dict[str, object]:
    """Embed the corpus and questions, then score source-page retrieval."""
    documents = load_documents()
    if not documents:
        raise ValueError(f"No PDF or HTML documents found under {ROOT / 'data'}")
    chunks = chunk_pages(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Documents were loaded, but produced no non-empty chunks")

    answerable = [
        question
        for question in questions
        if question["source"] is not None and question["page"] is not None
    ]
    if not answerable:
        raise ValueError("No answerable questions with expected source and page")

    model = load_embedding_model(model_name)
    chunk_embeddings = embed_passages(model, [chunk["text"] for chunk in chunks])
    retrieved = {
        question["id"]: rank_chunks(
            chunks,
            chunk_embeddings,
            embed_query(model, question["question"]),
            k,
        )
        for question in answerable
    }
    per_question = []
    for question in answerable:
        expected = (question["source"], question["page"])
        results = retrieved[question["id"]]
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if (result["source"], result["page"]) == expected
            ),
            None,
        )
        per_question.append(
            {
                "id": question["id"],
                "expected_source": question["source"],
                "expected_page": question["page"],
                "relevant_rank": rank,
                "top_results": results,
            }
        )
    return {
        "config": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "k": k,
            "embedding_model": model_name,
        },
        "metrics": calculate_metrics(questions, retrieved),
        "per_question": per_question,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Sentence-transformers model name or local model path",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k must be greater than zero")

    result = run_evaluation(
        load_questions(),
        k=args.k,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    result["run_at"] = datetime.now(timezone.utc).isoformat()
    with RESULTS_PATH.open("a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nSaved run to {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
