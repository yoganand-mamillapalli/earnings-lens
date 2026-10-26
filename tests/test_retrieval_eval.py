import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embeddings import (
    QUERY_PREFIX,
    embed_passages,
    embed_query,
    plot_embeddings,
    rank_chunks,
)
from eval.run_eval import calculate_metrics, load_questions
from rag import (
    DEFAULT_GROQ_MODEL,
    REFUSAL,
    _iter_questions,
    answer_question,
    build_context,
    retrieve_chunks,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.sentences: list[str] = []
        self.options: dict[str, bool] = {}

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.sentences = sentences
        self.options = {
            "normalize_embeddings": normalize_embeddings,
            "convert_to_numpy": convert_to_numpy,
            "show_progress_bar": show_progress_bar,
        }
        return [[1.0, 0.0] for _ in sentences]


class FakeGroqClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.request: dict[str, object] | None = None
        self.chat = self
        self.completions = self

    def create(self, **request: object) -> object:
        self.request = request
        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": self.content})()},
                    )()
                ]
            },
        )()


class RetrievalEvaluationTests(unittest.TestCase):
    def test_interactive_questions_continue_until_blank_input(self) -> None:
        with patch(
            "builtins.input",
            side_effect=["First question?", "Second question?", ""],
        ) as prompt:
            self.assertEqual(
                list(_iter_questions(None)),
                ["First question?", "Second question?"],
            )

        self.assertEqual(prompt.call_count, 3)

    def test_query_argument_yields_only_one_question_without_prompting(self) -> None:
        with patch("builtins.input") as prompt:
            self.assertEqual(list(_iter_questions("One question?")), ["One question?"])

        prompt.assert_not_called()

    def test_query_uses_bge_prefix_and_normalized_embeddings(self) -> None:
        model = FakeEncoder()

        vector = embed_query(model, "How much revenue?")

        self.assertEqual(
            model.sentences,
            [QUERY_PREFIX + "How much revenue?"],
        )
        self.assertTrue(model.options["normalize_embeddings"])
        self.assertEqual(vector, [1.0, 0.0])

    def test_passages_are_embedded_without_query_prefix(self) -> None:
        model = FakeEncoder()

        vectors = embed_passages(model, ["Quarterly revenue was $100 million."])

        self.assertEqual(model.sentences, ["Quarterly revenue was $100 million."])
        self.assertEqual(vectors, [[1.0, 0.0]])

    def test_rank_chunks_orders_normalized_vectors(self) -> None:
        chunks = [
            {
                "source": "a.pdf",
                "company": "Example",
                "quarter": "Q1",
                "page": 1,
                "chunk": 0,
                "text": "First passage.",
            },
            {
                "source": "b.pdf",
                "company": "Example",
                "quarter": "Q1",
                "page": 2,
                "chunk": 1,
                "text": "Second passage.",
            },
        ]

        ranked = rank_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]], [0.0, 1.0], 2)

        self.assertEqual([item["source"] for item in ranked], ["b.pdf", "a.pdf"])
        self.assertEqual(ranked[0]["score"], 1.0)

    def test_comparison_retrieval_includes_each_named_company(self) -> None:
        chunks = [
            {
                "source": "goldman.html",
                "company": "Goldman Sachs",
                "quarter": "Q3 2026",
                "page": 1,
                "chunk": 0,
                "text": "Goldman revenue.",
            },
            {
                "source": "jpmorgan.html",
                "company": "JPMorgan",
                "quarter": "Q3 2026",
                "page": 1,
                "chunk": 1,
                "text": "JPMorgan revenue.",
            },
            {
                "source": "goldman.html",
                "company": "Goldman Sachs",
                "quarter": "Q3 2026",
                "page": 1,
                "chunk": 2,
                "text": "Another Goldman passage.",
            },
            {
                "source": "goldman.html",
                "company": "Goldman Sachs",
                "quarter": "Q3 2026",
                "page": 1,
                "chunk": 3,
                "text": "Third Goldman passage.",
            },
            {
                "source": "jpmorgan.html",
                "company": "JPMorgan",
                "quarter": "Q3 2026",
                "page": 1,
                "chunk": 4,
                "text": "Another JPMorgan passage.",
            },
        ]

        results = retrieve_chunks(
            "Compare JPMorgan and Goldman Sachs revenue.",
            chunks,
            [
                [1.0, 0.0],
                [0.6, 0.8],
                [0.9, 0.1],
                [0.7, 0.7],
                [0.3, 0.95],
            ],
            [1.0, 0.0],
            4,
        )

        self.assertEqual(
            {result["source"] for result in results},
            {"goldman.html", "jpmorgan.html"},
        )
        self.assertEqual(len(results), 4)
        self.assertEqual(
            sum(result["source"] == "goldman.html" for result in results),
            2,
        )
        self.assertEqual(
            sum(result["source"] == "jpmorgan.html" for result in results),
            2,
        )

    def test_metrics_measure_source_page_hits_and_rank(self) -> None:
        questions = [
            {
                "id": "hit",
                "question": "Question?",
                "expected_answer": "Answer",
                "source": "report.pdf",
                "company": "Example",
                "quarter": "Q1",
                "page": 2,
                "type": "lookup",
            },
            {
                "id": "miss",
                "question": "Another?",
                "expected_answer": "Answer",
                "source": "other.pdf",
                "company": "Example",
                "quarter": "Q1",
                "page": 3,
                "type": "lookup",
            },
            {
                "id": "unknown",
                "question": "Unknown?",
                "expected_answer": "I don't know.",
                "source": None,
                "company": "Example",
                "quarter": "Q1",
                "page": None,
                "type": "not_in_doc",
            },
        ]
        retrieved = {
            "hit": [
                {"source": "other.pdf", "page": 1, "chunk": 0, "score": 0.9},
                {"source": "report.pdf", "page": 2, "chunk": 1, "score": 0.8},
            ],
            "miss": [{"source": "other.pdf", "page": 1, "chunk": 0, "score": 0.7}],
        }

        self.assertEqual(
            calculate_metrics(questions, retrieved),
            {
                "answerable_questions": 2,
                "hit_rate_at_k": 0.5,
                "mrr_at_k": 0.25,
            },
        )

    def test_load_questions_reports_invalid_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            valid_question = {
                "id": "ok",
                "question": "Question?",
                "expected_answer": "Answer",
                "source": "report.pdf",
                "company": "Example",
                "quarter": "Q1",
                "page": 1,
                "type": "lookup",
            }
            path.write_text(
                json.dumps(valid_question) + "\nnot json\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_questions(path)

    def test_rank_chunks_rejects_mismatched_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching dimensions"):
            rank_chunks(
                [
                    {
                        "source": "report.pdf",
                        "company": "Example",
                        "quarter": "Q1",
                        "page": 1,
                        "chunk": 0,
                        "text": "Passage.",
                    }
                ],
                [[1.0, 0.0]],
                [1.0],
                1,
            )

    def test_plot_rejects_dimensions_above_three(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 or 3"):
            plot_embeddings([], [], [], [], dimensions=4)

    def test_context_includes_citation_source_and_page(self) -> None:
        chunks = [
            {
                "source": "report.pdf",
                "company": "Example",
                "quarter": "Q1",
                "page": 2,
                "chunk": 0,
                "text": "Revenue was $100 million.",
            }
        ]

        self.assertEqual(
            build_context(
                [{"source": "report.pdf", "page": 2, "chunk": 0, "score": 0.9}],
                chunks,
            ),
            "[1] Source: report.pdf, page 2\nRevenue was $100 million.",
        )

    def test_answer_prompt_requires_supported_answers_and_citations(self) -> None:
        client = FakeGroqClient("I don't know based on the provided documents.")
        chunks = [
            {
                "source": "report.pdf",
                "company": "Example",
                "quarter": "Q1",
                "page": 2,
                "chunk": 0,
                "text": "Revenue was $100 million.",
            }
        ]
        results = [{"source": "report.pdf", "page": 2, "chunk": 0, "score": 0.9}]

        answer = answer_question("What is the stock target?", results, chunks, client)

        self.assertEqual(answer, "I don't know based on the provided documents.")
        self.assertIsNotNone(client.request)
        self.assertEqual(client.request["model"], DEFAULT_GROQ_MODEL)
        messages = client.request["messages"]
        self.assertIn("Retrieved passages may be irrelevant or insufficient", messages[0]["content"])
        self.assertIn("[1] Source: report.pdf, page 2", messages[1]["content"])

    def test_answer_without_retrieved_passages_refuses_without_calling_llm(self) -> None:
        client = FakeGroqClient("Unexpected")

        self.assertEqual(answer_question("Question?", [], [], client), REFUSAL)
        self.assertIsNone(client.request)


if __name__ == "__main__":
    unittest.main()
