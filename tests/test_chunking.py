import unittest

from chunking import chunk_pages
from loader import DocumentPage


def make_page(text: str) -> DocumentPage:
    return {
        "source": "report.pdf",
        "company": "Example",
        "quarter": "Q1 2026",
        "page": 2,
        "text": text,
    }


class ChunkPagesTests(unittest.TestCase):
    def test_chunks_with_requested_overlap_and_metadata(self) -> None:
        page = make_page(" ".join(f"word{i}" for i in range(7)))
        chunks = chunk_pages([page], chunk_size=4, overlap=1)

        self.assertEqual(
            [chunk["text"] for chunk in chunks],
            ["word0 word1 word2 word3", "word3 word4 word5 word6"],
        )
        self.assertEqual(
            [
                (chunk["source"], chunk["company"], chunk["quarter"], chunk["page"])
                for chunk in chunks
            ],
            [("report.pdf", "Example", "Q1 2026", 2)] * 2,
        )
        self.assertEqual([chunk["chunk"] for chunk in chunks], [0, 1])

    def test_pages_are_not_joined_and_short_pages_stay_one_chunk(self) -> None:
        chunks = chunk_pages(
            [make_page("first page"), {**make_page("second page"), "page": 3}],
            chunk_size=10,
            overlap=2,
        )

        self.assertEqual(
            [chunk["text"] for chunk in chunks],
            ["first page", "second page"],
        )
        self.assertEqual([chunk["page"] for chunk in chunks], [2, 3])

    def test_empty_pages_produce_no_chunks(self) -> None:
        self.assertEqual(chunk_pages([make_page(" \n\t")]), [])

    def test_invalid_chunk_settings_raise(self) -> None:
        page = make_page("content")
        for chunk_size, overlap in [(0, 0), (4, -1), (4, 4)]:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_pages([page], chunk_size=chunk_size, overlap=overlap)


if __name__ == "__main__":
    unittest.main()
