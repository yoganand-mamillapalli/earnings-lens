"""Split loaded document pages into overlapping word chunks."""

from typing import TypedDict

from loader import DocumentPage


class DocumentChunk(TypedDict):
    source: str
    company: str
    quarter: str
    page: int
    chunk: int
    text: str


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Split each page independently, preserving its citation metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            text = " ".join(words[start : start + chunk_size])
            chunks.append(
                {
                    "source": page["source"],
                    "company": page["company"],
                    "quarter": page["quarter"],
                    "page": page["page"],
                    "chunk": len(chunks),
                    "text": text,
                }
            )
            if start + chunk_size >= len(words):
                break
            start += step
    return chunks


def main() -> int:
    from loader import load_documents

    pages = load_documents()
    chunks = chunk_pages(pages)
    print(f"Loaded {len(pages)} pages into {len(chunks)} chunks.")
    for chunk in chunks[:3]:
        print(
            f"\n{chunk['source']} - page {chunk['page']} "
            f"(chunk {chunk['chunk']})"
        )
        print(chunk["text"][:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
