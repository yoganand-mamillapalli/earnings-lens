"""Load PDF and HTML source documents as text pages."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IGNORED_HTML_TAGS = {"head", "script", "style", "noscript"}
VOID_HTML_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
BLOCK_HTML_TAGS = {
    "address",
    "article",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "p",
    "section",
    "tr",
}


class DocumentPage(TypedDict):
    source: str
    company: str
    quarter: str
    page: int
    text: str


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_tags: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if self.ignored_tags:
            if tag not in VOID_HTML_TAGS:
                self.ignored_tags.append(tag)
            return
        if tag in IGNORED_HTML_TAGS:
            self.ignored_tags.append(tag)
        elif tag in BLOCK_HTML_TAGS:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_tags:
            if tag == self.ignored_tags[-1]:
                self.ignored_tags.pop()
            return
        if tag in BLOCK_HTML_TAGS:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_data(self, data: str) -> None:
        if not self.ignored_tags:
            self.parts.append(data)

    def get_text(self) -> str:
        lines = []
        for line in "".join(self.parts).replace("\xa0", " ").splitlines():
            cleaned = re.sub(r"[ \t]+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines)


def _read_metadata(path: Path) -> dict[str, object]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not metadata_path.exists():
        return {}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}")
    return metadata


def _load_pdf(path: Path, metadata: dict[str, object]) -> list[DocumentPage]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF loading requires pypdf; install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    reader = PdfReader(path)
    return [
        {
            "source": path.name,
            "company": str(metadata.get("company", "")),
            "quarter": str(metadata.get("quarter", "")),
            "page": page_number,
            "text": page.extract_text() or "",
        }
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def _load_html(path: Path, metadata: dict[str, object]) -> list[DocumentPage]:
    parser = _HTMLTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return [
        {
            "source": path.name,
            "company": str(metadata.get("company", "")),
            "quarter": str(metadata.get("quarter", "")),
            "page": 1,
            "text": parser.get_text(),
        }
    ]


def load_documents(data_dir: Path = DATA_DIR) -> list[DocumentPage]:
    """Load supported documents from data/pdf and data/html beneath data_dir."""
    pages: list[DocumentPage] = []
    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".htm", ".html"}
    )
    for path in paths:
        metadata = _read_metadata(path)
        if path.suffix.lower() == ".pdf":
            pages.extend(_load_pdf(path, metadata))
        else:
            pages.extend(_load_html(path, metadata))
    return pages


def main() -> int:
    pages = load_documents()
    sources = sorted({page["source"] for page in pages})
    for source in sources:
        source_pages = [page for page in pages if page["source"] == source]
        print(f"{source}: {len(source_pages)} page(s)")

    print("\nPreview:")
    for page in pages[:3]:
        print(f"\n{page['source']} - page {page['page']}")
        print(page["text"][:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
