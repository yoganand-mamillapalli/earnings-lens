"""Download the configured PDF and HTML source documents."""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from local_env import load_local_env
from sources import SOURCES, Source

ROOT = Path(__file__).resolve().parent
OUTPUT_DIRS = {
    ".pdf": ROOT / "data" / "pdf",
    ".htm": ROOT / "data" / "html",
    ".html": ROOT / "data" / "html",
}
DEFAULT_USER_AGENT = "earnings-lens/0.1 (public earnings-document downloader)"


def download_source(source: Source) -> Path:
    """Download one source document and write its metadata alongside it."""
    filename = source["filename"]
    suffix = Path(filename).suffix.lower()
    output_dir = OUTPUT_DIRS.get(suffix)
    if output_dir is None:
        raise ValueError(f"Unsupported source file type for {filename!r}")

    parsed_url = urlparse(source["url"])
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    hostname = (parsed_url.hostname or "").lower()
    if hostname == "sec.gov" or hostname.endswith(".sec.gov"):
        sec_user_agent = os.environ.get("SEC_USER_AGENT")
        if not sec_user_agent:
            raise ValueError(
                "Set SEC_USER_AGENT to a value identifying you (name and email) "
                "before downloading SEC documents."
            )
        headers["User-Agent"] = sec_user_agent

    request = urllib.request.Request(source["url"], headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
        content_type = response.headers.get("Content-Type", "")

    if not content:
        raise ValueError(f"Downloaded empty source document: {source['url']}")
    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError(
            f"Expected a PDF from {source['url']}, received Content-Type "
            f"{content_type or 'unknown'}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    temporary_path = destination.with_name(destination.name + ".part")
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(destination)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise

    metadata = {
        **source,
        "content_type": content_type,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    load_local_env()

    failures: list[tuple[str, Exception]] = []
    for source in SOURCES:
        try:
            destination = download_source(source)
        except (OSError, urllib.error.URLError, ValueError) as error:
            failures.append((source["filename"], error))
            print(f"FAILED {source['filename']}: {error}", file=sys.stderr)
        else:
            print(f"Saved {destination.relative_to(ROOT)}")

    if failures:
        print(f"{len(failures)} of {len(SOURCES)} source(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
