# earnings-lens

Ask questions about company earnings releases and financial supplements. The
project extracts text from local PDF and HTML files, retrieves relevant passages,
and uses Groq to answer with source citations.

## Getting started

Install the dependencies from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in your environment or in a project-root `.env` file. For
downloading SEC documents, also set `SEC_USER_AGENT` to a value identifying you.
See [.env.example](.env.example) for the expected format. Keep your actual `.env`
private.

Source documents are listed in [sources.py](sources.py). To download them, run:

```powershell
python fetch_sources.py
```

The downloader saves PDFs to `data/pdf/`, HTML files to `data/html/`, and writes a
metadata sidecar for each file. It downloads the full configured list. Existing
local documents can be used without downloading again.

## Ask questions

Run an interactive session to ask questions:

```powershell
python rag.py
```

Submit a blank question to quit. For a one-time question, use `--query`:

```powershell
python rag.py --query "What were Tradeweb's total revenues in Q2 2026?"
```

`--k` sets the number of passages to retrieve (default `5`). The default Groq model
is `openai/gpt-oss-20b`; use `--llm-model` to select another model available to
your API key. Retrieved passages are printed before each answer.

For example:

> Compare JPMorgan and Goldman Sachs Q3 2026 total revenues. Which reported more,
> and by how much?

Example answer: JPMorgan reported total net revenue of **$50.172 billion** [5],
compared with Goldman Sachs' net revenues of **$18.679 billion** [1]. JPMorgan
reported **$31.493 billion more**. These citation numbers match this retrieval
run; they can change when the retrieved passages change.

To check how the system handles missing information, ask:

> According to these filings, what was JPMorgan's closing share price the day
> after its Q3 2026 earnings release?

That price is not provided in the earnings documents, so the expected response is
`I don't know based on the provided documents.`

Answers are generated from retrieved text and can be wrong. Check the printed
passages and verify important figures against the original filing.

## Pipeline

- [sources.py](sources.py) defines the documents to download.
- [fetch_sources.py](fetch_sources.py) downloads PDF and HTML files.
- [loader.py](loader.py) extracts PDF text by page and HTML text as one page.
- [chunking.py](chunking.py) splits pages into 300-word chunks with 50-word overlap.
- [embeddings.py](embeddings.py) embeds passages and ranks them by cosine similarity
  using the local `BAAI/bge-small-en-v1.5` model.
- [rag.py](rag.py) sends the question and retrieved passages to Groq and prints the
  answer with citations. For comparisons that name multiple companies, retrieval
  reserves results for the most relevant passages from each named company when
  `--k` allows it.

The document embeddings are built in memory when `rag.py` starts and reused for
each question in that session.

## Inspect and evaluate

Preview extracted text and page counts:

```powershell
python loader.py
```

Preview chunks:

```powershell
python chunking.py
```

Inspect retrieval for a question, or open a PCA plot:

```powershell
python embeddings.py --query "What were Tradeweb's total revenues in Q2 2026?"
python embeddings.py --query "What were Tradeweb's total revenues in Q2 2026?" --plot
```

Run the retrieval evaluation:

```powershell
python -m eval.run_eval
```

Questions and expected source pages are in [eval/questions.jsonl](eval/questions.jsonl).
Evaluation results, including hit-rate@k and mean reciprocal rank, are appended to
`eval/results.jsonl`.

Run the tests:

```powershell
python -m unittest discover -s tests
```

## Notes

- Retrieval searches loaded documents in memory; there is no persistent vector
  index or general company/quarter filter. Comparisons naming multiple companies
  reserve result slots for each named company when the result limit permits.
- Scanned PDFs are not OCR-processed. HTML tables may lose their structure during
  text extraction, and HTML citations refer to page 1.
- Evaluation measures retrieval relevance, not generated-answer accuracy.
