"""Source documents used by the earnings-lens project."""

from typing import TypedDict


class Source(TypedDict):
    company: str
    quarter: str
    doc_type: str
    url: str
    filename: str


SOURCES: list[Source] = [
    {
        "company": "Tradeweb",
        "quarter": "Q2 2026",
        "doc_type": "earnings_release",
        "url": "https://www.tradeweb.com/4a22bd/globalassets/newsroom/07.29.26-earnings/tw-2026-q2_news-release.pdf",
        "filename": "tw-2026-q2_news-release.pdf",
    },
    {
        "company": "MarketAxess",
        "quarter": "Q2 2026",
        "doc_type": "earnings_release",
        "url": "https://s201.q4cdn.com/767283836/files/doc_financials/2026/q2/Q2_2026-Earnings-Release-FINAL.pdf",
        "filename": "Q2_2026-Earnings-Release-FINAL.pdf",
    },
    {
        "company": "Morgan Stanley",
        "quarter": "Q3 2026",
        "doc_type": "earnings_release",
        "url": "https://www.morganstanley.com/content/dam/msdotcom/en/about-us-ir/shareholder/3q2026.pdf",
        "filename": "3q2026.pdf",
    },
    {
        "company": "JPMorgan",
        "quarter": "Q3 2026",
        "doc_type": "financial_supplement",
        "url": "https://www.sec.gov/Archives/edgar/data/0000019617/000162828026065582/a3q26erfex992supplement.htm",
        "filename": "jpmorgan_q3_2026_financial_supplement.html",
    },
    {
        "company": "Goldman Sachs",
        "quarter": "Q3 2026",
        "doc_type": "earnings_results",
        "url": "https://www.sec.gov/Archives/edgar/data/0000886982/000088698226000529/a3q26gsearningsresults.htm",
        "filename": "goldman_sachs_q3_2026_earnings_results.html",
    },
]
