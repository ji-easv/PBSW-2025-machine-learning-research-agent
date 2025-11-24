import logging
import re
from typing import List
from ddgs import DDGS
from ratelimit import limits, sleep_and_retry

from datamodel.search_result import SearchResult


def is_blocked_content(results: list) -> bool:
    block_patterns = [
        r"verify you are human",
        r"captcha",
        r"robot check",
        r"unusual traffic",
        r"please enable cookies",
        r"access denied",
    ]
    for result in results:
        if result.get("title") or result.get("href") or result.get("body"):
            for field in ("title", "href", "body", "description"):
                value = result.get(field, "")
                for pattern in block_patterns:
                    if re.search(pattern, str(value), re.IGNORECASE):
                        return True
            # If at least one result looks normal, not blocked
            return False
    # If all results are empty or suspicious, treat as blocked
    return True


@sleep_and_retry
@limits(calls=10, period=30)
def search_web(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    Search the web. Supports advanced search syntax:

    Exact phrases:
        - Use quotes: "[your topic]" - searches for the exact phrase

    Exclude words:
        - Use minus: [topic] -[unwanted term] - excludes results with unwanted terms

    Site-specific search:
        - site:edu [your topic] - search only educational sites
        - site:arxiv.org [your topic] - search only arxiv.org
        - site:researchgate.net [your topic] - search ResearchGate

    Combine operators:
        - "[exact phrase]" site:edu -opinion - exact phrase, only .edu sites, no opinion pieces

    File type search:
        - [topic] filetype:pdf - search for PDF files only
        - research filetype:pdf site:edu - PDFs from educational sites

    Time-based (add year to query):
        - [topic] research 2020 - likely to return results from that year
        - "[topic]" 2024 - recent results

    Examples (diverse research domains):
        - "neural networks" deep learning 2024
        - "CRISPR gene editing" site:edu filetype:pdf
        - quantum computing algorithms -tutorial
        - "climate change models" 2023 site:nature.com
        - protein folding site:arxiv.org

    Args:
        query: Search query (supports syntax above)
        num_results: Maximum number of results to return (default: 10)

    Returns:
        Formatted string with search results including titles, URLs, and snippets.
    """
    try:

        results: List[SearchResult] = []
        with DDGS() as ddgs:
            content = ddgs.text(query, max_results=num_results, backend="mojeek")

            if is_blocked_content(content):
                logging.error("Search blocked by Brave. Detected bot protection.")
                return []

            for result in content:
                results.append(
                    SearchResult(
                        query=query,
                        title=result["title"],
                        url=result["href"],
                        snippet=result["body"],
                    )
                )

        return results

    except Exception as e:
        logging.error("Error searching the web: %s", str(e))
        return []
