import logging
import re
import requests
from typing import List
from ddgs import DDGS
from ratelimit import limits, sleep_and_retry
from bs4 import BeautifulSoup

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


def check_pages_for_relevance(search_results: List[SearchResult]) -> List[SearchResult]:
    for result in search_results:
        response = requests.get(result.url, timeout=10)
        if response.status_code != 200:
            continue

        page_content = response.text

        try:
            soup = BeautifulSoup(page_content, "html.parser")

            # Try to get title
            if soup.title:
                result.title = soup.title.string.strip()

            # Try to find authors (look for meta tags or common patterns)
            meta_authors = soup.find("meta", attrs={"name": "citation_author"})
            if meta_authors:
                result.authors = meta_authors.get("content")
            else:
                # Try to find by regex in the text
                text = soup.get_text(separator=" ", strip=True)
                author_match = re.search(
                    r"Authors?:\s*([A-Z][a-zA-Z\-,. ]{3,100})", text
                )
                if author_match:
                    result.authors = author_match.group(1).strip()

            # Try to find citation count (look for common patterns)
            citation_match = re.search(
                r"(Cited by|Citations|Times cited)\s*:?\s*(\d+)",
                page_content,
                re.IGNORECASE,
            )
            if citation_match:
                result.citations = int(citation_match.group(2))

            # Try to find publication year (look for 4-digit year near 2000-2025)
            year_match = re.search(r"(19|20)\d{2}", page_content)
            if year_match:
                y = int(year_match.group(0))
                if 1990 < y <= 2025:
                    result.publication_year = y

        except Exception as e:
            logging.error(f"Error parsing page {result.url}: {e}")
    return search_results


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
        - site:researchgate.net [your topic] - search ResearchGate

    Combine operators:
        - "[exact phrase]" site:edu -opinion - exact phrase, only .edu sites, no opinion pieces

    File type search:
        - [topic] filetype:pdf - search for PDF files only
        - research filetype:pdf site:edu - PDFs from educational sites

    Time-based (add year to query):
        - [topic] research after:2020 - results after 2020
        - [topic] before:2015 - results before 2015
        - "[topic]" 2025 - recent results

    Examples (diverse research domains):
        - "neural networks" deep learning before:2020 site:researchgate.net
        - "CRISPR gene editing" site:edu filetype:pdf
        - quantum computing algorithms -tutorial
        - "climate change models" 2023 site:nature.com

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
