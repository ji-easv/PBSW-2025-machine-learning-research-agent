import logging
from ratelimit import limits, sleep_and_retry
import time
import requests
from typing import List

from datamodel.search_result import SearchResult

base_url = "https://api.semanticscholar.org/graph/v1/paper/search"


@sleep_and_retry
@limits(calls=1, period=10)
def search_semantic_scholar(
    query: str,
    min_citations: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
    fields_of_study: List[str] | None = None,
) -> List[SearchResult]:
    """
    Search Semantic Scholar for research papers with citation data.

    Semantic Scholar covers ALL academic disciplines and provides:
    - Citation counts
    - Publication years
    - Author information
    - Abstracts
    - DOI and URLs

    Use this when citation thresholds (e.g., "over 10 citations") or
    year constraints (e.g., "after 2003") are important, or when the

    If no results are found, the message will echo the effective filters
    (query, year range, minimum citations) so the caller can decide how
    to relax constraints or adjust the query.

    Args:
        query: Search query (keywords, phrases)
        min_citations: Minimum number of citations (default: 0)
        year_from: Earliest publication year (e.g., 2003)
        year_to: Latest publication year (e.g., 2023)
        max_results: Maximum number of results (default: 10)
        fields_of_study: List of fields to filter (e.g., ["Engineering", "Computer Science"])

    Returns:
        Formatted string with paper information including citation counts
    """
    try:
        data = call_semantic_scholar_api(
            query=query,
            min_citations=min_citations,
            year_from=year_from,
            year_to=year_to,
            max_results=max_results,
            fields_of_study=fields_of_study,
        )

        if not data or "data" not in data or not data["data"]:
            logging.info(
                "No results found for query: '%s' with filters year_from=%s, year_to=%s, min_citations=%d, max_results=%d.",
                query,
                year_from,
                year_to,
                min_citations,
                max_results,
            )
            return []

        papers = data["data"]
        results: List[SearchResult] = []

        for paper in papers:
            title = paper.get("title", "No title")
            year = paper.get("year", "Unknown")
            citations = paper.get("citationCount", 0)

            # Get authors
            authors = paper.get("authors", [])
            author_names = [a.get("name", "Unknown") for a in authors[:3]]
            authors_text = ", ".join(author_names)
            if len(authors) > 3:
                authors_text += f" et al. ({len(authors)} total)"

            # Get URL
            url = paper.get("url", "N/A")

            # Get abstract
            abstract = paper.get("abstract", "No abstract available")
            if abstract and len(abstract) > 200:
                abstract = abstract[:200] + "..."

            results.append(
                SearchResult(
                    query=query,
                    url=url,
                    title=title,
                    snippet=abstract,
                    authors=author_names,
                    publication_year=year if isinstance(year, int) else None,
                    citations=citations,
                )
            )

        return results

    except requests.exceptions.RequestException as e:
        logging.error("Error accessing Semantic Scholar API: %s", str(e))
        return []
    except Exception as e:
        logging.error("Unexpected error: %s", str(e))
        return []


def call_semantic_scholar_api(
    query: str,
    min_citations: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
    fields_of_study: List[str] | None = None,
) -> dict | None:
    # Build query parameters
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,citationCount,abstract,url,externalIds,fieldsOfStudy",
    }

    year_filter_desc = "none"
    if year_from:
        params["year"] = f"{year_from}-"
        year_filter_desc = f">= {year_from}"
        if year_to:
            params["year"] = f"{year_from}-{year_to}"
            year_filter_desc = f"{year_from}–{year_to}"

    if min_citations > 0:
        params["minCitationCount"] = min_citations

    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    for attempt in range(1, 4):
        logging.info(
            "Calling Semantic Scholar API (attempt %d) with query: '%s', year=%s, min_citations=%d, max_results=%d.",
            attempt,
            query,
            year_filter_desc,
            min_citations,
            max_results,
        )
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                logging.error(
                    "Rate limit exceeded when calling Semantic Scholar API on attempt %d.",
                    attempt,
                )
                sleep_seconds = 10 * attempt
                logging.info(
                    "Sleeping for %d seconds before retrying...", sleep_seconds
                )
                time.sleep(sleep_seconds)
            else:
                logging.error(
                    "Error calling Semantic Scholar API on attempt %d: %s",
                    attempt,
                    str(e),
                )
            if attempt == 3:
                raise
