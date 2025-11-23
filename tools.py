import requests
import xml.etree.ElementTree as ET
from typing import List
from bs4 import BeautifulSoup


def search_web(query: str, num_results: int = 10) -> str:
    """
    Search the web using DuckDuckGo.

    Supports advanced search syntax:

    Exact phrases:
        - Use quotes: "speed bumps" - searches for the exact phrase

    Exclude words:
        - Use minus: speed bumps -motorcycle - excludes results with "motorcycle"

    Site-specific search:
        - site:edu speed bumps - search only educational sites
        - site:arxiv.org machine learning - search only arxiv.org

    Combine operators:
        - "traffic safety" site:edu -opinion - exact phrase, only .edu sites, no opinion pieces

    File type search:
        - speed bumps filetype:pdf - search for PDF files only
        - research filetype:pdf site:edu - PDFs from educational sites

    Time-based (add year to query):
        - speed bumps research 2020 - likely to return results from that year
        - "machine learning" 2024 - recent results

    Examples:
        - "speed bumps" safety research
        - traffic calming site:edu filetype:pdf
        - "neural networks" 2024 -tutorial
        - machine learning site:arxiv.org

    Args:
        query: Search query (supports syntax above)
        num_results: Maximum number of results to return (default: 10)

    Returns:
        Formatted string with search results including titles, URLs, and snippets.
    """
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.post(url, data=params, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for i, result in enumerate(soup.find_all("div", class_="result"), 1):
            if i > num_results:
                break

            title_elem = result.find("a", class_="result__a")
            snippet_elem = result.find("a", class_="result__snippet")

            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get("href", "")
                snippet = (
                    snippet_elem.get_text(strip=True)
                    if snippet_elem
                    else "No description"
                )

                results.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

        if not results:
            return f"No results found for query: '{query}'"

        return "\n".join(results)

    except Exception as e:
        return f"Error searching DuckDuckGo: {str(e)}"


api_base_url = "https://export.arxiv.org/api/query"


def search_research_papers_api(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> str:
    """
    Search arXiv for research papers.

    Args:
        query: Search query. Can use field prefixes like:
               - ti: (title)
               - au: (author)
               - abs: (abstract)
               - cat: (category)
               - all: (all fields)
               Example: "ti:speed bumps AND cat:physics"
        max_results: Number of results to return (default: 10)
        sort_by: Sort by "relevance", "lastUpdatedDate", or "submittedDate"
        sort_order: "ascending" or "descending"

    Returns:
        Formatted string with parsed paper information or helpful error message
    """
    try:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        response = requests.get(api_base_url, params=params, timeout=30)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)

        # Define namespaces
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        # Get total results
        total_results = root.find("opensearch:totalResults", ns)
        total_count = (
            int(total_results.text)
            if total_results is not None and total_results.text
            else 0
        )

        # Get entries
        entries = root.findall("atom:entry", ns)

        if total_count == 0 or not entries:
            return f"No results found for query: '{query}'"

        # Format results
        results = [f"Found {total_count} total results. Showing top {len(entries)}:\n"]

        for i, entry in enumerate(entries, 1):
            title = entry.find("atom:title", ns)
            title_text = "No title"
            if title is not None and title.text:
                title_text = title.text.strip().replace("\n", " ")

            # Get authors
            authors = entry.findall("atom:author", ns)
            author_names: List[str] = []
            for a in authors:
                name_elem = a.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    author_names.append(name_elem.text)

            authors_text = ", ".join(author_names[:3])  # First 3 authors
            if len(author_names) > 3:
                authors_text += f" et al. ({len(author_names)} total)"

            # Get published date
            published = entry.find("atom:published", ns)
            pub_date = "Unknown"
            if published is not None and published.text:
                pub_date = published.text[:10]

            # Get arXiv ID
            id_elem = entry.find("atom:id", ns)
            arxiv_id = "Unknown"
            if id_elem is not None and id_elem.text:
                arxiv_id = id_elem.text.replace("http://arxiv.org/abs/", "")

            # Get summary (abstract)
            summary = entry.find("atom:summary", ns)
            summary_text = "No abstract"
            if summary is not None and summary.text:
                summary_text = summary.text.strip()[:200] + "..."

            # Get primary category
            primary_cat = entry.find("arxiv:primary_category", ns)
            category = primary_cat.get("term") if primary_cat is not None else "Unknown"

            # Get PDF link
            pdf_link = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_link = link.get("href")
                    break

            results.append(
                f"\n{i}. {title_text}\n"
                f"   Authors: {authors_text}\n"
                f"   Published: {pub_date}\n"
                f"   arXiv ID: {arxiv_id}\n"
                f"   Category: {category}\n"
                f"   URL: https://arxiv.org/abs/{arxiv_id}\n"
                f"   PDF: {pdf_link or 'N/A'}\n"
                f"   Abstract: {summary_text}\n"
            )

        return "\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"Error accessing arXiv API: {str(e)}\nPlease try again later."
    except ET.ParseError as e:
        return f"Error parsing arXiv response: {str(e)}\nThe API may have returned invalid data."
    except Exception as e:
        return f"Unexpected error: {str(e)}"
