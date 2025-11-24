from ddgs import DDGS


def search_web(query: str, num_results: int = 10) -> str:
    """
    Search the web using DuckDuckGo.

    Supports advanced search syntax:

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

        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=num_results))

            if not search_results:
                return (
                    f"No results found for query: '{query}'. "
                    "Consider trying simpler keywords, synonyms, or domain filters "
                    "(e.g., site:edu, site:gov), or using a research-specific API if "
                    "you need academic papers."
                )

            for i, result in enumerate(search_results, 1):
                title = result.get("title", "No title")
                url = result.get("href", result.get("link", "No URL"))
                snippet = result.get(
                    "body", result.get("description", "No description")
                )

                results.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

        return "\n".join(results)

    except ImportError:
        return "Error: ddgs library not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Error searching DuckDuckGo: {str(e)}"
