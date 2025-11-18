import os
import requests


def search_web(query: str, num_results: int = 50) -> str:
    """
    Search the web using SearXNG instance.

    Args:
        query: The search query string
        num_results: Maximum number of results to return (default: 50)

    Returns:
        Formatted string with search results
    """
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")

    try:
        response = requests.get(
            f"{searxng_url}/search",
            params={
                "q": query,
                "format": "json",
                "language": "en",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return f"No results found for query: '{query}'"

        # Format the results
        formatted_results = [f"Web search results for '{query}':\n"]
        for i, result in enumerate(results[:num_results], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No description")

            formatted_results.append(
                f"{i}. {title}\n" f"   URL: {url}\n" f"   {content}\n"
            )

        return "\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        return f"Error searching the web: {str(e)}\nMake sure SearXNG is running at {searxng_url}"


def search_research_papers_api(query: str) -> str:
    # Placeholder function to simulate research paper search
    return f"Research papers for '{query}'"
