import requests
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional


# arXiv subject categories and related keywords
ARXIV_SUBJECTS = {
    "physics": [
        "physics",
        "astrophysics",
        "astronomy",
        "cosmology",
        "relativity",
        "quantum",
        "particle",
        "nuclear",
        "plasma",
        "optics",
        "fluid dynamics",
        "thermodynamics",
        "mechanics",
        "electromagnetism",
        "condensed matter",
        "superconductivity",
        "photonics",
        "acoustics",
        "waves",
        "gravity",
    ],
    "mathematics": [
        "mathematics",
        "algebra",
        "geometry",
        "topology",
        "calculus",
        "analysis",
        "number theory",
        "combinatorics",
        "optimization",
        "probability",
        "statistics",
        "differential equations",
        "numerical methods",
        "graph theory",
        "logic",
        "mathematical modeling",
    ],
    "computer_science": [
        "computer science",
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "neural network",
        "algorithm",
        "data structure",
        "programming",
        "software",
        "computer vision",
        "natural language processing",
        "nlp",
        "robotics",
        "cryptography",
        "cybersecurity",
        "database",
        "distributed systems",
        "networking",
        "graphics",
        "human computer interaction",
        "information retrieval",
        "computational",
    ],
    "quantitative_biology": [
        "biology",
        "genomics",
        "bioinformatics",
        "molecular",
        "cellular",
        "neuroscience",
        "ecology",
        "evolution",
        "population genetics",
        "systems biology",
        "proteomics",
        "biophysics",
        "computational biology",
    ],
    "quantitative_finance": [
        "finance",
        "economics",
        "financial modeling",
        "portfolio",
        "risk management",
        "derivatives",
        "trading",
        "market",
        "econometrics",
        "financial mathematics",
    ],
    "statistics": [
        "statistics",
        "statistical",
        "bayesian",
        "inference",
        "regression",
        "time series",
        "multivariate",
        "hypothesis testing",
        "sampling",
        "statistical learning",
        "data analysis",
    ],
    "electrical_engineering": [
        "signal processing",
        "image processing",
        "audio processing",
        "control systems",
        "electrical engineering",
        "circuits",
        "telecommunications",
        "radar",
    ],
}

# Topics NOT typically in arXiv
NON_ARXIV_TOPICS = [
    "civil engineering",
    "mechanical engineering",
    "traffic",
    "road",
    "transportation",
    "construction",
    "architecture",
    "urban planning",
    "speed bump",
    "speed hump",
    "medicine",
    "clinical",
    "health",
    "disease",
    "treatment",
    "therapy",
    "patient",
    "psychology",
    "sociology",
    "anthropology",
    "education",
    "business",
    "management",
    "marketing",
    "accounting",
    "law",
    "political science",
    "history",
    "literature",
    "chemistry",
    "materials science",
    "aerospace",
    "automotive",
    "manufacturing",
]


def is_arxiv_suitable(query: str) -> Tuple[Optional[bool], str]:
    """
    Determine if a research topic is likely to be found in arXiv.

    Args:
        query: The search query or topic

    Returns:
        Tuple of (is_suitable: Optional[bool], reason: str)
        - True: definitely in arXiv
        - False: definitely NOT in arXiv
        - None: uncertain, may or may not be in arXiv
    """
    query_lower = query.lower()

    # Check for explicitly non-arXiv topics
    for non_topic in NON_ARXIV_TOPICS:
        if non_topic in query_lower:
            return (
                False,
                f"Topic '{non_topic}' is not typically covered by arXiv (which focuses on physics, math, CS, etc.)",
            )

    # Check for arXiv-suitable topics
    matching_subjects = []
    for subject, keywords in ARXIV_SUBJECTS.items():
        for keyword in keywords:
            if keyword in query_lower:
                matching_subjects.append(subject)
                break

    if matching_subjects:
        return True, f"Topic matches arXiv subjects: {', '.join(matching_subjects)}"

    # If no clear match, default to uncertain (will try arXiv but with low confidence)
    return (
        None,
        "Topic may or may not be in arXiv - suggest trying web search or Semantic Scholar as a primary method",
    )


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
               Examples: "ti:quantum computing AND cat:quant-ph"
                        "au:Hinton AND ti:neural networks"
                        "abs:reinforcement learning"
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
            return (
                f"No results found for query: '{query}'. "
                "This topic may be outside arXiv's typical subject areas "
                "(physics, math, CS, etc.). Consider using Semantic Scholar or "
                "a general web search for this topic."
            )

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


def search_semantic_scholar(
    query: str,
    min_citations: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
    fields_of_study: List[str] | None = None,
) -> str:
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
    topic is outside arXiv's core areas (e.g., traffic, civil engineering).

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
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

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

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "data" not in data or not data["data"]:
            return (
                f"No results found for query: '{query}' with specified filters. "
                f"Filters used: year={year_filter_desc}, min_citations={min_citations}, "
                f"max_results={max_results}. Consider broadening the query, relaxing "
                f"the citation threshold, or widening the year range."
            )

        papers = data["data"]
        total = data.get("total", len(papers))

        header_filters = f"year={year_filter_desc}, min_citations={min_citations}, max_results={max_results}"
        results = [
            f"Found {total} total results. Showing top {len(papers)} (filters: {header_filters}):\n"
        ]

        for i, paper in enumerate(papers, 1):
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

            # Get DOI if available
            external_ids = paper.get("externalIds", {})
            doi = external_ids.get("DOI", "N/A")

            # Get abstract
            abstract = paper.get("abstract", "No abstract available")
            if abstract and len(abstract) > 200:
                abstract = abstract[:200] + "..."

            # Get fields of study
            fields = paper.get("fieldsOfStudy", [])
            fields_text = ", ".join(fields) if fields else "Unknown"

            results.append(
                f"\n{i}. {title}\n"
                f"   Authors: {authors_text}\n"
                f"   Year: {year}\n"
                f"   Citations: {citations}\n"
                f"   Fields: {fields_text}\n"
                f"   URL: {url}\n"
                f"   DOI: {doi}\n"
                f"   Abstract: {abstract}\n"
            )

        return "\n".join(results)

    except requests.exceptions.RequestException as e:
        return (
            f"Error accessing Semantic Scholar API: {str(e)}\nPlease try again later."
        )
    except Exception as e:
        return f"Unexpected error: {str(e)}"
