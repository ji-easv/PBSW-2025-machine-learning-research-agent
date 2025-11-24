from mistralai import dataclass


@dataclass
class SearchResult:
    query: str
    url: str
    title: str
    snippet: str
    authors: list[str] | None = None
    publication_year: int | None = None
    citations: int | None = None
