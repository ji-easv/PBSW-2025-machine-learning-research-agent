class SearchResult:
    def __init__(
        self,
        query: str,
        url: str,
        title: str,
        snippet: str,
        authors: list[str] | None = None,
        publication_year: int | None = None,
        citations: int | None = None,
    ) -> None:
        self.query = query
        self.url = url
        self.title = title
        self.snippet = snippet
        self.citations = citations
        self.authors = authors
        self.publication_year = publication_year
