"""
Academic paper search tool using ArXiv.
"""

from langchain_core.tools import tool


@tool
def arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Search ArXiv for academic papers.

    Args:
        query: The search query (e.g., "quantum computing")
        max_results: Maximum number of papers to return

    Returns:
        Formatted list of papers with title, authors, abstract
    """
    try:
        import arxiv

        # Create a search query
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in search.results():
            result_str = (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join([author.name for author in paper.authors])}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"ArXiv ID: {paper.arxiv_id}\n"
                f"Abstract: {paper.summary[:500]}...\n"
            )
            results.append(result_str)

        return "\n---\n".join(results) if results else "No papers found."

    except Exception as e:
        return f"Error searching ArXiv: {e}"
