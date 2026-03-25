"""
Web search tool using Tavily (primary) or DuckDuckGo (fallback).
"""

import os
from typing import Optional

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if tavily_api_key:
        return _tavily_search(query, max_results)
    else:
        return _duckduckgo_search(query, max_results)


def _tavily_search(query: str, max_results: int) -> str:
    """Search using Tavily API."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query, max_results=max_results)

        results = []
        for result in response.get("results", []):
            results.append(
                f"Title: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content']}\n"
            )

        return "\n---\n".join(results) if results else "No results found."

    except Exception as e:
        return f"Error searching with Tavily: {e}. Falling back to DuckDuckGo."


def _duckduckgo_search(query: str, max_results: int) -> str:
    """Search using DuckDuckGo (free, no API key needed)."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                results.append(
                    f"Title: {result['title']}\n"
                    f"URL: {result['href']}\n"
                    f"Content: {result['body']}\n"
                )

        return "\n---\n".join(results) if results else "No results found."

    except Exception as e:
        return f"Error searching with DuckDuckGo: {e}"
