"""
Research agent tools for external information gathering.

Includes:
- Web search (via Tavily or DuckDuckGo)
- Academic paper search (via Arxiv)
- Basic calculator
"""

from .search import web_search
from .arxiv_tool import arxiv_search
from .calculator import calculator

__all__ = ["web_search", "arxiv_search", "calculator"]
