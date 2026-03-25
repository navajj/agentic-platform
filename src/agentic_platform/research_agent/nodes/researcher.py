"""
Researcher node: searches for answers to sub-questions.

Demonstrates:
- Tool calling (web_search + arxiv_search)
- Handling multiple tool results
- Returning state updates
"""

import logging
from typing import Optional

from langchain_core.language_model import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from research_agent.state import ResearchState, SubQuestion
from research_agent.tools import web_search, arxiv_search

logger = logging.getLogger(__name__)


def researcher_node(
    state: ResearchState, sub_question: SubQuestion, llm: Optional[BaseLanguageModel] = None
) -> dict:
    """
    Research a single sub-question using web and academic searches.

    Args:
        state: Current research state
        sub_question: The specific sub-question to research
        llm: Language model (for synthesis)

    Returns:
        Update dict with findings for this sub-question
    """
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    logger.info(f"Researching: {sub_question['question']}")

    # Perform searches
    web_results = web_search.invoke({"query": sub_question["question"], "max_results": 3})
    arxiv_results = arxiv_search.invoke(
        {"query": sub_question["question"], "max_results": 2}
    )

    combined_results = f"Web Results:\n{web_results}\n\nAcademic Papers:\n{arxiv_results}"

    # Use LLM to synthesize the findings into concise text
    messages = [
        SystemMessage(
            content="You are a research assistant. Synthesize research findings into "
            "a clear, factual summary."
        ),
        HumanMessage(
            content=f"""Question: {sub_question['question']}

Research Results:
{combined_results}

Summarize the key findings in 2-3 sentences, focusing on the most relevant information."""
        ),
    ]

    response = llm.invoke(messages)
    findings = response.content

    logger.info(f"Completed research for: {sub_question['id']}")

    # Update the specific sub-question with findings
    updated_sub_questions = []
    for sq in state.get("sub_questions", []):
        if sq["id"] == sub_question["id"]:
            updated_sq = sq.copy()
            updated_sq["findings"] = findings
            updated_sq["sources"] = [
                "web_search",
                "arxiv_search",
            ]  # Simplified for demo
            updated_sub_questions.append(updated_sq)
        else:
            updated_sub_questions.append(sq)

    return {
        "sub_questions": updated_sub_questions,
    }
