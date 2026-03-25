"""
Synthesizer node: merges findings from all sub-questions into a draft report.

Demonstrates:
- Aggregating multi-source data
- LLM-based synthesis
- Structured output generation
"""

import logging
from typing import Optional

from langchain_core.language_model import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.state import ResearchState

logger = logging.getLogger(__name__)


def synthesizer_node(
    state: ResearchState, llm: Optional[BaseLanguageModel] = None
) -> dict:
    """
    Synthesize sub-question findings into a cohesive draft report.

    Args:
        state: Current research state with filled sub_questions
        llm: Language model to use

    Returns:
        Update dict with draft_report
    """
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    logger.info("Synthesizing findings into draft report...")

    # Format all findings
    findings_text = ""
    for sq in state.get("sub_questions", []):
        findings_text += f"\n### {sq['question']}\n{sq.get('findings', 'No findings.')}\n"

    # Use LLM to synthesize into a report
    messages = [
        SystemMessage(
            content="You are a research analyst. Create a well-structured research report "
            "that synthesizes findings into a coherent narrative."
        ),
        HumanMessage(
            content=f"""Topic: {state['topic']}

Research Findings:
{findings_text}

Create a comprehensive report that:
1. Provides an executive summary
2. Discusses key findings organized by sub-topic
3. Highlights connections between findings
4. Concludes with implications and recommendations

Format the report in clear sections with headers."""
        ),
    ]

    response = llm.invoke(messages)
    draft_report = response.content

    logger.info("Draft report completed")

    return {
        "draft_report": draft_report,
        "quality": None,
        "human_feedback": None,
    }
