"""
Quality check node: evaluates the draft report and decides if it passes.

Demonstrates:
- State inspection for conditional routing
- Scoring/evaluation logic
- Returning quality metadata
"""

import logging
from typing import Optional

from langchain_core.language_model import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.state import QualityScore, ResearchState

logger = logging.getLogger(__name__)


def quality_check_node(
    state: ResearchState, llm: Optional[BaseLanguageModel] = None
) -> dict:
    """
    Evaluate the quality of the draft report.

    Args:
        state: Current state with draft_report
        llm: Language model for evaluation

    Returns:
        Update dict with quality score and feedback
    """
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    logger.info("Evaluating draft report quality...")

    draft = state.get("draft_report", "")

    # Use LLM to score the report
    messages = [
        SystemMessage(
            content="You are a quality assurance evaluator for research reports. "
            "Evaluate reports on completeness, accuracy, and clarity. "
            "Response: Score (0-100) | Feedback | Passed (yes/no)"
        ),
        HumanMessage(
            content=f"""Topic: {state['topic']}

Draft Report:
{draft}

Evaluate this report on:
1. Completeness: Does it address the topic comprehensively?
2. Clarity: Is it well-structured and easy to understand?
3. Accuracy: Are claims supported by the research?
4. Relevance: Is all content relevant to the topic?

Provide a score (0-100), brief feedback, and whether it passes (yes/no)."""
        ),
    ]

    response = llm.invoke(messages)
    evaluation = response.content

    # Simple parsing (could be more robust with structured output)
    try:
        lines = evaluation.strip().split("\n")
        parts = lines[0].split("|")

        score_str = parts[0].strip().replace("Score ", "").split()[0]
        score = int(score_str)

        feedback = parts[1].strip() if len(parts) > 1 else "No feedback"
        passed = "yes" in evaluation.lower()

        # Require score > 70 to pass
        if score < 70:
            passed = False

    except (ValueError, IndexError):
        # Fallback if parsing fails
        score = 50
        feedback = "Unable to parse evaluation"
        passed = False

    quality: QualityScore = {
        "score": score,
        "feedback": feedback,
        "passed": passed,
    }

    logger.info(f"Quality check complete: Score={score}, Passed={passed}")

    return {"quality": quality}
