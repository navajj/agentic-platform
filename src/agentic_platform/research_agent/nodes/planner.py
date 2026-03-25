"""
Planner node: breaks a research topic into sub-questions.

Demonstrates:
- LLM integration
- Structured output (Pydantic models)
- Returning state updates
"""

import json
import logging
from typing import Optional

from langchain_core.language_model import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.state import ResearchState, SubQuestion

logger = logging.getLogger(__name__)


class SubQuestionList(BaseModel):
    """Structured output for sub-questions."""

    questions: list[dict] = Field(description="List of sub-questions")


def planner_node(
    state: ResearchState, llm: Optional[BaseLanguageModel] = None
) -> dict:
    """
    Plan sub-questions for the research topic.

    Args:
        state: Current research state
        llm: Language model to use (for testing)

    Returns:
        Update dict with sub_questions field
    """
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    logger.info(f"Planning sub-questions for topic: {state['topic']}")

    # Generate sub-questions
    messages = [
        SystemMessage(
            content="You are a research assistant. Break down research topics into "
            "focused sub-questions. Return valid JSON only."
        ),
        HumanMessage(
            content=f"""Topic: {state['topic']}

Generate {state.get('max_sub_questions', 3)} specific, research-focused sub-questions.

Return ONLY valid JSON in this format:
{{
  "questions": [
    {{"question": "What is...", "id": "q1"}},
    {{"question": "How does...", "id": "q2"}},
    {{"question": "Why is...", "id": "q3"}}
  ]
}}

Make questions specific, answerable, and complementary."""
        ),
    ]

    response = llm.invoke(messages)
    content = response.content

    # Parse JSON from response
    try:
        # Try to extract JSON from the response
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        json_str = content[start_idx:end_idx]
        parsed = json.loads(json_str)
        questions_data = parsed.get("questions", [])
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        # Fallback: create generic sub-questions
        questions_data = [
            {"question": f"What are the key aspects of {state['topic']}?", "id": "q1"},
            {
                "question": f"What is the historical context of {state['topic']}?",
                "id": "q2",
            },
            {"question": f"What are current developments in {state['topic']}?", "id": "q3"},
        ]

    # Convert to SubQuestion objects
    sub_questions: list[SubQuestion] = []
    for q in questions_data:
        sub_q: SubQuestion = {
            "id": q.get("id", f"q{len(sub_questions) + 1}"),
            "question": q.get("question", ""),
            "findings": "",
            "sources": [],
        }
        sub_questions.append(sub_q)

    logger.info(f"Generated {len(sub_questions)} sub-questions")

    return {
        "sub_questions": sub_questions,
        "retry_count": state.get("retry_count", 0) + 1,
    }
