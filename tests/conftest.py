"""
Test fixtures and configuration.
"""

import pytest
from langchain_core.language_model import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage

from agentic_platform.research_agent.state import ResearchState


class MockLLM(BaseLanguageModel):
    """Mock LLM for testing (no API calls)."""

    def _generate(self, messages, **kwargs):
        # Simple mock: return a fixed response
        from langchain_core.outputs import LLMResult, Generation

        return LLMResult(generations=[[Generation(text="Mock response")]])

    def _llm_type(self):
        return "mock"

    @property
    def _identifying_params(self):
        return {}

    def invoke(self, messages, **kwargs):
        # Mock response based on message content
        from langchain_core.messages import AIMessage

        if messages and hasattr(messages[-1], "content"):
            content = messages[-1].content.lower()
        else:
            content = ""

        # Simple mocking logic
        if "question" in content or "sub-question" in content:
            return AIMessage(
                content='{"questions": [{"question": "What is topic?", "id": "q1"}, '
                '{"question": "How does topic work?", "id": "q2"}, '
                '{"question": "Why is topic important?", "id": "q3"}]}'
            )
        elif "score" in content or "evaluate" in content:
            return AIMessage(content="Score 85 | Good quality | yes")
        else:
            return AIMessage(content="Mock response: This is a test response.")


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM."""
    return MockLLM()


@pytest.fixture
def sample_research_state():
    """Fixture providing a sample research state."""
    state: ResearchState = {
        "topic": "Test Topic",
        "max_sub_questions": 3,
        "sub_questions": [],
        "messages": [],
        "draft_report": "",
        "quality": None,
        "human_feedback": None,
        "retry_count": 0,
        "error": None,
        "final_report": "",
    }
    return state
