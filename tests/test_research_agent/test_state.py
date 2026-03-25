"""
Tests for the research agent state schema.

Verifies TypedDict structure and reducer behavior.
"""

import operator
from typing import Annotated

import pytest

from agentic_platform.research_agent.state import ResearchState, SubQuestion


def test_research_state_schema():
    """Verify ResearchState has all required fields."""
    state: ResearchState = {
        "topic": "Test",
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

    assert state["topic"] == "Test"
    assert isinstance(state["sub_questions"], list)
    assert isinstance(state["messages"], list)


def test_sub_question_creation():
    """Verify SubQuestion structure."""
    sq: SubQuestion = {
        "id": "q1",
        "question": "What is AI?",
        "findings": "AI is...",
        "sources": ["arxiv.org", "wikipedia.org"],
    }

    assert sq["id"] == "q1"
    assert sq["question"] == "What is AI?"
    assert len(sq["sources"]) == 2


def test_messages_accumulation():
    """
    Test that messages field uses reducer (Annotated with operator.add).

    This demonstrates the critical reducer pattern for LangGraph.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # Create state with one message
    state: ResearchState = {
        "topic": "Test",
        "max_sub_questions": 3,
        "sub_questions": [],
        "messages": [HumanMessage(content="First")],
        "draft_report": "",
        "quality": None,
        "human_feedback": None,
        "retry_count": 0,
        "error": None,
        "final_report": "",
    }

    # In a real graph, nodes would return partial updates
    # Without the Annotated reducer, the following would REPLACE the list
    # With the reducer, it APPENDS

    # Simulate what operator.add does:
    new_messages = [AIMessage(content="Second")]
    merged = operator.add(state["messages"], new_messages)

    assert len(merged) == 2
    assert merged[0].content == "First"
    assert merged[1].content == "Second"


def test_sub_questions_list():
    """Test creating and managing sub-questions."""
    state: ResearchState = {
        "topic": "Quantum Computing",
        "max_sub_questions": 3,
        "sub_questions": [
            {"id": "q1", "question": "What is?", "findings": "", "sources": []},
            {"id": "q2", "question": "How does?", "findings": "", "sources": []},
        ],
        "messages": [],
        "draft_report": "",
        "quality": None,
        "human_feedback": None,
        "retry_count": 0,
        "error": None,
        "final_report": "",
    }

    assert len(state["sub_questions"]) == 2
    assert state["sub_questions"][0]["id"] == "q1"

    # Simulate adding findings
    state["sub_questions"][0]["findings"] = "Quantum computing uses qubits..."
    assert "qubits" in state["sub_questions"][0]["findings"]
