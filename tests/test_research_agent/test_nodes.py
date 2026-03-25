"""
Tests for research agent nodes.

Demonstrates how to test node functions in isolation.
"""

import pytest

from agentic_platform.research_agent.state import ResearchState
from agentic_platform.research_agent.nodes.planner import planner_node


def test_planner_node(sample_research_state, mock_llm):
    """Test the planner node creates sub-questions."""
    state = sample_research_state

    # Call planner node
    result = planner_node(state, llm=mock_llm)

    # Verify structure
    assert "sub_questions" in result
    assert isinstance(result["sub_questions"], list)
    assert "retry_count" in result

    # If mock LLM returns questions, verify count
    if result["sub_questions"]:
        for sq in result["sub_questions"]:
            assert "id" in sq
            assert "question" in sq
            assert "findings" in sq
            assert "sources" in sq


def test_planner_node_creates_ids(sample_research_state, mock_llm):
    """Verify planner creates valid sub-question IDs."""
    state = sample_research_state

    result = planner_node(state, llm=mock_llm)

    # All sub-questions should have unique IDs
    ids = [sq["id"] for sq in result["sub_questions"]]
    assert len(ids) == len(set(ids))  # All unique


def test_planner_increments_retry_count(sample_research_state, mock_llm):
    """Verify retry_count is incremented."""
    state = sample_research_state.copy()
    state["retry_count"] = 0

    result = planner_node(state, llm=mock_llm)

    assert result["retry_count"] == 1


def test_planner_preserves_topic(sample_research_state, mock_llm):
    """Verify planner doesn't modify the topic."""
    state = sample_research_state
    topic = state["topic"]

    result = planner_node(state, llm=mock_llm)

    # topic is not in result (node only returns what it changed)
    # but original state is unchanged
    assert state["topic"] == topic
