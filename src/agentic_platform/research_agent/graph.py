"""
Research Agent Graph

Assembles the complete research agent graph with all nodes, edges, and conditional routing.

Key features:
- Fan-out via Send API: one researcher per sub-question in parallel
- Conditional routing: quality check → pass or retry
- Checkpointing: SQLite persistence
- Human-in-the-loop: interrupt_before quality_check
"""

import logging
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, Send

from research_agent.state import ResearchState
from research_agent.nodes import (
    planner_node,
    researcher_node,
    synthesizer_node,
    quality_check_node,
    formatter_node,
)

logger = logging.getLogger(__name__)


def route_to_researchers(state: ResearchState) -> list:
    """
    Route from planner to researcher nodes (fan-out via Send).

    This demonstrates the Send API for parallel execution:
    each sub-question gets its own researcher instance.
    """
    return [
        Send("researcher", {"sub_question": sq})
        for sq in state.get("sub_questions", [])
    ]


def route_after_quality(state: ResearchState) -> str:
    """
    Route based on quality check result.

    - If passed: go to formatter
    - If failed and retries remain: go back to planner
    - If failed and no retries: go to formatter anyway
    """
    quality = state.get("quality", {})
    if quality.get("passed", False):
        logger.info("Quality check passed → formatter")
        return "formatter"
    elif state.get("retry_count", 0) < 2:
        logger.info("Quality check failed → replanning with feedback")
        return "planner"
    else:
        logger.warning("Max retries reached → formatter with lower quality")
        return "formatter"


def build_graph(
    checkpointer: Optional[SqliteSaver] = None,
    with_human_review: bool = True,
) -> any:
    """
    Build the research agent graph.

    Args:
        checkpointer: SQLite checkpointer for persistence. If None, no persistence.
        with_human_review: If True, pause before quality_check for human review.

    Returns:
        Compiled graph.Runnable
    """
    # Create the state graph
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("formatter", formatter_node)

    # Add edges
    # START → planner
    graph.add_edge(START, "planner")

    # Planner → researchers (fan-out via Send)
    graph.add_conditional_edges(
        "planner",
        route_to_researchers,
        ["researcher"],  # All Send calls target "researcher"
    )

    # Researchers → synthesizer (fan-in, automatic)
    graph.add_edge("researcher", "synthesizer")

    # Synthesizer → quality_check
    graph.add_edge("synthesizer", "quality_check")

    # Quality check → formatter or planner (conditional)
    graph.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {
            "formatter": "formatter",
            "planner": "planner",
        },
    )

    # Formatter → END
    graph.add_edge("formatter", END)

    # Compile the graph
    interrupt_before = ["quality_check"] if with_human_review else None

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )

    logger.info("Graph compiled successfully")
    return compiled


def visualize_graph():
    """
    Visualize the graph structure using Mermaid.

    Call this to see the graph topology.
    """
    graph = build_graph()
    try:
        mermaid = graph.get_graph().draw_mermaid()
        print(mermaid)
    except Exception as e:
        logger.warning(f"Could not visualize graph: {e}")
