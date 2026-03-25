"""
Module 04 — Example 1: Checkpointing and Resumability

Shows how to persist graph state and resume interrupted runs.

Key concept: thread_id in config tells LangGraph which thread to save/resume.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


class LongRunningState(TypedDict):
    """State for a multi-step operation."""
    step: int
    data: str


def step_1(state: LongRunningState) -> dict:
    """First step."""
    print(f"Step 1: Starting with data '{state['data']}'")
    return {"step": 1, "data": state["data"] + " [step_1]"}


def step_2(state: LongRunningState) -> dict:
    """Second step."""
    print(f"Step 2: Processing {state['data']}")
    return {"step": 2, "data": state["data"] + " [step_2]"}


def step_3(state: LongRunningState) -> dict:
    """Third step."""
    print(f"Step 3: Final processing {state['data']}")
    return {"step": 3, "data": state["data"] + " [step_3]"}


def build_graph(checkpointer):
    """Build a graph with checkpointing."""
    graph = StateGraph(LongRunningState)

    graph.add_node("step_1", step_1)
    graph.add_node("step_2", step_2)
    graph.add_node("step_3", step_3)

    graph.add_edge(START, "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "step_3")
    graph.add_edge("step_3", END)

    # CRITICAL: pass checkpointer to enable persistence
    return graph.compile(checkpointer=checkpointer)


def demonstrate_checkpointing():
    """Show checkpointing and resumability."""
    print("=" * 70)
    print("CHECKPOINTING DEMO")
    print("=" * 70)
    print()

    # Use in-memory SQLite for demo
    checkpointer = SqliteSaver(":memory:")

    compiled = build_graph(checkpointer)

    # Setup initial state
    input_state = {
        "step": 0,
        "data": "initial",
    }

    # Use a thread_id to identify this run
    thread_id = "demo-run-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("RUN 1: First execution")
    print("-" * 70)

    # Run the graph
    result_1 = compiled.invoke(input_state, config=config)
    print(f"After run 1: {result_1}")
    print()

    # Now, simulate resuming the same thread
    # In a real scenario, the graph might have been interrupted,
    # and we want to resume it

    print("RUN 2: Resume the same thread (same thread_id)")
    print("-" * 70)

    # Create a new input (empty, since we're resuming)
    # The graph will load the previous state
    result_2 = compiled.invoke({}, config=config)
    print(f"After run 2 (resumed): {result_2}")
    print()

    print("=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("""
1. CHECKPOINTER
   Saves graph state after each step.

   With: SqliteSaver("path/to/db.db") → persists to disk
   Without: None → no persistence

2. THREAD_ID
   Identifies a unique run/conversation.

   config = {"configurable": {"thread_id": "run-123"}}

   Same thread_id → resume from saved state
   New thread_id → start fresh

3. USE CASES
   - Long-running processes that might timeout
   - Resumable workflows
   - Conversation history (each conversation = thread)
   - Debugging (replay a thread)

4. GETTING PREVIOUS STATE
   state = compiled.get_state(config)
   → Returns the last checkpoint for this thread

5. UPDATING STATE (for human-in-the-loop)
   compiled.update_state(config, {"field": new_value})
   → Modifies the saved state before resuming

6. IN PRODUCTION
   - Use SqliteSaver(db_path) for SQLite
   - Use langgraph-checkpoint-postgres for Postgres
   - Set appropriate cleanup policies for old checkpoints
    """)


if __name__ == "__main__":
    demonstrate_checkpointing()
