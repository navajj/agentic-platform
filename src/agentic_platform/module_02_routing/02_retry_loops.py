"""
Module 02 — Example 2: Retry Loops

Demonstrates how LangGraph makes retry loops natural.

This is IMPOSSIBLE in LangChain chains (they're linear).
LangGraph makes it easy with conditional_edges.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class ValidationState(TypedDict):
    """State for a validation retry loop."""
    data: str
    is_valid: bool
    attempts: int
    error_message: str


def process_node(state: ValidationState) -> dict:
    """Try to process the data."""
    print(f"Attempt {state['attempts']}: Processing '{state['data']}'")

    # Simple validation: data must be all uppercase
    is_valid = state["data"].isupper()
    error = "" if is_valid else "Data must be all uppercase"

    return {
        "is_valid": is_valid,
        "error_message": error,
        "attempts": state["attempts"] + 1,
    }


def fix_node(state: ValidationState) -> dict:
    """Fix the data if it's not valid."""
    print(f"Fixing data: '{state['data']}' → '{state['data'].upper()}'")
    return {"data": state["data"].upper()}


def success_node(state: ValidationState) -> dict:
    """Handle success."""
    print(f"✓ SUCCESS: Data is valid after {state['attempts']} attempts")
    return {}


def failure_node(state: ValidationState) -> dict:
    """Handle permanent failure."""
    print(f"✗ FAILURE: Could not validate after {state['attempts']} attempts")
    return {}


def route_after_validation(state: ValidationState) -> str:
    """
    Decide: retry or succeed?

    - If valid: success
    - If invalid and retries remain: fix and retry
    - If max retries reached: failure
    """
    if state["is_valid"]:
        return "success"
    elif state["attempts"] < 3:
        return "fix"  # → fix_node → back to process_node
    else:
        return "failure"


def build_graph():
    """Build a retry loop graph."""
    graph = StateGraph(ValidationState)

    # Add nodes
    graph.add_node("process", process_node)
    graph.add_node("fix", fix_node)
    graph.add_node("success", success_node)
    graph.add_node("failure", failure_node)

    # Add edges
    graph.add_edge(START, "process")

    # CONDITIONAL ROUTING: After validation, decide what to do
    graph.add_conditional_edges(
        "process",
        route_after_validation,
        {
            "success": "success",
            "fix": "fix",
            "failure": "failure",
        },
    )

    # THE KEY: fix node → back to process (retry loop!)
    graph.add_edge("fix", "process")

    graph.add_edge("success", END)
    graph.add_edge("failure", END)

    return graph.compile()


if __name__ == "__main__":
    print("=" * 70)
    print("RETRY LOOPS DEMO")
    print("=" * 70)
    print()

    compiled = build_graph()

    # Test case 1: Already valid
    print("TEST 1: Already valid data")
    print("-" * 70)
    result = compiled.invoke(
        {
            "data": "HELLO",
            "is_valid": False,
            "attempts": 0,
            "error_message": "",
        }
    )
    print()

    # Test case 2: Needs fixing
    print("TEST 2: Data that needs fixing")
    print("-" * 70)
    result = compiled.invoke(
        {
            "data": "hello",
            "is_valid": False,
            "attempts": 0,
            "error_message": "",
        }
    )
    print()

    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. RETRY LOOPS ARE NATURAL IN LANGGRAPH
   In LangChain, you'd need to:
   - Wrap everything in a while loop
   - Manually track state
   - Use get_state() / get_lc_namespace() tricks

   In LangGraph, you just:
   - Add a conditional edge back to the same node
   - Return a routing decision

2. GRAPH STRUCTURE FOR RETRIES
   START → process
      ↓
   [conditional]
      ├→ success → END
      ├→ fix → [back to process] (LOOP!)
      └→ failure → END

3. THIS ENABLES:
   - Validation retries with feedback
   - LLM re-planning loops
   - Multi-attempt strategies
   - Error recovery

4. KEY PATTERN
   def route_fn(state) -> str:
       if state["is_valid"]:
           return "success"
       elif state["attempts"] < max_attempts:
           return "fix"   # Returns to START of fix → process loop
       else:
           return "failure"

   graph.add_edge("fix", "process")  # Close the loop
    """)
