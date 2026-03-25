"""
Module 01 — Example 1: Hello Graph

The absolute minimum LangGraph example.
Shows: StateGraph, add_node, add_edge, set_entry_point, compile, invoke

This is what a "hello world" looks like in LangGraph.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# Step 1: Define your state schema
class State(TypedDict):
    """Minimal state: just a message string."""
    message: str


# Step 2: Define node functions
# Each node receives the full state and returns a dict of only the keys it changed.
def node_a(state: State) -> dict:
    """First node: append to message."""
    print(f"Node A running. Current message: {state['message']}")
    return {"message": state["message"] + " processed by A"}


def node_b(state: State) -> dict:
    """Second node: append to message."""
    print(f"Node B running. Current message: {state['message']}")
    return {"message": state["message"] + " -> processed by B"}

def node_c(state: State) -> dict:
    """Third node: append to message."""
    print(f"Node C running. Current message: {state['message']}")
    return {"message": state["message"] + " -> processed by C"}

# Step 3: Build the graph
def build_graph():
    """Construct the StateGraph."""
    graph = StateGraph(State)

    # Add nodes (node name, node function)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_node("node_c", node_c)

    # Add edges (from -> to)
    # START is a special constant meaning "start here"
    # END is a special constant meaning "we're done"
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", "node_c")
    graph.add_edge("node_c", END)

    # Compile: returns a Runnable (same interface as LangChain)
    return graph.compile()


if __name__ == "__main__":
    # Build and run
    compiled = build_graph()

    # Invoke with initial state
    input_state = {"message": "Hello"}
    result = compiled.invoke(input_state)

    print(f"\nFinal state: {result}")
    # Expected: "Hello processed by A -> processed by B"

    # Key differences from LangChain:
    # 1. Explicit graph construction (nodes + edges)
    # 2. Each node returns a partial dict (only changed keys)
    # 3. START and END are explicit
    # 4. invoke() interface is the same, but the graph is graph-first, not chain-first
