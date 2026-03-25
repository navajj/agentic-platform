"""
Module 01 — Example 3: Edges and Flow

Shows:
1. Multiple edges and graph structure
2. Graph visualization using Mermaid
3. How to inspect the graph topology
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class ValidationState(TypedDict):
    """State for a validation pipeline."""
    data: str
    is_valid: bool
    errors: list[str]


def process_node(state: ValidationState) -> dict:
    """Process the input data."""
    print(f"Processing: {state['data']}")
    # Just transform to uppercase for demo
    return {"data": state["data"].upper()}


def validate_node(state: ValidationState) -> dict:
    """Validate the processed data."""
    print(f"Validating: {state['data']}")
    # Simple validation: data must be non-empty
    is_valid = len(state["data"]) > 0
    errors = [] if is_valid else ["Data is empty"]
    return {"is_valid": is_valid, "errors": errors}


def format_node(state: ValidationState) -> dict:
    """Format the valid data."""
    print(f"Formatting: {state['data']}")
    return {"data": f"FORMATTED({state['data']})"}


def error_handler_node(state: ValidationState) -> dict:
    """Handle validation errors."""
    print(f"Handling errors: {state['errors']}")
    return {"data": "ERROR_STATE"}


def build_graph():
    """Build a graph with multiple paths."""
    graph = StateGraph(ValidationState)

    # Add nodes
    graph.add_node("process", process_node)
    graph.add_node("validate", validate_node)
    graph.add_node("format", format_node)
    graph.add_node("error_handler", error_handler_node)

    # Add edges
    graph.add_edge(START, "process")
    graph.add_edge("process", "validate")

    # Conditional edges: we'll implement this more fully in Module 02
    # For now, use simple edges
    graph.add_edge("validate", "format")  # Success path
    # graph.add_edge("validate", "error_handler")  # Error path (would be conditional)

    graph.add_edge("format", END)
    graph.add_edge("error_handler", END)

    return graph


if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH STRUCTURE DEMO")
    print("=" * 70)

    # Build the graph
    graph = build_graph()
    compiled = graph.compile()

    # Visualize the graph structure
    print("\n1. Mermaid Diagram:")
    print("-" * 70)
    try:
        mermaid_str = compiled.get_graph().draw_mermaid()
        print(mermaid_str)
    except Exception as e:
        print(f"(Mermaid visualization not available: {e})")

    # Run the graph
    print("\n2. Execution Trace:")
    print("-" * 70)

    input_state = {
        "data": "hello",
        "is_valid": False,
        "errors": [],
    }

    result = compiled.invoke(input_state)
    print(f"\nFinal state: {result}")

    # Inspect graph topology programmatically
    print("\n3. Graph Topology:")
    print("-" * 70)

    g = compiled.get_graph()
    print(f"Nodes: {list(g.nodes.keys())}")
    print(f"Edges: {list(g.edges)}")

    print("\n4. Node Details:")
    print("-" * 70)
    for node_name, node in g.nodes.items():
        print(f"  {node_name}: {node.metadata if hasattr(node, 'metadata') else 'user-defined'}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Graph topology is EXPLICIT:
   - add_node("name", fn) registers a node
   - add_edge(a, b) creates a transition
   - START and END are special nodes

2. Visualization helps understand flow:
   - Use get_graph().draw_mermaid() to visualize
   - Useful for debugging complex agents

3. You can inspect the graph programmatically:
   - compiled.get_graph().nodes
   - compiled.get_graph().edges
   - This is useful for testing and introspection

4. Next step: Conditional edges (Module 02)
   - So far all paths are fixed
   - Conditional edges let nodes decide which path to take
    """)
