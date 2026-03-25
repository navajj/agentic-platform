"""
Module 02 — Example 1: Conditional Edges

Shows how to route between nodes based on state inspection.

Key concept: add_conditional_edges takes a routing function that inspects state
and returns a string indicating which node to go to next.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class ClassifierState(TypedDict):
    """State for a message classifier."""
    message: str
    intent: str


def classifier_node(state: ClassifierState) -> dict:
    """Classify the message intent."""
    msg = state["message"].lower()

    if any(word in msg for word in ["hello", "hi", "hey"]):
        intent = "greeting"
    elif any(word in msg for word in ["help", "support", "issue"]):
        intent = "support"
    elif any(word in msg for word in ["purchase", "buy", "order"]):
        intent = "purchase"
    else:
        intent = "general"

    print(f"Classified '{state['message']}' as: {intent}")
    return {"intent": intent}


def greeting_node(state: ClassifierState) -> dict:
    """Handle greeting intent."""
    print(f"GREETING HANDLER: Hello there! Nice to meet you.")
    return {"message": state["message"] + " [GREETED]"}


def support_node(state: ClassifierState) -> dict:
    """Handle support intent."""
    print(f"SUPPORT HANDLER: We're here to help! Creating support ticket...")
    return {"message": state["message"] + " [SUPPORT_TICKET_CREATED]"}


def purchase_node(state: ClassifierState) -> dict:
    """Handle purchase intent."""
    print(f"PURCHASE HANDLER: Processing your order...")
    return {"message": state["message"] + " [ORDER_PROCESSED]"}


def general_node(state: ClassifierState) -> dict:
    """Handle general intent."""
    print(f"GENERAL HANDLER: Thanks for your message.")
    return {"message": state["message"] + " [ACKNOWLEDGED]"}


def route_by_intent(state: ClassifierState) -> str:
    """
    Routing function: inspects state and returns target node name.

    This is called by add_conditional_edges to decide which node to go to next.
    """
    intent_to_node = {
        "greeting": "greeting",
        "support": "support",
        "purchase": "purchase",
        "general": "general",
    }
    return intent_to_node.get(state["intent"], "general")


def build_graph():
    """Build a routing graph."""
    graph = StateGraph(ClassifierState)

    # Add nodes
    graph.add_node("classifier", classifier_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("support", support_node)
    graph.add_node("purchase", purchase_node)
    graph.add_node("general", general_node)

    # Add edges
    graph.add_edge(START, "classifier")

    # CONDITIONAL EDGES: After classifier, route based on intent
    graph.add_conditional_edges(
        "classifier",
        route_by_intent,  # routing function
        {
            "greeting": "greeting",
            "support": "support",
            "purchase": "purchase",
            "general": "general",
        },  # target mapping
    )

    # All intent handlers go to END
    graph.add_edge("greeting", END)
    graph.add_edge("support", END)
    graph.add_edge("purchase", END)
    graph.add_edge("general", END)

    return graph.compile()


if __name__ == "__main__":
    print("=" * 70)
    print("CONDITIONAL EDGES DEMO")
    print("=" * 70)

    compiled = build_graph()

    # Test different intents
    test_messages = [
        "Hello, how are you?",
        "I need technical support",
        "I want to buy something",
        "Just chatting",
    ]

    for msg in test_messages:
        print(f"\nProcessing: '{msg}'")
        print("-" * 70)

        result = compiled.invoke({"message": msg, "intent": ""})
        print(f"Final: {result['message']}")

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("""
1. add_conditional_edges takes THREE arguments:
   - Source node name: "classifier"
   - Routing function: returns a string (target node name)
   - Target mapping: dict of possible routes

2. The routing function inspects state and returns a STRING:
   def route_fn(state) -> str:
       return "node_name"

3. This is DIFFERENT from add_edge:
   add_edge("a", "b")              → always goes to "b"
   add_conditional_edges(...)      → goes to different nodes based on logic

4. Use cases:
   - Classify then route (this example)
   - Decision trees
   - Conditional retries (Module 02-03)
   - State-based branching
    """)
