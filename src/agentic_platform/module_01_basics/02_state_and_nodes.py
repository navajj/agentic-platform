"""
Module 01 — Example 2: State and Nodes (Reducers)

THE KEY LEARNING: How to accumulate state across multiple nodes.

This demonstrates the #1 gotcha for LangChain developers:
- Without a reducer, each node REPLACES the list
- With a reducer (Annotated[list, operator.add]), each node APPENDS to the list

This is how message history, findings, and other accumulating data work.
"""

import operator
from typing import Annotated, TypedDict


# WITHOUT reducer: list gets REPLACED (BAD)
class BadState(TypedDict):
    """Each node return value will REPLACE the items list."""
    items: list[str]


# WITH reducer: list gets APPENDED (GOOD)
class GoodState(TypedDict):
    """Each node return value will APPEND to the items list."""
    items: Annotated[list[str], operator.add]


def demonstrate_bad_accumulation():
    """Show what happens WITHOUT a reducer."""
    print("=" * 60)
    print("BAD: Without Annotated[list, operator.add]")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END

    def node_1(state: BadState) -> dict:
        print(f"  Node 1 - current items: {state['items']}")
        # Return a new item
        return {"items": ["item_1"]}

    def node_2(state: BadState) -> dict:
        print(f"  Node 2 - current items: {state['items']}")
        # Return another item
        return {"items": ["item_2"]}

    def node_3(state: BadState) -> dict:
        print(f"  Node 3 - current items: {state['items']}")
        # Return another item
        return {"items": ["item_3"]}

    graph = StateGraph(BadState)
    graph.add_node("node_1", node_1)
    graph.add_node("node_2", node_2)
    graph.add_node("node_3", node_3)

    graph.add_edge(START, "node_1")
    graph.add_edge("node_1", "node_2")
    graph.add_edge("node_2", "node_3")
    graph.add_edge("node_3", END)

    compiled = graph.compile()
    result = compiled.invoke({"items": []})

    print(f"  Final items: {result['items']}")
    print(f"  ❌ PROBLEM: Only the LAST item survived! [item_3]")
    print()


def demonstrate_good_accumulation():
    """Show what happens WITH a reducer."""
    print("=" * 60)
    print("GOOD: With Annotated[list, operator.add]")
    print("=" * 60)

    from langgraph.graph import StateGraph, START, END

    def node_1(state: GoodState) -> dict:
        print(f"  Node 1 - current items: {state['items']}")
        # Return a new item
        return {"items": ["item_1"]}

    def node_2(state: GoodState) -> dict:
        print(f"  Node 2 - current items: {state['items']}")
        # Return another item
        return {"items": ["item_2"]}

    def node_3(state: GoodState) -> dict:
        print(f"  Node 3 - current items: {state['items']}")
        # Return another item
        return {"items": ["item_3"]}

    graph = StateGraph(GoodState)
    graph.add_node("node_1", node_1)
    graph.add_node("node_2", node_2)
    graph.add_node("node_3", node_3)

    graph.add_edge(START, "node_1")
    graph.add_edge("node_1", "node_2")
    graph.add_edge("node_2", "node_3")
    graph.add_edge("node_3", END)

    compiled = graph.compile()
    result = compiled.invoke({"items": []})

    print(f"  Final items: {result['items']}")
    print(f"  ✅ SUCCESS: All items accumulated! [item_1, item_2, item_3]")
    print()


def demonstrate_reducer_detail():
    """Explain how reducers work in detail."""
    print("=" * 60)
    print("REDUCER DETAIL: How Annotated[list, operator.add] works")
    print("=" * 60)

    print("""
The syntax: Annotated[list[str], operator.add]

What it means:
  - Annotated is a type hint that adds metadata to a type
  - list[str] is the actual type (list of strings)
  - operator.add is the REDUCER function

How it works:
  When a node returns {"items": ["new"]}, LangGraph applies the reducer:
    current_items = state["items"]
    new_items = node_return["items"]
    result = operator.add(current_items, new_items)
    # result = [*current_items, *new_items]

Common reducers:
  - operator.add        → list concatenation (APPEND)
  - operator.mul        → list multiplication
  - lambda a, b: b      → always use the new value (REPLACE)

For your own reducer function:
  def my_reducer(current, new):
      return custom_merge(current, new)

  items: Annotated[list[str], my_reducer]
    """)


if __name__ == "__main__":
    demonstrate_bad_accumulation()
    demonstrate_good_accumulation()
    demonstrate_reducer_detail()

    print("=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
When building LangGraph agents:

1. ALWAYS use Annotated[list[T], operator.add] for accumulating lists
   (messages, findings, errors, etc.)

2. ALWAYS return only the keys you changed from a node:
   return {"key": new_value}

3. LangGraph handles the merging based on your reducer.
   Without the reducer, you lose data!

This is different from LangChain chains where state threading is implicit.
In LangGraph, you control the accumulation explicitly via reducers.
    """)
