# Module 01: LangGraph Basics

Learn the fundamental concepts of LangGraph:
- **StateGraph**: The core graph abstraction
- **TypedDict State**: Explicit state schema
- **Nodes**: Functions that transform state
- **Edges**: Transitions between nodes
- **Reducers**: How state updates merge

## Running the Examples

```bash
# Example 1: Minimal 2-node graph
uv run python -m agentic_platform.module_01_basics.01_hello_graph

# Example 2: State and reducers (MOST IMPORTANT)
uv run python -m agentic_platform.module_01_basics.02_state_and_nodes

# Example 3: Graph structure and visualization
uv run python -m agentic_platform.module_01_basics.03_edges_and_flow
```

## Key Concepts

### StateGraph
The container for your graph. Define nodes and edges, then compile it.

```python
from langgraph.graph import StateGraph

graph = StateGraph(MyState)
graph.add_node("name", node_fn)
graph.add_edge("prev", "next")
compiled = graph.compile()
```

### State Schema (TypedDict)
Your explicit state is a TypedDict. Every node receives it and returns a partial update.

```python
from typing import TypedDict

class MyState(TypedDict):
    field1: str
    field2: int

def my_node(state: MyState) -> dict:
    # Return ONLY the keys you changed
    return {"field1": "new_value"}
```

### Nodes
Functions that take state and return a dict of updates. No mutations!

```python
def my_node(state: MyState) -> dict:
    # Read from state
    value = state["field1"]
    # Compute
    result = transform(value)
    # Return partial update
    return {"field1": result}
```

### Edges
Explicit transitions. Can be unconditional (`add_edge`) or conditional (`add_conditional_edges`).

```python
graph.add_edge(START, "node_a")          # Always go to node_a
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)
```

### Reducers (Critical!)
Control how state updates merge when a node returns a list.

```python
from typing import Annotated
import operator

class State(TypedDict):
    # Default: node return REPLACES the value
    messages: list[str]

    # With reducer: node return APPENDS to the list
    history: Annotated[list[str], operator.add]
```

Without the reducer, you lose data. With it, you accumulate. **This is the #1 gotcha.**

## Progression

1. **01_hello_graph.py**: Minimal graph, introduces StateGraph + edges
2. **02_state_and_nodes.py**: Deep dive on reducers (most important!)
3. **03_edges_and_flow.py**: Graph structure, visualization, inspection

## Mental Model Shift

Coming from LangChain, you're used to:
```python
result = prompt | llm | parser
```

In LangGraph, you think:
```python
graph = StateGraph(State)
graph.add_node("prompt", prompt_fn)
graph.add_node("llm", llm_fn)
graph.add_node("parser", parser_fn)
graph.add_edge(START, "prompt")
graph.add_edge("prompt", "llm")
graph.add_edge("llm", "parser")
graph.add_edge("parser", END)
compiled = graph.compile()
```

Why is this better?
- **Explicit**: Every transition is clear
- **Flexible**: Easy to add cycles, conditional paths, parallelism
- **Debuggable**: You can visualize and introspect the graph
- **Persistent**: With checkpointing, you can resume interrupted runs

## Next Steps

- Read Example 2 very carefully — the reducer pattern is used everywhere
- Run each example and modify it (change the message, add a node, etc.)
- Proceed to Module 02 when you're comfortable with these concepts
