# Module 02: Conditional Routing

Master `add_conditional_edges` to enable branching, decision-making, and loops.

## Running the Examples

```bash
uv run python -m agentic_platform.module_02_routing.01_conditional_edges
uv run python -m agentic_platform.module_02_routing.02_retry_loops
```

## Key Concepts

### Conditional Edges

Route to different nodes based on state inspection:

```python
def route_fn(state) -> str:
    if state["field"] == "value":
        return "node_a"
    else:
        return "node_b"

graph.add_conditional_edges(
    "source_node",
    route_fn,
    {"node_a": "node_a", "node_b": "node_b"},
)
```

### Retry Loops

Edges can go backwards, enabling retries:

```python
graph.add_edge("fix", "process")  # Retry loop!

# With conditional routing:
def route_fn(state):
    if state["is_valid"]:
        return "end"
    elif state["attempts"] < 3:
        return "fix"  # Will loop back to process
    else:
        return "failure"
```

### Use Cases

1. **Classification**: Classify then route to different handlers
2. **Validation**: Validate, fix if needed, retry
3. **Decision Trees**: Multi-level branching
4. **LLM-Based Decisions**: Let the LLM choose the next step
5. **Error Handling**: Route to recovery nodes

## LangChain Comparison

In LangChain, this is awkward:
```python
# Can't do conditional branching easily in chains
# Usually resort to LLMChain + manual if/else outside the chain
```

In LangGraph, it's natural:
```python
# Conditional edges are first-class
graph.add_conditional_edges("source", route_fn, {...})
```

## Next Steps

Proceed to Module 03 to learn tool integration.
