# Module 04: Production Patterns

Master checkpointing, streaming, human-in-the-loop, and error handling for production agents.

## Running the Examples

```bash
uv run python -m agentic_platform.module_04_production.01_checkpointing
```

## Key Concepts

### Checkpointing

Persist graph state for resumability:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver("./checkpoints.db")
graph = graph.compile(checkpointer=checkpointer)

# Run with thread_id
config = {"configurable": {"thread_id": "run-1"}}
result = graph.invoke(input, config=config)

# Later, resume the same thread
result = graph.invoke({}, config=config)  # Loads from checkpoint
```

### Streaming

Stream updates as the graph executes:

```python
for chunk in graph.stream(input, config=config):
    print(chunk)  # Each step's output
```

### Human-in-the-Loop

Pause before a node for human review:

```python
graph = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["quality_check"]
)

# Graph pauses before quality_check
# Human reviews, then:
graph.update_state(config, {"human_feedback": "..."})
# Resume execution
```

### Error Handling

Try/except in nodes, route to recovery:

```python
def node(state):
    try:
        result = risky_operation(state)
    except Exception as e:
        return {"error": str(e)}
    return {"result": result}

# Route based on error:
def route(state):
    if state.get("error"):
        return "error_recovery"
    return "success"
```

## Production Checklist

- [ ] Use SqliteSaver or better (Postgres) for checkpointing
- [ ] Set appropriate cleanup policies for old checkpoints
- [ ] Implement human-in-the-loop for critical decisions
- [ ] Add comprehensive error handling
- [ ] Enable streaming for user feedback
- [ ] Set up logging with structlog or similar
- [ ] Enable LangSmith tracing for debugging
- [ ] Test resumability and edge cases
- [ ] Monitor for timeout/interrupt edge cases

## Next Steps

See the Research Agent capstone for a real example using all production patterns.
