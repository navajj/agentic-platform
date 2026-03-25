# Module 03: Tool Integration

Master tool binding, tool calling, and parallel execution with the Send API.

## Running the Examples

```bash
uv run python -m agentic_platform.module_03_tools.01_react_agent
```

## Key Concepts

### Tool Binding

Attach tools to an LLM so it can decide to call them:

```python
from langchain_core.tools import tool

@tool
def my_tool(arg: str) -> str:
    """Tool description."""
    return f"Result: {arg}"

model_with_tools = model.bind_tools([my_tool])
```

### ToolNode

Automatically calls tools based on LLM output:

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([my_tool])
# Takes AIMessage with tool_calls, returns ToolMessage with results
```

### ReAct Pattern

Reason + Act loop:
1. LLM decides if a tool is needed
2. Tool is called if requested
3. Result fed back to LLM
4. Repeat until done

### Send API (Parallelism)

Fan out parallel executions:

```python
from langgraph.graph import Send

def route_fn(state):
    return [
        Send("researcher", {"item": item})
        for item in state["items"]
    ]
```

## Use Cases

- Web search agents
- Code execution agents
- API-based tools
- Parallel research (like the capstone)

## Next Steps

See the Research Agent for a full example using tools with parallel execution.
