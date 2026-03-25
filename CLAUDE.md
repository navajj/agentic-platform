# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A progressive LangGraph learning platform built around a Research Agent capstone. Each module teaches one LangGraph concept independently, then the `research_agent` package combines everything into a production-quality multi-agent system.

**Key idea**: If you know LangChain, LangGraph requires a mental shift from "chains" to "graphs." This platform teaches that shift through concrete, working code.

---

## Quick Start

### Setup
```bash
uv sync                          # install dependencies
cp .env.example .env             # add your API keys
```

### Run the research agent
```bash
uv run research "Quantum computing applications in cryptography"
```

### Run a specific learning module
```bash
uv run python -m agentic_platform.module_01_basics.01_hello_graph
```

### Tests & Quality
```bash
uv run pytest                    # all tests
uv run pytest -v --cov=src      # with coverage
uv run ruff check src/
uv run mypy src/
```

---

## Architecture Overview

### Mental Model: LangChain → LangGraph

| Aspect | LangChain | LangGraph |
|---|---|---|
| Thinking | "What chain of steps?" | "What is my state, how does it flow?" |
| Composition | Left-to-right piping (`\|`) | Explicit graph (nodes + edges) |
| State | Implicit threading through pipes | Explicit TypedDict, passed to every node |
| Cycles | Awkward (need LLM to decide when) | Natural (add_conditional_edges) |
| Parallelism | Sequential or complex workarounds | First-class with `Send` API |
| Persistence | Not built-in | First-class: checkpointing + thread_id |

### Key Concept: Reducers

This is the #1 gotcha for LangChain users:

```python
# Default: node return value REPLACES the state key
class State(TypedDict):
    messages: list[str]

def node(state: State) -> dict:
    return {"messages": ["new"]}  # replaces entire list

# To APPEND instead of replace, use Annotated + operator:
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]

def node(state: State) -> dict:
    return {"messages": ["new"]}  # APPENDS to the list
```

Without the reducer, message history gets lost. This is how the graph naturally accumulates state across multiple nodes.

### Project Structure

```
src/agentic_platform/
├── module_01_basics/      Learning: StateGraph, TypedDict, nodes, edges
│   ├── 01_hello_graph.py     (simplest possible graph: 2 nodes)
│   ├── 02_state_and_nodes.py (partial updates + reducers)
│   └── 03_edges_and_flow.py  (visualization, graph structure)
│
├── module_02_routing/     Learning: conditional_edges, cycles
│   ├── 01_conditional_edges.py
│   ├── 02_routing_agent.py
│   └── 03_cycles_and_loops.py  (retry loops impossible in LangChain)
│
├── module_03_tools/       Learning: ToolNode, tool binding, parallelism
│   ├── 01_tool_node.py
│   ├── 02_react_agent.py
│   └── 03_parallel_tool_calls.py  (Send API for fan-out)
│
├── module_04_production/  Learning: checkpointing, streaming, HITL, errors
│   ├── 01_checkpointing.py
│   ├── 02_streaming.py
│   ├── 03_human_in_the_loop.py
│   └── 04_error_handling.py
│
└── research_agent/        Capstone: uses all concepts
    ├── state.py           (ResearchState TypedDict + reducers)
    ├── nodes/
    │   ├── planner.py        (breaks topic into sub-questions)
    │   ├── researcher.py     (searches per sub-question, parallel)
    │   ├── synthesizer.py    (merges findings)
    │   ├── quality_check.py  (grades the report)
    │   └── formatter.py      (final markdown)
    ├── tools/
    │   ├── search.py         (Tavily / DuckDuckGo)
    │   ├── arxiv_tool.py     (academic papers)
    │   └── calculator.py     (numeric reasoning)
    ├── graph.py           (StateGraph assembly, Send fan-out)
    ├── checkpointer.py    (SQLite setup)
    └── main.py            (CLI: Typer + Rich)
```

---

## Common Development Tasks

### Adding a New Node to the Research Agent

1. **Create node function** in `research_agent/nodes/{name}.py`:
```python
from research_agent.state import ResearchState

def my_node(state: ResearchState) -> dict:
    # Read from state, compute, return partial update
    result = do_something(state["topic"])
    return {"key_to_update": result}
```

2. **Register in graph** (`research_agent/graph.py`):
```python
graph.add_node("my_node", my_node)
```

3. **Wire edges**:
```python
graph.add_edge("previous_node", "my_node")
graph.add_edge("my_node", "next_node")
```
Or conditional:
```python
graph.add_conditional_edges(
    "my_node",
    route_fn,  # returns a string: "target_node_name"
    {"path_a": "node_a", "path_b": "node_b"},
)
```

4. **Test** in `tests/test_research_agent/test_nodes.py`:
```python
from research_agent.nodes.my_node import my_node

def test_my_node():
    state = {"topic": "test", ...}
    result = my_node(state)
    assert result["key_to_update"] == expected
```

---

## Core LangGraph Concepts

| Concept | File | Why it matters |
|---|---|---|
| `StateGraph` | `research_agent/graph.py` | The graph itself — nodes + edges |
| `TypedDict` state schema | `research_agent/state.py` | Single source of truth for what flows through the graph |
| Reducers (`Annotated[list, operator.add]`) | `research_agent/state.py`, every node | How to accumulate state (messages, findings, etc.) without overwriting |
| `add_node`, `add_edge` | `research_agent/graph.py` | Explicit graph construction |
| `add_conditional_edges` | `module_02_routing/`, `graph.py` | Decision points: route based on state |
| `Send` API | `module_03_tools/03_parallel_tool_calls.py`, `graph.py` | Parallel fan-out (one researcher per sub-question) |
| `checkpointer` (`SqliteSaver`) | `research_agent/checkpointer.py`, `graph.py` | Persistence + resumability via `thread_id` |
| `interrupt_before` | `research_agent/graph.py` | Human-in-the-loop: pause before a node |
| `graph.stream()` | `research_agent/main.py` | Streaming output (not `chain.stream()`) |

---

## LangGraph vs LangChain Cheat Sheet

| Task | LangChain | LangGraph |
|---|---|---|
| Build a chain | `prompt \| llm \| parser` | `graph.add_node()`, `graph.add_edge()`, `graph.compile()` |
| Pass state | Implicit (piped through) | Explicit `TypedDict`, partial updates from nodes |
| Accumulate (e.g., messages) | Use `RunnablePassthrough` + manual merging | `Annotated[list, operator.add]` reducer |
| Branch on state | `RunnableBranch` | `add_conditional_edges` + routing function |
| Retry or loop | Not natural; use `langchain_core.Runnable.retry_if` | `add_conditional_edges` back to earlier node |
| Run in parallel | Complex (`map`/`batch`) | `Send` API in conditional edges |
| Pause for human | Not built-in | `interrupt_before`, `update_state()` |
| Persist & resume | Not built-in | `checkpointer` + `thread_id` config |

---

## Environment Variables

```bash
OPENAI_API_KEY=...              # primary LLM provider
ANTHROPIC_API_KEY=...           # fallback/alternate
TAVILY_API_KEY=...              # web search tool
LANGSMITH_API_KEY=...           # tracing + debugging
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-platform
SQLITE_DB_PATH=./checkpoints.db
LOG_LEVEL=INFO
```

Set in `.env` (never commit).

---

## Logging & Observability

Uses `structlog` for structured logging:
- **Console (dev)**: Pretty, colored output
- **JSON (prod)**: Logs include full context
- **LangSmith**: Set `LANGCHAIN_TRACING_V2=true` to trace every node invocation

Enable debug logs:
```bash
LOG_LEVEL=DEBUG uv run research "topic"
```

This shows state transitions between nodes.

---

## Testing Strategy

- **Unit tests**: Test nodes in isolation with a mock state dict (no API calls)
- **Integration tests**: Full graph with a mocked LLM (`langchain_core.FakeListLLM`)
- **End-to-end**: Full graph with real APIs (requires valid `.env`)

All tests use `pytest` + `pytest-asyncio`. Run:
```bash
uv run pytest tests/ -v --cov=src
```

See `tests/conftest.py` for fixtures (test graph, mock LLM, etc.).

---

## Common Gotchas

### Gotcha 1: Returning Full State
**Wrong:**
```python
def node(state: State) -> dict:
    return state  # Returns entire state
```

**Correct:**
```python
def node(state: State) -> dict:
    return {"key_to_update": new_value}  # Only changed keys
```

### Gotcha 2: Forgetting Reducers
**Wrong:**
```python
class State(TypedDict):
    messages: list[str]  # each node call REPLACES this
```

**Correct:**
```python
class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # APPENDS
```

### Gotcha 3: Thread ID for Checkpointing
**Wrong:**
```python
graph.invoke(input)  # No persistence
```

**Correct:**
```python
graph.invoke(input, config={"configurable": {"thread_id": "run-001"}})
```

### Gotcha 4: `interrupt_before` Timing
`interrupt_before=["node_name"]` pauses **before** the node runs, not after. You can inject feedback via `graph.update_state()` before resuming.

### Gotcha 5: `Send` Requires Routing
`Send` is only used inside `add_conditional_edges` callbacks. Don't try to use it in a normal edge.

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain/LangGraph Concepts](https://python.langchain.com/docs/concepts/architecture)
- Module README files (each has learning notes specific to that concept)

---

## Future Work / Extensions

- Add Postgres checkpointer (`langgraph-checkpoint-postgres`)
- Implement streaming token output in CLI
- Add example with ReAct agent from `langgraph.prebuilt`
- Expand tool suite (image generation, code execution, etc.)
- Production deployment guide (Docker, cloud platforms)
