# Agentic Platform — LangGraph Learning & Practice

A structured learning platform for building multi-agent systems with **LangGraph**, designed for developers who know LangChain but want to master LangGraph's graph-first paradigm.

## The Challenge

Coming from LangChain, you're used to thinking in **chains**: left-to-right composition, implicit state, sequential flow. LangGraph requires a mindset shift to **graphs**: explicit nodes, explicit state, natural cycles and parallelism.

This platform teaches that shift through:
1. **Progressive modules** (01–04) that each teach one concept
2. **A capstone research agent** that uses everything together
3. **Production patterns** (checkpointing, streaming, human-in-the-loop)

## Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager ([install](https://docs.astral.sh/uv/))
- API keys (OpenAI or Anthropic, optional for basic examples)

### Installation
```bash
git clone <repo>
cd agentic-platform
uv sync
cp .env.example .env
# Add your API keys to .env
```

### Run the Research Agent
```bash
uv run research "Quantum computing applications in cryptography"
```

This kicks off a multi-agent system that:
1. Plans 3 sub-questions about your topic
2. Dispatches parallel researcher agents
3. Synthesizes findings into a report
4. Quality-checks the output
5. Formats and returns the final report

### Run a Learning Module
```bash
# Simplest graph: 2 nodes, no APIs
uv run python -m agentic_platform.module_01_basics.01_hello_graph

# Conditional routing example
uv run python -m agentic_platform.module_02_routing.02_routing_agent

# Tool integration with ReAct
uv run python -m agentic_platform.module_03_tools.02_react_agent

# Checkpointing & resumability
uv run python -m agentic_platform.module_04_production.01_checkpointing
```

---

## Project Structure

```
agentic-platform/
├── README.md                (this file)
├── CLAUDE.md                (Claude Code guidance)
├── pyproject.toml
├── .env.example
│
├── src/agentic_platform/
│   ├── module_01_basics/       LangGraph fundamentals
│   │   ├── 01_hello_graph.py     First graph: 2 nodes
│   │   ├── 02_state_and_nodes.py TypedDict + reducers
│   │   └── 03_edges_and_flow.py  Visualization + structure
│   │
│   ├── module_02_routing/      Conditional routing & cycles
│   │   ├── 01_conditional_edges.py
│   │   ├── 02_routing_agent.py
│   │   └── 03_cycles_and_loops.py
│   │
│   ├── module_03_tools/        Tool integration & parallelism
│   │   ├── 01_tool_node.py
│   │   ├── 02_react_agent.py
│   │   └── 03_parallel_tool_calls.py
│   │
│   ├── module_04_production/   Production patterns
│   │   ├── 01_checkpointing.py
│   │   ├── 02_streaming.py
│   │   ├── 03_human_in_the_loop.py
│   │   └── 04_error_handling.py
│   │
│   └── research_agent/         Capstone multi-agent system
│       ├── state.py            (TypedDict + reducers)
│       ├── nodes/              (planner, researcher, synthesizer, etc.)
│       ├── tools/              (search, arxiv, calculator)
│       ├── graph.py            (StateGraph assembly)
│       ├── checkpointer.py     (SQLite persistence)
│       └── main.py             (CLI entry point)
│
├── tests/
│   ├── test_module_*.py
│   └── test_research_agent/
│
└── docs/
    ├── 00-langgraph-vs-langchain.md
    ├── 01-graph-mental-model.md
    ├── 02-state-and-reducers.md
    └── 03-production-patterns.md
```

---

## Learning Path

### Module 01: LangGraph Basics
**Goal**: Understand `StateGraph`, `TypedDict` state, nodes, edges

- `01_hello_graph.py` — Minimal 2-node graph, shows `add_node`, `add_edge`, `compile`, `invoke`
- `02_state_and_nodes.py` — State schemas with `TypedDict`, partial node updates, **reducers** (the key concept!)
- `03_edges_and_flow.py` — Multi-node graph, Mermaid visualization

**Key insight**: In LangGraph, every node must return a *dict of only the keys it changed*, not the full state.

### Module 02: Conditional Routing
**Goal**: Use `add_conditional_edges` to branch, route, and loop

- `01_conditional_edges.py` — Route to different nodes based on state
- `02_routing_agent.py` — Multi-path agent (e.g., customer support classifier)
- `03_cycles_and_loops.py` — **Retry loops** (impossible in LangChain; natural in LangGraph)

**Key insight**: `add_conditional_edges` enables decision-making within the graph itself. The routing function inspects state and returns a string matching one of the target node names.

### Module 03: Tool Integration
**Goal**: Bind tools to the LLM, handle tool calls, understand `ToolNode` and parallel execution

- `01_tool_node.py` — Manual tool integration: bind → call → handle results
- `02_react_agent.py` — Use `langgraph.prebuilt.create_react_agent` shortcut
- `03_parallel_tool_calls.py` — Fan-out with `Send` API: one action per sub-item in parallel

**Key insight**: `Send` is how you achieve true parallelism in LangGraph. Return a list of `Send` objects from a conditional edge callback.

### Module 04: Production Patterns
**Goal**: Checkpointing, streaming, human-in-the-loop, error handling

- `01_checkpointing.py` — SQLite persistence via `thread_id`; resume interrupted runs
- `02_streaming.py` — `graph.stream()` with `mode="updates"` vs `mode="values"`
- `03_human_in_the_loop.py` — `interrupt_before` to pause, inspect, inject feedback
- `04_error_handling.py` — Try/except in nodes, error recovery paths

**Key insight**: Checkpointing + `thread_id` makes graphs resumable. `interrupt_before` + `update_state()` enables human oversight.

### Capstone: Research Agent
**Goal**: Build a real multi-agent system using every concept

A system that takes a research topic and:
1. **Planner node** breaks it into 3 sub-questions
2. **Researcher nodes** (fanout via `Send`) search for info on each sub-question in parallel
3. **Synthesizer node** (fanin) merges findings
4. **Quality check node** grades the report (conditional: pass → format, fail → replan)
5. **Formatter node** produces final Markdown

**Graph topology**:
```
START → planner → [Send] → researcher(1..N) → synthesizer
      → quality_check → [conditional] → formatter → END
                                     ↘ planner (retry)
```

**Key concepts demonstrated**:
- State schema with reducers (`Annotated[list, operator.add]`)
- Fan-out via `Send` API
- Conditional edges with retry logic
- Checkpointing + resumability
- Tool calls (search APIs)
- Human-in-the-loop (interrupt before quality_check)

---

## The Big Shift: LangChain → LangGraph

### LangChain Thinking
```python
from langchain import PromptTemplate, OpenAI
from langchain.chains import LLMChain

prompt = PromptTemplate(...)
llm = OpenAI()
chain = prompt | llm | parser
result = chain.invoke(input)
```
- Left-to-right composition
- Implicit state threading
- Sequential only

### LangGraph Thinking
```python
from langgraph.graph import StateGraph

graph = StateGraph(MyState)
graph.add_node("step1", step1_fn)
graph.add_node("step2", step2_fn)
graph.add_edge("step1", "step2")
compiled = graph.compile(checkpointer=...)

result = compiled.invoke(input, config={"configurable": {"thread_id": "run-1"}})
```
- Explicit graph (nodes + edges)
- Explicit `TypedDict` state
- Natural cycles, parallelism, persistence

**The reducer trick** — the #1 gotcha:
```python
# In LangGraph, you MUST use reducers to accumulate state
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # APPENDS, not replaces

def my_node(state: State) -> dict:
    return {"messages": ["new"]}  # This APPENDS, doesn't overwrite
```

---

## Commands

### Development
```bash
uv sync                          # Install dependencies
uv sync --extra dev              # Include dev tools
cp .env.example .env             # Set up environment

uv run pytest                    # Run tests
uv run pytest -v --cov=src      # With coverage
uv run ruff check src/           # Lint
uv run mypy src/                 # Type check

uv run python -m agentic_platform.module_01_basics.01_hello_graph
```

### Run Research Agent
```bash
uv run research "Your research topic here"
uv run research --help
```

---

## Environment Variables

Create `.env` from `.env.example`:

```bash
# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Tools
TAVILY_API_KEY=tvly-...

# Observability (optional)
LANGSMITH_API_KEY=ls_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-platform

# Logging
LOG_LEVEL=INFO

# Database
SQLITE_DB_PATH=./checkpoints.db
```

---

## Key Concepts at a Glance

| Concept | What it does | Why it matters |
|---------|-------------|-----------------|
| **StateGraph** | Defines nodes and edges | The graph itself |
| **TypedDict State** | Single source of truth for data flow | No hidden state; everything explicit |
| **Reducers** | Control how updates merge (`Annotated[list, operator.add]`) | Without them, message history gets lost |
| **add_conditional_edges** | Route based on state | Enables branching, retries, loops |
| **Send API** | Parallel fan-out | One sub-agent per item |
| **Checkpointer** | SQLite persistence | Resume interrupted runs via `thread_id` |
| **interrupt_before** | Pause before a node | Human-in-the-loop validation |
| **stream()** | Stream updates as they happen | Real-time progress visibility |

---

## Common Patterns

### Pattern 1: Accumulate Messages
```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def node(state: State) -> dict:
    new_msg = HumanMessage(...)
    return {"messages": [new_msg]}  # Appends
```

### Pattern 2: Route Based on State
```python
def route_fn(state: State) -> str:
    if state["is_urgent"]:
        return "escalate"
    return "standard_path"

graph.add_conditional_edges(
    "classifier",
    route_fn,
    {"escalate": "escalation_node", "standard_path": "standard_node"},
)
```

### Pattern 3: Parallel Sub-Agents (Fan-Out)
```python
from langgraph.graph import Send

def fan_out_fn(state: State):
    return [
        Send("researcher", {"sub_question": q, ...})
        for q in state["sub_questions"]
    ]

graph.add_conditional_edges(
    "planner",
    fan_out_fn,
    ["researcher"],
)
```

### Pattern 4: Checkpointing & Resumability
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver(":memory:")
graph = graph.compile(checkpointer=checkpointer)

# First run
result = graph.invoke(input, config={"configurable": {"thread_id": "run-1"}})

# Later: resume the same thread
graph.invoke(input, config={"configurable": {"thread_id": "run-1"}})
```

---

## Testing

All tests run with `pytest`:

```bash
uv run pytest tests/ -v
uv run pytest tests/test_research_agent/ -v
uv run pytest --cov=src --cov-report=html
```

Each module has unit tests for its nodes (mocked, no API calls).

---

## Next Steps

1. **Start with Module 01** to internalize StateGraph and reducers
2. **Play with the examples** — modify them, add nodes, change routing logic
3. **Build the Research Agent** from scratch using the plan in `CLAUDE.md`
4. **Extend it** — add more tools, change the research strategy, add persistence to a real database
5. **Deploy** — containerize and run on your favorite cloud platform

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'agentic_platform'`
Make sure you've run `uv sync` and are running with `uv run`.

### Partial state not updating
Check your node signature — it must return `dict`, not `State`. Only return the keys you changed.

### Reducers not appending
Forgot `Annotated[list, operator.add]`? See the reducer section of CLAUDE.md.

### API key errors
Make sure `.env` has valid keys. Set `LOG_LEVEL=DEBUG` to see detailed errors.

---

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Python Docs](https://python.langchain.com/)
- [LangSmith Tracing](https://docs.smith.langchain.com/)
- Each module's README for concept-specific notes

---

## Contributing

This is a learning platform. Feel free to:
- Modify examples to experiment
- Add new nodes to the research agent
- Create new tools
- Write tests
- Improve docs

---

## License

MIT
