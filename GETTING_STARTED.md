# Getting Started with Agentic Platform

Your LangGraph learning platform is ready! Here's your step-by-step learning path.

## 1. Initial Setup

```bash
cd /Users/jose.nava/Documents/pruebas-git/agentic-platform

# Install dependencies
uv sync

# Create .env file and add your API keys
cp .env.example .env
# Then edit .env and add:
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...
#   LANGSMITH_API_KEY=ls_... (optional for tracing)
```

## 2. Learn the Fundamentals (Module 01)

Start here to understand the core LangGraph concepts:

```bash
# Example 1: Simplest possible graph (2 nodes)
uv run python -m agentic_platform.module_01_basics.01_hello_graph

# Example 2: STATE & REDUCERS (MOST IMPORTANT!)
# This demonstrates the #1 gotcha for LangChain users
uv run python -m agentic_platform.module_01_basics.02_state_and_nodes

# Example 3: Graph structure and visualization
uv run python -m agentic_platform.module_03_tools.01_react_agent
```

**Read the code comments carefully** — they explain the mental shift from chains to graphs.

## 3. Master Conditional Routing (Module 02)

Learn how to branch and route based on state:

```bash
# Example 1: Route messages to different handlers
uv run python -m agentic_platform.module_02_routing.01_conditional_edges

# Example 2: Retry loops (impossible in LangChain!)
uv run python -m agentic_platform.module_02_routing.02_retry_loops
```

## 4. Understand Tool Integration (Module 03)

See how agents call external tools:

```bash
# Example: ReAct agent with tool calling
uv run python -m agentic_platform.module_03_tools.01_react_agent
```

## 5. Production Patterns (Module 04)

Learn about persistence, streaming, and human-in-the-loop:

```bash
# Example: Checkpointing for resumable runs
uv run python -m agentic_platform.module_04_production.01_checkpointing
```

## 6. See It All Together: Research Agent

Now run the capstone project that uses ALL the concepts:

```bash
uv run research "Your research topic here"

# Examples:
uv run research "Quantum computing applications in cryptography"
uv run research "History of artificial intelligence"
uv run research "Climate change mitigation strategies"
```

The agent will:
1. **Plan**: Break your topic into 3 sub-questions
2. **Research**: Search the web and arxiv in parallel (using Send API)
3. **Synthesize**: Merge findings into a draft report
4. **Quality Check**: Evaluate the report (with human-in-the-loop option)
5. **Format**: Produce a final markdown report

## Next Steps

### To Deepen Your Understanding

1. **Read CLAUDE.md** — Complete guide for future AI-assisted development
2. **Read README.md** — Comprehensive project documentation
3. **Examine the code comments** — Each node has detailed explanations
4. **Study the state.py** — Understand TypedDict + reducers deeply

### To Extend the Platform

1. **Add new tools** in `src/agentic_platform/research_agent/tools/`
   - Implement the `@tool` decorator
   - Add to the research agent tools list

2. **Add new nodes** in `src/agentic_platform/research_agent/nodes/`
   - Follow the signature: `(state: ResearchState) -> dict`
   - Return only changed keys
   - Register in `graph.py`

3. **Modify the graph** in `src/agentic_platform/research_agent/graph.py`
   - Add edges, conditional routes
   - Change the flow

4. **Create new modules** for additional learning topics
   - Copy the pattern from Module 01-04
   - Add your own examples

### To Deploy

1. Set up environment variables (API keys, database path)
2. Use `SqliteSaver` or upgrade to PostgreSQL checkpointer
3. Enable LangSmith tracing for debugging
4. Containerize with Docker for cloud deployment

## Key Insights

### The Mental Shift: Chains → Graphs

**LangChain thinking:**
```python
result = prompt | llm | parser
```

**LangGraph thinking:**
```python
graph.add_node("llm", llm_fn)
graph.add_node("parser", parser_fn)
graph.add_edge("llm", "parser")
compiled = graph.compile()
result = compiled.invoke(input)
```

### Why This Matters

Graphs enable:
- **Cycles**: Retry loops, human-in-the-loop feedback
- **Parallelism**: Fan-out with `Send` API
- **Persistence**: Checkpoints with `thread_id`
- **Flexibility**: Easy to visualize and modify

### The #1 Gotcha: Reducers

Without: `messages: list[str]` → each node REPLACES the list
With: `messages: Annotated[list[str], operator.add]` → each node APPENDS

This is how history accumulates! Don't forget it.

## Troubleshooting

### `ModuleNotFoundError`
```bash
# Make sure you're using uv run:
uv run python -m agentic_platform.module_01_basics.01_hello_graph
```

### API Key Errors
```bash
# Check your .env file has valid keys
cat .env

# Or set them directly:
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

### Research agent fails to run
```bash
# Check dependencies are installed:
uv sync

# Run with debug logging:
LOG_LEVEL=DEBUG uv run research "test"
```

## Resources

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangChain Docs**: https://python.langchain.com/
- **LangSmith**: https://smith.langchain.com/ (tracing & debugging)

## Questions?

1. **Check CLAUDE.md** for common patterns and gotchas
2. **Read the example code comments** — they're detailed!
3. **Examine conftest.py** for testing patterns
4. **Look at the node implementations** for real examples

---

**Happy learning!** You're about to become a LangGraph expert. 🚀
