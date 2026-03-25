"""
Module 03 — Example 1: ReAct Agent using LangGraph

Demonstrates using langgraph.prebuilt.create_react_agent,
which is a built-in agent that handles tool calling automatically.

ReAct = Reasoning + Acting:
1. LLM reasons about the problem
2. Decides to call a tool
3. Tool result is fed back to LLM
4. Loop until LLM says "done"
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# Define some simple tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def get_weather(location: str) -> str:
    """Get weather for a location (simulated)."""
    # Simulated weather
    weather_data = {
        "san francisco": "Cloudy, 65°F",
        "new york": "Rainy, 58°F",
        "london": "Overcast, 52°F",
    }
    return weather_data.get(location.lower(), "Unknown location")


def main():
    """Run the ReAct agent."""
    print("=" * 70)
    print("REACT AGENT DEMO")
    print("=" * 70)
    print()

    # Setup tools
    tools = [add, multiply, get_weather]

    # Setup LLM
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the agent (one line!)
    # This handles all the graph construction, tool calling, etc.
    agent = create_react_agent(model, tools)

    # Test queries
    queries = [
        "What is 25 * 4?",
        "Add 10 and 20, then multiply the result by 3",
        "What's the weather in San Francisco?",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 70)

        # Run the agent
        result = agent.invoke({"messages": [("human", query)]})

        # Extract the final response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}")
        print()

    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. create_react_agent IS A SHORTCUT
   It handles:
   - Tool binding to the LLM
   - Checking for tool_calls in LLM response
   - Calling the tools
   - Feeding results back to LLM
   - Stopping when done

   Without it, you'd manually wire all of that (see example 01_tool_node.py)

2. UNDER THE HOOD
   The graph looks like:
   START → llm_with_tools
      ↓
   [LLM response has tool_calls?]
      ├→ YES: tool_node → [back to llm_with_tools]
      └→ NO: return

3. THIS IS A PATTERN YOU CAN EXTEND
   You can do:
   graph = create_react_agent(model, tools)
   graph.add_node("custom", custom_fn)
   graph.add_edge(...)
   # Now you have a custom agent with the react core

4. WHEN TO USE create_react_agent
   - Quick tool-using agents
   - Standard ReAct pattern
   - No custom routing needed

   When to build from scratch:
   - Complex custom logic
   - Multiple agent types
   - Research Agent pattern (like our capstone)
    """)


if __name__ == "__main__":
    main()
