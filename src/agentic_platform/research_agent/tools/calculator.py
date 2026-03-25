"""
Simple calculator tool for numeric operations.
"""

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A Python expression (e.g., "2 + 2", "10 ** 2")

    Returns:
        The result of the calculation
    """
    try:
        # Only allow safe operations
        allowed_names = {
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            "len": len,
            "__builtins__": {},
        }

        result = eval(expression, allowed_names, {})
        return str(result)

    except Exception as e:
        return f"Error evaluating expression: {e}"
