"""
Research agent node functions.

Each node processes the state and returns a dict of updates.
Nodes demonstrate different graph patterns:
- planner: LLM-based planning
- researcher: Tool calling + fan-out via Send
- synthesizer: Merging findings
- quality_check: Conditional logic
- formatter: Final formatting
"""

from .planner import planner_node
from .researcher import researcher_node
from .synthesizer import synthesizer_node
from .quality_check import quality_check_node
from .formatter import formatter_node

__all__ = [
    "planner_node",
    "researcher_node",
    "synthesizer_node",
    "quality_check_node",
    "formatter_node",
]
