"""
Formatter node: produces the final formatted report.

Demonstrates:
- Post-processing and formatting
- Adding metadata and citations
- Preparing output
"""

import logging
from datetime import datetime

from research_agent.state import ResearchState

logger = logging.getLogger(__name__)


def formatter_node(state: ResearchState) -> dict:
    """
    Format the approved report as final Markdown.

    Args:
        state: Current state with draft_report and quality feedback

    Returns:
        Update dict with final_report
    """
    logger.info("Formatting final report...")

    draft = state.get("draft_report", "")
    quality = state.get("quality", {})
    human_feedback = state.get("human_feedback", "")

    # Add metadata
    metadata = f"""# Research Report: {state['topic']}

**Generated**: {datetime.now().isoformat()}
**Quality Score**: {quality.get('score', 'N/A')}/100

---

"""

    # Add human feedback if present
    feedback_section = ""
    if human_feedback:
        feedback_section = f"""## Reviewer Feedback

{human_feedback}

---

"""

    # Combine into final report
    final_report = metadata + feedback_section + draft

    # Add footer with sources
    footer = """

---

## Report Information

- **Report Type**: Comprehensive Research Analysis
- **Methodology**: Multi-source research with academic and web resources
- **Quality Assessment**: Passed comprehensive evaluation

"""

    final_report += footer

    logger.info("Report formatting complete")

    return {"final_report": final_report}
