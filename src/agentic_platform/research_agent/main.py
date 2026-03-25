"""
Research Agent CLI

Entry point for the research agent. Run research on any topic.

Usage:
    uv run research "Your topic here"
    uv run research --help
"""

import logging
import os
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from research_agent.state import ResearchState
from research_agent.graph import build_graph
from research_agent.checkpointer import get_checkpointer

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Research Agent: Break down topics and synthesize findings")
console = Console()


@app.command()
def research(
    topic: str = typer.Argument(..., help="The research topic"),
    max_questions: int = typer.Option(
        3, "--max-questions", "-q", help="Maximum sub-questions to plan"
    ),
    checkpointing: bool = typer.Option(
        True, "--checkpointing/--no-checkpointing", help="Enable result checkpointing"
    ),
    skip_human_review: bool = typer.Option(
        False,
        "--skip-review",
        help="Skip human review before quality check",
    ),
) -> None:
    """
    Run research on a topic.

    The agent will:
    1. Plan sub-questions
    2. Research each sub-question in parallel
    3. Synthesize findings
    4. Quality check the report
    5. Format and output the final report
    """
    try:
        # Validate topic
        if not topic or len(topic.strip()) == 0:
            console.print("[red]Error: Topic cannot be empty[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold cyan]Research Agent[/bold cyan]")
        console.print(f"Topic: [yellow]{topic}[/yellow]\n")

        # Initialize state
        input_state: ResearchState = {
            "topic": topic,
            "max_sub_questions": max_questions,
            "sub_questions": [],
            "messages": [],
            "draft_report": "",
            "quality": None,
            "human_feedback": None,
            "retry_count": 0,
            "error": None,
            "final_report": "",
        }

        # Build graph
        checkpointer = get_checkpointer() if checkpointing else None
        graph = build_graph(
            checkpointer=checkpointer,
            with_human_review=not skip_human_review,
        )

        # Execute graph
        console.print("[blue]Starting research process...[/blue]")

        config = None
        if checkpointing:
            import uuid

            thread_id = str(uuid.uuid4())[:8]
            config = {"configurable": {"thread_id": thread_id}}
            console.print(f"[dim]Thread ID: {thread_id}[/dim]")

        # Stream the execution
        try:
            for event in graph.stream(input_state, config=config):
                # Events are step outputs; log them for debugging
                logger.debug(f"Step: {event}")

            # Get final state
            final_state = graph.get_state(config)
            if final_state:
                result_state = final_state.values
            else:
                # If no state available, run one more time to get it
                result_state = graph.invoke(input_state, config=config)

        except Exception as e:
            # If human review paused the graph, try to complete it
            if "interrupt" in str(e).lower():
                console.print("\n[yellow]Graph interrupted for human review[/yellow]")
                # Try to get current state and offer to continue
                try:
                    final_state = graph.get_state(config)
                    if final_state:
                        result_state = final_state.values
                    else:
                        console.print(
                            "[red]Error: Could not retrieve state after interrupt[/red]"
                        )
                        raise typer.Exit(1)
                except Exception as state_error:
                    console.print(f"[red]Error retrieving state: {state_error}[/red]")
                    raise typer.Exit(1)
            else:
                raise

        # Display results
        if "final_report" in result_state and result_state["final_report"]:
            console.print("\n" + "=" * 70)
            console.print("[bold green]Research Complete[/bold green]")
            console.print("=" * 70 + "\n")

            # Display report
            report_md = Markdown(result_state["final_report"])
            console.print(report_md)

            # Display quality metrics if available
            if result_state.get("quality"):
                quality = result_state["quality"]
                console.print("\n" + "=" * 70)
                console.print("[bold]Quality Metrics[/bold]")
                console.print(f"Score: {quality.get('score', 'N/A')}/100")
                console.print(f"Passed: {'✓' if quality.get('passed') else '✗'}")
                if quality.get("feedback"):
                    console.print(f"Feedback: {quality['feedback']}")
                console.print("=" * 70)
        else:
            console.print("[yellow]No final report generated[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[red]Research interrupted by user[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Research failed")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
