"""
Checkpointer setup for research agent persistence.

Uses SQLite for local persistence. Configure via SQLITE_DB_PATH env var.
"""

import os
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer() -> SqliteSaver:
    """
    Get or create a SQLite checkpointer.

    Uses SQLITE_DB_PATH env var (default: ./checkpoints.db).

    Returns:
        Configured SqliteSaver instance
    """
    db_path = os.getenv("SQLITE_DB_PATH", "./checkpoints.db")
    return SqliteSaver(db_path)


def get_checkpointer_memory() -> SqliteSaver:
    """
    Get an in-memory SQLite checkpointer.

    Useful for testing (no disk writes).

    Returns:
        SqliteSaver with in-memory database
    """
    return SqliteSaver(":memory:")
