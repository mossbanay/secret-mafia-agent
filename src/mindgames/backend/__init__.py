"""
FastAPI backend for mindgames arena system.

Provides REST API endpoints for game management, agent tracking, and data queries.
"""

from .app import app, get_db_session

__all__ = ["app", "get_db_session"]
