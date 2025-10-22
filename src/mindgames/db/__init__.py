"""
Database package for mindgames arena system.

Implements SQLAlchemy-based database layer with models, session management,
and data access layer based on proposal.md schema.
"""

from .models import (
    Agent,
    Game,
    GamePlayer,
    GameTurn,
    LLMCall,
    Model,
    Kind,
    Prompt,
    AgentTrueSkill,
)
from .session import get_session, init_db, get_engine
from .repository import Repository

__all__ = [
    "Agent",
    "Game",
    "GamePlayer",
    "GameTurn",
    "LLMCall",
    "Model",
    "Kind",
    "Prompt",
    "AgentTrueSkill",
    "get_session",
    "init_db",
    "get_engine",
    "Repository",
]
