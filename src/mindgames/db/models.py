"""
SQLAlchemy models for mindgames arena database.

Based on the schema defined in proposal.md.

All timestamps are stored as UNIX timestamps (seconds since epoch) as integers.
"""

from datetime import datetime
from sqlalchemy import (
    Integer,
    String,
    Float,
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, DateTime


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Model(Base):
    """LLM model configurations."""

    __tablename__ = "models"
    __table_args__ = (
        CheckConstraint("active IN (0, 1)", name="check_models_active"),
        CheckConstraint("class IN ('open', 'small')", name="check_models_class"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Model name (e.g., 'llama3.1-8b', 'gpt-4', 'claude-3')",
    )
    litellm_model_name: Mapped[str] = mapped_column(
        String, nullable=False, comment="LiteLLM model identifier for API calls"
    )
    class_: Mapped[str] = mapped_column(
        "class",
        String,
        nullable=False,
        comment="Model class: 'open' (all models) or 'small' (<=8B parameters)",
    )
    active: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Whether this model is active for selection in new rounds",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when model was added"
    )

    # Relationships
    agents: Mapped[list["Agent"]] = relationship("Agent", back_populates="model")


class Kind(Base):
    """Agent kind/type configurations."""

    __tablename__ = "kinds"
    __table_args__ = (CheckConstraint("active IN (0, 1)", name="check_kinds_active"),)

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Kind name (e.g., 'standard', 'advanced', 'experimental')",
    )
    active: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Whether this kind is active for selection in new rounds",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when kind was added"
    )

    # Relationships
    agents: Mapped[list["Agent"]] = relationship("Agent", back_populates="kind")


class Prompt(Base):
    """Prompt templates and their configurations (immutable)."""

    __tablename__ = "prompts"
    __table_args__ = (CheckConstraint("active IN (0, 1)", name="check_prompts_active"),)

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    name: Mapped[str] = mapped_column(
        String, nullable=False, comment="Prompt version (e.g., 'v1', 'v2', 'v3')"
    )
    content: Mapped[str] = mapped_column(
        String, nullable=False, comment="Full prompt template content (immutable)"
    )
    forked_from: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("prompts.id"),
        nullable=True,
        comment="Reference to parent prompt if this was forked from another",
    )
    active: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Whether this prompt is active for selection in new rounds",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=func.now(),
        comment="Timestamp when prompt was created",
    )

    # Relationships
    parent_prompt: Mapped["Prompt | None"] = relationship(
        "Prompt", remote_side=[id], backref="forked_prompts"
    )
    agents: Mapped[list["Agent"]] = relationship("Agent", back_populates="prompt")


class Agent(Base):
    """Unique agent configurations (model + prompt + kind combinations)."""

    __tablename__ = "agents"
    __table_args__ = (
        UniqueConstraint("kind_id", "model_id", "prompt_id", name="uq_agent_config"),
        CheckConstraint("active IN (0, 1)", name="check_agents_active"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    kind_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("kinds.id"),
        nullable=False,
        comment="Reference to kinds table",
    )
    model_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("models.id"),
        nullable=False,
        comment="Reference to models table",
    )
    prompt_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prompts.id"),
        nullable=False,
        comment="Reference to prompts table",
    )
    alias: Mapped[str | None] = mapped_column(
        String, nullable=True, comment="User-defined alias (nullable)"
    )
    active: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Whether this agent is active for game selection (1=active, 0=inactive). Only active agents are selected for games.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when agent was first created"
    )

    # Relationships
    kind: Mapped["Kind"] = relationship("Kind", back_populates="agents")
    model: Mapped["Model"] = relationship("Model", back_populates="agents")
    prompt: Mapped["Prompt"] = relationship("Prompt", back_populates="agents")
    game_players: Mapped[list["GamePlayer"]] = relationship(
        "GamePlayer", back_populates="agent"
    )
    trueskill: Mapped["AgentTrueSkill | None"] = relationship(
        "AgentTrueSkill", back_populates="agent", uselist=False
    )


# Add index for common agent config queries
Index("idx_agent_config", Agent.kind_id, Agent.model_id, Agent.prompt_id)


class Game(Base):
    """Core game metadata and outcomes."""

    __tablename__ = "games"
    __table_args__ = (
        CheckConstraint(
            "status IN ('in_progress', 'completed', 'failed')",
            name="check_games_status",
        ),
        CheckConstraint(
            "winning_team IN ('Villager', 'Mafia') OR winning_team IS NULL",
            name="check_games_winning_team",
        ),
        CheckConstraint(
            "environment_crashed IN (0, 1)", name="check_games_environment_crashed"
        ),
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="check_games_finish_time",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when game started"
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="Timestamp when game finished (NULL if in progress)",
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="Game status: 'in_progress', 'completed', 'failed'",
    )
    total_turns: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Total number of turns in the game"
    )
    winning_team: Mapped[str | None] = mapped_column(
        String, nullable=True, comment="Winning team: 'Villager' or 'Mafia'"
    )
    environment_crashed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Whether the environment crashed during the game",
    )
    game_info: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="Raw textarena game_info JSON"
    )

    # Relationships
    players: Mapped[list["GamePlayer"]] = relationship(
        "GamePlayer", back_populates="game", cascade="all, delete-orphan"
    )
    turns: Mapped[list["GameTurn"]] = relationship(
        "GameTurn", back_populates="game", cascade="all, delete-orphan"
    )


# Add indexes for common game queries
Index("idx_games_finished_at", Game.finished_at)
Index("idx_games_status", Game.status)
Index("idx_games_winning_team", Game.winning_team)


class GamePlayer(Base):
    """Links agents to games with role and performance data."""

    __tablename__ = "game_players"
    __table_args__ = (
        CheckConstraint(
            "player_index >= 0 AND player_index < 6",
            name="check_game_players_player_index",
        ),
        CheckConstraint(
            "role IN ('Mafia', 'Villager', 'Doctor', 'Detective') OR role IS NULL",
            name="check_game_players_role",
        ),
        UniqueConstraint("game_id", "player_index", name="uq_game_player"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    game_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
        comment="Reference to games table",
    )
    agent_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("agents.id", ondelete="RESTRICT"),
        nullable=False,
        comment="Reference to agents table",
    )
    player_index: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Player position in game (0-5)"
    )
    role: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        comment="Game role: 'Mafia', 'Villager', 'Doctor', 'Detective'",
    )
    reward: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Final reward/score for this player"
    )

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="players")
    agent: Mapped["Agent"] = relationship("Agent", back_populates="game_players")


# Add indexes for common game_players queries
Index("idx_agent_games", GamePlayer.agent_id)


class GameTurn(Base):
    """Turn-by-turn logging with observations, actions, and timing."""

    __tablename__ = "game_turns"
    __table_args__ = (
        CheckConstraint(
            "player_index >= 0 AND player_index < 6",
            name="check_game_turns_player_index",
        ),
        CheckConstraint("response_time_ms >= 0", name="check_game_turns_response_time"),
        CheckConstraint(
            "tokens_used > 0 OR tokens_used IS NULL",
            name="check_game_turns_tokens_used",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    game_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
        comment="Reference to games table",
    )
    turn_number: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Turn number (1-indexed)"
    )
    player_index: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Which player acted (0-5)"
    )
    observation: Mapped[dict] = mapped_column(
        JSON, nullable=False, comment="Full textarena observation JSON"
    )
    action: Mapped[str] = mapped_column(
        String, nullable=False, comment="Action taken by the player"
    )
    step_info: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="textarena step_info JSON"
    )
    response_time_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Time to generate action in milliseconds"
    )
    tokens_used: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Tokens in the response of the action (may differ from internal LLM call tokens)",
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when turn was executed"
    )

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="turns")
    llm_calls: Mapped[list["LLMCall"]] = relationship(
        "LLMCall", back_populates="game_turn", cascade="all, delete-orphan"
    )


# Add indexes for common game_turns queries
Index("idx_game_turns", GameTurn.game_id, GameTurn.turn_number)
Index("idx_turn_player", GameTurn.game_id, GameTurn.player_index)


class LLMCall(Base):
    """Track individual LLM API calls for each turn."""

    __tablename__ = "llm_calls"
    __table_args__ = (
        CheckConstraint(
            "total_response_time_ms >= 0", name="check_llm_calls_total_response_time"
        ),
        CheckConstraint(
            "time_to_first_token_ms >= 0 OR time_to_first_token_ms IS NULL",
            name="check_llm_calls_time_to_first_token",
        ),
        CheckConstraint(
            "tokens_per_second >= 0 OR tokens_per_second IS NULL",
            name="check_llm_calls_tokens_per_second",
        ),
        CheckConstraint("response_tokens > 0", name="check_llm_calls_response_tokens"),
        CheckConstraint(
            "total_tokens_consumed > 0", name="check_llm_calls_total_tokens"
        ),
        CheckConstraint(
            "response_received_at >= request_sent_at",
            name="check_llm_calls_response_time",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key",
    )
    game_turn_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("game_turns.id", ondelete="CASCADE"),
        nullable=False,
        comment="Reference to game_turns table",
    )
    model: Mapped[str] = mapped_column(
        String, nullable=False, comment="LLM model used for this call"
    )
    messages: Mapped[dict] = mapped_column(
        JSON, nullable=False, comment="Full messages array sent to LLM"
    )
    response: Mapped[dict] = mapped_column(
        JSON, nullable=False, comment="Full response object from LLM"
    )
    request_sent_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when request was sent"
    )
    response_received_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when response was received"
    )
    total_response_time_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Total time for request in milliseconds"
    )
    time_to_first_token_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time to first token in milliseconds (if available)",
    )
    tokens_per_second: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Tokens generated per second (if available)"
    )
    response_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Number of tokens in the response"
    )
    total_tokens_consumed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Total tokens consumed (may differ from response_tokens for models that hide reasoning tokens)",
    )

    # Relationships
    game_turn: Mapped["GameTurn"] = relationship("GameTurn", back_populates="llm_calls")


# Add indexes for common llm_calls queries
Index("idx_llm_call_turn", LLMCall.game_turn_id)
Index("idx_llm_call_model", LLMCall.model)


class AgentTrueSkill(Base):
    """Track TrueSkill ratings for each agent."""

    __tablename__ = "agent_trueskill"

    agent_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("agents.id", ondelete="CASCADE"),
        primary_key=True,
        comment="Reference to agents table",
    )
    mu: Mapped[float] = mapped_column(
        Float, nullable=False, default=25.0, comment="TrueSkill skill mean"
    )
    sigma: Mapped[float] = mapped_column(
        Float, nullable=False, default=8.333, comment="TrueSkill skill uncertainty"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="Timestamp when rating was last updated"
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="trueskill")
