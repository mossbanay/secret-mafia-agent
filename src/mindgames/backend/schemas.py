"""
Pydantic schemas for API request/response models.

Based on data_models.md requirements.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ===== Header Stats =====


class HeaderStatsResponse(BaseModel):
    """Header statistics shown across all pages."""

    total_spent: float = Field(..., description="Total money spent in USD")
    total_games: int = Field(..., description="Total number of games played")


# ===== Agent Models =====


class AgentResponse(BaseModel):
    """Agent data for the agents list page."""

    id: str = Field(..., description="Unique agent identifier")
    alias: Optional[str] = Field(None, description="Optional user-defined alias")
    kind: str = Field(..., description="Agent type")
    model: str = Field(..., description="LLM model name")
    prompt: str = Field(..., description="Prompt version")
    class_: str = Field(
        ..., alias="class", description="Access class: 'open' or 'restricted'"
    )
    active: bool = Field(..., description="Whether this agent is active")
    games: int = Field(..., description="Number of games played")
    trueskill: float = Field(..., description="TrueSkill rating score")
    mu: float = Field(..., description="TrueSkill mu (skill mean)")
    sigma: float = Field(..., description="TrueSkill sigma (skill uncertainty)")

    class Config:
        populate_by_name = True


class AgentListResponse(BaseModel):
    """Response for listing agents."""

    agents: List[AgentResponse]
    total: int


class UpdateAliasRequest(BaseModel):
    """Request to update an agent's alias."""

    alias: str = Field(..., description="New alias for the agent")


class UpdateActiveRequest(BaseModel):
    """Request to update an agent's active status."""

    active: bool = Field(..., description="Whether this agent is active")


# ===== Rollout/Game Models =====


class RolloutResponse(BaseModel):
    """Rollout (game summary) data."""

    game_id: str = Field(..., description="Unique game identifier")
    agent_1: str = Field(..., description="Agent alias or ID for player 1")
    agent_2: str = Field(..., description="Agent alias or ID for player 2")
    agent_3: str = Field(..., description="Agent alias or ID for player 3")
    agent_4: str = Field(..., description="Agent alias or ID for player 4")
    agent_5: str = Field(..., description="Agent alias or ID for player 5")
    agent_6: str = Field(..., description="Agent alias or ID for player 6")
    finished_at: int = Field(..., description="UNIX timestamp (seconds since epoch)")
    winning_team: str = Field(..., description="'Villager' or 'Mafia'")


class RolloutListResponse(BaseModel):
    """Response for listing rollouts/games."""

    rollouts: List[RolloutResponse]
    total: int


# ===== Game Details Models =====


class GamePlayerResponse(BaseModel):
    """Player information in a game."""

    agent_id: str = Field(..., description="Agent identifier")
    alias: Optional[str] = Field(None, description="Optional agent alias")
    kind: str = Field(..., description="Agent type")
    model: str = Field(..., description="LLM model used")
    prompt: str = Field(..., description="Prompt version used")
    role: str = Field(
        ..., description="Game role: 'Mafia', 'Villager', 'Doctor', 'Detective'"
    )
    reward: float = Field(..., description="Final reward for this player")
    avg_response_time: int = Field(
        ..., description="Average response time in milliseconds"
    )


class GameMessageResponse(BaseModel):
    """Message/turn in a game."""

    turn_id: int = Field(..., description="Database ID of the turn")
    turn: int = Field(..., description="Turn number (1-indexed)")
    player_index: int = Field(..., description="Player index (0-5)")
    agent: str = Field(..., description="Agent name or alias")
    kind: str = Field(..., description="Agent kind")
    model: str = Field(..., description="LLM model used")
    prompt: str = Field(..., description="Prompt version used")
    role: str = Field(..., description="Player's role")
    content: str = Field(..., description="Message content/text")
    response_time: int = Field(..., description="Response time in milliseconds")
    observation_tokens: int = Field(
        ..., description="Number of tokens in the observation/input"
    )
    action_tokens: int = Field(..., description="Number of tokens in the action/output")
    tokens: int = Field(
        ..., description="Total number of tokens used (observation + action)"
    )
    llm_call_count: int = Field(
        ..., description="Number of LLM calls made for this turn"
    )


class GameDetailsResponse(BaseModel):
    """Full game details."""

    game_id: str = Field(..., description="Unique game identifier")
    turns: int = Field(..., description="Total number of turns")
    winning_team: str = Field(..., description="'Villager' or 'Mafia'")
    players: List[GamePlayerResponse] = Field(..., description="Array of 6 players")
    messages: List[GameMessageResponse] = Field(
        ..., description="All game messages chronologically"
    )


# ===== Filter Models =====


class FilterValuesResponse(BaseModel):
    """Response for filter enumeration endpoints."""

    values: List[str] = Field(..., description="Array of available filter values")


# ===== Model Management =====


class ModelResponse(BaseModel):
    """Model configuration data."""

    id: int = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Model name (e.g., 'llama3.1-8b', 'gpt-4')")
    litellm_model_name: str = Field(..., description="LiteLLM model identifier")
    class_: str = Field(
        ..., alias="class", description="Model class: 'open' or 'small'"
    )
    active: bool = Field(..., description="Whether this model is active")
    created_at: str = Field(..., description="ISO timestamp when model was added")

    class Config:
        populate_by_name = True


class ModelListResponse(BaseModel):
    """Response for listing models."""

    models: List[ModelResponse]
    total: int


class CreateModelRequest(BaseModel):
    """Request to create a new model."""

    name: str = Field(..., description="Model name (e.g., 'llama3.1-8b')")
    litellm_model_name: str = Field(..., description="LiteLLM model identifier")
    class_: str = Field(
        ..., alias="class", description="Model class: 'open' or 'small'"
    )

    class Config:
        populate_by_name = True


class UpdateModelRequest(BaseModel):
    """Request to update a model."""

    name: Optional[str] = Field(None, description="Model name")
    litellm_model_name: Optional[str] = Field(
        None, description="LiteLLM model identifier"
    )
    class_: Optional[str] = Field(
        None, alias="class", description="Model class: 'open' or 'small'"
    )
    active: Optional[bool] = Field(None, description="Whether this model is active")

    class Config:
        populate_by_name = True


# ===== Prompt Management =====


class PromptResponse(BaseModel):
    """Prompt configuration data."""

    id: int = Field(..., description="Unique prompt identifier")
    name: str = Field(..., description="Prompt version (e.g., 'v1', 'v2')")
    content: str = Field(..., description="Full prompt template content")
    forked_from: Optional[int] = Field(None, description="Parent prompt ID if forked")
    active: bool = Field(..., description="Whether this prompt is active")
    created_at: str = Field(..., description="ISO timestamp when prompt was created")


class PromptListResponse(BaseModel):
    """Response for listing prompts."""

    prompts: List[PromptResponse]
    total: int


class CreatePromptRequest(BaseModel):
    """Request to create a new prompt."""

    name: str = Field(..., description="Prompt version (e.g., 'v1', 'v2')")
    content: str = Field(..., description="Full prompt template content")
    forked_from: Optional[int] = Field(None, description="Parent prompt ID if forked")


class UpdatePromptRequest(BaseModel):
    """Request to update a prompt."""

    name: Optional[str] = Field(None, description="Prompt version")
    content: Optional[str] = Field(None, description="Prompt template content")
    active: Optional[bool] = Field(None, description="Whether this prompt is active")


# ===== Kind Management =====


class KindResponse(BaseModel):
    """Kind (agent type) configuration data."""

    id: int = Field(..., description="Unique kind identifier")
    name: str = Field(
        ..., description="Kind name (e.g., 'standard', 'advanced', 'experimental')"
    )
    active: bool = Field(..., description="Whether this kind is active")
    created_at: str = Field(..., description="ISO timestamp when kind was created")


class KindListResponse(BaseModel):
    """Response for listing kinds."""

    kinds: List[KindResponse]
    total: int


class CreateKindRequest(BaseModel):
    """Request to create a new kind."""

    name: str = Field(..., description="Kind name (e.g., 'standard', 'advanced')")


class UpdateKindRequest(BaseModel):
    """Request to update a kind."""

    name: Optional[str] = Field(None, description="Kind name")
    active: Optional[bool] = Field(None, description="Whether this kind is active")


# ===== Game Summary =====


class GameSummaryResponse(BaseModel):
    """Markdown summary of a game."""

    markdown: str = Field(..., description="Markdown-formatted game summary")


# ===== LLM Call Models =====


class LLMCallResponse(BaseModel):
    """Individual LLM API call data."""

    id: int = Field(..., description="Unique LLM call identifier")
    game_turn_id: int = Field(..., description="Associated game turn ID")
    model: str = Field(..., description="LLM model used")
    messages: list = Field(..., description="Request messages sent to LLM")
    response: dict = Field(..., description="Full response from LLM")
    request_sent_at: str = Field(..., description="ISO timestamp when request was sent")
    response_received_at: str = Field(
        ..., description="ISO timestamp when response was received"
    )
    total_response_time_ms: int = Field(
        ..., description="Total response time in milliseconds"
    )
    time_to_first_token_ms: Optional[int] = Field(
        None, description="Time to first token in milliseconds"
    )
    tokens_per_second: Optional[float] = Field(
        None, description="Tokens generated per second"
    )
    response_tokens: int = Field(..., description="Number of tokens in response")
    total_tokens_consumed: int = Field(..., description="Total tokens consumed")


class LLMCallListResponse(BaseModel):
    """Response for listing LLM calls."""

    llm_calls: List[LLMCallResponse]
    total: int
