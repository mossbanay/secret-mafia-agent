"""
API routes for the mindgames arena backend.

Implements all endpoints required by the UI as specified in data_models.md.
"""

from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, distinct
from sqlalchemy.orm import Session, joinedload

from mindgames.db import Repository, get_session
from mindgames.db.models import Agent, Game, GamePlayer, GameTurn, Kind, Model, Prompt
from .schemas import (
    HeaderStatsResponse,
    AgentResponse,
    AgentListResponse,
    UpdateAliasRequest,
    UpdateActiveRequest,
    RolloutResponse,
    RolloutListResponse,
    GameDetailsResponse,
    GamePlayerResponse,
    GameMessageResponse,
    GameSummaryResponse,
    FilterValuesResponse,
    ModelResponse,
    ModelListResponse,
    CreateModelRequest,
    UpdateModelRequest,
    PromptResponse,
    PromptListResponse,
    CreatePromptRequest,
    UpdatePromptRequest,
    KindResponse,
    KindListResponse,
    CreateKindRequest,
    UpdateKindRequest,
    LLMCallResponse,
    LLMCallListResponse,
)

# Create router
router = APIRouter(prefix="/api")


# Dependency to get database session
def get_db_session() -> Session:
    """
    Dependency that provides a database session.

    Yields:
        SQLAlchemy session instance
    """
    with get_session() as session:
        yield session


# ===== Stats Endpoints =====


@router.get("/stats", response_model=HeaderStatsResponse)
async def get_stats(session: Session = Depends(get_db_session)):
    """
    Get header statistics.

    Returns total completed games and total spent (currently placeholder).
    """
    # Get total completed games count (matches what's shown in rollouts)
    total_games = (
        session.query(func.count(Game.id)).filter(Game.status == "completed").scalar()
        or 0
    )

    # TODO: Calculate total_spent from LLM calls when cost tracking is implemented
    total_spent = 0.0

    return HeaderStatsResponse(total_spent=total_spent, total_games=total_games)


# ===== Agent Endpoints =====


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    kind: Optional[str] = Query(None, description="Filter by kind"),
    model: Optional[str] = Query(None, description="Filter by model"),
    prompt: Optional[str] = Query(None, description="Filter by prompt"),
    class_: Optional[str] = Query(
        None, alias="class", description="Filter by class (open=all models, small=<=8B)"
    ),
    sort: Optional[str] = Query("trueskill", description="Sort column"),
    order: Optional[str] = Query("desc", description="Sort order: asc or desc"),
    limit: Optional[int] = Query(100, description="Maximum results"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
    session: Session = Depends(get_db_session),
):
    """
    List all agents with optional filtering and sorting.

    Supports filtering by kind, model, prompt, and class.
    """
    # Build query with joins
    query = (
        select(Agent)
        .join(Kind, Agent.kind_id == Kind.id)
        .join(Model, Agent.model_id == Model.id)
        .join(Prompt, Agent.prompt_id == Prompt.id)
        .options(
            joinedload(Agent.kind),
            joinedload(Agent.model),
            joinedload(Agent.prompt),
            joinedload(Agent.trueskill),
        )
    )

    # Apply filters
    if kind and kind != "all":
        query = query.where(Kind.name == kind)
    if model and model != "all":
        query = query.where(Model.name == model)
    if prompt and prompt != "all":
        query = query.where(Prompt.name == prompt)
    # Class filter: "open" shows all models, "small" shows only small models
    if class_ == "small":
        query = query.where(Model.class_ == "small")

    # Get total count before pagination
    count_query = select(func.count()).select_from(query.subquery())
    total = session.execute(count_query).scalar() or 0

    # Apply sorting
    if sort == "id":
        query = query.order_by(Agent.id.desc() if order == "desc" else Agent.id.asc())
    elif sort == "kind":
        query = query.order_by(Kind.name.desc() if order == "desc" else Kind.name.asc())
    elif sort == "model":
        query = query.order_by(
            Model.name.desc() if order == "desc" else Model.name.asc()
        )
    elif sort == "prompt":
        query = query.order_by(
            Prompt.name.desc() if order == "desc" else Prompt.name.asc()
        )
    elif sort == "class":
        query = query.order_by(
            Model.class_.desc() if order == "desc" else Model.class_.asc()
        )
    elif sort == "games":
        # Count completed games per agent
        games_subquery = (
            select(
                GamePlayer.agent_id, func.count(GamePlayer.game_id).label("game_count")
            )
            .join(Game, GamePlayer.game_id == Game.id)
            .where(Game.status == "completed")
            .group_by(GamePlayer.agent_id)
            .subquery()
        )
        query = query.outerjoin(games_subquery, Agent.id == games_subquery.c.agent_id)
        query = query.order_by(
            games_subquery.c.game_count.desc()
            if order == "desc"
            else games_subquery.c.game_count.asc()
        )
    elif sort == "trueskill":
        # Sort by TrueSkill rating (mu - 3*sigma)
        from mindgames.db.models import AgentTrueSkill

        query = query.outerjoin(AgentTrueSkill, Agent.id == AgentTrueSkill.agent_id)
        trueskill_rating = AgentTrueSkill.mu - 3 * AgentTrueSkill.sigma
        query = query.order_by(
            trueskill_rating.desc() if order == "desc" else trueskill_rating.asc()
        )
    elif sort == "mu":
        # Sort by mu
        from mindgames.db.models import AgentTrueSkill

        query = query.outerjoin(AgentTrueSkill, Agent.id == AgentTrueSkill.agent_id)
        query = query.order_by(
            AgentTrueSkill.mu.desc() if order == "desc" else AgentTrueSkill.mu.asc()
        )
    elif sort == "sigma":
        # Sort by sigma
        from mindgames.db.models import AgentTrueSkill

        query = query.outerjoin(AgentTrueSkill, Agent.id == AgentTrueSkill.agent_id)
        query = query.order_by(
            AgentTrueSkill.sigma.desc()
            if order == "desc"
            else AgentTrueSkill.sigma.asc()
        )

    # Apply pagination
    query = query.limit(limit).offset(offset)

    # Execute query
    result = session.execute(query).unique()
    agents = result.scalars().all()

    # Build response
    agent_responses = []
    for agent in agents:
        # Count completed games for this agent
        games_count = (
            session.query(func.count(GamePlayer.id))
            .join(Game, GamePlayer.game_id == Game.id)
            .filter(GamePlayer.agent_id == agent.id, Game.status == "completed")
            .scalar()
            or 0
        )

        # Get TrueSkill rating
        trueskill_rating = 0.0
        mu = 25.0
        sigma = 8.333
        if agent.trueskill:
            trueskill_rating = agent.trueskill.mu - 3 * agent.trueskill.sigma
            mu = agent.trueskill.mu
            sigma = agent.trueskill.sigma

        agent_responses.append(
            AgentResponse(
                id=str(agent.id),
                alias=agent.alias,
                kind=agent.kind.name,
                model=agent.model.name,
                prompt=agent.prompt.name,
                class_=agent.model.class_,
                active=bool(agent.active),
                games=games_count,
                trueskill=round(trueskill_rating, 2),
                mu=round(mu, 2),
                sigma=round(sigma, 2),
            )
        )

    return AgentListResponse(agents=agent_responses, total=total)


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, session: Session = Depends(get_db_session)):
    """Get a specific agent by ID."""
    repo = Repository(session)
    agent = repo.get_agent_by_id(agent_id, with_relations=True)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Count completed games
    games_count = (
        session.query(func.count(GamePlayer.id))
        .join(Game, GamePlayer.game_id == Game.id)
        .filter(GamePlayer.agent_id == agent.id, Game.status == "completed")
        .scalar()
        or 0
    )

    # Get TrueSkill rating
    trueskill_rating = 0.0
    mu = 25.0
    sigma = 8.333
    if agent.trueskill:
        trueskill_rating = agent.trueskill.mu - 3 * agent.trueskill.sigma
        mu = agent.trueskill.mu
        sigma = agent.trueskill.sigma

    return AgentResponse(
        id=str(agent.id),
        alias=agent.alias,
        kind=agent.kind.name,
        model=agent.model.name,
        prompt=agent.prompt.name,
        class_=agent.model.class_,
        active=bool(agent.active),
        games=games_count,
        trueskill=round(trueskill_rating, 2),
        mu=round(mu, 2),
        sigma=round(sigma, 2),
    )


@router.patch("/agents/{agent_id}/alias", response_model=AgentResponse)
async def update_agent_alias(
    agent_id: int,
    request: UpdateAliasRequest,
    session: Session = Depends(get_db_session),
):
    """Update an agent's alias."""
    repo = Repository(session)
    agent = repo.get_agent_by_id(agent_id, with_relations=True)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update alias
    agent.alias = request.alias
    session.flush()

    # Count completed games
    games_count = (
        session.query(func.count(GamePlayer.id))
        .join(Game, GamePlayer.game_id == Game.id)
        .filter(GamePlayer.agent_id == agent.id, Game.status == "completed")
        .scalar()
        or 0
    )

    # Get TrueSkill rating
    trueskill_rating = 0.0
    mu = 25.0
    sigma = 8.333
    if agent.trueskill:
        trueskill_rating = agent.trueskill.mu - 3 * agent.trueskill.sigma
        mu = agent.trueskill.mu
        sigma = agent.trueskill.sigma

    return AgentResponse(
        id=str(agent.id),
        alias=agent.alias,
        kind=agent.kind.name,
        model=agent.model.name,
        prompt=agent.prompt.name,
        class_=agent.model.class_,
        active=bool(agent.active),
        games=games_count,
        trueskill=round(trueskill_rating, 2),
        mu=round(mu, 2),
        sigma=round(sigma, 2),
    )


@router.patch("/agents/{agent_id}/active", response_model=AgentResponse)
async def update_agent_active(
    agent_id: int,
    request: UpdateActiveRequest,
    session: Session = Depends(get_db_session),
):
    """Update an agent's active status."""
    repo = Repository(session)
    agent = repo.get_agent_by_id(agent_id, with_relations=True)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update active status
    agent.active = 1 if request.active else 0
    session.flush()

    # Count completed games
    games_count = (
        session.query(func.count(GamePlayer.id))
        .join(Game, GamePlayer.game_id == Game.id)
        .filter(GamePlayer.agent_id == agent.id, Game.status == "completed")
        .scalar()
        or 0
    )

    # Get TrueSkill rating
    trueskill_rating = 0.0
    mu = 25.0
    sigma = 8.333
    if agent.trueskill:
        trueskill_rating = agent.trueskill.mu - 3 * agent.trueskill.sigma
        mu = agent.trueskill.mu
        sigma = agent.trueskill.sigma

    return AgentResponse(
        id=str(agent.id),
        alias=agent.alias,
        kind=agent.kind.name,
        model=agent.model.name,
        prompt=agent.prompt.name,
        class_=agent.model.class_,
        active=bool(agent.active),
        games=games_count,
        trueskill=round(trueskill_rating, 2),
        mu=round(mu, 2),
        sigma=round(sigma, 2),
    )


# ===== Rollout/Game List Endpoints =====


@router.get("/rollouts", response_model=RolloutListResponse)
async def list_rollouts(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    sort: Optional[str] = Query("finished_at", description="Sort column"),
    order: Optional[str] = Query("desc", description="Sort order: asc or desc"),
    session: Session = Depends(get_db_session),
):
    """
    List all rollouts (completed games).

    Returns games with their 6 players and outcome.
    """
    # Build query for completed games
    query = (
        select(Game)
        .where(Game.status == "completed")
        .options(joinedload(Game.players).joinedload(GamePlayer.agent))
    )

    # Apply sorting
    if sort == "game_id":
        query = query.order_by(Game.id.desc() if order == "desc" else Game.id.asc())
    elif sort == "finished_at":
        query = query.order_by(
            Game.finished_at.desc() if order == "desc" else Game.finished_at.asc()
        )
    elif sort == "winning_team":
        query = query.order_by(
            Game.winning_team.desc() if order == "desc" else Game.winning_team.asc()
        )
    else:
        # Default to finished_at desc
        query = query.order_by(Game.finished_at.desc())

    # Get total count
    count_query = select(func.count(Game.id)).where(Game.status == "completed")
    total = session.execute(count_query).scalar() or 0

    # Apply pagination
    offset = (page - 1) * limit
    query = query.limit(limit).offset(offset)

    # Execute query
    result = session.execute(query).unique()
    games = result.scalars().all()

    # Build response
    rollout_responses = []
    for game in games:
        # Get players ordered by player_index
        players = sorted(game.players, key=lambda p: p.player_index)

        # Ensure we have exactly 6 players
        if len(players) != 6:
            continue

        # Convert naive datetime to UTC-aware before getting timestamp
        finished_timestamp = 0
        if game.finished_at:
            # SQLite stores datetimes as naive, so we need to tell Python it's UTC
            utc_finished = game.finished_at.replace(tzinfo=timezone.utc)
            finished_timestamp = int(utc_finished.timestamp())

        # Use alias if available, otherwise agent ID
        agent_displays = [
            players[i].agent.alias or str(players[i].agent_id) for i in range(6)
        ]

        rollout_responses.append(
            RolloutResponse(
                game_id=str(game.id),
                agent_1=agent_displays[0],
                agent_2=agent_displays[1],
                agent_3=agent_displays[2],
                agent_4=agent_displays[3],
                agent_5=agent_displays[4],
                agent_6=agent_displays[5],
                finished_at=finished_timestamp,
                winning_team=game.winning_team or "Unknown",
            )
        )

    return RolloutListResponse(rollouts=rollout_responses, total=total)


# ===== Game Details Endpoint =====


@router.get("/games/{game_id}", response_model=GameDetailsResponse)
async def get_game_details(game_id: int, session: Session = Depends(get_db_session)):
    """
    Get full details for a specific game.

    Includes players, turns, and messages.
    """
    repo = Repository(session)
    game = repo.get_game_by_id(game_id, with_relations=True)

    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Get players ordered by player_index
    players = sorted(game.players, key=lambda p: p.player_index)

    # Build player responses
    player_responses = []
    for player in players:
        # Calculate average response time for this player
        avg_response_time = (
            session.query(func.avg(GameTurn.response_time_ms))
            .filter(
                GameTurn.game_id == game.id,
                GameTurn.player_index == player.player_index,
            )
            .scalar()
            or 0
        )

        player_responses.append(
            GamePlayerResponse(
                agent_id=str(player.agent_id),
                alias=player.agent.alias,
                kind=player.agent.kind.name,
                model=player.agent.model.name,
                prompt=player.agent.prompt.name,
                role=player.role or "Unknown",
                reward=float(player.reward),
                avg_response_time=int(avg_response_time),
            )
        )

    # Get turns ordered by turn_number
    turns = repo.get_game_turns(game.id)

    # Build message responses
    message_responses = []
    for turn in turns:
        # Find the player for this turn
        player = next((p for p in players if p.player_index == turn.player_index), None)
        if not player:
            continue

        # Use alias if available, otherwise agent_id
        agent_name = player.agent.alias or str(player.agent_id)

        # Extract action text from observation or use action directly
        content = turn.action

        # Calculate token breakdown from LLM calls
        # If we have LLM call data, use it; otherwise fall back to turn.tokens_used
        observation_tokens = 0
        action_tokens = 0
        total_tokens = 0

        if turn.llm_calls and len(turn.llm_calls) > 0:
            # Use the first (and typically only) LLM call for this turn
            llm_call = turn.llm_calls[0]
            action_tokens = llm_call.response_tokens
            total_tokens = llm_call.total_tokens_consumed
            observation_tokens = total_tokens - action_tokens
        else:
            # Fallback to turn.tokens_used if no LLM call data
            total_tokens = turn.tokens_used or 0
            action_tokens = total_tokens  # Assume all tokens are action tokens
            observation_tokens = 0

        # Count LLM calls for this turn
        llm_call_count = len(turn.llm_calls) if turn.llm_calls else 0

        message_responses.append(
            GameMessageResponse(
                turn_id=turn.id,
                turn=turn.turn_number,
                player_index=turn.player_index,
                agent=agent_name,
                kind=player.agent.kind.name,
                model=player.agent.model.name,
                prompt=player.agent.prompt.name,
                role=player.role or "Unknown",
                content=content,
                response_time=turn.response_time_ms,
                observation_tokens=observation_tokens,
                action_tokens=action_tokens,
                tokens=total_tokens,
                llm_call_count=llm_call_count,
            )
        )

    return GameDetailsResponse(
        game_id=str(game.id),
        turns=game.total_turns,
        winning_team=game.winning_team or "Unknown",
        players=player_responses,
        messages=message_responses,
    )


@router.get("/games/{game_id}/summary", response_model=GameSummaryResponse)
async def get_game_summary(game_id: int, session: Session = Depends(get_db_session)):
    """
    Get a Markdown-formatted summary of a game.

    Includes player roles and all actions in sequence.
    """
    repo = Repository(session)
    game = repo.get_game_by_id(game_id, with_relations=True)

    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Get players ordered by player_index
    players = sorted(game.players, key=lambda p: p.player_index)

    # Get turns ordered by turn_number
    turns = repo.get_game_turns(game.id)

    # Build markdown summary
    markdown_lines = []

    # Header
    markdown_lines.append(f"# Game {game.id} Summary")
    markdown_lines.append("")
    markdown_lines.append(f"**Winning Team:** {game.winning_team or 'Unknown'}")
    markdown_lines.append(f"**Total Turns:** {game.total_turns}")
    markdown_lines.append("")

    # Players section
    markdown_lines.append("## Players")
    markdown_lines.append("")
    for player in players:
        agent_name = player.agent.alias or f"Agent {player.agent_id}"
        markdown_lines.append(f"- **Player {player.player_index}** ({agent_name})")
        markdown_lines.append(f"  - **Role:** {player.role or 'Unknown'}")
        markdown_lines.append(f"  - **Kind:** {player.agent.kind.name}")
        markdown_lines.append(f"  - **Model:** {player.agent.model.name}")
        markdown_lines.append(f"  - **Prompt:** {player.agent.prompt.name}")
        markdown_lines.append(f"  - **Reward:** {player.reward}")
        markdown_lines.append("")

    # Actions section
    markdown_lines.append("## Game Actions")
    markdown_lines.append("")

    for turn in turns:
        # Find the player for this turn
        player = next((p for p in players if p.player_index == turn.player_index), None)
        if not player:
            continue

        agent_name = player.agent.alias or f"Agent {player.agent_id}"

        markdown_lines.append(f"### Turn {turn.turn_number}")
        markdown_lines.append(
            f"**Player {turn.player_index}** ({agent_name}) - {player.role or 'Unknown'}"
        )
        markdown_lines.append("")
        markdown_lines.append(f"**Action:** {turn.action}")
        markdown_lines.append("")

    markdown = "\n".join(markdown_lines)

    return GameSummaryResponse(markdown=markdown)


@router.get("/turns/{turn_id}/llm-calls", response_model=LLMCallListResponse)
async def get_turn_llm_calls(turn_id: int, session: Session = Depends(get_db_session)):
    """
    Get all LLM API calls for a specific game turn.

    Returns the complete request/response data for each LLM call made during this turn.
    """
    repo = Repository(session)

    # Verify turn exists
    turn = session.get(GameTurn, turn_id)
    if not turn:
        raise HTTPException(status_code=404, detail="Game turn not found")

    # Get all LLM calls for this turn
    llm_calls = repo.get_turn_llm_calls(turn_id)

    # Build response
    llm_call_responses = []
    for llm_call in llm_calls:
        llm_call_responses.append(
            LLMCallResponse(
                id=llm_call.id,
                game_turn_id=llm_call.game_turn_id,
                model=llm_call.model,
                messages=llm_call.messages,
                response=llm_call.response,
                request_sent_at=llm_call.request_sent_at.isoformat(),
                response_received_at=llm_call.response_received_at.isoformat(),
                total_response_time_ms=llm_call.total_response_time_ms,
                time_to_first_token_ms=llm_call.time_to_first_token_ms,
                tokens_per_second=llm_call.tokens_per_second,
                response_tokens=llm_call.response_tokens,
                total_tokens_consumed=llm_call.total_tokens_consumed,
            )
        )

    return LLMCallListResponse(
        llm_calls=llm_call_responses, total=len(llm_call_responses)
    )


# ===== Filter Endpoints =====


@router.get("/filters/kinds", response_model=FilterValuesResponse)
async def get_filter_kinds(session: Session = Depends(get_db_session)):
    """Get all available agent kinds for filtering."""
    kinds = session.query(distinct(Kind.name)).all()
    return FilterValuesResponse(values=[k[0] for k in kinds])


@router.get("/filters/models", response_model=FilterValuesResponse)
async def get_filter_models(session: Session = Depends(get_db_session)):
    """Get all available models for filtering."""
    models = session.query(distinct(Model.name)).all()
    return FilterValuesResponse(values=[m[0] for m in models])


@router.get("/filters/prompts", response_model=FilterValuesResponse)
async def get_filter_prompts(session: Session = Depends(get_db_session)):
    """Get all available prompt versions for filtering."""
    prompts = session.query(distinct(Prompt.name)).all()
    return FilterValuesResponse(values=[p[0] for p in prompts])


@router.get("/filters/classes", response_model=FilterValuesResponse)
async def get_filter_classes(session: Session = Depends(get_db_session)):
    """Get all available model classes for filtering."""
    classes = session.query(distinct(Model.class_)).all()
    return FilterValuesResponse(values=[c[0] for c in classes if c[0]])


# ===== Model Management Endpoints =====


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    active_only: bool = Query(False, description="Filter to only active models"),
    session: Session = Depends(get_db_session),
):
    """
    List all models.

    By default returns all models. Set active_only=true to filter to only active models.
    """
    query = select(Model).order_by(Model.created_at.desc())

    if active_only:
        query = query.where(Model.active == 1)

    result = session.execute(query)
    models = result.scalars().all()

    model_responses = [
        ModelResponse(
            id=model.id,
            name=model.name,
            litellm_model_name=model.litellm_model_name,
            class_=model.class_,
            active=bool(model.active),
            created_at=model.created_at.isoformat(),
        )
        for model in models
    ]

    return ModelListResponse(models=model_responses, total=len(model_responses))


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, session: Session = Depends(get_db_session)):
    """Get a specific model by ID."""
    model = session.query(Model).filter(Model.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelResponse(
        id=model.id,
        name=model.name,
        litellm_model_name=model.litellm_model_name,
        class_=model.class_,
        active=bool(model.active),
        created_at=model.created_at.isoformat(),
    )


@router.post("/models", response_model=ModelResponse, status_code=201)
async def create_model(
    request: CreateModelRequest, session: Session = Depends(get_db_session)
):
    """Create a new model."""
    # Validate class value
    if request.class_ not in ["open", "small"]:
        raise HTTPException(status_code=400, detail="class must be 'open' or 'small'")

    # Check if model name already exists
    existing = session.query(Model).filter(Model.name == request.name).first()
    if existing:
        raise HTTPException(
            status_code=400, detail=f"Model with name '{request.name}' already exists"
        )

    # Create new model
    model = Model(
        name=request.name,
        litellm_model_name=request.litellm_model_name,
        class_=request.class_,
        active=1,
        created_at=datetime.now(timezone.utc),
    )

    session.add(model)
    session.flush()

    return ModelResponse(
        id=model.id,
        name=model.name,
        litellm_model_name=model.litellm_model_name,
        class_=model.class_,
        active=bool(model.active),
        created_at=model.created_at.isoformat(),
    )


@router.patch("/models/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    request: UpdateModelRequest,
    session: Session = Depends(get_db_session),
):
    """Update a model."""
    model = session.query(Model).filter(Model.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Update fields if provided
    if request.name is not None:
        # Check if new name conflicts with existing model
        existing = (
            session.query(Model)
            .filter(Model.name == request.name, Model.id != model_id)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Model with name '{request.name}' already exists",
            )
        model.name = request.name

    if request.litellm_model_name is not None:
        model.litellm_model_name = request.litellm_model_name

    if request.class_ is not None:
        if request.class_ not in ["open", "small"]:
            raise HTTPException(
                status_code=400, detail="class must be 'open' or 'small'"
            )
        model.class_ = request.class_

    if request.active is not None:
        model.active = 1 if request.active else 0

    return ModelResponse(
        id=model.id,
        name=model.name,
        litellm_model_name=model.litellm_model_name,
        class_=model.class_,
        active=bool(model.active),
        created_at=model.created_at.isoformat(),
    )


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(model_id: int, session: Session = Depends(get_db_session)):
    """
    Delete a model.

    Note: This will fail if there are any agents using this model due to foreign key constraints.
    Consider deactivating the model instead using PATCH /models/{model_id} with active=false.
    """
    model = session.query(Model).filter(Model.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        session.delete(model)
        session.flush()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete model that is in use by agents. Deactivate it instead.",
        )


# ===== Prompt Management Endpoints =====


@router.get("/prompts", response_model=PromptListResponse)
async def list_prompts(
    active_only: bool = Query(False, description="Filter to only active prompts"),
    session: Session = Depends(get_db_session),
):
    """
    List all prompts.

    By default returns all prompts. Set active_only=true to filter to only active prompts.
    """
    query = select(Prompt).order_by(Prompt.created_at.desc())

    if active_only:
        query = query.where(Prompt.active == 1)

    result = session.execute(query)
    prompts = result.scalars().all()

    prompt_responses = [
        PromptResponse(
            id=prompt.id,
            name=prompt.name,
            content=prompt.content,
            forked_from=prompt.forked_from,
            active=bool(prompt.active),
            created_at=prompt.created_at.isoformat(),
        )
        for prompt in prompts
    ]

    return PromptListResponse(prompts=prompt_responses, total=len(prompt_responses))


@router.get("/prompts/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: int, session: Session = Depends(get_db_session)):
    """Get a specific prompt by ID."""
    prompt = session.query(Prompt).filter(Prompt.id == prompt_id).first()

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return PromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        forked_from=prompt.forked_from,
        active=bool(prompt.active),
        created_at=prompt.created_at.isoformat(),
    )


@router.post("/prompts", response_model=PromptResponse, status_code=201)
async def create_prompt(
    request: CreatePromptRequest, session: Session = Depends(get_db_session)
):
    """Create a new prompt."""
    # Check if prompt name already exists
    existing = session.query(Prompt).filter(Prompt.name == request.name).first()
    if existing:
        raise HTTPException(
            status_code=400, detail=f"Prompt with name '{request.name}' already exists"
        )

    # If forked_from is provided, validate parent exists
    if request.forked_from is not None:
        parent = session.query(Prompt).filter(Prompt.id == request.forked_from).first()
        if not parent:
            raise HTTPException(
                status_code=400,
                detail=f"Parent prompt with ID {request.forked_from} not found",
            )

    # Create new prompt
    prompt = Prompt(
        name=request.name,
        content=request.content,
        forked_from=request.forked_from,
        active=1,
        created_at=datetime.now(timezone.utc),
    )

    session.add(prompt)
    session.flush()

    return PromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        forked_from=prompt.forked_from,
        active=bool(prompt.active),
        created_at=prompt.created_at.isoformat(),
    )


@router.patch("/prompts/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
    prompt_id: int,
    request: UpdatePromptRequest,
    session: Session = Depends(get_db_session),
):
    """Update a prompt."""
    prompt = session.query(Prompt).filter(Prompt.id == prompt_id).first()

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # Update fields if provided
    if request.name is not None:
        # Check if new name conflicts with existing prompt
        existing = (
            session.query(Prompt)
            .filter(Prompt.name == request.name, Prompt.id != prompt_id)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt with name '{request.name}' already exists",
            )
        prompt.name = request.name

    if request.content is not None:
        prompt.content = request.content

    if request.active is not None:
        prompt.active = 1 if request.active else 0

    session.flush()

    return PromptResponse(
        id=prompt.id,
        name=prompt.name,
        content=prompt.content,
        forked_from=prompt.forked_from,
        active=bool(prompt.active),
        created_at=prompt.created_at.isoformat(),
    )


@router.delete("/prompts/{prompt_id}", status_code=204)
async def delete_prompt(prompt_id: int, session: Session = Depends(get_db_session)):
    """
    Delete a prompt.

    Note: This will fail if there are any agents using this prompt due to foreign key constraints.
    Consider deactivating the prompt instead using PATCH /prompts/{prompt_id} with active=false.
    """
    prompt = session.query(Prompt).filter(Prompt.id == prompt_id).first()

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    try:
        session.delete(prompt)
        session.flush()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete prompt that is in use by agents. Deactivate it instead.",
        )


# ===== Kind Management Endpoints =====


@router.get("/kinds", response_model=KindListResponse)
async def list_kinds(
    active_only: bool = Query(False, description="Filter to only active kinds"),
    session: Session = Depends(get_db_session),
):
    """
    List all kinds (agent types).

    By default returns all kinds. Set active_only=true to filter to only active kinds.
    """
    query = select(Kind).order_by(Kind.created_at.desc())

    if active_only:
        query = query.where(Kind.active == 1)

    result = session.execute(query)
    kinds = result.scalars().all()

    kind_responses = [
        KindResponse(
            id=kind.id,
            name=kind.name,
            active=bool(kind.active),
            created_at=kind.created_at.isoformat(),
        )
        for kind in kinds
    ]

    return KindListResponse(kinds=kind_responses, total=len(kind_responses))


@router.get("/kinds/{kind_id}", response_model=KindResponse)
async def get_kind(kind_id: int, session: Session = Depends(get_db_session)):
    """Get a specific kind by ID."""
    kind = session.query(Kind).filter(Kind.id == kind_id).first()

    if not kind:
        raise HTTPException(status_code=404, detail="Kind not found")

    return KindResponse(
        id=kind.id,
        name=kind.name,
        active=bool(kind.active),
        created_at=kind.created_at.isoformat(),
    )


@router.post("/kinds", response_model=KindResponse, status_code=201)
async def create_kind(
    request: CreateKindRequest, session: Session = Depends(get_db_session)
):
    """Create a new kind (agent type)."""
    # Check if kind name already exists
    existing = session.query(Kind).filter(Kind.name == request.name).first()
    if existing:
        raise HTTPException(
            status_code=400, detail=f"Kind with name '{request.name}' already exists"
        )

    # Create new kind
    kind = Kind(name=request.name, active=1, created_at=datetime.now(timezone.utc))

    session.add(kind)
    session.flush()

    return KindResponse(
        id=kind.id,
        name=kind.name,
        active=bool(kind.active),
        created_at=kind.created_at.isoformat(),
    )


@router.patch("/kinds/{kind_id}", response_model=KindResponse)
async def update_kind(
    kind_id: int, request: UpdateKindRequest, session: Session = Depends(get_db_session)
):
    """Update a kind."""
    kind = session.query(Kind).filter(Kind.id == kind_id).first()

    if not kind:
        raise HTTPException(status_code=404, detail="Kind not found")

    # Update fields if provided
    if request.name is not None:
        # Check if new name conflicts with existing kind
        existing = (
            session.query(Kind)
            .filter(Kind.name == request.name, Kind.id != kind_id)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Kind with name '{request.name}' already exists",
            )
        kind.name = request.name

    if request.active is not None:
        kind.active = 1 if request.active else 0

    session.flush()

    return KindResponse(
        id=kind.id,
        name=kind.name,
        active=bool(kind.active),
        created_at=kind.created_at.isoformat(),
    )


@router.delete("/kinds/{kind_id}", status_code=204)
async def delete_kind(kind_id: int, session: Session = Depends(get_db_session)):
    """
    Delete a kind.

    Note: This will fail if there are any agents using this kind due to foreign key constraints.
    Consider deactivating the kind instead using PATCH /kinds/{kind_id} with active=false.
    """
    kind = session.query(Kind).filter(Kind.id == kind_id).first()

    if not kind:
        raise HTTPException(status_code=404, detail="Kind not found")

    try:
        session.delete(kind)
        session.flush()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete kind that is in use by agents. Deactivate it instead.",
        )
