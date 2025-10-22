"""
Data access layer (repository pattern) for database operations.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session, joinedload

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


class Repository:
    """
    Repository for database operations.

    Provides high-level methods for common database queries and operations.
    """

    def __init__(self, session: Session):
        """
        Initialize repository with a database session.

        Args:
            session: SQLAlchemy session instance
        """
        self.session = session

    # ===== Model Operations =====

    def create_model(
        self,
        name: str,
        litellm_model_name: str,
        class_: str = "open",
        active: bool = True,
    ) -> Model:
        """Create a new model."""
        model = Model(
            name=name,
            litellm_model_name=litellm_model_name,
            class_=class_,
            active=1 if active else 0,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(model)
        self.session.flush()
        return model

    def get_model_by_id(self, model_id: int) -> Model | None:
        """Get model by ID."""
        return self.session.get(Model, model_id)

    def get_model_by_name(self, name: str) -> Model | None:
        """Get model by name."""
        stmt = select(Model).where(Model.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_active_models(self) -> list[Model]:
        """Get all active models."""
        stmt = select(Model).where(Model.active == 1)
        return list(self.session.execute(stmt).scalars())

    def set_model_active(self, model_id: int, active: bool) -> bool:
        """Set model active status."""
        model = self.get_model_by_id(model_id)
        if model:
            model.active = 1 if active else 0
            self.session.flush()
            return True
        return False

    # ===== Kind Operations =====

    def create_kind(self, name: str, active: bool = True) -> Kind:
        """Create a new kind."""
        kind = Kind(
            name=name,
            active=1 if active else 0,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(kind)
        self.session.flush()
        return kind

    def get_kind_by_id(self, kind_id: int) -> Kind | None:
        """Get kind by ID."""
        return self.session.get(Kind, kind_id)

    def get_kind_by_name(self, name: str) -> Kind | None:
        """Get kind by name."""
        stmt = select(Kind).where(Kind.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_active_kinds(self) -> list[Kind]:
        """Get all active kinds."""
        stmt = select(Kind).where(Kind.active == 1)
        return list(self.session.execute(stmt).scalars())

    def set_kind_active(self, kind_id: int, active: bool) -> bool:
        """Set kind active status."""
        kind = self.get_kind_by_id(kind_id)
        if kind:
            kind.active = 1 if active else 0
            self.session.flush()
            return True
        return False

    # ===== Prompt Operations =====

    def create_prompt(
        self,
        name: str,
        content: str,
        forked_from: int | None = None,
        active: bool = True,
    ) -> Prompt:
        """Create a new prompt."""
        prompt = Prompt(
            name=name,
            content=content,
            forked_from=forked_from,
            active=1 if active else 0,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(prompt)
        self.session.flush()
        return prompt

    def get_prompt_by_id(self, prompt_id: int) -> Prompt | None:
        """Get prompt by ID."""
        return self.session.get(Prompt, prompt_id)

    def get_prompt_by_name(self, name: str) -> Prompt | None:
        """Get prompt by name."""
        stmt = select(Prompt).where(Prompt.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_active_prompts(self) -> list[Prompt]:
        """Get all active prompts."""
        stmt = select(Prompt).where(Prompt.active == 1)
        return list(self.session.execute(stmt).scalars())

    def set_prompt_active(self, prompt_id: int, active: bool) -> bool:
        """Set prompt active status."""
        prompt = self.get_prompt_by_id(prompt_id)
        if prompt:
            prompt.active = 1 if active else 0
            self.session.flush()
            return True
        return False

    # ===== Agent Operations =====

    def create_agent(
        self,
        kind_id: int,
        model_id: int,
        prompt_id: int,
        alias: str | None = None,
    ) -> Agent:
        """Create a new agent."""
        agent = Agent(
            kind_id=kind_id,
            model_id=model_id,
            prompt_id=prompt_id,
            alias=alias,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(agent)
        self.session.flush()
        return agent

    def get_agent_by_id(
        self, agent_id: int, with_relations: bool = False
    ) -> Agent | None:
        """
        Get agent by ID.

        Args:
            agent_id: Agent ID
            with_relations: If True, eagerly load related kind, model, and prompt

        Returns:
            Agent instance or None
        """
        if with_relations:
            stmt = (
                select(Agent)
                .where(Agent.id == agent_id)
                .options(
                    joinedload(Agent.kind),
                    joinedload(Agent.model),
                    joinedload(Agent.prompt),
                )
            )
            return self.session.execute(stmt).scalar_one_or_none()
        return self.session.get(Agent, agent_id)

    def get_agent_by_config(
        self, kind_id: int, model_id: int, prompt_id: int
    ) -> Agent | None:
        """Get agent by configuration (kind, model, prompt).

        If multiple agents exist with the same configuration, returns the first one.
        """
        stmt = (
            select(Agent)
            .where(
                and_(
                    Agent.kind_id == kind_id,
                    Agent.model_id == model_id,
                    Agent.prompt_id == prompt_id,
                )
            )
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_or_create_agent(
        self,
        kind_id: int,
        model_id: int,
        prompt_id: int,
        alias: str | None = None,
    ) -> Agent:
        """
        Get existing agent or create new one with given configuration.

        Handles race conditions by catching integrity errors and retrying the get.
        """
        # First try to get existing agent
        agent = self.get_agent_by_config(kind_id, model_id, prompt_id)
        if agent:
            return agent

        # Try to create new agent
        try:
            new_agent = self.create_agent(kind_id, model_id, prompt_id, alias)
            return new_agent
        except Exception as e:
            # Handle race condition - another process may have created it
            # Roll back the failed transaction
            self.session.rollback()

            # Try to get the agent again
            agent = self.get_agent_by_config(kind_id, model_id, prompt_id)
            if agent:
                return agent

            # If still not found, re-raise the original error
            raise e

    def get_active_agents(self) -> list[Agent]:
        """Get all active agents."""
        stmt = select(Agent).where(Agent.active == 1)
        return list(self.session.execute(stmt).scalars())

    def set_agent_active(self, agent_id: int, active: bool) -> bool:
        """Set agent active status."""
        agent = self.get_agent_by_id(agent_id)
        if agent:
            agent.active = 1 if active else 0
            self.session.flush()
            return True
        return False

    def get_random_agents(self, count: int = 6) -> list[Agent]:
        """
        Get random agents by composing random active kinds, models, and prompts.

        Creates new agents if the combination doesn't exist.

        Args:
            count: Number of agents to generate

        Returns:
            List of Agent instances
        """
        agents = []
        for _ in range(count):
            # Get random active configurations
            kind_stmt = (
                select(Kind).where(Kind.active == 1).order_by(func.random()).limit(1)
            )
            kind = self.session.execute(kind_stmt).scalar_one()

            model_stmt = (
                select(Model).where(Model.active == 1).order_by(func.random()).limit(1)
            )
            model = self.session.execute(model_stmt).scalar_one()

            prompt_stmt = (
                select(Prompt)
                .where(Prompt.active == 1)
                .order_by(func.random())
                .limit(1)
            )
            prompt = self.session.execute(prompt_stmt).scalar_one()

            # Get or create agent with this combination
            agent = self.get_or_create_agent(kind.id, model.id, prompt.id)
            agents.append(agent)

        return agents

    # ===== Game Operations =====

    def create_game(
        self,
        started_at: datetime | None = None,
        status: str = "in_progress",
    ) -> Game:
        """Create a new game."""
        game = Game(
            started_at=started_at or datetime.now(timezone.utc),
            status=status,
            total_turns=0,
            environment_crashed=0,
        )
        self.session.add(game)
        self.session.flush()
        return game

    def get_game_by_id(self, game_id: int, with_relations: bool = False) -> Game | None:
        """
        Get game by ID.

        Args:
            game_id: Game ID
            with_relations: If True, eagerly load players and turns

        Returns:
            Game instance or None
        """
        if with_relations:
            stmt = (
                select(Game)
                .where(Game.id == game_id)
                .options(
                    joinedload(Game.players).joinedload(GamePlayer.agent),
                    joinedload(Game.turns),
                )
            )
            return self.session.execute(stmt).unique().scalar_one_or_none()
        return self.session.get(Game, game_id)

    def finish_game(
        self,
        game_id: int,
        status: str = "completed",
        winning_team: str | None = None,
        environment_crashed: bool = False,
        game_info: dict[str, Any] | None = None,
    ) -> bool:
        """Finish a game and set final state."""
        game = self.get_game_by_id(game_id)
        if game:
            game.finished_at = datetime.now(timezone.utc)
            game.status = status
            game.winning_team = winning_team
            game.environment_crashed = 1 if environment_crashed else 0
            if game_info is not None:
                game.game_info = game_info
            self.session.flush()
            return True
        return False

    def get_games(
        self,
        status: str | None = None,
        winning_team: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Game]:
        """
        Get games with optional filters.

        Args:
            status: Filter by game status
            winning_team: Filter by winning team
            limit: Maximum number of games to return
            offset: Number of games to skip

        Returns:
            List of Game instances
        """
        stmt = select(Game)

        if status:
            stmt = stmt.where(Game.status == status)
        if winning_team:
            stmt = stmt.where(Game.winning_team == winning_team)

        stmt = stmt.order_by(Game.started_at.desc())

        if limit:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)

        return list(self.session.execute(stmt).scalars())

    # ===== GamePlayer Operations =====

    def add_game_player(
        self,
        game_id: int,
        agent_id: int,
        player_index: int,
        role: str | None = None,
        reward: float | None = None,
    ) -> GamePlayer:
        """Add a player to a game."""
        player = GamePlayer(
            game_id=game_id,
            agent_id=agent_id,
            player_index=player_index,
            role=role,
            reward=reward,
        )
        self.session.add(player)
        self.session.flush()
        return player

    def update_game_player(
        self,
        game_id: int,
        player_index: int,
        role: str | None = None,
        reward: float | None = None,
    ) -> bool:
        """Update a game player's role and reward."""
        stmt = select(GamePlayer).where(
            and_(
                GamePlayer.game_id == game_id,
                GamePlayer.player_index == player_index,
            )
        )
        player = self.session.execute(stmt).scalar_one_or_none()
        if player:
            if role is not None:
                player.role = role
            if reward is not None:
                player.reward = reward
            self.session.flush()
            return True
        return False

    # ===== GameTurn Operations =====

    def add_game_turn(
        self,
        game_id: int,
        turn_number: int,
        player_index: int,
        observation: dict[str, Any],
        action: str,
        response_time_ms: int,
        step_info: dict[str, Any] | None = None,
        tokens_used: int | None = None,
        timestamp: datetime | None = None,
    ) -> GameTurn:
        """Add a turn to a game."""
        turn = GameTurn(
            game_id=game_id,
            turn_number=turn_number,
            player_index=player_index,
            observation=observation,
            action=action,
            step_info=step_info,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self.session.add(turn)
        self.session.flush()

        # Update game total_turns
        game = self.get_game_by_id(game_id)
        if game:
            game.total_turns = max(game.total_turns, turn_number)

        return turn

    def get_game_turns(self, game_id: int) -> list[GameTurn]:
        """Get all turns for a game, ordered by turn number."""
        stmt = (
            select(GameTurn)
            .where(GameTurn.game_id == game_id)
            .order_by(GameTurn.turn_number)
        )
        return list(self.session.execute(stmt).scalars())

    # ===== LLMCall Operations =====

    def add_llm_call(
        self,
        game_turn_id: int,
        model: str,
        messages: dict[str, Any],
        response: dict[str, Any],
        request_sent_at: datetime,
        response_received_at: datetime,
        total_response_time_ms: int,
        response_tokens: int,
        total_tokens_consumed: int,
        time_to_first_token_ms: int | None = None,
        tokens_per_second: float | None = None,
    ) -> LLMCall:
        """Add an LLM call to a game turn."""
        llm_call = LLMCall(
            game_turn_id=game_turn_id,
            model=model,
            messages=messages,
            response=response,
            request_sent_at=request_sent_at,
            response_received_at=response_received_at,
            total_response_time_ms=total_response_time_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            tokens_per_second=tokens_per_second,
            response_tokens=response_tokens,
            total_tokens_consumed=total_tokens_consumed,
        )
        self.session.add(llm_call)
        self.session.flush()
        return llm_call

    def get_turn_llm_calls(self, game_turn_id: int) -> list[LLMCall]:
        """Get all LLM calls for a game turn."""
        stmt = select(LLMCall).where(LLMCall.game_turn_id == game_turn_id)
        return list(self.session.execute(stmt).scalars())

    # ===== AgentTrueSkill Operations =====

    def create_or_update_trueskill(
        self,
        agent_id: int,
        mu: float = 25.0,
        sigma: float = 8.333,
    ) -> AgentTrueSkill:
        """Create or update TrueSkill rating for an agent."""
        trueskill = self.session.get(AgentTrueSkill, agent_id)
        if trueskill:
            trueskill.mu = mu
            trueskill.sigma = sigma
            trueskill.updated_at = datetime.now(timezone.utc)
        else:
            trueskill = AgentTrueSkill(
                agent_id=agent_id,
                mu=mu,
                sigma=sigma,
                updated_at=datetime.now(timezone.utc),
            )
            self.session.add(trueskill)
        self.session.flush()
        return trueskill

    def get_agent_trueskill(self, agent_id: int) -> AgentTrueSkill | None:
        """Get TrueSkill rating for an agent."""
        return self.session.get(AgentTrueSkill, agent_id)

    def get_top_agents_by_trueskill(
        self, limit: int = 10
    ) -> list[tuple[Agent, AgentTrueSkill]]:
        """
        Get top agents by TrueSkill rating (mu - 3*sigma).

        Args:
            limit: Maximum number of agents to return

        Returns:
            List of (Agent, AgentTrueSkill) tuples
        """
        stmt = (
            select(Agent, AgentTrueSkill)
            .join(AgentTrueSkill, Agent.id == AgentTrueSkill.agent_id)
            .order_by((AgentTrueSkill.mu - 3 * AgentTrueSkill.sigma).desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).all())
