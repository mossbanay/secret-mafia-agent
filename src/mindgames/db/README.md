# Database Module

SQLAlchemy-based database layer for the mindgames arena system.

## Overview

This module provides a complete database implementation based on the schema defined in `proposal.md`. It uses SQLAlchemy ORM for type-safe database operations and includes:

- **Models**: SQLAlchemy ORM models for all database tables
- **Session Management**: Context managers for database sessions
- **Repository Pattern**: High-level API for common database operations

## Quick Start

```python
from mindgames.db import init_db, get_session, Repository

# Initialize database (creates tables)
init_db("mindgames.db")

# Use session context manager
with get_session("mindgames.db") as session:
    repo = Repository(session)

    # Create a model
    model = repo.create_model(
        name="cerebras-llama3.1-8b",
        litellm_model_name="cerebras/llama3.1-8b",
        active=True
    )

    # Query data
    active_models = repo.get_active_models()
```

## Module Structure

```
mindgames/db/
├── __init__.py      # Package exports
├── models.py        # SQLAlchemy ORM models
├── session.py       # Session management and initialization
├── repository.py    # Data access layer (Repository pattern)
├── example.py       # Example usage
└── README.md        # This file
```

## Core Components

### Models (`models.py`)

SQLAlchemy ORM models representing database tables:

- `Model`: LLM model configurations
- `Kind`: Agent kind/type configurations
- `Prompt`: Prompt templates (immutable with forking)
- `Agent`: Agent configurations (kind + model + prompt)
- `Game`: Game metadata and outcomes
- `GamePlayer`: Links agents to games with roles and rewards
- `GameTurn`: Turn-by-turn game logging
- `LLMCall`: Individual LLM API call tracking
- `AgentTrueSkill`: TrueSkill ratings for agents

### Session Management (`session.py`)

- `init_db(db_path, drop_all)`: Initialize database schema
- `get_session(db_path)`: Context manager for database sessions
- `get_engine(db_path)`: Get SQLAlchemy engine instance

### Repository (`repository.py`)

High-level API for database operations:

#### Model Operations
- `create_model(name, litellm_model_name, active)`
- `get_model_by_id(model_id)`
- `get_model_by_name(name)`
- `get_active_models()`
- `set_model_active(model_id, active)`

#### Kind Operations
- `create_kind(name, active)`
- `get_kind_by_id(kind_id)`
- `get_kind_by_name(name)`
- `get_active_kinds()`
- `set_kind_active(kind_id, active)`

#### Prompt Operations
- `create_prompt(name, content, forked_from, active)`
- `get_prompt_by_id(prompt_id)`
- `get_prompt_by_name(name)`
- `get_active_prompts()`
- `set_prompt_active(prompt_id, active)`

#### Agent Operations
- `create_agent(kind_id, model_id, prompt_id, alias, class_)`
- `get_agent_by_id(agent_id, with_relations)`
- `get_agent_by_config(kind_id, model_id, prompt_id)`
- `get_or_create_agent(...)`
- `get_random_agents(count)`

#### Game Operations
- `create_game(started_at, status)`
- `get_game_by_id(game_id, with_relations)`
- `finish_game(game_id, status, winning_team, environment_crashed, game_info)`
- `get_games(status, winning_team, limit, offset)`

#### GamePlayer Operations
- `add_game_player(game_id, agent_id, player_index, role, reward)`
- `update_game_player(game_id, player_index, role, reward)`

#### GameTurn Operations
- `add_game_turn(game_id, turn_number, player_index, observation, action, ...)`
- `get_game_turns(game_id)`

#### LLMCall Operations
- `add_llm_call(game_turn_id, model, messages, response, ...)`
- `get_turn_llm_calls(game_turn_id)`

#### TrueSkill Operations
- `create_or_update_trueskill(agent_id, mu, sigma)`
- `get_agent_trueskill(agent_id)`
- `get_top_agents_by_trueskill(limit)`

## Usage Examples

### Initialize Database

```python
from mindgames.db import init_db

# Create database with schema
init_db("mindgames.db")

# Drop and recreate all tables (use with caution!)
init_db("mindgames.db", drop_all=True)
```

### Create Configuration Data

```python
from mindgames.db import get_session, Repository

with get_session() as session:
    repo = Repository(session)

    # Create a model
    model = repo.create_model(
        name="cerebras-llama3.1-8b",
        litellm_model_name="cerebras/llama3.1-8b"
    )

    # Create a kind
    kind = repo.create_kind(name="standard")

    # Create a prompt
    prompt = repo.create_prompt(
        name="v1",
        content="You are playing Secret Mafia..."
    )

    # Create an agent
    agent = repo.create_agent(
        kind_id=kind.id,
        model_id=model.id,
        prompt_id=prompt.id,
        alias="Agent Alpha"
    )
```

### Run a Game

```python
from datetime import datetime
from mindgames.db import get_session, Repository

with get_session() as session:
    repo = Repository(session)

    # Create game
    game = repo.create_game()

    # Add players
    agents = repo.get_random_agents(count=6)
    for i, agent in enumerate(agents):
        repo.add_game_player(
            game_id=game.id,
            agent_id=agent.id,
            player_index=i
        )

    # Log turns
    for turn_num in range(1, 10):
        for player_idx in range(6):
            turn = repo.add_game_turn(
                game_id=game.id,
                turn_number=turn_num,
                player_index=player_idx,
                observation={"state": "..."},
                action="vote player 3",
                response_time_ms=1500,
                tokens_used=150
            )

            # Track LLM calls
            repo.add_llm_call(
                game_turn_id=turn.id,
                model="cerebras/llama3.1-8b",
                messages={...},
                response={...},
                request_sent_at=datetime.utcnow(),
                response_received_at=datetime.utcnow(),
                total_response_time_ms=1500,
                response_tokens=20,
                total_tokens_consumed=200
            )

    # Finish game
    repo.finish_game(
        game_id=game.id,
        status="completed",
        winning_team="Villager"
    )
```

### Query Data

```python
from mindgames.db import get_session, Repository

with get_session() as session:
    repo = Repository(session)

    # Get active configurations
    models = repo.get_active_models()
    kinds = repo.get_active_kinds()
    prompts = repo.get_active_prompts()

    # Get completed games
    games = repo.get_games(status="completed", limit=10)

    # Get game with full details
    game = repo.get_game_by_id(game_id, with_relations=True)
    for player in game.players:
        print(f"Player {player.player_index}: {player.agent.alias}")

    # Get top agents
    top_agents = repo.get_top_agents_by_trueskill(limit=10)
    for agent, trueskill in top_agents:
        rating = trueskill.mu - 3 * trueskill.sigma
        print(f"{agent.alias}: {rating:.2f}")
```

## Database Schema

The schema is defined in `proposal.md` and includes:

### Configuration Tables
- **models**: LLM model configurations with LiteLLM names
- **kinds**: Agent types/kinds
- **prompts**: Immutable prompt templates with forking support

### Agent Tables
- **agents**: Unique agent configurations (kind + model + prompt)
- **agent_trueskill**: TrueSkill ratings per agent

### Game Tables
- **games**: Game metadata and outcomes
- **game_players**: Links agents to games (6 players per game)
- **game_turns**: Turn-by-turn logging
- **llm_calls**: Individual LLM API call tracking

## Key Features

### Type Safety
All models use SQLAlchemy 2.0+ typed mappings for full type checking support.

### Data Integrity
- Foreign key constraints with appropriate ON DELETE behavior
- CHECK constraints for enums and value ranges
- Unique constraints for game players

### Performance
- Indexes on frequently queried columns
- Eager loading support for relationships
- SQLite WAL mode for concurrent access

### Immutability
- Prompts are immutable with forking support
- Game history is preserved (CASCADE delete only on game removal)

## Migration from Old Database

The old `database.py` used a different schema. To migrate:

1. Export data from old database
2. Transform to new schema format
3. Import using Repository methods

See `example.py` for reference implementation.

## Running the Example

```bash
uv run python -m mindgames.db.example
```

This will create `example_mindgames.db` and demonstrate all major operations.
