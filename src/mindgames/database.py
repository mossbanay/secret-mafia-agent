"""
Database module for mindgames agent system.

Implements SQLite schema for storing agents, strategies, games, and performance data
as specified in SPEC.md section 3.1.
"""

import sqlite3
import uuid
import json
import hashlib
from pathlib import Path
from typing import Any


class MindGamesDatabase:
    """SQLite database interface for mindgames agent system."""

    def __init__(self, db_path: str = "mindgames.db"):
        """Initialize database connection and create schema if needed."""
        self.db_path = Path(db_path)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable dict-like access
        self._enable_foreign_keys()
        self._create_schema()

    def _enable_foreign_keys(self):
        """Enable foreign key constraints."""
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.commit()

    def _create_schema(self):
        """Create all database tables if they don't exist."""
        schema_sql = """
        -- Core entity tables
        CREATE TABLE IF NOT EXISTS agent_types (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            is_disabled BOOLEAN NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS llm_models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kwargs TEXT, -- JSON string for model parameters
            is_disabled BOOLEAN NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS strategy_prompts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_disabled BOOLEAN NOT NULL DEFAULT 0
        );

        -- Composite agent definitions
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            type_id TEXT NOT NULL REFERENCES agent_types(id),
            model_id TEXT NOT NULL REFERENCES llm_models(id),
            prompt_id TEXT NOT NULL REFERENCES strategy_prompts(id),
            agent_hash TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Game execution tracking
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            environment TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds REAL,
            winner_team TEXT,
            final_rewards TEXT, -- JSON string
            game_state_final TEXT -- JSON string
        );

        CREATE TABLE IF NOT EXISTS game_participants (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL REFERENCES games(id),
            agent_id TEXT NOT NULL REFERENCES agents(id),
            player_index INTEGER NOT NULL,
            assigned_role TEXT,
            final_reward REAL
        );

        CREATE TABLE IF NOT EXISTS game_actions (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL REFERENCES games(id),
            agent_id TEXT NOT NULL REFERENCES agents(id),
            turn_number INTEGER NOT NULL,
            phase TEXT,
            observation TEXT,
            raw_response TEXT,
            parsed_action TEXT,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            processing_time_ms INTEGER
        );

        -- Indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_agents_hash ON agents(agent_hash);
        CREATE INDEX IF NOT EXISTS idx_game_participants_game ON game_participants(game_id);
        CREATE INDEX IF NOT EXISTS idx_game_participants_agent ON game_participants(agent_id);
        CREATE INDEX IF NOT EXISTS idx_game_actions_game ON game_actions(game_id);
        CREATE INDEX IF NOT EXISTS idx_game_actions_agent ON game_actions(agent_id);
        CREATE INDEX IF NOT EXISTS idx_game_actions_turn ON game_actions(game_id, turn_number);
        """

        self.connection.executescript(schema_sql)
        self.connection.commit()

    def generate_uuid(self) -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())

    def generate_agent_hash(
        self, agent_type: str, model_name: str, prompt_content: str
    ) -> str:
        """Generate deterministic hash for agent combination."""
        content = f"{agent_type}:{model_name}:{prompt_content}"
        return hashlib.sha256(content.encode()).hexdigest()

    def current_timestamp(self):
        """Get current timestamp (SQLite will handle this automatically)."""
        return None  # Let SQLite handle with CURRENT_TIMESTAMP

    # Agent type methods
    def create_agent_type(self, name: str) -> str:
        """Create a new agent type and return its ID."""
        agent_type_id = self.generate_uuid()
        self.connection.execute(
            "INSERT INTO agent_types (id, name) VALUES (?, ?)", (agent_type_id, name)
        )
        self.connection.commit()
        return agent_type_id

    def get_agent_type_by_name(self, name: str) -> dict[str, Any] | None:
        """Get agent type by name."""
        cursor = self.connection.execute(
            "SELECT * FROM agent_types WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_or_create_agent_type(self, name: str) -> str:
        """Get existing agent type ID or create new one."""
        existing = self.get_agent_type_by_name(name)
        if existing:
            return existing["id"]
        return self.create_agent_type(name)

    def disable_agent_type(self, name: str) -> bool:
        """Disable an agent type by name. Returns True if successful."""
        cursor = self.connection.execute(
            "UPDATE agent_types SET is_disabled = 1 WHERE name = ?", (name,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def enable_agent_type(self, name: str) -> bool:
        """Enable an agent type by name. Returns True if successful."""
        cursor = self.connection.execute(
            "UPDATE agent_types SET is_disabled = 0 WHERE name = ?", (name,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    # LLM model methods
    def create_llm_model(self, name: str, kwargs: dict[str, Any]) -> str:
        """Create a new LLM model and return its ID."""
        model_id = self.generate_uuid()
        kwargs_json = json.dumps(kwargs) if kwargs else "{}"
        self.connection.execute(
            "INSERT INTO llm_models (id, name, kwargs) VALUES (?, ?, ?)",
            (model_id, name, kwargs_json),
        )
        self.connection.commit()
        return model_id

    def get_llm_model_by_name(self, name: str) -> dict[str, Any] | None:
        """Get LLM model by name."""
        cursor = self.connection.execute(
            "SELECT * FROM llm_models WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["kwargs"] = json.loads(result["kwargs"]) if result["kwargs"] else {}
            return result
        return None

    def get_or_create_llm_model(
        self, name: str, kwargs: dict[str, Any] | None = None
    ) -> str:
        """Get existing LLM model ID or create new one."""
        existing = self.get_llm_model_by_name(name)
        if existing:
            return existing["id"]
        return self.create_llm_model(name, kwargs or {})

    def disable_llm_model(self, name: str) -> bool:
        """Disable an LLM model by name. Returns True if successful."""
        cursor = self.connection.execute(
            "UPDATE llm_models SET is_disabled = 1 WHERE name = ?", (name,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def enable_llm_model(self, name: str) -> bool:
        """Enable an LLM model by name. Returns True if successful."""
        cursor = self.connection.execute(
            "UPDATE llm_models SET is_disabled = 0 WHERE name = ?", (name,)
        )
        self.connection.commit()
        return cursor.rowcount > 0

    # Strategy prompt methods
    def create_strategy_prompt(self, name: str, content: str, version: int = 1) -> str:
        """Create a new strategy prompt and return its ID."""
        prompt_id = self.generate_uuid()
        self.connection.execute(
            "INSERT INTO strategy_prompts (id, name, content, version) VALUES (?, ?, ?, ?)",
            (prompt_id, name, content, version),
        )
        self.connection.commit()
        return prompt_id

    def get_strategy_prompt_by_name(
        self, name: str, version: int | None = None
    ) -> dict[str, Any] | None:
        """Get strategy prompt by name and optional version (latest if not specified)."""
        if version:
            cursor = self.connection.execute(
                "SELECT * FROM strategy_prompts WHERE name = ? AND version = ?",
                (name, version),
            )
        else:
            cursor = self.connection.execute(
                "SELECT * FROM strategy_prompts WHERE name = ? ORDER BY version DESC LIMIT 1",
                (name,),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def disable_strategy_prompt(self, name: str, version: int | None = None) -> bool:
        """Disable a strategy prompt by name and optional version. Returns True if successful."""
        if version:
            cursor = self.connection.execute(
                "UPDATE strategy_prompts SET is_disabled = 1 WHERE name = ? AND version = ?",
                (name, version),
            )
        else:
            cursor = self.connection.execute(
                "UPDATE strategy_prompts SET is_disabled = 1 WHERE name = ?", (name,)
            )
        self.connection.commit()
        return cursor.rowcount > 0

    def enable_strategy_prompt(self, name: str, version: int | None = None) -> bool:
        """Enable a strategy prompt by name and optional version. Returns True if successful."""
        if version:
            cursor = self.connection.execute(
                "UPDATE strategy_prompts SET is_disabled = 0 WHERE name = ? AND version = ?",
                (name, version),
            )
        else:
            cursor = self.connection.execute(
                "UPDATE strategy_prompts SET is_disabled = 0 WHERE name = ?", (name,)
            )
        self.connection.commit()
        return cursor.rowcount > 0

    # Agent methods
    def create_agent(
        self,
        agent_type_name: str,
        model_name: str,
        prompt_name: str,
        model_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Create a new agent and return its ID."""
        # Get or create component IDs
        type_id = self.get_or_create_agent_type(agent_type_name)
        model_id = self.get_or_create_llm_model(model_name, model_kwargs)

        # Get prompt (latest version)
        prompt = self.get_strategy_prompt_by_name(prompt_name)
        if not prompt:
            raise ValueError(f"Strategy prompt '{prompt_name}' not found")

        # Generate agent hash and ID
        agent_hash = self.generate_agent_hash(
            agent_type_name, model_name, prompt["content"]
        )
        agent_id = self.generate_uuid()

        self.connection.execute(
            "INSERT INTO agents (id, type_id, model_id, prompt_id, agent_hash) VALUES (?, ?, ?, ?, ?)",
            (agent_id, type_id, model_id, prompt["id"], agent_hash),
        )
        self.connection.commit()
        return agent_id

    def get_agent_by_hash(self, agent_hash: str) -> dict[str, Any] | None:
        """Get agent by hash with full details."""
        cursor = self.connection.execute(
            """
            SELECT a.*, at.name as agent_type, lm.name as model_name, lm.kwargs as model_kwargs,
                   sp.name as prompt_name, sp.content as prompt_content, sp.version as prompt_version
            FROM agents a
            JOIN agent_types at ON a.type_id = at.id
            JOIN llm_models lm ON a.model_id = lm.id
            JOIN strategy_prompts sp ON a.prompt_id = sp.id
            WHERE a.agent_hash = ?
        """,
            (agent_hash,),
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["model_kwargs"] = (
                json.loads(result["model_kwargs"]) if result["model_kwargs"] else {}
            )
            return result
        return None

    def get_random_agents(self, count: int = 6) -> list[dict[str, Any]]:
        """Get random agents for game simulation by randomly composing agent types, models, and prompts."""
        results = []

        for _ in range(count):
            # Randomly select one from each category, excluding disabled entries
            agent_type_cursor = self.connection.execute(
                "SELECT * FROM agent_types WHERE is_disabled = 0 ORDER BY RANDOM() LIMIT 1"
            )
            agent_type_row = agent_type_cursor.fetchone()
            if not agent_type_row:
                raise ValueError("No enabled agent types available")
            agent_type = dict(agent_type_row)

            model_cursor = self.connection.execute(
                "SELECT * FROM llm_models WHERE is_disabled = 0 ORDER BY RANDOM() LIMIT 1"
            )
            model_row = model_cursor.fetchone()
            if not model_row:
                raise ValueError("No enabled LLM models available")
            model = dict(model_row)

            prompt_cursor = self.connection.execute(
                "SELECT * FROM strategy_prompts WHERE is_disabled = 0 ORDER BY RANDOM() LIMIT 1"
            )
            prompt_row = prompt_cursor.fetchone()
            if not prompt_row:
                raise ValueError("No enabled strategy prompts available")
            prompt = dict(prompt_row)

            # Generate agent hash for this combination
            agent_hash = self.generate_agent_hash(
                agent_type["name"], model["name"], prompt["content"]
            )

            # Check if agent already exists
            existing_agent = self.get_agent_by_hash(agent_hash)

            if existing_agent:
                # Use existing agent
                results.append(existing_agent)
            else:
                # Create new agent
                agent_id = self.generate_uuid()
                self.connection.execute(
                    "INSERT INTO agents (id, type_id, model_id, prompt_id, agent_hash) VALUES (?, ?, ?, ?, ?)",
                    (agent_id, agent_type["id"], model["id"], prompt["id"], agent_hash),
                )
                self.connection.commit()

                # Build result dict with all the info
                result = {
                    "id": agent_id,
                    "type_id": agent_type["id"],
                    "model_id": model["id"],
                    "prompt_id": prompt["id"],
                    "agent_hash": agent_hash,
                    "agent_type": agent_type["name"],
                    "model_name": model["name"],
                    "model_kwargs": json.loads(model["kwargs"])
                    if model["kwargs"]
                    else {},
                    "prompt_name": prompt["name"],
                    "prompt_content": prompt["content"],
                    "prompt_version": prompt["version"],
                }
                results.append(result)

        return results

    # Game tracking methods
    def create_game(self, environment: str = "SecretMafia-v0") -> str:
        """Create a new game record and return its ID."""
        game_id = self.generate_uuid()
        self.connection.execute(
            "INSERT INTO games (id, environment) VALUES (?, ?)", (game_id, environment)
        )
        self.connection.commit()
        return game_id

    def finish_game(
        self,
        game_id: str,
        winner_team: str,
        final_rewards: dict[str, Any],
        game_state_final: dict[str, Any],
    ):
        """Update game with final results."""
        # Calculate duration using SQLite datetime functions
        self.connection.execute(
            """
            UPDATE games
            SET end_time = CURRENT_TIMESTAMP,
                duration_seconds = (julianday(CURRENT_TIMESTAMP) - julianday(start_time)) * 86400.0,
                winner_team = ?,
                final_rewards = ?,
                game_state_final = ?
            WHERE id = ?
        """,
            (
                winner_team,
                json.dumps(final_rewards),
                json.dumps(game_state_final),
                game_id,
            ),
        )
        self.connection.commit()

    def add_game_participant(
        self,
        game_id: str,
        agent_id: str,
        player_index: int,
        assigned_role: str | None = None,
        final_reward: float | None = None,
    ) -> str:
        """Add a participant to a game."""
        participant_id = self.generate_uuid()
        self.connection.execute(
            """
            INSERT INTO game_participants (id, game_id, agent_id, player_index, assigned_role, final_reward)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                participant_id,
                game_id,
                agent_id,
                player_index,
                assigned_role,
                final_reward,
            ),
        )
        self.connection.commit()
        return participant_id

    def update_game_participant(
        self,
        game_id: str,
        player_index: int,
        assigned_role: str | None = None,
        final_reward: float | None = None,
    ) -> bool:
        """Update a game participant's role and final reward."""
        cursor = self.connection.execute(
            """
            UPDATE game_participants
            SET assigned_role = ?, final_reward = ?
            WHERE game_id = ? AND player_index = ?
            """,
            (assigned_role, final_reward, game_id, player_index),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def add_game_action(
        self,
        game_id: str,
        agent_id: str,
        turn_number: int,
        phase: str,
        observation: str,
        raw_response: str,
        parsed_action: str,
        processing_time_ms: int | None = None,
    ) -> str:
        """Add an action to the game log."""
        action_id = self.generate_uuid()
        self.connection.execute(
            """
            INSERT INTO game_actions (id, game_id, agent_id, turn_number, phase, observation,
                                    raw_response, parsed_action, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                action_id,
                game_id,
                agent_id,
                turn_number,
                phase,
                observation,
                raw_response,
                parsed_action,
                processing_time_ms,
            ),
        )
        self.connection.commit()
        return action_id

    def close(self):
        """Close database connection."""
        self.connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def initialize_database(db_path: str = "mindgames.db") -> MindGamesDatabase:
    """Initialize database with default data."""
    db = MindGamesDatabase(db_path)

    # Create default agent type
    db.get_or_create_agent_type("basic")

    # Create default model
    db.get_or_create_llm_model(
        "local/qwen3-30b-a3b", {"temperature": 0.7, "max_tokens": 512}
    )

    # Create default strategy prompt
    if not db.get_strategy_prompt_by_name("basic_mafia"):
        db.create_strategy_prompt(
            name="basic_mafia",
            content="You are playing Secret Mafia. Play strategically according to your role.\nBe brief about your reasoning, you only have 100 words to respond.",
            version=1,
        )

    return db


if __name__ == "__main__":
    # Test database creation
    with initialize_database("test_mindgames.db") as db:
        print("Database schema created successfully!")

        # Test agent creation
        agent_id = db.create_agent(
            agent_type_name="basic",
            model_name="gpt-4o-mini",
            prompt_name="basic_mafia",
            model_kwargs={"temperature": 0.7, "max_tokens": 512},
        )
        print(f"Created test agent: {agent_id}")

        # Test random agent selection
        agents = db.get_random_agents(3)
        print(f"Random agents: {len(agents)} found")
