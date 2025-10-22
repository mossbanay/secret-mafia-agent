import argparse
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from random import sample

import textarena as ta
import trueskill
from tqdm import tqdm

from mindgames.agents.basic import ThinkingAgent
from mindgames.agents.remembering import ThinkingAndRememberingAgent
from mindgames.db.session import get_session, init_db
from mindgames.db.repository import Repository
from mindgames.db.models import Agent


@dataclass
class ActiveAgentData:
    """Structured data for an active agent in a game."""

    id: int
    model_name: str
    model_litellm_name: str
    kind_name: str
    prompt_content: str


N_PLAYERS = 6


def update_trueskill_ratings(
    repo: Repository, agents: list[Agent], rewards: dict
) -> None:
    """
    Update TrueSkill ratings for all agents based on game outcome.

    Args:
        repo: Repository instance for database access
        agents: List of agents who participated in the game
        rewards: Dict mapping player_index to reward value (+1 for win, -1 for loss)
    """
    # Get or initialize TrueSkill ratings for each agent
    ratings = []
    for player_index, agent in enumerate(agents):
        # Get existing rating or use defaults
        agent_full = repo.get_agent_by_id(agent.id, with_relations=True)
        if agent_full and agent_full.trueskill:
            rating = trueskill.Rating(
                mu=agent_full.trueskill.mu, sigma=agent_full.trueskill.sigma
            )
        else:
            # Default TrueSkill rating: mu=25.0, sigma=8.333
            rating = trueskill.Rating()
        ratings.append(rating)

    # Split players into winners and losers based on rewards
    winners = [ratings[idx] for idx, reward in rewards.items() if reward > 0]
    losers = [ratings[idx] for idx, reward in rewards.items() if reward < 0]

    # Rate the match: winners vs losers
    # rating_groups is a list where each element represents a team/rank
    # Lower index = better rank, so winners come first
    if winners and losers:
        rating_groups = [winners, losers]
        new_ratings_nested = trueskill.rate(rating_groups)

        # Flatten back to list matching original order
        winner_indices = [idx for idx, reward in rewards.items() if reward > 0]
        loser_indices = [idx for idx, reward in rewards.items() if reward < 0]

        new_ratings = [None] * len(agents)

        # Map winners back
        for i, player_idx in enumerate(winner_indices):
            new_ratings[player_idx] = new_ratings_nested[0][i]

        # Map losers back
        for i, player_idx in enumerate(loser_indices):
            new_ratings[player_idx] = new_ratings_nested[1][i]

        # Update database with new ratings
        for player_index, agent in enumerate(agents):
            if new_ratings[player_index] is not None:
                new_rating = new_ratings[player_index]
                repo.create_or_update_trueskill(
                    agent_id=agent.id, mu=new_rating.mu, sigma=new_rating.sigma
                )


def generate_game_hash(game_id: int) -> str:
    """Generate a short hash for game identification."""
    full_hash = hashlib.md5(
        f"game_{game_id}_{datetime.now(timezone.utc)}".encode()
    ).hexdigest()
    return full_hash[:8]


def create_agent_instance(agent: Agent, repo: Repository):
    """
    Create an agent instance from database configuration.

    Args:
        agent: Database Agent object with loaded relationships
        repo: Repository instance for database access

    Returns:
        Agent instance (ThinkingAgent or ThinkingAndRememberingAgent) configured from database
    """
    # Load agent with all relationships
    agent_full = repo.get_agent_by_id(agent.id, with_relations=True)
    if not agent_full:
        raise ValueError(f"Agent {agent.id} not found")

    # Create the appropriate agent type based on kind
    if agent_full.kind.name == "remembering":
        return ThinkingAndRememberingAgent(
            agent_db=agent_full,
            system_prompt=agent_full.prompt.content,
            enable_structured_memory=True,
        )
    else:
        # Default to ThinkingAgent for "thinking" or any other kind
        return ThinkingAgent(
            agent_db=agent_full, system_prompt=agent_full.prompt.content
        )


def run_game(
    db_url: str | None = None,
    verbose: bool = True,
) -> tuple[bool, int, str | None]:
    """
    Run a single game, logging the results to the database.

    Args:
        db_url: Database URL (if None, uses DATABASE_URL env var)
        verbose: Whether to print detailed progress messages

    Returns:
        Tuple of (success: bool, game_id: int, winning_team: Optional[str])
    """
    if verbose:
        print("\nStarting game ...")

    # Create game record with short transaction
    with get_session(db_url) as session:
        repo = Repository(session)
        game = repo.create_game(started_at=datetime.now(timezone.utc))
        game_id = game.id

    game_hash = generate_game_hash(game_id)
    if verbose:
        print(f"📝 Game ID: {game_id} (hash: {game_hash})")

    try:
        # Get active agents for selection
        with get_session(db_url) as session:
            repo = Repository(session)

            # Get all active agents
            active_agents = repo.get_active_agents()

            if len(active_agents) < N_PLAYERS:
                raise ValueError(
                    f"Insufficient active agents for game. Need at least {N_PLAYERS} unique agents, "
                    f"but only {len(active_agents)} are marked as active. "
                    f"Please activate more agents using: UPDATE agents SET active = 1 WHERE id IN (...);"
                )

            # Randomly select N_PLAYERS unique agents
            selected_agents = sample(active_agents, N_PLAYERS)

            # Extract agent IDs
            all_agent_ids = [agent.id for agent in selected_agents]

            # Create agent configs with positions
            agent_configs = [(i, agent_id) for i, agent_id in enumerate(all_agent_ids)]

        # Add all players to game and collect agent data in single transaction
        active_agents_data: list[ActiveAgentData] = []
        with get_session(db_url) as session:
            repo = Repository(session)
            for player_index, agent_id in agent_configs:
                # Add player to game
                _ = repo.add_game_player(
                    game_id=game_id, agent_id=agent_id, player_index=player_index
                )
                # Load agent for game use
                if agent := repo.get_agent_by_id(agent_id, with_relations=True):
                    active_agents_data.append(
                        ActiveAgentData(
                            id=agent.id,
                            model_name=agent.model.name,
                            model_litellm_name=agent.model.litellm_model_name,
                            kind_name=agent.kind.name,
                            prompt_content=agent.prompt.content,
                        )
                    )

        if verbose:
            print(f"👥 Selected {len(active_agents_data)} agents for the game")

        # Create agent instances - we need to reload agents in session for create_agent_instance
        agent_instances = []
        for agent_data in active_agents_data:
            with get_session(db_url) as session:
                repo = Repository(session)
                # Reload the agent with all relationships in this session
                agent = repo.get_agent_by_id(agent_data.id, with_relations=True)
                # Use create_agent_instance to get the proper agent type based on kind
                agent_instance = create_agent_instance(agent, repo)
            agent_instances.append(agent_instance)

        # Initialize the game environment
        env = ta.make(env_id="SecretMafia-v0")
        env.reset(num_players=N_PLAYERS)

        done = False
        turn_count = 0

        while not done:
            turn_count += 1
            if verbose:
                print(f"\n🤖 Turn {turn_count}")

            # Get observation
            player_id, observation = env.get_observation()
            agent_data = active_agents_data[player_id]
            if verbose:
                print(
                    f"Player {player_id}: Agent #{agent_data.id} (Model: {agent_data.model_name}, Kind: {agent_data.kind_name})"
                )

            # Record turn start time
            turn_start = time.time()
            action_start = datetime.now(timezone.utc)

            # Get action from agent
            action, llm_metadata = agent_instances[player_id](
                observation, return_metadata=True
            )

            # Calculate response time
            response_time_ms = int((time.time() - turn_start) * 1000)
            if verbose:
                print(
                    f"Action: {action[:100]}..."
                    if len(action) > 100
                    else f"Action: {action}"
                )

            # Step the environment
            done, step_info = env.step(action=action)

            # Log the turn to database with short transaction
            with get_session(db_url) as session:
                repo = Repository(session)
                turn = repo.add_game_turn(
                    game_id=game_id,
                    turn_number=turn_count,
                    player_index=player_id,
                    observation={"text": observation},  # Store as dict
                    action=action,
                    response_time_ms=response_time_ms,
                    step_info=step_info,
                    tokens_used=llm_metadata.get("tokens_used"),
                    timestamp=action_start,
                )
                turn_id = turn.id

                # Log all LLM calls if metadata available
                # Support both new format (llm_calls list) and old format (single llm_call)
                if llm_metadata and "llm_calls" in llm_metadata:
                    # New format: multiple calls
                    for llm_call in llm_metadata["llm_calls"]:
                        repo.add_llm_call(
                            game_turn_id=turn_id,
                            model=agent_data.model_litellm_name,
                            messages=llm_call.get("messages", {}),
                            response=llm_call.get("response", {}),
                            request_sent_at=llm_call.get(
                                "request_sent_at", action_start
                            ),
                            response_received_at=llm_call.get(
                                "response_received_at", datetime.now(timezone.utc)
                            ),
                            total_response_time_ms=llm_call.get(
                                "total_response_time_ms", response_time_ms
                            ),
                            response_tokens=llm_call.get("response_tokens", 0),
                            total_tokens_consumed=llm_call.get(
                                "total_tokens_consumed", 0
                            ),
                            time_to_first_token_ms=llm_call.get(
                                "time_to_first_token_ms"
                            ),
                            tokens_per_second=llm_call.get("tokens_per_second"),
                        )
                elif llm_metadata and "llm_call" in llm_metadata:
                    # Old format: single call (backward compatibility)
                    llm_call = llm_metadata["llm_call"]
                    repo.add_llm_call(
                        game_turn_id=turn_id,
                        model=agent_data.model_litellm_name,
                        messages=llm_call.get("messages", {}),
                        response=llm_call.get("response", {}),
                        request_sent_at=llm_call.get("request_sent_at", action_start),
                        response_received_at=llm_call.get(
                            "response_received_at", datetime.now(timezone.utc)
                        ),
                        total_response_time_ms=llm_call.get(
                            "total_response_time_ms", response_time_ms
                        ),
                        response_tokens=llm_call.get("response_tokens", 0),
                        total_tokens_consumed=llm_call.get("total_tokens_consumed", 0),
                        time_to_first_token_ms=llm_call.get("time_to_first_token_ms"),
                        tokens_per_second=llm_call.get("tokens_per_second"),
                    )

        # Get final game results
        rewards, game_info = env.close()

        # Update player rewards and roles with short transaction
        with get_session(db_url) as session:
            repo = Repository(session)
            # rewards is a dict: {0: -1, 1: 1, 2: 1, ...}
            for player_index in range(N_PLAYERS):
                # Extract role from game_info for this player
                # game_info structure: {0: {role: "Detective", ...}, 1: {role: "Mafia", ...}, ...}
                role = None
                if game_info and player_index in game_info:
                    role = game_info[player_index].get("role")

                # Get reward for this player
                reward = rewards.get(player_index, 0.0)

                repo.update_game_player(
                    game_id=game_id,
                    player_index=player_index,
                    role=role,
                    reward=float(reward),
                )

        # Determine winning team from game_info
        # The textarena game_info structure has per-player data with a "reason" field
        # that contains the win condition (e.g., "Mafia wins!" or "Village wins!")
        winning_team = None
        if game_info and len(game_info) > 0:
            # Get the reason from first player (all players have same reason)
            first_player_key = list(game_info.keys())[0]
            reason = game_info[first_player_key].get("reason", "")

            if "Mafia win" in reason:
                winning_team = "Mafia"
            elif ("Village win" in reason or "Villager" in reason) and "win" in reason:
                winning_team = "Villager"

        # Mark game as completed with short transaction
        with get_session(db_url) as session:
            repo = Repository(session)
            _ = repo.finish_game(
                game_id=game_id,
                status="completed",
                winning_team=winning_team,
                game_info=game_info,
            )

        # Update TrueSkill ratings with short transaction
        if verbose:
            print("\n📈 Updating TrueSkill ratings...")
        with get_session(db_url) as session:
            repo = Repository(session)
            # Load agents fresh for TrueSkill update
            active_agents = []
            for agent_data in active_agents_data:
                agent = repo.get_agent_by_id(agent_data.id, with_relations=True)
                active_agents.append(agent)
            update_trueskill_ratings(repo, active_agents, rewards)

        if verbose:
            print(f"\n✅ Game {game_id} completed!")
            print(f"🏆 Winning team: {winning_team}")
            print(f"📊 Total turns: {turn_count}")
            print(f"🎯 Rewards: {rewards}")

        return (True, game_id, winning_team)

    except Exception as e:
        # Mark game as failed with short transaction
        with get_session(db_url) as session:
            repo = Repository(session)
            repo.finish_game(game_id=game_id, status="failed", environment_crashed=True)
        if verbose:
            print(f"❌ Game {game_id} failed: {e}")
        return (False, game_id, None)


def _run_game_worker(
    args_tuple: tuple[str | None],
) -> tuple[bool, int, str | None]:
    """
    Worker function for running a game in a separate process.

    Args:
        args_tuple: Tuple of (game_number, n_games_total, db_url)

    Returns:
        Tuple of (success: bool, game_id: int, winning_team: Optional[str])
    """
    (db_url,) = args_tuple
    try:
        return run_game(db_url, verbose=False)
    except Exception as e:
        print(f"❌ Error in game: {e}")
        return (False, -1, None)


def run_arena() -> None:
    """Run the mindgames arena with database integration."""
    parser = argparse.ArgumentParser(description="Run mindgames arena")
    parser.add_argument(
        "-n",
        "--n-games",
        type=int,
        default=1,
        help="Number of games to run (default: 1)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Number of games to run in parallel (default: 1 for sequential)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database schema before running",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Database URL (defaults to DATABASE_URL env var or sqlite:///mindgames.db)",
    )
    args = parser.parse_args()

    # Get database URL from args or environment
    db_url = args.db_url or os.environ.get("DATABASE_URL")

    # Initialize database if requested
    if args.init_db:
        print("🔧 Initializing database schema...")
        init_db(db_url)

    if args.parallel > 1:
        print(
            f"🚀 Running {args.n_games} game(s) with {args.parallel} parallel workers"
        )

        # Run games in parallel using ProcessPoolExecutor
        successful_games = 0
        failed_games = 0
        results = []

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all games
            future_to_game = {
                executor.submit(_run_game_worker, (db_url,)): game_num
                for game_num in range(1, args.n_games + 1)
            }

            # Use tqdm to show progress
            with tqdm(total=args.n_games, desc="Running games", unit="game") as pbar:
                for future in as_completed(future_to_game):
                    game_num = future_to_game[future]
                    try:
                        success, game_id, winning_team = future.result()
                        if success:
                            successful_games += 1
                            results.append((game_id, winning_team))
                        else:
                            failed_games += 1
                        pbar.update(1)
                    except Exception as e:
                        failed_games += 1
                        print(f"\n❌ Game {game_num} raised exception: {e}")
                        pbar.update(1)

        print(f"\n🏁 Completed running {args.n_games} game(s)")
        print(f"✅ Successful: {successful_games}")
        print(f"❌ Failed: {failed_games}")

    else:
        print(f"🚀 Running {args.n_games} game(s) sequentially")

        # Run games sequentially
        successful_games = 0
        failed_games = 0

        for game_number in range(1, args.n_games + 1):
            try:
                success, game_id, winning_team = run_game(db_url, verbose=True)
                if success:
                    successful_games += 1
                else:
                    failed_games += 1
            except Exception as e:
                print(f"❌ Error in game {game_number}: {e}")
                failed_games += 1
                continue

        print(f"\n🏁 Completed running {args.n_games} game(s)")
        print(f"✅ Successful: {successful_games}")
        print(f"❌ Failed: {failed_games}")


def main() -> None:
    try:
        run_arena()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")


if __name__ == "__main__":
    main()
