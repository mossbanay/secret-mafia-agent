"""
Example usage of the database module.

This script demonstrates how to:
1. Initialize the database
2. Create models, kinds, and prompts
3. Create agents
4. Create and track games
5. Query data using the repository
"""

from datetime import datetime
from mindgames.db import init_db, get_session, Repository


def example_usage():
    """Demonstrate database usage."""

    # Initialize database (creates tables if they don't exist)
    print("Initializing database...")
    init_db("example_mindgames.db")

    # Use session context manager
    with get_session("example_mindgames.db") as session:
        repo = Repository(session)

        # ===== Step 1: Create configuration data =====
        print("\n1. Creating models, kinds, and prompts...")

        # Create models
        cerebras_model = repo.create_model(
            name="cerebras-llama3.1-8b",
            litellm_model_name="cerebras/llama3.1-8b",
            active=True,
        )

        gpt4_model = repo.create_model(
            name="gpt-4o",
            litellm_model_name="gpt-4o",
            active=True,
        )

        # Create kinds
        standard_kind = repo.create_kind(name="standard", active=True)
        advanced_kind = repo.create_kind(name="advanced", active=True)

        # Create prompts
        prompt_v1 = repo.create_prompt(
            name="v1",
            content="You are playing Secret Mafia. Play strategically according to your role.",
            active=True,
        )

        prompt_v2 = repo.create_prompt(
            name="v2",
            content="You are playing Secret Mafia. Analyze the game state carefully and make strategic decisions based on your role.",
            forked_from=prompt_v1.id,
            active=True,
        )

        print(f"  Created {len(repo.get_active_models())} models")
        print(f"  Created {len(repo.get_active_kinds())} kinds")
        print(f"  Created {len(repo.get_active_prompts())} prompts")

        # ===== Step 2: Create agents =====
        print("\n2. Creating agents...")

        agent1 = repo.create_agent(
            kind_id=standard_kind.id,
            model_id=cerebras_model.id,
            prompt_id=prompt_v1.id,
            alias="Agent Alpha",
            class_="open",
        )

        agent2 = repo.create_agent(
            kind_id=advanced_kind.id,
            model_id=gpt4_model.id,
            prompt_id=prompt_v2.id,
            alias="Agent Beta",
            class_="open",
        )

        print(f"  Created agent {agent1.id}: {agent1.alias}")
        print(f"  Created agent {agent2.id}: {agent2.alias}")

        # ===== Step 3: Create a game =====
        print("\n3. Creating a game...")

        game = repo.create_game(started_at=datetime.utcnow())
        print(f"  Created game {game.id}")

        # Add players to the game
        for i, agent in enumerate([agent1, agent2]):
            repo.add_game_player(
                game_id=game.id,
                agent_id=agent.id,
                player_index=i,
                role=None,  # Will be set later
            )

        print(f"  Added {len(game.players)} players")

        # ===== Step 4: Simulate game turns =====
        print("\n4. Simulating game turns...")

        for turn_num in range(1, 4):
            for player_idx in range(2):
                turn = repo.add_game_turn(
                    game_id=game.id,
                    turn_number=turn_num,
                    player_index=player_idx,
                    observation={"state": f"turn_{turn_num}_player_{player_idx}"},
                    action=f"action_{turn_num}_{player_idx}",
                    response_time_ms=1500,
                    tokens_used=150,
                )

                # Add LLM call for this turn
                repo.add_llm_call(
                    game_turn_id=turn.id,
                    model="cerebras/llama3.1-8b" if player_idx == 0 else "gpt-4o",
                    messages={"messages": [{"role": "user", "content": "Play mafia"}]},
                    response={"response": "I will vote to eliminate player 3"},
                    request_sent_at=datetime.utcnow(),
                    response_received_at=datetime.utcnow(),
                    total_response_time_ms=1500,
                    response_tokens=20,
                    total_tokens_consumed=200,
                )

        print(f"  Simulated {game.total_turns} turns")

        # ===== Step 5: Finish the game =====
        print("\n5. Finishing the game...")

        repo.update_game_player(game.id, 0, role="Villager", reward=1.0)
        repo.update_game_player(game.id, 1, role="Mafia", reward=-1.0)

        repo.finish_game(
            game_id=game.id,
            status="completed",
            winning_team="Villager",
            environment_crashed=False,
            game_info={"notes": "Example game"},
        )

        print(f"  Game finished: {game.status}, winner: {game.winning_team}")

        # ===== Step 6: Update TrueSkill ratings =====
        print("\n6. Updating TrueSkill ratings...")

        repo.create_or_update_trueskill(agent1.id, mu=26.5, sigma=7.8)
        repo.create_or_update_trueskill(agent2.id, mu=24.2, sigma=8.1)

        print("  Updated TrueSkill ratings for 2 agents")

        # ===== Step 7: Query data =====
        print("\n7. Querying data...")

        # Get completed games
        completed_games = repo.get_games(status="completed")
        print(f"  Found {len(completed_games)} completed games")

        # Get game with relations
        game_with_data = repo.get_game_by_id(game.id, with_relations=True)
        if game_with_data:
            print(f"  Game {game_with_data.id} has:")
            print(f"    - {len(game_with_data.players)} players")
            print(f"    - {len(game_with_data.turns)} turns")

            for player in game_with_data.players:
                print(
                    f"    - Player {player.player_index}: {player.agent.alias} ({player.role}) - Reward: {player.reward}"
                )

        # Get top agents
        top_agents = repo.get_top_agents_by_trueskill(limit=5)
        print("\n  Top agents by TrueSkill:")
        for agent, trueskill in top_agents:
            rating = trueskill.mu - 3 * trueskill.sigma
            print(
                f"    - {agent.alias}: μ={trueskill.mu:.2f}, σ={trueskill.sigma:.2f}, rating={rating:.2f}"
            )

    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    example_usage()
