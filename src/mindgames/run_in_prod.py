#!/usr/bin/env python3
"""
Production deployment script for MindGames agents.

This script connects your agent to the online Secret Mafia competition,
supporting multiple agent types with configurable parameters.
"""

import argparse
import sys
import traceback

from dataclasses import dataclass

import textarena as ta
from mindgames.agents.basic import ThinkingAgent
from mindgames.constants import DEFAULT_API_KEY, DEFAULT_BASE_URL, TEAM_HASH
from mindgames.prompts import V3_PROMPT, V4_PROMPT


@dataclass
class DeploymentConfig:
    """Configuration for a production deployment."""

    deployment_name: str
    system_prompt: str
    model_name: str
    agent_kind: str
    small_category: bool


# Model configurations for deployments
DEPLOYMENTS: dict[str, DeploymentConfig] = {
    "tungsten_008s": DeploymentConfig(
        deployment_name="tungsten_008s",
        system_prompt=V4_PROMPT,
        model_name="Qwen/Qwen3-8B",
        agent_kind="thinking",
        small_category=True,
    ),
}

def main():
    """Main entry point for production deployment."""
    parser = argparse.ArgumentParser(
        description="Connect your agent to the Secret Mafia online competition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run run_in_prod.py --deployment tungsten_008s
        """,
    )

    # Required arguments
    _ = parser.add_argument(
        "--deployment",
        required=True,
        choices=DEPLOYMENTS.keys(),
        help="Which deployment to use",
    )

    args = parser.parse_args()

    deployment_config = DEPLOYMENTS.get(args.deployment)
    assert deployment_config is not None, (
        f"Failed to lookup deployment '{args.deployment}'"
    )

    # Set defaults for model description
    model_description = f"Model description: {deployment_config.deployment_name}"

    print(f"🚀 Deploying {deployment_config.deployment_name} agent to production...")
    print("📊 Track: Social Detection (SecretMafia-v0)")
    print(f"🤖 Model: {deployment_config.model_name}")
    print(f"📄 Small Category: {deployment_config.small_category}")
    print("-" * 60)

    try:
        # Create the agent
        print("🔧 Initializing agent...")

        if deployment_config.agent_kind == "thinking":
            agent = ThinkingAgent(
                model_name=deployment_config.model_name,
                system_prompt=deployment_config.system_prompt,
            )
        else:
            assert False, f"Couldn't load agent kind '{deployment_config.agent_kind}'"

        print("✅ Agent initialized successfully")

        # Connect to online environment
        print("🌐 Connecting to online environment...")
        env = ta.make_mgc_online(
            track="Social Detection",
            model_name=deployment_config.model_name,
            model_description=model_description,
            team_hash=TEAM_HASH,
            agent=agent,
            small_category=deployment_config.small_category,
        )

        print("✅ Connected to online environment")

        # Reset environment
        env.reset(num_players=1)  # Always 1 for online play
        print("🎮 Environment reset, starting game...")

        # Main game loop
        done = False
        turn_count = 0

        while not done:
            turn_count += 1
            if args.verbose:
                print(f"\n🔄 Turn {turn_count}")

            # Get observation
            player_id, observation = env.get_observation()

            if args.verbose:
                print(f"📥 Observation for Player {player_id}:")
                print(
                    f"   {observation[:200]}..."
                    if len(observation) > 200
                    else f"   {observation}"
                )

            # Get agent action
            action = agent(observation)
            print("##################")
            print(action)
            print("##################")

            if args.verbose:
                print("📤 Agent response:")
                print(f"   {action}")

            # Execute action
            done, step_info = env.step(action=action)

        # Game finished
        rewards, game_info = env.close()

        print("\n🏁 Game completed!")
        print(f"🏆 Final rewards: {rewards}")
        print(f"📊 Game info: {game_info}")

    except KeyboardInterrupt:
        print("\n👋 Game interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    n = 10
    while n > 0:
        print(f"{n=}")
        main()
        n -= 1
