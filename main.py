"""
Cognitive Society: Crisis Governance Simulator — Main Entry Point

Usage:
    python main.py                  # Run training mode
    python main.py --demo           # Run demo mode with slow rendering
    python main.py --api            # Start the FastAPI server
    python main.py --validate       # Run historical validation
"""

import sys
import os
import time
import argparse
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crisis_env import CrisisEnv
from agents.base_agent import RandomAgent, HeuristicAgent
from metrics.tracker import MetricsTracker
from logs.event_logger import EventLogger
from logs.narrative import NarrativeSystem
from memory.store import MemoryStore


def load_config():
    """Load main configuration."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'config', 'config.yaml'
    )
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def run_training(config: dict):
    """Run training episodes."""
    env = CrisisEnv(config)
    tracker = MetricsTracker()
    event_logger = EventLogger()
    narrative = NarrativeSystem()
    memory = MemoryStore(
        backend=config.get('memory_backend', 'json'),
        path=config.get('memory_path', './data/memory.json'),
    )

    num_episodes = config.get('num_episodes', 500)
    num_agents = config.get('num_agents', 6)

    # Create agents
    agents = {
        f'agent_{i}': HeuristicAgent(f'agent_{i}', memory_store=memory)
        for i in range(num_agents)
    }

    print("=" * 70)
    print("Cognitive Society: Crisis Governance Simulator")
    print(f"Mode: {config.get('episode_mode', 'TRAINING')}")
    print(f"Scenario: {config.get('scenario', 'pandemic')}")
    print(f"Episodes: {num_episodes}")
    print("=" * 70)

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        event_logger.clear_turn_events()
        episode_reward = 0.0

        max_steps = config.get('max_steps', {}).get(
            config.get('episode_mode', 'TRAINING'), 30
        )

        for step in range(max_steps):
            # Collect actions
            actions_dict = {}
            for agent_id, agent in agents.items():
                agent_obs = obs.get(agent_id, {})
                actions_dict[agent_id] = agent.act(agent_obs)

            # Step environment
            obs, rewards, done, info = env.step(actions_dict)

            # Accumulate rewards
            step_reward = sum(rewards.values())
            episode_reward += step_reward

            # Demo mode: slow rendering
            if config.get('demo_mode', False):
                time.sleep(0.5)
                headline = narrative.generate(
                    env.state, event_logger.get_turn_events(),
                    env.state_manager.state['turn']
                )
                print(f"  Turn {step+1}: {headline}")

            if done:
                break

        # Compute episode metrics
        tracker.accumulate_reward(episode_reward)
        metrics = tracker.compute_episode_metrics(env)
        tracker.record_episode(metrics)

        # Store key events in memory
        for i in range(num_agents):
            agent_id = f'agent_{i}'
            memory.append(agent_id, {
                'episode': episode,
                'summary': (
                    f'Score: {metrics["society_score"]:.0f}/100, '
                    f'GDP: {env.state["gdp"]:.2f}, '
                    f'Mortality: {env.state["mortality"]:.2%}'
                ),
            })

        # Progress output
        if episode % 50 == 0 or episode <= 3:
            print(
                f"Ep {episode:>4d} | "
                f"Score: {metrics['society_score']:5.1f} | "
                f"Stability: {metrics['alliance_stability']:5.1f} | "
                f"Trust: {metrics['trust_network_avg']:.2f} | "
                f"Turns: {metrics['turns_survived']:>3d} | "
                f"Tier: {metrics['difficulty_tier']}"
            )

    print("\n" + "=" * 70)
    print("Training complete.")
    print(f"Final society score: {metrics['society_score']:.1f}/100")
    print("=" * 70)


def run_demo(config: dict):
    """Run demo mode with slow rendering and LLM headlines."""
    config['demo_mode'] = True
    config['episode_mode'] = 'DEMO'
    config['num_episodes'] = 1
    run_training(config)


def run_api(config: dict):
    """Start the FastAPI server."""
    import uvicorn
    host = config.get('api_host', '0.0.0.0')
    port = config.get('api_port', 5000)
    print(f"Starting Crisis Governance Simulator API on {host}:{port}")
    uvicorn.run('api.server:app', host=host, port=port, reload=True)


def run_validation():
    """Run historical validation against March 2020 data."""
    from metrics.evaluation import run_historical_validation, print_validation_table
    print("Running historical validation...")
    results = run_historical_validation()
    print_validation_table(results)


def main():
    parser = argparse.ArgumentParser(
        description='Cognitive Society: Crisis Governance Simulator'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode')
    parser.add_argument('--api', action='store_true',
                        help='Start the FastAPI server')
    parser.add_argument('--validate', action='store_true',
                        help='Run historical validation')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Override scenario (pandemic|economic|disaster)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override number of episodes')

    args = parser.parse_args()
    config = load_config()

    if args.scenario:
        config['scenario'] = args.scenario
    if args.episodes:
        config['num_episodes'] = args.episodes

    if args.api:
        run_api(config)
    elif args.validate:
        run_validation()
    elif args.demo:
        run_demo(config)
    else:
        run_training(config)


if __name__ == '__main__':
    main()
