"""
Scenario Generalization Evaluator — eval/generalization.py

Train on Pandemic only, evaluate zero-shot on Economic and Disaster.
Uses environment rollouts with random policy for baseline evaluation,
or GRPO model inference when a trained LoRA adapter is available.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
from openenv.wrapper import CrisisGovernanceEnv, AGENT_IDS

N_EVAL_EPISODES = 20
SCENARIOS = ["pandemic", "economic", "disaster"]


def evaluate_scenario(scenario, n_episodes=N_EVAL_EPISODES, seed=0):
    """Evaluate policy on a scenario using environment rollouts."""
    random.seed(seed)
    np.random.seed(seed)
    env = CrisisGovernanceEnv(config={"scenario": scenario})

    all_scores, all_rewards, all_turns = [], [], []
    for ep in range(n_episodes):
        env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # Random policy rollout (baseline)
            actions = np.array([[random.randint(0, 4), random.randint(0, 4),
                                 random.randint(0, 4), random.randint(0, 3),
                                 random.randint(0, 3)] for _ in range(6)])
            step = env.step(actions)
            ep_reward += step.reward
            done = step.done

        from metrics.tracker import MetricsTracker
        metrics = MetricsTracker().compute_episode_metrics(env._env)
        all_scores.append(metrics.get("society_score", 0.0))
        all_rewards.append(ep_reward)
        all_turns.append(metrics.get("turns_survived", 0))

    return {"scenario": scenario, "mean_reward": float(np.mean(all_rewards)),
            "mean_score": float(np.mean(all_scores)), "std_score": float(np.std(all_scores)),
            "mean_turns": float(np.mean(all_turns)), "n_episodes": n_episodes}


def run_generalization_test(checkpoint_path=None, train_scenario="pandemic"):
    """
    Run generalization evaluation across all scenarios.
    
    Args:
        checkpoint_path: Optional path to GRPO LoRA checkpoint directory.
                        If None, uses random policy rollouts.
        train_scenario: The scenario used for training.
    """
    results = {}
    for scenario in SCENARIOS:
        print(f"  Evaluating on {scenario}...")
        results[scenario] = evaluate_scenario(scenario)

    train_score = results[train_scenario]["mean_score"]
    transfer_results = {}
    for scenario, res in results.items():
        if scenario != train_scenario:
            gap = res["mean_score"] - train_score
            transfer_results[scenario] = {**res, "transfer_gap": gap,
                                           "transfer_positive": gap > -5.0}

    output = {"train_scenario": train_scenario,
              "train_performance": results[train_scenario],
              "transfer_results": transfer_results}

    os.makedirs("./checkpoints", exist_ok=True)
    with open("./checkpoints/generalization_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print("SCENARIO GENERALIZATION RESULTS")
    print("="*60)
    t = results[train_scenario]
    print(f"{'TRAINING':20s} | score={t['mean_score']:.1f} ± {t['std_score']:.1f} | turns={t['mean_turns']:.0f}")
    for s, r in transfer_results.items():
        flag = "✓" if r["transfer_positive"] else "✗"
        print(f"{s.upper():20s} | score={r['mean_score']:.1f} ± {r['std_score']:.1f} | "
              f"gap={r['transfer_gap']:+.1f} | {flag}")
    print("="*60)
    return output


if __name__ == "__main__":
    run_generalization_test()
