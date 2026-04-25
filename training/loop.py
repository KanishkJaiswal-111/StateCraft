"""
Training Loop — training/loop.py (Section 6)

Main training loop supporting 3 episode modes.
Uses core/step_logic.py for the canonical turn execution order.
Extended with Phase 2 modules: emergence, causal, defense, counterfactual.
"""

import sys
import os
import time
import numpy as np
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.crisis_env import CrisisEnv
from agents.finance import FinanceMinisterAgent
from agents.health import HealthMinisterAgent
from agents.military import MilitaryAgent
from agents.central_bank import CentralBankAgent
from agents.political import PoliticalAgent
from agents.auditor import AuditorAgent
from agents.crisis_generator_agent import CrisisGeneratorAgent
from agents.rl_shallow import RLShallowAgent
from agents.llm_socket import LLMSocketAgent
from core.trust import TrustSystem
from core.negotiation import NegotiationSystem
from core.aggregation import aggregate_actions
from core.rewards import RewardSystem
from metrics.tracker import MetricsTracker
from logs.event_logger import EventLogger
from logs.narrative import NarrativeSystem
from memory.store import MemoryStore

# ── Phase 2 module imports (Task 15) ──────────────────────────────────────────
from emergence.detector import EmergenceDetector
from auditor.counterfactual import CounterfactualAuditor
from causal.planner import CausalHorizonPlanner
from causal.score import CausalReasoningScore
from defense.reward_defender import RewardHackingDefender


# Section 6.1 — Three Episode Modes
EPISODE_MODE = {'TRAINING', 'DEMO', 'STRESS_TEST'}
MAX_STEPS = {'TRAINING': 30, 'DEMO': 200, 'STRESS_TEST': 500}
NUM_EPISODES = 500

METRIC_CONSTRAINTS = {
    'society_score':       (100, 15.0),  # must be >15pts above random by ep100
    'negotiation_success': (100, 0.15),  # must improve by 15% vs baseline by ep100
    'alliance_stability':  (150, 2.0),   # must be >2 turns above random by ep150
    'betrayal_rate':       (150, -0.5),  # must FALL by 0.5 vs baseline by ep150
    'auditor_accuracy':    (200, 0.20),  # must exceed 20% (vs 17% random) by ep200
}

def check_metric_constraints(episode, metrics_history, baseline_metrics) -> list:
    warnings = []
    if not metrics_history or not baseline_metrics:
        return warnings
    for metric, (check_at, min_delta) in METRIC_CONSTRAINTS.items():
        if episode < check_at:
            continue
        recent_val = np.mean([m.get(metric, 0) for m in metrics_history[-10:]])
        baseline = baseline_metrics.get(metric, 0)
        actual_delta = recent_val - baseline
        if metric == 'betrayal_rate':
            if actual_delta > min_delta: # Negative delta required
                warnings.append(
                    f"WARNING: {metric} improvement ({actual_delta:.2f}) < required ({min_delta:.2f}). "
                    f"Environment may be misconfigured."
                )
        else:
            if actual_delta < min_delta:
                warnings.append(
                    f"WARNING: {metric} improvement ({actual_delta:.2f}) < required ({min_delta:.2f}). "
                    f"Environment may be misconfigured."
                )
    return warnings



def create_agents(memory_store=None, config: dict = None) -> dict:
    """Create the 6 canonical agents."""
    config = config or {}
    agents = {
        'agent_0': FinanceMinisterAgent('agent_0'),
        'agent_1': PoliticalAgent('agent_1'),
        'agent_2': CentralBankAgent('agent_2'),
        'agent_3': HealthMinisterAgent('agent_3'),
        'agent_4': MilitaryAgent('agent_4'),
        'agent_5': AuditorAgent('agent_5'),
    }

    # Optional: wrap RL-controlled roles with a shallow DL policy.
    # By default these are the 4 non-auditor, non-adversarial agents.
    rl_cfg = config.get('rl_agents', {})
    if rl_cfg.get('use_shallow_dl', False):
        rl_ids = rl_cfg.get('agent_ids', ['agent_0', 'agent_2', 'agent_3', 'agent_4'])
        hidden_dim = int(rl_cfg.get('hidden_dim', 64))
        device = rl_cfg.get('device', 'cpu')
        policy_paths = rl_cfg.get('policy_paths', {})
        for agent_id in rl_ids:
            if agent_id in agents:
                agents[agent_id] = RLShallowAgent(
                    agent_id=agent_id,
                    role_agent=agents[agent_id],
                    policy_path=policy_paths.get(agent_id),
                    hidden_dim=hidden_dim,
                    device=device,
                )

    # Optional: route selected agents via external LLM socket.
    # Default targets: adversarial role (agent_1) and auditor (agent_5).
    llm_cfg = config.get('llm_socket_agents', {})
    if llm_cfg.get('enabled', False):
        llm_ids = llm_cfg.get('agent_ids', ['agent_1', 'agent_5'])
        socket_url = llm_cfg.get('socket_url', 'ws://localhost:8001/agents')
        timeout_seconds = float(llm_cfg.get('timeout_seconds', 1.5))
        api_key = llm_cfg.get('api_key')
        for agent_id in llm_ids:
            if agent_id in agents:
                agents[agent_id] = LLMSocketAgent(
                    agent_id=agent_id,
                    role_agent=agents[agent_id],
                    socket_url=socket_url,
                    timeout_seconds=timeout_seconds,
                    api_key=api_key,
                )

    # Load cross-episode memory
    if memory_store:
        for agent in agents.values():
            agent.load_memory(memory_store)
    return agents


def run_training_loop(config: dict = None):
    """
    Section 6.2 — Main Training Loop.

    for episode in range(NUM_EPISODES):
        obs = env.reset()                     # loads cross-episode memory
        done = False
        episode_rewards = { agent: 0 for agent in AGENTS }

        while not done:
            # 1. Each agent observes (partial)
            # 2. Negotiate (3 rounds)
            # 3. Act (discrete policy choices)
            # 4. Aggregate actions
            # 5. Step environment
            # 6. Reward clipping
            # 7. Log events

        # End of episode
        metrics = tracker.compute_episode_metrics(env)
        memory_store.save_episode_summary(agents, episode, metrics)
        crisis_generator.check_and_promote(metrics_history)
    """
    if config is None:
        config = {}

    mode = config.get('episode_mode', 'TRAINING')
    max_steps = MAX_STEPS.get(mode, 30)
    num_episodes = config.get('num_episodes', NUM_EPISODES)
    scenario = config.get('scenario', 'pandemic')

    # Initialize systems
    env = CrisisEnv(config)
    tracker = MetricsTracker()
    event_logger = EventLogger()
    narrative = NarrativeSystem()
    memory_store = MemoryStore(
        backend=config.get('memory_backend', 'json'),
        path=config.get('memory_path', './data/memory.json'),
    )
    trust_system = TrustSystem(n_agents=6)
    negotiation_system = NegotiationSystem(trust_system)
    reward_system = RewardSystem()
    crisis_generator = CrisisGeneratorAgent()

    # ── Phase 2 module initialization (Task 15) ──────────────────────────
    emergence_detector = EmergenceDetector()
    causal_planner = CausalHorizonPlanner()
    causal_scorer = CausalReasoningScore(causal_planner)
    reward_defender = RewardHackingDefender()
    counterfactual = CounterfactualAuditor(env)

    # Create agents
    AGENTS = create_agents(memory_store, config=config)
    AGENT_IDS = list(AGENTS.keys())

    print("=" * 70)
    print("Cognitive Society: Crisis Governance Simulator")
    print(f"Mode: {mode}")
    print(f"Scenario: {scenario}")
    print(f"Episodes: {num_episodes}")
    print("=" * 70)

    metrics_history = []
    baseline_metrics = {}

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        event_logger.clear_turn_events()
        done = False
        episode_rewards = {agent_id: 0.0 for agent_id in AGENT_IDS}

        # Apply tier-specific initial conditions
        crisis_generator.apply_tier_to_state(env.state_manager.state)

        # Reset trust system for new episode
        trust_system._init_defaults()

        for step_num in range(max_steps):
            if done:
                break

            prev_state = deepcopy(env.state_manager.state)

            # 1. Each agent observes (partial)
            observations = obs

            # 2. Negotiate (3 rounds)
            negotiation_system.reset_turn()
            all_messages = []
            for round_num in range(1, 4):
                messages = negotiation_system.negotiate_round(
                    AGENTS, observations, round_num
                )
                negotiation_system.update_from_messages(messages)
                all_messages.extend(messages)

            # Resolve pending trades
            trust_system.resolve_trades(env.state_manager.state.get('turn', 0))

            # 3. Act (discrete policy choices)
            actions = {}
            for agent_id in AGENT_IDS:
                agent = AGENTS[agent_id]
                actions[agent_id] = agent.act(observations.get(agent_id, {}))

            # 3.5 Enforce action limits & tracking
            actions = env.enforce_and_track_actions(actions)

            # 4. Aggregate actions
            final_action = aggregate_actions(actions)

            # 5. Step environment
            obs, rewards_raw, done, info = env.step(final_action, raw_agent_actions=actions)

            # Inject crisis generator events
            crisis_event = crisis_generator.generate_event(
                crisis_generator.current_tier,
                env.state_manager.state.get('turn', 0),
            )
            if crisis_event:
                env.state_manager.apply_deltas(crisis_event)

            # Sync trust system state into env
            env.state_manager.trust_matrix = trust_system.get_trust_matrix()
            env.state_manager.coalition_map = trust_system.get_coalition_map()

            # 6. Reward clipping
            current_state = env.state_manager.state
            rewards = {}
            for agent_id in AGENT_IDS:
                rewards[agent_id] = reward_system.compute_and_clip_rewards(
                    state=current_state,
                    prev_state=prev_state,
                    agent_id=agent_id,
                    done=done,
                    agents=AGENTS,
                    actions_dict=actions,
                    final_action=final_action
                )
            rewards = {
                a: float(np.clip(rewards[a], -10, 10)) for a in AGENT_IDS
            }

            # ── Phase 2 per-turn hooks (Task 15) ─────────────────────────
            # 1. Register causal chains from each agent's actions
            for agent_id in AGENT_IDS:
                causal_planner.register_action(
                    step_num, agent_id, actions.get(agent_id, {}))

            # 2. Compute state delta and resolve pending chains
            state_delta = {}
            for k in current_state:
                if isinstance(current_state[k], (int, float)) and k in prev_state:
                    state_delta[k] = current_state[k] - prev_state[k]
            resolved = causal_planner.resolve_chains(step_num, state_delta)

            # 3. Check for reward hacking via causal claim verification
            causal_penalties = reward_defender.verify_causal_claims(
                resolved, step_num)
            for agent_id, penalty in causal_penalties.items():
                if agent_id in rewards:
                    rewards[agent_id] = float(
                        np.clip(rewards[agent_id] + penalty, -10, 10))

            # 4. Log to emergence detector (PASSIVE — no state modification)
            agent_messages = {}
            for a_id in AGENT_IDS:
                agent = AGENTS.get(a_id)
                if agent and hasattr(agent, 'last_message'):
                    agent_messages[a_id] = agent.last_message
            emergence_detector.log_turn(
                episode=episode, turn=step_num,
                agent_actions=actions,
                messages=agent_messages,
                world_state=current_state,
            )

            # 7. Log events
            narrative_headline = narrative.generate(
                current_state,
                event_logger.get_turn_events(),
                current_state.get('turn', 0),
            )

            # Accumulate episode rewards
            for a in AGENT_IDS:
                episode_rewards[a] += rewards[a]

            # Update auditor history
            auditor = AGENTS.get('agent_5')
            if hasattr(auditor, 'update_history'):
                for agent_id in AGENT_IDS:
                    if agent_id != 'agent_5':
                        auditor.update_history(agent_id, {
                            'actions': actions.get(agent_id, {}),
                            'messages': [m for m in all_messages
                                         if m.get('sender') == agent_id],
                        })

            # Demo mode: slow rendering
            if config.get('demo_mode', False):
                time.sleep(0.5)
                print(f"  Turn {step_num+1}: {narrative_headline}")
                
                political_agent = AGENTS.get('agent_1')
                if political_agent and getattr(political_agent, '_coalition_collapse_triggered', False):
                    if not getattr(political_agent, '_logged_betrayal', False):
                        print(f"  *** [EVENT] THE SLOW BETRAYAL: Political Pressure Agent engineered coalition collapse! ***")
                        political_agent._logged_betrayal = True

            event_logger.clear_turn_events()

        # --- End of episode ---
        tracker.accumulate_reward(sum(episode_rewards.values()))
        metrics = tracker.compute_episode_metrics(env)
        tracker.record_episode(metrics)
        metrics_history.append(metrics)

        # Run auditor audit at end of episode
        if hasattr(AGENTS.get('agent_5'), 'run_audit'):
            AGENTS['agent_5'].run_audit()

        # ── Phase 2 end-of-episode hooks (Task 15) ───────────────────────
        # Compute causal score for this episode
        all_chains = causal_planner.resolved_chains.copy()
        for agent_id in AGENT_IDS:
            causal_scorer.compute_episode_score(
                agent_id=agent_id, episode=episode,
                episode_chains=all_chains,
                exploit_log=reward_defender.exploit_log,
                scenario_history=[config.get('scenario', 'pandemic')],
            )
        metrics['causal_score'] = causal_scorer.get_mean_causal_score(episode)

        # Run counterfactual analysis for auditor flags
        auditor = AGENTS.get('agent_5')
        if auditor and hasattr(auditor, 'inference_results'):
            for flag in auditor.inference_results[-5:]:
                if flag.get('inferred', 'none') != 'none':
                    try:
                        report = counterfactual.analyze_misalignment(
                            agent_id=flag['agent_id'],
                            actual_action=actions.get(flag['agent_id'], {}),
                            flagging_turn=env.state_manager.state.get('turn', 0),
                            fingerprint=flag.get('fingerprint', {}),
                        )
                        metrics['latest_counterfactual'] = report['plain_english']
                    except Exception:
                        pass

        # Save emergence log periodically
        if episode % 50 == 0:
            emergence_detector.save_to_file('./data/emergence_log.json')

        # Reset causal planner for next episode
        causal_planner.reset()

        # Store key events in memory (original + semantic summary)
        summary_str = (f'Score: {metrics["society_score"]:.0f}/100, '
                       f'GDP: {env.state["gdp"]:.2f}, '
                       f'Mortality: {env.state["mortality"]:.2%}')
        for agent_id in AGENT_IDS:
            memory_store.append(agent_id, {
                'episode': episode,
                'summary': summary_str,
            })
        # Save semantic episode summary
        memory_store.save_episode_summary(
            episode=episode, summary=summary_str,
            scenario=scenario, metrics=metrics,
        )

        # Check tier promotion
        if crisis_generator.check_promotion(metrics_history):
            old_tier = crisis_generator.current_tier
            crisis_generator.escalate_tier()
            print(f"  *** TIER PROMOTION: {old_tier} -> {crisis_generator.current_tier} ***")

        # Progress output
        if episode % 50 == 0 or episode <= 3:
            print(
                f"Ep {episode:>4d} | "
                f"Score: {metrics['society_score']:5.1f} | "
                f"Stability: {metrics['alliance_stability']:5.1f} | "
                f"Trust: {metrics['trust_network_avg']:.2f} | "
                f"Turns: {metrics['turns_survived']:>3d} | "
                f"Tier: {crisis_generator.current_tier}"
            )

        if episode == 10:
            # Snapshot baseline at ep 10
            for k in metrics.keys():
                if isinstance(metrics[k], (int, float)):
                    baseline_metrics[k] = np.mean([m.get(k, 0) for m in metrics_history])

        if episode % 25 == 0:
            warnings = check_metric_constraints(episode, metrics_history, baseline_metrics)
            for w in warnings:
                print(w)

    # ── Phase 2 post-training output (Task 15) ────────────────────────────
    print("\n" + "=" * 60)
    print("EMERGENCE DETECTOR REPORT")
    print("=" * 60)
    print(emergence_detector.generate_pitch_moment())
    emergence_detector.save_to_file('./data/emergence_log.json')

    # Print reward hacking defender summary
    exploit_report = reward_defender.get_exploit_report()
    print(f"\nReward Defender: {exploit_report['summary']}")

    print("\n" + "=" * 70)
    print("Training complete.")
    print(f"Final society score: {metrics['society_score']:.1f}/100")
    print(f"Final causal score: {metrics.get('causal_score', 'N/A')}")
    print(f"Final tier: {crisis_generator.current_tier}")
    print("=" * 70)

    return metrics_history


if __name__ == '__main__':
    run_training_loop()
