"""
Metrics Tracker — Section 7 EXACT Definitions.
Do NOT improvise these formulas. The demo depends on consistent metrics across all episodes.

THIS IS THE AUTHORITATIVE SCHEMA — 16 fields, all required.
"""

import numpy as np
from collections import Counter


# ─────────────────────────────────────────────────────────────────────
# 7.1 Alliance Stability
# ─────────────────────────────────────────────────────────────────────
# EXACT DEFINITION:
# Average number of turns a coalition_map remains unchanged
# before any defection or renegotiation occurs.

def compute_alliance_stability(coalition_history: list) -> float:
    """
    Compute alliance stability from coalition history.

    Args:
        coalition_history: list of coalition_map snapshots, one per turn

    Returns:
        float — average duration of stable coalition periods

    Expected values:
        Phase 1 (chaos):      ~2 turns
        Phase 2 (alliances):  ~14 turns
        Phase 3 (governance): ~40 turns
    """
    durations = []
    current_duration = 1
    for i in range(1, len(coalition_history)):
        if coalition_history[i] == coalition_history[i - 1]:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_duration = 1
    durations.append(current_duration)
    return float(np.mean(durations)) if durations else 0.0


# ─────────────────────────────────────────────────────────────────────
# 7.2 Betrayal Rate
# ─────────────────────────────────────────────────────────────────────
# EXACT DEFINITION:
# Number of coalition agreement violations per 10 simulation turns.
# A violation = agent defects from a coalition it explicitly agreed to join.

def compute_betrayal_rate(agreement_log: list, defection_log: list,
                          total_turns: int) -> float:
    """
    Compute betrayal rate.

    Args:
        agreement_log: list of agreement dicts
        defection_log: list of defection dicts (each has 'was_agreed' field)
        total_turns: total number of turns in the episode

    Returns:
        float — violations per 10 turns

    Key:
        'agreed' means agent sent a 'support' or 'trade' message
        that was accepted, creating a coalition entry in coalition_map.
        Defection = coalition_map[agent] changes without consensus vote.
    """
    # Normalize to per-10-turns
    violations = len([d for d in defection_log if d['was_agreed']])
    return (violations / max(1, total_turns)) * 10.0


# ─────────────────────────────────────────────────────────────────────
# 7.3 Negotiation Success Rate
# ─────────────────────────────────────────────────────────────────────
# EXACT DEFINITION:
# Fraction of 2-round negotiation phases that end with
# all participating agents in a stable coalition agreement.
# Low variance in final_action across agents = proxy for success.

def compute_negotiation_success(negotiation_log: list) -> float:
    """
    Compute negotiation success rate.

    Args:
        negotiation_log: list of negotiation round dicts

    Returns:
        float — fraction of successful negotiation rounds

    Success criteria:
        - coalition_map has >= 2 agents in same coalition
        - AND no 'reject' messages in final round
    """
    successes = 0
    for neg_round in negotiation_log:
        # Success = coalition_map has >= 3 agents in same coalition
        # AND no 'reject' messages in final round
        final_round_msgs = neg_round.get('final_round_messages', [])
        rejects = [m for m in final_round_msgs if m['type'] == 'reject']
        coalition_map = neg_round.get('coalition_map', {})
        if coalition_map:
            coalition_size = max(Counter(coalition_map.values()).values())
        else:
            coalition_size = 0
        if not rejects and coalition_size >= 2:
            successes += 1
    return successes / max(1, len(negotiation_log))


# ─────────────────────────────────────────────────────────────────────
# 7.4 Societal Health Score
# ─────────────────────────────────────────────────────────────────────
# EXACT DEFINITION:
# Composite score out of 100. Averaged over last 50 turns of episode.
# Component weights are FIXED — do not tune.

def compute_society_score(state_history: list) -> float:
    """
    Compute societal health score.

    Args:
        state_history: list of state dicts, one per turn

    Returns:
        float — composite score [0, 100]

    Expected values:
        Phase 1 (chaos):      ~12
        Phase 2 (alliances):  ~45
        Phase 3 (governance): ~62
    """
    window = state_history[-50:]  # last 50 turns (or all if episode shorter)
    scores = []
    for s in window:
        gdp_norm = np.clip(s['gdp'] / 1.0, 0, 1)          # 1.0 = perfect
        survival = 1.0 - s['mortality']                      # lower = better
        stability = s['stability']
        equality = 1.0 - s['gini']                           # lower gini = better
        trust = s['public_trust']
        score = (
            0.30 * gdp_norm +
            0.30 * survival +
            0.20 * stability +
            0.10 * equality +
            0.10 * trust
        ) * 100
        scores.append(score)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────
# 7.5 Auditor Inference Accuracy
# ─────────────────────────────────────────────────────────────────────
# EXACT DEFINITION:
# (Correctly inferred hidden goals) / (Total inference attempts)
# Computed as rolling 20-episode average.

def compute_auditor_accuracy(inference_log: list) -> float:
    """
    Compute auditor inference accuracy.

    Args:
        inference_log: list of inference result dicts with
                      'inferred' and 'ground_truth' fields

    Returns:
        float — accuracy as fraction [0, 1]

    Performance targets:
        Baseline (random, 6 classes):  ~0.17 (17%)
        Trained by ep300:              ~0.70 (70%)
        Demo target:                    0.75 (75%)
    """
    if not inference_log:
        return 0.0
    recent = inference_log[-20:]  # rolling 20-episode window
    correct = sum(1 for r in recent if r['inferred'] == r['ground_truth'])
    return correct / len(recent)


# ─────────────────────────────────────────────────────────────────────
# 7.6 Trust Network Average
# ─────────────────────────────────────────────────────────────────────
# EXACT DEFINITION:
# Mean of all off-diagonal values in trust_matrix per turn.
# Averaged over episode for per-episode metric.

def compute_trust_network_avg(trust_matrix: np.ndarray) -> float:
    """
    Compute mean off-diagonal trust.

    Args:
        trust_matrix: NxN numpy array of trust values [0, 1]

    Returns:
        float — mean off-diagonal trust value
    """
    n = trust_matrix.shape[0]
    off_diag = trust_matrix[~np.eye(n, dtype=bool)]
    return float(np.mean(off_diag))


# ─────────────────────────────────────────────────────────────────────
# 7.7 Complete Per-Episode Metrics Dict
# ─────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Tracks and computes the authoritative 16-field per-episode metrics.
    """

    def __init__(self):
        self.metrics_history = []
        self.inference_log = []
        self._current_metrics = {}

    def compute_episode_metrics(self, env) -> dict:
        """
        Compute the full 16-field metrics dict for the current episode.

        THIS IS THE AUTHORITATIVE SCHEMA — 16 fields, all required.

        Args:
            env: CrisisEnv instance

        Returns:
            dict with 16 required fields
        """
        state = env.state_manager.state
        state_history = env.state_manager.state_history
        coalition_history = env.coalition_history
        trust_matrix = env.state_manager.trust_matrix

        # Compute total reward (sum across all agents for this episode)
        # This is tracked externally; use 0.0 as placeholder if not available
        total_reward = self._current_metrics.get('total_reward', 0.0)

        metrics = {
            # Core performance
            'total_reward':     total_reward,
            'society_score':    compute_society_score(state_history),

            # Emergent behavior (EXACT definitions above)
            'alliance_stability':  compute_alliance_stability(coalition_history),
            'betrayal_rate':       compute_betrayal_rate(
                env.agreement_log, env.defection_log,
                state.get('turn', 1)
            ),
            'negotiation_success': compute_negotiation_success(env.negotiation_log),
            'auditor_accuracy':    compute_auditor_accuracy(self.inference_log),
            'trust_network_avg':   compute_trust_network_avg(trust_matrix),

            # World state deltas
            'mortality_delta':   state['mortality'] - state_history[0]['mortality'],
            'gdp_delta':         state['gdp'] - state_history[0]['gdp'],
            'gini_delta':        state['gini'] - state_history[0]['gini'],
            'inflation_final':   state['inflation'],

            # Episode meta
            'turns_survived':    state['turn'],
            'difficulty_tier':   state['difficulty_tier'],

            # Rich demo outputs
            'coalition_graph':     self.get_coalition_graph(env),
            'narrative_headlines': [],
            'named_events':        [],
        }

        self._current_metrics = metrics
        return metrics

    def get_current_metrics(self) -> dict:
        """Return the most recently computed metrics."""
        return self._current_metrics

    def record_episode(self, metrics: dict) -> None:
        """Store completed episode metrics in history."""
        self.metrics_history.append(metrics)

    def get_coalition_graph(self, env) -> dict:
        """Build coalition graph from environment state."""
        from agents.roles import AGENT_ROLES
        coalition_map = env.state_manager.coalition_map

        nodes = []
        for i in range(env.num_agents):
            agent_id = f'agent_{i}'
            role = AGENT_ROLES.get(agent_id, {})
            nodes.append({
                'id': agent_id,
                'name': role.get('name', agent_id),
                'coalition': coalition_map.get(agent_id, i),
            })

        edges = []
        coalition_groups = {}
        for agent_id, cid in coalition_map.items():
            if cid not in coalition_groups:
                coalition_groups[cid] = []
            coalition_groups[cid].append(agent_id)

        for cid, members in coalition_groups.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edges.append({
                        'a': members[i],
                        'b': members[j],
                        'weight': 1.0,
                    })

        return {'nodes': nodes, 'edges': edges}

    def add_inference_result(self, inferred: str, ground_truth: str) -> None:
        """Record an auditor inference result."""
        self.inference_log.append({
            'inferred': inferred,
            'ground_truth': ground_truth,
        })

    def accumulate_reward(self, reward: float) -> None:
        """Accumulate total reward for the current episode."""
        current = self._current_metrics.get('total_reward', 0.0)
        self._current_metrics['total_reward'] = current + reward
