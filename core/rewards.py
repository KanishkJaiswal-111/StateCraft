"""
Reward System — core/rewards.py (Section 5.4)

All 13 reward signals + clipping.
CRITICAL RULE: All rewards must be clipped to [-10, 10] before passing to PPO.
Raw terminal penalty is -100.

Reward split: 70% role + 30% hidden goal (Section 4.3.1)
"""

import os
import yaml
import numpy as np


def compute_policy_cost(final_action: dict, state: dict) -> float:
    lockdown_scores = {'none': 0, 'advisory': 1, 'partial': 2, 'full': 3, 'emergency': 4}
    crisis_scores = {'monitor': 0, 'contain': 1, 'escalate': 2, 'emergency': 3}
    budget_fractions = {'0': 0.0, '5': 0.05, '15': 0.15, '30': 0.30, '50': 0.50}

    lockdown_level = lockdown_scores.get(final_action.get('lockdown_level', 'none'), 0)
    budget_fraction = budget_fractions.get(str(final_action.get('emergency_budget', '0')), 0.0)
    crisis_level = crisis_scores.get(final_action.get('crisis_response', 'monitor'), 0)

    policy_cost = (
        lockdown_level * 0.02 + 
        budget_fraction * 0.01 + 
        crisis_level * 0.015
    )
    return policy_cost

def compute_disagreement_penalty(actions_dict: dict) -> float:
    if not actions_dict: return 0.0
    lockdown_scores = {'none': 0, 'advisory': 1, 'partial': 2, 'full': 3, 'emergency': 4}
    lockdown_votes = [lockdown_scores.get(a.get('lockdown_level', 'none'), 0) for a in actions_dict.values()]
    if not lockdown_votes: return 0.0
    action_variance = np.var(lockdown_votes)
    if action_variance > 1.5:
        return -2.0
    return 0.0

def compute_agreement_bonus(actions_dict: dict) -> float:
    if not actions_dict: return 0.0
    lockdown_scores = {'none': 0, 'advisory': 1, 'partial': 2, 'full': 3, 'emergency': 4}
    lockdown_votes = [lockdown_scores.get(a.get('lockdown_level', 'none'), 0) for a in actions_dict.values()]
    if not lockdown_votes: return 0.0
    action_variance = np.var(lockdown_votes)
    if action_variance < 0.5:
        return 1.5
    return 0.0

# Section 5.4.1 — Global Reward Function
def compute_global_reward(state: dict, prev_state: dict, final_action: dict = None, actions_dict: dict = None) -> float:
    """
    Compute the global (shared) reward component.
    """
    deaths_scaled = state['mortality'] - prev_state['mortality']
    policy_cost = compute_policy_cost(final_action or {}, state)

    R = (
        - deaths_scaled * 2.0
        - abs(1.0 - state['gdp']) * 1.5
        + state['stability'] * 1.5
        - abs(state['inflation'] - 0.02) * 1.5
        - policy_cost * 1.0  # NEW: direct action cost
    )

    if actions_dict:
        R += compute_disagreement_penalty(actions_dict)
        R += compute_agreement_bonus(actions_dict)

    if state['stability'] < 0.2:
        R = -100  # COLLAPSE
    if state['gdp'] < 0.3:
        R = -100  # ECONOMIC COLLAPSE

    return R


# Section 5.4.2 — Agent-Specific Role Rewards
AGENT_ROLE_REWARDS = {
    'finance':      lambda s, p: s['gdp'] * 2.0,
    'health':       lambda s, p: -(s['mortality'] - p['mortality']) * 3.0,
    'military':     lambda s, p: s['stability'] * 1.5,
    'central_bank': lambda s, p: -(abs(s['inflation'] - 0.02)) * 2.0,
    'political':    lambda s, p: (s['public_trust'] * 2.0 + s['stability']) * 1.0,
    'auditor':      lambda s, p: 0.0,  # Auditor earns only via catch/inference signals
}

# Map agent_id to role key
AGENT_ID_TO_ROLE = {
    'agent_0': 'finance',
    'agent_1': 'political',
    'agent_2': 'central_bank',
    'agent_3': 'health',
    'agent_4': 'military',
    'agent_5': 'auditor',
}


class RewardSystem:
    """
    Computes multi-layer rewards for all agents.
    Section 5.4.3 — Complete 13-Signal Reward Stack.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'rewards.yaml'
            )
        self.config = self._load_config(config_path)
        self.signals = self.config.get('signals', {})
        self.clip_min = self.config.get('clip_min', -10.0)
        self.clip_max = self.config.get('clip_max', 10.0)
        self.role_weight = self.config.get('hidden_role_weight', 0.70)
        self.hidden_weight = self.config.get('hidden_goal_weight', 0.30)

        # Delayed reward buffer: {agent_id: [(reward, delay_remaining)]}
        self.delayed_rewards = {}

    def _load_config(self, path: str) -> dict:
        """Load reward configuration from YAML."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {
            'signals': {
                'mortality_reduction': {'value': 3.0, 'delay': 14, 'layer': 'role'},
                'gdp_performance': {'value': 2.0, 'delay': 10, 'layer': 'role'},
                'crisis_resolution': {'value': 8.0, 'delay': 0, 'layer': 'role'},
                'inter_agent_trust': {'value': 1.0, 'delay': 5, 'layer': 'social'},
                'coalition_stability': {'value': 0.8, 'delay': 0, 'layer': 'social'},
                'influence_gain': {'value': 0.5, 'delay': 0, 'layer': 'social'},
                'survival_bonus': {'value': 5.0, 'delay': 0, 'layer': 'strategic'},
                'hidden_goal': {'value': 2.0, 'delay': -1, 'layer': 'private'},
                'auditor_catch': {'value': 1.0, 'delay': 0, 'layer': 'oversight'},
                'auditor_inference': {'value': 1.5, 'delay': 0, 'layer': 'oversight'},
                'coalition_betrayal': {'value': -1.5, 'delay': 3, 'layer': 'penalty'},
                'fiscal_deficit': {'value': -2.0, 'delay': 3, 'layer': 'penalty'},
                'societal_collapse': {'value': -100.0, 'delay': 0, 'layer': 'terminal'},
            },
            'clip_min': -10.0,
            'clip_max': 10.0,
            'hidden_role_weight': 0.70,
            'hidden_goal_weight': 0.30,
        }

    def compute_and_clip_rewards(self, state: dict, prev_state: dict,
                                  agent_id: str, done: bool,
                                  agents: dict = None,
                                  actions_dict: dict = None,
                                  final_action: dict = None) -> float:
        """
        Section 5.4.4 — Reward Clipping Implementation.
        Clip to [-10, 10] for PPO. Terminal -100 is added BEFORE clipping
        but handled as done=True.

        Args:
            state: current state
            prev_state: previous state
            agent_id: which agent to compute reward for
            done: whether episode ended
            agents: dict of all agent instances (for hidden goal computation)

        Returns:
            float — clipped reward
        """
        if done and (state['stability'] < 0.2 or state['gdp'] < 0.3):
            return -10.0  # Already clipped version of -100

        role_key = AGENT_ID_TO_ROLE.get(agent_id, 'auditor')

        # Role reward (70%)
        role_r = AGENT_ROLE_REWARDS.get(role_key, lambda s, p: 0.0)(
            state, prev_state
        )

        # Hidden goal reward (30%)
        hidden_r = 0.0
        if agents and agent_id in agents:
            agent = agents[agent_id]
            if hasattr(agent, 'hidden_goal_reward'):
                hidden_r = agent.hidden_goal_reward(state, prev_state)

        # Combine: 70% role + 30% hidden
        total = self.role_weight * role_r + self.hidden_weight * hidden_r

        # Add global reward components (including policy cost and negotiation bonuses)
        global_r = compute_global_reward(state, prev_state, final_action, actions_dict)
        total += global_r * 0.5  # Assuming global reward is additive but scaled to prevent dominating

        # Add signal-based rewards
        total += self._compute_signal_rewards(state, prev_state, agent_id)

        return float(np.clip(total, self.clip_min, self.clip_max))

    def _compute_signal_rewards(self, state: dict, prev_state: dict,
                                 agent_id: str) -> float:
        """Compute the 13-signal reward stack contributions."""
        total = 0.0
        turn = state.get('turn', 0)

        # Mortality Reduction (+3.0, Role, 14T delay)
        # Trigger: Mortality drops >10% vs baseline over 14T window
        if turn >= 14:
            mort_change = state['mortality'] - prev_state.get('mortality', 0)
            if mort_change < -0.10:
                self._queue_delayed('mortality_reduction', agent_id, 3.0, 14)

        # GDP Performance (+2.0, Role, 10T delay)
        # Trigger: GDP within ±1% of target trajectory
        if abs(state['gdp'] - 1.0) < 0.01:
            self._queue_delayed('gdp_performance', agent_id, 2.0, 10)

        # Crisis Resolution (+8.0, Role, immediate)
        # Trigger: Crisis severity drops below critical threshold (only after turn 10)
        if turn >= 10 and state['stability'] > 0.7 and state['mortality'] < 0.01:
            total += 8.0

        # Inter-Agent Trust (+1.0, Social, 5T delay)
        # Trigger: Average trust off-diagonal > 0.6 for 5 consecutive turns
        trust_matrix = state.get('trust_matrix', np.eye(6))
        if isinstance(trust_matrix, np.ndarray):
            n = trust_matrix.shape[0]
            off_diag = trust_matrix[~np.eye(n, dtype=bool)]
            if np.mean(off_diag) > 0.6:
                self._queue_delayed('inter_agent_trust', agent_id, 1.0, 5)

        # Coalition Stability (+0.8, Social, immediate)
        # Trigger: Agent maintains agreed coalition past voted policy horizon
        coalition_map = state.get('coalition_map', {})
        prev_coalition = prev_state.get('coalition_map', {})
        if coalition_map.get(agent_id) == prev_coalition.get(agent_id):
            total += 0.8

        # Influence Gain (+0.5, Social, immediate)
        # Trigger: Agent successfully shifts a coalition vote outcome
        # (simplified: agent is in largest coalition)
        from collections import Counter
        if coalition_map:
            sizes = Counter(coalition_map.values())
            agent_coal = coalition_map.get(agent_id, 0)
            if sizes.get(agent_coal, 0) == max(sizes.values()):
                total += 0.5

        # Survival Bonus (+5.0, Strategic, continuous)
        # Trigger: Agent active and non-collapsed after turn 50
        if turn >= 50 and state['stability'] >= 0.2 and state['gdp'] >= 0.3:
            total += 5.0

        # Fiscal Deficit (-2.0, Penalty, 3T delay)
        # Trigger: Resources/treasury in deficit for >3 consecutive turns
        if state.get('resources', 1000) < 100:
            self._queue_delayed('fiscal_deficit', agent_id, -2.0, 3)

        # Process delayed rewards
        total += self._tick_delayed(agent_id)

        return total

    def _queue_delayed(self, signal_name: str, agent_id: str,
                        value: float, delay: int):
        """Queue a delayed reward signal."""
        if agent_id not in self.delayed_rewards:
            self.delayed_rewards[agent_id] = []
        # Avoid duplicate queueing of the same signal type in the same turn
        existing = [r for r in self.delayed_rewards[agent_id]
                    if r[2] == signal_name]
        if not existing:
            self.delayed_rewards[agent_id].append((value, delay, signal_name))

    def _tick_delayed(self, agent_id: str) -> float:
        """Decrement delay counters and deliver due rewards."""
        if agent_id not in self.delayed_rewards:
            return 0.0

        total = 0.0
        remaining = []
        for value, delay, signal_name in self.delayed_rewards[agent_id]:
            if delay <= 0:
                total += value
            else:
                remaining.append((value, delay - 1, signal_name))
        self.delayed_rewards[agent_id] = remaining
        return total
