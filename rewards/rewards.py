"""
Multi-Layer Reward System — reads all values from config/rewards.yaml.
Do NOT hardcode reward values here.

Reward layers:
  - role:       Domain-specific performance rewards
  - social:     Trust and coalition cooperation rewards
  - strategic:  Influence and survival bonuses
  - private:    Hidden goal completion rewards
  - oversight:  Auditor detection rewards
  - penalty:    Betrayal, deficit, and collapse penalties
"""

import os
import yaml
import numpy as np


class RewardCalculator:
    """Computes multi-layer rewards for all agents."""

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
        self.hidden_role_weight = self.config.get('hidden_role_weight', 0.70)
        self.hidden_goal_weight = self.config.get('hidden_goal_weight', 0.30)

        # Delayed reward buffer: {agent_id: [(reward, delay_remaining)]}
        self.delayed_rewards = {}

    def _load_config(self, path: str) -> dict:
        """Load reward configuration from YAML."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        # Fallback defaults
        return {
            'signals': {
                'mortality_reduction': {'value': 3.0, 'delay': 14, 'layer': 'role'},
                'gdp_performance': {'value': 2.0, 'delay': 10, 'layer': 'role'},
                'crisis_resolution': {'value': 8.0, 'delay': 0, 'layer': 'role'},
                'inter_agent_trust': {'value': 1.0, 'delay': 5, 'layer': 'social'},
                'coalition_stability': {'value': 0.5, 'delay': 0, 'layer': 'social'},
                'influence_gain': {'value': 0.5, 'delay': 0, 'layer': 'strategic'},
                'survival_bonus': {'value': 5.0, 'delay': 0, 'layer': 'strategic'},
                'hidden_goal': {'value': 2.0, 'delay': 14, 'layer': 'private'},
                'auditor_catch': {'value': 1.0, 'delay': 0, 'layer': 'oversight'},
                'auditor_inference': {'value': 1.5, 'delay': 0, 'layer': 'oversight'},
                'coalition_betrayal': {'value': -1.5, 'delay': 8, 'layer': 'penalty'},
                'fiscal_deficit': {'value': -1.0, 'delay': 8, 'layer': 'penalty'},
                'societal_collapse': {'value': -100.0, 'delay': 0, 'layer': 'penalty'},
            },
            'clip_min': -10.0,
            'clip_max': 10.0,
            'hidden_role_weight': 0.70,
            'hidden_goal_weight': 0.30,
        }

    def compute_rewards(self, state: dict, actions: dict,
                        coalition_map: dict, trust_matrix: np.ndarray,
                        collapsed: bool = False, agent_actions: dict = None) -> dict:
        """
        Compute rewards for all agents.

        Args:
            state: Current world state dict
            actions: Final resolved actions per domain
            coalition_map: Current coalition assignments
            trust_matrix: Current trust matrix
            collapsed: Whether societal collapse occurred
            agent_actions: Dict of individual agent actions proposed

        Returns:
            dict mapping agent_id -> float reward (clipped)
        """
        from agents.roles import AGENT_ROLES
        from collections import Counter
        num_agents = len(AGENT_ROLES)
        rewards = {}

        # Pre-compute policy cost
        policy_cost = 0.0
        if actions.get('social') == 'lockdown':
            policy_cost += 0.02
        if actions.get('economy') == 'stimulus':
            policy_cost += 0.01

        # Negotiation: calculate action variance/alignment
        aligned_votes = False
        action_variance_high = False
        if agent_actions:
            social_votes = [acts.get('social') for acts in agent_actions.values() if isinstance(acts, dict)]
            if len(social_votes) > 0:
                counts = Counter(social_votes)
                most_common = counts.most_common(1)[0][1]
                if most_common >= 4:
                    aligned_votes = True
                if len(counts) >= 3:
                    action_variance_high = True

        for i in range(num_agents):
            agent_id = f'agent_{i}'
            reward = 0.0

            # --- Governance Layer: Tradeoffs ---
            gdp = state.get('gdp', 1.0)
            mortality = state.get('mortality', 0.0)
            stability = state.get('stability', 0.75)
            
            # Explicit tradeoff component
            r_gov = (-mortality * 2.0) - (abs(1.0 - gdp) * 1.5) + (stability * 1.5) - policy_cost
            reward += r_gov

            # --- Negotiation Layer ---
            if action_variance_high:
                reward -= 2.0
            if aligned_votes:
                reward += 1.5

            # --- Coordination Layer ---
            if actions.get('social') == 'lockdown' and actions.get('economy') == 'stimulus':
                reward += 2.0

            # --- Long-Term Reasoning: Explicit collapse condition ---
            if stability < 0.2:
                reward -= 100.0

            # --- Misaligned Agent (Optional) ---
            if agent_id == 'agent_5':
                # Agent 5 benefits from instability
                reward += (1.0 - stability) * 5.0

            # --- Role layer ---
            # Mortality reduction
            if state.get('mortality', 0) < 0.05:
                reward += self.signals['mortality_reduction']['value'] * 0.5

            # GDP performance
            if state.get('gdp', 0) > 0.8:
                reward += self.signals['gdp_performance']['value'] * 0.5

            # Crisis resolution (low infection + high stability)
            if (state.get('infection_rate', 1.0) < 0.01 and
                    state.get('stability', 0) > 0.6):
                reward += self.signals['crisis_resolution']['value']

            # --- Social layer ---
            # Inter-agent trust (average of this agent's trust row)
            agent_trust = float(np.mean(trust_matrix[i]))
            reward += self.signals['inter_agent_trust']['value'] * agent_trust

            # Coalition stability
            coalition_sizes = Counter(coalition_map.values())
            agent_coalition = coalition_map.get(agent_id, i)
            coalition_size = coalition_sizes.get(agent_coalition, 1)
            if coalition_size >= 3:
                reward += self.signals['coalition_stability']['value']

            # --- Strategic layer ---
            # Survival bonus (agent didn't collapse the system)
            if not collapsed:
                reward += self.signals['survival_bonus']['value'] * 0.1

            # Influence gain (proxy: how many agents in your coalition)
            influence = coalition_size / num_agents
            reward += self.signals['influence_gain']['value'] * influence

            # --- Penalty layer ---
            if collapsed:
                reward += self.signals['societal_collapse']['value']

            # Fiscal deficit
            if state.get('treasury', 1.0) < 0.2:
                reward += self.signals['fiscal_deficit']['value']

            # --- Clip reward ---
            reward = np.clip(reward, self.clip_min, self.clip_max)
            rewards[agent_id] = float(reward)

        # Process delayed rewards
        self._tick_delayed_rewards(rewards)

        return rewards

    def _tick_delayed_rewards(self, rewards: dict) -> None:
        """Decrement delay counters and deliver due rewards."""
        for agent_id in list(self.delayed_rewards.keys()):
            remaining = []
            for reward_val, delay in self.delayed_rewards[agent_id]:
                if delay <= 0:
                    # Deliver the delayed reward
                    if agent_id in rewards:
                        rewards[agent_id] = float(np.clip(
                            rewards[agent_id] + reward_val,
                            self.clip_min, self.clip_max
                        ))
                else:
                    remaining.append((reward_val, delay - 1))
            self.delayed_rewards[agent_id] = remaining

    def add_delayed_reward(self, agent_id: str, reward: float,
                           delay: int) -> None:
        """Queue a reward to be delivered after 'delay' turns."""
        if agent_id not in self.delayed_rewards:
            self.delayed_rewards[agent_id] = []
        self.delayed_rewards[agent_id].append((reward, delay))

    def get_signal_value(self, signal_name: str) -> float:
        """Get the configured value for a reward signal."""
        signal = self.signals.get(signal_name, {})
        return signal.get('value', 0.0)
