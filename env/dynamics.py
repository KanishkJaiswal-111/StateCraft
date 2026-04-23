"""
World Dynamics — computes state transitions based on agent actions.
Handles action aggregation, crisis injection, and natural decay.
"""

import numpy as np


class WorldDynamics:
    """Computes world state transitions from agent actions."""

    # Domain-to-state mapping: which actions affect which state fields
    ACTION_EFFECTS = {
        'healthcare': {
            'invest': {'mortality': -0.01, 'healthcare_capacity': 0.05, 'treasury': -0.05},
            'cut': {'mortality': 0.005, 'healthcare_capacity': -0.03, 'treasury': 0.03},
            'maintain': {},
        },
        'economy': {
            'stimulus': {'gdp': 0.03, 'treasury': -0.08, 'unemployment': -0.01, 'inflation': 0.01},
            'austerity': {'gdp': -0.01, 'treasury': 0.05, 'unemployment': 0.02, 'inflation': -0.005},
            'maintain': {},
        },
        'social': {
            'lockdown': {'stability': -0.05, 'infection_rate': -0.02, 'gdp': -0.03, 'unemployment': 0.02},
            'open': {'stability': 0.02, 'infection_rate': 0.01, 'gdp': 0.02, 'unemployment': -0.01},
            'maintain': {},
        },
        'monetary': {
            'lower_rates': {'gdp': 0.02, 'inflation': 0.02, 'gini': 0.01},
            'raise_rates': {'gdp': -0.01, 'inflation': -0.02, 'gini': -0.005},
            'maintain': {},
        },
        'fiscal': {
            'spend': {'treasury': -0.06, 'public_trust': 0.03, 'stability': 0.02},
            'save': {'treasury': 0.04, 'public_trust': -0.02, 'stability': -0.01},
            'maintain': {},
        },
        'communication': {
            'transparent': {'public_trust': 0.04, 'stability': 0.02},
            'suppress': {'public_trust': -0.05, 'stability': -0.02},
            'maintain': {},
        },
    }

    DOMAINS = list(ACTION_EFFECTS.keys())

    def __init__(self):
        self.rng = np.random.default_rng(42)

    def compute_action_effects(self, actions: dict, state: dict = None) -> dict:
        """
        Aggregate effects from all agent actions into state deltas.

        Args:
            actions: dict mapping domain -> action_name (already resolved
                     from agent votes/negotiations)
            state: current world state

        Returns:
            dict of state field deltas
        """
        total_deltas = {}

        for domain, action in actions.items():
            if domain in self.ACTION_EFFECTS:
                effects = self.ACTION_EFFECTS[domain].get(action, {})
                for field, delta in effects.items():
                    total_deltas[field] = total_deltas.get(field, 0.0) + delta

        # Coordination: Joint action quality (lockdown without stimulus)
        if actions.get('social') == 'lockdown' and actions.get('economy') != 'stimulus':
            total_deltas['gdp'] = total_deltas.get('gdp', 0.0) - 0.05

        if state is not None:
            # Long-Term Reasoning: delayed penalty for early no lockdown
            if state.get('turn', 0) < 5 and actions.get('social') == 'open':
                total_deltas['infection_rate'] = total_deltas.get('infection_rate', 0.0) + 0.05
            
            # Anti-cheat constraints: Limits on max lockdown duration
            if actions.get('social') == 'lockdown' and state.get('lockdown_duration', 0) >= 10:
                total_deltas['stability'] = total_deltas.get('stability', 0.0) - 0.20
                total_deltas['public_trust'] = total_deltas.get('public_trust', 0.0) - 0.10

        # Anti-cheat constraints: Action cost for stimulus
        if actions.get('economy') == 'stimulus':
            total_deltas['treasury'] = total_deltas.get('treasury', 0.0) - 0.02

        return total_deltas

    def apply_natural_dynamics(self, state: dict) -> dict:
        """
        Apply natural world dynamics (decay, momentum, etc.) each turn.
        These happen regardless of agent actions.

        Returns:
            dict of state field deltas from natural dynamics
        """
        deltas = {}

        # Infection spreads naturally if > 0
        if state.get('infection_rate', 0) > 0.01:
            deltas['infection_rate'] = state['infection_rate'] * 0.05  # 5% growth
            deltas['mortality'] = state['infection_rate'] * 0.02  # mortality from infection

        # Healthcare decay under load
        if state.get('mortality', 0) > 0.05:
            deltas['healthcare_capacity'] = -0.01

        # Public trust naturally decays during crisis
        if state.get('stability', 1.0) < 0.5:
            deltas['public_trust'] = -0.01

        # Unemployment creates instability
        if state.get('unemployment', 0) > 0.10:
            deltas['stability'] = -0.005
            deltas['public_trust'] = -0.005

        # High inflation erodes trust
        if state.get('inflation', 0) > 0.05:
            deltas['public_trust'] = deltas.get('public_trust', 0) - 0.005

        # GDP recovery momentum
        if state.get('gdp', 1.0) < 0.8:
            deltas['gdp'] = 0.005  # Slow natural recovery

        # Add small stochastic noise
        for field in ['gdp', 'stability', 'public_trust']:
            noise = self.rng.normal(0, 0.005)
            deltas[field] = deltas.get(field, 0) + noise

        return deltas

    def resolve_conflicting_actions(self, agent_actions: dict) -> dict:
        """
        Resolve potentially conflicting actions from multiple agents.

        Args:
            agent_actions: dict mapping agent_id -> {domain: action}

        Returns:
            dict mapping domain -> final_action (majority vote or coalition leader)
        """
        from collections import Counter

        domain_votes = {}

        for agent_id, actions in agent_actions.items():
            for domain, action in actions.items():
                if domain == 'messages':
                    continue
                if domain not in domain_votes:
                    domain_votes[domain] = []
                domain_votes[domain].append(action)

        final_actions = {}
        for domain, votes in domain_votes.items():
            # Majority vote; ties go to 'maintain'
            counter = Counter(votes)
            most_common = counter.most_common(1)
            if most_common:
                action, count = most_common[0]
                if count > len(votes) / 2:
                    final_actions[domain] = action
                else:
                    final_actions[domain] = 'maintain'
            else:
                final_actions[domain] = 'maintain'

        return final_actions

    def get_available_actions(self, domain: str) -> list:
        """Get list of available actions for a domain."""
        if domain in self.ACTION_EFFECTS:
            return list(self.ACTION_EFFECTS[domain].keys())
        return ['maintain']
