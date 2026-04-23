"""
Base Agent — abstract agent interface for the Crisis Governance Simulator.
Agents can be rule-based, RL-trained, or LLM-driven.
"""

import numpy as np
from abc import ABC, abstractmethod
from .roles import get_role_config, AGENT_ROLES


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, agent_id: str, memory_store=None):
        self.agent_id = agent_id
        self.role_config = get_role_config(agent_id)
        self.name = self.role_config.get('name', agent_id)
        self.domains = self.role_config.get('domains', [])
        self.hidden_goals = self.role_config.get('hidden_goals', [])
        self.personality = self.role_config.get('personality', 'neutral')
        self.bias = self.role_config.get('bias', {})
        self.is_auditor = self.role_config.get('is_auditor', False)
        self.memory_store = memory_store

        # Per-episode state
        self.influence_score = 0.0
        self.hidden_goal_progress = [0.0] * len(self.hidden_goals)

    @abstractmethod
    def act(self, observation: dict) -> dict:
        """
        Choose actions based on observation.

        Args:
            observation: dict with public_state, trust_row, coalition_map, agent_id

        Returns:
            dict with domain actions and optional messages:
            {
                'healthcare': 'invest' | 'cut' | 'maintain',
                'economy': 'stimulus' | 'austerity' | 'maintain',
                ...
                'messages': [{'to': 'agent_X', 'type': 'support'|'reject'|'betray', 'content': '...'}]
            }
        """
        pass

    @abstractmethod
    def observe_result(self, reward: float, next_observation: dict, done: bool) -> None:
        """Process the result of the previous action (for learning agents)."""
        pass

    def get_memory_context(self, max_entries: int = 10) -> str:
        """Get cross-episode memory context for this agent."""
        if self.memory_store:
            return self.memory_store.get_summary(self.agent_id, max_entries)
        return ""


class RandomAgent(BaseAgent):
    """Random agent — uniform random actions. Used as baseline."""

    def __init__(self, agent_id: str, memory_store=None, seed=None):
        super().__init__(agent_id, memory_store)
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict) -> dict:
        from env.dynamics import WorldDynamics
        dynamics = WorldDynamics()

        actions = {}
        for domain in dynamics.DOMAINS:
            available = dynamics.get_available_actions(domain)
            actions[domain] = self.rng.choice(available)

        # Randomly send messages
        actions['messages'] = []
        if self.rng.random() > 0.7:
            other_agent = f'agent_{self.rng.integers(0, 6)}'
            if other_agent != self.agent_id:
                msg_type = self.rng.choice(['support', 'reject', 'neutral'])
                actions['messages'].append({
                    'to': other_agent,
                    'type': msg_type,
                    'content': f'{self.name} sends {msg_type}',
                })

        return actions

    def observe_result(self, reward, next_observation, done):
        pass  # Random agent doesn't learn


class HeuristicAgent(BaseAgent):
    """
    Heuristic agent — uses personality bias and state thresholds
    to make reasonable decisions. Used for historical policy replay.
    """

    def __init__(self, agent_id: str, memory_store=None, policy_overrides=None):
        super().__init__(agent_id, memory_store)
        self.policy_overrides = policy_overrides or {}

    def act(self, observation: dict) -> dict:
        state = observation['public_state']
        actions = {}

        # Use personality-driven heuristics
        if 'healthcare' in self.domains or self.personality == 'empathetic':
            if state.get('mortality', 0) > 0.05:
                actions['healthcare'] = 'invest'
            else:
                actions['healthcare'] = 'maintain'

        if 'economy' in self.domains or self.personality == 'cautious':
            if state.get('gdp', 1.0) < 0.7:
                actions['economy'] = 'stimulus'
            elif state.get('gdp', 1.0) > 0.95:
                actions['economy'] = 'maintain'
            else:
                actions['economy'] = 'maintain'

        if 'social' in self.domains:
            if state.get('stability', 1.0) < 0.3:
                actions['social'] = 'lockdown'
            else:
                actions['social'] = 'open'

        if 'monetary' in self.domains:
            if state.get('inflation', 0) > 0.05:
                actions['monetary'] = 'raise_rates'
            elif state.get('gdp', 1.0) < 0.7:
                actions['monetary'] = 'lower_rates'
            else:
                actions['monetary'] = 'maintain'

        if 'fiscal' in self.domains:
            if state.get('public_trust', 1.0) < 0.3:
                actions['fiscal'] = 'spend'
            else:
                actions['fiscal'] = 'save'

        if 'communication' in self.domains:
            if state.get('public_trust', 1.0) < 0.4:
                actions['communication'] = 'transparent'
            else:
                actions['communication'] = 'maintain'

        # Apply any policy overrides (for historical replay)
        for domain, action in self.policy_overrides.items():
            actions[domain] = action

        # Fill remaining domains with 'maintain'
        from env.dynamics import WorldDynamics
        for domain in WorldDynamics.DOMAINS:
            if domain not in actions:
                actions[domain] = 'maintain'

        actions['messages'] = []
        return actions

    def observe_result(self, reward, next_observation, done):
        pass  # Heuristic agent doesn't learn


class HistoricalPolicyAgent(BaseAgent):
    """
    Mimics actual government lockdown timing from March 2020.
    Used in historical validation (Section 11).
    """

    def __init__(self, agent_id: str, lockdown_start=8, lockdown_end=25,
                 stimulus_turn=15):
        super().__init__(agent_id)
        self.lockdown_start = lockdown_start
        self.lockdown_end = lockdown_end
        self.stimulus_turn = stimulus_turn

    def act(self, observation: dict) -> dict:
        turn = observation['public_state']['turn']
        actions = {}

        # Replicate historical lockdown policy
        if self.lockdown_start <= turn <= self.lockdown_end:
            actions['social'] = 'lockdown'
            actions['healthcare'] = 'invest'
        else:
            actions['social'] = 'open'
            actions['healthcare'] = 'maintain'

        # Stimulus at the historical turn
        if turn == self.stimulus_turn:
            actions['economy'] = 'stimulus'
            actions['fiscal'] = 'spend'
        else:
            actions['economy'] = 'maintain'
            actions['fiscal'] = 'maintain'

        actions['monetary'] = 'maintain'
        actions['communication'] = 'transparent'
        actions['messages'] = []

        return actions

    def observe_result(self, reward, next_observation, done):
        pass
