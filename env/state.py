"""
State Manager — manages the 12-field world state dict.
All state transitions go through this module.
"""

import numpy as np
from copy import deepcopy


class StateManager:
    """Manages the world state for the Crisis Governance Simulator."""

    # The 12 canonical state fields
    STATE_FIELDS = [
        'gdp', 'mortality', 'stability', 'gini', 'public_trust',
        'inflation', 'treasury', 'healthcare_capacity', 'unemployment',
        'infection_rate', 'turn', 'difficulty_tier', 'past_actions', 'lockdown_duration'
    ]

    def __init__(self, num_agents: int = 6):
        self.num_agents = num_agents
        self.state = {}
        self.trust_matrix = np.zeros((num_agents, num_agents))
        self.coalition_map = {}  # agent_id -> coalition_id
        self.state_history = []

    def initialize(self, scenario_state: dict = None) -> dict:
        """Initialize state from scenario or defaults."""
        defaults = {
            'gdp': 1.0,
            'mortality': 0.0,
            'stability': 0.75,
            'gini': 0.39,
            'public_trust': 0.62,
            'inflation': 0.02,
            'treasury': 0.80,
            'healthcare_capacity': 0.70,
            'unemployment': 0.036,
            'infection_rate': 0.0,
            'turn': 0,
            'difficulty_tier': 1,
            'past_actions': [],
            'lockdown_duration': 0,
        }

        if scenario_state:
            defaults.update(scenario_state)

        self.state = defaults
        self.state['turn'] = 0

        # Initialize trust matrix — slight positive bias on diagonal neighbors
        self.trust_matrix = np.full(
            (self.num_agents, self.num_agents), 0.5
        )
        np.fill_diagonal(self.trust_matrix, 1.0)

        # Everyone starts in their own coalition
        self.coalition_map = {f'agent_{i}': i for i in range(self.num_agents)}

        self.state_history = [deepcopy(self.state)]

        return deepcopy(self.state)

    def get_state(self) -> dict:
        """Return a copy of the current state."""
        state_copy = deepcopy(self.state)
        state_copy['trust_matrix'] = self.trust_matrix.copy()
        state_copy['coalition_map'] = deepcopy(self.coalition_map)
        return state_copy

    def apply_deltas(self, deltas: dict) -> None:
        """Apply a dict of field deltas to the state, clamping to valid ranges."""
        for field, delta in deltas.items():
            if field in self.state and isinstance(self.state[field], (int, float)):
                self.state[field] += delta

        # Clamp values to valid ranges
        self.state['gdp'] = max(0.0, self.state['gdp'])
        self.state['mortality'] = np.clip(self.state['mortality'], 0.0, 1.0)
        self.state['stability'] = np.clip(self.state['stability'], 0.0, 1.0)
        self.state['gini'] = np.clip(self.state['gini'], 0.0, 1.0)
        self.state['public_trust'] = np.clip(self.state['public_trust'], 0.0, 1.0)
        self.state['inflation'] = max(-0.1, self.state['inflation'])
        self.state['treasury'] = np.clip(self.state['treasury'], 0.0, 1.0)
        self.state['healthcare_capacity'] = np.clip(
            self.state['healthcare_capacity'], 0.0, 1.0
        )
        self.state['unemployment'] = np.clip(self.state['unemployment'], 0.0, 1.0)
        self.state['infection_rate'] = np.clip(self.state['infection_rate'], 0.0, 1.0)

    def advance_turn(self) -> None:
        """Increment turn counter and snapshot state."""
        self.state['turn'] += 1
        self.state_history.append(deepcopy(self.state))

    def update_trust(self, agent_a: int, agent_b: int, delta: float) -> None:
        """Update trust between two agents symmetrically."""
        self.trust_matrix[agent_a][agent_b] = np.clip(
            self.trust_matrix[agent_a][agent_b] + delta, 0.0, 1.0
        )
        self.trust_matrix[agent_b][agent_a] = np.clip(
            self.trust_matrix[agent_b][agent_a] + delta, 0.0, 1.0
        )

    def update_coalition(self, agent_id: str, coalition_id: int) -> None:
        """Move an agent to a new coalition."""
        self.coalition_map[agent_id] = coalition_id

    def check_collapse(self) -> bool:
        """Check if societal collapse has occurred."""
        s = self.state
        if s['mortality'] > 0.5:
            return True
        if s['gdp'] < 0.2:
            return True
        if s['stability'] < 0.05:
            return True
        if s['public_trust'] < 0.05:
            return True
        return False

    def compute_difficulty_tier(self) -> int:
        """Compute current difficulty tier [1-5] based on state."""
        s = self.state
        score = (
            (1.0 - s['mortality']) * 0.3 +
            min(s['gdp'], 1.0) * 0.3 +
            s['stability'] * 0.2 +
            s['public_trust'] * 0.2
        )
        if score > 0.8:
            return 1
        elif score > 0.6:
            return 2
        elif score > 0.4:
            return 3
        elif score > 0.2:
            return 4
        else:
            return 5
