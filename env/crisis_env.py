"""
CrisisEnv — Main simulation environment.
Orchestrates agents, state, dynamics, and scenarios per episode.
"""

import os
import yaml
import numpy as np
from copy import deepcopy

from .state import StateManager
from .scenarios import ScenarioLoader
from .dynamics import WorldDynamics


class CrisisEnv:
    """
    Crisis Governance Simulator Environment.

    Manages 6 agents governing through a multi-domain crisis.
    Each step:
      1. Agents submit actions per domain
      2. Negotiation resolves conflicts
      3. World dynamics compute state transitions
      4. Rewards are calculated
      5. Events are logged
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = self._load_default_config()

        self.config = config
        self.scenario_name = config.get('scenario', 'pandemic')
        self.num_agents = config.get('num_agents', 6)
        self.episode_mode = config.get('episode_mode', 'TRAINING')
        self.max_steps = config.get('max_steps', {}).get(self.episode_mode, 30)
        self.demo_mode = config.get('demo_mode', False)

        self.state_manager = StateManager(num_agents=self.num_agents)
        self.scenario_loader = ScenarioLoader()
        self.dynamics = WorldDynamics()

        self.state = {}
        self.scenario = None
        self.episode = 0
        self.done = False

        # Logs for metrics computation
        self.coalition_history = []
        self.agreement_log = []
        self.defection_log = []
        self.negotiation_log = []
        self.state_history_ref = []

    def _load_default_config(self) -> dict:
        """Load config from config/config.yaml."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'config.yaml'
        )
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        # Fallback defaults
        return {
            'scenario': 'pandemic',
            'num_agents': 6,
            'episode_mode': 'TRAINING',
            'max_steps': {'TRAINING': 30, 'DEMO': 200, 'STRESS_TEST': 500},
            'demo_mode': False,
            'reward_clip': 10.0,
            'promotion_threshold': 0.70,
            'promotion_window': 20,
            'initial_tier': 1,
        }

    def reset(self, config=None) -> dict:
        """
        Initialize or reset the environment for a new episode.

        Args:
            config: Optional ResetConfig to override scenario/settings.

        Returns:
            observations: dict of per-agent observations.
        """
        if config:
            if hasattr(config, 'scenario') and config.scenario:
                self.scenario_name = config.scenario
            if hasattr(config, 'episode_mode') and config.episode_mode:
                self.episode_mode = config.episode_mode
                self.max_steps = self.config.get('max_steps', {}).get(
                    self.episode_mode, 30
                )

        self.episode += 1
        self.done = False

        # Load scenario
        self.scenario = self.scenario_loader.load_scenario(self.scenario_name)

        # Initialize state
        self.state = self.state_manager.initialize(
            self.scenario.get('initial_state', {})
        )

        # Reset logs
        self.coalition_history = [deepcopy(self.state_manager.coalition_map)]
        self.agreement_log = []
        self.defection_log = []
        self.negotiation_log = []
        self.state_history_ref = self.state_manager.state_history

        # Build per-agent observations
        observations = self._build_observations()
        return observations

    def load_historical_scenario(self, scenario_id: str) -> None:
        """Load a historical scenario for validation runs."""
        historical = self.scenario_loader.load_historical_scenario(scenario_id)
        self.scenario = {
            'initial_state': historical.get('initial_state', {}),
            'crisis_events': {},
        }
        # Convert timeline to crisis_events dict
        for event in historical.get('crisis_timeline', []):
            turn = event.pop('turn')
            event.pop('event', None)
            self.scenario['crisis_events'][turn] = {
                k.replace('_delta', ''): v
                for k, v in event.items()
            }

        self.state = self.state_manager.initialize(
            self.scenario.get('initial_state', {})
        )
        self.done = False
        self.coalition_history = [deepcopy(self.state_manager.coalition_map)]
        self.state_history_ref = self.state_manager.state_history

    def step(self, actions_dict: dict) -> tuple:
        """
        Advance the simulation by one turn.

        Args:
            actions_dict: dict mapping agent_id -> {domain: action}

        Returns:
            tuple: (observations, rewards, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        turn = self.state_manager.state['turn']

        # 1. Resolve conflicting actions via negotiation/voting
        final_actions = self.dynamics.resolve_conflicting_actions(actions_dict)

        # Calculate agreement level for dependency
        economy_votes = [a.get('economy') for a in actions_dict.values() if isinstance(a, dict)]
        agreement_level = economy_votes.count(final_actions.get('economy')) / max(1, len(economy_votes))

        # 2. Compute action effects
        action_deltas = self.dynamics.compute_action_effects(final_actions, self.state_manager.state)

        # Negotiation: Dependency (stimulus effectiveness *= agreement_level)
        if final_actions.get('economy') == 'stimulus' and 'gdp' in action_deltas:
            # Assume positive GDP impact from stimulus is scaled by agreement
            if action_deltas['gdp'] > 0:
                action_deltas['gdp'] *= agreement_level

        # Update decision memory (past_actions) and lockdown_duration
        past = self.state_manager.state.get('past_actions', [])
        past.append(deepcopy(final_actions))
        if len(past) > 5:
            past.pop(0)
        self.state_manager.state['past_actions'] = past

        if final_actions.get('social') == 'lockdown':
            self.state_manager.state['lockdown_duration'] = self.state_manager.state.get('lockdown_duration', 0) + 1
        else:
            self.state_manager.state['lockdown_duration'] = 0

        # 3. Apply crisis events for this turn
        crisis_deltas = self.scenario_loader.get_crisis_events(self.scenario, turn + 1)
        for field, delta in crisis_deltas.items():
            action_deltas[field] = action_deltas.get(field, 0) + delta

        # 4. Apply natural dynamics
        natural_deltas = self.dynamics.apply_natural_dynamics(self.state_manager.state)
        for field, delta in natural_deltas.items():
            action_deltas[field] = action_deltas.get(field, 0) + delta

        # 5. Apply all deltas to state
        self.state_manager.apply_deltas(action_deltas)

        # 6. Process negotiation messages and update trust/coalitions
        messages = self._process_negotiations(actions_dict)

        # 7. Update difficulty tier
        self.state_manager.state['difficulty_tier'] = \
            self.state_manager.compute_difficulty_tier()

        # 8. Advance turn
        self.state_manager.advance_turn()

        # 9. Record coalition snapshot
        self.coalition_history.append(
            deepcopy(self.state_manager.coalition_map)
        )

        # 10. Check termination
        collapsed = self.state_manager.check_collapse()
        max_reached = self.state_manager.state['turn'] >= self.max_steps
        self.done = collapsed or max_reached

        # 11. Get updated state
        self.state = self.state_manager.get_state()

        # 12. Compute rewards
        from rewards.rewards import RewardCalculator
        reward_calc = RewardCalculator()
        rewards = reward_calc.compute_rewards(
            state=self.state,
            actions=final_actions,
            coalition_map=self.state_manager.coalition_map,
            trust_matrix=self.state_manager.trust_matrix,
            collapsed=collapsed,
            agent_actions=actions_dict
        )

        # 13. Generate headline
        from logs.narrative import generate_headline
        headline = generate_headline(self.state, [], self.state_manager.state['turn'])

        # 14. Build observations
        observations = self._build_observations()

        info = {
            'final_action': final_actions,
            'messages': messages,
            'headline': headline,
            'collapsed': collapsed,
        }

        return observations, rewards, self.done, info

    def _build_observations(self) -> dict:
        """Build per-agent observations (partial observability)."""
        observations = {}
        state = self.state_manager.state

        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            # Each agent sees: public state + own trust row + coalition map
            obs = {
                'public_state': {
                    'gdp': state['gdp'],
                    'mortality': state['mortality'],
                    'stability': state['stability'],
                    'public_trust': state['public_trust'],
                    'turn': state['turn'],
                    'difficulty_tier': state['difficulty_tier'],
                },
                'trust_row': self.state_manager.trust_matrix[i].tolist(),
                'coalition_map': deepcopy(self.state_manager.coalition_map),
                'agent_id': agent_id,
            }
            observations[agent_id] = obs

        return observations

    def _process_negotiations(self, actions_dict: dict) -> list:
        """
        Process negotiation messages embedded in agent actions.
        Updates trust matrix and coalition map based on messages.

        Returns:
            list of message dicts
        """
        messages = []

        for agent_id, actions in actions_dict.items():
            agent_msgs = actions.get('messages', [])
            for msg in agent_msgs:
                messages.append({
                    'from': agent_id,
                    'to': msg.get('to', 'all'),
                    'type': msg.get('type', 'neutral'),
                    'content': msg.get('content', ''),
                })

                # Update trust based on message type
                try:
                    from_idx = int(agent_id.split('_')[1])
                    to_idx = int(msg.get('to', 'all').split('_')[1]) \
                        if msg.get('to', 'all') != 'all' else None
                except (ValueError, IndexError):
                    continue

                if to_idx is not None:
                    if msg.get('type') == 'support':
                        self.state_manager.update_trust(from_idx, to_idx, 0.05)
                        # Check for coalition formation
                        self.agreement_log.append({
                            'turn': self.state_manager.state['turn'],
                            'from': agent_id,
                            'to': msg['to'],
                            'was_agreed': True,
                        })
                    elif msg.get('type') == 'reject':
                        self.state_manager.update_trust(from_idx, to_idx, -0.03)
                    elif msg.get('type') == 'betray':
                        self.state_manager.update_trust(from_idx, to_idx, -0.10)
                        self.defection_log.append({
                            'turn': self.state_manager.state['turn'],
                            'agent': agent_id,
                            'was_agreed': True,
                        })

        # Record negotiation round
        if messages:
            self.negotiation_log.append({
                'turn': self.state_manager.state['turn'],
                'final_round_messages': messages,
                'coalition_map': deepcopy(self.state_manager.coalition_map),
            })

        return messages
