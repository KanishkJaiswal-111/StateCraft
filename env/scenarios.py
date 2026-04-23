"""
Scenario Loader — loads and manages crisis scenarios.
Supports pandemic, economic, and disaster scenarios.
"""

import os
import yaml
from copy import deepcopy


class ScenarioLoader:
    """Loads crisis scenarios from YAML configuration files."""

    # Built-in scenario defaults (used when no YAML file is available)
    BUILTIN_SCENARIOS = {
        'pandemic': {
            'initial_state': {
                'gdp': 1.0,
                'mortality': 0.0,
                'stability': 0.75,
                'gini': 0.39,
                'public_trust': 0.62,
                'inflation': 0.02,
                'treasury': 0.80,
                'healthcare_capacity': 0.70,
                'unemployment': 0.036,
                'infection_rate': 0.001,
            },
            'crisis_events': {
                3:  {'infection_rate': 0.02, 'stability': -0.05},
                5:  {'infection_rate': 0.04, 'public_trust': -0.08, 'gdp': -0.02},
                8:  {'gdp': -0.08, 'unemployment': 0.04, 'stability': -0.10},
                12: {'mortality': 0.03, 'healthcare_capacity': -0.30},
                15: {'treasury': -0.25, 'gdp': 0.02, 'public_trust': 0.05},
                20: {'stability': -0.08, 'public_trust': -0.06},
                25: {'gdp': 0.03, 'infection_rate': 0.02},
            },
        },
        'economic': {
            'initial_state': {
                'gdp': 0.85,
                'mortality': 0.01,
                'stability': 0.60,
                'gini': 0.45,
                'public_trust': 0.50,
                'inflation': 0.08,
                'treasury': 0.40,
                'healthcare_capacity': 0.80,
                'unemployment': 0.12,
                'infection_rate': 0.0,
            },
            'crisis_events': {
                2:  {'gdp': -0.05, 'unemployment': 0.03},
                5:  {'inflation': 0.04, 'public_trust': -0.10},
                8:  {'treasury': -0.15, 'stability': -0.08},
                12: {'gini': 0.05, 'stability': -0.06},
                18: {'gdp': -0.08, 'unemployment': 0.05},
                22: {'public_trust': -0.10, 'stability': -0.10},
            },
        },
        'disaster': {
            'initial_state': {
                'gdp': 0.95,
                'mortality': 0.0,
                'stability': 0.70,
                'gini': 0.38,
                'public_trust': 0.65,
                'inflation': 0.03,
                'treasury': 0.75,
                'healthcare_capacity': 0.65,
                'unemployment': 0.05,
                'infection_rate': 0.0,
            },
            'crisis_events': {
                1:  {'mortality': 0.05, 'healthcare_capacity': -0.40, 'stability': -0.15},
                3:  {'gdp': -0.10, 'treasury': -0.20},
                6:  {'public_trust': -0.10, 'unemployment': 0.06},
                10: {'stability': -0.05, 'gini': 0.04},
                15: {'gdp': 0.03, 'public_trust': 0.05},
                20: {'healthcare_capacity': 0.10, 'stability': 0.05},
            },
        },
    }

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config'
            )
        self.config_dir = config_dir
        self.historical_dir = os.path.join(config_dir, 'historical_scenarios')

    def load_scenario(self, scenario_name: str) -> dict:
        """Load a scenario by name. Falls back to built-in if no YAML found."""
        if scenario_name in self.BUILTIN_SCENARIOS:
            return deepcopy(self.BUILTIN_SCENARIOS[scenario_name])
        raise ValueError(f"Unknown scenario: {scenario_name}")

    def load_historical_scenario(self, scenario_id: str) -> dict:
        """Load a historical scenario from YAML for validation."""
        yaml_path = os.path.join(self.historical_dir, f'{scenario_id}.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            return data
        raise FileNotFoundError(f"Historical scenario not found: {yaml_path}")

    def get_crisis_events(self, scenario: dict, turn: int) -> dict:
        """Get crisis events scheduled for a specific turn."""
        events = scenario.get('crisis_events', {})
        return events.get(turn, {})

    def list_scenarios(self) -> list:
        """List all available scenario names."""
        return list(self.BUILTIN_SCENARIOS.keys())
