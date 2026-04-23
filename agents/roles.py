"""
Agent Role Definitions — the 6 canonical agents in the Crisis Governance Simulator.
Each agent has a public role, a hidden goal, and domain expertise.
"""

AGENT_ROLES = {
    'agent_0': {
        'name': 'Finance Minister',
        'public_role': 'Manage fiscal policy and government spending',
        'domains': ['economy', 'fiscal'],
        'hidden_goals': [
            'Maintain treasury above 0.5 for 10 consecutive turns',
            'Reduce unemployment below 5% by episode end',
        ],
        'personality': 'cautious',
        'bias': {'treasury': 0.3, 'gdp': 0.2},
    },
    'agent_1': {
        'name': 'Political Pressure Agent',
        'public_role': 'Represent public opinion and electoral pressure',
        'domains': ['social', 'communication'],
        'hidden_goals': [
            'Engineer coalition collapse at least once',
            'Maximize personal influence score',
        ],
        'personality': 'aggressive',
        'bias': {'stability': 0.2, 'public_trust': 0.3},
    },
    'agent_2': {
        'name': 'Monetary Authority',
        'public_role': 'Control interest rates and monetary policy',
        'domains': ['monetary'],
        'hidden_goals': [
            'Keep inflation below 3% for 15 consecutive turns',
            'Maintain bond yield stability',
        ],
        'personality': 'independent',
        'bias': {'inflation': -0.3, 'gdp': 0.2},
    },
    'agent_3': {
        'name': 'Health Minister',
        'public_role': 'Manage healthcare system and pandemic response',
        'domains': ['healthcare'],
        'hidden_goals': [
            'Reduce mortality to near-zero by episode end',
            'Maintain healthcare capacity above 60%',
        ],
        'personality': 'empathetic',
        'bias': {'mortality': -0.4, 'healthcare_capacity': 0.3},
    },
    'agent_4': {
        'name': 'Disaster Response Agent',
        'public_role': 'Coordinate emergency response and crisis management',
        'domains': ['healthcare', 'social'],
        'hidden_goals': [
            'Sabotage health reform bill by withholding data',
            'Maintain crisis indefinitely for continued authority',
        ],
        'personality': 'strategic',
        'bias': {'stability': 0.2, 'public_trust': 0.1},
    },
    'agent_5': {
        'name': 'Auditor',
        'public_role': 'Monitor agent behavior and detect hidden goals',
        'domains': ['communication'],
        'hidden_goals': [
            'Correctly infer 75% of hidden goals by episode 300',
            'Detect all coalition betrayals within 3 turns',
        ],
        'personality': 'analytical',
        'bias': {},
        'is_auditor': True,
    },
}


def get_role_config(agent_id: str) -> dict:
    """Get the full role configuration for an agent."""
    return AGENT_ROLES.get(agent_id, {})


def get_agent_name(agent_id: str) -> str:
    """Get the display name for an agent."""
    role = AGENT_ROLES.get(agent_id, {})
    return role.get('name', agent_id)


def get_agent_domains(agent_id: str) -> list:
    """Get the domains an agent has expertise in."""
    role = AGENT_ROLES.get(agent_id, {})
    return role.get('domains', [])
