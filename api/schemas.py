"""
Complete Step Response Schema — api/schemas.py (Section 9.3)
"""

from pydantic import BaseModel
from typing import Optional


class ResetConfig(BaseModel):
    """Configuration for resetting the environment."""
    scenario: Optional[str] = None
    episode_mode: Optional[str] = None
    num_agents: Optional[int] = None


class ActionsPayload(BaseModel):
    """Payload for submitting agent actions."""
    actions_dict: dict  # agent_id -> {domain: action, messages: [...]}


class StepResponse(BaseModel):
    """
    Complete step response schema.
    """
    state:           dict   # Full 12-field state dict
    trust_matrix:    list   # 6x6 float matrix as nested list
    coalition_graph: dict   # { nodes:[...], edges:[{a,b,weight}] }
    events:          list   # List of event dicts this turn
    actions:         dict   # Final aggregated action per domain
    messages:        list   # All negotiation messages this turn
    metrics:         dict   # Current episode metrics (16 fields)
    headline:        str    # Society Newspaper headline
    done:            bool   # Episode terminated?
    auditor_report:  dict   # Auditor's fingerprint scores + flags this turn
