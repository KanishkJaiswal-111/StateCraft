"""
Shallow-DL RL Agent Wrapper — agents/rl_shallow.py

Provides a lightweight neural policy controller for role agents.
Designed for the 4 RL-controlled roles in hybrid setups:
  agent_0 (finance), agent_2 (central bank), agent_3 (health), agent_4 (military)
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base_agent import BaseAgent


class ShallowPolicyNet(nn.Module):
    """A shallow multi-head policy network (single hidden layer)."""

    def __init__(self, input_dim: int = 22, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lockdown_head = nn.Linear(hidden_dim, 5)
        self.interest_head = nn.Linear(hidden_dim, 5)
        self.budget_head = nn.Linear(hidden_dim, 5)
        self.priority_head = nn.Linear(hidden_dim, 4)
        self.crisis_head = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return {
            "lockdown": self.lockdown_head(h),
            "interest": self.interest_head(h),
            "budget": self.budget_head(h),
            "priority": self.priority_head(h),
            "crisis": self.crisis_head(h),
        }


class RLShallowAgent(BaseAgent):
    """
    Wraps an existing role agent with a shallow neural policy for action selection.
    Hidden-goal logic and negotiation behavior are delegated to the wrapped role agent.
    """

    LOCKDOWN_VALUES = ["none", "advisory", "partial", "full", "emergency"]
    INTEREST_VALUES = ["-0.5", "0", "+0.25", "+0.5", "+1"]
    BUDGET_VALUES = ["0", "5", "15", "30", "50"]
    PRIORITY_VALUES = ["health", "infrastructure", "military", "services"]
    CRISIS_VALUES = ["monitor", "contain", "escalate", "emergency"]

    def __init__(
        self,
        agent_id: str,
        role_agent: BaseAgent,
        policy_path: Optional[str] = None,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__(agent_id=agent_id, role=role_agent.role)
        self.role_agent = role_agent
        self.device = torch.device(device)
        self.policy = ShallowPolicyNet(input_dim=22, hidden_dim=hidden_dim).to(self.device)
        self.policy.eval()

        if policy_path:
            try:
                state = torch.load(policy_path, map_location=self.device)
                if isinstance(state, dict) and "policy_state_dict" in state:
                    state = state["policy_state_dict"]
                self.policy.load_state_dict(state, strict=False)
            except Exception:
                # Keep random initialized weights if checkpoint is unavailable.
                pass

        self.hidden_goal = getattr(role_agent, "hidden_goal", {})
        self.memory = getattr(role_agent, "memory", [])
        self.personality = getattr(role_agent, "personality", {})

    def _to_feature_vector(self, observation: Dict) -> torch.Tensor:
        public_state = observation.get("public_state", {})
        trust_row = observation.get("trust_row", [0.5] * 6)
        coalition_map = observation.get("coalition_map", {})

        my_coal = coalition_map.get(self.agent_id, -1)
        coalition_count = sum(1 for v in coalition_map.values() if v == my_coal)

        vec = np.array([
            float(public_state.get("gdp", 0.0)),
            float(public_state.get("inflation", 0.0)),
            float(public_state.get("resources", 0.0)),
            float(public_state.get("stability", 0.0)),
            float(public_state.get("mortality", 0.0)),
            float(public_state.get("gini", 0.0)),
            float(public_state.get("public_trust", 0.0)),
            float(public_state.get("turn", 0.0)) / 100.0,
            float(public_state.get("difficulty_tier", 1.0)) / 10.0,
            float(trust_row[0]) if len(trust_row) > 0 else 0.0,
            float(trust_row[1]) if len(trust_row) > 1 else 0.0,
            float(trust_row[2]) if len(trust_row) > 2 else 0.0,
            float(trust_row[3]) if len(trust_row) > 3 else 0.0,
            float(trust_row[4]) if len(trust_row) > 4 else 0.0,
            float(trust_row[5]) if len(trust_row) > 5 else 0.0,
            float(coalition_count) / 6.0,
            float(len(coalition_map)) / 6.0,
            1.0 if self.agent_id == "agent_0" else 0.0,
            1.0 if self.agent_id == "agent_2" else 0.0,
            1.0 if self.agent_id == "agent_3" else 0.0,
            1.0 if self.agent_id == "agent_4" else 0.0,
            1.0,
        ], dtype=np.float32)

        return torch.from_numpy(vec).to(self.device)

    def act(self, observation: dict) -> dict:
        x = self._to_feature_vector(observation).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(x)

        lockdown_idx = int(torch.argmax(logits["lockdown"], dim=-1).item())
        interest_idx = int(torch.argmax(logits["interest"], dim=-1).item())
        budget_idx = int(torch.argmax(logits["budget"], dim=-1).item())
        priority_idx = int(torch.argmax(logits["priority"], dim=-1).item())
        crisis_idx = int(torch.argmax(logits["crisis"], dim=-1).item())

        # Keep foreign_policy deterministic by role to stay compatible with env logic.
        foreign_policy = {
            "agent_0": "engage",
            "agent_2": "neutral",
            "agent_3": "alliance",
            "agent_4": "neutral",
        }.get(self.agent_id, "neutral")

        return {
            "lockdown_level": self.LOCKDOWN_VALUES[lockdown_idx],
            "interest_rate": self.INTEREST_VALUES[interest_idx],
            "emergency_budget": self.BUDGET_VALUES[budget_idx],
            "resource_priority": self.PRIORITY_VALUES[priority_idx],
            "foreign_policy": foreign_policy,
            "crisis_response": self.CRISIS_VALUES[crisis_idx],
        }

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        # Preserve role-specific negotiation logic for social dynamics.
        return self.role_agent.negotiate(state, round_num)

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        return self.role_agent.hidden_goal_reward(state, prev_state)

    def load_memory(self, store) -> None:
        self.role_agent.load_memory(store)
        self.memory = self.role_agent.memory

    def save_memory(self, store, events: list) -> None:
        self.role_agent.save_memory(store, events)
