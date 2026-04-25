"""
LLM Socket Agent Wrapper — agents/llm_socket.py

Routes agent decisions through an external LLM service over WebSocket.
Falls back to wrapped agent behavior when socket is unavailable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent

try:
    from websockets.sync.client import connect as ws_connect
except Exception:
    ws_connect = None


VALID_MSG_TYPES = {"support", "threat", "trade", "reject", "inform"}
VALID_LOCKDOWN = {"none", "advisory", "partial", "full", "emergency"}
VALID_INTEREST = {"-0.5", "-0.25", "0", "+0.25", "+0.5", "+1", "+2"}
VALID_BUDGET = {"0", "5", "15", "30", "50"}
VALID_PRIORITY = {"health", "infrastructure", "military", "services"}
VALID_FOREIGN = {"isolate", "neutral", "engage", "alliance"}
VALID_CRISIS = {"monitor", "contain", "escalate", "emergency"}


class LLMSocketAgent(BaseAgent):
    """
    Wraps a role agent and queries an external LLM server for decisions.

    Expected request:
      {
        "kind": "act" | "negotiate",
        "agent_id": "agent_x",
        "role": "...",
        "round_num": 1,
        "observation": {...}
      }

    Expected response for act:
      {
        "action": {
          "lockdown_level": "...",
          "emergency_budget": "...",
          "resource_priority": "...",
          "interest_rate": "...",
          "foreign_policy": "...",
          "crisis_response": "..."
        }
      }

    Expected response for negotiate:
      {
        "messages": [{"target":"...", "type":"...", "content":"..."}, ...]
      }
    """

    def __init__(
        self,
        agent_id: str,
        role_agent: BaseAgent,
        socket_url: str,
        timeout_seconds: float = 1.5,
        api_key: Optional[str] = None,
    ):
        super().__init__(agent_id=agent_id, role=role_agent.role)
        self.role_agent = role_agent
        self.socket_url = socket_url
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

        self.hidden_goal = getattr(role_agent, "hidden_goal", {})
        self.memory = getattr(role_agent, "memory", [])
        self.personality = getattr(role_agent, "personality", {})

    def _socket_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if ws_connect is None:
            return None

        headers = None
        if self.api_key:
            headers = [("Authorization", f"Bearer {self.api_key}")]

        try:
            with ws_connect(self.socket_url, additional_headers=headers, open_timeout=self.timeout_seconds) as ws:
                ws.send(json.dumps(payload))
                response = ws.recv(timeout=self.timeout_seconds)
                if not response:
                    return None
                parsed = json.loads(response)
                return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _is_valid_action(self, action: Dict[str, Any]) -> bool:
        if not isinstance(action, dict):
            return False
        return (
            action.get("lockdown_level") in VALID_LOCKDOWN
            and action.get("interest_rate") in VALID_INTEREST
            and action.get("emergency_budget") in VALID_BUDGET
            and action.get("resource_priority") in VALID_PRIORITY
            and action.get("foreign_policy") in VALID_FOREIGN
            and action.get("crisis_response") in VALID_CRISIS
        )

    def _sanitize_messages(self, messages: Any) -> List[Dict[str, str]]:
        if not isinstance(messages, list):
            return []
        out: List[Dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            target = str(msg.get("target", "all"))
            msg_type = str(msg.get("type", "inform"))
            content = str(msg.get("content", ""))
            if msg_type not in VALID_MSG_TYPES:
                msg_type = "inform"
            out.append({"target": target, "type": msg_type, "content": content[:300]})
        return out

    def act(self, observation: dict) -> dict:
        payload = {
            "kind": "act",
            "agent_id": self.agent_id,
            "role": self.role,
            "observation": observation,
        }
        response = self._socket_request(payload)
        if response:
            action = response.get("action", {})
            if self._is_valid_action(action):
                return action
        return self.role_agent.act(observation)

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        payload = {
            "kind": "negotiate",
            "agent_id": self.agent_id,
            "role": self.role,
            "round_num": round_num,
            "observation": state,
        }
        response = self._socket_request(payload)
        if response:
            msgs = self._sanitize_messages(response.get("messages", []))
            if msgs:
                return msgs
        return self.role_agent.negotiate(state, round_num)

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        return self.role_agent.hidden_goal_reward(state, prev_state)

    def load_memory(self, store) -> None:
        self.role_agent.load_memory(store)
        self.memory = self.role_agent.memory

    def save_memory(self, store, events: list) -> None:
        self.role_agent.save_memory(store, events)
