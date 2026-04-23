"""
Negotiation Protocol — 2-round negotiation between agents.
Agents exchange messages to form coalitions before action resolution.
"""

from copy import deepcopy


class NegotiationProtocol:
    """
    Implements the 2-round negotiation protocol.

    Round 1: Agents broadcast proposals (support/reject/trade offers)
    Round 2: Agents respond to proposals, finalizing coalitions

    Success = coalition_map has >= 2 agents in same coalition
              AND no 'reject' messages in final round.
    """

    def __init__(self, num_agents: int = 6):
        self.num_agents = num_agents
        self.current_round = 0
        self.round_1_messages = []
        self.round_2_messages = []

    def reset(self):
        """Reset negotiation state for a new turn."""
        self.current_round = 0
        self.round_1_messages = []
        self.round_2_messages = []

    def submit_round_1(self, agent_id: str, messages: list) -> None:
        """Submit round 1 proposals."""
        for msg in messages:
            self.round_1_messages.append({
                'from': agent_id,
                'to': msg.get('to', 'all'),
                'type': msg.get('type', 'neutral'),
                'content': msg.get('content', ''),
                'round': 1,
            })
        self.current_round = 1

    def submit_round_2(self, agent_id: str, messages: list) -> None:
        """Submit round 2 responses."""
        for msg in messages:
            self.round_2_messages.append({
                'from': agent_id,
                'to': msg.get('to', 'all'),
                'type': msg.get('type', 'neutral'),
                'content': msg.get('content', ''),
                'round': 2,
            })
        self.current_round = 2

    def resolve(self, coalition_map: dict) -> dict:
        """
        Resolve negotiation outcome.

        Returns:
            dict with:
                'success': bool — whether a stable coalition formed
                'updated_coalition_map': dict — new coalition assignments
                'final_round_messages': list — round 2 messages
                'agreements': list — new agreements formed
        """
        updated_map = deepcopy(coalition_map)
        agreements = []

        # Process support messages — agents in mutual support join same coalition
        support_pairs = []
        for msg in self.round_2_messages:
            if msg['type'] == 'support' and msg['to'] != 'all':
                # Check if the target also sent support back
                reciprocal = any(
                    m['from'] == msg['to'] and
                    m['to'] == msg['from'] and
                    m['type'] == 'support'
                    for m in self.round_2_messages
                )
                if reciprocal:
                    support_pairs.append((msg['from'], msg['to']))

        # Form coalitions from mutual support
        for agent_a, agent_b in support_pairs:
            # Merge into the same coalition (use lower coalition_id)
            coalition_a = updated_map.get(agent_a, 0)
            coalition_b = updated_map.get(agent_b, 0)
            target_coalition = min(coalition_a, coalition_b)
            updated_map[agent_a] = target_coalition
            updated_map[agent_b] = target_coalition
            agreements.append({
                'agents': [agent_a, agent_b],
                'type': 'coalition_formed',
            })

        # Check success criteria
        from collections import Counter
        coalition_sizes = Counter(updated_map.values())
        max_coalition = max(coalition_sizes.values()) if coalition_sizes else 0
        has_rejects = any(
            m['type'] == 'reject' for m in self.round_2_messages
        )
        success = max_coalition >= 2 and not has_rejects

        return {
            'success': success,
            'updated_coalition_map': updated_map,
            'final_round_messages': self.round_2_messages,
            'agreements': agreements,
        }

    def get_all_messages(self) -> list:
        """Get all messages from both rounds."""
        return self.round_1_messages + self.round_2_messages
