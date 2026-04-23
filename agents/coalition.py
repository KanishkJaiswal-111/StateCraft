"""
Coalition Manager — tracks coalition formation, defection, and stability.
"""

from collections import Counter
from copy import deepcopy


class CoalitionManager:
    """Manages coalition tracking and defection detection."""

    def __init__(self, num_agents: int = 6):
        self.num_agents = num_agents
        self.coalition_map = {f'agent_{i}': i for i in range(num_agents)}
        self.history = [deepcopy(self.coalition_map)]
        self.defection_log = []
        self.agreement_log = []

    def update(self, new_map: dict) -> list:
        """
        Update coalition map and detect defections.

        Returns:
            list of defection events detected
        """
        defections = []

        for agent_id, new_coalition in new_map.items():
            old_coalition = self.coalition_map.get(agent_id)
            if old_coalition is not None and old_coalition != new_coalition:
                # Agent changed coalition — check if it was agreed upon
                was_agreed = any(
                    a['agents'] and agent_id in a['agents']
                    for a in self.agreement_log[-5:]  # Recent agreements
                ) if self.agreement_log else False

                if not was_agreed:
                    defection = {
                        'turn': len(self.history),
                        'agent': agent_id,
                        'from_coalition': old_coalition,
                        'to_coalition': new_coalition,
                        'was_agreed': False,
                    }
                    defections.append(defection)
                    self.defection_log.append(defection)

        self.coalition_map = deepcopy(new_map)
        self.history.append(deepcopy(new_map))

        return defections

    def add_agreement(self, agreement: dict) -> None:
        """Record a coalition agreement."""
        self.agreement_log.append(agreement)

    def get_coalition_graph(self) -> dict:
        """
        Build a coalition graph for visualization.

        Returns:
            dict with 'nodes' and 'edges'
        """
        from agents.roles import AGENT_ROLES

        nodes = []
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            role = AGENT_ROLES.get(agent_id, {})
            nodes.append({
                'id': agent_id,
                'name': role.get('name', agent_id),
                'coalition': self.coalition_map.get(agent_id, i),
            })

        edges = []
        # Agents in the same coalition get an edge
        coalition_groups = {}
        for agent_id, coalition_id in self.coalition_map.items():
            if coalition_id not in coalition_groups:
                coalition_groups[coalition_id] = []
            coalition_groups[coalition_id].append(agent_id)

        for coalition_id, members in coalition_groups.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edges.append({
                        'a': members[i],
                        'b': members[j],
                        'weight': 1.0,
                    })

        return {'nodes': nodes, 'edges': edges}

    def get_largest_coalition_size(self) -> int:
        """Get the size of the largest coalition."""
        if not self.coalition_map:
            return 0
        counter = Counter(self.coalition_map.values())
        return max(counter.values())

    def get_coalition_members(self, coalition_id: int) -> list:
        """Get all agents in a specific coalition."""
        return [
            agent_id for agent_id, cid in self.coalition_map.items()
            if cid == coalition_id
        ]
