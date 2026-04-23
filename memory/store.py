"""
Cross-Episode Memory — memory/store.py (Section 12)

This directly satisfies Theme #2: 'beyond context memory limits'.
A rolling buffer inside one episode does NOT qualify.
A persistent external store does.

JSON file-based persistent storage.
"""

import os
import json


class MemoryStore:
    '''JSON file-based persistent memory store.'''

    def __init__(self, backend='json', path='./data/memory.json'):
        self.backend = backend
        self.path    = path
        self._store = {}
        self._load()

    def _load(self):
        """Load existing memory from file."""
        if self.backend == 'json' and os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._store = {}

    def _save(self):
        """Persist memory to file."""
        if self.backend == 'json':
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump(self._store, f, indent=2)

    def append(self, agent_id: str, event: dict) -> None:
        '''Persist a key event to the agent's memory'''
        # events are compressed summaries, not raw transcripts
        if agent_id not in self._store:
            self._store[agent_id] = []
        self._store[agent_id].append(event)
        self._save()

    def get(self, agent_id: str) -> list:
        '''Load agent's memory history at episode start'''
        return self._store.get(agent_id, [])

    def get_summary(self, agent_id: str, max_entries=10) -> str:
        '''Return formatted memory string for context injection'''
        entries = self.get(agent_id)[-max_entries:]
        return '\n'.join([f'Ep{e["episode"]}: {e["summary"]}' for e in entries])

    def clear(self, agent_id: str = None):
        """Clear memory for a specific agent or all agents."""
        if agent_id:
            self._store.pop(agent_id, None)
        else:
            self._store = {}
        self._save()

    def get_all_agents(self) -> list:
        """Get list of all agent IDs with stored memories."""
        return list(self._store.keys())
