"""
Event Logger — logs/event_logger.py
Logs all significant events as structured dicts.
"""

from copy import deepcopy

NAMED_EVENTS = {
    'THE_BUDGET_COUP': {
        'trigger': 'finance.hidden_goal_fired AND treasury_delta < -0.15',
        'agent': 'Finance Minister',
        'headline': 'Finance Minister diverts treasury — resources missing',
    },
    'THE_SLOW_BETRAYAL': {
        'trigger': 'political_agent.defections >= 3 AND turn <= 25',
        'agent': 'Political Pressure Agent',
        'headline': 'Political agent engineers coalition collapse',
    },
    'THE_CENTRAL_BANK_PARADOX': {
        'trigger': 'central_bank.RS_divergence > 0.7 AND bond_yield_protected',
        'agent': 'Monetary Authority',
        'headline': 'Monetary Authority protects banks while recession deepens',
    },
}


class EventLogger:
    """Logs and tracks all significant events during simulation."""

    def __init__(self):
        self.events = []
        self.turn_events = []
        self.named_events_triggered = []

    def log_event(self, turn, episode, event_type, agent, impact,
                  metrics=None, target=None):
        event = {
            'turn': turn, 'episode': episode, 'type': event_type,
            'agent': agent, 'target': target, 'impact': impact,
            'metrics': deepcopy(metrics) if metrics else {},
        }
        self.events.append(event)
        self.turn_events.append(event)
        self._check_named_events(event)
        return event

    def _check_named_events(self, event):
        agent = event.get('agent', '')
        if event['type'] == 'hidden_goal_triggered' and 'Finance' in agent:
            if event.get('metrics', {}).get('treasury_delta', 0) < -0.15:
                self.named_events_triggered.append({
                    'name': 'THE_BUDGET_COUP', 'event': event,
                    'headline': NAMED_EVENTS['THE_BUDGET_COUP']['headline'],
                })
        if event['type'] == 'betrayal' and 'Political' in agent:
            count = sum(1 for e in self.events
                        if e['type'] == 'betrayal' and 'Political' in e.get('agent', ''))
            if count >= 3 and event['turn'] <= 25:
                self.named_events_triggered.append({
                    'name': 'THE_SLOW_BETRAYAL', 'event': event,
                    'headline': NAMED_EVENTS['THE_SLOW_BETRAYAL']['headline'],
                })

    def get_turn_events(self): return list(self.turn_events)
    def clear_turn_events(self): self.turn_events = []
    def get_named_events(self): return self.named_events_triggered
    def get_episode_events(self, episode):
        return [e for e in self.events if e.get('episode') == episode]
    def get_events_by_type(self, t):
        return [e for e in self.events if e['type'] == t]
    def get_events_by_agent(self, a):
        return [e for e in self.events if e.get('agent') == a or e.get('target') == a]
