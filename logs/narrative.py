"""
Narrative System — logs/narrative.py (Section 8.2)
Generates 2-sentence newspaper headlines after each turn.
"""

from metrics.tracker import compute_society_score


def generate_headline(state: dict, events: list, turn: int) -> str:
    """
    Generate a 2-sentence headline after each turn.
    Template-based OR LLM call (if API available).
    TEMPLATE fallback (always works):
    """
    # TEMPLATE fallback (always works):
    if events:
        event = events[-1]
        return (
            f'{event["agent"]} {event["impact"]}. '
            f'Stability: {state["stability"]:.0%}. '
            f'Society score: {compute_society_score([state]):.0f}.'
        )
    else:
        return (
            f'Turn {turn}: Coalition holds. '
            f'GDP at {state["gdp"]:.2f}. '
            f'Public trust {state["public_trust"]:.0%}.'
        )

    # LLM call (optional, uses Claude/HF API if DEMO_MODE=True):
    # Prompt: 'Write a 2-sentence newspaper headline for this governance event: {json}'
    # Strip to <2 sentences. Cache to avoid latency.


class NarrativeSystem:
    """Manages narrative generation across an episode."""

    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.headlines = []
        self.last_headline = ""

    def generate(self, state: dict, events: list, turn: int) -> str:
        headline = generate_headline(state, events, turn)
        self.headlines.append(headline)
        self.last_headline = headline
        return headline

    def get_all_headlines(self) -> list:
        return list(self.headlines)
