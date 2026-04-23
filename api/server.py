"""
API Layer — api/server.py (Section 9)

FastAPI endpoints and WebSocket for the Crisis Governance Simulator.
"""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import StepResponse, ActionsPayload, ResetConfig
from env.crisis_env import CrisisEnv
from metrics.tracker import MetricsTracker
from logs.event_logger import EventLogger
from logs.narrative import NarrativeSystem

# ─────────────────────────────────────────────────────────────────────
# 9.1 FastAPI Endpoints
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(title='Crisis Governance Simulator API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
env = CrisisEnv()
tracker = MetricsTracker()
event_logger = EventLogger()
narrative = NarrativeSystem()
metrics_history = []


# POST /reset — initialize or reset environment
@app.post('/reset')
async def reset(config: ResetConfig = None):
    obs = env.reset(config)
    event_logger.clear_turn_events()
    return {'observations': obs, 'state': env.state}


# POST /step — advance one turn
@app.post('/step')
async def step(actions: ActionsPayload):
    event_logger.clear_turn_events()
    obs, rewards, done, info = env.step(actions.actions_dict)
    metrics = tracker.compute_episode_metrics(env)
    events = event_logger.get_turn_events()

    # Generate headline
    headline = narrative.generate(env.state, events, env.state['turn'])

    response = StepResponse(
        state=env.state,
        trust_matrix=env.state_manager.trust_matrix.tolist(),
        coalition_graph=tracker.get_coalition_graph(env),
        events=events,
        actions=info['final_action'],
        messages=info['messages'],
        metrics=metrics,
        headline=headline,
        done=done,
        auditor_report={},
    )

    if done:
        tracker.record_episode(metrics)
        metrics_history.append(metrics)

    return response


# GET /metrics — Current episode metrics
@app.get('/metrics')
async def get_metrics():
    return tracker.compute_episode_metrics(env)


# GET /history — all episode metrics
@app.get('/history')
async def get_history():
    return {'episodes': metrics_history}


# ─────────────────────────────────────────────────────────────────────
# 9.2 WebSocket — Real-Time Streaming
# ─────────────────────────────────────────────────────────────────────

@app.websocket('/ws/stream')
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Push state update after every env.step()
            payload = {
                'turn':         env.state_manager.state.get('turn', 0),
                'state':        env.state_manager.state,
                'trust_matrix': env.state_manager.trust_matrix.tolist(),
                'events':       event_logger.get_turn_events(),
                'headline':     narrative.last_headline,
                'metrics':      tracker.get_current_metrics(),
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.0)  # yield — demo mode uses sleep(0.5)
    except WebSocketDisconnect:
        pass
