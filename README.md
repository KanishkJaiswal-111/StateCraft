# Cognitive Society: Crisis Governance Simulator

A multi-agent simulation platform for modeling crisis governance dynamics. Six AI agents with distinct roles, hidden goals, and competing incentives must negotiate, form coalitions, and govern through pandemic, economic, and disaster scenarios.

## Architecture

```
cognitive-society/
├── main.py                         # Entry point (training/demo/API/validation)
├── requirements.txt                # Python dependencies
├── config/
│   ├── config.yaml                 # Main runtime configuration
│   ├── rewards.yaml                # All reward signal values (single source of truth)
│   └── historical_scenarios/
│       └── pandemic_march_2020.yaml  # March 2020 COVID-19 real data
├── env/
│   ├── crisis_env.py               # Main simulation environment
│   ├── state.py                    # 12-field world state manager
│   ├── scenarios.py                # Scenario loader (pandemic/economic/disaster)
│   └── dynamics.py                 # World dynamics & action resolution
├── agents/
│   ├── base_agent.py               # Agent base class + Random/Heuristic/Historical
│   ├── roles.py                    # 6 canonical agent role definitions
│   ├── negotiation.py              # 2-round negotiation protocol
│   └── coalition.py                # Coalition tracking & defection detection
├── rewards/
│   └── rewards.py                  # Multi-layer reward system (6 layers)
├── metrics/
│   ├── tracker.py                  # 16-field authoritative metrics schema
│   └── evaluation.py               # Historical validation vs March 2020
├── logs/
│   ├── event_logger.py             # Structured event logging + named events
│   └── narrative.py                # Society Newspaper headline generator
├── memory/
│   └── store.py                    # Cross-episode persistent memory (JSON)
├── api/
│   ├── server.py                   # FastAPI REST + WebSocket endpoints
│   └── schemas.py                  # Pydantic response schemas
└── data/
    └── memory.json                 # Persistent memory store
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (500 episodes, pandemic scenario)
python main.py

# Run with custom settings
python main.py --scenario economic --episodes 100

# Run demo mode (slow rendering + headlines)
python main.py --demo

# Start API server (FastAPI on port 5000)
python main.py --api

# Run historical validation (March 2020 COVID-19)
python main.py --validate
```

## The 6 Agents

| Agent | Role | Hidden Goals |
|-------|------|-------------|
| **Finance Minister** | Fiscal policy & spending | Maintain treasury > 0.5; reduce unemployment < 5% |
| **Political Pressure Agent** | Public opinion & electoral pressure | Engineer coalition collapse; maximize influence |
| **Monetary Authority** | Interest rates & monetary policy | Keep inflation < 3% for 15 turns; bond yield stability |
| **Health Minister** | Healthcare & pandemic response | Reduce mortality to near-zero; maintain healthcare > 60% |
| **Disaster Response Agent** | Emergency response & crisis management | Sabotage health reform; maintain crisis for authority |
| **Auditor** | Monitor behavior & detect hidden goals | Infer 75% of hidden goals; detect betrayals within 3 turns |

## Metrics System (16 Fields)

All metrics follow **EXACT** definitions — do not improvise formulas.

### Emergent Behavior Metrics
- **Alliance Stability** — Avg turns a coalition holds unchanged
- **Betrayal Rate** — Coalition violations per 10 turns
- **Negotiation Success** — Fraction of rounds forming stable coalitions
- **Auditor Accuracy** — Rolling 20-episode hidden goal inference accuracy
- **Trust Network Avg** — Mean off-diagonal trust matrix value

### Societal Health Score (Composite 0-100)
```
Score = (0.30 * GDP + 0.30 * Survival + 0.20 * Stability + 0.10 * Equality + 0.10 * Trust) * 100
```

## Reward Layers

| Layer | Signals | Description |
|-------|---------|-------------|
| **Role** | mortality_reduction, gdp_performance, crisis_resolution | Domain-specific performance |
| **Social** | inter_agent_trust, coalition_stability | Cooperation rewards |
| **Strategic** | influence_gain, survival_bonus | Self-interest incentives |
| **Private** | hidden_goal | Hidden goal completion |
| **Oversight** | auditor_catch, auditor_inference | Detection rewards |
| **Penalty** | coalition_betrayal, fiscal_deficit, societal_collapse | Punishments |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Initialize or reset environment |
| `POST` | `/step` | Advance one turn with agent actions |
| `GET` | `/metrics` | Current episode metrics (16 fields) |
| `GET` | `/history` | All episode metrics history |
| `WS` | `/ws/stream` | Real-time state streaming |

## Named Events (Auto-Detected)

- **THE_BUDGET_COUP** — Finance Minister diverts treasury (hidden goal + treasury delta < -0.15)
- **THE_SLOW_BETRAYAL** — Political agent engineers coalition collapse (3+ defections by turn 25)
- **THE_CENTRAL_BANK_PARADOX** — Monetary Authority protects banks while recession deepens

## Historical Validation

Compares trained agents against real March 2020 COVID-19 government responses:

| Policy | Mortality | GDP Impact | Stability | Composite |
|--------|-----------|------------|-----------|-----------|
| What govts did (Mar 2020) | Baseline (0%) | -6.1% | 52/100 | 48/100 |
| Trained Agent (ep300+) | -18% vs baseline | -4.2% | 71/100 | 74/100 |
| Random Baseline | +34% vs baseline | -9.8% | 21/100 | 18/100 |

## Configuration

Edit `config/config.yaml` for runtime settings and `config/rewards.yaml` for reward signal tuning. All reward values are read from YAML — never hardcoded.

## Cross-Episode Memory

Agents persist compressed event summaries across episodes via `memory/store.py` using a JSON file backend. This satisfies the "beyond context memory limits" requirement.
