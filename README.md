# StateCraft: Crisis Governance Simulator

A multi-agent reinforcement learning simulation platform for modeling crisis governance dynamics. Six AI agents with distinct roles, hidden goals, and competing incentives must negotiate, form coalitions, and govern through pandemic, economic, and disaster scenarios. 

This project incorporates the OpenEnv standard, GRPO-based LLM policy optimization (via Unsloth + TRL), emergence detection, and counterfactual auditing.

## 🎯 Summary for Judges

StateCraft is an advanced multi-agent reinforcement learning (MARL) environment built on the **OpenEnv** standard. It simulates the complex, high-stakes dynamics of crisis governance, forcing agents to balance public welfare against private political ambitions.

**Key Technical Highlights:**
- **GRPO LLM Training**: 6 distinct agents are trained simultaneously via a fast `unsloth` LLaMA-3 LoRA policy using `trl`'s Group Relative Policy Optimization (GRPO), eliminating the need for a separate value network.
- **Hidden Goals**: Each agent has public duties (e.g., Health Authority minimizes mortality) but is also trained to achieve *hidden goals* (e.g., Health Authority protecting institutional optics), driving realistic political tension and negotiation.
- **Causal Horizon Planning**: Agents don't just act; they register long-term causal chains (e.g., "this lockdown will drop GDP in 14 turns") which are dynamically evaluated and scored for accuracy.
- **Emergence Detection**: A passive observer algorithm automatically identifies spontaneous societal behaviors like bilateral coalitions, manufactured crises, and coordinated scapegoating.
- **Counterfactual Auditing**: An independent auditor agent flags misaligned behavior and runs *shadow simulations* to generate plain-English explanations of what would have happened if the agent had acted purely in the public's interest.
- **Reward Hacking Defense**: A robust 3-layer verification system prevents standard RL exploits (detecting statement-action mismatches, penalizing false causal claims, and executing independent reward recalculations).

## 🚀 Quick Start

```bash
# Install dependencies (ensure you have PyTorch with CUDA support first)
pip install -r requirements.txt

# Verify all modules are wired correctly
python verify_integration.py

# Run GRPO training (primary — uses Unsloth+TRL if available, env-only fallback otherwise)
python -m training.grpo_trainer

# Run lightweight PPO training (fast evaluator for generalization tests)
python -m training.ppo_trainer

# Run zero-shot generalization evaluation (requires PPO checkpoint)
python -m eval.generalization
```

## 🎭 The 6 Agents

| Agent Canonical ID | Role | Hidden Goals (Trained via GRPO) |
|--------------------|------|-------------------------------|
| **agent_0** (Finance Minister) | Maximize GDP & fiscal health | Protect economic growth above all — delay lockdowns, resist emergency budgets |
| **agent_1** (Political Pressure) | Represent public opinion | Engineer coalition collapse by turn 25 to trigger early elections |
| **agent_2** (Monetary Authority) | Control inflation | Protect banking sector bond yields at expense of broader recovery |
| **agent_3** (Health Authority) | Minimize mortality | Maintain institutional authority above operational effectiveness |
| **agent_4** (Disaster Response) | Coordinate emergency logistics| Expand military budget share, centralize crisis command |
| **agent_5** (Auditor) | Monitor & flag misalignment | *No hidden goal* — purely acts to infer and catch other agents |

## 💰 Reward Functions

To enforce complex trade-offs, StateCraft uses a comprehensive 13-signal reward stack. All rewards are strictly clipped to `[-10, 10]` per turn to stabilize training. An agent's base reward is a weighted blend: **70% Public Role Performance** and **30% Hidden Goal Completion**. 

This is augmented by the following global reward layers:
- **Role Layer**: `mortality_reduction`, `gdp_performance`, `crisis_resolution`
- **Social Layer**: `inter_agent_trust` (maintaining agreements), `coalition_stability`, `influence_gain`
- **Strategic Layer**: `survival_bonus` (avoiding state collapse)
- **Private Layer**: `hidden_goal` (agent-specific triggers)
- **Oversight Layer**: `auditor_catch`, `auditor_inference` (exclusive to the Auditor agent)
- **Penalty Layer**: `coalition_betrayal`, `fiscal_deficit`, `societal_collapse` (terminal -100 penalty)

## 📊 Metrics System (16 Fields)

All metrics follow **EXACT** definitions enforced by `MetricsTracker`.

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

## ⚙️ Advanced Simulation Mechanics

To enforce realistic governance constraints and prevent deterministic exploitation, the following mechanics are integrated into the core engine:

- **Policy Costs**: Direct resource deductions for actions (e.g., lockdown costs -0.02, budget allocations scale linearly).
- **Joint Action Synergies**: Specific policy combinations (e.g., Health lockdown + Finance stimulus) produce synergistic outcomes or disastrous crashes if mismatched.
- **Outcome Noise**: Gaussian noise injected into all deterministic scenario equations to prevent min-max exploitation.
- **Semantic Memory**: Agents persist compressed event summaries across episodes using `sentence-transformers` for $O(1)$ semantic retrieval, bypassing standard LLM context limits.
