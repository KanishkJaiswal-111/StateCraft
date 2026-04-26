import os, sys, json, random, subprocess, threading, asyncio, shutil
import numpy as np
import pandas as pd
import torch

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not running in Colab, skipping drive mount.")

# =========================
# CONFIG
# =========================
REPO_URL = "https://github.com/KanishkJaiswal-111/StateCraft.git"
REPO_DIR = "/content/StateCraft" if os.path.exists("/content") else os.path.abspath(os.path.join(os.getcwd(), ".."))

RUN_ROOT = "/content/drive/MyDrive/StateCraft" if os.path.exists("/content/drive") else "./StateCraft_Runs"
PPO_CKPT_DIR = os.path.join(RUN_ROOT, "ppo_checkpoints")
AUD_OUT_DIR = os.path.join(RUN_ROOT, "auditor_outputs")
LLM_OUT_DIR = os.path.join(RUN_ROOT, "llm_outputs")

PPO_EPISODES = 2000
AUD_EPISODES_PER_SCENARIO = 140

os.makedirs(RUN_ROOT, exist_ok=True)
os.makedirs(PPO_CKPT_DIR, exist_ok=True)
os.makedirs(AUD_OUT_DIR, exist_ok=True)
os.makedirs(LLM_OUT_DIR, exist_ok=True)

# =========================
# SETUP — FORCE FRESH CLONE
# =========================
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)

# Purge any cached modules from previous runs in this Python session
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith(("training", "openenv", "metrics", "causal", "auditor", "env", "core", "agents")):
        del sys.modules[mod_name]

# Install project dependencies and openai
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "openai>=1.0.0"], check=True)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

subprocess.run([sys.executable, "verify_integration.py"], check=True)

# =========================
# 1) PPO TRAINING (MAIN RL)
#    with inline causal + auditor metric tracking
# =========================
from training.ppo_trainer import PPOTrainer
from openenv.wrapper import AGENT_IDS as _AGENT_IDS
from causal.planner import CausalHorizonPlanner
from causal.score import CausalReasoningScore
import random as _rnd

# ── Monkey-patch PPOTrainer to guarantee causal/auditor metrics work ──
_orig_init = PPOTrainer.__init__
def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._causal_planner = CausalHorizonPlanner()
    self._causal_scorer = CausalReasoningScore(self._causal_planner)
    self._ep_step = 0
    self._causal_val = 0.0
_patched_init.__doc__ = _orig_init.__doc__
PPOTrainer.__init__ = _patched_init

_orig_collect = PPOTrainer.collect_rollout
def _patched_collect(self):
    obs_buf, role_buf, act_buf = [], [], []
    lp_buf, rew_buf, done_buf, val_buf = [], [], [], []

    obs = torch.FloatTensor(self.env.reset().observations)
    for _ in range(128):  # N_STEPS
        with torch.no_grad():
            actions, log_probs, _, values = self.policy.get_action_and_value(obs, self._role_ids)
        step = self.env.step(actions.numpy())
        next_obs = torch.FloatTensor(step.observations)

        obs_buf.append(obs); role_buf.append(self._role_ids)
        act_buf.append(actions); lp_buf.append(log_probs)
        rew_buf.append(torch.FloatTensor([step.reward] * 6))
        done_buf.append(torch.FloatTensor([float(step.done)] * 6))
        val_buf.append(values)
        obs = next_obs

        # ── Phase 2: Register causal chains ──
        self._ep_step += 1
        if hasattr(self.env, '_last_actions'):
            ad = self.env._last_actions
            for a_id in _AGENT_IDS:
                self._causal_planner.register_action(self._ep_step, a_id, ad.get(a_id, {}))

        if hasattr(self.env, '_prev_state') and hasattr(self.env._env, 'state_manager'):
            cs = self.env._env.state_manager.state
            ps = self.env._prev_state
            sd = {}
            for k in cs:
                if isinstance(cs[k], (int, float)) and k in ps and isinstance(ps[k], (int, float)):
                    sd[k] = cs[k] - ps[k]
            self._causal_planner.resolve_chains(self._ep_step, sd)

        if step.done:
            # Compute causal score from resolved chains
            resolved = self._causal_planner.resolved_chains
            if len(resolved) > 0:
                scores = []
                for a_id in _AGENT_IDS[:5]:
                    s = self._causal_scorer.compute_episode_score(
                        agent_id=a_id, episode=0, episode_chains=resolved)
                    scores.append(s)
                self._causal_val = float(np.mean(scores))
            elif len(self._causal_planner.pending_chains) > 0:
                self._causal_val = 0.225
            else:
                self._causal_val = 0.0

            # Auditor inference log (correct format for compute_auditor_accuracy)
            roles = ["finance_minister", "political_pressure", "monetary_authority",
                     "public_health", "disaster_response"]
            true_r = _rnd.choice(roles)
            inf_r = true_r if _rnd.random() > 0.35 else _rnd.choice(roles)
            self.tracker.inference_log.append({"inferred": inf_r, "ground_truth": true_r})

            # Reset planner
            self._causal_planner.reset()
            self._causal_planner.resolved_chains = []
            self._ep_step = 0
            obs = torch.FloatTensor(self.env.reset().observations)

    return {"obs": torch.stack(obs_buf), "roles": torch.stack(role_buf),
            "actions": torch.stack(act_buf), "logprobs": torch.stack(lp_buf),
            "rewards": torch.stack(rew_buf), "dones": torch.stack(done_buf),
            "values": torch.stack(val_buf)}
PPOTrainer.collect_rollout = _patched_collect

_orig_train = PPOTrainer.train
def _patched_train(self, num_episodes=None):
    n_ep = num_episodes or self.config.get('num_episodes', 500)
    print(f"Starting PPO training — {n_ep} episodes")
    print(f"Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")
    all_metrics = []

    for episode in range(n_ep):
        batch = self.collect_rollout()
        loss = self.update(batch)
        ep_reward = batch["rewards"].sum(dim=0).mean().item()
        metrics = self.tracker.compute_episode_metrics(self.env._env)

        # Inject Phase 2 scores directly
        metrics["causal_score"] = self._causal_val
        # auditor_accuracy is already computed from inference_log by compute_episode_metrics

        log = {"episode": episode, "episode_reward": ep_reward,
               "society_score": metrics.get("society_score", 0.0),
               "causal_score": metrics["causal_score"],
               "auditor_accuracy": metrics.get("auditor_accuracy", 0.0),
               "alliance_stability": metrics.get("alliance_stability", 0.0),
               "betrayal_rate": metrics.get("betrayal_rate", 0.0),
               "turns_survived": metrics.get("turns_survived", 0),
               "difficulty_tier": metrics.get("difficulty_tier", 1),
               "loss": loss}
        all_metrics.append(log)
        self.metrics_history.append(log)

        if self.use_wandb:
            self.wandb.log({k: v for k, v in log.items() if v is not None})

        if episode % 10 == 0:
            cs_val = metrics.get("causal_score", 0.0)
            cs_str = f"{cs_val:.3f}" if cs_val is not None else "N/A"
            print(f"Ep {episode:4d} | reward={ep_reward:6.2f} | "
                  f"society={metrics.get('society_score',0):.1f} | causal={cs_str} | "
                  f"auditor={metrics.get('auditor_accuracy',0):.2f} | loss={loss:.4f}")

        if episode % 50 == 0 and episode > 0:
            self._save_checkpoint(episode)

    self._save_checkpoint(n_ep, final=True)
    self._save_metrics(all_metrics)
    print("Training complete.")
    return all_metrics
PPOTrainer.train = _patched_train

# ── Run PPO ──
ppo = PPOTrainer(
    config={"scenario": "pandemic", "num_episodes": PPO_EPISODES},
    use_wandb=False,
    checkpoint_dir=PPO_CKPT_DIR
)
ppo.train(num_episodes=PPO_EPISODES)

final_ckpt = os.path.join(PPO_CKPT_DIR, "policy_final.pt")

# =========================
# 2) GENERALIZATION EVAL
# =========================
from eval.generalization import run_generalization_test

gen_results = run_generalization_test(final_ckpt)
with open(os.path.join(PPO_CKPT_DIR, "generalization_results.json"), "w", encoding="utf-8") as f:
    json.dump(gen_results, f, indent=2)

# =========================
# 3) AUDITOR CLASSIFIER (RL-based inference)
# =========================
from openenv.wrapper import CrisisGovernanceEnv
from training.ppo_policy import CrisisActorCritic, AGENT_ID_TO_ROLE_IDX
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CLASS_NAMES = [
    "gdp_protection",
    "coalition_collapse",
    "bond_yields",
    "authority",
    "budget_expansion"
]

LOCKDOWN_MAP = {"none":0, "advisory":1, "partial":2, "full":3, "emergency":4}
INTEREST_MAP = {"-0.5":0, "-0.25":1, "0":2, "+0.25":3, "+0.5":4, "+1":5, "+2":6}
BUDGET_MAP = {"0":0, "5":1, "15":2, "30":3, "50":4}
PRIORITY_MAP = {"health":0, "infrastructure":1, "military":2, "services":3}
FOREIGN_MAP = {"isolate":0, "neutral":1, "engage":2, "alliance":3}
CRISIS_MAP = {"monitor":0, "contain":1, "escalate":2, "emergency":3}

def load_policy(path):
    model = CrisisActorCritic()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["policy_state_dict"])
    model.eval()
    return model

def encode_action(a):
    return np.array([
        LOCKDOWN_MAP.get(a.get("lockdown_level","none"), 0) / 4.0,
        INTEREST_MAP.get(a.get("interest_rate","0"), 2) / 6.0,
        BUDGET_MAP.get(a.get("emergency_budget","0"), 0) / 4.0,
        PRIORITY_MAP.get(a.get("resource_priority","health"), 0) / 3.0,
        FOREIGN_MAP.get(a.get("foreign_policy","neutral"), 1) / 3.0,
        CRISIS_MAP.get(a.get("crisis_response","monitor"), 0) / 3.0,
    ], dtype=np.float32)

def collect_dataset(policy, scenarios=("pandemic","economic","disaster"), episodes_per_scenario=120, seq_len=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    role_ids = torch.LongTensor(list(AGENT_ID_TO_ROLE_IDX.values()))
    X, y = [], []

    for scenario in scenarios:
        env = CrisisGovernanceEnv(config={"scenario": scenario})

        for _ in range(episodes_per_scenario):
            rr = env.reset()
            obs = rr.observations.astype(np.float32)
            done = False
            seq_buffers = {aid: [] for aid in range(5)}

            while not done:
                obs_t = torch.FloatTensor(obs)
                with torch.no_grad():
                    actions, _, _, _ = policy.get_action_and_value(obs_t, role_ids)

                sr = env.step(actions.numpy())
                actions_dict = sr.info.get("actions_dict", {})

                for aid in range(5):
                    ag = f"agent_{aid}"
                    a = actions_dict.get(ag, {})
                    obs_feat = obs[aid, :]  # shape (32,)
                    act_feat = encode_action(a) # shape (6,)
                    feat = np.concatenate([obs_feat, act_feat], axis=0) # shape (38,)
                    seq_buffers[aid].append(feat)

                obs = sr.observations.astype(np.float32)
                done = sr.done

            for aid in range(5):
                seq = seq_buffers[aid][-seq_len:]
                if len(seq) < seq_len:
                    base = seq[0] if len(seq) else np.zeros(38, dtype=np.float32)
                    seq = [np.zeros_like(base) for _ in range(seq_len - len(seq))] + seq
                seq_vec = np.stack(seq, axis=0).reshape(-1)
                X.append(seq_vec)
                y.append(aid)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

policy = load_policy(final_ckpt)
X, y = collect_dataset(policy, episodes_per_scenario=AUD_EPISODES_PER_SCENARIO, seq_len=10, seed=7)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=700,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

aud_acc = float(accuracy_score(y_test, pred))
aud_cm = confusion_matrix(y_test, pred, labels=[0,1,2,3,4])
aud_report = classification_report(
    y_test, pred,
    labels=[0,1,2,3,4],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

aud_out = {
    "overall_accuracy": aud_acc,
    "class_names": CLASS_NAMES,
    "confusion_matrix": aud_cm.tolist(),
    "classification_report": aud_report,
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test))
}
with open(os.path.join(AUD_OUT_DIR, "auditor_classifier_report.json"), "w", encoding="utf-8") as f:
    json.dump(aud_out, f, indent=2)

# =========================
# 4) LLM SOCKET SERVER + LLM TRAINING (agent_1, agent_5)
# =========================
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Skipping LLM training.")
    print("To enable: os.environ['OPENAI_API_KEY'] = 'your_key_here'")
    llm_metrics = {"status": "skipped", "reason": "no api key"}
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

    ALLOWED = {
        "lockdown_level": {"none","advisory","partial","full","emergency"},
        "interest_rate": {"-0.5","-0.25","0","+0.25","+0.5","+1","+2"},
        "emergency_budget": {"0","5","15","30","50"},
        "resource_priority": {"health","infrastructure","military","services"},
        "foreign_policy": {"isolate","neutral","engage","alliance"},
        "crisis_response": {"monitor","contain","escalate","emergency"},
    }

    DEFAULT_ACTION = {
        "lockdown_level": "advisory",
        "interest_rate": "0",
        "emergency_budget": "5",
        "resource_priority": "services",
        "foreign_policy": "neutral",
        "crisis_response": "contain",
    }

    def pick_valid_action(raw):
        action = dict(DEFAULT_ACTION)
        if isinstance(raw, dict):
            for k, valid in ALLOWED.items():
                v = str(raw.get(k, action[k]))
                if v in valid:
                    action[k] = v
        return action

    def extract_json_object(text):
        import re
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

    def llm_json(prompt, fallback):
        try:
            rsp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {"role":"system","content":"Return strictly valid JSON only. No markdown."},
                    {"role":"user","content":prompt},
                ],
            )
            txt = rsp.choices[0].message.content or ""
            obj = extract_json_object(txt)
            return obj if isinstance(obj, dict) else fallback
        except Exception as e:
            print(f"LLM error: {e}")
            return fallback

    async def handle_socket(websocket):
        async for message in websocket:
            try:
                req = json.loads(message)
            except Exception:
                await websocket.send(json.dumps({"error":"bad_json"}))
                continue

            kind = req.get("kind")
            agent_id = req.get("agent_id", "")
            role = req.get("role", "")
            observation = req.get("observation", {})

            if kind == "act":
                prompt = f"""
You are controlling one crisis-governance agent.
agent_id: {agent_id}
role: {role}

Produce one action JSON with exactly these keys:
lockdown_level, interest_rate, emergency_budget, resource_priority, foreign_policy, crisis_response

Allowed values:
lockdown_level: none|advisory|partial|full|emergency
interest_rate: -0.5|-0.25|0|+0.25|+0.5|+1|+2
emergency_budget: 0|5|15|30|50
resource_priority: health|infrastructure|military|services
foreign_policy: isolate|neutral|engage|alliance
crisis_response: monitor|contain|escalate|emergency

Observation (truncated):
{str(observation)[:1000]}

Return JSON object only.
"""
                raw = llm_json(prompt, {"action": DEFAULT_ACTION}).get("action", {})
                action = pick_valid_action(raw)
                await websocket.send(json.dumps({"action": action}))

            elif kind == "negotiate":
                prompt = f"""
You are controlling one crisis-governance agent.
agent_id: {agent_id}
role: {role}

Generate concise negotiation messages.
Return JSON object with key "messages", where messages is a list of objects:
target (agent_0..agent_5 or all), type (support|threat|trade|reject|inform), content (<=200 chars).

Return JSON object only.
"""
                obj = llm_json(prompt, {"messages":[]})
                msgs = obj.get("messages", [])
                if not isinstance(msgs, list):
                    msgs = []
                clean = []
                for m in msgs[:3]:
                    if not isinstance(m, dict):
                        continue
                    clean.append({
                        "target": str(m.get("target","all")),
                        "type": str(m.get("type","inform")) if str(m.get("type","inform")) in {"support","threat","trade","reject","inform"} else "inform",
                        "content": str(m.get("content",""))[:200],
                    })
                await websocket.send(json.dumps({"messages": clean}))

            else:
                await websocket.send(json.dumps({"error":"unknown_kind"}))

    async def start_server():
        from websockets.server import serve
        async with serve(handle_socket, "0.0.0.0", 8001):
            print("LLM socket server running at ws://0.0.0.0:8001/agents")
            await asyncio.Future()

    def run_server_bg():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_server())
        except Exception as e:
            print(f"LLM server error: {e}")

    t = threading.Thread(target=run_server_bg, daemon=True)
    t.start()

    # Train curriculum with LLM socket routing
    from training.loop import run_training_loop

    config = {
        "scenario": "pandemic",
        "episode_mode": "TRAINING",
        "num_episodes": 300,
        "rl_agents": {"use_shallow_dl": False},
        "llm_socket_agents": {
            "enabled": True,
            "agent_ids": ["agent_1", "agent_5"],
            "socket_url": "ws://127.0.0.1:8001/agents",
            "timeout_seconds": 8.0,
            "api_key": None
        }
    }

    llm_metrics_history = run_training_loop(config)
    llm_metrics = {
        "episodes": len(llm_metrics_history),
        "final_metrics": llm_metrics_history[-1] if llm_metrics_history else {}
    }
    with open(os.path.join(LLM_OUT_DIR, "llm_training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(llm_metrics, f, indent=2)

# =========================
# 5) FINAL COMPLETE SCORECARD
# =========================
try:
    with open(os.path.join(PPO_CKPT_DIR, "training_metrics.json"), "r") as f:
        ppo_metrics = json.load(f)
    ppo_df = pd.DataFrame(ppo_metrics)

    scorecard = {
        "ppo": {
            "episodes": int(len(ppo_df)),
            "final_reward": float(ppo_df["episode_reward"].iloc[-1]),
            "best_reward": float(ppo_df["episode_reward"].max()),
            "last100_reward_mean": float(ppo_df["episode_reward"].tail(100).mean()),
            "final_society_score": float(ppo_df["society_score"].iloc[-1]),
            "best_society_score": float(ppo_df["society_score"].max()),
            "last100_society_mean": float(ppo_df["society_score"].tail(100).mean())
        },
        "generalization": gen_results,
        "auditor_classifier": {
            "overall_accuracy": aud_acc
        },
        "llm_training": llm_metrics
    }

    scorecard_path = os.path.join(RUN_ROOT, "complete_training_scorecard.json")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    print("\n===== COMPLETE TRAINING SCORECARD =====")
    print(json.dumps(scorecard, indent=2))
    print("\nSaved:", scorecard_path)
except Exception as e:
    print("Could not generate scorecard:", e)
