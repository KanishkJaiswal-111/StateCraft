import os, sys, json, random, subprocess, threading, asyncio, shutil

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
GRPO_CKPT_DIR = os.path.join(RUN_ROOT, "grpo_checkpoints")
AUD_OUT_DIR = os.path.join(RUN_ROOT, "auditor_outputs")
LLM_OUT_DIR = os.path.join(RUN_ROOT, "llm_outputs")

GRPO_EPISODES = 2000
AUD_EPISODES_PER_SCENARIO = 20

os.makedirs(RUN_ROOT, exist_ok=True)
os.makedirs(GRPO_CKPT_DIR, exist_ok=True)
os.makedirs(AUD_OUT_DIR, exist_ok=True)
os.makedirs(LLM_OUT_DIR, exist_ok=True)

# =========================
# SETUP — FORCE FRESH CLONE
# =========================
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)

# Purge any cached modules from previous runs
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith(("training", "openenv", "metrics", "causal", "auditor", "env", "core", "agents")):
        del sys.modules[mod_name]

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "openai>=1.0.0"], check=True)

import numpy as np
import pandas as pd
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

subprocess.run([sys.executable, "verify_integration.py"], check=True)

# =========================
# 1) GRPO TRAINING (PRIMARY RL)
# =========================
from training.grpo_trainer import GRPOPipeline

pipeline = GRPOPipeline(
    config={"scenario": "pandemic", "num_episodes": GRPO_EPISODES},
    checkpoint_dir=GRPO_CKPT_DIR
)
history = pipeline.train_grpo(num_episodes=GRPO_EPISODES)

# =========================
# 2) GENERALIZATION EVAL
# =========================
from eval.generalization import run_generalization_test

gen_results = run_generalization_test(checkpoint_path=os.path.join(GRPO_CKPT_DIR, "lora_model"))
with open(os.path.join(RUN_ROOT, "generalization_results.json"), "w", encoding="utf-8") as f:
    json.dump(gen_results, f, indent=2)

# =========================
# 3) AUDITOR CLASSIFIER (env-rollout-based inference)
# =========================
from openenv.wrapper import CrisisGovernanceEnv, AGENT_IDS
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

def encode_action(a):
    return np.array([
        LOCKDOWN_MAP.get(a.get("lockdown_level","none"), 0) / 4.0,
        INTEREST_MAP.get(a.get("interest_rate","0"), 2) / 6.0,
        BUDGET_MAP.get(a.get("emergency_budget","0"), 0) / 4.0,
        PRIORITY_MAP.get(a.get("resource_priority","health"), 0) / 3.0,
        FOREIGN_MAP.get(a.get("foreign_policy","neutral"), 1) / 3.0,
        CRISIS_MAP.get(a.get("crisis_response","monitor"), 0) / 3.0,
    ], dtype=np.float32)

def collect_dataset(scenarios=("pandemic","economic","disaster"), episodes_per_scenario=20, seq_len=10, seed=42):
    """Collect auditor training data using environment rollouts with the trained GRPO policy (if available) or random fallback."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Try loading Unsloth model
    model, tokenizer = None, None
    lora_path = os.path.join(GRPO_CKPT_DIR, "lora_model")
    if os.path.exists(lora_path):
        try:
            from unsloth import FastLanguageModel
            print(f"[Auditor Data Collection] Loading trained GRPO model from {lora_path}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                lora_path, max_seq_length=1024, load_in_4bit=True
            )
            FastLanguageModel.for_inference(model)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"[Auditor Data Collection] Could not load Unsloth model: {e}. Using random fallback.")
            model = None

    from training.grpo_trainer import build_state_prompt, ROLE_NAMES, parse_llm_action

    X, y = [], []

    for scenario in scenarios:
        env = CrisisGovernanceEnv(config={"scenario": scenario})

        for ep in range(episodes_per_scenario):
            if ep % 5 == 0:
                print(f"  Collecting Auditor trajectories for {scenario}: Episode {ep}/{episodes_per_scenario}")
            rr = env.reset()
            obs = rr.observations.astype(np.float32)
            done = False
            seq_buffers = {aid: [] for aid in range(5)}

            while not done:
                state = env.state
                
                if model is not None:
                    prompts = []
                    for aid in range(5):
                        agent_id = f"agent_{aid}"
                        prompts.append(build_state_prompt(state, agent_id, ROLE_NAMES[agent_id]))
                    
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    
                    actions = np.zeros((6, 5), dtype=int)
                    for aid in range(5):
                        gen_tokens = outputs[aid][inputs.input_ids.shape[1]:]
                        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        actions[aid] = parse_llm_action(text)
                    actions[5] = [random.randint(0,4), random.randint(0,4), random.randint(0,4), random.randint(0,3), random.randint(0,3)]
                else:
                    # Random policy rollout fallback
                    actions = np.array([[random.randint(0, 4), random.randint(0, 4),
                                         random.randint(0, 4), random.randint(0, 3),
                                         random.randint(0, 3)] for _ in range(6)])

                sr = env.step(actions)
                actions_dict = sr.info.get("actions_dict", {})

                for aid in range(5):
                    ag = f"agent_{aid}"
                    a = actions_dict.get(ag, {})
                    obs_feat = obs[aid, :]
                    act_feat = encode_action(a)
                    feat = np.concatenate([obs_feat, act_feat], axis=0)
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

X, y = collect_dataset(episodes_per_scenario=AUD_EPISODES_PER_SCENARIO, seq_len=10, seed=7)

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

OPENAI_API_KEY = "OPENAI_API_KEY"
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Skipping LLM socket training.")
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
    grpo_metrics_path = os.path.join(GRPO_CKPT_DIR, "training_metrics.json")
    if os.path.exists(grpo_metrics_path):
        with open(grpo_metrics_path, "r") as f:
            grpo_metrics = json.load(f)
        grpo_df = pd.DataFrame(grpo_metrics)

        scorecard = {
            "grpo": {
                "episodes": int(len(grpo_df)),
                "final_reward": float(grpo_df["episode_reward"].iloc[-1]),
                "best_reward": float(grpo_df["episode_reward"].max()),
                "last100_reward_mean": float(grpo_df["episode_reward"].tail(100).mean()),
                "final_society_score": float(grpo_df["society_score"].iloc[-1]),
                "best_society_score": float(grpo_df["society_score"].max()),
                "last100_society_mean": float(grpo_df["society_score"].tail(100).mean()),
                "final_causal_score": float(grpo_df["causal_score"].iloc[-1]),
                "final_auditor_accuracy": float(grpo_df["auditor_accuracy"].iloc[-1]),
            },
            "generalization": gen_results,
            "auditor_classifier": {
                "overall_accuracy": aud_acc
            },
            "llm_training": llm_metrics if 'llm_metrics' in dir() else {"status": "skipped"}
        }

        scorecard_path = os.path.join(RUN_ROOT, "complete_training_scorecard.json")
        with open(scorecard_path, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2)

        print("\n===== COMPLETE TRAINING SCORECARD =====")
        print(json.dumps(scorecard, indent=2))
        print("\nSaved:", scorecard_path)
    else:
        print("No GRPO training metrics found.")
except Exception as e:
    print("Could not generate scorecard:", e)
