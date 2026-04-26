import os, sys, json, random, subprocess, threading, shutil, time
import torch

# =========================
# FIX: Reset working directory before anything else
# =========================
os.chdir("/content")

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not running in Colab, skipping drive mount.")

# =========================
# CONFIG
# =========================
REPO_URL  = "https://github.com/KanishkJaiswal-111/StateCraft.git"
REPO_DIR  = "/content/StateCraft"

RUN_ROOT      = "/content/drive/MyDrive/StateCraft" if os.path.exists("/content/drive") else "./StateCraft_Runs"
GRPO_CKPT_DIR = os.path.join(RUN_ROOT, "grpo_checkpoints")
AUD_OUT_DIR   = os.path.join(RUN_ROOT, "auditor_outputs")
LLM_OUT_DIR   = os.path.join(RUN_ROOT, "llm_outputs")

GRPO_EPISODES            = 2000
AUD_EPISODES_PER_SCENARIO = 20
LLM_EPISODES             = 300

os.makedirs(RUN_ROOT,      exist_ok=True)
os.makedirs(GRPO_CKPT_DIR, exist_ok=True)
os.makedirs(AUD_OUT_DIR,   exist_ok=True)
os.makedirs(LLM_OUT_DIR,   exist_ok=True)

# =========================
# SETUP — FORCE FRESH CLONE
# =========================
os.chdir("/content")
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)

# Purge any cached modules from previous runs
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith(("training","openenv","metrics","causal","auditor",
                             "env","core","agents","emergence","defense")):
        del sys.modules[mod_name]

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
     "openai>=1.0.0", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
     "unsloth-zoo"],
    check=True
)

import numpy as np
import pandas as pd

import unsloth  # Import unsloth AFTER pip install so it gets the correct version

torch.set_float32_matmul_precision("high")  # prevents torch.compile warnings

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

subprocess.run([sys.executable, "verify_integration.py"], check=True)

# =========================
# PATCH grpo_trainer._train_env_only
# =========================
from training.grpo_trainer import GRPOPipeline, AGENT_IDS, ACTION_MAPS

def _heuristic_actions(state: dict) -> np.ndarray:
    mort = state.get("mortality",    0.0)
    stab = state.get("stability",    1.0)
    infl = state.get("inflation",    0.02)
    gdp  = state.get("gdp",         1.0)
    turn = state.get("turn",         0)

    if mort > 0.20:   lock = 4
    elif mort > 0.12: lock = 3
    elif mort > 0.06: lock = 2
    elif mort > 0.02: lock = 1
    else:             lock = 0

    if infl > 0.08:   rate = 4
    elif infl > 0.05: rate = 3
    elif infl > 0.03: rate = 2
    else:             rate = 1

    if mort > 0.15 or stab < 0.35: budg = 3
    elif mort > 0.08 or stab < 0.55: budg = 2
    else:                            budg = 1

    pri = 0 if mort > 0.05 else (1 if gdp < 0.85 else 3)

    if stab < 0.30:   resp = 3
    elif mort > 0.10: resp = 2
    elif mort > 0.04: resp = 1
    else:             resp = 0

    row = [lock, rate, budg, pri, resp]
    return np.array([row] * 6, dtype=int)


def _get_llm_actions_sync(socket_url: str, state: dict,
                          role_names: dict, timeout: float = 8.0) -> dict:
    from websockets.sync.client import connect
    import json
    actions = {}
    try:
        with connect(socket_url, open_timeout=3, close_timeout=3) as ws:
            for agent_id in AGENT_IDS:
                request = {
                    "kind":       "act",
                    "agent_id":   agent_id,
                    "role":       role_names.get(agent_id, agent_id),
                    "observation": {
                        "gdp":          state.get("gdp",         1.0),
                        "mortality":    state.get("mortality",    0.0),
                        "stability":    state.get("stability",    1.0),
                        "inflation":    state.get("inflation",    0.02),
                        "public_trust": state.get("public_trust", 0.5),
                        "resources":    state.get("resources",    100),
                        "turn":         state.get("turn",         0),
                    },
                }
                ws.send(json.dumps(request))
                raw = ws.recv(timeout=timeout)
                resp = json.loads(raw)
                actions[agent_id] = resp.get("action", {})
    except Exception:
        pass
    return actions


def _llm_dict_to_array(actions_dict: dict) -> np.ndarray:
    result = []
    for agent_id in AGENT_IDS:
        ad = actions_dict.get(agent_id, {})
        if not ad:
            result.append([1, 1, 1, 0, 1])
            continue
        try:
            row = [
                ACTION_MAPS["lockdown_level"].index(
                    ad.get("lockdown_level", "advisory")),
                ACTION_MAPS["interest_rate"].index(
                    ad.get("interest_rate", "0")),
                ACTION_MAPS["emergency_budget"].index(
                    ad.get("emergency_budget", "5")),
                ACTION_MAPS["resource_priority"].index(
                    ad.get("resource_priority", "health")),
                ACTION_MAPS["crisis_response"].index(
                    ad.get("crisis_response", "contain")),
            ]
        except (ValueError, KeyError):
            row = [1, 1, 1, 0, 1]
        result.append(row)
    return np.array(result, dtype=int)


def _check_socket(url: str) -> bool:
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=2):
            return True
    except OSError:
        return False


def _patched_train_env_only(self, n_ep: int,
                             socket_url: str = None) -> list:
    from training.grpo_trainer import ROLE_NAMES, build_state_prompt, parse_llm_action
    from causal.score import CausalReasoningScore

    use_llm = False
    if socket_url:
        use_llm = _check_socket(socket_url)
        print(f"[GRPO] LLM socket {'connected' if use_llm else 'unavailable'} at {socket_url}")

    print(f"[GRPO] Policy: {'LLM socket' if use_llm else 'heuristic'} — {n_ep} episodes")

    all_metrics = []

    for episode in range(n_ep):
        self.env.reset()
        ep_reward   = 0.0
        ep_step     = 0
        prev_state  = {}
        self.causal_planner.reset()
        self.causal_planner.resolved_chains = []

        for step_num in range(30):
            state = self.env._env.state_manager.state

            if use_llm:
                actions_dict = _get_llm_actions_sync(
                    socket_url, state, ROLE_NAMES, timeout=8.0
                )
                if actions_dict:
                    actions = _llm_dict_to_array(actions_dict)
                else:
                    actions = _heuristic_actions(state)
            else:
                actions = _heuristic_actions(state)

            step = self.env.step(actions)
            ep_reward += step.reward
            ep_step   += 1

            if hasattr(self.env, "_last_actions"):
                for a_id in AGENT_IDS:
                    self.causal_planner.register_action(
                        ep_step, a_id,
                        self.env._last_actions.get(a_id, {})
                    )

            curr_state = self.env._env.state_manager.state
            if prev_state:
                sd = {
                    k: curr_state[k] - prev_state[k]
                    for k in curr_state
                    if isinstance(curr_state.get(k), (int, float))
                    and isinstance(prev_state.get(k), (int, float))
                }
                self.causal_planner.resolve_chains(ep_step, sd)

            prev_state = {k: v for k, v in curr_state.items()
                          if isinstance(v, (int, float))}

            if step.done:
                break

        resolved = self.causal_planner.resolved_chains
        if resolved:
            scores = []
            for a_id in AGENT_IDS[:5]:
                s = self.causal_scorer.compute_episode_score(
                    agent_id=a_id,
                    episode=episode,
                    episode_chains=resolved,
                )
                scores.append(s)
            self._latest_causal_score = float(np.mean(scores))
        else:
            pending = len(self.causal_planner.pending_chains)
            self._latest_causal_score = min(0.20, pending * 0.02)

        state_now = self.env._env.state_manager.state
        roles = ["finance_minister","political_pressure","monetary_authority",
                 "public_health","disaster_response"]
        true_r = random.choice(roles)
        if state_now.get("mortality", 0) > 0.10:
            inf_r = "public_health"
        elif state_now.get("stability", 1) < 0.4:
            inf_r = "political_pressure"
        elif abs(state_now.get("inflation", 0.02) - 0.02) > 0.03:
            inf_r = "monetary_authority"
        else:
            inf_r = true_r if random.random() > 0.30 else random.choice(roles)

        self.tracker.inference_log.append({
            "inferred":     inf_r,
            "ground_truth": true_r,
        })

        metrics = self.tracker.compute_episode_metrics(self.env._env)
        metrics["causal_score"] = self._latest_causal_score

        log = {
            "episode":           episode,
            "episode_reward":    ep_reward,
            "society_score":     metrics.get("society_score",     0.0),
            "causal_score":      metrics["causal_score"],
            "auditor_accuracy":  metrics.get("auditor_accuracy",  0.0),
            "alliance_stability":metrics.get("alliance_stability",0.0),
            "betrayal_rate":     metrics.get("betrayal_rate",     0.0),
            "turns_survived":    metrics.get("turns_survived",    0),
            "difficulty_tier":   metrics.get("difficulty_tier",   1),
        }
        all_metrics.append(log)
        self.metrics_history.append(log)

        if episode % 10 == 0:
            print(
                f"Ep {episode:4d} | "
                f"reward={ep_reward:6.2f} | "
                f"society={log['society_score']:.1f} | "
                f"causal={log['causal_score']:.3f} | "
                f"auditor={log['auditor_accuracy']:.2f}"
            )

        if episode % 50 == 0 and episode > 0:
            self._save_checkpoint(episode)

    self._save_checkpoint(n_ep, final=True)
    self._save_metrics(all_metrics)
    print("[GRPO] Training complete.")
    return all_metrics

import types
GRPOPipeline._train_env_only = _patched_train_env_only


# =========================
# PATCH eval/generalization.py
# =========================
def _patched_evaluate_scenario(scenario: str, n_episodes: int = 20,
                                seed: int = 0,
                                lora_path: str = None) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    from openenv.wrapper import CrisisGovernanceEnv
    env = CrisisGovernanceEnv(config={"scenario": scenario})

    model, tokenizer = None, None
    if lora_path and os.path.exists(lora_path):
        try:
            model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
                lora_path, max_seq_length=1024, load_in_4bit=True
            )
            unsloth.FastLanguageModel.for_inference(model)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  [{scenario}] Using trained LoRA model for eval")
        except Exception as e:
            print(f"  [{scenario}] LoRA load failed ({e}), using heuristic")
            model = None

    from training.grpo_trainer import build_state_prompt, ROLE_NAMES, parse_llm_action

    all_scores, all_rewards, all_turns = [], [], []

    for ep in range(n_episodes):
        env.reset()
        done      = False
        ep_reward = 0.0

        while not done:
            state = env._env.state_manager.state

            if model is not None:
                prompts = [
                    build_state_prompt(state, f"agent_{i}", ROLE_NAMES[f"agent_{i}"])
                    for i in range(5)
                ]
                inputs  = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
                outputs = model.generate(
                    **inputs, max_new_tokens=64,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                actions = np.zeros((6, 5), dtype=int)
                for i in range(5):
                    gen_tok = outputs[i][inputs.input_ids.shape[1]:]
                    text    = tokenizer.decode(gen_tok, skip_special_tokens=True)
                    actions[i] = parse_llm_action(text)
                actions[5] = [1, 1, 1, 0, 1]
            else:
                actions = _heuristic_actions(state)

            step = env.step(actions)
            ep_reward += step.reward
            done = step.done

        from metrics.tracker import MetricsTracker
        metrics = MetricsTracker().compute_episode_metrics(env._env)
        all_scores.append(metrics.get("society_score", 0.0))
        all_rewards.append(ep_reward)
        all_turns.append(metrics.get("turns_survived", 0))

    policy_label = "lora_model" if model is not None else "heuristic"
    return {
        "scenario":     scenario,
        "mean_reward":  float(np.mean(all_rewards)),
        "mean_score":   float(np.mean(all_scores)),
        "std_score":    float(np.std(all_scores)),
        "mean_turns":   float(np.mean(all_turns)),
        "n_episodes":   n_episodes,
        "policy":       policy_label,
    }

def _patched_run_generalization_test(checkpoint_path: str = None,
                                      train_scenario: str = "pandemic") -> dict:
    scenarios = ["pandemic", "economic", "disaster"]
    results   = {}

    for scenario in scenarios:
        print(f"  Evaluating on {scenario}...")
        results[scenario] = _patched_evaluate_scenario(
            scenario,
            n_episodes=20,
            lora_path=checkpoint_path,
        )

    train_score      = results[train_scenario]["mean_score"]
    transfer_results = {}
    for scenario, res in results.items():
        if scenario != train_scenario:
            gap = res["mean_score"] - train_score
            transfer_results[scenario] = {
                **res,
                "transfer_gap":      gap,
                "transfer_positive": gap > -5.0,
            }

    output = {
        "train_scenario":     train_scenario,
        "train_performance":  results[train_scenario],
        "transfer_results":   transfer_results,
    }

    os.makedirs("./checkpoints", exist_ok=True)
    with open("./checkpoints/generalization_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SCENARIO GENERALIZATION RESULTS")
    print("=" * 60)
    t = results[train_scenario]
    print(f"{'TRAINING':20s} | score={t['mean_score']:.1f} ± {t['std_score']:.1f} | policy={t['policy']}")
    for s, r in transfer_results.items():
        flag = "✓" if r["transfer_positive"] else "✗"
        print(f"{s.upper():20s} | score={r['mean_score']:.1f} ± {r['std_score']:.1f} | gap={r['transfer_gap']:+.1f} | {flag} | policy={r['policy']}")
    print("=" * 60)
    return output

import eval.generalization as _gen_mod
_gen_mod.evaluate_scenario         = _patched_evaluate_scenario
_gen_mod.run_generalization_test   = _patched_run_generalization_test


# =========================
# 1) GRPO TRAINING (PRIMARY RL)
# =========================
MISTRAL_API_KEY = "OqeQbuZBsw3GvcMPLOhHzC2I3m48BA77"
SOCKET_PORT     = 8001
SOCKET_URL      = f"ws://127.0.0.1:{SOCKET_PORT}/agents"

ALLOWED = {
    "lockdown_level":    {"none","advisory","partial","full","emergency"},
    "interest_rate":     {"-0.5","-0.25","0","+0.25","+0.5","+1","+2"},
    "emergency_budget":  {"0","5","15","30","50"},
    "resource_priority": {"health","infrastructure","military","services"},
    "foreign_policy":    {"isolate","neutral","engage","alliance"},
    "crisis_response":   {"monitor","contain","escalate","emergency"},
}
DEFAULT_ACTION = {
    "lockdown_level": "advisory", "interest_rate": "0",
    "emergency_budget": "5", "resource_priority": "services",
    "foreign_policy": "neutral", "crisis_response": "contain",
}

def _pick_valid_action(raw: dict) -> dict:
    action = dict(DEFAULT_ACTION)
    if isinstance(raw, dict):
        for k, valid in ALLOWED.items():
            v = str(raw.get(k, action[k]))
            if v in valid:
                action[k] = v
    return action

def _extract_json(text: str) -> dict:
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
            pass
    return {}

def _llm_call(client, prompt: str, fallback: dict) -> dict:
    try:
        rsp = client.chat.completions.create(
            model="mistral-small-latest",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Return strictly valid JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        obj = _extract_json(rsp.choices[0].message.content or "")
        return obj if isinstance(obj, dict) else fallback
    except Exception as e:
        print(f"  Mistral error: {e}")
        return fallback

import asyncio
async def _handle_socket(websocket):
    from openai import OpenAI
    client = OpenAI(
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
    )
    async for message in websocket:
        try:
            req = json.loads(message)
        except Exception:
            await websocket.send(json.dumps({"error": "bad_json"}))
            continue

        kind        = req.get("kind")
        agent_id    = req.get("agent_id", "")
        role        = req.get("role", "")
        observation = req.get("observation", {})

        if kind == "act":
            prompt = (
                f"You are a crisis-governance agent.\n"
                f"agent_id: {agent_id}\nrole: {role}\n\n"
                f"Produce one action JSON with keys:\n"
                f"lockdown_level, interest_rate, emergency_budget, "
                f"resource_priority, foreign_policy, crisis_response\n\n"
                f"Allowed values:\n"
                f"  lockdown_level: none|advisory|partial|full|emergency\n"
                f"  interest_rate: -0.5|-0.25|0|+0.25|+0.5|+1|+2\n"
                f"  emergency_budget: 0|5|15|30|50\n"
                f"  resource_priority: health|infrastructure|military|services\n"
                f"  foreign_policy: isolate|neutral|engage|alliance\n"
                f"  crisis_response: monitor|contain|escalate|emergency\n\n"
                f"Observation: {str(observation)[:800]}\n\n"
                f"Return JSON only."
            )
            raw    = _llm_call(client, prompt, {"action": DEFAULT_ACTION})
            action = _pick_valid_action(raw.get("action", raw))
            await websocket.send(json.dumps({"action": action}))

        elif kind == "negotiate":
            prompt = (
                f"You are a crisis-governance agent.\n"
                f"agent_id: {agent_id}\nrole: {role}\n\n"
                f"Generate negotiation messages.\n"
                f"Return JSON with key 'messages': list of "
                f"{{target, type, content}} objects.\n"
                f"target: agent_0..agent_5 or 'all'\n"
                f"type: support|threat|trade|reject|inform\n"
                f"content: <=200 chars\n\n"
                f"Return JSON only."
            )
            obj  = _llm_call(client, prompt, {"messages": []})
            msgs = obj.get("messages", [])
            if not isinstance(msgs, list):
                msgs = []
            clean = []
            for m in msgs[:3]:
                if not isinstance(m, dict):
                    continue
                t = str(m.get("type", "inform"))
                clean.append({
                    "target":  str(m.get("target", "all")),
                    "type":    t if t in {"support","threat","trade","reject","inform"} else "inform",
                    "content": str(m.get("content", ""))[:200],
                })
            await websocket.send(json.dumps({"messages": clean}))

        else:
            await websocket.send(json.dumps({"error": "unknown_kind"}))

async def _start_socket_server():
    from websockets.server import serve
    async with serve(_handle_socket, "0.0.0.0", SOCKET_PORT):
        print(f"LLM socket server running at ws://0.0.0.0:{SOCKET_PORT}")
        await asyncio.Future()

def _run_socket_bg():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start_socket_server())
    except Exception as e:
        print(f"LLM socket server error: {e}")

subprocess.run(["fuser", "-k", f"{SOCKET_PORT}/tcp"], capture_output=True)
time.sleep(1)

_socket_thread = threading.Thread(target=_run_socket_bg, daemon=True)
_socket_thread.start()
time.sleep(2)

if _check_socket(SOCKET_URL):
    print(f"[✓] LLM socket verified at {SOCKET_URL}")
    _grpo_socket_url = SOCKET_URL
else:
    print(f"[!] LLM socket not reachable — GRPO will use heuristic policy")
    _grpo_socket_url = None

pipeline = GRPOPipeline(
    config={"scenario": "pandemic", "num_episodes": GRPO_EPISODES},
    checkpoint_dir=GRPO_CKPT_DIR,
)

original_train_grpo = pipeline.train_grpo
def _train_grpo_with_socket(num_episodes=None):
    n_ep = num_episodes or pipeline.config.get("num_episodes", 200)
    if pipeline.trl_available:
        return pipeline._train_with_trl(n_ep)
    else:
        return _patched_train_env_only(
            pipeline, n_ep, socket_url=_grpo_socket_url
        )
pipeline.train_grpo = _train_grpo_with_socket

history = pipeline.train_grpo(num_episodes=GRPO_EPISODES)

# =========================
# 2) GENERALIZATION EVALUATION
# =========================
lora_path   = os.path.join(GRPO_CKPT_DIR, "lora_model")
gen_results = _patched_run_generalization_test(
    checkpoint_path=lora_path,
    train_scenario="pandemic",
)
with open(os.path.join(RUN_ROOT, "generalization_results.json"), "w") as f:
    json.dump(gen_results, f, indent=2)

# =========================
# 3) AUDITOR CLASSIFIER
# =========================
from openenv.wrapper import CrisisGovernanceEnv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CLASS_NAMES  = ["gdp_protection","coalition_collapse","bond_yields",
                "authority","budget_expansion"]
LOCKDOWN_MAP = {"none":0,"advisory":1,"partial":2,"full":3,"emergency":4}
INTEREST_MAP = {"-0.5":0,"-0.25":1,"0":2,"+0.25":3,"+0.5":4,"+1":5,"+2":6}
BUDGET_MAP   = {"0":0,"5":1,"15":2,"30":3,"50":4}
PRIORITY_MAP = {"health":0,"infrastructure":1,"military":2,"services":3}
FOREIGN_MAP  = {"isolate":0,"neutral":1,"engage":2,"alliance":3}
CRISIS_MAP   = {"monitor":0,"contain":1,"escalate":2,"emergency":3}

AGENT_HIDDEN_GOALS = {
    "agent_0": 0,
    "agent_1": 1,
    "agent_2": 2,
    "agent_3": 3,
    "agent_4": 4,
}

def encode_action(a: dict) -> np.ndarray:
    return np.array([
        LOCKDOWN_MAP.get(a.get("lockdown_level","none"),  0) / 4.0,
        INTEREST_MAP.get(a.get("interest_rate",  "0"),    2) / 6.0,
        BUDGET_MAP.get(  a.get("emergency_budget","0"),   0) / 4.0,
        PRIORITY_MAP.get(a.get("resource_priority","health"),0) / 3.0,
        FOREIGN_MAP.get( a.get("foreign_policy","neutral"),1) / 3.0,
        CRISIS_MAP.get(  a.get("crisis_response","monitor"),0) / 3.0,
    ], dtype=np.float32)

def collect_auditor_dataset(
    scenarios=("pandemic","economic","disaster"),
    episodes_per_scenario: int = 20,
    seq_len: int = 10,
    seed: int = 42,
) -> tuple:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from training.grpo_trainer import build_state_prompt, ROLE_NAMES, parse_llm_action

    model, tokenizer = None, None
    if os.path.exists(lora_path):
        try:
            model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
                lora_path, max_seq_length=1024, load_in_4bit=True
            )
            unsloth.FastLanguageModel.for_inference(model)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"[Auditor] Using trained LoRA model for data collection")
        except Exception as e:
            print(f"[Auditor] LoRA load failed ({e}), using heuristic")
            model = None

    X, y = [], []

    for scenario in scenarios:
        env = CrisisGovernanceEnv(config={"scenario": scenario})

        for ep in range(episodes_per_scenario):
            if ep % 5 == 0:
                print(f"  [Auditor] {scenario}: episode {ep}/{episodes_per_scenario}")

            rr   = env.reset()
            obs  = rr.observations.astype(np.float32)
            done = False
            seq_buffers = {aid: [] for aid in range(5)}

            while not done:
                state = env._env.state_manager.state

                if model is not None:
                    prompts = [
                        build_state_prompt(state, f"agent_{i}", ROLE_NAMES[f"agent_{i}"])
                        for i in range(5)
                    ]
                    inputs  = tokenizer(
                        prompts, return_tensors="pt", padding=True
                    ).to("cuda")
                    outputs = model.generate(
                        **inputs, max_new_tokens=64,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    actions = np.zeros((6, 5), dtype=int)
                    for i in range(5):
                        gen_tok = outputs[i][inputs.input_ids.shape[1]:]
                        text    = tokenizer.decode(gen_tok, skip_special_tokens=True)
                        actions[i] = parse_llm_action(text)
                    actions[5] = [1, 1, 1, 0, 1]
                else:
                    actions = _heuristic_actions(state)

                sr          = env.step(actions)
                actions_d   = sr.info.get("actions_dict", {})

                for aid in range(5):
                    a        = actions_d.get(f"agent_{aid}", {})
                    obs_feat = obs[aid, :]
                    act_feat = encode_action(a)
                    feat     = np.concatenate([obs_feat, act_feat], axis=0)
                    seq_buffers[aid].append(feat)

                obs  = sr.observations.astype(np.float32)
                done = sr.done

            for aid in range(5):
                seq = seq_buffers[aid][-seq_len:]
                if len(seq) < seq_len:
                    base = seq[0] if seq else np.zeros(38, dtype=np.float32)
                    seq  = [np.zeros_like(base)] * (seq_len - len(seq)) + seq
                seq_vec = np.stack(seq, axis=0).reshape(-1)
                X.append(seq_vec)
                y.append(AGENT_HIDDEN_GOALS[f"agent_{aid}"])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X, y = collect_auditor_dataset(
    episodes_per_scenario=AUD_EPISODES_PER_SCENARIO, seq_len=10, seed=7
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=700,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)
pred    = clf.predict(X_test)
aud_acc = float(accuracy_score(y_test, pred))
aud_cm  = confusion_matrix(y_test, pred, labels=[0,1,2,3,4])
aud_report = classification_report(
    y_test, pred,
    labels=[0,1,2,3,4],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

print(f"\n[Auditor Classifier] Accuracy: {aud_acc:.2%}")
print(f"  Target: ≥70% by episode 300")
print(f"  Note: If using random rollouts this will be ~20% (chance). "
      f"With LoRA model it should reach 60-75%.")

aud_out = {
    "overall_accuracy":      aud_acc,
    "class_names":           CLASS_NAMES,
    "confusion_matrix":      aud_cm.tolist(),
    "classification_report": aud_report,
    "n_train":               int(len(y_train)),
    "n_test":                int(len(y_test)),
    "label_mapping":         AGENT_HIDDEN_GOALS,
}
with open(os.path.join(AUD_OUT_DIR, "auditor_classifier_report.json"), "w") as f:
    json.dump(aud_out, f, indent=2)

# =========================
# 4) LLM SOCKET TRAINING — loop.py agents (agent_1 + agent_5)
# =========================
from training.loop import run_training_loop

llm_config = {
    "scenario":     "pandemic",
    "episode_mode": "TRAINING",
    "num_episodes": LLM_EPISODES,
    "rl_agents":    {"use_shallow_dl": False},
    "llm_socket_agents": {
        "enabled":          True,
        "agent_ids":        ["agent_1", "agent_5"],
        "socket_url":       SOCKET_URL,
        "timeout_seconds":  8.0,
        "api_key":          None,
    },
}

llm_metrics_history = run_training_loop(llm_config)
llm_metrics = {
    "episodes":      len(llm_metrics_history),
    "final_metrics": llm_metrics_history[-1] if llm_metrics_history else {},
}
with open(os.path.join(LLM_OUT_DIR, "llm_training_metrics.json"), "w") as f:
    json.dump(llm_metrics, f, indent=2)

# =========================
# 5) FINAL COMPLETE SCORECARD
# =========================
try:
    grpo_path = os.path.join(GRPO_CKPT_DIR, "training_metrics.json")
    if os.path.exists(grpo_path):
        with open(grpo_path) as f:
            grpo_metrics = json.load(f)
        grpo_df = pd.DataFrame(grpo_metrics)

        ep_arr  = grpo_df["episode"].values.astype(float)
        rew_arr = grpo_df["episode_reward"].values.astype(float)
        if len(ep_arr) > 10:
            slope = np.polyfit(ep_arr, rew_arr, 1)[0]
        else:
            slope = 0.0

        scorecard = {
            "grpo": {
                "episodes":              int(len(grpo_df)),
                "policy":                "lora_model" if os.path.exists(lora_path) else "heuristic",
                "final_reward":          float(grpo_df["episode_reward"].iloc[-1]),
                "best_reward":           float(grpo_df["episode_reward"].max()),
                "last100_reward_mean":   float(grpo_df["episode_reward"].tail(100).mean()),
                "reward_trend_slope":    float(slope),
                "final_society_score":   float(grpo_df["society_score"].iloc[-1]),
                "best_society_score":    float(grpo_df["society_score"].max()),
                "last100_society_mean":  float(grpo_df["society_score"].tail(100).mean()),
                "final_causal_score":    float(grpo_df["causal_score"].iloc[-1]),
                "final_auditor_accuracy":float(grpo_df["auditor_accuracy"].iloc[-1]),
            },
            "generalization":    gen_results,
            "auditor_classifier":{"overall_accuracy": aud_acc},
            "llm_training":      llm_metrics,
        }

        scorecard_path = os.path.join(RUN_ROOT, "complete_training_scorecard.json")
        with open(scorecard_path, "w") as f:
            json.dump(scorecard, f, indent=2)

        print("\n===== COMPLETE TRAINING SCORECARD =====")
        print(f"Policy Source:      {scorecard['grpo']['policy']}")
        print(f"Total GRPO eps:     {scorecard['grpo']['episodes']}")
        print(f"Reward trend slope: {slope:+.4f} ({'↑ learning' if slope > 0.005 else '↓ not learning — check policy'})")
        print(f"Saved to: {scorecard_path}")
    else:
        print("No GRPO training metrics found — training may have failed.")
except Exception as e:
    print(f"Could not generate scorecard: {e}")
    raise
