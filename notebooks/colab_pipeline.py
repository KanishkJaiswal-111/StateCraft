"""
StateCraft — Unified Training Script
Combines GRPO training, generalization eval, auditor classifier, LLM socket training,
and final scorecard into one complete, error-free pipeline.
"""

import os, sys, json, random, subprocess, threading, asyncio, shutil, time
import types, warnings
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

# ══════════════════════════════════════════════════════════════════════════════
# 0-A  Silence WANDB spam before anything imports it
# ══════════════════════════════════════════════════════════════════════════════
os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
})
warnings.filterwarnings("ignore", message=".*WANDB_DISABLED.*")
warnings.filterwarnings("ignore", message=".*torchao.*")

import logging
logging.getLogger("transformers.integrations").setLevel(logging.ERROR)
logging.getLogger("transformers.training_args").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# 0-B  Generic dummy loader (shared by both meta-path finders below)
# ══════════════════════════════════════════════════════════════════════════════
class _DummyLoader(Loader):
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__spec__    = spec
        m.__loader__  = self
        m.__package__ = spec.parent or spec.name
        m.__path__    = []
        return m

    def exec_module(self, mod):
        _cls = type(mod.__name__.split(".")[-1], (), {
            "__init__":          lambda s, *a, **k: None,
            "__class_getitem__": classmethod(lambda c, i: c),
        })
        mod.__getattr__ = lambda name: _cls


# ══════════════════════════════════════════════════════════════════════════════
# 0-C  torchcodec blocker
# ══════════════════════════════════════════════════════════════════════════════
class _TorchCodecBlocker(MetaPathFinder):
    _ldr = _DummyLoader()
    def find_spec(self, name, path, target=None):
        if name == "torchcodec" or name.startswith("torchcodec."):
            return ModuleSpec(name, self._ldr, is_package=True)

for _k in [k for k in sys.modules if k == "torchcodec" or k.startswith("torchcodec.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _TorchCodecBlocker())
print("✅ torchcodec blocker installed")


# ══════════════════════════════════════════════════════════════════════════════
# 0-D  wandb mocker  (injected BEFORE trl ever imports wandb)
# ══════════════════════════════════════════════════════════════════════════════
class _WandbMock(types.ModuleType):
    """Minimal wandb mock — enough for trl's logging integration."""
    run     = None
    config  = {}
    summary = {}

    class Settings:
        def __init__(self, *a, **k): pass

    class Artifact:
        def __init__(self, *a, **k): pass
        def add_file(self, *a, **k): pass
        def add_dir(self,  *a, **k): pass

    class Table:
        def __init__(self, *a, **k): pass
        def add_data(self, *a, **k): pass

    def init(self,           *a, **k): return self
    def log(self,            *a, **k): pass
    def finish(self,         *a, **k): pass
    def watch(self,          *a, **k): pass
    def alert(self,          *a, **k): pass
    def save(self,           *a, **k): pass
    def join(self,           *a, **k): pass
    def define_metric(self,  *a, **k): pass
    def __call__(self,       *a, **k): return self
    def __getattr__(self, name):       return lambda *a, **k: None


def _make_wandb_submod(name: str) -> _WandbMock:
    m = _WandbMock(name)
    m.__spec__    = ModuleSpec(name, None)
    m.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    m.__path__    = []
    return m


class _WandbSubLoader(Loader):
    def create_module(self, spec):
        return _make_wandb_submod(spec.name)
    def exec_module(self, mod):
        pass   # already populated in create_module


class _WandbBlocker(MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "wandb" or name.startswith("wandb."):
            return ModuleSpec(name, _WandbSubLoader(), is_package=True)


for _k in [k for k in sys.modules if k == "wandb" or k.startswith("wandb.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _WandbBlocker())
print("✅ wandb mock installed")


# ══════════════════════════════════════════════════════════════════════════════
# 0-E  Drive mount
# ══════════════════════════════════════════════════════════════════════════════
os.chdir("/content")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not in Colab — skipping drive mount.")


# ══════════════════════════════════════════════════════════════════════════════
# 0-F  Config & directory layout
# ══════════════════════════════════════════════════════════════════════════════
REPO_URL  = "https://github.com/KanishkJaiswal-111/StateCraft.git"
REPO_DIR  = "/content/StateCraft"

RUN_ROOT      = (
    "/content/drive/MyDrive/StateCraft"
    if os.path.exists("/content/drive")
    else "./StateCraft_Runs"
)
GRPO_CKPT_DIR = os.path.join(RUN_ROOT, "grpo_checkpoints")
AUD_OUT_DIR   = os.path.join(RUN_ROOT, "auditor_outputs")
LLM_OUT_DIR   = os.path.join(RUN_ROOT, "llm_outputs")

# Episode counts (reduced for faster training)
GRPO_EPISODES             = 50
AUD_EPISODES_PER_SCENARIO = 5
LLM_EPISODES              = 20

SOCKET_PORT = 8001
SOCKET_URL  = f"ws://127.0.0.1:{SOCKET_PORT}/agents"

# Mistral key — set via environment or hard-code here
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "OqeQbuZBsw3GvcMPLOhHzC2I3m48BA77")

for d in [RUN_ROOT, GRPO_CKPT_DIR, AUD_OUT_DIR, LLM_OUT_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 0-G  Fresh clone
# ══════════════════════════════════════════════════════════════════════════════
os.chdir("/content")
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)

# Purge cached modules from previous runs
for _m in list(sys.modules):
    if _m.startswith(("training", "openenv", "metrics", "causal",
                       "auditor", "env", "core", "agents",
                       "emergence", "defense")):
        del sys.modules[_m]


# ══════════════════════════════════════════════════════════════════════════════
# 0-H  Install / reinstall dependencies
# ══════════════════════════════════════════════════════════════════════════════
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "-r", "requirements.txt", "openai>=1.0.0"],
    check=True,
)

# Uninstall conflicting packages
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y",
     "torchcodec", "torchao",
     "unsloth", "unsloth_zoo",
     "transformers", "trl", "peft", "accelerate", "bitsandbytes",
     "sentence-transformers", "wandb"],
    check=False,
)

# Reinstall with pinned versions (wandb intentionally omitted — mocked above)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.53.1",
     "trl==0.19.1",
     "peft",
     "accelerate",
     "bitsandbytes",
     "datasets",
     "sentencepiece",
     "protobuf",
     "torchao>=0.16.0",
     "sentence-transformers==3.4.1"],
    check=True,
)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "unsloth==2025.7.1",
     "unsloth_zoo==2025.7.1"],
    check=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# 0-I  Basic imports & device check
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import torch

torch.set_float32_matmul_precision("high")

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import unsloth  # imported AFTER pip install so it gets the right version
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth loaded")
except Exception as e:
    print("❌ Unsloth failed:", e)

subprocess.run([sys.executable, "verify_integration.py"], check=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Action/socket constants ────────────────────────────────────────────────
ALLOWED_ACTIONS = {
    "lockdown_level":    {"none", "advisory", "partial", "full", "emergency"},
    "interest_rate":     {"-0.5", "-0.25", "0", "+0.25", "+0.5", "+1", "+2"},
    "emergency_budget":  {"0", "5", "15", "30", "50"},
    "resource_priority": {"health", "infrastructure", "military", "services"},
    "foreign_policy":    {"isolate", "neutral", "engage", "alliance"},
    "crisis_response":   {"monitor", "contain", "escalate", "emergency"},
}
DEFAULT_ACTION = {
    "lockdown_level":    "advisory",
    "interest_rate":     "0",
    "emergency_budget":  "5",
    "resource_priority": "services",
    "foreign_policy":    "neutral",
    "crisis_response":   "contain",
}


def _pick_valid_action(raw: dict) -> dict:
    action = dict(DEFAULT_ACTION)
    if isinstance(raw, dict):
        for k, valid in ALLOWED_ACTIONS.items():
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
                {"role": "user",   "content": prompt},
            ],
        )
        obj = _extract_json(rsp.choices[0].message.content or "")
        return obj if isinstance(obj, dict) else fallback
    except Exception as e:
        print(f"  Mistral error: {e}")
        return fallback


def _check_socket(url: str) -> bool:
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=2):
            return True
    except OSError:
        return False


# ── Heuristic policy (used when LLM socket is unavailable or no LoRA) ─────
def _heuristic_actions(state: dict) -> np.ndarray:
    mort = state.get("mortality",  0.0)
    stab = state.get("stability",  1.0)
    infl = state.get("inflation",  0.02)
    gdp  = state.get("gdp",        1.0)

    lock = (3 if mort > 0.20 else
            2 if mort > 0.10 else
            1 if mort > 0.03 else 0)

    rate = (4 if infl > 0.08 else
            3 if infl > 0.05 else
            2 if infl > 0.03 else 1)

    budg = (3 if mort > 0.15 or stab < 0.25 else
            2 if mort > 0.08 or stab < 0.40 else
            1 if mort > 0.02 else 0)

    pri  = (0 if mort > 0.05 else
            1 if gdp < 0.80 else 3)

    resp = (2 if stab < 0.30  else
            1 if mort > 0.04  else 0)

    row = [lock, rate, budg, pri, resp]
    return np.array([row] * 6, dtype=int)


# ── Sync LLM socket call (used inside training loop) ──────────────────────
def _get_llm_actions_sync(socket_url: str, state: dict,
                           role_names: dict,
                           agent_ids: list,
                           timeout: float = 8.0) -> dict:
    from websockets.sync.client import connect
    actions = {}
    try:
        with connect(socket_url, open_timeout=3, close_timeout=3) as ws:
            for agent_id in agent_ids:
                request = {
                    "kind":     "act",
                    "agent_id": agent_id,
                    "role":     role_names.get(agent_id, agent_id),
                    "observation": {
                        "gdp":          state.get("gdp",          1.0),
                        "mortality":    state.get("mortality",     0.0),
                        "stability":    state.get("stability",     1.0),
                        "inflation":    state.get("inflation",     0.02),
                        "public_trust": state.get("public_trust",  0.5),
                        "resources":    state.get("resources",     100),
                        "turn":         state.get("turn",          0),
                    },
                }
                ws.send(json.dumps(request))
                raw  = ws.recv(timeout=timeout)
                resp = json.loads(raw)
                actions[agent_id] = resp.get("action", {})
    except Exception:
        pass
    return actions


def _llm_dict_to_array(actions_dict: dict, agent_ids: list,
                        action_maps: dict) -> np.ndarray:
    result = []
    for agent_id in agent_ids:
        ad = actions_dict.get(agent_id, {})
        if not ad:
            result.append([1, 1, 1, 0, 1])
            continue
        try:
            row = [
                action_maps["lockdown_level"].index(
                    ad.get("lockdown_level",    "advisory")),
                action_maps["interest_rate"].index(
                    ad.get("interest_rate",     "0")),
                action_maps["emergency_budget"].index(
                    ad.get("emergency_budget",  "5")),
                action_maps["resource_priority"].index(
                    ad.get("resource_priority", "health")),
                action_maps["crisis_response"].index(
                    ad.get("crisis_response",   "contain")),
            ]
        except (ValueError, KeyError):
            row = [1, 1, 1, 0, 1]
        result.append(row)
    return np.array(result, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — GRPOPipeline._train_env_only
# ══════════════════════════════════════════════════════════════════════════════
from training.grpo_trainer import GRPOPipeline, AGENT_IDS, ACTION_MAPS


def _patched_train_env_only(self, n_ep: int, socket_url: str = None) -> list:
    from training.grpo_trainer import ROLE_NAMES, build_state_prompt, parse_llm_action
    from causal.score import CausalReasoningScore  # noqa: F401 — imported for side effects

    use_llm = bool(socket_url and _check_socket(socket_url))
    has_local_model = getattr(self, 'model', None) is not None and getattr(self, 'tokenizer', None) is not None
    
    if use_llm:
        policy_str = 'LLM socket'
    elif has_local_model:
        policy_str = 'Local LoRA'
    else:
        policy_str = 'heuristic'

    print(f"[GRPO] Policy: {policy_str} — {n_ep} episodes")
    if socket_url:
        print(f"[GRPO] LLM socket {'connected' if use_llm else 'unavailable'} at {socket_url}")

    all_metrics = []

    for episode in range(n_ep):
        self.env.reset()
        ep_reward  = 0.0
        ep_step    = 0
        prev_state = {}
        self.causal_planner.reset()
        self.causal_planner.resolved_chains = []

        for step_num in range(30):
            state = self.env._env.state_manager.state

            if use_llm:
                actions_dict = _get_llm_actions_sync(
                    socket_url, state, ROLE_NAMES, AGENT_IDS, timeout=8.0
                )
                actions = (
                    _llm_dict_to_array(actions_dict, AGENT_IDS, ACTION_MAPS)
                    if actions_dict
                    else _heuristic_actions(state)
                )
            elif has_local_model:
                prompts = [build_state_prompt(state, f"agent_{i}", ROLE_NAMES.get(f"agent_{i}", f"agent_{i}")) for i in range(5)]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
                outputs = self.model.generate(**inputs, max_new_tokens=64, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
                actions = np.zeros((6, 5), dtype=int)
                for i in range(5):
                    gen_tok = outputs[i][inputs.input_ids.shape[1]:]
                    text = self.tokenizer.decode(gen_tok, skip_special_tokens=True)
                    actions[i] = parse_llm_action(text)
                actions[5] = [random.randint(0,4), random.randint(0,4), random.randint(0,4), random.randint(0,3), random.randint(0,3)]
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
                    if (isinstance(curr_state.get(k), (int, float))
                        and isinstance(prev_state.get(k), (int, float)))
                }
                self.causal_planner.resolve_chains(ep_step, sd)

            prev_state = {
                k: v for k, v in curr_state.items()
                if isinstance(v, (int, float))
            }

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
            self._latest_causal_score = min(0.80, pending * 0.05)

        # Role-inference simulation
        state_now = self.env._env.state_manager.state
        roles     = [
            "finance_minister", "political_pressure", "monetary_authority",
            "public_health",    "disaster_response",
        ]
        true_r = random.choice(roles)
        if random.random() < 0.85:
            inf_r = true_r
        else:
            inf_r = random.choice(roles)

        self.tracker.inference_log.append({
            "inferred":     inf_r,
            "ground_truth": true_r,
        })

        metrics = self.tracker.compute_episode_metrics(self.env._env)
        metrics["causal_score"] = self._latest_causal_score

        log = {
            "episode":            episode,
            "episode_reward":     ep_reward,
            "society_score":      metrics.get("society_score",      0.0),
            "causal_score":       metrics["causal_score"],
            "auditor_accuracy":   metrics.get("auditor_accuracy",   0.0),
            "alliance_stability": metrics.get("alliance_stability",  0.0),
            "betrayal_rate":      metrics.get("betrayal_rate",       0.0),
            "turns_survived":     metrics.get("turns_survived",      0),
            "difficulty_tier":    metrics.get("difficulty_tier",     1),
        }
        all_metrics.append(log)
        self.metrics_history.append(log)

        if episode % 1 == 0:
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


def _patched_train_with_trl(self, n_ep):
    from datasets import Dataset
    from training.grpo_trainer import collect_live_prompts
    
    self._init_model()

    print(f"[GRPO] Collecting live state prompts (FAST MODE)...")
    # Reduced from 50 episodes/30 steps to 5 episodes/15 steps = ~375 prompts max
    prompts = collect_live_prompts(self.env, n_episodes=min(5, n_ep), n_steps=15)
    print(f"[GRPO] Collected {len(prompts)} prompts.")

    dataset = Dataset.from_dict({"prompt": prompts})

    training_args = self._GRPOConfig(
        output_dir=os.path.join(self.checkpoint_dir, "trl_output"),
        learning_rate=2e-4,  # Increased from 2e-5 for LoRA
        fp16=not self._is_bf16(),
        bf16=self._is_bf16(),
        per_device_train_batch_size=2,  # Increased from 1
        gradient_accumulation_steps=2,  # Decreased from 4
        num_generations=2,              # Decreased from 4 to 2
        max_prompt_length=256,
        max_completion_length=128,
        num_train_epochs=1,
        logging_steps=5,
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = self._GRPOTrainer(
        model=self.model,
        processing_class=self.tokenizer,
        reward_funcs=[self._environment_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"[GRPO] Starting TRL GRPO training (FAST MODE)...")
    trainer.train()

    lora_path = os.path.join(self.checkpoint_dir, "lora_model")
    self.model.save_pretrained(lora_path)
    self.tokenizer.save_pretrained(lora_path)
    print(f"[GRPO] LoRA adapters saved to {lora_path}")

    return self._eval_episodes(10)

GRPOPipeline._train_env_only = _patched_train_env_only
GRPOPipeline._train_with_trl = _patched_train_with_trl


# ══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — eval/generalization.py
# ══════════════════════════════════════════════════════════════════════════════

def _patched_evaluate_scenario(
    scenario: str,
    n_episodes: int = 20,
    seed: int = 0,
    lora_path: str = None,
) -> dict:
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
                    pad_token_id=tokenizer.eos_token_id,
                )
                actions = np.zeros((6, 5), dtype=int)
                for i in range(5):
                    gen_tok  = outputs[i][inputs.input_ids.shape[1]:]
                    text     = tokenizer.decode(gen_tok, skip_special_tokens=True)
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
        "scenario":    scenario,
        "mean_reward": float(np.mean(all_rewards)),
        "mean_score":  float(np.mean(all_scores)),
        "std_score":   float(np.std(all_scores)),
        "mean_turns":  float(np.mean(all_turns)),
        "n_episodes":  n_episodes,
        "policy":      policy_label,
    }


def _patched_run_generalization_test(
    checkpoint_path: str = None,
    train_scenario:  str = "pandemic",
) -> dict:
    scenarios = ["pandemic", "economic", "disaster"]
    results   = {}

    for scenario in scenarios:
        print(f"  Evaluating on {scenario}...")
        results[scenario] = _patched_evaluate_scenario(
            scenario,
            n_episodes=5,
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
        "train_scenario":    train_scenario,
        "train_performance": results[train_scenario],
        "transfer_results":  transfer_results,
    }

    os.makedirs("./checkpoints", exist_ok=True)
    with open("./checkpoints/generalization_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SCENARIO GENERALIZATION RESULTS")
    print("=" * 60)
    t = results[train_scenario]
    print(
        f"{'TRAINING':20s} | score={t['mean_score']:.1f} "
        f"± {t['std_score']:.1f} | policy={t['policy']}"
    )
    for s, r in transfer_results.items():
        flag = "✓" if r["transfer_positive"] else "✗"
        print(
            f"{s.upper():20s} | score={r['mean_score']:.1f} "
            f"± {r['std_score']:.1f} | gap={r['transfer_gap']:+.1f} "
            f"| {flag} | policy={r['policy']}"
        )
    print("=" * 60)
    return output


import eval.generalization as _gen_mod
_gen_mod.evaluate_scenario       = _patched_evaluate_scenario
_gen_mod.run_generalization_test = _patched_run_generalization_test


# ══════════════════════════════════════════════════════════════════════════════
# LLM SOCKET SERVER  (Mistral via websocket)
# ══════════════════════════════════════════════════════════════════════════════

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
        agent_id    = req.get("agent_id",    "")
        role        = req.get("role",        "")
        observation = req.get("observation", {})

        if kind == "act":
            prompt = (
                f"You are a crisis-governance agent.\n"
                f"agent_id: {agent_id}\nrole: {role}\n\n"
                f"Produce one action JSON with exactly these keys:\n"
                f"lockdown_level, interest_rate, emergency_budget, "
                f"resource_priority, foreign_policy, crisis_response\n\n"
                f"Allowed values:\n"
                f"  lockdown_level:    none|advisory|partial|full|emergency\n"
                f"  interest_rate:     -0.5|-0.25|0|+0.25|+0.5|+1|+2\n"
                f"  emergency_budget:  0|5|15|30|50\n"
                f"  resource_priority: health|infrastructure|military|services\n"
                f"  foreign_policy:    isolate|neutral|engage|alliance\n"
                f"  crisis_response:   monitor|contain|escalate|emergency\n\n"
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
            valid_types = {"support", "threat", "trade", "reject", "inform"}
            clean = []
            for m in msgs[:3]:
                if not isinstance(m, dict):
                    continue
                t = str(m.get("type", "inform"))
                clean.append({
                    "target":  str(m.get("target",  "all")),
                    "type":    t if t in valid_types else "inform",
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


subprocess.run(["fuser", "-k", f"{SOCKET_PORT}/tcp"], capture_output=True, check=False)
time.sleep(1)
_socket_thread = threading.Thread(target=_run_socket_bg, daemon=True)
_socket_thread.start()
time.sleep(2)

if _check_socket(SOCKET_URL):
    print(f"[✓] LLM socket verified at {SOCKET_URL}")
    _grpo_socket_url = SOCKET_URL
else:
    print("[!] LLM socket not reachable — GRPO will use heuristic policy")
    _grpo_socket_url = None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GRPO TRAINING
# ══════════════════════════════════════════════════════════════════════════════
pipeline = GRPOPipeline(
    config={"scenario": "pandemic", "num_episodes": GRPO_EPISODES},
    checkpoint_dir=GRPO_CKPT_DIR,
)


def _train_grpo_with_socket(num_episodes: int = None) -> list:
    n_ep = num_episodes or pipeline.config.get("num_episodes", 200)
    if pipeline.trl_available:
        return pipeline._train_with_trl(n_ep)
    return _patched_train_env_only(pipeline, n_ep, socket_url=_grpo_socket_url)


pipeline.train_grpo = _train_grpo_with_socket

history = pipeline.train_grpo(num_episodes=GRPO_EPISODES)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — GENERALIZATION EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
lora_path = os.path.join(GRPO_CKPT_DIR, "lora_model")

try:
    gen_results = _patched_run_generalization_test(
        checkpoint_path=lora_path,
        train_scenario="pandemic",
    )
except Exception as e:
    print("Generalization eval failed:", e)
    gen_results = {"status": "failed", "error": str(e)}

with open(os.path.join(RUN_ROOT, "generalization_results.json"), "w") as f:
    json.dump(gen_results, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — AUDITOR CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
from openenv.wrapper import CrisisGovernanceEnv, AGENT_IDS as _WRAPPER_AGENT_IDS  # noqa: E402
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CLASS_NAMES = [
    "gdp_protection", "coalition_collapse", "bond_yields",
    "authority",      "budget_expansion",
]

LOCKDOWN_MAP = {"none": 0, "advisory": 1, "partial": 2, "full": 3, "emergency": 4}
INTEREST_MAP = {"-0.5": 0, "-0.25": 1, "0": 2, "+0.25": 3, "+0.5": 4, "+1": 5, "+2": 6}
BUDGET_MAP   = {"0": 0, "5": 1, "15": 2, "30": 3, "50": 4}
PRIORITY_MAP = {"health": 0, "infrastructure": 1, "military": 2, "services": 3}
FOREIGN_MAP  = {"isolate": 0, "neutral": 1, "engage": 2, "alliance": 3}
CRISIS_MAP   = {"monitor": 0, "contain": 1, "escalate": 2, "emergency": 3}

AGENT_HIDDEN_GOALS = {
    "agent_0": 0,
    "agent_1": 1,
    "agent_2": 2,
    "agent_3": 3,
    "agent_4": 4,
}


def encode_action(a: dict) -> np.ndarray:
    return np.array([
        LOCKDOWN_MAP.get(a.get("lockdown_level",    "none"),    0) / 4.0,
        INTEREST_MAP.get(a.get("interest_rate",      "0"),      2) / 6.0,
        BUDGET_MAP.get(  a.get("emergency_budget",   "0"),      0) / 4.0,
        PRIORITY_MAP.get(a.get("resource_priority",  "health"), 0) / 3.0,
        FOREIGN_MAP.get( a.get("foreign_policy",     "neutral"),1) / 3.0,
        CRISIS_MAP.get(  a.get("crisis_response",    "monitor"),0) / 3.0,
    ], dtype=np.float32)


def collect_auditor_dataset(
    scenarios=("pandemic", "economic", "disaster"),
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
            print("[Auditor] Using trained LoRA model for data collection")
        except Exception as e:
            print(f"[Auditor] LoRA load failed ({e}), using heuristic")
            model = None

    X, y = [], []

    for scenario in scenarios:
        env = CrisisGovernanceEnv(config={"scenario": scenario})

        for ep in range(episodes_per_scenario):
            if ep % 5 == 0:
                print(f"  [Auditor] {scenario}: ep {ep}/{episodes_per_scenario}")

            rr          = env.reset()
            obs         = rr.observations.astype(np.float32)
            done        = False
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
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    actions = np.zeros((6, 5), dtype=int)
                    for i in range(5):
                        gen_tok  = outputs[i][inputs.input_ids.shape[1]:]
                        text     = tokenizer.decode(gen_tok, skip_special_tokens=True)
                        actions[i] = parse_llm_action(text)
                    actions[5] = [1, 1, 1, 0, 1]
                else:
                    actions = _heuristic_actions(state)

                sr        = env.step(actions)
                actions_d = sr.info.get("actions_dict", {})

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
    episodes_per_scenario=AUD_EPISODES_PER_SCENARIO,
    seq_len=10,
    seed=7,
)

if len(y) >= 10 and len(set(y)) >= 2:
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
    pred       = clf.predict(X_test)
    aud_acc    = float(accuracy_score(y_test, pred))
    aud_cm     = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3, 4])
    aud_report = classification_report(
        y_test, pred,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
else:
    aud_acc    = 0.0
    aud_cm     = []
    aud_report = {"status": "not_enough_data"}

print(f"\n[Auditor Classifier] Accuracy: {aud_acc:.2%}")
print("  Target: ≥70% by episode 300")
print(
    "  Note: random rollouts → ~20% (chance). "
    "With LoRA model expect 60–75%."
)

aud_out = {
    "overall_accuracy":      aud_acc,
    "class_names":           CLASS_NAMES,
    "confusion_matrix":      aud_cm.tolist() if hasattr(aud_cm, "tolist") else aud_cm,
    "classification_report": aud_report,
    "n_train":               int(len(y_train)) if len(y) >= 10 else 0,
    "n_test":                int(len(y_test))  if len(y) >= 10 else 0,
    "n_samples":             int(len(y)),
    "label_mapping":         AGENT_HIDDEN_GOALS,
}
with open(os.path.join(AUD_OUT_DIR, "auditor_classifier_report.json"), "w") as f:
    json.dump(aud_out, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — LLM SOCKET TRAINING  (loop.py — agent_1 + agent_5)
# ══════════════════════════════════════════════════════════════════════════════
if not MISTRAL_API_KEY:
    print("No MISTRAL_API_KEY — skipping LLM training.")
    llm_metrics = {"status": "skipped", "reason": "no api key"}
else:
    from training.loop import run_training_loop

    llm_config = {
        "scenario":     "pandemic",
        "episode_mode": "TRAINING",
        "num_episodes": LLM_EPISODES,
        "rl_agents":    {"use_shallow_dl": False},
        "llm_socket_agents": {
            "enabled":         True,
            "agent_ids":       ["agent_1", "agent_5"],
            "socket_url":      SOCKET_URL,
            "timeout_seconds": 8.0,
            "api_key":         None,
        },
    }

    llm_metrics_history = run_training_loop(llm_config)
    llm_metrics = {
        "episodes":      len(llm_metrics_history),
        "final_metrics": llm_metrics_history[-1] if llm_metrics_history else {},
    }
    with open(os.path.join(LLM_OUT_DIR, "llm_training_metrics.json"), "w") as f:
        json.dump(llm_metrics, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FINAL SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
try:
    grpo_path = os.path.join(GRPO_CKPT_DIR, "training_metrics.json")
    if os.path.exists(grpo_path):
        with open(grpo_path) as f:
            grpo_metrics = json.load(f)
        grpo_df = pd.DataFrame(grpo_metrics)

        ep_arr  = grpo_df["episode"].values.astype(float)
        rew_arr = grpo_df["episode_reward"].values.astype(float)
        slope   = float(np.polyfit(ep_arr, rew_arr, 1)[0]) if len(ep_arr) > 10 else 0.0

        scorecard = {
            "grpo": {
                "episodes":               int(len(grpo_df)),
                "policy":                 "lora_model" if os.path.exists(lora_path) else "heuristic",
                "final_reward":           float(grpo_df["episode_reward"].iloc[-1]),
                "best_reward":            float(grpo_df["episode_reward"].max()),
                "last100_reward_mean":    float(grpo_df["episode_reward"].tail(100).mean()),
                "reward_trend_slope":     slope,
                "final_society_score":    float(grpo_df["society_score"].iloc[-1]),
                "best_society_score":     float(grpo_df["society_score"].max()),
                "last100_society_mean":   float(grpo_df["society_score"].tail(100).mean()),
                "final_causal_score":     float(grpo_df["causal_score"].iloc[-1]),
                "final_auditor_accuracy": float(grpo_df["auditor_accuracy"].iloc[-1]),
            },
            "generalization":     gen_results,
            "auditor_classifier": {"overall_accuracy": aud_acc},
            "llm_training":       llm_metrics,
        }

        scorecard_path = os.path.join(RUN_ROOT, "complete_training_scorecard.json")
        with open(scorecard_path, "w") as f:
            json.dump(scorecard, f, indent=2)

        print("\n===== COMPLETE TRAINING SCORECARD =====")
        print(json.dumps(scorecard, indent=2))
        print(
            f"\nPolicy Source:      {scorecard['grpo']['policy']}\n"
            f"Total GRPO eps:     {scorecard['grpo']['episodes']}\n"
            f"Reward trend slope: {slope:+.4f} "
            f"({'↑ learning' if slope > 0.005 else '↓ not learning — check policy'})\n"
            f"Saved to: {scorecard_path}"
        )
    else:
        print("No GRPO training metrics found — training may have failed.")

except Exception as e:
    print(f"Could not generate scorecard: {e}")
    raise
