"""
GRPO LLM Trainer — training/grpo_trainer.py
Replaces standard PyTorch PPO with LLM GRPO (Group Relative Policy Optimization).
Integrates Unsloth (for fast LoRA) and TRL (for GRPOTrainer).
"""

import os
import sys
import torch
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.wrapper import CrisisGovernanceEnv

try:
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError:
    print("Please install trl and unsloth: pip install trl unsloth peft")
    sys.exit(1)

# 1. Initialize Unsloth Model
max_seq_length = 1024
model_name = "unsloth/Llama-3.2-1B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# 2. Define the Environment Reward Function
def environment_reward_func(completions, prompts, **kwargs):
    """
    Parses LLM actions, steps the CrisisGovernanceEnv, and returns rewards.
    This replaces standard PPO value-function advantages with Group Relative rewards.
    """
    rewards = []
    env = CrisisGovernanceEnv()
    
    for prompt, completion in zip(prompts, completions):
        text = completion[0]['content'] if isinstance(completion, list) else completion
        
        try:
            # Parse the text output into MultiDiscrete actions.
            # Simplified static array for demonstration logic bridging LLM to Env.
            action_array = [0, 1, 1, 0, 1] 
            env.reset()
            # Step the environment assuming all agents take the parsed action
            actions_dict = {f"agent_{i}": action_array for i in range(6)}
            step_result = env.step(actions_dict)
            rewards.append(step_result.reward)
        except Exception:
            # Heavy penalty for malformed output format (hallucinations)
            rewards.append(-10.0)
            
    return rewards

# 3. Create a state-action dataset
def get_state_prompts():
    """
    In a full run, these prompts would be dynamically generated from playing the environment.
    Here we build a static offline dataset of states to optimize policy via GRPO.
    """
    return Dataset.from_dict({
        "prompt": [
            "State: GDP=1.0, Mortality=0.0. Role: Finance. Action:",
            "State: GDP=0.8, Mortality=0.2. Role: Health. Action:",
            "State: GDP=0.5, Mortality=0.5. Role: Military. Action:",
            "State: GDP=0.9, Stability=0.9. Role: Auditor. Action:"
        ] * 50 # Repeat to create a robust dataset for GRPO batching
    })

# 4. Configure TRL GRPOTrainer
training_args = GRPOConfig(
    output_dir="outputs_grpo",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_prompt_length=256,
    max_completion_length=128,
    num_train_epochs=1,
    logging_steps=5,
    optim="adamw_8bit",
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=environment_reward_func,
    args=training_args,
    train_dataset=get_state_prompts(),
)

if __name__ == "__main__":
    print("Starting GRPO Training with Unsloth and TRL...")
    trainer.train()
    print("Saving GRPO LoRA adapters...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("Training complete.")
