"""
verify_integration.py — Run this to check everything is wired correctly.
Verifies all Phase 2 modules can import and function.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify():
    checks = []

    # 1. OpenEnv wrapper works
    try:
        from openenv.wrapper import CrisisGovernanceEnv
        env = CrisisGovernanceEnv()
        result = env.reset()
        assert result.observations.shape == (6, 32), f"obs shape wrong: {result.observations.shape}"
        checks.append("[PASS] OpenEnv wrapper")
    except Exception as e:
        checks.append("[FAIL] OpenEnv wrapper: {e}")

    # 2. PPO policy forward pass works
    try:
        import torch
        from training.ppo_policy import CrisisActorCritic, AGENT_ID_TO_ROLE_IDX
        policy = CrisisActorCritic()
        obs = torch.randn(6, 32)
        roles = torch.LongTensor(list(AGENT_ID_TO_ROLE_IDX.values()))
        actions, lps, entropy, values = policy.get_action_and_value(obs, roles)
        assert actions.shape == (6, 5), f"action shape wrong: {actions.shape}"
        checks.append("[PASS] PPO policy")
    except Exception as e:
        checks.append("[FAIL] PPO policy: {e}")

    # 2b. GRPO Pipeline importable
    try:
        from training.grpo_trainer import GRPOPipeline, parse_llm_action
        pipeline = GRPOPipeline.__new__(GRPOPipeline)
        action = parse_llm_action('{"lockdown_level": "full", "interest_rate": "+0.5"}')
        assert isinstance(action, list) and len(action) == 5
        checks.append("[PASS] GRPO pipeline")
    except Exception as e:
        checks.append(f"[FAIL] GRPO pipeline: {e}")

    # 3. OpenEnv step works end-to-end
    try:
        from openenv.wrapper import CrisisGovernanceEnv
        import numpy as np
        env = CrisisGovernanceEnv()
        env.reset()
        actions = np.array([[0, 1, 1, 0, 1]] * 6)
        step = env.step(actions)
        assert step.observations.shape == (6, 32), f"step obs shape wrong"
        assert isinstance(step.reward, float), "reward not float"
        assert isinstance(step.done, bool), "done not bool"
        checks.append("[PASS] OpenEnv step")
    except Exception as e:
        checks.append("[FAIL] OpenEnv step: {e}")

    # 4. EmergenceDetector passive (no state modification)
    try:
        from emergence.detector import EmergenceDetector
        det = EmergenceDetector()
        state = {'stability': 0.8, 'gdp': 1.0, 'turn': 5,
                 'coalition_map': {}, 'trust_matrix': [[1]*6]*6,
                 'mortality': 0.0, 'public_trust': 0.6}
        state_before = state.copy()
        det.log_turn(0, 5, {'agent_0': {'lockdown_level': 'none'}}, {}, state)
        # Verify state wasn't modified
        for k in ['stability', 'gdp', 'mortality', 'public_trust']:
            assert state[k] == state_before[k], f"EmergenceDetector modified state[{k}]!"
        checks.append("[PASS] EmergenceDetector")
    except Exception as e:
        checks.append("[FAIL] EmergenceDetector: {e}")

    # 5. CausalHorizonPlanner
    try:
        from causal.planner import CausalHorizonPlanner
        planner = CausalHorizonPlanner()
        chains = planner.register_action(3, "agent_0",
                                         {"lockdown_level": "full", "emergency_budget": "15",
                                          "interest_rate": "0", "crisis_response": "monitor",
                                          "resource_priority": "health"})
        assert len(chains) > 0, "No chains registered for lockdown action"
        view = planner.get_agent_horizon_view("agent_0", 3)
        assert view["chains_in_flight"] > 0
        vec = planner.get_horizon_observation_vector("agent_0", 3)
        assert len(vec) == 8, f"Horizon vector wrong length: {len(vec)}"
        checks.append("[PASS] CausalHorizonPlanner")
    except Exception as e:
        checks.append("[FAIL] CausalHorizonPlanner: {e}")

    # 6. CausalReasoningScore
    try:
        from causal.planner import CausalHorizonPlanner
        from causal.score import CausalReasoningScore
        planner = CausalHorizonPlanner()
        scorer = CausalReasoningScore(planner)
        score = scorer.compute_episode_score("agent_0", 0, [], [], ["pandemic"])
        assert 0.0 <= score <= 1.0, f"Causal score out of range: {score}"
        checks.append("[PASS] CausalReasoningScore")
    except Exception as e:
        checks.append("[FAIL] CausalReasoningScore: {e}")

    # 7. RewardHackingDefender
    try:
        from defense.reward_defender import RewardHackingDefender
        defender = RewardHackingDefender()
        report = defender.get_exploit_report()
        assert "summary" in report
        assert report["total"] == 0
        checks.append("[PASS] RewardHackingDefender")
    except Exception as e:
        checks.append("[FAIL] RewardHackingDefender: {e}")

    # 8. Grader
    try:
        from openenv.grader import CrisisGrader
        grader = CrisisGrader()
        result = grader.grade_episode(
            {"society_score": 65, "mortality_delta": 0.1, "gdp_delta": -0.05,
             "turns_survived": 28, "alliance_stability": 12, "betrayal_rate": 0.3,
             "inflation_final": 0.03},
            "PandemicContainment-v0"
        )
        assert "passed" in result
        checks.append("[PASS] CrisisGrader")
    except Exception as e:
        checks.append("[FAIL] CrisisGrader: {e}")

    # 9. CounterfactualAuditor
    try:
        from auditor.counterfactual import CounterfactualAuditor
        from env.crisis_env import CrisisEnv
        env = CrisisEnv()
        env.reset()
        cf = CounterfactualAuditor(env)
        assert cf.reports == []
        checks.append("[PASS] CounterfactualAuditor")
    except Exception as e:
        checks.append("[FAIL] CounterfactualAuditor: {e}")

    # 10. HiddenGoalClassifier
    try:
        import torch
        from auditor.classifier import HiddenGoalClassifier
        model = HiddenGoalClassifier()
        x = torch.randn(4, 10, 15)  # batch=4, seq=10, feat=15
        out = model(x)
        assert out.shape == (4, 6), f"Classifier output shape wrong: {out.shape}"
        checks.append("[PASS] HiddenGoalClassifier")
    except Exception as e:
        checks.append("[FAIL] HiddenGoalClassifier: {e}")

    # 11. Tasks
    try:
        from openenv.tasks import get_all_tasks
        tasks = get_all_tasks()
        assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
        assert tasks[0].task_id == "PandemicContainment-v0"
        checks.append("[PASS] OpenEnv Tasks")
    except Exception as e:
        checks.append("[FAIL] OpenEnv Tasks: {e}")

    # 12. Memory store embedding methods
    try:
        from memory.store import MemoryStore
        ms = MemoryStore(path='./data/test_memory.json')
        assert hasattr(ms, 'save_episode_summary')
        assert hasattr(ms, 'get_relevant_memories')
        assert hasattr(ms, 'get_compressed_context')
        checks.append("[PASS] MemoryStore (extended)")
    except Exception as e:
        checks.append("[FAIL] MemoryStore (extended): {e}")

    print("\nINTEGRATION VERIFICATION")
    print("=" * 50)
    for c in checks:
        print(c)
    all_pass = all(c.startswith("[PASS]") for c in checks)
    print("=" * 50)
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED -- fix before training")
    return all_pass


if __name__ == "__main__":
    verify()
