import numpy as np
from src.env.environment import run_episode
from src.env.state import init_state, update_state


def _make_record():
    return {"item_responses": np.array([1.0, 0.0, 1.0, 0.0]), "label": 1,
            "label_source": "questionnaire",
            "covariates": {"age_band": "1-2", "sex": "M"},
            "missing_mask": np.array([False] * 4)}


def _predictor(state):
    s = sum(int(state["value"][j]) for j in range(state["n"]) if state["mask"][j] == 1)
    return 0.3 + 0.2 * s


def _policy(state, legal):
    items = [a for a in legal if a != -1]
    return items[0] if items else -1


def test_trace_belief_update():
    rec = _make_record()
    result = run_episode(rec, question_budget=2, b_min=0, policy=_policy, predictor=_predictor)

    # --- Independent replay ---------------------------------------------
    # Reconstruct every pre-step state from the record alone and evaluate the
    # predictor on it. The recorded belief must equal that output. This is the
    # assertion AGENT_PROGRESS.md (known open item 8) recorded as commented out.
    state = init_state(rec)
    state["questions_remaining"] = 2
    state["budget"] = 2
    for t in result["trace"]:
        expected_before = _predictor(state)
        assert abs(t["belief_before"] - expected_before) < 1e-12, (
            f"step {t['step']}: belief_before {t['belief_before']} != predictor "
            f"output on the independently reconstructed pre-step state "
            f"{expected_before}")
        j = t["item_idx"]
        assert int(rec["item_responses"][j]) == t["value"], (
            f"step {t['step']}: traced value {t['value']} != record value "
            f"{int(rec['item_responses'][j])}")
        state = update_state(state, j, t["value"], state["questions_remaining"] - 1)
        state["questions_remaining"] -= 1
        state["budget"] = 2
        expected_after = _predictor(state)
        assert abs(t["belief_after"] - expected_after) < 1e-12, (
            f"step {t['step']}: belief_after {t['belief_after']} != predictor "
            f"output on the reconstructed post-step state {expected_after}")

    # --- Sequential consistency -------------------------------------------
    # belief_after of step i and belief_before of step i+1 are the same state
    # evaluated twice; they must agree exactly. A mismatch here means the trace
    # is not a faithful record of the episode the environment actually ran.
    tr = result["trace"]
    for i in range(len(tr) - 1):
        assert abs(tr[i]["belief_after"] - tr[i + 1]["belief_before"]) < 1e-12, (
            "belief_after of step i must equal belief_before of step i+1: both "
            "are the predictor evaluated on the same intermediate state")

    # --- Terminal consistency ---------------------------------------------
    # The final trace belief_after must equal the episode's reported p_hat,
    # because STOP evaluates the predictor on exactly that state.
    assert abs(tr[-1]["belief_after"] - result["p_hat"]) < 1e-12
