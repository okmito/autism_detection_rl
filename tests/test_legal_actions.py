import numpy as np
from src.env.state import init_state, get_legal_items, UNASKED, MISSING, OBSERVED
from src.env.environment import run_episode

def test_missing_never_legal():
    rec = {"item_responses": np.array([0.0, np.nan, 1.0]), "label":0, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"M"}, "missing_mask": np.array([False,True,False])}
    s = init_state(rec)
    legal = get_legal_items(s)
    assert 1 not in legal  # MISSING
    assert 0 in legal and 2 in legal

def test_stop_illegal_below_bmin():
    rec = {"item_responses": np.array([0.0,1.0]), "label":0, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"M"}, "missing_mask": np.array([False,False])}
    calls=[]
    def policy(state, legal):
        calls.append(legal.copy())
        # STOP should not be in legal on first call when b_min=1 and no items asked yet
        if len(calls)==1:
            assert -1 not in legal, "STOP should be illegal below B_min on first step"
        return [a for a in legal if a!=-1][0] if any(a!=-1 for a in legal) else -1
    predictor = lambda s: 0.5
    run_episode(rec, question_budget=2, b_min=1, policy=policy, predictor=predictor)
    # second call should have STOP legal
    assert -1 in calls[1]

def test_stop_forced_when_no_legal():
    """STOP must be available once no unasked item remains, even below B_min.

    Behaviour change (2026-10-01, P0-4): the environment used to *ask the policy*
    in this situation and rely on it returning STOP. It now forces STOP itself,
    because STOP is the only legal action and consulting the policy could only
    ever produce an illegal answer. `stop_reason` therefore reports
    ``budget_exhausted``, which was previously unreachable whenever b_min == 0.
    """
    rec = {"item_responses": np.array([0.0]), "label":0, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"M"}, "missing_mask": np.array([False])}
    calls=[]
    def policy(state, legal):
        calls.append(legal.copy())
        assert 0 in legal, "the only unasked item must be offered"
        assert -1 not in legal, "STOP is illegal while an unasked item remains"
        return 0
    predictor=lambda s:0.5
    result = run_episode(rec, question_budget=5, b_min=5, policy=policy, predictor=predictor)

    # b_min=5 is never met (only 1 item exists) yet the episode terminates.
    assert result["items_asked"] == [0]
    assert result["stop_reason"] == "budget_exhausted"
    # The policy was consulted exactly once: only STOP remained on the second
    # iteration, so no choice was offered.
    assert len(calls) == 1


def test_stop_reason_distinguishes_voluntary_from_exhausted():
    """P0-4: `stop_reason` must tell the two causes of termination apart.

    With b_min == 0 the old expression `"policy_stop" if stop_legal else
    "budget_exhausted"` always evaluated to `policy_stop`, so the `budget_exhausted`
    branch was dead code and any metric keyed on it was a constant.
    """
    rec = {"item_responses": np.zeros(6, dtype=float), "label":0,
           "label_source":"questionnaire", "covariates":{"age_band":"1-2","sex":"M"},
           "missing_mask": np.zeros(6, dtype=bool)}

    def greedy_never_stop(state, legal):
        items = [a for a in legal if a != -1]
        return items[0] if items else -1

    def stop_at_two(state, legal):
        if int((state["mask"] == 1).sum()) >= 2 and -1 in legal:
            return -1
        items = [a for a in legal if a != -1]
        return items[0] if items else -1

    predictor = lambda s: 0.3

    exhausted = run_episode(rec, question_budget=4, b_min=0, policy=greedy_never_stop,
                            predictor=predictor)
    assert len(exhausted["items_asked"]) == 4
    assert exhausted["stop_reason"] == "budget_exhausted"

    voluntary = run_episode(rec, question_budget=4, b_min=0, policy=stop_at_two,
                            predictor=predictor)
    assert len(voluntary["items_asked"]) == 2
    assert voluntary["stop_reason"] == "policy_stop"
