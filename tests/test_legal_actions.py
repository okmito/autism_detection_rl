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
    rec = {"item_responses": np.array([0.0]), "label":0, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"M"}, "missing_mask": np.array([False])}
    calls=[]
    def policy(state, legal):
        calls.append(legal.copy())
        # second call should have STOP legal
        if len(calls)==2:
            assert -1 in legal
        return legal[0] if 0 in legal else -1
    predictor=lambda s:0.5
    run_episode(rec, question_budget=5, b_min=5, policy=policy, predictor=predictor)
    # After asking the single item, only STOP should be legal even though B_min not met? Actually after asking 1, items_asked=1 < B_min=5 but no legal items left => stop_legal True
    assert any(-1 in c for c in calls)
