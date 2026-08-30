import numpy as np
from src.env.environment import run_episode

def test_budget_exhaustion():
    rec = {"item_responses": np.array([0.0,1.0,0.0,1.0]), "label":1, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"M"}, "missing_mask": np.array([False]*4)}
    def greedy(state, legal):
        # always pick item if available
        items=[a for a in legal if a!=-1]
        return items[0] if items else -1
    predictor=lambda s:0.6
    result = run_episode(rec, question_budget=2, b_min=0, policy=greedy, predictor=predictor)
    assert len(result["items_asked"])<=2
    assert len(result["items_asked"])==2 or result["stop_reason"]=="policy_stop"
    # No repeated selection
    assert len(set(result["items_asked"]))==len(result["items_asked"])

def test_no_repeat():
    rec = {"item_responses": np.array([0.0,1.0,1.0]), "label":0, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"F"}, "missing_mask": np.array([False]*3)}
    def policy(state, legal):
        # try to pick same item twice would be illegal — ensure env prevents
        items=[a for a in legal if a!=-1]
        return items[0] if items else -1
    result=run_episode(rec, question_budget=3, b_min=0, policy=policy, predictor=lambda s:0.5)
    assert len(set(result["items_asked"]))==len(result["items_asked"])
