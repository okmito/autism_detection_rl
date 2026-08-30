import numpy as np
from src.env.environment import run_episode

def test_reward_bounds_uniform():
    rec = {"item_responses": np.array([0.0,1.0]), "label":1, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"M"}, "missing_mask": np.array([False,False])}
    lam=0.1
    def policy(state, legal):
        items=[a for a in legal if a!=-1]
        return items[0] if items else -1
    for p in [0.0,0.5,1.0]:
        pred=lambda s, pv=p: pv
        r=run_episode(rec, question_budget=1, b_min=0, policy=policy, predictor=pred, lambda_cost=lam)
        asked=len(r["items_asked"])
        # R in [-lambda*sum c_j, 1]
        assert r["R"] <= 1.0 + 1e-9
        assert r["R"] >= -lam*asked -1e-9

def test_reward_bounds_nonuniform():
    rec={"item_responses":np.array([1.0,0.0,1.0]),"label":0,"label_source":"questionnaire",
         "covariates":{"age_band":"1-2","sex":"F"},"missing_mask":np.array([False]*3)}
    weights=[2.0,1.0,0.5]
    lam=0.05
    def policy(s,legal):
        items=[a for a in legal if a!=-1]
        return items[0] if items else -1
    r=run_episode(rec, question_budget=2,b_min=0,policy=policy,predictor=lambda s:0.2, lambda_cost=lam, cost_mode="weighted", cost_weights=weights)
    asked_cost=sum(weights[j] for j in r["items_asked"])
    assert r["R"]>= -lam*asked_cost -1e-9
    assert r["R"]<=1.0+1e-9
