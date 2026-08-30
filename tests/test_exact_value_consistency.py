import numpy as np
from src.solvers.exact_custom import ExactDP

def test_exact_value_consistency():
    records=[
        {"item_responses":np.array([0.0,0.0]),"label":0,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([0.0,1.0]),"label":0,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([1.0,0.0]),"label":1,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([1.0,1.0]),"label":1,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
    ]
    dp=ExactDP(records, n_items=2, budget=1, b_min=0, lambda_cost=0.0)
    res=dp.solve()
    v_star=res["V_star"]
    # re-evaluate via dp.evaluate_policy using DP's own policy
    def dp_policy(state, legal):
        return dp.get_action(state)
    v_eval=dp.evaluate_policy(dp_policy)
    assert abs(v_star - v_eval) < 1e-9, f"{v_star} vs {v_eval}"
