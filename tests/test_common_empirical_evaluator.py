import numpy as np
from src.solvers.exact_custom import ExactDP

def test_common_evaluator():
    records=[
        {"item_responses":np.array([0.0,1.0]),"label":0,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([1.0,0.0]),"label":1,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([1.0,1.0]),"label":1,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"F"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([0.0,0.0]),"label":0,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"F"},"missing_mask":np.array([False,False])},
    ]
    dp=ExactDP(records, n_items=2, budget=1, b_min=0, lambda_cost=0.05)
    dp.solve()
    # Two policies evaluated with same evaluator should be comparable
    def policy_always_0(state, legal):
        return 0 if 0 in legal else -1
    def policy_always_1(state, legal):
        return 1 if 1 in legal else -1
    v0=dp.evaluate_policy(policy_always_0)
    v1=dp.evaluate_policy(policy_always_1)
    # V* should be >= both
    v_star=dp._memo[(tuple([0,0]), tuple([-1,-1]),1)]  # root value
    assert v_star >= v0 -1e-9
    assert v_star >= v1 -1e-9
    # evaluator uses same costs and lambda for both
    assert isinstance(v0, float) and isinstance(v1, float)
