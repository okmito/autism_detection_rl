import numpy as np
import itertools
from src.solvers.exact_custom import ExactDP

def brute_force_best(records, n_items, budget, b_min, lambda_cost):
    # enumerate all deterministic policies via brute force search over decision tree?
    # For small n,budget we enumerate all possible policies as mapping from state to action and compute value via DP simulation.
    # Simpler: compare DP value vs enumeration of all fixed item subsets? Not sufficient.
    # Instead enumerate all possible adaptive policies via recursive enumeration of action choices at each state.
    # For tiny n=2, budget=1, we can enumerate.
    dp = ExactDP(records, n_items=n_items, budget=budget, b_min=b_min, lambda_cost=lambda_cost)
    res = dp.solve()
    v_star = res["V_star"]
    # brute force: enumerate all deterministic policies (for n=2,B=1 there are few)
    # Policy is defined by root action and child actions (but depth 1 so only root)
    # For budget 1, any legal item or STOP (if allowed)
    # Evaluate each via dp.evaluate_policy
    candidates=[]
    # STOP
    def make_policy(action_at_root):
        def pol(state, legal):
            # only root state matters for B=1
            if action_at_root in legal:
                return action_at_root
            # fallback
            return legal[0]
        return pol
    # enumerate items+STOP
    for a in list(range(n_items))+[-1]:
        # check legality at root (MISSING not legal but we assume no missing in toy)
        # evaluate
        v = dp.evaluate_policy(make_policy(a))
        candidates.append(v)
    # For B=1, optimum should be max candidates
    assert v_star >= max(candidates) -1e-9, f"DP {v_star} < brute {max(candidates)}"
    # Also assert DP >= each
    for c in candidates:
        assert v_star +1e-9 >= c

def test_exact_optimality_small():
    # toy dataset: 4 records, n=2
    records=[
        {"item_responses":np.array([0.0,0.0]),"label":0,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([0.0,1.0]),"label":0,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([1.0,0.0]),"label":1,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
        {"item_responses":np.array([1.0,1.0]),"label":1,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False,False])},
    ]
    brute_force_best(records, n_items=2, budget=1, b_min=0, lambda_cost=0.0)
    brute_force_best(records, n_items=2, budget=1, b_min=0, lambda_cost=0.1)
