"""Exhaustive fixed-subset oracle vs enumeration — §17 #4"""
import numpy as np
import itertools
from sklearn.metrics import brier_score_loss

def fixed_subset_utility(records, subset, lambda_cost=0.0):
    # Train: compute p_emp for each pattern? Simplified: use full-training logistic? Instead mimic §14 utility: for subset S, terminal p_emp is mean y of support for each pattern.
    # Evaluate empirical utility as 1 - Brier of p_emp on training data with that subset observation.
    # For each record, its pattern is its values on subset, support = records sharing same pattern, p_emp = mean label of support.
    X=np.stack([r["item_responses"] for r in records])
    y=np.array([r["label"] for r in records])
    subset=list(subset)
    if not subset:
        p_emp=y.mean()
        return 1 - np.mean((p_emp - y)**2) - lambda_cost*len(subset)
    # group by pattern
    patterns={}
    for i,row in enumerate(X):
        key=tuple(row[j] for j in subset)
        patterns.setdefault(key, []).append(i)
    brier=0
    for key, idx in patterns.items():
        p=float(y[idx].mean())
        for i in idx:
            brier+=(p - y[i])**2
    brier/=len(y)
    return 1 - brier - lambda_cost*len(subset)

def exhaustive_best_fixed(records, n_items, budget, lambda_cost=0.0):
    best=None; best_subset=None
    for subset in itertools.combinations(range(n_items), budget):
        u=fixed_subset_utility(records, subset, lambda_cost)
        if best is None or u>best:
            best=u; best_subset=subset
    return best, best_subset

def test_exact_fixed_subset_small():
    rng=np.random.default_rng(1)
    n=5; budget=2
    records=[]
    for _ in range(20):
        x=rng.integers(0,2,size=n).astype(float)
        label=int(x[0]==1)  # informative item 0
        records.append({"item_responses":x,"label":label,"label_source":"questionnaire","covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False]*n)})
    best, subset = exhaustive_best_fixed(records, n, budget, 0.0)
    # verify that best is indeed >= any other subset
    for s in itertools.combinations(range(n), budget):
        u=fixed_subset_utility(records, s, 0.0)
        assert best+1e-9 >= u
    assert len(subset)==budget
