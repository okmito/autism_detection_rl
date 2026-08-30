import numpy as np
from src.env.state import init_state, encode_state, UNASKED, OBSERVED, MISSING, update_state

def test_unasked_missing_observed_distinct():
    rec = {"item_responses": np.array([0.0, 1.0, np.nan]), "label":0, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"M"}, "missing_mask": np.array([False,False,True])}
    s = init_state(rec)
    assert s["mask"][0]==UNASKED
    assert s["mask"][2]==MISSING
    # encode before observation
    v0 = encode_state({**s, "questions_remaining":3, "budget":3})
    # observe item 0 with value 0
    s2 = update_state(s, 0, 0, 2)
    s2["questions_remaining"]=2; s2["budget"]=3
    v1 = encode_state(s2)
    # observe another with value 1 would be different vector? Check that observed 0 vs UNASKED vs MISSING distinct
    # mask one-hot ensures distinct
    assert not np.allclose(v0, v1)
    # Now test OBSERVED(0) vs OBSERVED(1) distinct
    s3 = update_state(s, 1, 1, 2)
    s3["questions_remaining"]=2; s3["budget"]=3
    v2 = encode_state(s3)
    assert not np.allclose(v1, v2)

def test_round_trip():
    rec = {"item_responses": np.array([1.0,0.0]), "label":1, "label_source":"questionnaire",
           "covariates":{"age_band":"1-2","sex":"F"}, "missing_mask": np.array([False,False])}
    s = init_state(rec)
    s["questions_remaining"]=2; s["budget"]=2
    s2 = update_state(s, 0, 1, 1)
    s2["questions_remaining"]=1; s2["budget"]=2
    assert s2["mask"][0]==OBSERVED
    assert s2["value"][0]==1
    assert s2["mask"][1]==UNASKED
