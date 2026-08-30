import numpy as np
import torch
from src.models.masked_predictor import MaskedPredictor
from src.env.state import init_state, update_state

def test_predictor_all_masks():
    n=4
    pred=MaskedPredictor(n_items=n, hidden=[32,16])
    # need at least some training to avoid random output NaN? Random weights still produce valid prob
    # Test all-UNASKED
    rec={"item_responses":np.array([0.0,1.0,0.0,1.0]),"label":0,"label_source":"questionnaire",
         "covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False]*4)}
    s=init_state(rec); s["questions_remaining"]=4; s["budget"]=4
    p=pred(s)
    assert 0<=p<=1
    # partially observed
    s2=update_state(s,0,0,3); s2["questions_remaining"]=3; s2["budget"]=4
    p2=pred(s2)
    assert 0<=p2<=1
    # fully observed
    s3=s
    for j in range(n):
        s3=update_state(s3,j,int(rec["item_responses"][j]),0)
        s3["budget"]=4; s3["questions_remaining"]=0
    s3["budget"]=4; s3["questions_remaining"]=0
    p3=pred(s3)
    assert 0<=p3<=1
    # with missing
    rec2={"item_responses":np.array([0.0,np.nan,1.0,0.0]),"label":1,"label_source":"questionnaire",
          "covariates":{"age_band":"1-2","sex":"F"},"missing_mask":np.array([False,True,False,False])}
    s=init_state(rec2); s["questions_remaining"]=3; s["budget"]=4
    p=pred(s)
    assert 0<=p<=1
