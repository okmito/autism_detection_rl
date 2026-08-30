import numpy as np
from src.env.state import init_state, update_state
from src.explain.counterfactual import find_counterfactual

def test_counterfactual_flips():
    n=3
    # predictor that thresholds sum
    def predictor(state):
        s=sum(int(state["value"][j]) for j in range(n) if state["mask"][j]==1)
        # map sum to prob: 0.2 +0.2*s
        return 0.2+0.2*s
    # state with 2 observed items 1,1 => p=0.6 => REFER (>=0.5)
    rec={"item_responses":np.array([1.0,1.0,0.0]),"label":1,"label_source":"questionnaire",
         "covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False]*3)}
    s=init_state(rec); s["questions_remaining"]=1; s["budget"]=3
    s=update_state(s,0,1,1); s["questions_remaining"]=1; s["budget"]=3
    s=update_state(s,1,1,0); s["questions_remaining"]=0; s["budget"]=3
    s["questions_remaining"]=0; s["budget"]=3
    p=predictor(s)
    assert p>=0.5
    cf=find_counterfactual(s, p, predictor, tau=0.5)
    if cf is not None:
        # apply flip and verify decision flips
        s2={"mask":s["mask"].copy(),"value":s["value"].copy(),"n":s["n"],"questions_remaining":0,"budget":3}
        s2["value"][cf["item_idx"]]=cf["flipped_value"]
        p2=predictor(s2)
        assert (p2>=0.5) != (p>=0.5)
    else:
        # if None, verify robust: all single flips keep same decision
        for j in range(n):
            if s["mask"][j]!=1: continue
            s2={"mask":s["mask"].copy(),"value":s["value"].copy(),"n":s["n"],"questions_remaining":0,"budget":3}
            s2["value"][j]=1-s2["value"][j]
            p2=predictor(s2)
            assert (p2>=0.5)==(p>=0.5)

def test_counterfactual_robust():
    n=2
    def predictor(state):
        return 0.9  # always high
    rec={"item_responses":np.array([0.0,0.0]),"label":1,"label_source":"questionnaire",
         "covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False]*2)}
    s=init_state(rec); s["questions_remaining"]=0; s["budget"]=2
    s=update_state(s,0,0,0); s["questions_remaining"]=0; s["budget"]=2
    cf=find_counterfactual(s, predictor(s), predictor, tau=0.5)
    assert cf is None
