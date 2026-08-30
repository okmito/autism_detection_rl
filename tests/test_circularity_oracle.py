import numpy as np
from src.audits.circularity import audit_circularity

def test_circularity_oracle():
    # deterministic threshold labels: sum >=2 =>1
    rng=np.random.default_rng(0)
    records=[]
    for _ in range(100):
        x=rng.integers(0,2,size=5).astype(float)
        label=int(x.sum()>=2)
        records.append({"item_responses":x,"label":label,"label_source":"questionnaire",
                        "covariates":{"age_band":"1-2","sex":"M"},"missing_mask":np.array([False]*5)})
    report=audit_circularity(records, thresholds=range(6))
    assert report["classification"]=="Deterministic"
    assert report["exact_match_rate"]==1.0
    assert report["best_threshold"]==2
