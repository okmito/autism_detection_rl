from src.audits.leakage import LeakageTracker, poisoned_control_demo
import numpy as np

def test_no_leakage_pass():
    tracker=LeakageTracker()
    tracker.record_fit("scaler", train_indices=[0,1,2])
    # held-out is 3,4 -> no overlap
    violations=tracker.check_no_leakage([3,4])
    assert violations==[]

def test_no_leakage_fail():
    tracker=LeakageTracker()
    tracker.record_fit("scaler", train_indices=[0,1,2,3])
    violations=tracker.check_no_leakage([2,3])
    assert len(violations)>0

def test_poisoned_control():
    assert poisoned_control_demo() is True
