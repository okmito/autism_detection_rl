from src.env.state import reachable_state_count

def test_binary_10_6():
    assert reachable_state_count(10,6,m=2)==26025

def test_qchat25_canonical():
    # 24 items with 5 categories + item 4 with 6 => 25 items
    m_list=[5]*25
    m_list[3]=6  # item 4 (0-indexed 3)
    count = reachable_state_count(25,6,m_list=m_list)
    assert count==3081146397

def test_heterogeneous_matches_formula():
    # simple check small n
    assert reachable_state_count(3,2,m_list=[2,3,2])== 1 + (2+3+2) + (2*3+2*2+3*2)  # k=0 + k=1 + k=2

def test_adi_r_theoretical():
    # 93 binary items, B=6
    assert reachable_state_count(93,6,m=2)==50494563219
