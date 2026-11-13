"""Repeated-split cost selection (src/rl/selection.py): same rule as the protocol, lower-variance estimate."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.rl.experiment import DEFAULT_PROTOCOL, Study
from src.rl.selection import repeated_inner_table
from tests.test_rl_mdp import synth


def _study(X, y):
    P = copy.deepcopy(DEFAULT_PROTOCOL)
    P["cost_grid"] = [0.0, 0.02, 0.1]          # small grid keeps the test fast
    return Study(X, y, P, include_dqn=False)


def test_single_repeat_reproduces_the_protocol_selection():
    X, y = synth(n_rec=180, seed=3)
    st = _study(X, y)
    ref = st.inner_tables(X, y)
    rep = repeated_inner_table(st, X, y, repeats=1)
    assert rep["selected"] == ref["selected"]["vi"]
    assert rep["full_ba"] == pytest.approx(ref["full_ba"])
    for c in st.P["cost_grid"]:
        for k in ("balanced_accuracy", "mean_questions", "auroc"):
            assert rep["table"][c][k] == pytest.approx(ref["tables"]["vi"][c][k])


def test_repeats_average_the_per_split_tables_and_count_rule_passes():
    X, y = synth(n_rec=180, seed=4)
    st = _study(X, y)
    rep = repeated_inner_table(st, X, y, repeats=3)
    assert len(rep["per_repeat"]) == 3
    d = st.P["selection"]["delta_select"]
    for c in st.P["cost_grid"]:
        vals = [r["table"][c]["balanced_accuracy"] for r in rep["per_repeat"]]
        assert rep["table"][c]["balanced_accuracy"] == pytest.approx(np.mean(vals))
        manual = sum(r["table"][c]["balanced_accuracy"] >= r["full_ba"] - d for r in rep["per_repeat"])
        assert rep["meets_rule_in_n_repeats"][c] == manual
    assert rep["meets_rule_in_n_repeats"][0.0] == 3            # cost 0 asks everything: equals the full questionnaire
    assert rep["selected"] == st.select(rep["table"], rep["full_ba"])


def test_repeats_must_be_positive():
    X, y = synth(n_rec=60, seed=5)
    with pytest.raises(ValueError):
        repeated_inner_table(_study(X, y), X, y, repeats=0)
