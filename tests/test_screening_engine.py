"""The shared screening engine (audit findings T1, T8 and the C4 guard at its source).

``src/screening`` is used by both the web app and the demo server. The pure-unit tests below need no
dataset; the last block checks the app's use of it and needs the (gitignored) Saudi data.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.env.environment import STOP
from src.env.state import MISSING, OBSERVED, UNASKED, get_legal_items
from src.screening import session as engine
from src.screening.diagnostics import ITEMS, item_payload, selection_diagnostic, selection_summary

N = 10


def make_records(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        items = rng.integers(0, 2, size=N).astype(float)
        out.append({"item_responses": items, "missing_mask": np.zeros(N, dtype=bool),
                    "label": int(items.sum() >= 4)})
    return out


def make_ctx(predict=lambda state: 0.5, budget=3, **over):
    records = make_records()
    return engine.Context(predict=predict, predictor=None, train=records, records=records, n_items=N,
                          budget=budget, tau=0.5, tag="test", input_reference="test", **over)


# ---------------------------------------------------------------- states (T8)


def test_blank_state_has_every_item_unasked_and_nothing_bound_to_a_record():
    state = engine.blank_state(N, 6)
    assert list(state["mask"]) == [UNASKED] * N and list(state["value"]) == [-1] * N
    assert state["questions_remaining"] == 6 and state["budget"] == 6
    assert get_legal_items(state) == list(range(N))


def test_record_state_inherits_that_records_missingness():
    rec = make_records(1)[0]
    rec["missing_mask"][3] = True
    state = engine.record_state(rec, 6)
    assert state["mask"][3] == MISSING
    assert 3 not in get_legal_items(state)


# ---------------------------------------------------------------- C4 guard


@pytest.mark.parametrize("bad", [STOP, -1, N, N + 5])
def test_require_legal_item_rejects_stop_and_out_of_range(bad):
    with pytest.raises(ValueError):
        engine.require_legal_item(bad, list(range(N)) + [STOP], N)


def test_require_legal_item_rejects_an_item_that_is_not_currently_legal():
    with pytest.raises(ValueError):
        engine.require_legal_item(2, [0, 1], N)
    assert engine.require_legal_item(1, [0, 1], N) == 1


def test_next_question_never_returns_stop_even_for_a_policy_that_wants_it():
    ctx = make_ctx()
    state = engine.blank_state(N, 3)
    with pytest.raises(ValueError):
        engine.next_question(ctx, lambda s, legal: STOP, state)


# ---------------------------------------------------------------- stepping


def test_apply_answer_records_the_answer_and_does_not_mutate_the_old_state():
    from src.policies.greedy import GreedyIGPolicy
    ctx = make_ctx(predict=lambda s: 0.1 + 0.05 * int((s["mask"] == OBSERVED).sum()))
    policy = GreedyIGPolicy(ctx.train, N)
    state = engine.blank_state(N, 3)
    j, _ = engine.next_question(ctx, policy, state)
    before = state["mask"].copy()
    new_state, step = engine.apply_answer(ctx, policy, state, j, 1, asked_before=0)
    assert np.array_equal(state["mask"], before), "input state must be untouched"
    assert new_state["mask"][j] == OBSERVED and new_state["value"][j] == 1
    assert new_state["questions_remaining"] == 2 and new_state["budget"] == 3
    assert step["step"] == 1 and step["item"] == f"A{j + 1}" and step["value"] == 1
    assert step["belief_before"] == pytest.approx(0.1) and step["belief_after"] == pytest.approx(0.15)
    assert set(step["selection"]) >= {"applicable", "support_size", "ig_spread"}


def test_is_exhausted_by_budget_and_by_running_out_of_items():
    ctx = make_ctx(budget=3)
    state = engine.blank_state(N, 3)
    assert not engine.is_exhausted(ctx, state, 2) and engine.is_exhausted(ctx, state, 3)
    state["mask"][:] = OBSERVED
    assert engine.is_exhausted(ctx, state, 0), "no legal items left"


def test_decision_threshold_is_inclusive():
    assert engine.decision_for(0.5, 0.5) == engine.REFERRAL
    assert engine.decision_for(0.4999, 0.5) == engine.NO_REFERRAL


def test_replayed_diagnostics_only_for_greedy():
    from src.policies.random_policy import RandomPolicy
    ctx = make_ctx()
    rec = make_records(1)[0]
    assert engine.replayed_diagnostics(ctx, RandomPolicy(seed=0), rec, [0, 1]) == []


# ---------------------------------------------------------------- diagnostics


def test_item_payload_matches_the_questionnaire_contract():
    assert len(ITEMS) == N
    p = item_payload(0)
    assert p == {"index": 0, "code": "A1", "text": ITEMS[0]}


def test_selection_summary_handles_empty_and_diagnosed_traces():
    assert selection_summary([])["steps_diagnosed"] == 0
    trace = [{"selection": {"ig_spread": 0.0, "criterion_vacuous": True, "support_is_pure": True, "support_size": 12}},
             {"selection": {"ig_spread": 0.3, "criterion_vacuous": False, "support_is_pure": False, "support_size": 5}}]
    s = selection_summary(trace)
    assert (s["steps_diagnosed"], s["steps_criterion_vacuous"], s["steps_support_pure"]) == (2, 1, 1)
    assert s["support_size_at_stop"] == 5


def test_a_policy_without_support_gets_a_not_applicable_diagnostic():
    out = selection_diagnostic(lambda s, l: 0, engine.blank_state(N, 3), list(range(N)))
    assert out["applicable"] is False and out["top_items"] == []


# ---------------------------------------------------------------- the app on the engine (needs the dataset)


def _ok(resp):
    body = resp.get_json()
    assert resp.status_code == 200, body
    return body["data"]


def test_interactive_sessions_are_not_bound_to_a_dataset_record(screening_api):
    """T8: a real respondent must not inherit an unrelated held-out record's label or missingness."""
    from app import db
    sid = _ok(screening_api.post("/api/screening/start", json={"mode": "interactive", "policy": "greedy"}))["session_id"]
    row = db.get_conn().execute("SELECT record_index, true_label, state_json FROM screening_sessions WHERE id=?", (sid,)).fetchone()
    assert row["record_index"] is None and row["true_label"] is None
    import json
    state = json.loads(row["state_json"])
    assert state["mask"] == [UNASKED] * N and state["value"] == [-1] * N


def test_interactive_results_carry_no_true_label_but_auto_results_do(screening_api):
    from tests.test_app_screening import answer, start
    first = start(screening_api)
    sid, idx = first["session_id"], first["question"]["index"]
    answer(screening_api, sid, 1, idx)
    result = answer(screening_api, sid, None, stop=True)["result"]
    assert "true_label" not in result
    auto = _ok(screening_api.post("/api/screening/start", json={"mode": "auto", "policy": "greedy"}))["result"]
    assert auto["true_label"] in (0, 1)
    assert "selection_summary" in result and "selection_summary" in auto


def test_auto_results_attach_a_diagnostic_to_every_greedy_step(screening_api):
    auto = _ok(screening_api.post("/api/screening/start", json={"mode": "auto", "policy": "greedy"}))["result"]
    assert auto["trace"] and all(t.get("selection") for t in auto["trace"])


# ---------------------------------------------------------------- the RL lab replay must not drift from run_episode (T1)


@pytest.mark.parametrize("scenario", ["syn_typical", "syn_mixed", "syn_elevated", "syn_borderline"])
@pytest.mark.parametrize("budget", [2, 6])
def test_rl_lab_replay_agrees_with_run_episode(screening_api, scenario, budget):
    """The lab keeps its own instrumented loop (it records a snapshot per phase), so pin its outcome to the
    canonical ``run_episode`` for the same record, policy and budget."""
    from app.config import TAU_DEFAULT
    from app.services import rl_lab, screening
    from src.env.environment import run_episode

    screening.ensure_ready()
    record = rl_lab._case_record(scenario)
    policy = screening.make_policy("greedy")
    ep = run_episode(record, budget, 0, policy, screening.predict, 0.0, tau=TAU_DEFAULT)
    out = rl_lab.replay(scenario, "greedy", budget)
    assert out["ok"]
    assert [t["item"] for t in out["trace"]] == [f"A{j + 1}" for j in ep["items_asked"]]
    assert out["summary"]["final_p_hat"] == pytest.approx(ep["p_hat"], abs=1e-9)
    assert out["summary"]["stop_reason"] == ep["stop_reason"]
    assert out["summary"]["decision"] == engine.decision_for(ep["p_hat"], TAU_DEFAULT)


def test_interactive_start_never_reads_the_held_out_records(screening_api, monkeypatch):
    """T8 on any dataset: with the held-out records gone, an interactive session still starts. (On the Saudi
    data a record-seeded state happens to equal a blank one, so a state comparison alone cannot tell them apart.)"""
    from app.services import screening
    screening.ensure_ready()
    monkeypatch.setitem(screening._STATE, "test", [])
    monkeypatch.setitem(screening._STATE, "counter", 0)
    out = _ok(screening_api.post("/api/screening/start", json={"mode": "interactive", "policy": "greedy"}))
    assert out["question"]["code"].startswith("A")
    assert screening._STATE["counter"] == 0, "interactive starts must not consume the replay record counter"
