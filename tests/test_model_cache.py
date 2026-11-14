"""Verified predictor cache (audit finding T12) shared by the app and the demo (T1).

Synthetic records and temp directories only: no dataset, no ``results/`` files, no real database.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from src.data import ingest
from src.models import artifact_integrity as ai
from src.screening import model_cache as mc

KW = dict(seed=0, n_items=10, calibration="platt")


@pytest.fixture
def records(monkeypatch):
    recs = ingest._synthetic_records(120, 10, label_source="synthetic", seed=3)
    monkeypatch.setattr(mc, "load_dataset", lambda name: list(recs))
    return recs


class Spy:
    """Counts MaskedPredictor.fit calls so a cache hit can be told from a retrain."""
    def __init__(self, monkeypatch):
        self.fits = 0
        real = mc.MaskedPredictor.fit

        def counting(inner, *a, **k):
            self.fits += 1
            return real(inner, *a, **k)
        monkeypatch.setattr(mc.MaskedPredictor, "fit", counting)


def build(tmp_path, **over):
    msgs = {"info": [], "warn": []}
    out = mc.build_predictor(tmp_path, info=msgs["info"].append, warn=msgs["warn"].append, **{**KW, **over})
    return out, msgs


def files(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir())


def probe(pred):
    from src.env.state import init_state
    rec = {"item_responses": np.zeros(10), "missing_mask": np.zeros(10, dtype=bool)}
    st = init_state(rec)
    st["questions_remaining"] = 6
    st["budget"] = 6
    return float(pred(st))


# ------------------------------------------------------------------ hit / miss


def test_a_miss_trains_and_writes_weights_calibrator_and_manifest(records, tmp_path):
    (_, _, _, _, pred), msgs = build(tmp_path)
    names = files(tmp_path)
    assert len(names) == 3 and {n.rsplit(".", 1)[-1] for n in names} == {"pt", "pkl", "json"}
    manifest = json.loads(next(p for p in tmp_path.glob("*.manifest.json")).read_text())
    assert manifest["weights_sha256"] == ai.sha256_file(next(tmp_path.glob("*.pt")))
    assert manifest["calibrator_sha256"] == ai.sha256_file(next(tmp_path.glob("*.pkl")))
    assert manifest["dataset_fingerprint"] and manifest["predictor_version"] == mc.MaskedPredictor.VERSION
    assert any("training" in m for m in msgs["info"]) and not msgs["warn"]


def test_a_hit_loads_without_retraining_and_predicts_the_same(records, tmp_path, monkeypatch):
    (_, _, _, _, first), _ = build(tmp_path)
    spy = Spy(monkeypatch)
    (_, _, _, _, second), msgs = build(tmp_path)
    assert spy.fits == 0, "a verified cache must be used"
    assert probe(first) == pytest.approx(probe(second), abs=1e-9)
    assert any("verified cached predictor" in m for m in msgs["info"])


def test_no_temp_files_are_left_behind(records, tmp_path):
    build(tmp_path)
    assert not [n for n in files(tmp_path) if n.endswith(".tmp")]


# ------------------------------------------------------------------ tampering and legacy caches


def test_tampered_weights_are_rejected_and_replaced(records, tmp_path, monkeypatch):
    build(tmp_path)
    pt = next(tmp_path.glob("*.pt"))
    pt.write_bytes(pt.read_bytes() + b"\x00")
    spy = Spy(monkeypatch)
    _, msgs = build(tmp_path)
    assert spy.fits == 1 and any("does not match" in m for m in msgs["warn"])
    spy2 = Spy(monkeypatch)
    _, msgs2 = build(tmp_path)
    assert spy2.fits == 0 and not msgs2["warn"], "the rewritten cache verifies"


class Boom:
    """Unpickling this runs code. If the hash check works it must never run."""
    ran = False

    def __reduce__(self):
        return (_set_ran, ())


def _set_ran():
    Boom.ran = True
    return None


def test_a_swapped_calibrator_is_rejected_before_it_is_unpickled(records, tmp_path):
    build(tmp_path)
    pkl = next(tmp_path.glob("*.pkl"))
    pkl.write_bytes(pickle.dumps(Boom()))
    Boom.ran = False
    _, msgs = build(tmp_path)
    assert Boom.ran is False, "the pickle must not be executed when its hash does not match"
    assert any("calibrator" in m and "does not match" in m for m in msgs["warn"])


def test_a_cache_without_a_manifest_is_not_trusted(records, tmp_path, monkeypatch):
    build(tmp_path)
    next(tmp_path.glob("*.manifest.json")).unlink()
    spy = Spy(monkeypatch)
    _, msgs = build(tmp_path)
    assert spy.fits == 1 and any("no manifest" in m for m in msgs["warn"])


@pytest.mark.parametrize("field, value", [("seed", 99), ("calibration", "isotonic"), ("dataset_fingerprint", "0" * 16),
                                          ("predictor_version", 0)])
def test_a_manifest_for_a_different_configuration_is_rejected(records, tmp_path, monkeypatch, field, value):
    build(tmp_path)
    mp = next(tmp_path.glob("*.manifest.json"))
    m = json.loads(mp.read_text())
    m[field] = value
    mp.write_text(json.dumps(m))
    spy = Spy(monkeypatch)
    build(tmp_path)
    assert spy.fits == 1


def test_a_manifest_missing_a_hash_is_refused(records, tmp_path, monkeypatch):
    build(tmp_path)
    mp = next(tmp_path.glob("*.manifest.json"))
    m = json.loads(mp.read_text())
    del m["weights_sha256"]
    mp.write_text(json.dumps(m))
    spy = Spy(monkeypatch)
    _, msgs = build(tmp_path)
    assert spy.fits == 1 and any("no valid recorded SHA-256" in x for x in msgs["warn"])


# ------------------------------------------------------------------ cache key and frozen files


def test_changing_the_dataset_changes_the_key_and_keeps_the_old_cache(records, tmp_path, monkeypatch):
    build(tmp_path)
    before = files(tmp_path)
    changed = [dict(r, item_responses=r["item_responses"].copy()) for r in records]
    changed[0]["item_responses"][0] = 1.0 - changed[0]["item_responses"][0]
    monkeypatch.setattr(mc, "load_dataset", lambda name: list(changed))
    spy = Spy(monkeypatch)
    build(tmp_path)
    assert spy.fits == 1, "weights trained on other data must not be reused"
    after = files(tmp_path)
    assert set(before) < set(after) and len(after) == 6, "both caches coexist"


def test_fingerprint_is_stable_and_sensitive(records):
    from src.data.splits import stratified_split
    parts = stratified_split(records, seed=0)
    base = mc.dataset_fingerprint(records, *parts)
    assert base == mc.dataset_fingerprint(records, *parts) and len(base) == 16
    altered = [dict(r) for r in records]
    altered[5] = dict(altered[5], label=1 - int(altered[5]["label"]))
    assert mc.dataset_fingerprint(altered, *parts) != base
    other_parts = stratified_split(records, seed=1)
    assert mc.dataset_fingerprint(records, *other_parts) != base, "a different split is a different model"


def test_legacy_frozen_cache_files_are_never_read_or_modified(records, tmp_path):
    legacy = {"demo_model_saudi_seed0_platt_v2.pt": b"frozen weights", "demo_model_saudi_seed0_platt_v2.pkl": b"frozen cal"}
    for name, data in legacy.items():
        (tmp_path / name).write_bytes(data)
    build(tmp_path)
    for name, data in legacy.items():
        assert (tmp_path / name).read_bytes() == data
    build(tmp_path, retrain=True)
    for name, data in legacy.items():
        assert (tmp_path / name).read_bytes() == data


def test_retrain_flag_ignores_a_valid_cache(records, tmp_path, monkeypatch):
    build(tmp_path)
    spy = Spy(monkeypatch)
    build(tmp_path, retrain=True)
    assert spy.fits == 1


def test_an_unwritable_cache_location_warns_but_still_returns_a_working_predictor(records, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    (_, _, _, _, pred), msgs = build(blocker)
    assert 0.0 <= probe(pred) <= 1.0
    assert any("could not write predictor cache" in m for m in msgs["warn"])


# ------------------------------------------------------------------ integrity helpers


def test_verify_sha256_accepts_a_match_and_rejects_everything_else(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    good = ai.sha256_file(f)
    assert ai.verify_sha256(f, good) == good and ai.verify_sha256(f, good.upper()) == good
    for bad in (None, "", "abc", "z" * 64, "0" * 64, 5):
        with pytest.raises(ai.ArtifactIntegrityError):
            ai.verify_sha256(f, bad)
    with pytest.raises(ai.ArtifactIntegrityError, match="missing"):
        ai.verify_sha256(tmp_path / "absent", good)


def test_load_state_dict_refuses_pickled_objects(tmp_path):
    import torch
    ok = tmp_path / "ok.pt"
    torch.save({"w": torch.zeros(2)}, ok)
    assert "w" in ai.load_state_dict(ok)
    evil = tmp_path / "evil.pt"
    torch.save({"w": Boom()}, evil)
    Boom.ran = False
    with pytest.raises(Exception):
        ai.load_state_dict(evil)
    assert Boom.ran is False


def test_the_frozen_predictor_loader_checks_recorded_hashes(tmp_path):
    """load_frozen_predictor verifies a recorded hash before deserialising, and blocks on a mismatch."""
    import torch
    from src.eval.external_validation import ExternalValidationBlocked, load_frozen_predictor
    pt, pk = tmp_path / "w.pt", tmp_path / "c.pkl"
    torch.save({"w": torch.zeros(1)}, pt)
    Boom.ran = False
    pk.write_bytes(pickle.dumps(Boom()))
    manifest = {"n_items": 10, "hidden": [128, 64], "calibration": "platt", "seed": 0}
    with pytest.raises(ExternalValidationBlocked, match="SHA-256"):
        load_frozen_predictor(pt, pk, manifest, weights_sha256="0" * 64, calibrator_sha256=ai.sha256_file(pk))
    with pytest.raises(ExternalValidationBlocked, match="SHA-256"):
        load_frozen_predictor(pt, pk, manifest, weights_sha256=ai.sha256_file(pt), calibrator_sha256="1" * 64)
    assert Boom.ran is False, "a mismatching pickle must never be executed"


def test_the_frozen_predictor_loader_without_hashes_keeps_its_old_contract(tmp_path):
    from src.eval.external_validation import ExternalValidationBlocked, load_frozen_predictor
    manifest = {"n_items": 10, "hidden": [128, 64], "calibration": "platt", "seed": 0}
    with pytest.raises(ExternalValidationBlocked, match="weights missing"):
        load_frozen_predictor(tmp_path / "a.pt", tmp_path / "a.pkl", manifest, weights_sha256="0" * 64)
