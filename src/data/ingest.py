"""Dataset ingest — one loader per tier in §15.
All loaders return list[dict] conforming to common schema (§15 + schema.py).
Do NOT modify raw files; all transforms are in-memory.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import warnings

# ---------------------------------------------------------------------------
# Paths — verified local artifacts (do not rename raw files)
# ---------------------------------------------------------------------------
SAUDI_CSV = Path(r"data/raw/Q-CHAT Saudi Arabia/Autism Spectrum Disorder Screening Data for Toddlers in Saudi Arabia Data Set.csv")
POLISH_CSV = Path(r"data/raw/Q-CHAT Polish/polish_qchat.csv")
UCI_CHILD_ARFF = Path(r"data/raw/UCI/Autism-Child-Data.arff")
# NZ 1,054-row Toddler Autism dataset July 2018.csv — NOT PRESENT LOCALLY
NZ_TODDLER_CSV = Path(r"data/raw/Q-CHAT NZ/Toddler Autism dataset July 2018.csv")
NZ_COMBINED_CSV = Path(r"data/raw/Q-CHAT NZ/Autism_Screening_Data_Combined.csv")  # 6075 pooled — NOT primary cohort

# ---------------------------------------------------------------------------
# Binary mapping for Q-CHAT-10 primary representation (§15)
# ---------------------------------------------------------------------------
def qchat10_binary_map(raw: np.ndarray) -> np.ndarray:
    out = np.full_like(raw, np.nan, dtype=float)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            if np.isnan(v):
                continue
            v = int(v)
            if j < 9:
                out[i, j] = 1 if v >= 2 else 0
            else:
                out[i, j] = 1 if v <= 2 else 0
    return out


def _synthetic_records(n: int, n_items: int, label_source: str = "questionnaire", seed: int = 0) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        raw = rng.integers(0, 2, size=n_items).astype(float)
        missing = rng.random(n_items) < 0.05
        raw[missing] = np.nan
        label = int(rng.integers(0, 2))
        records.append({
            "item_responses": raw,
            "label": label,
            "label_source": label_source,
            "covariates": {"age_band": rng.choice(["1-2", "2-3", "3-5"]), "sex": rng.choice(["M", "F"])},
            "missing_mask": missing,
            "provenance": f"synthetic_{n}_{n_items}_{seed}",
        })
    return records


# ---------------------------------------------------------------------------
# NZ — PRIMARY 1,054 COHORT
# ---------------------------------------------------------------------------
def load_nz(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """NZ Q-CHAT-10 — 1,054 rows (spec §15: 728/326). Requires V-1."""
    if synthetic:
        return _synthetic_records(1054, 10, "questionnaire", seed=1)
    if NZ_TODDLER_CSV.exists():
        return _load_nz_toddler_csv(NZ_TODDLER_CSV)
    raise FileNotFoundError(
        "HUMAN ACTION REQUIRED — NZ primary cohort missing: "
        f"Expected 1,054-row file at '{NZ_TODDLER_CSV}' (Toddler Autism dataset July 2018.csv). "
        f"The only file present is '{NZ_COMBINED_CSV}' with 6075 pooled rows — "
        "do NOT use it as the primary NZ cohort per instructions. "
        "Obtain the 1,054-row toddler file and place it at the path above."
    )


def _load_nz_toddler_csv(path: Path) -> List[Dict[str, Any]]:
    # Schema: Case_No, A1..A10 (0/1), Age_Mons, Qchat-10-Score, Sex, Ethnicity, Jaundice, Family_mem_with_ASD, Who completed the test, Class/ASD Traits
    # Not executed until file present; implement faithfully when available
    df = pd.read_csv(path, encoding="utf-8")
    raise NotImplementedError("NZ toddler loader pending file provenance verification — do not infer mapping until file inspected.")


# ---------------------------------------------------------------------------
# SAUDI — VERIFIED 506 ROWS
# ---------------------------------------------------------------------------
def load_saudi(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """Saudi Q-CHAT-10 — 506 rows (spec §15). Verified at SAUDI_CSV."""
    if synthetic:
        return _synthetic_records(506, 10, "questionnaire", seed=2)
    if SAUDI_CSV.exists():
        return _load_saudi_csv(SAUDI_CSV)
    raise FileNotFoundError(f"HUMAN ACTION REQUIRED: Saudi CSV not found at '{SAUDI_CSV}'.")


def _load_saudi_csv(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    # Header order is A10..A1 reversed — map explicitly to A1..A10
    # Columns: A10,A9,A8,A7,A6,A5,A4,A3,A2,A1, Region, Family mem..., Who..., Age, Gender, Screening Score, Class
    required = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Saudi CSV missing expected column {c}; columns={list(df.columns)}")
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        # Binary 0/1 already recoded; preserve as float
        vals = np.array([float(row[f"A{j}"]) for j in range(1, 11)], dtype=float)  # A1..A10 in order
        missing = np.isnan(vals)  # none expected; 0 verified
        # Class is 0/1 int, Screening Score == sum validates determinism
        label = int(row["Class"])
        # Age is months? Already toddler 12-36; keep as age_band for covariates
        age = int(row["Age"])
        # Map age to band per spec (not used for policy state, but for subgroup)
        if age <= 18:
            age_band = "12-18"
        elif age <= 24:
            age_band = "18-24"
        elif age <= 30:
            age_band = "24-30"
        else:
            age_band = "30-36"
        sex_raw = str(row["Gender"]).strip().lower()
        sex = "M" if sex_raw.startswith("m") else ("F" if sex_raw.startswith("f") else "unknown")
        records.append({
            "item_responses": vals,
            "label": label,
            "label_source": "questionnaire",
            "covariates": {"age_band": age_band, "sex": sex, "region": str(row.get("Region", "")), "raw_age": age},
            "missing_mask": missing,
            "provenance": f"saudi:{path.name}:{idx}",
            "raw_screening_score": int(row["Screening Score"]),
        })
    # Provenance check: Screening Score == sum(A) must hold 506/506 — log warning if not
    mism = sum(1 for r in records if int(r["item_responses"].sum()) != r["raw_screening_score"])
    if mism:
        warnings.warn(f"Saudi: {mism}/506 rows where sum(A)!=Screening Score — provenance issue.")
    return records


# ---------------------------------------------------------------------------
# POLISH — VERIFIED 252 ROWS, Q-CHAT-25 MULTI-CATEGORY
# ---------------------------------------------------------------------------
_VALID_QCHAT_VALUES = {
    # Normalised by stripping quotes; raw file values include Polish strings
    # Item 1: 5 levels
    "qchat1recode": {"zawsze", "zazwyczaj", "czasami", "rzadko", "nigdy"},
    "qchat2recode": {"b. trudno", "dość trudno", "dość łatwo", "b.łatwo"},
    "qchat3recode": {"zawsze", "zazwyczaj", "czasami", "rzadko", "nigdy"},
    "qchat4recode": {"zawsze", "zazwyczaj", "czasami", "rzadko", "nigdy", "nigdy lub nie mówi"},
    "qchat5recode": {"nigdy", "mniej niż raz/tydzień", "kilka razy/tydzień", "kilka razy/dzień", "wiele razy/dzień"},
    # remaining items follow 5-level pattern — validated generically below
}

# Polish categorical strings are loaded as raw strings; for sequential env the
# categorical representation is preserved. The loader keeps raw strings and also
# provides an integer-encoded array for convenience, with NaN for invalid.
_POLISH_INVALID_LOG: List[Dict[str, Any]] = []


def load_polish(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """Polish Q-CHAT-25 — 252 rows (135 ASD / 117 control). Verified at POLISH_CSV.
    qchat4 value '11.0' is treated as invalid data value and logged, not silently converted.
    """
    if synthetic:
        return _synthetic_records(252, 25, "clinical", seed=6)
    if POLISH_CSV.exists():
        return _load_polish_csv(POLISH_CSV)
    raise FileNotFoundError(f"HUMAN ACTION REQUIRED: Polish CSV not found at '{POLISH_CSV}'.")


def _load_polish_csv(path: Path) -> List[Dict[str, Any]]:
    global _POLISH_INVALID_LOG
    _POLISH_INVALID_LOG = []
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    # Verify shape
    if len(df) != 252:
        warnings.warn(f"Polish CSV row count {len(df)} != expected 252 (spec §15 V-7).")
    q_cols = [f"qchat{i}recode" for i in range(1, 26)]
    for c in q_cols:
        if c not in df.columns:
            raise ValueError(f"Polish CSV missing column {c}")

    # Precompute per-item vocabulary (sorted, excluding invalid 11.0) for stable integer encoding
    vocab_per_item: List[Dict[str, int]] = []
    for col in q_cols:
        vals = sorted(v for v in df[col].astype(str).unique() if v != "11.0")
        vocab_per_item.append({v: i for i, v in enumerate(vals)})
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        age = float(row["age"])
        sex_raw = str(row["sex"]).strip().lower()
        sex = "M" if sex_raw.startswith("m") else ("F" if sex_raw.startswith("f") else "unknown")
        age_band = f"{int(age)}m"
        group = str(row["group"]).strip()
        label = 1 if group == "ASD" else 0
        raw_vals = []
        missing_mask_items = np.zeros(25, dtype=bool)
        encoded = np.full(25, np.nan, dtype=float)
        for j, col in enumerate(q_cols):
            raw = str(row[col]).strip()
            raw_vals.append(raw)
            if raw == "11.0":
                _POLISH_INVALID_LOG.append({"row": int(idx), "child_id": str(row.get("child_id", "")), "col": col, "value": raw})
                missing_mask_items[j] = True
                encoded[j] = np.nan
            elif raw in ("", "nan", "NaN", "None"):
                missing_mask_items[j] = True
                encoded[j] = np.nan
            else:
                missing_mask_items[j] = False
                # deterministic integer encoding for valid categorical response
                encoded[j] = float(vocab_per_item[j][raw])
        records.append({
            "item_responses": encoded,
            "label": label,
            "label_source": "clinical",
            "covariates": {"age_band": age_band, "sex": sex},
            "missing_mask": missing_mask_items,
            "provenance": f"polish:{path.name}:{idx}",
            "raw_qchat": raw_vals,
            "group": group,
        })
    if _POLISH_INVALID_LOG:
        warnings.warn(f"Polish: {_POLISH_INVALID_LOG} invalid values logged (qchat4 11.0 treated as missing).")
    return records


def get_polish_invalid_log() -> List[Dict[str, Any]]:
    return list(_POLISH_INVALID_LOG)


# ---------------------------------------------------------------------------
# UCI — VERIFIED 292 CHILD ROWS (ID 419)
# ---------------------------------------------------------------------------
def load_uci_child(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if synthetic:
        return _synthetic_records(292, 10, "questionnaire", seed=3)
    if UCI_CHILD_ARFF.exists():
        return _load_uci_arff(UCI_CHILD_ARFF)
    raise FileNotFoundError(f"HUMAN ACTION REQUIRED: UCI Child ARFF not found at '{UCI_CHILD_ARFF}'.")


def _load_uci_arff(path: Path) -> List[Dict[str, Any]]:
    # Minimal ARFF parser for this file (avoid scipy dependency)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Find @data
    data_start = None
    for i, l in enumerate(lines):
        if l.strip().lower() == "@data":
            data_start = i
            break
    if data_start is None:
        raise ValueError("ARFF missing @data")
    header = lines[:data_start]
    # Attributes in order: A1..A10, age, gender, ethnicity, jundice, austim, contry_of_res, used_app_before, result, age_desc, relation, Class/ASD
    records: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines[data_start + 1:]):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        # Split respecting quoted commas (simple: split by comma not inside ')
        import csv, io
        reader = csv.reader(io.StringIO(line), quotechar="'", skipinitialspace=True)
        fields = next(reader)
        # fields length 21
        if len(fields) != 21:
            raise ValueError(f"UCI row {idx} has {len(fields)} fields, expected 21: {line[:200]}")
        # A1..A10 are 0/1
        a_vals = []
        missing_items = []
        for j in range(10):
            v = fields[j].strip()
            if v == "?" or v == "":
                a_vals.append(np.nan)
                missing_items.append(True)
            else:
                a_vals.append(float(int(v)))
                missing_items.append(False)
        a_arr = np.array(a_vals, dtype=float)
        miss_arr = np.array(missing_items, dtype=bool)
        # Demographics: age, gender, ethnicity (may be ?)
        gender = fields[11].strip()
        sex = {"m": "M", "f": "F"}.get(gender.lower(), "unknown") if gender != "?" else "unknown"
        # age numeric
        try:
            age_f = float(fields[10])
            age_band = "4-11 years"  # all are child
        except Exception:
            age_band = "unknown"
        # label
        label_raw = fields[20].strip()
        label = 1 if label_raw.upper() == "YES" else 0
        # ethnicity/relation may contain ? — track overall row missing for covariates but item missing already above
        records.append({
            "item_responses": a_arr,
            "label": label,
            "label_source": "questionnaire",
            "covariates": {"age_band": age_band, "sex": sex, "ethnicity": fields[12], "relation": fields[19]},
            "missing_mask": miss_arr,
            "provenance": f"uci_child:{path.name}:{idx}",
            "raw_fields": fields,
        })
    if len(records) != 292:
        warnings.warn(f"UCI Child ARFF row count {len(records)} != expected 292.")
    return records


def load_uci_adol(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if synthetic:
        return _synthetic_records(104, 10, "questionnaire", seed=4)
    raise FileNotFoundError("HUMAN ACTION REQUIRED — UCI Adolescent 104-row ARFF not present locally (spec §15: UCI AQ-10 adolescent). Place at data/raw/UCI/Autism-Adolescent-Data.arff.")


def load_uci_adult(synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if synthetic:
        return _synthetic_records(704, 10, "questionnaire", seed=5)
    raise FileNotFoundError("HUMAN ACTION REQUIRED — UCI Adult 704-row ARFF not present locally (spec §15: UCI AQ-10 adult). Place at data/raw/UCI/Autism-Adult-Data.arff.")


REGISTRY = {
    "nz": load_nz,
    "saudi": load_saudi,
    "uci_child": load_uci_child,
    "uci_adol": load_uci_adol,
    "uci_adult": load_uci_adult,
    "polish": load_polish,
}
LOADERS = REGISTRY


def load_dataset(name: str, synthetic: bool = False, **kwargs) -> List[Dict[str, Any]]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name}. Available: {list(LOADERS)}")
    return LOADERS[name](synthetic=synthetic, **kwargs)
