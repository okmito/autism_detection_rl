"""Authoritative backward-induction DP — §14.1
Exact Bellman recursion over empirical training support.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
from collections import defaultdict

STOP = -1

class ExactDP:
    def __init__(self, records: List[Dict[str, Any]], n_items: int, budget: int, b_min: int = 0,
                 lambda_cost: float = 0.0, cost_mode: str = "uniform", cost_weights=None,
                 max_states: int = 50_000_000, max_hours: float = 24, max_ram_gb: float = 32):
        self.records = records
        self.n_items = n_items
        self.B = budget
        self.b_min = b_min
        self.lambda_cost = lambda_cost
        from src.env.costs import get_costs
        self.costs = get_costs(n_items, mode=cost_mode, weights=cost_weights)
        self.max_states = max_states
        self.max_hours = max_hours
        # Precompute arrays
        self.X = np.stack([r["item_responses"] for r in records])  # (N,n) with NaN
        self.y = np.array([r["label"] for r in records], dtype=int)
        self.missing_mask = np.stack([r["missing_mask"] for r in records])
        self.N = len(records)
        self._memo: Dict[Tuple, float] = {}
        self._policy: Dict[Tuple, int] = {}
        self.n_states_visited = 0
        self.n_evals = 0
        self.t0 = None

    def _state_key(self, mask: np.ndarray, value: np.ndarray, b: int) -> Tuple:
        # Only encode OBSERVED positions; UNASKED vs MISSING distinguished via mask
        # Use tuple of (mask tuple, value tuple for observed, b)
        # Optimize: mask as tuple
        return (tuple(mask.tolist()), tuple(value.tolist()), b)

    def _support_indices(self, mask: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Return indices of training records consistent with observed state.
        A record is consistent if for every OBSERVED j, its value matches and not missing.
        MISSING items in state correspond to structurally missing — they match only missing records?
        Actually MISSING means structurally absent from source record; for DP we only consider
        states reachable from actual records. So during DP, missing handling: if state has MISSING at j,
        then consistent records must have missing_mask True at j. If state has OBSERVED, record must have same value.
        UNASKED matches anything.
        """
        # vectorized
        # Start with all True
        consistent = np.ones(self.N, dtype=bool)
        for j in range(self.n_items):
            m = mask[j]
            if m == 1:  # OBSERVED
                v = value[j]
                # record must have not missing and value == v
                consistent &= (~self.missing_mask[:, j]) & (self.X[:, j] == v)
            elif m == 2:  # MISSING
                consistent &= self.missing_mask[:, j]
            # UNASKED => no constraint
            if not np.any(consistent):
                break
        return np.where(consistent)[0]

    def _p_emp_and_utility(self, idx: np.ndarray) -> Tuple[float, float]:
        if len(idx) == 0:
            return 0.5, 0.0  # unreachable — should not be called
        y_s = self.y[idx]
        p_emp = float(y_s.mean())
        # U_stop = 1 - mean((p_emp - y)^2)
        brier = np.mean((p_emp - y_s) ** 2)
        return p_emp, float(1 - brier)

    def _V(self, mask: np.ndarray, value: np.ndarray, b: int, asked_count: int, asked_set: Tuple[int, ...]) -> float:
        key = self._state_key(mask, value, b)
        if key in self._memo:
            return self._memo[key]
        # resource limits
        if self.max_states and len(self._memo) >= self.max_states:
            raise RuntimeError(f"Exceeded max_states {self.max_states}")
        if self.t0 and (time.time() - self.t0) > self.max_hours * 3600:
            raise RuntimeError(f"Exceeded max_hours {self.max_hours}")

        idx = self._support_indices(mask, value)
        if len(idx) == 0:
            # unreachable
            self._memo[key] = float("-inf")
            return float("-inf")
        n_obs = int(np.sum(mask == 1))
        # Determine legal items
        legal = [j for j in range(self.n_items) if mask[j] == 0]
        stop_legal = (n_obs >= self.b_min) or (len(legal) == 0)
        # Forced stop if b==0 or no legal
        forced_stop = (b == 0) or (len(legal) == 0)

        _, u_stop = self._p_emp_and_utility(idx)
        # cost already incurred is stored via asked_set
        # For Bellman, child value includes future costs; current cost is sunk.

        best_val = float("-inf")
        best_action = STOP

        if stop_legal or forced_stop:
            best_val = u_stop
            best_action = STOP

        if not forced_stop:
            # consider each legal item
            for j in legal:
                # compute empirical response distribution for item j among support
                # Only records where j is not missing
                sub_mask = ~self.missing_mask[idx, j]
                # If all missing, then asking j leads to MISSING observation? But per spec MISSING never legal.
                # Actually legal items exclude MISSING states, so asking a MISSING item shouldn't happen.
                # But for support where item is missing, that branch is MISSING state.
                # We handle both branches.
                # Branch 1: observed values
                # For binary primary, values 0/1; for multi, collect unique values
                vals, counts = np.unique(self.X[idx[sub_mask], j], return_counts=True) if np.any(sub_mask) else (np.array([]), np.array([]))
                # Branch for MISSING: records where missing
                n_missing_branch = int(np.sum(self.missing_mask[idx, j]))
                # Expected value: sum_v P(v|s,j) * V(T(s,j,v), b-1)
                # Need to include cost penalty -lambda*c_j at this step
                exp_val = 0.0
                total = len(idx)
                has_branch = False
                for v, c in zip(vals, counts):
                    has_branch = True
                    p = c / total
                    # transition to OBSERVED(v)
                    mask2 = mask.copy(); value2 = value.copy()
                    mask2[j] = 1; value2[j] = int(v)
                    # asked_count+1
                    v_next = self._V(mask2, value2, b - 1, asked_count + 1, asked_set + (j,))
                    if v_next == float("-inf"):
                        continue
                    exp_val += p * v_next
                if n_missing_branch > 0:
                    has_branch = True
                    p_miss = n_missing_branch / total
                    mask2 = mask.copy(); value2 = value.copy()
                    mask2[j] = 2; value2[j] = -1
                    v_next = self._V(mask2, value2, b - 1, asked_count + 1, asked_set + (j,))
                    if v_next != float("-inf"):
                        exp_val += p_miss * v_next
                if not has_branch:
                    continue
                # subtract lambda cost for asking j
                exp_val = exp_val - self.lambda_cost * float(self.costs[j])
                self.n_evals += 1
                if exp_val > best_val:
                    best_val = exp_val
                    best_action = j

        self._memo[key] = best_val
        self._policy[key] = best_action
        self.n_states_visited = len(self._memo)
        return best_val

    def solve(self) -> Dict[str, Any]:
        self.t0 = time.time()
        self._memo.clear(); self._policy.clear()
        # Initial state: UNASKED where not structurally missing? Actually initial mask
        # For exact DP, we need to consider initial distribution over missing patterns.
        # The root state has mask UNASKED for non-missing items? We start with all UNASKED
        # but support filtering will handle missing patterns as we observe MISSING branches.
        # However some records have structurally missing items — they are not UNASKED initially?
        # Per §9, MISSING means structurally absent from source record. At init, we don't know
        # which items are missing until we try to ask them and discover MISSING branch.
        # Simpler: root is all UNASKED, and missingness discovered on ask.
        # Alternative from spec: INIT_STATE maps missing_mask True => MISSING immediately.
        # For exact DP we treat initial missing-aware state as distribution over initial masks.
        # To match empirical distribution, we average over initial missing patterns.
        # Approach: compute value as expectation over initial states weighted by their frequency.
        # But spec defines exact state as observation pattern + consistent training support set.
        # So we can treat root as empty observation and support = all records.
        mask0 = np.zeros(self.n_items, dtype=int)  # all UNASKED
        value0 = np.full(self.n_items, -1, dtype=int)
        try:
            v_star = self._V(mask0, value0, self.B, 0, ())
        except RuntimeError as e:
            return {"status": "limit_reached", "error": str(e), "n_states": len(self._memo), "time_sec": time.time()-self.t0, "n_evals": self.n_evals}
        elapsed = time.time() - self.t0
        try:
            import psutil, os
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:
            mem_mb = None
        return {
            "status": "optimal",
            "V_star": float(v_star),
            "n_states": len(self._memo),
            "n_evals": self.n_evals,
            "time_sec": elapsed,
            "mem_mb": mem_mb,
            "policy": dict(self._policy),  # may be large
        }

    def get_action(self, state: Dict[str, Any]) -> int:
        """Return optimal action for given runtime state dict."""
        mask = state["mask"]
        value = state["value"]
        b = state.get("questions_remaining", self.B)
        key = self._state_key(mask, value, b)
        if key in self._policy:
            return self._policy[key]
        # fallback: if not in policy (unseen support), default to STOP if legal else first legal
        legal = [j for j in range(self.n_items) if mask[j] == 0]
        n_obs = int(np.sum(mask == 1))
        stop_legal = (n_obs >= self.b_min) or (len(legal) == 0)
        if stop_legal:
            return STOP
        return legal[0] if legal else STOP

    def evaluate_policy(self, policy_fn, records: List[Dict[str, Any]] | None = None) -> float:
        """Evaluate a deterministic policy under same empirical evaluator.
        Computes mean terminal utility over training distribution using exact support evaluation.
        For optimality gap: uses empirical transition model, not predictor.
        Here we approximate by Monte Carlo over training records using policy decisions.
        """
        if records is None:
            records = self.records
        # For each record, simulate policy path and compute p_emp at terminal state
        total = 0.0
        for rec in records:
            # simulate
            from src.env.state import init_state, update_state
            state = init_state(rec)
            state["questions_remaining"] = self.B
            state["budget"] = self.B
            questions_remaining = self.B
            items_asked = []
            while True:
                mask = state["mask"]; value = state["value"]
                legal = [j for j in range(self.n_items) if mask[j]==0]
                n_obs = int(np.sum(mask==1))
                stop_legal = (n_obs >= self.b_min) or (len(legal)==0)
                if questions_remaining==0:
                    action = STOP
                else:
                    # policy_fn takes state dict and legal list (including STOP)
                    legal_with_stop = legal + ([STOP] if stop_legal else [])
                    action = policy_fn(state, legal_with_stop)
                if action == STOP:
                    idx = self._support_indices(state["mask"], state["value"])
                    if len(idx)==0:
                        total += 0
                    else:
                        _, u = self._p_emp_and_utility(idx)
                        # subtract lambda cost of asked items? For V_emp, utility includes terminal Brier only?
                        # Per §14.3 Gap = V* - V_emp where both use same empirical utility.
                        # V* already includes -lambda*c_j during recursion. For policy evaluation we must similarly subtract.
                        asked_cost = sum(float(self.costs[j]) for j in items_asked)
                        total += u - self.lambda_cost * asked_cost
                    break
                # ask item
                v = int(rec["item_responses"][action])
                # handle missing: if record missing at action, transition to MISSING
                if rec["missing_mask"][action]:
                    state["mask"][action]=2; state["value"][action]=-1
                else:
                    state["mask"][action]=1; state["value"][action]=v
                state["questions_remaining"]=questions_remaining-1
                state["budget"]=self.B
                questions_remaining-=1
                items_asked.append(action)
        return total / len(records) if records else 0.0
