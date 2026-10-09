"""Deterministic text rendering of an outcome explanation — P3-outcome.

Presentation only. Every sentence is assembled from the fields of the
explanation object produced by :func:`src.explain.outcome.explain_outcome`;
nothing is computed here and no claim can appear that is not in the evidence
object. This is the layer a future LLM verbaliser would have to respect, and
the reason it can be unit-tested without a model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _points(phi: float) -> str:
    return f"{100.0 * float(phi):+.1f} points"


def _contribution_phrase(entry: Dict[str, Any],
                         item_texts: Optional[Dict[str, str]] = None) -> str:
    item = entry["item"]
    resp = entry.get("response")
    resp_word = "atypical" if resp == 1 else ("typical" if resp == 0
                                              else f"response {resp}")
    text = f" ({item_texts[item]})" if item_texts and item in item_texts else ""
    return f"{item} answered {resp_word}{text}: {_points(entry['phi'])}"


def render_text(explanation: Dict[str, Any],
                item_texts: Optional[Dict[str, str]] = None) -> str:
    """Human-readable rendering of an outcome explanation.

    Parameters
    ----------
    explanation:
        The object from ``explain_outcome``.
    item_texts:
        Optional {item_code: short question wording}. When absent, items are
        referred to by code only.
    """
    out: List[str] = []
    oc = explanation["outcome"]
    decision = oc["decision"]
    p_hat = oc["p_hat"]
    tau = oc["tau"]
    verdict = ("referral recommended" if decision == "REFERRAL_RECOMMENDED"
               else "no referral indicated")
    out.append(
        f"Screening result: risk estimate {_pct(p_hat)} "
        f"(operating threshold {_pct(tau)}) -> {verdict}.")

    contribs = explanation.get("contributions", [])
    if contribs:
        ranked = ", ".join(_contribution_phrase(c, item_texts) for c in contribs)
        out.append(f"What moved this estimate the most: {ranked}.")
    else:
        out.append("No observed responses were available to attribute.")

    ev = explanation.get("evidence", {})
    unasked = ev.get("unobserved_items_marginalised", [])
    if unasked:
        out.append(
            f"Not asked in this session: {', '.join(unasked)}. The estimate "
            f"accounts for them through the model's prior over unobserved "
            f"items, not through any answer.")

    unc = explanation.get("uncertainty", {})
    if unc.get("available"):
        lo, hi = unc["interval"]
        out.append(
            f"How stable the estimate is: over resamples of the model's prior "
            f"at the training sample size it moves between {_pct(lo)} and "
            f"{_pct(hi)}. This is model stability, not clinical certainty.")
    else:
        out.append(
            "How stable the estimate is: no per-session interval is available "
            "for this predictor version; the point estimate is reported "
            "without one.")

    cf = explanation.get("counterfactuals", {})
    single = [f for f in cf.get("single_flips", []) if f.get("flips_decision")]
    if single:
        parts = []
        for f in single[:3]:
            to_word = "atypical" if f["flipped_value"] == 1 else "typical"
            parts.append(
                f"if {f['item']} had been answered {to_word} the estimate would "
                f"have been {_pct(f['new_p'])}")
        out.append(
            "What would change the model's output: " + "; ".join(parts) +
            ". These describe the model's behaviour under different answers, "
            "not the person, and are not advice.")
    elif cf.get("robust_to_all_single_flips"):
        out.append(
            "What would change the model's output: no single change to any "
            "answer given would move it across the operating threshold.")
    ask_more = cf.get("ask_more", [])
    if ask_more and not single:
        entry = ask_more[0]
        opts = entry.get("options", {})
        if opts:
            atyp = opts.get("1", opts.get("0"))
            out.append(
                f"An unasked question ({entry['item']}) would move the estimate "
                f"to about {_pct(atyp)} if answered atypically.")

    if explanation.get("limitations"):
        out.append("Limitations: " + " ".join(explanation["limitations"]))

    out.append(explanation.get("disclaimer", "").strip())

    for w in explanation.get("warnings", []) or []:
        out.append(f"Warning: {w}")
    return "\n\n".join(out)
