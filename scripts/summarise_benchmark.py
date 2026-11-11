import json
import numpy as np

d = json.load(open("results/benchmark_multiseed_saudi.json"))
ps = d["per_seed"]["B6"]
arms = ["greedy", "beta_greedy", "irt_cat", "static_rfe", "exact_fixed_subset", "hybrid_irt_evoi", "random"]
print(f"fingerprint={d['split_fingerprint']} wall={d['wall_seconds']:.1f}s completed={d['n_completed_runs']} failed={d['n_failed_runs']}")
for arm in arms:
    b = [ps[f"seed{s}"][arm]["metrics"]["brier"] for s in range(10)]
    u = [ps[f"seed{s}"][arm]["metrics"]["uar"] for s in range(10)]
    a = [ps[f"seed{s}"][arm]["metrics"]["auroc"] for s in range(10)]
    asked = [ps[f"seed{s}"][arm]["burden"]["mean"] for s in range(10)]
    stop = [ps[f"seed{s}"][arm]["stop_early_frac"] for s in range(10)]
    print(f"{arm:20s} brier={np.mean(b):.4f}+/-{np.std(b,ddof=1):.4f} "
          f"uar={np.mean(u):.4f}+/-{np.std(u,ddof=1):.4f} auroc={np.mean(a):.4f} "
          f"asked={np.mean(asked):.2f} stop={np.mean(stop):.3f}")
print()
pb = json.load(open("results/paired_bootstrap_hybrid.json"))
for k, v in pb.items():
    print(k, f"diff={v['observed_diff_brier']:+.4f} 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
