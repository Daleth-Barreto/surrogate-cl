"""Localization consistency analysis (Paper 1).

Reads canonical fault_*.npz (neural/rate captures) and quantifies whether the
fault signature localizes consistently across seeds, reporting:
  - per-fault argmax channel across seeds (drift vs joint index)
  - top-k channel hit rate (k=3, k=5)
  - majority channel across seeds
  - per-channel23 delta (collapse-telegraph marker)
  - leave-one-seed-out nearest-centroid classifier accuracy (chance = 0.25)
  - per-fault confusion (which channels appear in top-3 most often)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fault_analysis import load_runs, JOINT_NAMES, T_FAULT, TPS, N_CH

RES = Path(__file__).resolve().parent.parent / "results"

SEEDS = [7, 29, 13]
FAILURES = ["freeze_lknee", "freeze_rknee", "freeze_lhip", "degrade_50"]
JOINT_MAP = {"freeze_lknee": 3, "freeze_rknee": 9, "freeze_lhip": 1,
             "degrade_50": 3}

PRE_WIN = (slice(int(0.5 * TPS), int(T_FAULT * TPS)),)
POST_WIN = (slice(int((T_FAULT + 1.0) * TPS), int((T_FAULT + 3.0) * TPS)),)


def delta_profile(fault_counts, healthy_counts):
    return (fault_counts[POST_WIN].mean(axis=0)
            - healthy_counts[PRE_WIN].mean(axis=0))


def main():
    runs = load_runs(["healthy"] + FAILURES, SEEDS)
    if not runs:
        raise SystemExit("no canonical captures found — run fault_detection.py first")

    healthy = {s: runs[("healthy", s)]["counts"] for s in SEEDS
               if ("healthy", s) in runs}

    out = {"seeds": SEEDS, "failures": FAILURES, "tps": TPS, "results": {}}

    for failure in FAILURES:
        joint = JOINT_MAP[failure]
        profiles, seeds_avail = [], []
        ch23_deltas, top_channels = [], []
        top3_set = {}
        for s in SEEDS:
            if (failure, s) not in runs or s not in healthy:
                continue
            d = delta_profile(runs[(failure, s)]["counts"], healthy[s])
            profiles.append(d)
            seeds_avail.append(s)
            top_ch = int(np.argmax(d))
            top_channels.append(top_ch)
            ch23_deltas.append(round(float(d[23]), 2))
            top3 = np.argsort(np.abs(d))[::-1][:5].tolist()
            for c in top3:
                top3_set[c] = top3_set.get(c, 0) + 1

        if not profiles:
            continue
        dmat = np.stack(profiles, axis=0)  # (n_seeds, 64)
        dmean = dmat.mean(axis=0)
        top_mean = int(np.argmax(dmean))
        majority = max(set(top_channels), key=top_channels.count)
        top1_hit = int(majority == joint)
        top3_sorted = sorted(top3_set.items(), key=lambda kv: -kv[1])[:5]
        top3_hit = int(joint in [c for c, _ in top3_sorted[:3]])
        top5_hit = int(joint in [c for c, _ in top3_sorted[:5]])

        # Per-seed top channels
        per_seed = [{"seed": s, "top_ch": top_channels[i],
                     "joint": joint, "ch23_delta": ch23_deltas[i]}
                    for i, s in enumerate(seeds_avail)]

        out["results"][failure] = {
            "joint": joint, "joint_name": JOINT_NAMES.get(joint),
            "seeds": seeds_avail,
            "top_channels_per_seed": top_channels,
            "majority_channel": majority,
            "majority_channel_name": JOINT_NAMES.get(majority),
            "top_mean_delta_ch": top_mean,
            "top_mean_delta_ch_name": JOINT_NAMES.get(top_mean),
            "top1_hit": top1_hit, "top3_hit": top3_hit, "top5_hit": top5_hit,
            "ch23_delta_per_seed": ch23_deltas,
            "ch23_delta_mean": round(float(np.mean(ch23_deltas)), 2),
            "top5_channels": top3_sorted,
            "per_seed": per_seed,
        }

    # Leave-one-seed-out nearest-centroid classifier
    all_profiles = []
    all_labels = []
    for failure in FAILURES:
        for i, s in enumerate(SEEDS):
            if (failure, s) not in runs or s not in healthy:
                continue
            d = delta_profile(runs[(failure, s)]["counts"], healthy[s])
            all_profiles.append(d)
            all_labels.append(failure)
    all_profiles = np.stack(all_profiles, axis=0)
    all_labels = np.array(all_labels)
    n = len(all_labels)
    loo_correct = 0
    per_class_correct = {f: 0 for f in FAILURES}
    per_class_total = {f: 0 for f in FAILURES}
    confusion = np.zeros((len(FAILURES), len(FAILURES)), dtype=int)
    label_idx = {f: i for i, f in enumerate(FAILURES)}

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        Xtr, Ytr = all_profiles[train_mask], all_labels[train_mask]
        Xt, yt = all_profiles[i], all_labels[i]
        # class centroids
        preds = []
        for f in FAILURES:
            cls_mask = Ytr == f
            if cls_mask.sum() == 0:
                preds.append(float("inf"))
            else:
                centroid = Xtr[cls_mask].mean(axis=0)
                preds.append(np.linalg.norm(Xt - centroid))
        pred = FAILURES[int(np.argmin(preds))]
        per_class_total[yt] += 1
        if pred == yt:
            loo_correct += 1
            per_class_correct[yt] += 1
        confusion[label_idx[yt], label_idx[pred]] += 1

    out["classifier"] = {
        "method": "nearest-centroid, leave-one-seed-out",
        "n_samples": n,
        "accuracy": round(loo_correct / max(n, 1), 3),
        "chance": round(1.0 / len(FAILURES), 3),
        "per_class_accuracy": {f: round(per_class_correct[f] / max(per_class_total[f], 1), 3)
                               for f in FAILURES},
        "confusion": {f: confusion[label_idx[f]].tolist() for f in FAILURES},
        "labels": FAILURES,
    }

    print("LOO classifier accuracy: %.3f (chance %.3f)" % (
        loo_correct / max(n, 1), 1.0 / len(FAILURES)))
    for failure in FAILURES:
        r = out["results"][failure]
        print("%-12s joint=%-11s majority_ch=%s(%s) top1=%d top3=%d top5=%d "
              "ch23_delta=%.2f" % (
                  failure, JOINT_NAMES.get(r["joint"]),
                  r["majority_channel"], r["majority_channel_name"],
                  r["top1_hit"], r["top3_hit"], r["top5_hit"],
                  r["ch23_delta_mean"]))

    with open(RES / "localization_analysis.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("saved:", RES / "localization_analysis.json")


if __name__ == "__main__":
    main()
