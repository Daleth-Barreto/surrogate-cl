"""Dead-substrate analysis: mirrors fault_analysis.py but reads fault_dead_*.npz
(Poisson dead-null) and reports AUC / det_t / lead / localization against the
live (neural) canonical results from fault_analysis.json.

Produces fault_analysis_dead.json with:
  - live numbers copied from canonical
  - dead numbers (same schema)
  - deltas (live AUC - dead AUC) for each failure
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fault_analysis import (channel_mi, spike_detector_auc, JOINT_NAMES,
                            T_FAULT, TPS, N_CH)

RES = Path(__file__).resolve().parent.parent / "results"

SEEDS = [7, 29, 13]
FAILURES = ["freeze_lknee", "freeze_rknee", "freeze_lhip", "degrade_50"]
JOINT_MAP = {"freeze_lknee": 3, "freeze_rknee": 9, "freeze_lhip": 1,
             "degrade_50": 3}


def load_runs(prefix, failures, seeds):
    runs = {}
    for failure in failures:
        for seed in seeds:
            p = RES / ("%s_%s_s%d.npz" % (prefix, failure, seed))
            if not p.exists():
                continue
            d = np.load(p)
            runs[(failure, seed)] = {
                "counts": d["counts"], "x": d["x"],
                "fallen_t": float(d["fallen_t"]) if d["fallen_t"] >= 0 else None,
            }
    return runs


def analyze_group(runs, failures, seeds, mi_perm=100):
    out = {"results": {}}
    pre_win = (slice(int(0.5 * TPS), int(T_FAULT * TPS)),)
    post_win = (slice(int((T_FAULT + 1.0) * TPS),
                      int((T_FAULT + 3.0) * TPS)),)

    healthy = [(s, runs[("healthy", s)]["counts"]) for s in seeds
               if ("healthy", s) in runs]

    for failure in failures:
        if failure == "healthy":
            continue
        joint = JOINT_MAP[failure]
        det_aucs, lead_times, det_times, fall_times = [], [], [], []
        ch_mi_across, ch_pv_across = [], []
        delta_profiles, localization_across = [], []
        per_seed = []
        for s in seeds:
            if (failure, s) not in runs:
                continue
            f = runs[(failure, s)]["counts"]
            pre_f = f[pre_win].sum(axis=1)
            post_f = f[post_win].sum(axis=1)
            h_pair = [c for (ss, c) in healthy if ss == s]
            if not h_pair:
                continue
            hc = h_pair[0]
            pre_h = hc[pre_win].sum(axis=1)
            mu_pre, sd_pre = float(pre_h.mean()), float(pre_h.std())
            auc, _, _ = spike_detector_auc(mu_pre, sd_pre, post_f, pre_h, post_f)
            det_aucs.append(auc)
            z_all = (f.sum(axis=1) - mu_pre) / (sd_pre + 1e-9)
            det_t = None
            for i in range(int(T_FAULT * TPS), len(z_all)):
                if z_all[i] > 3.0:
                    det_t = i / TPS
                    break
            det_times.append(det_t)
            fallen_t = runs[(failure, s)]["fallen_t"]
            fall_times.append(fallen_t)
            lead = None if det_t is None or fallen_t is None else fallen_t - det_t
            lead_times.append(lead)
            post_fc = f[post_win]
            post_hc = hc[post_win]
            mi, pv = channel_mi(post_fc, post_hc, n_perm=mi_perm)
            ch_mi_across.append(mi)
            ch_pv_across.append(pv)
            d = (post_fc.mean(axis=0) - post_hc.mean(axis=0))
            delta_profiles.append(d)
            localization_across.append(int(np.argmax(d)))
            per_seed.append({
                "seed": s, "fallen_t": fallen_t, "auc": round(auc, 3),
                "det_t": round(det_t, 2) if det_t else None,
                "lead_t": round(lead, 2) if lead else None,
                "mean_total_pre": round(float(pre_f.mean()), 1),
                "mean_total_post": round(float(post_f.mean()), 1),
                "top_ch_delta": int(np.argmax(d)),
                "top_delta_value": round(float(d.max()), 2),
                "top_channel23_delta": round(float(d[23]), 2),
                "top_ch_mi": int(np.argmax(mi)),
            })
        n_sig = int(np.mean(ch_pv_across, axis=0).min(axis=0) < 0.05) if ch_pv_across else 0
        mi_mean = np.mean(ch_mi_across, axis=0) if ch_mi_across else np.zeros(N_CH)
        pv_mean = np.mean(ch_pv_across, axis=0) if ch_pv_across else np.ones(N_CH)
        dmean = np.mean(delta_profiles, axis=0) if delta_profiles else np.zeros(N_CH)
        top_delta = int(np.argmax(dmean))
        n_sig_total = int(np.sum(pv_mean < 0.05))
        out["results"][failure] = {
            "joint": joint, "joint_name": JOINT_NAMES.get(joint),
            "auc_mean": round(float(np.mean(det_aucs)), 3),
            "auc_seeds": [round(float(v), 3) for v in det_aucs],
            "det_t_mean": round(float(np.mean([t for t in det_times
                                               if t is not None])), 2),
            "lead_t_mean": round(float(np.mean([t for t in lead_times
                                                if t is not None])), 2),
            "lead_t_seeds": [round(t, 2) if t is not None else None
                             for t in lead_times],
            "fall_t_seeds": [round(t, 2) if t is not None else None
                             for t in fall_times],
            "mean_total_pre": round(float(np.mean([p["mean_total_pre"]
                                                   for p in per_seed])), 1),
            "mean_total_post": round(float(np.mean([p["mean_total_post"]
                                                    for p in per_seed])), 1),
            "top_delta_ch": top_delta,
            "top_delta_ch_name": JOINT_NAMES.get(top_delta),
            "mi_mean_top_ch": round(float(mi_mean.max()), 3),
            "mi_sig_channels": n_sig_total,
            "top_ch23_delta_mean": round(float(np.mean(
                [d[23] for d in delta_profiles])), 2),
            "per_seed": per_seed,
        }
    return out


def main():
    live_runs = load_runs("fault", ["healthy"] + FAILURES, SEEDS)
    dead_runs = load_runs("fault_dead", ["healthy"] + FAILURES, SEEDS)

    if not dead_runs:
        raise SystemExit("no dead-null captures found — run fault_dead_null.py first")

    live_out = analyze_group(live_runs, ["healthy"] + FAILURES, SEEDS)
    dead_out = analyze_group(dead_runs, ["healthy"] + FAILURES, SEEDS)

    combined = {"seeds": SEEDS, "failures": FAILURES,
                "live": live_out["results"], "dead": dead_out["results"],
                "deltas": {}}
    for failure in FAILURES:
        if failure not in live_out["results"] or failure not in dead_out["results"]:
            continue
        l = live_out["results"][failure]
        d = dead_out["results"][failure]
        combined["deltas"][failure] = {
            "joint": l["joint"],
            "delta_auc": round(l["auc_mean"] - d["auc_mean"], 3),
            "delta_det_t": (round(l["det_t_mean"] - d["det_t_mean"], 2)
                            if d["det_t_mean"] is not None else None),
            "delta_lead_t": (round(l["lead_t_mean"] - d["lead_t_mean"], 2)
                             if d["lead_t_mean"] is not None else None),
            "live_auc": l["auc_mean"], "dead_auc": d["auc_mean"],
            "live_top_delta_ch": l["top_delta_ch"],
            "dead_top_delta_ch": d["top_delta_ch"],
            "live_mi_sig": l["mi_sig_channels"],
            "dead_mi_sig": d["mi_sig_channels"],
        }
        print("%-12s LIVE AUC=%.3f d_auc=%.3f (dead %.3f) d_lead=%s" % (
            failure, l["auc_mean"],
            l["auc_mean"] - d["auc_mean"], d["auc_mean"],
            (round(l["lead_t_mean"] - d["lead_t_mean"], 2)
             if d.get("lead_t_mean") is not None else "N/A")))

    with open(RES / "fault_analysis_dead.json", "w") as fp:
        json.dump(combined, fp, indent=2)
    print("saved:", RES / "fault_analysis_dead.json")


if __name__ == "__main__":
    main()
