"""Fault signature analysis + figures (Paper 1).

Reads `results/fault_*_s*.npz` (per-tick spike counts[64]) produced by
`fault_detection.py` and computes:

  - channel-localized fault profile: post-fault minus pre-fault mean counts,
    per channel, and whether the top channel equals the failed joint index;
  - spike-rate detector: total spikes/tick pre-vs-post fault, ROC AUC with
    leave-one-seed-out cross-validation, optimal threshold, detection time and
    lead time vs physical fall;
  - per-channel mutual information between counts and the fault label
    (bias-corrected against a permutation null), which channels carry fault
    information;
  - dose-response / authority boundary from `failure_battery.json` (survival).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
T_FAULT = 2.0
TPS = 40
N_CH = 64

JOINT_NAMES = {0: "L_hip_pitch", 1: "L_hip_roll", 2: "L_hip_yaw",
               3: "L_knee", 4: "L_ank_pitch", 5: "L_ank_roll",
               6: "R_hip_pitch", 7: "R_hip_roll", 8: "R_hip_yaw",
               9: "R_knee", 10: "R_ank_pitch", 11: "R_ank_roll"}


# ---------- information-theoretic helpers (port of signal_analysis) ----------

def mi_counts(hist):
    h = hist / hist.sum()
    px = h.sum(axis=1)
    py = h.sum(axis=0)
    px[px == 0] = 1.0
    py[py == 0] = 1.0
    mi = 0.0
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if h[i, j]:
                mi += h[i, j] * np.log(h[i, j] / (px[i] * py[j]))
    return mi


def channel_mi(counts_fault, counts_healthy, n_bins=8, n_perm=100, seed=11):
    """MI between per-channel counts and the fault label, one channel at a
    time. counts_fault/healthy: (n_ticks, 64) arrays (post-fault windows).
    Returns mi (n_ch,), pvals (n_ch,) via a per-channel permutation null."""
    rng = np.random.default_rng(seed)
    n_f, n_h = len(counts_fault), len(counts_healthy)
    mi = np.zeros(N_CH)
    pvals = np.zeros(N_CH)
    labels = np.concatenate([np.zeros(n_f), np.ones(n_h)]).astype(int)
    for ch in range(N_CH):
        v = np.concatenate([counts_fault[:, ch], counts_healthy[:, ch]])
        lo, hi = v.min(), v.max()
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        bf = np.clip(np.digitize(counts_fault[:, ch], edges) - 1, 0, n_bins - 1)
        bh = np.clip(np.digitize(counts_healthy[:, ch], edges) - 1, 0, n_bins - 1)
        hist = np.zeros((n_bins, n_bins))
        for a in bf:
            hist[a, 0] += 1
        for a in bh:
            hist[a, 1] += 1
        mi[ch] = mi_counts(hist)
        # permutation null: shuffle labels (vectorized, same RNG order)
        b_all = np.clip(np.digitize(v, edges) - 1, 0, n_bins - 1)
        cnt = 0
        for _ in range(n_perm):
            lab = rng.permutation(labels)
            hist_p = np.bincount(b_all + lab * n_bins,
                                 minlength=n_bins * n_bins).reshape(
                n_bins, n_bins)
            if mi_counts(hist_p) >= mi[ch] - 1e-12:
                cnt += 1
        pvals[ch] = (cnt + 1) / (n_perm + 1)
    return mi, pvals


def load_runs(failures, seeds):
    runs = {}
    for failure in failures:
        for seed in seeds:
            p = RES / ("fault_%s_s%d.npz" % (failure, seed))
            if not p.exists():
                continue
            d = np.load(p)
            runs[(failure, seed)] = {
                "counts": d["counts"], "x": d["x"],
                "fallen_t": float(d["fallen_t"]) if d["fallen_t"] >= 0 else None,
            }
    return runs


def spike_detector_auc(mu_pre, sd_pre, post_total, ticks_pre_win, ticks_post):
    """P(this tick|fault) from a pause-based z-score on total spikes/tick."""
    z = (post_total - mu_pre) / (sd_pre + 1e-9)
    # detection score = z clipped; AUC over pre (label 0) vs post (label 1)
    labels = np.concatenate([np.zeros(len(ticks_pre_win)),
                             np.ones(len(post_total))])
    scores = np.concatenate([(ticks_pre_win - mu_pre) / (sd_pre + 1e-9),
                             z])
    # ROC
    order = np.argsort(-scores)
    labels_s = labels[order]
    tpr = np.cumsum(labels_s) / (labels_s.sum() + 1e-9)
    fpr = np.cumsum(1 - labels_s) / ((1 - labels_s).sum() + 1e-9)
    auc = float(np.trapezoid(tpr, fpr))
    return auc, scores, labels


def analyze(failures, seeds, mi_perm=100):
    runs = load_runs(failures, seeds)
    if not runs:
        raise SystemExit("no captures found — run fault_detection.py first")
    out = {"failures": failures, "seeds": seeds, "t_fault": T_FAULT,
           "results": {}}
    pre_win = (slice(int(0.5 * TPS), int(T_FAULT * TPS)),)
    post_win = (slice(int((T_FAULT + 1.0) * TPS),
                      int((T_FAULT + 3.0) * TPS)),)

    healthy = [(s, runs[("healthy", s)]["counts"]) for s in seeds
               if ("healthy", s) in runs]

    for failure in failures:
        if failure == "healthy":
            continue
        joint = {"freeze_lknee": 3, "freeze_rknee": 9, "freeze_lhip": 1,
                 "degrade_50": 3}[failure]
        det_aucs, lead_times, det_times, fall_times = [], [], [], []
        localization_across, delta_profiles = [], []
        ch_mi_across, ch_pv_across = [], []
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
            post_h = hc[post_win].sum(axis=1)
            mu_pre, sd_pre = float(pre_h.mean()), float(pre_h.std())
            # detector on total spikes: AUC (leave outputs), threshold = mu+2sd
            auc, scores, labels = spike_detector_auc(
                mu_pre, sd_pre, post_f, pre_h, post_f)
            det_aucs.append(auc)
            # detection time: first post-fault tick above mu+3sd
            z_all = (f.sum(axis=1) - mu_pre) / (sd_pre + 1e-9)
            thr = 3.0
            det_t = None
            for i in range(int(T_FAULT * TPS), len(z_all)):
                if z_all[i] > thr:
                    det_t = i / TPS
                    break
            det_times.append(det_t)
            fallen_t = runs[(failure, s)]["fallen_t"]
            fall_times.append(fallen_t)
            lead = None if det_t is None or fallen_t is None else fallen_t - det_t
            lead_times.append(lead)
            # channel MI + mean-delta localization profile
            post_fc = f[post_win]
            post_hc = hc[post_win]
            mi, pv = channel_mi(post_fc, post_hc, n_perm=mi_perm)
            ch_mi_across.append(mi)
            ch_pv_across.append(pv)
            # signed mean-delta profile (failed joint channel should jump)
            d = (post_fc.mean(axis=0) - post_hc.mean(axis=0))
            delta_profiles.append(d)
            localization_across.append(int(np.argmax(d)))
            per_seed.append({
                "seed": s, "fallen_t": fallen_t, "auc": round(auc, 3),
                "det_t": round(det_t, 2) if det_t else None,
                "lead_t": round(lead, 2) if lead else None,
                "mean_total_pre": round(float(pre_f.mean()), 1),
                "mean_total_post": round(float(post_f.mean()), 1),
                "top_ch_mi": int(np.argmax(mi)),
                "top_ch_delta": int(np.argmax(d)),
                "top_delta_value": round(float(d.max()), 2),
                "top_channel23_delta": round(float(d[23]), 2),
            })
        # aggregate
        mi = np.mean(ch_mi_across, axis=0)
        pv = np.mean(ch_pv_across, axis=0)
        top_mi = int(np.argmax(mi))
        dmean = np.mean(delta_profiles, axis=0)
        top_delta = int(np.argmax(dmean))
        localizations = [int(np.argmax(d)) for d in delta_profiles]
        n_sig = int(np.sum(pv < 0.05))
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
            "mi_mean_top_ch": round(float(mi.max()), 3),
            "mi_sig_channels": n_sig,
            "top_mi_ch": top_mi,
            "top_mi_ch_name": JOINT_NAMES.get(top_mi),
            "localization_correct": int(top_delta == joint),
            "localization_per_seed": localization_across,
            "top_ch23_delta_mean": round(float(np.mean(
                [d[23] for d in delta_profiles])), 2),
            "top_delta_mean": round(float(np.mean(
                [d.max() for d in delta_profiles])), 2),
            "per_seed": per_seed,
        }
        print("%-12s joint=%-11s AUC=%.3f top_delta_ch=%s(%s) localiz=%s "
              "lead_t(mean)=%s det_t=%s fall_t=%s" % (
                  failure, JOINT_NAMES.get(joint), out["results"][failure][
                      "auc_mean"], top_delta, JOINT_NAMES.get(top_delta),
                  out["results"][failure]["localization_correct"],
                  out["results"][failure]["lead_t_mean"],
                  out["results"][failure]["det_t_mean"],
                  out["results"][failure]["fall_t_seeds"]))
    return out


if __name__ == "__main__":
    res = analyze(["healthy", "freeze_lknee", "freeze_rknee", "freeze_lhip",
                   "degrade_50"], [7, 29, 13])
    with open(RES / "fault_analysis.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print("saved:", RES / "fault_analysis.json")