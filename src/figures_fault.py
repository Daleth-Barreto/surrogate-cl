"""Paper 1 figures: fault signatures in the culture spike stream.

fig_fault_detection.png:
  (a) spike-count heatmap (64 ch x time) healthy vs freeze_lknee (seed 7)
  (b) total-spikes/tick trajectory with fault onset (t=2.0), detector
      threshold (mu+3sd) and physical fall marker
  (c) channel mean-delta topography (post-fault minus healthy); failed-joint
      channel + height channel highlighted
  (d) ROC AUC per failure + lead time inset
fig_authority.png:  survival under fault battery (dose-response / control
  authority boundary) from failure_battery.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
FIG = RES.parent / "figures"
FIG.mkdir(exist_ok=True)
T_FAULT = 2.0
TPS = 40
JOINT_NAMES = {0: "L_hip_pitch", 1: "L_hip_roll", 2: "L_hip_yaw",
               3: "L_knee", 4: "L_ank_pitch", 5: "L_ank_roll",
               6: "R_hip_pitch", 7: "R_hip_roll", 8: "R_hip_yaw",
               9: "R_knee", 10: "R_ank_pitch", 11: "R_ank_roll"}
CH24 = "height"
FAILS = [("freeze_lknee", 3, "freeze left knee"),
         ("freeze_rknee", 9, "freeze right knee"),
         ("freeze_lhip", 1, "freeze left hip_roll"),
         ("degrade_50", 3, "degrade L-leg 50%")]


def load(failure, seed):
    d = np.load(RES / ("fault_%s_s%d.npz" % (failure, seed)))
    return d["counts"], float(d["fallen_t"])


def fig_detection():
    f, h = load("freeze_lknee", 7)
    hc, _ = load("healthy", 7)
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5))

    # (a) heatmaps
    for ax, (cnt, title) in zip([axs[0, 0], axs[1, 0]],
                                [(hc, "healthy"), (f, "freeze L knee")]):
        t = np.arange(cnt.shape[0]) / TPS
        im = ax.imshow(cnt.T, aspect="auto", origin="lower",
                       cmap="viridis", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("channel")
        ax.set_xticks([0, TPS, 2 * TPS, 3 * TPS, 4 * TPS, 5 * TPS])
        ax.set_xticklabels(["0", "1", "2", "3", "4", "5"])
        ax.axvline(T_FAULT * TPS, color="r", lw=1.2, ls="--")
        ax.set_ylim(-0.5, 64)
    axs[0, 0].set_xlabel("time (s)")
    axs[1, 0].set_xlabel("time (s)")

    # (b) total spikes trajectory + detector
    ax = axs[0, 1]
    totals = f.sum(axis=1)
    htot = hc.sum(axis=1)
    mu_pre, sd_pre = htot[:int(T_FAULT * TPS)].mean(), htot[:int(T_FAULT * TPS)].std()
    z = (totals - mu_pre) / sd_pre
    t = np.arange(len(totals)) / TPS
    ax.plot(t, totals, lw=1, label="total spikes/tick")
    ax.axhline(mu_pre + 3 * sd_pre, color="k", ls=":", lw=1.2,
               label="$\\mu$+3$\\sigma$ (pre-fault)")
    ax.axvline(T_FAULT, color="r", ls="--", lw=1.3, label="fault onset (t=2 s)")
    det_i = int(np.argmax(z > 3)) if np.any(z > 3) else None
    if det_i is not None:
        ax.plot(t[det_i], totals[det_i], "o", mfc="white", mec="green",
                ms=7, label="detection")
    fb = float(np.load(RES / "fault_freeze_lknee_s7.npz")["fallen_t"])
    ax.axvline(fb, color="orange", ls="-.", lw=1.3, label="physical fall")
    ax.set_xlabel("time (s)"); ax.set_ylabel("spikes / tick")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("total spike-rate: detector")

    # (c) channel delta topography
    ax = axs[1, 1]
    x = np.arange(64)
    d = (f[120:200].mean(axis=0) - hc[120:200].mean(axis=0))
    bars = ax.bar(x, d, color="steelblue")
    bars[23].set_color("orange")          # height channel
    bars[3].set_color("red")              # failed joint (L_knee = 3)
    ax.set_title("channel delta topography (post-fault $-$ healthy, t\\in[3,5]s)")
    ax.set_xlabel("channel (0-5 L leg, 6-11 R leg, 23 height)")
    ax.set_ylabel("$\\Delta$ mean spikes/tick")
    first = {"0": "L\\_hip\\_pitch", "1": "L\\_hip\\_roll", "2": "L\\_hip\\_yaw",
             "3": "L\\_knee", "4": "L\\_ank\\_pitch", "5": "L\\_ank\\_roll",
             "6": "R\\_hip\\_pitch", "7": "R\\_hip\\_roll", "8": "R\\_hip\\_yaw",
             "9": "R\\_knee"}
    ax.set_xticks([0, 3, 6, 9, 23])
    ax.set_xticklabels([first["0"], first["3"], first["6"], first["9"], "height"],
                       rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.6)

    fig.suptitle("Fault signatures in the culture spike stream (neural::rate, seed 7)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig_fault_detection.png", dpi=200)
    print("saved", FIG / "fig_fault_detection.png")


def fig_auc():
    import json
    j = json.load(open(RES / "fault_analysis.json"))
    res = j["results"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = [JOINT_NAMES[res[k]["joint"]] for k in FAILS and [n for n, *_ in FAILS]]
    aucs = [res[k]["auc_mean"] for k, *_ in FAILS]
    leads = [res[k]["lead_t_mean"] for k, *_ in FAILS]
    bars = ax.bar(names, aucs, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection AUC (spike-rate detector)")
    for b, a, l in zip(bars, aucs, leads):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02,
                "AUC %.2f\nlead %.2fs" % (a, l), ha="center", fontsize=8)
    ax.set_title("Fault detection through the culture: AUC and lead time "
                 "before physical fall (mean, n=3 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_fault_auc.png", dpi=200)
    print("saved", FIG / "fig_fault_auc.png")


def fig_authority():
    import json
    j = json.load(open(RES / "failure_battery.json"))
    runs = j["run"]
    cond = "neural_rate"

    fig, ax = plt.subplots(figsize=(8.5, 4))
    labels, surv = [], []
    for o in ["healthy", "freeze_lknee", "freeze_rknee", "freeze_lhip",
              "degrade_50", "jitter_leg"]:
        if o not in runs:
            continue
        labels.append(o.replace("_", " "))
        surv.append(runs[o][cond]["survivors"])
    x = np.arange(len(labels))
    bars = ax.bar(x, surv, width=0.6, color="steelblue")
    bars[0].set_color("seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("survived runs / 4")
    ax.set_ylim(0, 4.5)
    for b, s in zip(bars, surv):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.1, str(s),
                ha="center", fontsize=9)
    ax.set_title("Control authority boundary: the culture loop cannot prevent "
                 "falls from severe joint faults (4 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_fault_authority.png", dpi=200)
    print("saved", FIG / "fig_fault_authority.png")


if __name__ == "__main__":
    fig_detection()
    fig_auc()
    fig_authority()