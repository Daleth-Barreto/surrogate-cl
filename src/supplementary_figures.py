"""Supplementary figures for Paper 1 dead-null / localization / recovery.

Reads the analysis JSONs produced alongside this script and draws three
figures into the figures/ directory:

  fig_dead_null.png     - live vs dead AUC, detection/lead time, total-spike
                          pre/post and ch23 delta, per failure mode
  fig_localization.png  - per-seed top-delta channel vs failed joint, and the
                          leave-one-seed-out nearest-centroid confusion matrix
  fig_recovery.png      - per-fault survival (halt OFF vs ON) and paired
                          detection-time / lead-time scatter

Run after fault_analysis_dead.py, localization.py, recovery_battery.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

FAILURES = ["freeze_lknee", "freeze_rknee", "freeze_lhip", "degrade_50"]
LABELS = {"freeze_lknee": "L knee", "freeze_rknee": "R knee",
          "freeze_lhip": "L hip", "degrade_50": "L leg (deg.)"}
JOINT_CH = {"freeze_lknee": 3, "freeze_rknee": 9, "freeze_lhip": 1,
            "degrade_50": 3}
COLORS = {"off": "#4477AA", "on": "#CC6677"}


def _load(name):
    with open(RES / name, encoding="utf-8") as fp:
        return json.load(fp)


def fig_dead_null(ad):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    live = ad["live"]
    dead = ad["dead"]
    x = np.arange(len(FAILURES))

    ax = axes[0][0]
    ax.bar(x - 0.2, [live[f]["auc_mean"] for f in FAILURES], 0.4,
           label="live substrate", color="#4477AA")
    ax.bar(x + 0.2, [dead[f]["auc_mean"] for f in FAILURES], 0.4,
           label="dead null", color="#AA7777")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("A. spike-detector AUC (ROC)")
    ax.legend(frameon=False)

    ax = axes[0][1]
    dets_live = [live[f]["det_t_mean"] for f in FAILURES]
    dets_dead = [dead[f]["det_t_mean"] for f in FAILURES]
    ax.bar(x - 0.2, dets_live, 0.4, label="live", color="#4477AA")
    ax.bar(x + 0.2, dets_dead, 0.4, label="dead", color="#AA7777")
    ax.axhline(2.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylabel("detection time (s)")
    ax.set_title("B. 3-sigma detection time")
    ax.legend(frameon=False)

    ax = axes[1][0]
    ch23_l = [live[f]["top_ch23_delta_mean"] for f in FAILURES]
    ch23_d = [dead[f]["top_ch23_delta_mean"] for f in FAILURES]
    ax.bar(x - 0.2, ch23_l, 0.4, label="live", color="#4477AA")
    ax.bar(x + 0.2, ch23_d, 0.4, label="dead", color="#AA7777")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylabel("mean delta ch23")
    ax.set_title("C. collapse-telegraph channel delta")
    ax.legend(frameon=False)

    ax = axes[1][1]
    tops = [live[f]["top_delta_ch"] if "top_delta_ch" in live[f]
            else live[f].get("top_delta_ch", 23) for f in FAILURES]
    ax.plot(x, [JOINT_CH[f] for f in FAILURES], "o", ms=9, color="k",
            label="failed joint index")
    ax.plot(x, tops, "s", ms=8, color="#CC6677", label="live top-delta ch")
    ax.plot(x, [dead[f]["top_delta_ch"] for f in FAILURES], "^", ms=8,
            color="#AA7777", label="dead top-delta ch")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylabel("channel index (0..63)")
    ax.set_title("D. localization channel")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(FIG / "fig_dead_null.png", dpi=150)
    plt.close(fig)
    print("saved:", FIG / "fig_dead_null.png")


def fig_localization(lz):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    res = lz["results"]

    ax = axes[0]
    x = np.arange(len(FAILURES))
    tops = [res[f]["top_channels_per_seed"] for f in FAILURES]
    joints = [res[f]["joint"] for f in FAILURES]
    for i, (xf, seeds) in enumerate(zip(x, tops)):
        for v in seeds:
            ax.scatter(xf + np.random.uniform(-0.15, 0.15), v, s=40,
                       color="#4477AA", alpha=0.8, zorder=3)
    ax.scatter([], [], s=40, color="#4477AA", alpha=0.8,
               label="per-seed top-delta channel")
    for xf, j in zip(x, joints):
        ax.plot([xf - 0.3, xf + 0.3], [j, j], color="k", ls="--", lw=1.2)
    ax.plot([], [], color="k", ls="--", lw=1.2, label="failed joint index")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylabel("channel index")
    ax.set_title("A. localization per seed")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    conf = np.array([res[f]["top5_channels"] for f in FAILURES]
                    ) if False else None
    labels = list(lz["classifier"]["labels"])
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    confull = lz["classifier"]["confusion"]
    for i, f in enumerate(labels):
        for j, g in enumerate(labels):
            mat[i][j] = confull[f][j]
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([LABELS[l] for l in labels], rotation=15)
    ax.set_yticklabels([LABELS[l] for l in labels])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("B. LOO nearest-centroid")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, mat[i][j], ha="center", va="center",
                    color="white" if mat[i][j] > mat.max() / 2 else "k")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    acc = lz["classifier"]["accuracy"]
    chance = lz["classifier"]["chance"]
    ax.bar([0, 1], [acc, chance], width=0.6, color=["#4477AA", "#999999"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["observed", "chance"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("C. cross-seed classification")

    fig.tight_layout()
    fig.savefig(FIG / "fig_localization.png", dpi=150)
    plt.close(fig)
    print("saved:", FIG / "fig_localization.png")


def fig_recovery(rc):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    runs = rc["runs"]
    x = np.arange(len(FAILURES))

    ax = axes[0]
    for i, f in enumerate(FAILURES):
        for j, recover in enumerate([False, True]):
            sub = [r for r in runs if r["failure"] == f and r["recover"] == recover]
            if not sub:
                continue
            surv = sum(1 for r in sub if not r["fallen"])
            ax.bar(i + 0.22 * (0.5 if recover else -0.5), surv / len(sub), 0.42,
                   color=COLORS["on" if recover else "off"],
                   label=("halt ON" if recover and i == 0 else "halt OFF"
                          if not recover and i == 0 else None))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("survival fraction (n=3)")
    ax.set_title("A. survival, halt OFF vs ON")
    ax.legend(frameon=False)

    ax = axes[1]
    for j, recover in enumerate([False, True]):
        xs, ys = [], []
        for f in FAILURES:
            for r in runs:
                if (r["failure"] == f and r["recover"] == recover
                        and r["det_t"] is not None):
                    xs.append(FAILURES.index(f) + 0.22 * (0.5 if recover else -0.5))
                    ys.append(r["det_t"])
        ax.scatter(xs, ys, s=42, color=COLORS["on" if recover else "off"],
                   alpha=0.85, label="halt ON" if recover else "halt OFF")
    ax.axhline(2.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylabel("detection time (s)")
    ax.set_title("B. online detection time")
    ax.legend(frameon=False)

    ax = axes[2]
    for j, recover in enumerate([False, True]):
        xs, ys = [], []
        for f in FAILURES:
            for r in runs:
                if (r["failure"] == f and r["recover"] == recover
                        and r["lead_t"] is not None):
                    xs.append(FAILURES.index(f) + 0.22 * (0.5 if recover else -0.5))
                    ys.append(r["lead_t"])
        ax.scatter(xs, ys, s=42, color=COLORS["on" if recover else "off"],
                   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[f] for f in FAILURES], rotation=15)
    ax.set_ylabel("lead time to fall (s)")
    ax.set_title("C. detection lead to fall")
    ax.axhline(0, color="k", lw=0.8)

    fig.tight_layout()
    fig.savefig(FIG / "fig_recovery.png", dpi=150)
    plt.close(fig)
    print("saved:", FIG / "fig_recovery.png")


if __name__ == "__main__":
    ad = _load("fault_analysis_dead.json")
    lz = _load("localization_analysis.json")
    rc = _load("recovery_battery.json")
    fig_dead_null(ad)
    fig_localization(lz)
    fig_recovery(rc)