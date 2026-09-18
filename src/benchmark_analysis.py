"""Benchmark analysis: cross-substrate comparison from failure_battery.json.

Produces:
  fig_benchmark_healthy.png   – bar charts comparing 4 substrates in healthy
  fig_benchmark_faultmatrix.png – survival heatmap (failures × conditions)
  fig_benchmark_radar.png     – radar portrait per substrate
  results/benchmark_summary.json – compact numeric summary
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RES = Path(__file__).resolve().parent.parent / "results"
FIG = RES.parent / "figures"
FIG.mkdir(exist_ok=True)

SUBSTRATE_LABELS = {
    "neural_rate": "Rate",
    "neural_izh": "Izh",
    "neural_mlp": "MLP",
    "poisson": "Poisson",
    "zero": "Zero",
}
SUBSTRATE_ORDER = ["neural_rate", "neural_izh", "neural_mlp", "poisson", "zero"]
SUBSTRATE_COLORS = {
    "neural_rate": "#2196F3",
    "neural_izh": "#FF9800",
    "neural_mlp": "#9C27B0",
    "poisson": "#4CAF50",
    "zero": "#9E9E9E",
}
FAILURE_ORDER = [
    "healthy", "freeze_lknee", "freeze_rknee", "freeze_lhip",
    "degrade_50", "degrade_25", "jitter_leg",
]
FAILURE_LABELS = {
    "healthy": "Healthy",
    "freeze_lknee": "Freeze L-knee",
    "freeze_rknee": "Freeze R-knee",
    "freeze_lhip": "Freeze L-hip",
    "degrade_50": "Degrade 50%",
    "degrade_25": "Degrade 25%",
    "jitter_leg": "Jitter leg",
}


def load_data():
    with open(RES / "failure_battery.json") as f:
        return json.load(f)


# ---------- fig 1: healthy comparison ----------

def fig_healthy(data):
    runs = data["run"]["healthy"]
    conds = SUBSTRATE_ORDER
    labels = [SUBSTRATE_LABELS[c] for c in conds]
    colors = [SUBSTRATE_COLORS[c] for c in conds]

    rmse_means = [runs[c]["rmse_mean"] for c in conds]
    rmse_stds = [runs[c]["rmse_std"] for c in conds]
    dist_means = [runs[c]["distance_mean"] for c in conds]
    dist_stds = [runs[c]["distance_std"] for c in conds]
    survivals = [runs[c]["survival"] for c in conds]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(conds))
    w = 0.55

    # (a) RMSE
    ax = axes[0]
    bars = ax.bar(x, rmse_means, w, yerr=rmse_stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE (command error)")
    ax.set_title("(a) Trajectory RMSE")
    for b, m, s in zip(bars, rmse_means, rmse_stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.005,
                f"{m:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(rmse_means) * 1.25)

    # (b) Distance
    ax = axes[1]
    bars = ax.bar(x, dist_means, w, yerr=dist_stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Distance walked (m)")
    ax.set_title("(b) Walking Distance")
    for b, m, s in zip(bars, dist_means, dist_stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.05,
                f"{m:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(dist_means) * 1.2)

    # (c) Survival
    ax = axes[2]
    bars = ax.bar(x, survivals, w, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Survival fraction")
    ax.set_title("(c) Survival")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="k", ls=":", lw=0.6)
    for b, s in zip(bars, survivals):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.03,
                f"{s:.2f}", ha="center", fontsize=9)

    fig.suptitle("Substrate Comparison — Healthy Condition (4 seeds, 12 s each)",
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG / "fig_benchmark_healthy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ---------- fig 2: fault survival matrix ----------

def fig_fault_matrix(data):
    runs = data["run"]
    fails = FAILURE_ORDER
    conds = SUBSTRATE_ORDER

    matrix = np.zeros((len(fails), len(conds)))
    for i, f in enumerate(fails):
        for j, c in enumerate(conds):
            matrix[i, j] = runs[f][c]["survival"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = LinearSegmentedColormap.from_list(
        "surv", ["#d32f2f", "#ff9800", "#4caf50"], N=256)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([SUBSTRATE_LABELS[c] for c in conds], fontsize=10, rotation=15, ha="right")
    ax.set_yticks(range(len(fails)))
    ax.set_yticklabels([FAILURE_LABELS[f] for f in fails], fontsize=9)

    for i in range(len(fails)):
        for j in range(len(conds)):
            v = matrix[i, j]
            txt = f"{v:.2f}"
            tc = "white" if v < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    fontweight="bold", color=tc)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Survival (0=fell, 1=walked)")
    ax.set_title("Survival Matrix: Failures x Substrates", fontsize=12, pad=12)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    out = FIG / "fig_benchmark_faultmatrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ---------- fig 3: radar chart ----------

def fig_radar(data):
    runs = data["run"]
    conds = SUBSTRATE_ORDER

    # Build axes (normalized 0-1, higher=better on all):
    #   1) RMSE accuracy (1 - normalized RMSE, lower RMSE → higher score)
    #   2) Walking distance (normalized to max)
    #   3) Robustness = mean survival across all failures
    #   4) Height maintenance (h_last_mean normalized)
    #   5) Speed consistency (1 - normalized RMSE std)

    healthy = runs["healthy"]
    rmse_vals = {c: healthy[c]["rmse_mean"] for c in conds}
    dist_vals = {c: healthy[c]["distance_mean"] for c in conds}
    hlast_vals = {c: healthy[c]["h_last_mean"] for c in conds}
    rmse_std_vals = {c: healthy[c]["rmse_std"] for c in conds}

    robust_vals = {}
    for c in conds:
        survs = [runs[f][c]["survival"] for f in FAILURE_ORDER]
        robust_vals[c] = np.mean(survs)

    axis_names = [
        "RMSE accuracy",
        "Distance",
        "Robustness",
        "Height ctrl",
        "Consistency",
    ]

    def norm(val_dict, invert=False):
        vals = np.array([val_dict[c] for c in conds])
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx > mn else 1.0
        normed = (vals - mn) / rng
        if invert:
            normed = 1.0 - normed
        # Ensure at least a small value so radar doesn't collapse to 0
        normed = np.clip(normed, 0.05, 1.0)
        return {c: float(normed[i]) for i, c in enumerate(conds)}

    rmse_norm = norm(rmse_vals, invert=True)
    dist_norm = norm(dist_vals)
    rob_norm = robust_vals  # already 0-1
    hlast_norm = norm(hlast_vals)
    rstd_norm = norm(rmse_std_vals, invert=True)

    n = len(axis_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for c in conds:
        vals = [rmse_norm[c], dist_norm[c], rob_norm[c],
                hlast_norm[c], rstd_norm[c]]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=SUBSTRATE_LABELS[c],
                color=SUBSTRATE_COLORS[c], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=SUBSTRATE_COLORS[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=9)
    ax.set_title("Substrate Portrait (normalized metrics)", fontsize=12, pad=20)
    fig.tight_layout()
    out = FIG / "fig_benchmark_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ---------- summary JSON + console ----------

def write_summary(data):
    runs = data["run"]
    summary = {"healthy": {}, "failure_survival": {}}

    # Healthy per substrate
    print("\n=== HEALTHY CONDITION (per substrate) ===")
    print(f"{'Substrate':<12} {'RMSE':>8} {'±std':>7} {'Distance':>9} {'±std':>7} {'Survival':>9}")
    print("-" * 55)
    for c in SUBSTRATE_ORDER:
        r = runs["healthy"][c]
        print(f"{SUBSTRATE_LABELS[c]:<12} {r['rmse_mean']:>8.3f} {r['rmse_std']:>7.3f} "
              f"{r['distance_mean']:>9.2f} {r['distance_std']:>7.2f} {r['survival']:>9.3f}")
        summary["healthy"][c] = {
            "rmse_mean": r["rmse_mean"],
            "rmse_std": r["rmse_std"],
            "distance_mean": r["distance_mean"],
            "distance_std": r["distance_std"],
            "survival": r["survival"],
        }

    # Failure survival matrix
    print("\n=== SURVIVAL BY FAILURE & SUBSTRATE ===")
    header = f"{'Failure':<16}" + "".join(f" {SUBSTRATE_LABELS[c]:>8}" for c in SUBSTRATE_ORDER)
    print(header)
    print("-" * len(header))
    for f in FAILURE_ORDER:
        row = f"{FAILURE_LABELS[f]:<16}"
        summary["failure_survival"][f] = {}
        for c in SUBSTRATE_ORDER:
            s = runs[f][c]["survival"]
            row += f" {s:>8.3f}"
            summary["failure_survival"][f][c] = s
        print(row)

    # Anomalies
    print("\n=== NOTABLE OBSERVATIONS ===")
    zero_h = runs["healthy"]["zero"]
    mlp_h = runs["healthy"]["neural_mlp"]
    print(f"  - 'zero' (no control) survives healthy: dist={zero_h['distance_mean']:.2f}m, "
          f"RMSE={zero_h['rmse_mean']:.3f} — robot coasts on initial velocity")
    print(f"  - 'mlp' now spiking (rate-matched): healthy dist={mlp_h['distance_mean']:.2f}m "
          f"RMSE={mlp_h['rmse_mean']:.3f}, distinct across faults "
          f"(0.89-2.12m), survival 0 on freeze/degrade")

    out_path = RES / "benchmark_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out_path}")
    return summary


def main():
    data = load_data()
    fig_healthy(data)
    fig_fault_matrix(data)
    fig_radar(data)
    write_summary(data)


if __name__ == "__main__":
    main()
