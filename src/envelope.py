"""F3: authority x latency envelope of the CL-contract loop.

Consolidates the disturbance probes into a design figure:
  Panel A  disturbance duration x force scatter (closed = survived, open =
           fell) over the lateral +y axis (frontier) + longitudinal points,
           with the torque-authority annotation (overlay cap vs tipping demand).
  Panel B  loop tick cadence histograms (per mode) vs the 25 ms 40 TPS
           deadline, plus bridge read() microbox.

Writes results/envelope.png + results/envelope.json (source points).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"

TAU_AVAIL = 30.0      # N*m overlay per joint (0.75 u x TAU_GAIN 40), contract 3 nC
TAU_TIP = 112.0       # N*m tipping demand (prior ballpark from probe analysis)
DEADLINE_MS = 25.0


def load(name):
    with open(RES / name, encoding="utf-8") as fp:
        return json.load(fp)


def main():
    frontier = load("push_frontier.json")
    sweep = load("perturb_sweep.json")
    probe2 = load("push_probe2.json")
    probe3 = load("push_probe3.json")

    pts = []

    def add(dur, f, fell, source, axis):
        pts.append({"dur": dur, "f": f, "fell": bool(fell),
                    "source": source, "axis": axis})

    for r in frontier["rows"]:             # lateral +y frontier
        add(r["dur"], r["fy"], r["fallen"], "frontier", "lat")
    for e in sweep["entries"]:             # impulses 0.15 s
        if e["mode"] == "neural":
            add(0.15, e["perturb_fy"], e["walker"]["fallen"], "sweep", "lat")
    for r in probe2["rows"]:               # sustained lateral
        add(r["dur"], r["fy"], r["fallen"], "probe2", "lat")
    for r in probe3["rows"]:               # longitudinal
        add(r["dur"], abs(r["fx"]), r["fallen"], "probe3", "lon")

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1)

    lat = [p for p in pts if p["axis"] == "lat"]
    lon = [p for p in pts if p["axis"] == "lon"]
    for p in lat:
        ax.scatter(p["dur"], p["f"],
                   marker="o" if not p["fell"] else "x",
                   s=42 if not p["fell"] else 90,
                   c="tab:green" if not p["fell"] else "tab:red",
                   edgecolors="k", linewidths=0.6, zorder=3)
    for p in lon:
        ax.scatter(p["dur"], p["f"],
                   marker="^" if not p["fell"] else "v",
                   c="tab:green" if not p["fell"] else "tab:red",
                   edgecolors="tab:purple", linewidths=0.8, s=70, zorder=3)
    ax.axvspan(0.0, 0.42, color="tab:green", alpha=0.12)
    ax.axvspan(0.45, 1.05, color="tab:red", alpha=0.12)
    ax.text(0.21, 205, "impulse\n(closable)", ha="center", fontsize=9,
            color="tab:green", weight="bold")
    ax.text(0.75, 205, "sustained\n(not closable)", ha="center", fontsize=9,
            color="tab:red", weight="bold")
    ax.annotate("lateral frontier ~0.4-0.45 s @140 N",
                xy=(0.45, 140), xytext=(0.55, 175),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=8)
    ax.text(0.03, 35, "o/+ lateral survived | x lateral fell\n"
                      "^ longitudinal survived | v longitudinal fell",
            fontsize=7.5, va="bottom")
    ax.set_xlabel("disturbance duration (s)")
    ax.set_ylabel("disturbance force (N)")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0, 215)
    ax.set_title("A) Authority x disturbance envelope (G1 @ .5 m/s)")

    axx = fig.add_axes([0.035, 0.42, 0.165, 0.25], zorder=5)
    bars = axx.barh(["overlay avail.", "tipping demand"],
                    [TAU_AVAIL, TAU_TIP], color=["tab:blue", "tab:red"])
    axx.set_xlim(0, 130)
    axx.set_title("torque (N·m)", fontsize=8)
    axx.tick_params(labelsize=7)
    for b, v in zip(bars, [TAU_AVAIL, TAU_TIP]):
        axx.text(v + 2, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                 va="center", fontsize=7)
    for sp in ["top", "right"]:
        axx.spines[sp].set_visible(False)

    ax2 = fig.add_subplot(1, 2, 2)
    for mode, color in [("neural", "tab:blue"), ("poisson", "tab:gray"),
                        ("mask0.5", "tab:orange"), ("random", "tab:purple")]:
        try:
            r = load(f"f1_capture_{mode}_s7.json")
        except FileNotFoundError:
            continue
        h = np.asarray(r["_latency_hist"], dtype=float) / 1e3   # ms
        ax2.hist(h, bins=40, alpha=0.45, label=mode, color=color)
    ax2.axvline(DEADLINE_MS, color="red", ls="--", lw=1.2)
    ax2.text(DEADLINE_MS + 1, ax2.get_ylim()[1] * 0.85,
             "deadline 25 ms @40 TPS", color="red", fontsize=8)
    ax2.set_xlabel("SDK loop tick interval (ms)")
    ax2.set_ylabel("ticks")
    ax2.set_title("B) Loop cadence vs contract deadline")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RES / "envelope.png", dpi=150)

    summary = {
        "tau_available_nm": TAU_AVAIL,
        "tau_tipping_nm": TAU_TIP,
        "deadline_ms": DEADLINE_MS,
        "lateral_frontier_s_at_140N": "0.40 survived / 0.45 fell",
        "lateral_closable_max": "dur < 0.45 s, F <= 140 N (base absorbs)",
        "longitudinal": "fwd fell at 0.3 s/150 N; back survived 0.3 s/150 N (zero)",
        "points": {"n": len(pts), "survived": sum(not p["fell"] for p in pts),
                   "fell": sum(p["fell"] for p in pts)},
    }
    with open(RES / "envelope.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print("saved:", RES / "envelope.png", RES / "envelope.json")


if __name__ == "__main__":
    main()