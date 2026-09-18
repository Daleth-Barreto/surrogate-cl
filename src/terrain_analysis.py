"""Merge + statistics for the Fase 0.3 terrain probes.

Reads results/terrain_probe.json (full 4-mode battery, 2 seeds) and
results/terrain_probe2.json (reference/neural power battery, 6 seeds) and emits:
  results/terrain_stats.json      per-scene/mode aggregates + hypothesis tests
  results/terrain_summary.png     combined survival/distance figure
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("C:/Proyectos/papers/surrogate_cl/results")
MODES = ["reference", "neural", "zero", "random"]


def _ln_choose(nn, kk):
    if kk < 0 or kk > nn or nn < 0:
        return float("-inf")
    return (math.lgamma(nn + 1) - math.lgamma(kk + 1)
            - math.lgamma(nn - kk + 1))


def fisher_one_sided(a, a_total, b, b_total):
    """P(neural survives <= a) under null. 2x2: [a, a_total-a; b, b_total-b]."""
    n_a, n_b = a_total, b_total
    surv_total = a + b
    group = n_a
    return sum(math.exp(_ln_choose(surv_total, k) + _ln_choose(n_a + n_b - surv_total, n_a - k) - _ln_choose(n_a + n_b, n_a))
               for k in range(0, a + 1))


def _betacf(a, b, x, itmax=300, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    tiny = 1e-300
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h  # not converged


def _betainc(a, b, x):
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    ln_g = lambda z: 0.5 * math.log(2 * math.pi * z) + (z - 0.5) * math.log(z) - z
    bt = math.exp(ln_g(a + b) - ln_g(a) - ln_g(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def welch_ttest(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = math.sqrt(vx / nx + vy / ny)
    t = (x.mean() - y.mean()) / se if se > 0 else float("inf")
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1)
                                     + (vy / ny) ** 2 / (ny - 1))
    p_two = 2.0 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return round(t, 3), round(df, 1), round(min(p_two, 1.0), 6)


def load_rows(path, scene_map=None):
    with open(path, encoding="utf-8") as fp:
        rows = json.load(fp)["rows"]
    if scene_map:
        for r in rows:
            if r["scene"] in scene_map:
                r["scene"] = scene_map[r["scene"]]
    return rows


def main():
    rows2 = load_rows(RES / "terrain_probe2.json")
    fresh = {(r["scene"], r["mode"], r["seed"]) for r in rows2}
    rows1 = load_rows(RES / "terrain_probe.json", scene_map={"curb": "curb10"})
    rows1 = [r for r in rows1 if (r["scene"], r["mode"], r["seed"]) not in fresh]
    rows = rows1 + rows2
    scenes = sorted({r["scene"] for r in rows})
    print("rows=%d scenes=%s" % (len(rows), scenes))

    stats = {"scenes": {}}
    for sc in scenes:
        srows = [r for r in rows if r["scene"] == sc]
        per = {}
        for m in MODES:
            mrows = [r for r in srows if r["mode"] == m]
            if not mrows:
                continue
            surv = sum(1 for r in mrows if not r["fallen"])
            dist = [r["distance_m"] for r in mrows]
            per[m] = {
                "n": len(mrows),
                "survivors": surv,
                "survival": round(surv / len(mrows), 3),
                "dist_mean": round(float(np.mean(dist)), 2),
                "dist_sd": round(float(np.std(dist, ddof=1)), 3)
                if len(dist) > 1 else 0.0,
                "vx_last_mean": round(float(np.mean([r["vx_last"] for r in mrows])), 3),
            }
        stats["scenes"][sc] = per
        print("\n%s" % sc)
        for m, v in per.items():
            print("  %-9s n=%d surv=%d/%-2d dist=%.2f+-%.2f vx_last=%.2f" % (
                m, v["n"], v["survivors"], v["n"], v["dist_mean"],
                v["dist_sd"], v["vx_last_mean"]))

    tests = {}

    def _surv(scene, mode):
        return sum(1 for r in rows
                   if r["scene"] == scene and r["mode"] == mode
                   and not r["fallen"])

    for sc in ("curb", "rough"):
        nneu = sum(1 for r in rows if r["scene"] == sc and r["mode"] == "neural")
        nref = sum(1 for r in rows if r["scene"] == sc and r["mode"] == "reference")
        a = _surv(sc, "neural")
        b = _surv(sc, "reference")
        p_le = fisher_one_sided(a, nneu, b, nref)
        p_ge = fisher_one_sided(b, nref, a, nneu)
        tests[sc] = {"neural_survivors": a, "n_neural": nneu,
                     "reference_survivors": b, "n_reference": nref,
                     "fisher_P(neural_less_eq_ref)": round(p_le, 5),
                     "fisher_P(neural_geeq_ref)": round(p_ge, 5)}
        print("%s: neural %d/%d vs reference %d/%d  "
              "(fisher lower p=%.4f, upper p=%.4f)"
              % (sc, a, nneu, b, nref, p_le, p_ge))

    ref = [r["distance_m"] for r in rows
           if r["scene"] == "rough" and r["mode"] == "reference"]
    neu = [r["distance_m"] for r in rows
           if r["scene"] == "rough" and r["mode"] == "neural"]
    t, dof, p2 = welch_ttest(neu, ref)
    tests["rough_dist_welch"] = {
        "t": t, "dof": dof, "p_two_sided": p2,
        "ref_mean": round(float(np.mean(ref)), 3),
        "neu_mean": round(float(np.mean(neu)), 3)}
    print("rough dist: ref %.2f vs neural %.2f  (t=%.2f df=%.1f p=%.6f)"
          % (np.mean(ref), np.mean(neu), t, dof, p2))

    gen = 0
    for r in rows:
        if r["scene"] in ("curb", "rough") and r["mode"] in ("reference", "neural"):
            gen += 1
    tests["rows_merged"] = gen
    stats["tests"] = tests
    with open(RES / "terrain_stats.json", "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2, ensure_ascii=False)

    _figure(rows, scenes)
    print("saved results/terrain_stats.json + terrain_summary.png")


def _figure(rows, scenes):
    colors = {"reference": "#3498db", "neural": "#1e7d32", "zero": "#c0392b",
              "random": "#8e44ad"}
    show = [s for s in ("curb", "rough", "ramp") if s in scenes]
    fig, axes = plt.subplots(1, len(show), figsize=(5.4 * len(show), 4.6),
                             squeeze=False)
    for ax, sc in zip(axes.flat, show):
        mr_all = [r for r in rows if r["scene"] == sc]
        modes = [m for m in MODES
                 if any(r["mode"] == m for r in mr_all)]
        x = np.arange(len(modes))
        dist = []
        sur = []
        for m in modes:
            mr = [r for r in mr_all if r["mode"] == m]
            dist.append(np.mean([r["distance_m"] for r in mr]))
            sur.append(100.0 * np.mean([int(not r["fallen"]) for r in mr]))
            ax.bar(x[len(dist) - 1], dist[-1], width=0.62, color=colors[m],
                   alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_title("%s   (survival %s)" % (
            sc, " / ".join("%d%%" % int(round(s)) for s in sur)))
        ax.set_ylabel("mean forward distance (m)")
        ax.grid(alpha=0.3)
        for xi, (d, s) in enumerate(zip(dist, sur)):
            ax.annotate("d=%.1f\nsurv=%d%%" % (d, s), (xi, d), ha="center",
                        va="bottom", fontsize=8)
    fig.suptitle("Terrain robustness under the CL1 contract "
                 "(12 s runs, 0.5 m/s duty; 6 seeds per group reference/neural)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(RES / "terrain_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()