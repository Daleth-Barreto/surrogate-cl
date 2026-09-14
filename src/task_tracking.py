"""H1 definitive: load-bearing via task-velocity tracking through the CL contract.

Multi-profile, multi-seed version with matched statistics. The task profile (a
desired forward-velocity schedule, like leader/obstacle telemetry) enters the
culture on channel 62 frames. The surrogate hub must re-emit it as muscle
command. Modes:
  neural  - hub decodes the profile (spike representation) -> tracks.
  random  - decoupled random readout -> large tracking error.
  mask0.5 - lesioned channels -> degraded decode.
  zero    - no function -> keeps the policy default (flat 0.5) -> error.

Metric: RMSE of commanded vx vs the desired profile plus walked distance,
aggregated per (profile, mode) over SEEDS. Paired statistics (exact
permutation, Cohen's dz, bootstrap CI) compare neural against each ablation on
matched seeds. Neural hub is deterministic (fixed Nengo seed): variance comes
from spike RNG, walker RNG and lesion draw.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ablate_loop import run_mode, MODES, TPS, DURATION_SEC

SEEDS = [1, 7, 13, 29, 55]
DEFAULT_SEED = 7
PASS_RMSE = 0.18

PROFILES = {
    "baseline": [(0.0, 0.5), (4.0, 0.8), (8.0, 0.5)],
    "multi_step": [(0.0, 0.5), (3.0, 0.65), (5.0, 0.8), (7.0, 0.65),
                   (9.0, 0.5)],
    "sine": [(i * 0.5,
              round(0.55 + 0.22 * math.sin(2 * math.pi * i * 0.5 / 5.0), 3))
             for i in range(24)],
    "descend": [(0.0, 0.8), (4.0, 0.5), (8.0, 0.6)],
    "pulse": [(0.0, 0.5), (2.0, 0.8), (2.5, 0.5), (5.0, 0.75),
              (5.6, 0.5), (8.0, 0.9), (8.6, 0.5)],
}
HEADLINE = "baseline"
PROFILE = PROFILES[HEADLINE]

RES = Path(__file__).resolve().parent.parent / "results"


def desired(sched, t):
    v = sched[0][1]
    for t0, vx in sched:
        if t >= t0:
            v = vx
    return v


def profile_sig(sched):
    ts = np.arange(0.0, DURATION_SEC, 1.0 / TPS) + 0.5 / TPS
    des = np.asarray([desired(sched, t) for t in ts], dtype=float)
    return der_std(des)


def der_std(des):
    return float(des.std())


def profile_windows(sched):
    wins = []
    for i, (t0, vx) in enumerate(sched):
        t1 = sched[i + 1][0] if i + 1 < len(sched) else DURATION_SEC
        wins.append((t0, vx, t1))
    return wins


def distance(ws):
    d = 0.0
    if not ws:
        return 0.0
    for i in range(1, len(ws)):
        d += (ws[i][0] - ws[i - 1][0]) * max(0.0, ws[i - 1][2])
    return d


def one_run(mode: str, seed: int, sched: list):
    r = run_mode(mode, seed=seed, task=sched)
    cmd = np.asarray(r["cmd_series"], dtype=float)
    ts = (np.arange(len(cmd)) + 0.5) / TPS
    des = np.asarray([desired(sched, t) for t in ts], dtype=float)
    rmse = float(np.sqrt(np.mean((cmd - des) ** 2)))
    ch62 = np.asarray(r.get("ch62_series", [0.0] * len(cmd)), dtype=float)
    segs = []
    for t0, vx, t1 in profile_windows(sched):
        m = (ts >= t0) & (ts < min(t1, DURATION_SEC))
        segs.append({
            "t0": t0, "vx": vx,
            "rmse": float(np.sqrt(np.mean((cmd[m] - vx) ** 2))),
            "cmd_mean": float(np.mean(cmd[m])),
            "ch62_mean": float(np.mean(ch62[m])),
            "n": int(m.sum()),
        })
    return {
        "seed": seed, "mode": mode,
        "rmse": round(rmse, 4),
        "nrmse": round(rmse / der_std(des), 3) if der_std(des) > 1e-9 else None,
        "segs": segs,
        "fallen": bool(r["walker"].get("fallen")),
        "distance_m": round(distance(r["walker_series"]), 2),
        "cmd_series": [round(float(v), 3) for v in cmd],
        "ch62_series": [round(float(v), 2) for v in ch62],
    }


def _stat(vals, nd):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return round(0.0, nd)
    return round(float(v.std(ddof=1)), nd)


def _ci_mean(vals, n_resamp=5000):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(123)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n_resamp)])
    return [round(float(np.percentile(means, 2.5)), 3),
            round(float(np.percentile(means, 97.5)), 3)]


def _exact_paired_p(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    T = abs(float(d.mean()))
    cnt = tot = 0
    for mask in range(1 << len(d)):
        signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(len(d))])
        Tp = abs(float((d * signs).mean()))
        cnt += Tp >= T - 1e-9
        tot += 1
    return cnt / tot


def _cohen_dz(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(d.std(ddof=1))
    return float(d.mean() / sd) if sd > 0 else 0.0


def aggregate(runs):
    rmses = np.array([r["rmse"] for r in runs])
    dists = np.array([r["distance_m"] for r in runs])
    nrm = [r["nrmse"] for r in runs if r["nrmse"] is not None]
    seg_stats = []
    for j in range(len(runs[0]["segs"])):
        entry = {"t0": runs[0]["segs"][j]["t0"], "vx": runs[0]["segs"][j]["vx"]}
        for stat in ("rmse", "cmd_mean", "ch62_mean"):
            vals = [r["segs"][j][stat] for r in runs]
            entry[stat] = _stat(vals, 2 if stat == "ch62_mean" else 3)
            entry[stat + "_mean"] = round(float(np.mean(vals)),
                                          2 if stat == "ch62_mean" else 3)
        seg_stats.append(entry)
    agg = {
        "mode": runs[0]["mode"],
        "n": len(runs),
        "rmse_mean": round(float(rmses.mean()), 3),
        "rmse_std": _stat(rmses, 3),
        "rmse_min": round(float(rmses.min()), 3),
        "rmse_max": round(float(rmses.max()), 3),
        "rmse_ci95": _ci_mean(rmses),
        "rmse_seeds": [round(float(v), 3) for v in rmses],
        "nrmse_mean": round(float(np.mean(nrm)), 2) if nrm else None,
        "fallen_any": any(r["fallen"] for r in runs),
        "distance_mean": round(float(dists.mean()), 2),
        "distance_std": _stat(dists, 2),
        "segs": seg_stats,
    }
    mid = int(np.argsort(rmses)[len(runs) // 2])
    agg["representative"] = {
        "seed": runs[mid]["seed"],
        "cmd_series": runs[mid]["cmd_series"],
        "ch62_series": runs[mid]["ch62_series"],
    }
    return agg


def paired_stats(neural_runs, ablated_runs):
    a = np.array([r["rmse"] for r in neural_runs], dtype=float)
    b = np.array([r["rmse"] for r in ablated_runs], dtype=float)
    return {
        "neural_mean": round(float(a.mean()), 3),
        "ablated_mean": round(float(b.mean()), 3),
        "delta_mean": round(float(a.mean() - b.mean()), 3),
        "cohen_dz": round(_cohen_dz(a, b), 2),
        "p_exact_perm": round(_exact_paired_p(a, b), 4),
        "ci95": _ci_mean(a - b),
    }


def run_profile(name, sched):
    rows, per_seed = {}, []
    for mode in MODES:
        runs = [one_run(mode, s, sched) for s in SEEDS]
        per_seed.extend(runs)
        rows[mode] = aggregate(runs)
        print("  %-8s rmse=%7.3f±%.3f ci=%s seeds=%s fallen=%s nrmse=%s"
              % (mode, rows[mode]["rmse_mean"], rows[mode]["rmse_std"],
                 rows[mode]["rmse_ci95"], rows[mode]["rmse_seeds"],
                 rows[mode]["fallen_any"], rows[mode]["nrmse_mean"]))
    stats = {}
    for ab in ("random", "mask0.5", "zero"):
        key = "neural_vs_" + ab
        stats[key] = paired_stats(per_seed_rows(per_seed, "neural"),
                                  per_seed_rows(per_seed, ab))
        print("    %s: d=%+.2f p=%s delta=%+.3f" % (
            key, stats[key]["cohen_dz"], stats[key]["p_exact_perm"],
            stats[key]["delta_mean"]))
    time.sleep(0.3)
    return rows, per_seed, stats


def per_seed_rows(per_seed, mode):
    return [r for r in per_seed if r["mode"] == mode]


def headline_figure(rows, per_seed, sched):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    ts = (np.arange(500) + 0.5) / TPS
    des = [desired(sched, t) for t in ts]
    for main, ax in (("neural", axes[0]), ("mask0.5", axes[1])):
        ax.plot(ts, des, "-", color="k", lw=2, label="target profile")
        for r in per_seed:
            if r["mode"] != main:
                continue
            ax.plot(ts[:len(r["cmd_series"])], r["cmd_series"], lw=0.7,
                    alpha=0.75)
        ax.set_ylabel("cmd vx")
        ax.set_title("%s (per seed, profile=%s)" % (main, HEADLINE))
        ax.legend(loc="upper right", fontsize=8)
    modes = [r["mode"] for r in rows]
    means = [r["rmse_mean"] for r in rows]
    stds = [r["rmse_std"] for r in rows]
    axes[2].bar(modes, means, yerr=stds, capsize=4, color="#4C72B0")
    axes[2].axhline(PASS_RMSE, color="r", ls="--", lw=1)
    axes[2].text(3.4, PASS_RMSE + 0.01, "pass RMSE=0.18", color="r", fontsize=8,
                 ha="center")
    for i, r in enumerate(rows):
        axes[2].text(i, (means[i] + stds[i]) * 1.02, "%.3f±%.3f" % (
            r["rmse_mean"], r["rmse_std"]), ha="center", fontsize=8)
    axes[2].set_ylabel("RMSE cmd vs profile")
    axes[2].set_title("multi-seed RMSE (%s)" % ", ".join(map(str, SEEDS)))
    fig.tight_layout()
    return fig


def multi_figure(profiles_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(profiles_data)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axs = axes.ravel()
    for i, name in enumerate(names):
        rows = profiles_data[name]["rows"]
        stats = profiles_data[name]["stats"]
        modes = [r["mode"] for r in rows]
        means = [r["rmse_mean"] for r in rows]
        stds = [r["rmse_std"] for r in rows]
        ci = [r["rmse_ci95"] for r in rows]
        ax = axs[i]
        ax.bar(modes, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85)
        for bx, r in zip(range(len(rows)), rows):
            y = r["rmse_seeds"]
            ax.scatter(np.full(len(y), bx) + 0.15, y, s=12, color="k", zorder=3)
            ax.errorbar(bx - 0.25, r["rmse_mean"], yerr=[[r["rmse_mean"] - ci[bx][0]],
                                                         [ci[bx][1] - r["rmse_mean"]]],
                        fmt="none", ecolor="firebrick", capsize=2)
        ax.axhline(PASS_RMSE, color="r", ls="--", lw=0.8)
        st = stats["neural_vs_random"]
        ax.set_title("%s\np=%.3f d=%.2f Δ=%.3f" % (name, st["p_exact_perm"],
                                                   st["cohen_dz"],
                                                   st["delta_mean"]), fontsize=9)
        ax.set_ylabel("RMSE" if i % 3 == 0 else "")
        ax.tick_params(axis="x", labelsize=8)
    ax = axs[5]
    x = np.arange(len(names))
    w = 0.2
    for j, mode in enumerate(MODES):
        vals = [profiles_data[n]["rows_j"][mode]["rmse_mean"]
                for n in names]
        ax.bar(x + j * w, vals, w, label=mode)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.axhline(PASS_RMSE, color="r", ls="--", lw=0.8)
    ax.set_ylabel("RMSE per profile")
    ax.set_title("cross-profile", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("H1 multi-profile: RMSE cmd vs desired profile (5 seeds, CI bootstrap)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    RES.mkdir(exist_ok=True)
    profiles_data = {}
    global per_seed_cache
    per_seed_cache = {}
    for name, sched in PROFILES.items():
        print("profile=%s sched=%s" % (name, sched))
        rows, per_seed, stats = run_profile(name, sched)
        profiles_data[name] = {
            "sched": sched,
            "rows": rows,
            "per_seed": [{k: v for k, v in r.items()
                          if k not in ("cmd_series", "ch62_series")}
                         for r in per_seed],
            "stats": stats,
            "rows_j": {m: {k: v for k, v in agg.items()
                           if k not in ("segs", "representative")}
                       for m, agg in rows.items()},
        }
        per_seed_cache[name] = per_seed

    global_rows = {}
    for name in PROFILES:
        global_rows[name] = profiles_data[name]["rows"]
    n_lt_random = sum(1 for n in PROFILES
                      if profiles_data[n]["stats"]["neural_vs_random"]["delta_mean"] < 0
                      and profiles_data[n]["stats"]["neural_vs_random"]["p_exact_perm"] <= 0.05)
    n_lt_zero = sum(1 for n in PROFILES
                    if profiles_data[n]["stats"]["neural_vs_zero"]["delta_mean"] < 0
                    and profiles_data[n]["stats"]["neural_vs_zero"]["p_exact_perm"] <= 0.05)
    n_lt_mask = sum(1 for n in PROFILES
                    if profiles_data[n]["stats"]["neural_vs_mask0.5"]["delta_mean"] < 0
                    and profiles_data[n]["stats"]["neural_vs_mask0.5"]["p_exact_perm"] <= 0.05)
    out = {
        "profile": PROFILES[HEADLINE],
        "headline": HEADLINE,
        "profiles": PROFILES,
        "pass_rmse": PASS_RMSE,
        "seeds": SEEDS,
        "default_seed": DEFAULT_SEED,
        "duration_sec": DURATION_SEC,
        "profiles_data": profiles_data,
        "global": {
            "n_profiles": len(PROFILES),
            "n_profiles_neural_lt_random_p005": n_lt_random,
            "n_profiles_neural_lt_zero_p005": n_lt_zero,
            "n_profiles_neural_lt_mask_p005": n_lt_mask,
        },
    }
    with open(RES / "task_tracking.json", "w") as fp:
        json.dump(out, fp, indent=2)

    rows = profiles_data[HEADLINE]["rows"]
    per_seed = per_seed_cache[HEADLINE]
    fig = headline_figure([rows[m] for m in MODES],
                          per_seed, PROFILES[HEADLINE])
    fig.savefig(RES / "task_tracking.png", dpi=140)
    fig2 = multi_figure(profiles_data)
    fig2.savefig(RES / "task_tracking_multi.png", dpi=140)
    print("saved:", RES / "task_tracking.json", RES / "task_tracking.png",
          RES / "task_tracking_multi.png")
    print("GLOBAL: neural<random p<=0.05 in %d/%d, "
          "<zero %d/%d, <mask0.5 %d/%d" % (
              n_lt_random, len(PROFILES), n_lt_zero, len(PROFILES),
              n_lt_mask, len(PROFILES)))


per_seed_cache = {}

if __name__ == "__main__":
    main()