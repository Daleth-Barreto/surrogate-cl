"""F2: information-theoretic validation - does the substrate compute?

Loads f1 capture runs and estimates, per mode:
  * MI (mutual information) between per-channel spike counts and plant state
    (vx, h) -> "coding geography" of the substrate.
  * MI(vx_cmd, vx_plant): motor-path coupling (the bug detector / fix verifier).
  * MI(vx_cmd, task_target): task alignment of the decoded command.
  * TE (transfer entropy) spikes -> command at successive lags (causal check
    that cultural activity precedes command changes) and command -> plant.

All estimates are histogram-based with equiprobable bins and bias-corrected
against a shuffle null (50 permutations). Reported in nats.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RES = BASE.parent / "results"

MODES = ["neural", "zero", "random", "mask0.5", "poisson"]
SEEDS = [7, 29]
PROFILE = [(0.0, 0.5), (4.0, 0.8), (8.0, 0.5)]
N_PERM = 50
RNG = np.random.default_rng(2026)


def load_runs():
    runs = {}
    for mode in MODES:
        runs[mode] = {}
        for seed in SEEDS:
            p = RES / f"f1_capture_{mode}_s{seed}.json"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as fp:
                r = json.load(fp)
            counts = np.asarray(r["counts_matrix"], dtype=float)          # T,64
            st = r["state_aligned"]
            runs[mode][seed] = {
                "counts": counts,
                "nspk": counts.sum(axis=1),
                "vx_cmd": np.asarray(r["cmd_series"], dtype=float),
                "vx_applied": np.asarray(st["cmd"], dtype=float),
                "vx": np.asarray(st["vx"], dtype=float),
                "h": np.asarray(st["h"], dtype=float),
                "pitch": np.asarray(st["pitch"], dtype=float),
                "roll": np.asarray(st["roll"], dtype=float),
            }
    return runs


def _bins(x, nb):
    qs = np.quantile(x, np.linspace(0, 1, nb + 1))
    qs[0], qs[-1] = x.min() - 1e-9, x.max() + 1e-9
    return np.digitize(x, qs[1:-1]).astype(int)


def mi_binned(a, b, nb=5):
    ba, bb = _bins(a, nb), _bins(b, nb)
    n = len(ba)
    pa = np.bincount(ba, minlength=nb) / n
    pb = np.bincount(bb, minlength=nb) / n
    pab = np.zeros((nb, nb))
    np.add.at(pab, (ba, bb), 1)
    pab /= n
    msk = pab > 0
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    hab = -np.sum(pab[msk] * np.log(pab[msk]))
    return max(0.0, ha + hb - hab)


def te_binned(a, b, k=1, nb=4):
    """TE_{a->b}(k) = sum p(b_{n+1}, b_n, a_{n-k})
                      * log p(b_{n+1}|b_n, a_{n-k}) / p(b_{n+1}|b_n).
    Equiprobable nb bins per variable. Estimated from joint counts."""
    bb_t = _bins(b, nb)            # current target state
    bb_t1 = np.roll(bb_t, -1)      # target next (drop last later)
    ba_tk = np.roll(_bins(a, nb), k)  # source shifted by lag k
    L = len(bb_t) - k - 1
    n1 = bb_t1[:L]
    n0 = bb_t[:L]
    nk = ba_tk[:L]
    c = np.zeros((nb, nb, nb))
    np.add.at(c, (n1, n0, nk), 1)
    p = c / c.sum()
    p_both = p.sum(axis=0, keepdims=True) + 1e-15              # p(b_n, a_{n-k})
    p_cond_both = p / p_both                                   # p(b_{n+1}|b_n,a)
    p_next_n = p.sum(axis=2) + 1e-15                           # p(b_{n+1}, b_n)
    p_n = p.sum(axis=(0, 2)) + 1e-15                           # p(b_n)
    p_cond_n = p_next_n / p_n[None, :]                         # p(b_{n+1}|b_n)
    term = p * (np.log(p_cond_both + 1e-15)
                - np.log(p_cond_n[:, :, None] + 1e-15))
    return float(np.clip(term.sum(), 0, None))


def bias_correct(fn, a, b, *args):
    base = fn(a, b, *args)
    null = np.mean([fn(RNG.permutation(a), RNG.permutation(b), *args)
                    for _ in range(N_PERM)])
    return max(0.0, base - null)


def channel_mi(counts, vx, h):
    nc = counts.shape[1]
    mi_vx, mi_h = np.zeros(nc), np.zeros(nc)
    for c in range(nc):
        x = counts[:, c]
        if x.std() < 1e-9:
            continue
        mi_vx[c] = bias_correct(mi_binned, x, vx)
        mi_h[c] = bias_correct(mi_binned, x, h)
    return mi_vx, mi_h


def main():
    runs = load_runs()
    summary = {m: {} for m in MODES}
    for mode in MODES:
        for seed, d in runs[mode].items():
            vx, h = d["vx"], d["h"]
            tgt = np.full_like(vx, 0.5)
            tgt[vx * 0 + np.arange(len(vx)) * 0.025 >= 8.0] = 0.5
            tgt[(np.arange(len(vx)) * 0.025 >= 4.0)
                & (np.arange(len(vx)) * 0.025 < 8.0)] = 0.8
            mi_vx, mi_h = channel_mi(d["counts"], vx, h)
            te_spk_cmd = {k: bias_correct(te_binned, d["nspk"], d["vx_cmd"], k)
                          for k in (1, 2, 4, 6, 8)}
            te_cmd_plant = {k: bias_correct(te_binned, d["vx_cmd"], d["vx"], k)
                            for k in (1, 2, 4, 6, 8)}
            summary[mode][seed] = {
                "mi_cmd_plant": bias_correct(mi_binned, d["vx_cmd"], vx),
                "mi_cmd_applied": bias_correct(mi_binned, d["vx_cmd"],
                                               d["vx_applied"]),
                "mi_cmd_task": bias_correct(mi_binned, d["vx_cmd"], tgt),
                "mi_nspk_vx": bias_correct(mi_binned, d["nspk"], vx),
                "mi_vx_cmd_vs_applied": bias_correct(mi_binned,
                                                     d["vx_cmd"],
                                                     d["vx_applied"]),
                "channel_mi_vx": mi_vx.tolist(),
                "channel_mi_h": mi_h.tolist(),
                "te_spikes_to_cmd": te_spk_cmd,
                "te_cmd_to_plant": te_cmd_plant,
            }

    # ---- aggregate per mode (mean across seeds) + Gate A probe ----
    agg = {}
    for mode in MODES:
        keys = [k for k in summary[mode]]
        agg[mode] = {k: float(np.mean([summary[mode][s][k] for s in keys]))
                     for k in ["mi_cmd_plant", "mi_cmd_task", "mi_nspk_vx",
                               "mi_cmd_applied"]}
        agg[mode]["te_spikes_to_cmd_lag2"] = float(np.mean(
            [summary[mode][s]["te_spikes_to_cmd"][2] for s in keys]))

    with open(RES / "f2_signal_summary.json", "w", encoding="utf-8") as fp:
        json.dump({"modes": MODES, "seeds": SEEDS, "n_perm": N_PERM,
                   "units": "nats, bias-corrected (shuffle null)",
                   "per_run": summary, "aggregate": agg}, fp, indent=1)

    # ---- figure ----
    fig, axes = plt.subplots(3, len(MODES), figsize=(14, 8), sharex=True)
    for j, mode in enumerate(MODES):
        r0 = summary[mode][SEEDS[0]]
        im = axes[0][j].imshow(np.asarray(r0["channel_mi_vx"]).reshape(16, 4),
                               aspect="auto", cmap="viridis")
        axes[0][j].set_title(f"{mode} (seed {SEEDS[0]})")
        axes[0][j].set_ylabel("ch block (4 col = 16 rows)")
        fig.colorbar(im, ax=axes[0][j], fraction=0.046)
        names = ["mi_cmd_plant", "mi_cmd_task", "mi_nspk_vx",
                 "te_spk_cmd_l2"]
        vals = [agg[mode][n] for n in
                ["mi_cmd_plant", "mi_cmd_task", "mi_nspk_vx"]]
        vals.append(float(np.mean(
            [summary[mode][s]["te_spikes_to_cmd"][2] for s in summary[mode]])))
        axes[1][j].bar(names, vals)
        axes[1][j].tick_params(axis="x", rotation=90, labelsize=8)
        lags = [1, 2, 4, 6, 8]
        for s, d in summary[mode].items():
            vals = [max(d["te_spikes_to_cmd"][k], 1e-4) for k in lags]
            axes[2][j].plot(lags, vals, marker="o", lw=0.6, label=f"seed{s}")
        axes[2][j].set_yscale("log")
        axes[2][j].set_xlabel("TE spikes->cmd, lag (ticks)")
        axes[2][j].legend(fontsize=7)
    axes[1][0].set_ylabel("nats (corr.)")
    fig.suptitle("Does the substrate compute? MI/TE per mode (task-driven capture)")
    fig.tight_layout()
    fig.savefig(RES / "f2_signal_fig.png", dpi=140)
    print("saved:", RES / "f2_signal_summary.json", RES / "f2_signal_fig.png")
    print("== Gate A (aggregate) ==")
    for m in MODES:
        print(f"  {m:<8} mi_cmd_plant={agg[m]['mi_cmd_plant']:.4f} "
              f"mi_cmd_task={agg[m]['mi_cmd_task']:.4f} "
              f"mi_nspk_vx={agg[m]['mi_nspk_vx']:.4f} "
              f"te_spk_cmd_lag2={agg[m]['te_spikes_to_cmd_lag2']:.4f}")


if __name__ == "__main__":
    main()