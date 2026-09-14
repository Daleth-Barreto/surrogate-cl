"""F1: aligned capture of per-tick spike counts + plant state, per mode.

Saves a JSON per (mode, seed): counts_matrix[64], x (hub input 5-dim), and
plant state (h, vx, roll, pitch, cmd) aligned per tick. Task-driven profile
(PROFILE) is used so that the command signal is identifiable for MI/TE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import ablate_loop as al  # noqa: E402  (registers bridge_g1 datasource)

SEEDS = [7, 29]
PROFILE = [(0.0, 0.5), (4.0, 0.8), (8.0, 0.5)]
RES = BASE.parent / "results"


def main():
    RES.mkdir(exist_ok=True)
    runs = []
    for mode in al.MODES:
        for seed in SEEDS:
            r = al.run_mode(mode, seed=seed, task=PROFILE, capture=True)
            name = f"f1_capture_{mode}_s{seed}.json"
            with open(RES / name, "w", encoding="utf-8") as fp:
                json.dump(r, fp)
            runs.append({"mode": mode, "seed": seed, "file": name,
                         "ticks": r["ticks"], "wall": r["_wall_step"],
                         "fallen": r["walker"]["fallen"],
                         "n_spikes_mean": r["mean_nspk"]})
            print(f"mode={mode:<8} seed={seed:<3} ticks={r['ticks']:<4} "
                  f"wall={r['_wall_step']:>5.1f}s fallen={r['walker']['fallen']} "
                  f"nspk={r['mean_nspk']}")
    with open(RES / "f1_capture_index.json", "w", encoding="utf-8") as fp:
        json.dump({"duration_sec": al.DURATION_SEC, "tps": al.TPS,
                   "profile": PROFILE, "seeds": SEEDS, "threshold": al.THR,
                   "poisson_lam": al.POISSON_LAM,
                   "notes": "task-driven capture; counts[64]+state per tick "
                            "(steps: counts,x aligned by tick.iteration; state "
                            "hold-last from bridge TSV at 50 Hz)",
                   "runs": runs}, fp, indent=2)
    print("saved index:", RES / "f1_capture_index.json")


if __name__ == "__main__":
    main()