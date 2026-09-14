"""F3: fine authority--disturbance frontier (isocontour of falling).

Lateral +y push at t0=5 s, sweeping push duration x force, mode neural as
primary actuator; zero included on a subset to show the envelope is physical
(base policy), not decoder-dependent. Used to trace the 'closable vs
not-closable' boundary of the authority x latency envelope.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ablate_loop import run_mode

SEED = 7
DUR_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]
FORCE_GRID = [100.0, 120.0, 140.0]
ZERO_CHECK = [(0.25, 140.0), (0.30, 140.0)]


def main():
    RES = Path(__file__).resolve().parent.parent / "results"
    rows = []

    def probe(dur, fy, mode):
        sch = [{"t0": 5.0, "dur": dur, "force": [0.0, fy, 0.0]}]
        r = run_mode(mode, seed=SEED, perturb=sch)
        w = r["walker"]
        rows.append({"dur": dur, "fy": fy, "mode": mode,
                     "fallen": w.get("fallen"), "h_last": w.get("h_last"),
                     "vx_last": w.get("vx_last"),
                     "mean_tau": r.get("mean_tau")})
        print("dur=%.2fs fy=%+.0fN mode=%-7s fallen=%s h=%.2f vx=%.2f" %
              (dur, fy, mode, w.get("fallen"), w.get("h_last") or -1.0,
               w.get("vx_last") or -1.0))
        time.sleep(0.5)

    for dur in DUR_GRID:
        for fy in FORCE_GRID:
            probe(dur, fy, "neural")
    for dur, fy in ZERO_CHECK:
        probe(dur, fy, "zero")

    out = {"t0_push": 5.0, "axis": "+y lateral", "seed": SEED,
           "rows": rows}
    with open(RES / "push_frontier.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=1)
    print("rows:", len(rows))


if __name__ == "__main__":
    main()