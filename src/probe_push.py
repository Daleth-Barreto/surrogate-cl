"""Probe 3: forward/backward pushes to test vx deceleration as load-bearing.

A backward push (force +x) pitches the walker forward; the neural hub reads
|pitch| and reduces commanded vx, reducing forward momentum. Zero stays at
cm=0, random fluctuates. If neural survives where zero/random fall, the
spike-decoded vx decision is load-bearing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import demo_walk
from ablate_loop import TPS, DURATION_SEC, run_mode

SEED = 7
CONFIGS = [
    (0.30, 150.0, +150.0),
    (0.50, 120.0, +120.0),
    (0.50, 150.0, +150.0),
    (0.50, 200.0, +200.0),
    (0.30, 150.0, -150.0),
]
MODES = ["neural", "zero", "random"]


def main():
    RES = Path(__file__).resolve().parent.parent / "results"
    RES.mkdir(exist_ok=True)
    rows = []
    for dur, fmag, fx in CONFIGS:
        sch = [{"t0": 5.0, "dur": dur, "force": [fx, 0.0, 0.0]}]
        for mode in MODES:
            r = run_mode(mode, seed=SEED, perturb=sch)
            w = r["walker"]
            rows.append({"dur": dur, "fx": fx, "fm": fmag, "mode": mode,
                         "fallen": w.get("fallen"), "h_last": w.get("h_last"),
                         "vx_last": w.get("vx_last"),
                         "mean_cmd": r.get("mean_cmd")})
            print("dur=%.2fs fx=%+.0fN mode=%-7s fallen=%s h=%.2f vx=%.2f "
                  "cmd=%.2f" % (dur, fx, mode, w.get("fallen"),
                                w.get("h_last") or -1.0,
                                w.get("vx_last") or -1.0,
                                r.get("mean_cmd") or 0.0))
            time.sleep(0.5)
    with open(RES / "push_probe3.json", "w") as fp:
        json.dump({"rows": rows}, fp, indent=2)
    print("rows:", len(rows))


if __name__ == "__main__":
    main()