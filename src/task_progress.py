"""H1b: minimum-progress task as the load-bearing operationalisation.

On flat ground the G1 policy is too robust for decoder ablations to show up as
falls, and under impulsive loads the loop's authority (vx channel + 30 Nm tau)
and latency are below what is needed. Reframe: the loop is load-bearing when
the TASK demands the decoded velocity — "walk >= 5.4 m in 12 s". zero (cmd=0)
and a decoupled random readout should degrade the task vs the spike-decoded
neural command. Distance is integrated from telemetry (t, vx) series.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ablate_loop import run_mode

SEED = 7
GOAL_M = 5.4   # 0.45 m/s x 12 s


def distance(ws):
    if not ws:
        return 0.0
    ts = [s[0] for s in ws]
    vx = [s[2] for s in ws]
    d = 0.0
    for i in range(1, len(ts)):
        d += (ts[i] - ts[i - 1]) * max(0.0, vx[i - 1])
    return d


def main():
    RES = Path(__file__).resolve().parent.parent / "results"
    RES.mkdir(exist_ok=True)
    rows = []
    for mode in ("neural", "zero", "random", "mask0.5"):
        r = run_mode(mode, seed=SEED)
        ws = r["walker_series"]
        dist = distance(ws)
        task = dist >= GOAL_M
        rows.append({"mode": mode, "dist": round(dist, 2), "task_ok": task,
                     "fallen": r["walker"].get("fallen"),
                     "mean_cmd": r["mean_cmd"], "vx_last": r["walker"].get("vx_last")})
        print("mode=%-8s dist=%.2fm task=%s fallen=%s cmd=%.2f vx_last=%.2f"
              % (mode, dist, task, r["walker"].get("fallen"),
                 r["mean_cmd"], r["walker"].get("vx_last") or 0.0))
        time.sleep(0.5)
    with open(RES / "task_progress.json", "w") as fp:
        json.dump({"goal_m": GOAL_M, "rows": rows}, fp, indent=2)
    print("goal_m=", GOAL_M)


if __name__ == "__main__":
    main()