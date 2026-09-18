"""Dead-substrate captures: per-tick spike counts[64] under actuator faults with
a Poisson dead null (no state-correlated signal).

Identical to fault_detection.py but mode='poisson' replaces spike counts with
independent Poisson draws (lambda=0.14) to serve as a matched dead-null control.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fault_detection import FAILURES, SEEDS, T_FAULT
from ablate_loop import run_mode, TPS

RES = Path(__file__).resolve().parent.parent / "results"


def capture(failure: str, seed: int):
    spec = FAILURES[failure]
    r = run_mode("poisson", seed=seed, task=[(0.0, 0.5)], substrate="rate",
                 failures=spec["spec"], capture=True)
    counts = np.asarray(r["counts_matrix"], dtype=float)
    x = np.asarray(r["x_matrix"], dtype=float)
    ws = r["walker_series"]
    fallen_t = None
    for i in range(1, len(ws)):
        if ws[i][1] < 0.35:
            fallen_t = ws[i][0]
            break
    return {"failure": failure, "joint": spec["joint"], "label": spec["label"],
            "seed": seed, "counts": counts, "x": x,
            "fallen_t": fallen_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--failures", default=",".join(FAILURES))
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    fails = [f for f in args.failures.split(",") if f in FAILURES]
    RES.mkdir(exist_ok=True)
    t0 = time.monotonic()
    out = {"seeds": seeds, "failures": fails, "t_fault": T_FAULT,
           "tps": TPS, "mode": "poisson", "runs": []}
    for failure in fails:
        for seed in seeds:
            r = capture(failure, seed)
            counts = np.asarray(r["counts"], dtype=np.float32)
            x = np.asarray(r["x"], dtype=np.float32)
            npz = RES / ("fault_dead_%s_s%d.npz" % (failure, seed))
            np.savez(npz, counts=counts, x=x,
                     fallen_t=r["fallen_t"] if r["fallen_t"] is not None else -1.0)
            meta = {k: v for k, v in r.items() if k not in ("counts", "x")}
            meta["npz"] = str(npz.name)
            out["runs"].append(meta)
            print("[%s s=%d] joint=%s fallen_t=%s (%.1fs)" % (
                failure, seed, FAILURES[failure]["joint"],
                r["fallen_t"], time.monotonic() - t0), flush=True)
        with open(RES / "fault_dead_captures.json", "w") as fp:
            json.dump(out, fp, indent=2)
    out["wall_sec"] = round(time.monotonic() - t0, 1)
    with open(RES / "fault_dead_captures.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("saved:", RES / "fault_dead_captures.json")


if __name__ == "__main__":
    main()
