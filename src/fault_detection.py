"""Fault signature captures: per-tick spike counts[64] under actuator faults.

Paper 1 protocol: inject a joint failure at t0=2 s while the closed loop runs,
capture per-tick spike counts[64] + plant telemetry, and test whether the
culture's spike stream carries a detectable, channel-localized fault signature.

Failures match the battery (`failure_battery.py`): a zero-torque joint maps to
a growing joint-error on the corresponding electrode channel index (0..11).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ablate_loop import run_mode, TPS

SEEDS = [7, 29, 13]
T_FAULT = 2.0

FAILURES = {
    "healthy": {"label": "healthy", "joint": None, "spec": []},
    "freeze_lknee": {"label": "left_knee", "joint": 3, "spec": [
        {"joints": [3], "kind": "zero", "t0": T_FAULT}]},
    "freeze_rknee": {"label": "right_knee", "joint": 9, "spec": [
        {"joints": [9], "kind": "zero", "t0": T_FAULT}]},
    "freeze_lhip": {"label": "left_hip_roll", "joint": 1, "spec": [
        {"joints": [1], "kind": "zero", "t0": T_FAULT}]},
    "degrade_50": {"label": "left_leg", "joint": 3, "spec": [
        {"joints": [0, 1, 2, 3, 4, 5], "kind": "degrade", "scale": 0.5,
         "t0": T_FAULT}]},
}

RES = Path(__file__).resolve().parent.parent / "results"


def capture(failure: str, seed: int, substrate: str = "rate"):
    spec = FAILURES[failure]
    r = run_mode("neural", seed=seed, task=[(0.0, 0.5)], substrate=substrate,
                 failures=spec["spec"], capture=True)
    counts = np.asarray(r["counts_matrix"], dtype=float)
    x = np.asarray(r["x_matrix"], dtype=float)
    ws = r["walker_series"]
    fallen_t = None
    for i in range(1, len(ws)):
        if ws[i][1] < 0.35:
            fallen_t = ws[i][0]
            break
    return {
        "failure": failure, "joint": spec["joint"], "label": spec["label"],
        "seed": seed, "substrate": substrate,
        "counts": counts, "x": x,
        "fallen_t": fallen_t,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--failures", default=",".join(FAILURES))
    parser.add_argument("--substrate", default="rate")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    fails = [f for f in args.failures.split(",") if f in FAILURES]
    RES.mkdir(exist_ok=True)
    t0 = time.monotonic()
    out = {"seeds": seeds, "failures": fails, "substrate": args.substrate,
           "t_fault": T_FAULT, "tps": TPS, "runs": []}
    for failure in fails:
        for seed in seeds:
            r = capture(failure, seed, args.substrate)
            counts = np.asarray(r["counts"], dtype=np.float32)
            x = np.asarray(r["x"], dtype=np.float32)
            npz = RES / ("fault_%s_s%d.npz" % (failure, seed))
            np.savez(npz, counts=counts, x=x,
                     fallen_t=r["fallen_t"] if r["fallen_t"] is not None else -1.0)
            meta = {k: v for k, v in r.items() if k not in ("counts", "x")}
            meta["npz"] = str(npz.name)
            out["runs"].append(meta)
            print("[%s s=%d] joint=%s fallen_t=%s (%.1fs)" % (
                failure, seed, FAILURES[failure]["joint"],
                r["fallen_t"], time.monotonic() - t0), flush=True)
        with open(RES / "fault_captures.json", "w") as fp:
            json.dump(out, fp, indent=2)
    out["wall_sec"] = round(time.monotonic() - t0, 1)
    with open(RES / "fault_captures.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("saved:", RES / "fault_captures.json")


if __name__ == "__main__":
    main()