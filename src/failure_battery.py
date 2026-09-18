"""Paper 1 battery: substrate-mediated failure recovery on the G1 humanoid.

Injects actuator failures into the closed loop and asks whether the
substrate-mediated decode (neural hub) survives hardware loss better than
the dead/nulled controls.

Failure types (actuator indices per g1_12dof.xml):
  healthy      - no failure (control)
  freeze_lknee - left knee torque zeroed (idx 3)
  freeze_rknee - right knee torque zeroed (idx 9)
  freeze_lhip  - left hip-roll torque zeroed (idx 1)
  degrade_50   - left leg joints (0..5) at 50% torque
  degrade_25   - left leg joints (0..5) at 25% torque
  jitter_leg   - left leg joints torque x (1 + N(0, 0.2)) per step

Substrates/modes:
  neural/rate  - canonical LIF hub on rate-code culture
  neural/izh   - LIF hub on Izhikevich culture
  neural/mlp   - LIF hub on MLP culture
  poisson/rate - dead Poisson substrate (null)
  zero/rate    - no command (null)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ablate_loop import run_mode, TPS, DURATION_SEC

BASE_VX = 0.5

SEEDS = [1, 7, 13, 29]
DEFAULT_PROFILE = [(0.0, BASE_VX)]

FAILURES = {
    "healthy": [],
    "freeze_lknee": [{"joints": [3], "kind": "zero", "t0": 2.0}],
    "freeze_rknee": [{"joints": [9], "kind": "zero", "t0": 2.0}],
    "freeze_lhip": [{"joints": [1], "kind": "zero", "t0": 2.0}],
    "degrade_50": [{"joints": [0, 1, 2, 3, 4, 5], "kind": "degrade",
                    "scale": 0.5, "t0": 2.0}],
    "degrade_25": [{"joints": [0, 1, 2, 3, 4, 5], "kind": "degrade",
                    "scale": 0.25, "t0": 2.0}],
    "jitter_leg": [{"joints": [0, 1, 2, 3, 4, 5], "kind": "jitter",
                    "sigma": 0.2, "t0": 2.0}],
}

CONDITIONS = {
    "neural_rate": {"mode": "neural", "substrate": "rate"},
    "neural_izh": {"mode": "neural", "substrate": "izh"},
    "neural_mlp": {"mode": "neural", "substrate": "mlp"},
    "poisson": {"mode": "poisson", "substrate": "rate"},
    "zero": {"mode": "zero", "substrate": "rate"},
}

RES = Path(__file__).resolve().parent.parent / "results"


def desired(sched, t):
    return BASE_VX


def one_run(cond: str, failure: str, seed: int, max_tries: int = 4):
    spec = CONDITIONS[cond]
    failures = FAILURES[failure]
    r = None
    for attempt in range(max_tries):
        try:
            r = run_mode(spec["mode"], seed=seed, task=[(0.0, BASE_VX)],
                         substrate=spec["substrate"], failures=failures)
            break
        except TimeoutError as e:
            print("  [retry %d/%d] %s/%s s=%d %s" % (
                attempt + 1, max_tries, failure, cond, seed, e), flush=True)
    if r is None:
        raise RuntimeError("run failed after %d tries: %s/%s s=%d" % (
            max_tries, failure, cond, seed))
    cmd = np.asarray(r["cmd_series"], dtype=float)
    ts = (np.arange(len(cmd)) + 0.5) / TPS
    des = np.full_like(cmd, BASE_VX)
    rmse = float(np.sqrt(np.mean((cmd - des) ** 2)))
    rmses = {}
    for w in ((0.0, 2.0), (2.0, 6.0), (6.0, None)):
        lo, hi = w
        m = (ts >= lo) & (ts < (hi if hi is not None else ts[-1] + 1e-9))
        rmses["%.0f-%.0f" % w if hi else "%.0f-end" % lo] = (
            float(np.sqrt(np.mean((cmd[m] - des[m]) ** 2))) if m.any() else None)
    ws = r["walker_series"]
    dist = 0.0
    for i in range(1, len(ws)):
        dist += (ws[i][0] - ws[i - 1][0]) * max(0.0, ws[i - 1][2])
    return {
        "cond": cond, "failure": failure, "seed": seed,
        "rmse": round(rmse, 4),
        "rmse_windows": {k: (round(float(v), 4) if v is not None else None)
                         for k, v in rmses.items()},
        "fallen": bool(r["walker"].get("fallen")),
        "t_end": float(r["walker"].get("t_end") or 0.0),
        "h_last": float(r["walker"].get("h_last") or 0.0),
        "vx_last": float(r["walker"].get("vx_last") or 0.0),
        "distance_m": round(dist, 2),
        "mean_cmd": float(r["mean_cmd"]),
        "cmd_series": [round(float(v), 3) for v in r["cmd_series"]],
    }


def aggregate(runs, cond, failure):
    rmses = np.array([r["rmse"] for r in runs])
    dists = np.array([r["distance_m"] for r in runs])
    fallen = [r["fallen"] for r in runs]
    return {
        "cond": cond, "failure": failure, "n": len(runs),
        "survival": round(float(np.mean([not f for f in fallen])), 3),
        "survivors": int(sum(1 for f in fallen if not f)),
        "total": len(fallen),
        "fallen_seeds": [int(r["seed"]) for r in runs if r["fallen"]],
        "rmse_mean": round(float(rmses.mean()), 3),
        "rmse_std": round(float(rmses.std(ddof=1)), 3),
        "distance_mean": round(float(dists.mean()), 2),
        "distance_std": round(float(dists.std(ddof=1)), 2),
        "distance_seeds": [round(float(v), 2) for v in dists],
        "h_last_mean": round(float(np.mean([r["h_last"] for r in runs])), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--failures", default=",".join(FAILURES))
    parser.add_argument("--resume", action="store_true",
                        help="skip (failure, condition) cells already present "
                             "with n == len(seeds) in failure_battery.json")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    conds = [c for c in args.conditions.split(",") if c in CONDITIONS]
    fails = [f for f in args.failures.split(",") if f in FAILURES]
    RES.mkdir(exist_ok=True)
    t0 = time.monotonic()
    out = {"seeds": seeds, "conditions": conds, "failures": fails,
           "duration_sec": DURATION_SEC, "base_vx": BASE_VX,
           "run": {}}
    if args.resume and (RES / "failure_battery.json").exists():
        try:
            prev = json.loads((RES / "failure_battery.json").read_text())
            for f, conds_map in prev.get("run", {}).items():
                for c, agg in conds_map.items():
                    if agg.get("n") == len(seeds):
                        out["run"].setdefault(f, {})[c] = agg
        except Exception:
            pass
    n_runs = 0
    for failure in fails:
        for cond in conds:
            run_list = []
            if args.resume and failure in out["run"] \
                    and cond in out["run"][failure] \
                    and out["run"][failure][cond].get("n") == len(seeds):
                print("[resume] skipping %s/%s (done)" % (failure, cond))
                continue
            for seed in seeds:
                r = one_run(cond, failure, seed)
                run_list.append(r)
                n_runs += 1
                print("[%s/%s s=%d] cond=%-12s fallen=%s rmse=%.4f h=%.3f "
                      "dist=%.2f (%.1fs elapsed)" % (
                          failure, cond, seed, cond, r["fallen"], r["rmse"],
                          r["h_last"], r["distance_m"],
                          time.monotonic() - t0))
            out["run"].setdefault(failure, {})[cond] = aggregate(
                run_list, cond, failure)
            out["run"][failure][cond]["seeds_data"] = [
                {k: v for k, v in r.items() if k != "cmd_series"}
                for r in run_list]
            print("  AGG %-14s surv=%d/%-4d dist=%.2f±%.2f rmse=%.3f±%.3f" % (
                cond, out["run"][failure][cond]["survivors"],
                out["run"][failure][cond]["total"],
                out["run"][failure][cond]["distance_mean"],
                out["run"][failure][cond]["distance_std"],
                out["run"][failure][cond]["rmse_mean"],
                out["run"][failure][cond]["rmse_std"]))
            out["total_runs"] = n_runs
            out["wall_sec"] = round(time.monotonic() - t0, 1)
            with open(RES / "failure_battery.json", "w") as fp:
                json.dump(out, fp, indent=2)
    out["total_runs"] = n_runs
    out["wall_sec"] = round(time.monotonic() - t0, 1)
    with open(RES / "failure_battery.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("saved: %s (%d runs, %.1f s wall)" % (
        RES / "failure_battery.json", n_runs, out["wall_sec"]))
    return out


if __name__ == "__main__":
    main()