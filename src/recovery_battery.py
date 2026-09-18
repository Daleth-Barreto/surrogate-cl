"""Recovery battery: detector-triggered halt experiment (Paper 1).

Runs each frozen-joint fault twice (recovery ON / OFF) for matched pairs.
When the online total-spike z-score detector fires (2 consecutive ticks
above 3-sigma from the run's own 0-1.5 s baseline), the recovery arm
commands vx->0.02 m/s (soft halt). The OFF arm runs identically but does
not act on the detection.

Results: results/recovery_battery.json with per-seed survival, detection
time, and lead time for each arm and failure mode.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import cl
from cl.sim import SimulatorDataSourceMetadata, set_simulator_data_source

from ablate_loop import TPS, THR, _stim_current
from bridge_g1 import N_CHANNELS, CH_VX_OVERRIDE
from demo_walk import BASE_VX, build_hub, HubThread, TAU_HIP_L, TAU_HIP_R
from fault_detection import FAILURES

RES = Path(__file__).resolve().parent.parent / "results"

DURATION_SEC = 12.0
BASELINE_SEC = 1.5
DETECT_SIGMA = 3.0
CONSEC_THRESHOLD = 2
# The SDK biphasic stim requires a nonzero current, so an exact 0.0 m/s
# command is not reachable through ch62; 0.02 m/s is the effective halt.
HALT_VX = 0.02


def run_one(seed, failure, recover):
    """One closed-loop run with optional detector-triggered halt."""
    spec = FAILURES[failure]
    config = {"duration_sec": DURATION_SEC, "cmd_vx": BASE_VX, "seed": seed,
              "perturb": None, "task": [(0.0, 0.5)], "capture_state": True,
              "scene": None, "record_path": None,
              "substrate": "rate", "failures": spec["spec"]}
    metadata = SimulatorDataSourceMetadata(
        channel_count=N_CHANNELS, frames_per_second=25000,
        uV_per_sample_unit=0.195, duration_frames=None,
        seekable=False, realtime_only=True, supports_accelerated=False)
    set_simulator_data_source("bridge_g1:create", config=config,
                              metadata=metadata)

    holder = {"last_x": np.zeros(5), "lock": __import__("threading").Lock()}
    inp, out, p, sim = build_hub(holder)
    hub = HubThread(inp, out, p, sim).start()

    result = {"fallen_t": None, "det_t": None, "fallen": None,
              "vx_last": None, "wall": None, "mean_nspk": None}
    try:
        prev = np.zeros(N_CHANNELS)
        baseline_totals, streak, det_t, halted = [], 0, None, False
        log_totals = []
        wall0 = time.monotonic()

        with cl.open() as neurons:
            for tick in neurons.loop(TPS,
                                     stop_after_seconds=DURATION_SEC + 0.5):
                frames = (tick.frames.astype(np.float32)
                          if tick.frames is not None
                          else np.zeros((0, N_CHANNELS)))
                below = frames < THR
                prev_on = (prev >= THR)[None, :]
                down_cross = below & np.vstack([prev_on, frames[:-1] >= THR])
                counts = down_cross.sum(axis=0).astype(float)
                prev = frames[-1] if len(frames) else prev
                totals_now = float(counts.sum())
                now_t = tick.iteration / TPS

                if now_t < BASELINE_SEC:
                    baseline_totals.append(totals_now)

                mu_pre = sd_pre = 0.0
                if len(baseline_totals) >= 10:
                    mu_pre = float(np.mean(baseline_totals))
                    sd_pre = float(np.std(baseline_totals)) + 1e-9

                if (not halted and now_t >= BASELINE_SEC and sd_pre > 1e-6):
                    z = (totals_now - mu_pre) / sd_pre
                    if z > DETECT_SIGMA:
                        streak += 1
                    else:
                        streak = 0
                    if streak >= CONSEC_THRESHOLD and det_t is None:
                        det_t = now_t
                        halted = True

                with holder["lock"]:
                    holder["last_x"] = np.array([
                        float(min(30.0, counts[12]) / 8.0),
                        float(min(30.0, counts[13]) / 8.0),
                        float(min(30.0, counts[CH_VX_OVERRIDE]) / 4.0),
                        float(min(30.0, counts.sum()) / 8.0),
                        float(min(5.0, counts[63]) / 2.0),
                    ])
                hub_l = np.asarray(hub.latest).ravel()
                hub_vx = float(max(0.0, min(2.2,
                                            hub_l[0] if len(hub_l) else 0.0)))
                hub_tau = float(hub_l[1] if len(hub_l) > 1 else 0.0)

                vx_cmd = HALT_VX if (halted and recover) else hub_vx
                tau_cmd = hub_tau

                if halted and recover:
                    neurons.stim(CH_VX_OVERRIDE, _stim_current(HALT_VX))
                elif abs(vx_cmd) > 1e-3:
                    neurons.stim(CH_VX_OVERRIDE, _stim_current(vx_cmd))
                if abs(tau_cmd) > 1e-3:
                    neurons.stim(TAU_HIP_L, _stim_current(tau_cmd))
                    neurons.stim(TAU_HIP_R, _stim_current(-tau_cmd))

                log_totals.append(totals_now)

        walk, walk_series = read_telemetry_poll()
        fallen_t = None
        for i in range(1, len(walk_series)):
            if walk_series[i][1] < 0.35:
                fallen_t = walk_series[i][0]
                break
        result = {"fallen_t": fallen_t, "det_t": det_t,
                  "fallen": walk.get("fallen"), "vx_last": walk.get("vx_last"),
                  "wall": round(time.monotonic() - wall0, 1),
                  "mean_nspk": round(float(np.mean(log_totals)), 1)}
    finally:
        hub.stop()
    return result


def read_telemetry_poll(retries=30):
    """Poll the bridge telemetry JSON (written by bridge_g1) until fresh."""
    import json as _json
    from bridge_g1 import TELEMETRY_PATH
    walk = {"fallen": None, "vx_last": None, "h_last": None}
    walk_series = []
    for _ in range(retries):
        try:
            tel = _json.load(open(TELEMETRY_PATH, encoding="utf-8"))
            if tel.get("n"):
                walk = {"fallen": tel["fallen"], "vx_last": tel.get("vx_last"),
                        "h_last": tel.get("h_last")}
                walk_series = tel.get("series", [])
                break
        except Exception:
            pass
        time.sleep(0.5)
    return walk, walk_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,13,29")
    parser.add_argument("--failures",
                        default="freeze_lknee,freeze_rknee,freeze_lhip")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    fails = [f for f in args.failures.split(",") if f in FAILURES]
    RES.mkdir(exist_ok=True)
    t0 = time.monotonic()
    out = {"seeds": seeds, "failures": fails,
           "baseline_sec": BASELINE_SEC, "detect_sigma": DETECT_SIGMA,
           "consec_threshold": CONSEC_THRESHOLD, "halt_vx": HALT_VX,
           "runs": []}

    for failure in fails:
        for seed in seeds:
            for recover in [False, True]:
                r = run_one(seed, failure, recover)
                lead = None
                if r["det_t"] is not None and r["fallen_t"] is not None:
                    lead = round(r["fallen_t"] - r["det_t"], 2)
                entry = {
                    "failure": failure, "joint": FAILURES[failure]["joint"],
                    "seed": seed, "recover": recover,
                    "fallen_t": r["fallen_t"], "det_t": r["det_t"],
                    "lead_t": lead, "fallen": r["fallen"],
                    "mean_nspk": r["mean_nspk"], "wall": r["wall"],
                }
                out["runs"].append(entry)
                print("[recover=%s s=%d %-14s] fallen=%s det_t=%s lead=%s "
                      "nspk=%.1f wall=%.1fs" % (
                          recover, seed, failure, r["fallen"], r["det_t"],
                          lead, r["mean_nspk"], r["wall"]), flush=True)
        runs_fail = [r for r in out["runs"] if r["failure"] == failure]
        off = [r for r in runs_fail if not r["recover"]]
        on = [r for r in runs_fail if r["recover"]]
        surv_off = sum(1 for r in off if not r["fallen"])
        surv_on = sum(1 for r in on if not r["fallen"])
        det_off = sorted([r["det_t"] for r in off if r["det_t"] is not None])
        det_on = sorted([r["det_t"] for r in on if r["det_t"] is not None])
        print("  %s: survival OFF=%d/%d  ON=%d/%d  det_t_off=%s  det_t_on=%s"
              % (failure, surv_off, len(off), surv_on, len(on),
                 [round(t, 2) for t in det_off],
                 [round(t, 2) for t in det_on]), flush=True)

    out["wall_sec"] = round(time.monotonic() - t0, 1)
    with open(RES / "recovery_battery.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("saved:", RES / "recovery_battery.json")


if __name__ == "__main__":
    main()