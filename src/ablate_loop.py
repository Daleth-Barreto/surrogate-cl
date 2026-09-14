"""M3/M4: closed-loop ablations + latency report for the F2 CL-contract demo.

Replicates the doom-neuron ablation protocol (SeanCole02/doom-neuron) on our
spiking CL-contract loop: the cultural substrate provides the spike signal; the
decoder turns it into the forward-velocity command.

Modes:
  neural  - real Nengo hub decode (baseline closed loop).
  zero    - decoder ablated to 0: no stims, no cultural influence on vx command.
  random  - decoder replaced by a random walk in [0, 2*BASE_VX] decoupled from
            the spike counts: tests whether decoded spike signal carries control.
  mask0.5 - lesion: 50% of spike channels silenced before the hub decodes
            (proxy for damaged culture / noisy readout).

Latency (M4) is reported per mode:
  - loop: SDK tick cadence (histogram + p50/p95/p99/max, overruns > 25 ms).
  - bridge: read() duration measured inside bridge_g1 (telemetry bridge_read).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cl
from cl.sim import (SimulatorDataSourceMetadata, set_simulator_data_source)

import bridge_g1
from bridge_g1 import (N_CHANNELS, CH_HEIGHT, CH_VX_OVERRIDE, TELEMETRY_PATH,
                       STIM_UA_PER_MS)
from demo_walk import BASE_VX, TPS, build_hub, HubThread, TAU_HIP_L, TAU_HIP_R

DURATION_SEC = 12.0
THR = -1400
OVERRUN_US = 25_000  # 1/TPS in microseconds
TAU_DEADBAND = 0.05
VX_DEADBAND = 0.1

MODES = ["neural", "zero", "random", "mask0.5"]
MASK_FRAC = 0.5


def _stim_current(value):
    return float(max(-0.75, min(0.75, value))) * STIM_UA_PER_MS


def read_telemetry(retries: int = 30):
    walk = {"t_end": None, "h_last": None, "vx_last": None, "fallen": None,
            "bridge_read": {}, "cmd_last": None}
    walk_series = []
    for _ in range(retries):
        try:
            tel = json.load(open(TELEMETRY_PATH, encoding="utf-8"))
            if tel.get("n"):
                walk = {"t_end": tel["t_end"], "h_last": tel["h_last"],
                        "vx_last": tel["vx_last"], "fallen": tel["fallen"],
                        "bridge_read": tel.get("bridge_read", {}),
                        "cmd_last": tel.get("cmd_last")}
                walk_series = tel.get("series", [])
                break
        except Exception:
            pass
        time.sleep(0.5)
    return walk, walk_series


def run_mode(mode: str, seed: int = 7, perturb: list | None = None,
             task: list | None = None):
    step0 = time.monotonic()
    set_simulator_data_source(
        "bridge_g1:create",
        config={"duration_sec": DURATION_SEC, "cmd_vx": BASE_VX, "seed": seed,
                "perturb": perturb, "task": task},
        metadata=SimulatorDataSourceMetadata(
            channel_count=N_CHANNELS,
            frames_per_second=25000,
            uV_per_sample_unit=0.195,
            duration_frames=None,
            seekable=False,
            realtime_only=True,
            supports_accelerated=False,
        ),
    )

    holder = {"last_x": np.zeros(5), "lock": __import__("threading").Lock()}
    inp, out, p, sim = build_hub(holder)
    hub = HubThread(inp, out, p, sim).start()

    rng = np.random.default_rng(seed)
    if MASK_FRAC and mode.startswith("mask"):
        lesion_ch = set(rng.choice(N_CHANNELS, int(N_CHANNELS * MASK_FRAC),
                                   replace=False).tolist())
    else:
        lesion_ch = set()

    log_t, log_cmd, log_nspk, log_h, log_tau, log_t62 = [], [], [], [], [], []
    prev = np.zeros(N_CHANNELS)
    applied_vx = 0.0
    applied_tau = 0.0
    last_iter_wall = None
    intervals_us = []
    wall0 = time.monotonic()
    with cl.open() as neurons:
        for tick in neurons.loop(TPS, stop_after_seconds=DURATION_SEC + 0.5):
            now = time.monotonic()
            if last_iter_wall is not None:
                intervals_us.append(1e6 * (now - last_iter_wall))
            last_iter_wall = now

            frames = tick.frames.astype(np.float32) if tick.frames is not None \
                else np.zeros((0, N_CHANNELS))
            below = frames < THR
            prev_on = (prev >= THR)[None, :]
            down_cross = below & np.vstack([prev_on, frames[:-1] >= THR])
            counts = down_cross.sum(axis=0).astype(float)
            prev = frames[-1] if len(frames) else prev

            if lesion_ch:
                counts[list(lesion_ch)] = 0.0

            nspk = float(counts.sum())
            x = np.array([
                float(min(30.0, counts[12]) / 8.0),
                float(min(30.0, counts[13]) / 8.0),
                float(min(30.0, counts[CH_VX_OVERRIDE]) / 4.0),
                float(min(30.0, counts.sum()) / 8.0),
                float(min(5.0, counts[63]) / 2.0),
            ])
            with holder["lock"]:
                holder["last_x"] = x
            hub_l = np.asarray(hub.latest).ravel()
            hub_vx = float(max(0.0, min(2.2, hub_l[0] if len(hub_l) else 0.0)))
            hub_tau = float(hub_l[1] if len(hub_l) > 1 else 0.0)

            if mode in ("neural",) or mode.startswith("mask"):
                vx_cmd, tau_cmd = hub_vx, hub_tau
            elif mode == "zero":
                vx_cmd, tau_cmd = 0.0, 0.0
            elif mode == "random":
                vx_cmd = float(rng.uniform(0.0, 0.75))
                tau_cmd = float(rng.uniform(-0.75, 0.75))
            else:
                vx_cmd, tau_cmd = hub_vx, hub_tau

            if abs(vx_cmd - applied_vx) > VX_DEADBAND:
                applied_vx = vx_cmd
                if applied_vx > 1e-3:
                    neurons.stim(CH_VX_OVERRIDE, _stim_current(applied_vx))
            if abs(tau_cmd - applied_tau) > TAU_DEADBAND:
                applied_tau = tau_cmd
                if abs(applied_tau) > 1e-3:
                    neurons.stim(TAU_HIP_L, _stim_current(applied_tau))
                    neurons.stim(TAU_HIP_R, _stim_current(-applied_tau))
            log_t.append(tick.iteration / TPS)
            log_cmd.append(vx_cmd)
            log_tau.append(applied_tau)
            log_nspk.append(nspk)
            log_h.append(float(counts[CH_HEIGHT]))
            log_t62.append(float(counts[CH_VX_OVERRIDE]))

    wall = time.monotonic() - wall0
    hub.stop()

    walk, walk_series = read_telemetry()
    iv = np.asarray(intervals_us, dtype=float)
    pct = np.percentile(iv, [50, 95, 99]) if iv.size else [None] * 3
    overruns = int(np.sum(iv > OVERRUN_US)) if iv.size else None
    stims = sum(1 for i in range(1, len(log_cmd))
                if abs(log_cmd[i] - log_cmd[i - 1]) > 0.01
                or abs(log_tau[i] - log_tau[i - 1]) > TAU_DEADBAND)

    res = {
        "mode": mode,
        "ticks": len(log_t),
        "duration_sec": DURATION_SEC,
        "wall_time": round(wall, 2),
        "mean_cmd": round(float(np.mean(log_cmd)), 3),
        "mean_tau": round(float(np.mean(np.abs(log_tau))), 3),
        "stim_events": stims,
        "mean_nspk": round(float(np.mean(log_nspk)), 2),
        "walker": walk,
        "latency_loop_us": {
            "mean": round(float(iv.mean()), 1) if iv.size else None,
            "p50": round(float(pct[0]), 1),
            "p95": round(float(pct[1]), 1),
            "p99": round(float(pct[2]), 1),
            "max": round(float(iv.max()), 1) if iv.size else None,
            "overruns_gt_25ms": overruns,
            "n": int(iv.size),
        },
        "bridge_read": walk.get("bridge_read", {}),
    }
    # short series for figures (keep cmd at full res, downsample the rest)
    res["cmd_series"] = [round(v, 3) for v in log_cmd]
    res["tau_series"] = [round(v, 3) for v in log_tau]
    res["ch62_series"] = [round(v, 2) for v in log_t62]
    res["height_channel_series"] = [round(v, 0) for v in log_h]
    res["walker_series"] = [[round(s[0], 3), round(s[1], 4), round(s[2], 4)]
                            for s in walk_series]
    res["_wall_step"] = round(time.monotonic() - step0, 1)
    res["_latency_hist"] = [round(float(v), 0) for v in intervals_us]
    return res


def figure(results):
    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    for r, ax in zip(results, axes):
        ax[0].plot(r["cmd_series"], lw=0.8)
        ax[0].set_ylabel("cmd vx")
        ax[0].set_title("%s: fallen=%s vx_last=%.2f" % (
            r["mode"], r["walker"]["fallen"], r["walker"].get("vx_last") or 0.0))
        ws = r["walker_series"]
        if ws:
            wh = [s[1] for s in ws]
            wv = [s[2] for s in ws]
            ax[1].plot(wh, lw=0.8)
            ax[2].plot(wv, lw=0.8)
        ax[1].set_ylabel("height (m)")
        ax[2].set_ylabel("vx (m/s)")
        ax[2].set_xlabel("tick / sampled ctrl")
    fig.tight_layout()
    return fig


def latency_figure(results, intervals_all):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for r, axx in zip(results, ax.ravel()):
        hist = intervals_all.get(r["mode"], [])
        axx.hist(hist, bins=60, log=True)
        axx.axvline(OVERRUN_US, color="r", ls="--",
                    label="deadline 25 ms")
        axx.set_title("%s: p50=%s p95=%s p99=%s max=%s over=%s" % (
            r["mode"], r["latency_loop_us"]["p50"], r["latency_loop_us"]["p95"],
            r["latency_loop_us"]["p99"], r["latency_loop_us"]["max"],
            r["latency_loop_us"]["overruns_gt_25ms"]))
        axx.set_yscale("log")
        axx.legend()
    fig.suptitle("SDK tick cadence (loop latency, us)")
    fig.tight_layout()
    return fig


def main():
    results = []
    intervals_all = {}
    RES = Path(__file__).resolve().parent.parent / "results"
    RES.mkdir(exist_ok=True)
    for mode in MODES:
        r = run_mode(mode)
        results.append(r)
        intervals_all[mode] = r.pop("_latency_hist", [])
        print("mode=%-8s ticks=%d wall=%.1fs cmd=%.2f stims=%d nspk=%.1f "
              "fallen=%s vx_last=%.2f loop_p95=%.0fus bridge_read_n=%s" % (
                  mode, r["ticks"], r["_wall_step"], r["mean_cmd"],
                  r["stim_events"], r["mean_nspk"], r["walker"]["fallen"],
                  r["walker"].get("vx_last"), r["latency_loop_us"]["p95"],
                  r["bridge_read"].get("n_calls")))

    out = {"modes": results,
           "overrun_threshold_us": OVERRUN_US,
           "notes": "random/zero = doom-neuron-style decoder ablations; "
                    "mask0.5 = 50%% channel lesion of spike counts."}
    with open(RES / "f2_ablation.json", "w") as fp:
        json.dump(out, fp, indent=2)

    fig = figure(results)
    fig.savefig(RES / "f2_ablation.png", dpi=140)
    fig2 = latency_figure(results, intervals_all)
    fig2.savefig(RES / "f2_ablation_latency.png", dpi=140)
    print("saved:", RES / "f2_ablation.json", RES / "f2_ablation.png",
          RES / "f2_ablation_latency.png")


if __name__ == "__main__":
    main()