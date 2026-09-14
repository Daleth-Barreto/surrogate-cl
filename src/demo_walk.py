"""CL1-contract closed loop: G1 walker under a spiking Nengo decision hub.

The MuJoCo walker runs inside the SDK's custom simulator data source
(`bridge_g1`). Its sensory state is coded into electrode "frames". The local
CL1 simulator detects cultural spikes on those frames. A small Nengo LIF hub
(for decoration of a biological substrate) reads spike counts and decides a
forward-velocity command, committed back to the environment as a Myo-electric
stimulation (channel 62 = vx override). This closes a hardware-faithful loop:
sensors -> electrodes -> cultural spikes -> spiking decision -> stims -> motors.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

import nengo

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import bridge_g1
from bridge_g1 import (N_CHANNELS, CH_HEIGHT, CH_VX_OVERRIDE)

import cl
from cl.sim import (SimulatorDataSourceMetadata, set_simulator_data_source)

DURATION_SEC = 12.0
TPS = 40
BASE_VX = 0.5
HUB_DT = 0.002
HUB_HORIZON_STEPS = 600   # 1.2 s of hub time per chunked run (amortises ~100ms call overhead)

TAU_HIP_L = 1   # left_hip_roll actuator id
TAU_HIP_R = 7   # right_hip_roll actuator id
TAU_SIGN = 1.0
TAU_UNITS = 0.75   # decoder-units cap (=> 3 uA stim, => TAU_GAIN*0.75 Nm torque)

T = {"t": 0.0}


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def decide(x):
    roll, pitch, thsig, nspk, _ = x
    gate = _sig(30.0 * (nspk - 0.15))
    slow = 1.0 - 0.9 * _sig(10.0 * (max(abs(roll), abs(pitch)) - 0.9))
    vx = BASE_VX * (1.0 + 1.0 * np.tanh(2.5 * (thsig - 0.15))) * gate * slow
    tau = TAU_UNITS * np.tanh(2.0 * TAU_SIGN * roll)
    return [float(vx), float(tau)]


def build_hub(holder):
    net = nengo.Network(seed=1)
    with net:
        inp = nengo.Node(lambda t: holder["last_x"], size_in=0, size_out=5)
        ensemble = nengo.Ensemble(1000, 5, radius=2.0, neuron_type=nengo.LIF())
        out = nengo.Node(size_in=2)
        p = nengo.Probe(out, "output", synapse=None)
        nengo.Connection(inp, ensemble, synapse=0.03)
        nengo.Connection(ensemble, out, function=decide, synapse=0.05)
    sim = nengo.Simulator(net, dt=HUB_DT)
    return inp, out, p, sim


class HubThread:
    """Steps the Nengo substrate in LARGE chunks (amortising Nengo's per-call
    overhead) while the main loop reads the last decoded command."""

    def __init__(self, inp, out, p, sim):
        self.inp, self.out, self.p, self.sim = inp, out, p, sim
        self.latest = np.array([0.0, 0.0])
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop = True
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.sim.close()

    def _run(self):
        while not self._stop:
            self.sim.run_steps(HUB_HORIZON_STEPS)
            self.latest = np.asarray(self.sim.data[self.p][-1]).ravel()


def main():
    set_simulator_data_source(
        "bridge_g1:create",
        config={"duration_sec": DURATION_SEC, "cmd_vx": BASE_VX},
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

    counts_holder = {"last_x": np.zeros(5), "lock": threading.Lock()}
    inp, out, p, sim = build_hub(counts_holder)
    hub = HubThread(inp, out, p, sim).start()

    log = {"t": [], "cmd": [], "h_ch_counts": [], "tot": [], "nspk": []}
    applied = 0.0
    wall0 = time.time()
    THR = -1400
    prev = np.zeros(N_CHANNELS)
    with cl.open() as neurons:
        for tick in neurons.loop(TPS, stop_after_seconds=DURATION_SEC + 0.5):
            T["t"] = tick.iteration / TPS
            frames = tick.frames.astype(np.float32) if tick.frames is not None \
                else np.zeros((0, N_CHANNELS))
            below = frames < THR
            prev_on = (prev >= THR)[None, :]
            down_cross = below & np.vstack([prev_on, frames[:-1] >= THR])
            counts = down_cross.sum(axis=0).astype(float)
            prev = frames[-1] if len(frames) else prev
            spikes = frames if len(frames) else np.zeros((0, N_CHANNELS))
            nspk = float(counts.sum())
            x = np.array([
                float(min(30.0, counts[12]) / 8.0),
                float(min(30.0, counts[13]) / 8.0),
                0.0,
                float(min(30.0, counts.sum()) / 8.0),
                float(min(5.0, counts[63]) / 2.0),
            ])
            with counts_holder["lock"]:
                counts_holder["last_x"] = x
            cmd = float(max(0.0, min(2.2, hub.latest[0])))
            if abs(cmd - applied) > 0.1:
                applied = cmd
                if applied > 1e-3:
                    neurons.stim(CH_VX_OVERRIDE, max(applied, 1e-3)
                                 * bridge_g1.STIM_UA_PER_MS)
            log["t"].append(T["t"])
            log["cmd"].append(cmd)
            log["h_ch_counts"].append(float(counts[23]))
            log["tot"].append(float(counts.sum()))
            log["nspk"].append(nspk)

    wall = time.time() - wall0
    hub.stop()

    hch = np.array(log["h_ch_counts"])
    mean_cmd = float(np.mean(log["cmd"]))
    n_stim = sum(1 for i in range(1, len(log["cmd"]))
                 if abs(log["cmd"][i] - log["cmd"][i - 1]) > 0.01)
    walk = {}
    walk_series = []
    for _ in range(10):
        try:
            import json as _json
            tel = _json.load(open(bridge_g1.TELEMETRY_PATH, encoding="utf-8"))
            walk = {"t_end": tel["t_end"], "h_last": tel["h_last"],
                    "vx_last": tel["vx_last"], "fallen": tel["fallen"]}
            walk_series = tel.get("series", [])
            if tel.get("n") or tel.get("fallen"):
                break
        except Exception:
            pass
        time.sleep(0.5)
    print("ticks=%d wall=%.1fs cmd=%.2f stims=%d mean_nspk=%.1f hch_last=%.1f tel=%s"
          % (len(log["t"]), wall, mean_cmd, n_stim,
             float(np.mean(log["nspk"])), float(hch[-1]), walk))

    out = {
        "ticks": len(log["t"]),
        "duration_sec": DURATION_SEC,
        "wall_time": round(wall, 2),
        "loop_tps": TPS,
        "mean_cmd_vx": round(mean_cmd, 3),
        "stim_events": n_stim,
        "mean_spikes_per_tick": round(float(np.mean(log["nspk"])), 2),
        "height_channel_decay": round(float(hch[-1]) - float(hch[0]), 3),
        "cmd_series": [round(v, 2) for v in log["cmd"]],
        "height_channel_series": [round(v, 0) for v in hch],
        "walker": {
            "t_end": walk.get("t_end"),
            "h_last": walk.get("h_last"),
            "vx_last": walk.get("vx_last"),
            "fallen": walk.get("fallen"),
        },
    }
    RES = BASE.parent / "results"
    RES.mkdir(exist_ok=True)
    import json
    with open(RES / "f2_demo.json", "w") as f:
        json.dump(out, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if walk_series:
        wt = [s[0] for s in walk_series]
        wh = [s[1] for s in walk_series]
        wv = [s[2] for s in walk_series]
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(log["t"], log["cmd"]); ax[0].set_ylabel("hub cmd vx")
        ax[1].plot(wt, wh); ax[1].set_ylabel("walker height (m)")
        ax[2].plot(wt, wv); ax[2].set_xlabel("t (s)")
        ax[2].set_ylabel("vx (m/s)")
    else:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(log["t"], log["cmd"]); ax[0].set_ylabel("cmd vx")
        ax[1].plot(log["t"], hch); ax[1].set_xlabel("t (s)")
        ax[1].set_ylabel("height-channel counts")
    fig.tight_layout()
    fig.savefig(RES / "f2_demo.png", dpi=140)
    print("saved:", RES / "f2_demo.json", RES / "f2_demo.png")


if __name__ == "__main__":
    main()