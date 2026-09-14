"""Coral/CL1 simulator data source owned by a MuJoCo G1 walker (Deploy12).

Encodes the walker's sensory state into electrode frames (64 channels) at the
SDK's 25 kHz sample rate, advances the physics inside `read()`, and applies
committed stimulations (channels <16 = torque overlay, channel 12..15 attitude,
channel 23 = height, channel 63 = hub context, channel 62 = vx override) to the
environment in `on_stim()`.

This is the "environment under the CL1 contract": the cultural brain on the
device reads frames, emits stimulations, and the loop closes through this data
source.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from cl.sim import (DataSourceBatch, DataSourceSpike, DataSourceStim,
                    SimulatorDataSource, SimulatorDataSourceMetadata)

_SRC = Path(__file__).resolve().parent
SNN_SRC = Path("C:/Proyectos/papers/Icra2027/01_snn/src")
TELEMETRY_PATH = str(Path.home() / "AppData" / "Local" / "Temp" / "opencode"
                     / "bridge_telemetry.json")
sys.path.insert(0, str(SNN_SRC))

CH_FSR, CH_ATT = 12, 3
CH_HEIGHT = 23
CH_VX_OVERRIDE = 62
CH_HUB_CTX = 63
N_CHANNELS = 64

SAMPLE_RATE = 25000
CTRL_HZ = 50
SAMPLES_PER_CTRL = SAMPLE_RATE // CTRL_HZ
STIM_UA_PER_MS = 4.0   # stimulation uA per 1 m/s of commanded velocity
TAU_GAIN = 40.0        # Nm per decoder unit on torque channels (<12)


class G1DataSource(SimulatorDataSource):
    def __init__(self, duration_sec: float = 14.0, cmd_vx: float = 0.8,
                 seed: int = 0, perturb: list | None = None,
                 task: list | None = None):
        self.duration_sec = duration_sec
        self.cmd_vx = float(cmd_vx)
        self.seed = seed
        self.perturbations = perturb or []
        self.task_sched = task or []
        self._pelvis_id = None
        self._perturbed_steps = 0
        self._rng = np.random.default_rng(seed)
        from deploy12 import Deploy12
        self.dep = Deploy12()
        self.dep.reset(cmd=[self.cmd_vx, 0.0, 0.0])
        self.dep.cmd_fn = None
        self.dep.obs_noise_std = 0.0
        self.tau_overlay = np.zeros(12)
        self.cmd_override = None
        self.hub_ctx = 0.5
        self._last_ctrl = 0
        self._physics_steps = 0
        self._global_sample = 0
        self.sensors = self._read_sensors()
        self.fallen = False
        self.phase = 0.0
        self._stream_ts = 0
        self.telemetry = []
        self._next_tr = np.ones(N_CHANNELS) * SAMPLE_RATE  # per-channel next transient frame
        self.read_stats = {"n": 0, "sum": 0.0, "min": float("inf"), "max": 0.0}
        self.samples_read = 0

    @property
    def metadata(self):
        return SimulatorDataSourceMetadata(
            channel_count=N_CHANNELS,
            frames_per_second=SAMPLE_RATE,
            uV_per_sample_unit=0.195,
            duration_frames=None,
            seekable=False,
            realtime_only=True,
            supports_accelerated=False,
        )

    def open(self):
        pass

    def close(self):
        self._save_telemetry()
        self.dep.close()

    def _channel_base(self, i):
        return 21000 + 900 * (i % 5)

    def _task_vx(self, t):
        v = None
        for t0, vx in self.task_sched:
            if t >= t0:
                v = vx
        return v

    def _read_sensors(self):
        d = self.dep.data
        err = self.dep.target - d.qpos[7:]
        roll = np.arctan2(2 * (d.qpos[7] * 0 + 0 * d.qpos[5]), 1)  # placeholder
        q = d.qpos[3:7]
        e = d.qvel[3:6]
        pitch = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]),
                           1 - 2 * (q[1] * q[1] + q[2] * q[2]))
        roll = np.arctan2(2 * (q[2] * q[3] + q[0] * q[1]),
                          1 - 2 * (q[1] * q[1] + q[2] * q[2]))
        att = np.array([roll, pitch, d.qvel[3]])
        h = d.qpos[2]
        s = np.zeros(64, dtype=np.float32)
        s[0:12] = np.tanh(4.0 * err)
        s[12:15] = np.tanh(0.8 * att)
        s[23] = np.tanh(6.0 * (h - 0.72))
        return s

    def _ctrl_step(self):
        self.dep.refresh_cmd()
        self.dep.cmd[0] = (self.cmd_override
                           if self.cmd_override is not None else self.cmd_vx)
        if self._pelvis_id is None:
            self._pelvis_id = self.dep.model.body("pelvis").id
        for _ in range(self.dep.dec):
            if self.dep.data.time > self.duration_sec:
                break
            t = self.dep.data.time
            f = np.zeros(3)
            for pw in self.perturbations:
                if pw["t0"] <= t < pw["t0"] + pw["dur"]:
                    f += np.asarray(pw.get("force", [0.0, 0.0, 0.0]), dtype=float)
            self.dep.data.xfrc_applied[self._pelvis_id, :3] = f
            if np.any(f != 0.0):
                self._perturbed_steps += 1
            self.dep.data.ctrl[:] = self.dep.analog_pd_tau() + self.tau_overlay
            import mujoco
            mujoco.mj_step(self.dep.model, self.dep.data)
            self._physics_steps += 1
            if self.dep.data.qpos[2] < 0.35:
                self.fallen = True
                self.tau_overlay[:] = 0.0
                break
            if self._physics_steps % self.dep.dec == 0:
                self.dep.policy_step()
        self.sensors = self._read_sensors()
        self.telemetry.append((float(self.dep.data.time),
                               float(self.dep.data.qpos[2]),
                               float(self.dep.data.qvel[0]),
                               (self.cmd_override if self.cmd_override is not None
                                else float(self.dep.cmd[0]))))
        if len(self.telemetry) % 25 == 0:
            self._save_telemetry()

    def _save_telemetry(self):
        try:
            import json
            tel = self.telemetry
            st = self.read_stats
            with open(TELEMETRY_PATH, "w") as fp:
                json.dump({
                    "n": len(tel),
                    "t_end": float(tel[-1][0]) if tel else 0.0,
                    "h_last": float(tel[-1][1]) if tel else None,
                    "vx_last": float(tel[-1][2]) if tel else None,
                    "fallen": bool(self.fallen),
                    "cmd_last": float(tel[-1][3]) if tel else None,
                    "series": [[round(x[0], 3), round(x[1], 4), round(x[2], 4)]
                               for x in tel[::25]],
                    "bridge_read": {
                        "n_calls": st["n"],
                        "samples": self.samples_read,
                        "mean_us": round(1e6 * st["sum"] / st["n"], 2) if st["n"] else None,
                        "max_us": round(1e6 * st["max"], 2) if st["max"] else None,
                        "min_us": round(1e6 * st["min"], 2) if st["min"] != float("inf") else None,
                    },
                    "perturbations": self.perturbations,
                    "perturbed_steps": self._perturbed_steps,
                    "task_sched": self.task_sched,
                }, fp)
        except Exception:
            pass

    def _encode(self, frame_count, first_sample):
        n = int(frame_count)
        n_ch = N_CHANNELS
        frames = np.zeros((n, n_ch), dtype=np.int16)
        rng = np.random.default_rng(self.seed + (first_sample // SAMPLE_RATE))
        noise = rng.integers(-60, 60, size=(n, n_ch)).astype(np.int16)
        s_vals = [float(self.sensors[i]) if i < len(self.sensors) else 0.0
                  for i in range(n_ch)]
        if CH_HUB_CTX < n_ch:
            s_vals[CH_HUB_CTX] = self.hub_ctx
        if CH_VX_OVERRIDE < n_ch:
            if self.task_sched:
                s_vals[CH_VX_OVERRIDE] = (self._task_vx(self.dep.data.time)
                                          or 0.0)
            else:
                s_vals[CH_VX_OVERRIDE] = (self.cmd_override
                                          if self.cmd_override is not None
                                          else 0.0)
        for i in range(n_ch):
            s = s_vals[i]
            if abs(s) < 0.1:
                continue
            mag = 1800.0 + 2200.0 * min(abs(s), 1.0)
            interval = 90.0 + 600.0 * (1.0 - min(abs(s), 1.0))
            jitter = rng.uniform(0.7, 1.3)
            next_tr = self._next_tr[i]
            step = 0
            while step < n:
                if first_sample + step < next_tr:
                    step += 1
                    continue
                nb = step
                ne = min(step + 6, n)
                seg = np.arange(ne - nb)
                frames[nb:ne, i] += (mag * np.sin(2 * np.pi * seg / 5.0)).astype(np.int16)
                self._next_tr[i] = next_tr + interval * jitter
                next_tr = self._next_tr[i]
                step += 1
        return frames + noise

    def read(self, from_timestamp: int, frame_count: int):
        t0 = time.monotonic()
        # Always keep stream position equal to SDK's from_timestamp so buffer
        # tracking stays aligned.
        self._stream_ts = int(from_timestamp)
        end = self._stream_ts + int(frame_count)
        # Advance physics across each control boundary (50 Hz).
        ctrl_blocks = end // SAMPLES_PER_CTRL
        if ctrl_blocks > self._last_ctrl:
            for _ in range(ctrl_blocks - self._last_ctrl):
                self._ctrl_step()
                if self.fallen or self.dep.data.time > self.duration_sec:
                    break
            self._last_ctrl = ctrl_blocks
        self._global_sample = end
        dur = time.monotonic() - t0
        st = self.read_stats
        st["n"] += 1
        st["sum"] += dur
        if dur < st["min"]:
            st["min"] = dur
        if dur > st["max"]:
            st["max"] = dur
        self.samples_read += int(frame_count)
        return DataSourceBatch(frames=self._encode(frame_count, self._stream_ts))

    def on_stim(self, stim: DataSourceStim):
        ch = stim.channel
        v = 0.0
        if stim.phase_currents_uA:
            v = float(stim.phase_currents_uA[-1]) / STIM_UA_PER_MS
        elif stim.phase_durations_us:
            v = float(np.sum(stim.phase_durations_us)) / 1e6 / 1e3
        if 0 <= ch < 12:
            self.tau_overlay[ch] = float(np.clip(v * TAU_GAIN, -40, 40))
            self.hub_ctx = 0.5
        elif ch == CH_VX_OVERRIDE:
            self.cmd_override = float(np.clip(v, 0.0, 0.75))
            self.hub_ctx = 0.9
        else:
            self.hub_ctx = min(1.0, self.hub_ctx + 0.1)

    def on_stims(self, stims):
        for s in stims:
            self.on_stim(s)

    def epoch_state(self):
        return {
            "t": float(self.dep.data.time),
            "h": float(self.dep.data.qpos[2]),
            "vx": float(self.dep.data.qvel[0]),
            "fallen": self.fallen,
            "cmd_override": self.cmd_override,
            "strides_after_physics": self._physics_steps,
        }


def create(duration_sec: float = 14.0, cmd_vx: float = 0.8, seed: int = 0,
           perturb: list | None = None, task: list | None = None):
    return G1DataSource(duration_sec=duration_sec, cmd_vx=cmd_vx, seed=seed,
                        perturb=perturb, task=task)