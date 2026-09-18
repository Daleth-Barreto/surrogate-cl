"""Gate B substrate #2: a real recurrent spiking-network culture (Izhikevich).

The canonical loop's "culture" is a per-channel rate/amplitude transducer
(`bridge_g1._encode`) that writes spike-like transients onto the 64 electrode
channels. This module provides a second, genuinely dynamical substrate: each
channel is read out from a recurrent Izhikevich microcircuit (6 regular-spiking
excitatory + 2 fast-spiking inhibitory) driven by the same sensor currents. The
microcircuit fires in volleys; a volley is rendered as one negative-going
transient so the SDK down-crossing spike detector and the decoder stay
identical.

Rates are calibrated (once, cached in results/_izh_calib.json) so that the
expected number of volleys per tick matches the canonical transducer across the
sensor range: events_per_channel == round(n_frames / interval(|s|)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 25000
DT_MS = 1.0 / SAMPLE_RATE * 1e3
VOLT = 30.0               # spike threshold (mV)
REFR_MS = 1.0             # min sample-spacing between rendered volleys
EXC_PER_CH = 6            # RS neurons per channel (drive + trigger transients)


class IzhikevichSubstrate:
    """Izhikevich population: one recurrent microcircuit per channel."""

    def __init__(self, n_ch=64, seed=11):
        self.n_ch = n_ch
        self.seed = seed
        self.calib = _load_or_compute_calib()
        self.exc = np.arange(n_ch * 8) % 8 < EXC_PER_CH
        self.reset(seed=seed)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        total = self.n_ch * 8
        self.n = total
        self.a = np.where(self.exc, 0.02, 0.1)
        self.b = np.where(self.exc, 0.2, 0.25)
        self.c = np.full(total, -65.0)
        self.d = np.full(total, 2.0)
        self.v = self.c + rng.normal(0.0, 2.0, total)
        self.u = self.b * self.v
        self.ch_of = np.repeat(np.arange(self.n_ch), 8)
        self.last_start = np.full(self.n_ch, -1e9)
        self._t = 0

    def _drive(self, s_vals, seed):
        s = np.abs(np.asarray(s_vals, dtype=float))
        s = np.clip(s, 0.0, 1.0)
        rate = np.zeros(self.n_ch)
        active = s >= 0.1
        rate[active] = SAMPLE_RATE / (90.0 + 600.0 * (1.0 - s[active])) / EXC_PER_CH
        I_ch = np.zeros(self.n_ch)
        I_ch[active] = self.calib.I_at(rate[active])
        I = np.repeat(I_ch, 8)
        I[~self.exc] *= 0.15
        rng = np.random.default_rng(seed)
        return I * rng.uniform(0.85, 1.15, size=self.n)

    def frames(self, s_vals, n_frames, rng_seed):
        n = int(n_frames)
        n_ch = self.n_ch
        out = np.zeros((n, n_ch), dtype=np.int16)
        if n <= 0:
            return out
        rng = np.random.default_rng(rng_seed)
        noise = rng.integers(-60, 60, size=(n, n_ch)).astype(np.int16)
        sv = np.clip(np.asarray(s_vals, dtype=float), -1.0, 1.0)
        mag = np.where(np.abs(sv) >= 0.1, 1800.0 + 2200.0 * np.abs(sv), 0.0)
        I = self._drive(sv, rng_seed + 1)
        refr_s = int(round(REFR_MS / 1e3 * SAMPLE_RATE))
        v, u = self.v, self.u
        b, c, d = self.b, self.c, self.d
        a = self.a
        dof = self.ch_of
        trig = self.exc
        last_start = self.last_start
        n_ch_loc = n_ch
        for s in range(n):
            v += 0.5 * DT_MS * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            v += 0.5 * DT_MS * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            u += DT_MS * a * (b * v - u)
            fired = v >= VOLT
            if fired.any():
                v[fired] = c[fired]
                u[fired] = u[fired] + d[fired]
                for g in np.unique(dof[fired & trig]):
                    if (self._t + s) - last_start[g] >= refr_s:
                        last_start[g] = self._t + s
                        ne = min(s + 6, n)
                        seg = np.arange(ne - s)
                        out[s:ne, g] += (mag[g] * np.sin(
                            2 * np.pi * seg / 5.0)).astype(np.int16)
        self.v, self.u = v, u
        self.last_start = last_start
        self._t += n
        return (out + noise).astype(np.int16)


class _Calibration:
    """Maps target burst rate (Hz per channel) -> drive current (arbitrary).

    Built from the steady-state firing rate of a representative RS neuron
    (rate monotonically increases with constant current).
    """

    def __init__(self):
        rng = np.random.default_rng(11)
        Is = np.linspace(4.0, 28.0, 13)
        rates = np.asarray([self._rate(I, rng) for I in Is], dtype=float)
        order = np.argsort(rates)
        keep = np.ones(len(rates), dtype=bool)
        for i in range(1, len(order)):
            if rates[order[i]] <= rates[order[i - 1]]:
                keep[order[i]] = False
        self._I = Is[keep]
        self._R = rates[keep]

    def I_at(self, rate_hz):
        return np.interp(np.clip(rate_hz, self._R[0], self._R[-1]),
                         self._R, self._I)

    def _rate(self, I, rng, n_ms=2500.0):
        v = -65.0 + rng.normal(0.0, 1.0)
        u = 0.2 * v
        dt = DT_MS
        spikes = 0
        for _ in range(int(n_ms / dt)):
            v += 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            v += 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
            u += dt * 0.02 * (0.2 * v - u)
            if v >= VOLT:
                v = -65.0
                u += 2.0
                spikes += 1
        return spikes * (1000.0 / n_ms)


_CALIB = None


def _load_or_compute_calib():
    global _CALIB
    if _CALIB is not None:
        return _CALIB
    p = Path(__file__).resolve().parent.parent / "results" / "_izh_calib.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            c = _Calibration.__new__(_Calibration)
            c._I = np.asarray(d["I"])
            c._R = np.asarray(d["R"])
            _CALIB = c
            return c
        except Exception:
            pass
    c = _Calibration()
    try:
        p.write_text(json.dumps({"I": c._I.tolist(), "R": c._R.tolist()}))
    except Exception:
        pass
    _CALIB = c
    return c