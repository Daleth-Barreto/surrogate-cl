"""Substrate #4: a feedforward MLP culture (non-spiking baseline).

Drop-in replacement for the rate-code transducer and the Izhikevich
microcircuit.  A small multi-layer perceptron maps the 64-channel sensor
values to per-channel activation magnitudes.  The magnitudes are then
rendered as the same negative-going sine transients the SDK spike detector
expects, so everything downstream (spike detector, hub, decoder) stays
identical.

The MLP is randomly initialised (no training) — it serves as a structured
but non-biological substrate to answer the question "does spiking dynamics
matter, or would any non-linear map suffice?"  Because the weights are
fixed per seed the substrate is deterministic given (s_vals, rng_seed).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 25000
HIDDEN = 128
REFR_MS = 1.0
OUT_GAIN = 2.0
ACTIVE_THRESH = 0.5
INTERVAL_SCALE = 5.0
_INIT_NEXT_TR = float(SAMPLE_RATE)


class MLPSubstrate:
    """Feedforward MLP: 64 -> HIDDEN -> 64, tanh output in [-1, 1].

    The network is randomly initialised (no training): it is a structured but
    non-biological substrate used to ask whether spiking dynamics matter, or
    whether any static nonlinear map over the same sensor->electrode contract
    suffices. Magic constants (OUT_GAIN, threshold, interval, magnitude) are
    calibrated so the delivered per-tick spike counts are comparable with the
    rate-code substrate (measured mean_nspk ~8-9/tick).
    """

    def __init__(self, n_ch: int = 64, seed: int = 42):
        self.n_ch = n_ch
        self.seed = seed
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / n_ch)
        scale2 = np.sqrt(2.0 / HIDDEN)
        self.W1 = rng.normal(0.0, scale1, (n_ch, HIDDEN)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = rng.normal(0.0, scale2, (HIDDEN, n_ch)).astype(np.float32)
        self.b2 = np.zeros(n_ch, dtype=np.float32)
        self._next_tr = np.full(n_ch, _INIT_NEXT_TR)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        scale1 = np.sqrt(2.0 / self.n_ch)
        scale2 = np.sqrt(2.0 / HIDDEN)
        self.W1 = rng.normal(0.0, scale1, (self.n_ch, HIDDEN)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = rng.normal(0.0, scale2, (HIDDEN, self.n_ch)).astype(np.float32)
        self.b2 = np.zeros(self.n_ch, dtype=np.float32)
        self._next_tr = np.full(self.n_ch, _INIT_NEXT_TR)

    def _forward(self, s_vals):
        x = np.asarray(s_vals, dtype=np.float32)
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        return out * OUT_GAIN

    def frames(self, s_vals, n_frames, rng_seed, first_sample=0):
        n = int(n_frames)
        n_ch = self.n_ch
        out = np.zeros((n, n_ch), dtype=np.int16)
        if n <= 0:
            return out
        rng = np.random.default_rng(rng_seed)
        noise = rng.integers(-60, 60, size=(n, n_ch)).astype(np.int16)
        activations = self._forward(s_vals)
        mag = (1800.0 + 2200.0 * np.abs(activations)).astype(np.float32)
        active = np.abs(activations) >= ACTIVE_THRESH
        refr_s = int(round(REFR_MS / 1e3 * SAMPLE_RATE))
        for i in range(n_ch):
            if not active[i]:
                continue
            interval = INTERVAL_SCALE * (90.0 + 600.0 * (1.0 - abs(activations[i])))
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
                out[nb:ne, i] += (mag[i] * np.sin(
                    2 * np.pi * seg / 5.0)).astype(np.int16)
                self._next_tr[i] = next_tr + interval * jitter
                next_tr = self._next_tr[i]
                step += 1
        return (out + noise).astype(np.int16)
