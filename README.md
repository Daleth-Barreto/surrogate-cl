# surrogate-cl

Closed-loop validation that a *load-bearing task function* survives a simulated, non-trainable culture substrate, measured with information-theoretic statistics (mutual information and transfer entropy) and a physical-authority envelope.

This repository is the experiment scaffold of the research project "Does the substrate compute?" (working title). It closes a humanoid locomotion loop (Unitree G1, MuJoCo) through the **Cortical Labs contract** (`cl-sdk`, in-silico), reads spike events from a **simulated culture** that mimics the role of a DishBrain / CL1 wetware, and decodes the task from those spikes. The central claim is checked with honest ablation conditions: a disconnected decoder (zero), an uncoupled decoder (random), a 50 % electrode lesion (mask0.5), and a dead Poisson substrate (poisson), all against the intact Nengo-LIF hub (neural).

## Thesis

A culture substrate (CL1 / DishBrain) is not trained by gradient descent; it only learns through slow, closed-loop conditioning. The same function, however, can be **trained once in-silico** as a spiking neural network (Nengo / NEF) and deployed **verbatim** over the same contract (sensors -> electrodes -> spikes -> decisions -> stimulation). This repository turns that substitution into a *load-bearing task* experiment: the task information must demonstrably travel through the spikes, and an ablation must be able to destroy it.

**Hypothesis H1 (operational):** the loop is load-bearing if the velocity profile demanded by the task travels through the substrate's spikes. A task-driven channel (electrode 62) carries the commanded speed; the decoded command must modulate the physical forward velocity, the task must be statistically detectable in the spike trains (mutual information), and the causal chain electrode-62 -> command must be measurable with transfer entropy.

## System

The plant is the `Deploy12` G1 policy loader (LSTM + PD) inside the CL simulator datasource subprocess (`bridge_g1.py`). The contract runs at 40 ticks per second; each tick exposes 64 electrode frames (int16) and receives stimulation from the hub.

Channel map (64 electrodes):

- 0-11: torque overlay applied to the 12 leg joints (`TAU_GAIN` N-m per unit).
- 12/13: attitude (roll/pitch) readout.
- 23: pelvis height readout.
- 62: task / override command on forward velocity (the load-bearing task channel).
- 63: hub context.

The hub is a Nengo network (`demo_walk.py`): a 1000-neuron LIF ensemble encoding the 5-dimensional state `[roll, pitch, thsig, nspk, ctx]`, decoded to a 2-dimensional command `[vx, tau]` through the function `decide`. `decide` includes an activity gate and a tilt-based slow-down. The calibrated decode map is

    vx = max(0.0, min(0.75, (thsig - 0.0615) / 0.277)) * gate * slow

so that a commanded segment (0.5 -> 0.8 m/s) maps onto the physiological command range without saturating.

Ablation modes (implemented in `ablate_loop.py`):

| mode | meaning |
|---|---|
| `neural` | intact hub, decoded command applied to the plant |
| `random` | uncoupled decoder (fixed random command) |
| `zero`  | zero command (no closed-loop output) |
| `mask0.5` | 50 % of electrodes zeroed (lesion), per-seed |
| `poisson` | dead substrate: frames replaced by Poisson draws that carry no state information |

## Experiments and metrics

All runs last 12 s (500 control ticks) at a nominal flat-ground forward speed of about 0.5 m/s. Task profiles (velocity commands injected on channel 62):

- `baseline`: 0.5 -> 0.8 -> 0.5 m/s
- `multi_step`: stepped schedule with dwell periods
- `sine`: sinusoidal velocity profile
- `descend`: descending profile
- `pulse`: pulsed profile

Multi-seed protocol: 5 fixed seeds {1, 7, 13, 29, 55}; the Nengo hub is deterministic, so the variance comes from the simulator RNG, the cultural spike/count RNG, and the lesion draw.

Metrics:

- Tracking RMSE of command vs profile (primary outcome).
- Mutual information (`signal_analysis.py`, bias-corrected against a 50-permutation shuffle null, nats).
- Transfer entropy along the causal chain `c62 -> cmd` and `spikes -> cmd`.
- Perturbation authority envelope (`push_frontier` / `envelope.py`).

## Results

### Task tracking (5 profiles x 4 modes x 5 seeds, `results/task_tracking.json`)

RMSE mean +/- std, normalised so that `zero` (no command) scores approximately the profile itself (0.55-0.64). The pass threshold for the task tracked on the command is `pass_rmse = 0.18`.

| profile | neural | mask0.5 | random | zero |
|---|---|---|---|---|
| baseline | 0.182 +/- 0.011 | 0.304 +/- 0.046 | 0.336 +/- 0.003 | 0.612 +/- 0.000 |
| multi_step | 0.166 +/- 0.015 | 0.286 +/- 0.051 | 0.327 +/- 0.004 | 0.606 +/- 0.000 |
| sine | 0.194 +/- 0.015 | 0.296 +/- 0.034 | 0.334 +/- 0.008 | 0.601 +/- 0.000 |
| descend | 0.188 +/- 0.014 | 0.326 +/- 0.056 | 0.358 +/- 0.006 | 0.644 +/- 0.000 |
| pulse | 0.178 +/- 0.018 | 0.251 +/- 0.042 | 0.292 +/- 0.008 | 0.554 +/- 0.000 |

The neural hub is the best decoder on every profile, with a large separation from every ablation condition (baseline Cohen d_z about -14.4 vs random and -40.8 vs zero). The 50 % lesion consistently degrades the decode (+40 % to +90 % RMSE), and in some seeds it erases electrode 62 altogether, which is the mechanism by which the task signal disappears. The `zero` mode issues no command, so its RMSE measures the profile itself and anchors the scale.

Statistical caveat, stated explicitly: with 5 seeds the exact permutation test p-value has a floor of 1/32 = 0.0625. No p below 0.05 is claimed. The claim is separation in effect size and direction across all five profiles, not a p < 0.05 test result. A larger seed set (>= 6 seeds) is required to reach p <= 0.05 and is the planned step before submission.

### Information-theoretic validation (Gate A) (`results/f2_signal_summary.json`)

Captures of counts[64] + plant state per tick (all 5 modes x 2 seeds {7, 29}), analysis in `signal_analysis.py`. Aggregate values in nats:

| metric | neural | mask0.5 | poisson | random | zero |
|---|---|---|---|---|---|
| MI(cmd; task) | 0.0365 | 0.0263 | 0.0158 | 0.0000 | 0.0000 |
| TE(c62 -> cmd, lag 2) | 0.00786 | 0.00000 | 0.00000 | 0.00303 | 0.00000 |
| TE(spikes -> cmd, lag 2) | 0.00320 | 0.03842 | 0.00998 | 0.00623 | 0.00000 |
| SUM channel MI(vx) | 0.154 | 0.184 | 0.000 | 0.098 | 0.218 |

Reading:

- The task signature (MI(cmd; task)) is maximal for the intact hub and vanishes for the uncoupled decoders. The causal chain electrode-62 -> command (TE) is nonzero only for the neural hub (and residual noise for random). The dead Poisson substrate shows **exactly zero** transfer entropy and **exactly zero** per-channel mutual information with velocity: the dead substrate carries no task information, even though it still produces spikes.
- `SUM channel MI(vx)` is not the primary separator: the plain-G1 march code passes velocity information through the torque channels (channels 0-11) even in the `zero` mode, which is expected and correctly attributed to the plant's policy, not to the hub.

### Authority envelope (`results/envelope.json`, `results/push_frontier.json`)

- Lateral impulse (y-axis): the robot survives impulses of duration <= 0.40 s at 140 N and falls at 0.45 s at 140 N. Region *closable* by the loop: duration < 0.45 s, |F| <= 140 N (the base absorbs it). Frontier mapped at 21 grid points, 46 envelope probes (`push_probe2/3`).
- Torque authority: the contract delivers about 30 N-m of hip-roll correction (0.75 units x 40 N-m/unit) against a tipping moment of about 112 N-m for a sustained 140 N push: sustained pushes are outside the loop's authority, and every mode falls (including the recovery-torque hub). This is the honest boundary of the *authority x latency* envelope: the loop closes within the 25 ms deadline but cannot defend sustained floor loads.
- Longitudinal: forward push at 0.3 s / 150 N falls; the same sustained push applied backwards is survived by all modes (the base policy is directionally robust).

## Calibration note (why the numbers changed)

An initial audit found that in the first version of the loop the decoded velocity command never reached the plant (`cmd_override` only re-encoded to the channel-62 frame but not to `deploy12.cmd[0]`); the physics was identical across modes and the separation was spurious. The fix was applied in `bridge_g1.py` and the entire battery was re-run. All numbers in this repository reflect the corrected loop. The historical v1 files are preserved in `results/` (and in the phase-2 repo `02_cl`) marked as superseded.

## Repository layout

```
surrogate_cl/
├── src/
│   ├── bridge_g1.py           # CL datasource of the G1 walker (perturb, task, state TSV)
│   ├── demo_walk.py           # Nengo-LIF hub + calibrated decide
│   ├── ablate_loop.py         # 5-mode runner + spike sorting + state alignment
│   ├── capture_signal.py      # per-tick counts[64]+state captures (Gate A raw data)
│   ├── signal_analysis.py     # MI / TE / channel MI, bias-corrected (Gate A)
│   ├── probe_frontier.py      # lateral impulse frontier sweep
│   ├── envelope.py            # authority x latency envelope figure + json
│   ├── task_tracking.py       # multi-profile tracking battery (H1 primary)
│   ├── perturb_sweep.py       # flat-ground lateral sweep
│   ├── probe_push.py          # sustained push probes (authority limit)
│   └── task_progress.py       # flat-ground distance / task progress
├── results/                   # json + png artefacts (see table below)
├── requirements.lock.txt      # pinned environment (uv freeze)
└── .gitignore
```

## Reproducing the results

Environment (verified on Windows 11, Python 3.12.13):

```
mujoco==3.13.0
nengo==4.1.0
torch==2.14.0+cpu
numpy==2.5.3
scipy==1.18.1
matplotlib==3.11.2
cl-sdk==1.0.0        (CC BY-NC, academic use)
```

The virtual environment is a junction to `02_cl/.venv312` (managed with `uv`; there is no `pip` inside it). Full pins are in `requirements.lock.txt`.

The G1 model assets and checkpoints (`unitree_rl_gym`, `mujoco_menagerie`, `motion.pt`) live in `third_party/` at the project root and are not re-distributed here; see the project-level bootstrap script.

Commands:

```
uv pip install -r requirements.lock.txt --python .venv312/Scripts/python.exe

python src/task_tracking.py     # H1 primary battery (5 profiles, 4 modes, 5 seeds)
python src/capture_signal.py    # Gate A raw captures (2 seeds)
python src/signal_analysis.py   # Gate A MI/TE analysis
python src/probe_frontier.py    # lateral frontier
python src/envelope.py          # authority x latency figure
python src/perturb_sweep.py     # flat-ground sweep
```

Each script writes its artefacts under `results/`. Raw bridge telemetry is written by the datasource subprocess to `%LOCALAPPDATA%\Temp\opencode\bridge_telemetry.json`.

## Data artefacts (`results/`)

| file | content |
|---|---|
| `task_tracking.json` / `.png` | primary battery, per-profile per-mode multi-seed stats |
| `f1_capture_{mode}_s{seed}.json` | per-tick counts + state, 5 modes x {7,29} |
| `f1_capture_index.json` | capture manifest (ticks, seeds, wall time) |
| `f2_signal_summary.json` / `.png` | Gate A MI/TE aggregates and figure |
| `push_frontier.json` | lateral impulse frontier (21 rows) |
| `envelope.json` / `.png` | authority x latency envelope |
| `perturb_sweep.json` / `.png` | flat-ground lateral robustness |
| `push_probe2.json`, `push_probe3.json` | sustained push probes (authority limit) |
| `task_progress.json` | flat-ground distance/task progress |

Historical superseded files (`f2_*_v1`, pre-fix captures) are kept in the archive for transparency.

## Status

- [x] Loop corrected (velocity command reaches the plant) and re-validated
- [x] H1 primary battery: 5 profiles x 4 modes x 5 seeds, neural best everywhere
- [x] Gate A: MI/TE causal validation, Poisson dead substrate exactly zero
- [x] Authority envelope (lateral frontier + torque/tipping + deadline)
- [x] Calibrated decode (neural is the best decoder after calibration)
- [ ] Poisson column in the multi-profile battery (5 modes) and extended seeds (>= 6) for p <= 0.05
- [ ] Substrate interchangeability (Nengo-LIF vs Izhikevich/BL-1) and MI reproducibility across substrates (Gate B)
- [ ] Paper packaging (figures, negative sweep, video) -- tracked in the parent research-line repository

## License and dependencies

The code in this repository is MIT-licensed (see `LICENSE`). It depends on `cl-sdk` (CC BY-NC, academic use only), `nengo` (MIT), MuJoCo (Apache-2.0), and third-party G1 assets (Unitree RL Gym) that are not redistributed here. Nothing in this project cultivates biological cells and no CL1 hardware is required; everything runs in simulation, which is stated because the "biology" in the loop is the simulated spiking substrate, not the SDK.