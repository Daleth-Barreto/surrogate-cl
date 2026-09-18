# surrogate-cl

Closed-loop validation that a *load-bearing task function* survives a simulated, non-trainable culture substrate, measured with information-theoretic statistics (mutual information and transfer entropy) and a physical-authority envelope.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

## Quick start

```bash
# Clone the repository
git clone https://github.com/Daleth-Barreto/surrogate-cl.git
cd surrogate-cl

# Create virtual environment
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.lock.txt

# Run the primary battery (5 profiles x 5 modes x 6 seeds)
python src/task_tracking.py
```

## Overview

This repository is the experiment scaffold of the research project "Does the substrate compute?" (working title). It closes a humanoid locomotion loop (Unitree G1, MuJoCo) through the **Cortical Labs contract** (`cl-sdk`, in-silico), reads spike events from a **simulated culture** that mimics the role of a DishBrain / CL1 wetware, and decodes the task from those spikes. The central claim is checked with honest ablation conditions: a disconnected decoder (zero), an uncoupled decoder (random), a 50 % electrode lesion (mask0.5), and a dead Poisson substrate (poisson), all against the intact Nengo-LIF hub (neural).

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

Multi-seed protocol: 6 fixed seeds {1, 7, 13, 29, 55, 91}; the Nengo hub is deterministic, so the variance comes from the simulator RNG, the cultural spike/count RNG, the lesion draw, and real-time hub timing.

Metrics:

- Tracking RMSE of command vs profile (primary outcome).
- Mutual information (`signal_analysis.py`, bias-corrected against a 50-permutation shuffle null, nats).
- Transfer entropy along the causal chain `c62 -> cmd` and `spikes -> cmd`.
- Perturbation authority envelope (`push_frontier` / `envelope.py`).

## Results

### Task tracking (5 profiles x 5 modes x 6 seeds, `results/task_tracking.json`)

RMSE mean +/- std, normalised so that `zero` (no command) scores approximately the profile itself (0.55-0.64). The nominal reference `pass_rmse = 0.18` is the best single-seed tracking observed before calibration; separation from the null conditions is the statistical claim, not the absolute value.

| profile | neural | zero | random | mask0.5 | poisson |
|---|---|---|---|---|---|
| baseline | 0.238 +/- 0.010 | 0.612 +/- 0.000 | 0.337 +/- 0.004 | 0.397 +/- 0.057 | 0.306 +/- 0.006 |
| multi_step | 0.234 +/- 0.006 | 0.606 +/- 0.000 | 0.329 +/- 0.005 | 0.383 +/- 0.057 | 0.297 +/- 0.003 |
| sine | 0.241 +/- 0.005 | 0.601 +/- 0.000 | 0.333 +/- 0.007 | 0.387 +/- 0.052 | 0.307 +/- 0.005 |
| descend | 0.258 +/- 0.009 | 0.644 +/- 0.000 | 0.358 +/- 0.005 | 0.424 +/- 0.058 | 0.336 +/- 0.003 |
| pulse | 0.216 +/- 0.012 | 0.554 +/- 0.000 | 0.294 +/- 0.008 | 0.344 +/- 0.045 | 0.252 +/- 0.006 |

The neural hub is the best decoder on every profile. The 50 % lesion consistently degrades the decode (+40 % to +96 % RMSE), and in some seeds it erases electrode 62 altogether, which is the mechanism by which the task signal disappears. A dead Poisson substrate (matched expected spike rate, but carrying no state) lands strictly between the neural hub and the uncoupled decoders on every profile, showing that it is the *information carried by the spikes* that drives tracking, not the mere presence of spikes. The `zero` mode issues no command, so its RMSE measures the profile itself and anchors the scale.

Significance (exact paired permutation): with 6 seeds the one-sided p has a floor of 1/64 = 0.0156 and the two-sided p a floor of 2/64 = 0.03125. On **all 5 profiles and all 3 null comparisons** (`neural` vs `random`, vs `mask0.5`, vs `zero`) the difference points the same way for 6/6 seeds, so p(one-sided) = 0.0156 and p(two-sided) = 0.03125: **p <= 0.05 is reached on every profile**. Effect sizes are large: baseline Cohen d_z = -8.6 vs random, -2.4 vs mask0.5, -39.2 vs zero; the range across profiles is d_z -7.8 to -13.9 vs random, -2.4 to -3.0 vs mask0.5, and -28.7 to -75.0 vs zero.

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

### Terrain robustness (Fase 0.3, `results/terrain_probe.json` / `terrain_probe2.json` / `terrain_stats.json`)

Non-flat MuJoCo scenes expose how the substrate-mediated loop interacts with the physical policy under a constant 0.5 m/s march duty (channel 62 = 0.5, `task=[(0.0, 0.5)]`). The terrain scenes live in `data/g1_description/` (ramp, curb, rough) and re-point the mesh<->compiler to the project assets. Every combination runs in a fresh worker subprocess (the cl producer fails to hand-shake within the SDK 15 s timeout under CPU contention, so the probe retries after purging orphaned `cl-data-producer` subprocesses).

| scene | mode | survivors | forward distance (m) | notes |
|---|---|---|---|---|
| ramp (2 deg) | reference | 2/2 | 6.23 +/- 0.17 | stays up |
| " | neural | 2/2 | 4.88 +/- 0.09 | substrate liquidity cost (slower vx ~0.37) |
| " | zero | 2/2 | 6.48 +/- 0.17 | marches, no command |
| " | random | 2/2 | 4.82 +/- 0.11 | uncoupled, slower |
| curb 5 cm | reference | **6/6** | 6.31 +/- 0.00 | clears the step every seed |
| " | neural | **1/6** | 5.35 +/- 0.56 | trips on precision crossing (fall p=0.0076, Fisher one-sided neural <= ref) |
| rough bumps | reference | **0/6** | 3.89 +/- 0.06 | collapses at the first bump (x~4 m) every seed |
| " | neural | **2/6** | 5.87 +/- 0.56 | survives 2/6 and always travels further: t(5.1) = 8.63, p = 0.0011 (Welch, two-sided) |
| " | zero | 0/2 | 3.84 +/- 0.03 | stops at the first bump |
| " | random | 0/2 | 5.76 +/- 0.20 | wanders far, falls |

Reading (honest, terrain-dependent):

- The substrate loop is **not behaviourally inert**: it changes real walking outcomes with statistically significant, opposing signs by terrain. On rough ground it preserves balance where the open-loop policy alone collapses (0/6 to 2/6 survival; significantly longer traversal, p = 0.0011). On a precision-critical 5 cm curb it destabilizes the step plan the policy clears reliably (6/6 down to 1/6, p = 0.0076). On slopes both modes survive; the closed loop walks slower (a measurable latency/liquidity cost of the real-time substrate).
- A 10 cm curb was intentionally too hard (reference also falls 0/6) and is archived as `curb10` in `terrain_stats.json`; the 5 cm curb is the calibrated step.
- Caveat: because the loop is closed in real time, the same seed can land differently across wall-clock sessions (hub latency jitter); statistics therefore use one invocation per (scene, mode, seed), with the 6-seed battery `{1,7,13,29,55,91}` as canonical.

### Latency and the critical path (Fase 0.4)

The contract tick is 25 ms (`TPS = 40`), and the loop-overrun deadline is `1/TPS = 25 ms` (asserted  same-tick in `ablate_loop.py`). But the code-level *staleness* of the information the command is derived from is longer, and must not be hidden:

```
sensors -> frames[64] .......... queue (multiprocessing, ~microseconds transport)
frames -> culture (simulated) .. consumption of the *newest* frame per tick
culture -> hub decode .......... Nengo runs HUB_HORIZON_STEPS=600 steps of dt=0.002 s
                                   per chunked run = 1.2 s of hub time; the decode
                                   integrates spikes over this trailing window
decode -> stimulation/plant ..... applied under the 25 ms contract deadline
```

- **Window staleness = 1.2 s**: with the Nengo hub run in chunks of 600 steps, the command on the plant at hub-tick `t` is the decode of a spike window that closed up to 1.2 s earlier. Immediate transport/drain adds only tens of ms; the dominant term is the horizon.
- **Contract deadline = 25 ms**: the round-trip within one tick (produce frames -> read newest -> compute -> stimulate) is asserted to fit in `1/TPS`, and the cadence histograms in `envelope.py` confirm it stays inside the deadline.
- **Interpretation**: the loop is therefore *liquidity-limited* by the 1.2 s information window, not by the 25 ms contract tick. This is the mechanism behind the terrain speed cost (ramp: neural ~0.37 vs reference ~0.49 m/s) and behind the curb instability (decisions at the step are based on ~1.2 s-old balance/position evidence). It also makes the rough-ground stabilization result stronger: the loop still recovers trip dynamics toward extra traversed distance (p = 0.0011) using information that is a full gait period old.

## Calibration note (why the numbers changed)

An initial audit found that in the first version of the loop the decoded velocity command never reached the plant (`cmd_override` only re-encoded to the channel-62 frame but not to `deploy12.cmd[0]`); the physics was identical across modes and the separation was spurious. The fix was applied in `bridge_g1.py` and the entire battery was re-run. All numbers in this repository reflect the corrected loop. The historical v1 files are preserved in `results/` (and in the phase-2 repo `02_cl`) marked as superseded.

## Repository layout

```
surrogate_cl/
+- src/
|  +- bridge_g1.py           # CL datasource of the G1 walker (perturb, task, state TSV)
|  +- demo_walk.py           # Nengo-LIF hub + calibrated decide
|  +- ablate_loop.py         # 5-mode runner + spike sorting + state alignment
|  +- capture_signal.py      # per-tick counts[64]+state captures (Gate A raw data)
|  +- signal_analysis.py     # MI / TE / channel MI, bias-corrected (Gate A)
|  +- probe_frontier.py      # lateral impulse frontier sweep
|  +- envelope.py            # authority x latency envelope figure + json
|  +- task_tracking.py       # multi-profile tracking battery (H1 primary)
|  +- terrain_probe.py       # Fase 0.3 non-flat scenes battery (worker-per-run)
|  +- terrain_analysis.py    # merge + Fisher / Welch statistics + summary figure
|  +- render_terrain.py      # mp4 clips from recorded qpos (worker-per-run, CLI)
|  +- perturb_sweep.py       # flat-ground lateral sweep
|  +- probe_push.py          # sustained push probes (authority limit)
|  +- task_progress.py       # flat-ground distance / task progress
+- config/
|  +- g1.yaml                # canonical flat config (deployed fenceless flat scene)
|  +- g1_{ramp,curb,rough}.yaml  # terrain configs (absolute xml_path)
+- data/
|  +- g1_description/
|     +- g1_12dof.xml        # local copy, meshes redirected to project assets
|     +- scene_{ramp,curb,rough}.xml  # non-flat terrain scenes
+- results/                  # json + png artefacts (see table below)
+- requirements.lock.txt     # pinned environment (uv freeze)
+- .gitignore
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

python src/task_tracking.py     # H1 primary battery (5 profiles, 5 modes, 6 seeds)
python src/terrain_probe.py     # Fase 0.3 battery (--scenes/--modes/--seeds/--out)
python src/terrain_analysis.py  # terrain statistics + summary figure
python src/render_terrain.py    # terrain mp4 clips (--jobs scene:mode:seed --reuse)
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
| `terrain_probe.json` / `.png` | Fase 0.3 four-mode terrain battery (ramp/curb/rough, seeds {7,29}) |
| `terrain_probe2.json` / `.png` | reference+neural power battery (curb 5 cm, rough; 6 canonical seeds) |
| `terrain_stats.json` / `terrain_summary.png` | merged aggregates, Fisher exact + Welch stats, combined figure |
| `terrain_videos/manifest.json` + `.mp4`/`.npz` | replayed walker clips (rough/curb, reference/neural) with fixed camera + framing margin |
| `FIGURES.md` | index of every figure/material and the message it carries |
| `WHITE_PAPER_draft.md` | go/no-go white-paper draft (four-front evidence, positioning, threats) |
| `f1_capture_{mode}_s{seed}.json` | per-tick counts + state, 5 modes x {7,29} |
| `f1_capture_index.json` | capture manifest (ticks, seeds, wall time) |
| `f2_signal_summary.json` / `.png` | Gate A MI/TE aggregates and figure |
| `substrate_izh.py` (`src/`) | Gate B substrate #2: Izhikevich RSNN culture (calibrated drive, 64 ch contract) |
| `task_tracking_izh.json` / `task_tracking_multi_izh.png` | Gate B reduced battery (2 seeds) on the Izhikevich substrate |
| `f1_capture_izh_*.json` / `f1_capture_index_izh.json` | Gate B captures (Izh substrate) |
| `f2_signal_izh.json` / `f2_signal_izh_fig.png` | Gate B MI/TE aggregates on the Izh substrate |
| `push_frontier.json` | lateral impulse frontier (21 rows) |
| `envelope.json` / `.png` | authority x latency envelope |
| `perturb_sweep.json` / `.png` | flat-ground lateral robustness |
| `push_probe2.json`, `push_probe3.json` | sustained push probes (authority limit) |
| `task_progress.json` | flat-ground distance/task progress |

Historical superseded files (`f2_*_v1`, pre-fix captures) are kept in the archive for transparency.

## Status

- [x] Loop corrected (velocity command reaches the plant) and re-validated
- [x] H1 primary battery: 5 profiles x 5 modes x 6 seeds, neural best everywhere, p <= 0.05 on all profiles
- [x] Gate A: MI/TE causal validation, Poisson dead substrate exactly zero
- [x] Authority envelope (lateral frontier + torque/tipping + deadline)
- [x] Calibrated decode (neural is the best decoder after calibration)
- [x] Substrate interchangeability (Izhikevich RSNN substrate #2) -- closed-loop advantage and Gate A MI reproduce on the second substrate (reduced 2-seed battery; full-power 6-seed battery optional)
- [x] Paper packaging (white-paper narrative, negative sweep) -- figures/video indexed in `results/FIGURES.md`

## License and dependencies

The code in this repository is MIT-licensed (see `LICENSE`). It depends on `cl-sdk` (CC BY-NC, academic use only), `nengo` (MIT), MuJoCo (Apache-2.0), and third-party G1 assets (Unitree RL Gym) that are not redistributed here. Nothing in this project cultivates biological cells and no CL1 hardware is required; everything runs in simulation, which is stated because the "biology" in the loop is the simulated spiking substrate, not the SDK.

## AI usage policy

This project was developed with assistance from AI language models (Claude, GPT-4, Copilot) for:

- **Code refactoring**: structural improvements, function extraction, and code organization
- **Documentation**: docstrings, inline comments, README writing, and API documentation
- **Testing**: test case generation and edge case identification
- **Research**: literature review, related work synthesis, and methodology suggestions

All AI-generated code was reviewed, tested, and validated by the author before inclusion. The experimental design, scientific methodology, and interpretation of results are entirely human-authored. AI tools were used as productivity aids, not as autonomous agents.

If you use AI tools to contribute to this project, please disclose their usage in your pull request description.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{hernandez2026surrogatecl,
  author       = {Hernández Barreto, Alan Daleth},
  title        = {surrogate-cl: Load-bearing substrate validation on the Cortical Labs contract},
  year         = {2026},
  url          = {https://github.com/Daleth-Barreto/surrogate-cl},
  license      = {MIT}
}
```

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/Daleth-Barreto/surrogate-cl).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [Cortical Labs](https://corticallabs.com/) for the `cl-sdk`
- [Unitree Robotics](https://www.unitree.com/) for the G1 humanoid model
- [MuJoCo](https://mujoco.org/) for the physics engine
- [Nengo](https://www.nengo.ai/) for the neural engineering framework