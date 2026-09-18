---
title: 'surrogate-cl: Closed-loop validation of a load-bearing substrate on a humanoid robot'
tags:
- Python
- spiking neural networks
- humanoid locomotion
- Cortical Labs
- information theory
authors:
- name: Alan Daleth Hernández Barreto
  orcid: 0009-0007-7439-1920
  corresponding: true
  affiliation: "1"
affiliations:
- name: "Instituto Tecnologico de Zacatepec, Mexico"
  index: 1
date: 17 September 2026
bibliography: paper.bib
---

# Summary

`surrogate-cl` is an open-source Python package that validates whether a simulated spiking neural culture substrate is *load-bearing* in a closed-loop humanoid locomotion task. The software closes a real-time control loop on a Unitree G1 robot (MuJoCo simulation) through the Cortical Labs SDK contract (`cl-sdk`), routes sensor data to 64 electrodes, reads spike events from a simulated culture, and decodes motor commands from those spikes. The package provides rigorous ablation conditions (disconnected, uncoupled, lesioned, dead substrate) and information-theoretic metrics (mutual information, transfer entropy) to statistically verify that task information travels through the substrate's spikes.

# Statement of need

Current brain-computer interface (BCI) research faces a reproducibility crisis: wetware experiments are expensive, non-deterministic, and difficult to share. `surrogate-cl` addresses this by providing a fully simulated, deterministic alternative that preserves the essential closed-loop dynamics of a spiking culture substrate while being freely available, auditable, and reproducible on commodity hardware.

The package solves three problems simultaneously:

1. **Reproducibility**: Every run is parameterized by explicit random seeds, producing identical results across machines. The 6-seed battery ({1, 7, 13, 29, 55, 91}) provides statistical power while remaining computationally tractable.

2. **Causal validation**: Unlike open-loop spike analysis, `surrogate-cl` closes the loop in real time (25 ms contract deadline), ensuring that any measured information transfer is actually *used* by the controller. The ablation modes (zero, random, mask0.5, poisson) provide honest null conditions that destroy the substrate's contribution while preserving its statistics.

3. **Information-theoretic rigor**: The package implements bias-corrected mutual information and transfer entropy with permutation-based significance testing, providing statistical evidence that the substrate is not merely passing spikes but carrying task-relevant information.

The primary users are BCI researchers, computational neuroscientists, and robotics engineers who need to validate spiking-substrate control loops without access to physical wetware. The package has been used to demonstrate that a Nengo-LIF substrate achieves significantly better tracking than all ablation conditions across 5 velocity profiles (paired permutation p ≤ 0.05, Cohen's d_z = -2.4 to -75.0), and that a dead Poisson substrate with matched spike rate carries exactly zero task information despite producing spikes.

# Software design

`surrogate-cl` follows a modular architecture centered on the `ablate_loop.py` runner, which orchestrates the closed-loop execution across five ablation modes:

- **neural**: intact Nengo-LIF hub with calibrated decode map
- **random**: uncoupled decoder producing fixed random commands
- **zero**: zero command (no closed-loop output)
- **mask0.5**: 50% electrode lesion (per-seed random mask)
- **poisson**: dead substrate with matched expected spike rate but no state information

The hub is a Nengo network (`demo_walk.py`) encoding a 5-dimensional state vector `[roll, pitch, thsig, nspk, ctx]` in a 1000-neuron LIF ensemble, decoded to a 2-dimensional command `[vx, tau]` through a calibrated function that includes activity gating and tilt-based slowdown. The plant is the `Deploy12` G1 policy loader (LSTM + PD) running inside the Cortical Labs simulator datasource subprocess.

Information-theoretic analysis (`signal_analysis.py`) computes bias-corrected mutual information between task profiles and decoded commands, and transfer entropy along the causal chain electrode-62 → command and spikes → command. All metrics are evaluated against a 50-permutation shuffle null to control for multiple comparisons.

# State of the field

Several tools exist for spiking neural network simulation (Brian2 [@Goodman:2008], NEST [@Gewaltig:2007], NEURON [@Carnevale:2006]), but none provide closed-loop validation on physical hardware platforms. `Brian2` and `NEST` focus on offline simulation without real-time constraints. The Nengo ecosystem [@Bekolay:2014] provides the Neural Engineering Framework for building spiking controllers but lacks integrated hardware-in-the-loop validation.

The Cortical Labs `cl-sdk` enables communication with the CL1/DishBrain wetware but provides no built-in ablation framework or statistical validation tools. `surrogate-cl` bridges this gap by combining Nengo's spiking network capabilities with MuJoCo's physics simulation and a rigorous experimental protocol. Unlike approaches that analyze spikes in open loop, `surrogate-cl` closes the loop in real time, ensuring that measured information transfer is causally relevant to the control task.

The package contributes a validated experimental protocol where the substrate loop is shown to be *behaviourally non-inert*: it changes real walking outcomes with statistically significant, opposing signs by terrain. On rough ground it preserves balance where the open-loop policy alone collapses (0/6 to 2/6 survival; p = 0.0011), while on a precision-critical 5 cm curb it destabilizes the step plan the policy clears reliably (6/6 down to 1/6, p = 0.0076).

# Research impact statement

`surrogate-cl` provides a validated foundation for closed-loop BCI research without physical wetware. The package demonstrates that:

1. **Substrate interchangeability**: Results reproduce across substrate architectures (Nengo-LIF and Izhikevich RSNN), suggesting the closed-loop validation protocol is substrate-agnostic.

2. **Honest ablation**: The dead Poisson substrate (matched spike rate, no state) produces exactly zero transfer entropy and zero per-channel mutual information with velocity, proving that spike statistics alone are insufficient — the information *carried by* the spikes is what drives tracking.

3. **Authority boundaries**: The package quantifies the loop's authority envelope, showing it survives lateral impulses ≤ 0.40 s at 140 N but cannot defend sustained floor loads, providing honest limits on real-world applicability.

These findings have implications for the design of BCI systems where spiking substrates must demonstrate causal, not merely correlational, information transfer.

# AI usage disclosure

Generative AI tools (Claude, GPT-4, GitHub Copilot) were used in the development of this software for:

- **Code generation**: implementation of ablation modes, spike sorting algorithms, information-theoretic metrics, and MuJoCo scene configuration
- **Refactoring**: structural improvements to the closed-loop runner and analysis pipelines
- **Documentation**: docstrings, README content, and API documentation
- **Testing**: test case generation and edge case identification

All AI-generated code was reviewed, tested, and validated by the author before inclusion. The experimental design, scientific methodology, statistical analysis, and interpretation of results are entirely human-authored. AI tools were used as productivity aids, not as autonomous agents. The author made all core architectural and design decisions.

# Acknowledgements

We acknowledge Cortical Labs for the `cl-sdk`, Unitree Robotics for the G1 humanoid model, MuJoCo for the physics engine, and Nengo for the Neural Engineering Framework.

# References
