# surrogate_cl — Sustituto SNN entrenable ↔ Wetware por contrato CL

Línea de investigación nueva dentro de `C:\Proyectos\papers`, paralela a F1 (`01_snn`,
planta MuJoCo G1/Deploy12) y F2 (`Icra2027/02_cl`, contrato CL1 con cl-sdk).

## Tesis

El sustrato de cultivo (CL1/DishBrain) **no se entrena por gradiente**: solo
condicionamiento closed-loop lento y costoso. Pero la misma función puede
**entrenarse una vez in-silico** con un SNN (Nengo/NEF) y desplegarse **verbatim**
sobre el mismo contrato (sensores → electrodos → spikes → decisiones → estímulos).
Esta carpeta convierte esa sustitución en un experimento de *tarea portante*
(load-bearing) con ablaciones honestas tipo doom-neuron.

## Hipótesis (H1, evolucionada tras los datos)

**H1 original:** una perturbación física (empuje) distingue al hub neural de los
decodificadores desconectados (random/zero).

**Lo que los datos mostraron (3 sondeos), y el rediseño de H1:**

1. **Bug crítico encontrado y reparado.** El canal de comando de vx nunca llegaba a la
   planta: `cmd_override` solo se re-codificaba al frame del canal 62, nunca a
   `deploy12.cmd[0]` (observado por la LSTM). La física era idéntica entre modos.
   Ahora `_ctrl_step` aplica `cmd_override` a `dep.cmd[0]` (`bridge_g1.py`).
2. **La planta es robusta en plano.** Impulsos laterales de 0.15 s hasta 140 N los
   sobreviven los 4 modos (el lazo es redundante en esa clase de disturbio, igual
   que doom-neuron). `results/perturb_sweep.json`.
3. **El lazo queda por debajo de la autoridad necesaria para defender cargas
   sostenidas.** Con empujes ≥0.5 s/140 N (≈112 N·m de volcamiento) caen todos,
   incluido el hub con canal de torque de recuperación (cadera-roll, ±30 N·m máximo
   disponible por el contrato: `TAU_GAIN=40`, tope de estímulo 3 µA).
   `results/push_probe2.json`, `results/push_probe3.json`.
4. **La tarea portante es el seguimiento de un perfil de velocidad** que entra al
   sustrato por el canal 62 (telemetría de líder/obstáculo). Ahí el hub decodificado
   por spikes rinde: `results/task_tracking.json`.

**H1 operacional (definitiva):** el lazo es tarea portante si el perfil de velocidad
que pide la tarea viaja por los spikes. RMSE comando-vs-perfil sobre 12 s (hub
calibrado: nominal plano ≈0.54 m/s):

Tabla **multi-seed definitiva** (media ± std sobre 5 semillas [1,7,13,29,55]; el
hub Nengo es determinista, la varianza proviene del RNG de spikes/cultura, del
simulador y del dibujo de lesión):

| modo     | RMSE         | seg 0.5 (rmse/cmd) | seg 0.8 (rmse/cmd) | ch62_mean (0.5→0.8) |
|----------|--------------|--------------------|--------------------|----------------------|
| neural   | 0.191 ± 0.008| 0.146 ± 0.022 / 0.550 | 0.233 ± 0.024 / 0.631 | 0.69 → 1.13      |
| random   | 0.336 ± 0.003| 0.247 / 0.378      | 0.474 / 0.375      | 0.67 → 1.10          |
| mask0.5  | 0.305 ± 0.036| 0.188 / 0.339      | 0.461 / 0.352      | 0.13 → 0.22          |
| zero     | 0.612 ± 0.000| 0.500 / 0.000      | 0.800 / 0.000      | 0.67 → 1.16          |

Lectura:
- neural es el **único modo que modula la velocidad con la tarea** (cmd 0.55→0.63 con
  el segmento, ch62 0.69→1.13) y tiene el mejor RMSE; random decodifica sin vínculo
  causal (plano ~0.38); zero no emite comando (RMSE ≈ el propio perfil);
- la **lesión del 50% de canales degrada el decode** (RMSE 0.305 vs 0.191, +60%): el
  ch62 queda parcialmente dañado según la semilla (0.13/0.22, no siempre borrado —
  ver `per_seed`), la información de tarea se degrada;
- la separación absoluta es conservadora (0.55→0.63): la codificación rate/ch62
  comprime el rango y el límite de paso (RMSE<0.18) cae justo en la media neural
  (0.180 min–0.201 max) — el claim del paper es **segregación + lesión**, no tracking
  analógico fino.

Sobre el canal de torque: se agregó como 2ª salida del hub (recuperación de actitud,
ch1/ch7 hip-roll), útil para ilustrar la *autoridad del contrato* más que como
mecanismo ganador de estabilidad en este régimen.

## Cómo ejecutar

El venv es una junction al F2: `C:\Proyectos\papers\surrogate_cl\.venv` →
`Icra2027\02_cl\.venv312`. La planta se referencia en su ruta original
(`SNN_SRC = C:/Proyectos/papers/Icra2027/01_snn/src` → Deploy12 + assets
unitree_rl_gym); no se duplican binarios.

```powershell
cd C:\Proyectos\papers\surrogate_cl
.venv\Scripts\python.exe src\task_tracking.py      # H1: perfil de velocidad
.venv\Scripts\python.exe src\perturb_sweep.py       # barrido lateral (plano)
.venv\Scripts\python.exe src\probe_push.py          # frontera de fallo (probe 2/3)
.venv\Scripts\python.exe src\task_progress.py       # distancia/tarea plano
```

Resultados y figuras en `results/`. Telemetría del bridge en
`%LOCALAPPDATA%\Temp\opencode\bridge_telemetry.json`.

## Arquitectura del contrato en este folder

- `bridge_g1.py` — datasource CL del G1; acepta `perturb` (fuerzas `xfrc_applied` en
  pelvis) y `task` (perfil vx(t) inyectado al canal 62); canales:
  0–11 torque-overlay (`TAU_GAIN` N·m por unidad), 12/13 actitud, 23 altura, 62 vx
  (task/override, tope 0.75 = 3 µA), 63 hub-ctx.
- `demo_walk.py` — hub Nengo 1000-LIF, salida 2-D [vx, tau]; `decide` con gate de
  actividad y desaceleración ante inclinación |roll|/|pitch|.
- `ablate_loop.py` — runner de modos neural/zero/random/mask0.5 + latencia SDK/bridge.
- `perturb_sweep.py`, `probe_push.py`, `task_progress.py`, `task_tracking.py` —
  experimentos.

## Qué se espera / novedad / impacto

**Esperado (resultado central para el paper):** la función entrenada in-silico
(velocidad + recuperación) viaja intacta por el contrato CL, la señala la lesión y
los decodificadores desacoplados no la reproducen; la *estabilidad* de planta es un
separador honesto (no se exagera el rol del lazo).

**Novedad:** primer lazo humanoide (G1 23-DoF) clausurado por contrato CL1 con un
sustituto SNN **entrenable e intercambiable** (misma interfaz spikes↔stims que el
wetware real), con ablaciones tipo doom-neuron y medición de sistemas (latencia,
autoridad de canal).

**Impacto:** (i) metodológico — separar *tarea portante* de *robustez de planta*
evita falsos positivos de función en biocomputación; (ii) de diseño de contrato —
los canales necesitan autoridad y latencia adecuadas al disturbio (τ≥volcamiento,
lazo ∠ impulso); (iii) para wetware — demuestra que la información de tarea es
espacialmente distribuida y sensible a lesión.

## Plan de publicación

- IEEE RA-L → vía principal (IF 5.3, sin APC); transferencia/ICRA Option o IROS
  2027 (1 mar 2027) como venías; **ICRA 2028** meta fija.
- Demos: HVAC 2027, Telluride 2027 (27 jun–16 jul 2027).

## Estado

- [x] H1 operacional (task tracking + lesión) — `results/task_tracking.json`
- [x] Frontera de perturbación (autoridad) — probes 2/3
- [x] Redundancia en plano (doom-neuron) — `perturb_sweep.json`
- [x] Bug del canal vx reparado y documentado
- [x] Hub calibrado (nominal plano ≈0.54) — `task_tracking.json` regenerado
- [x] Re-validar `Icra2027/02_cl` con el canal vx reparado → `f2_ablation_v2.*`,
      `f2_demo_v2.*` (la versión v1 queda intacta; ver `02_cl/README.md`)
- [x] Tabla definitiva multi-seed (5 semillas) — `results/task_tracking.json`/`task_tracking.png`
- [ ] Varios perfiles de tarea (más que 0.5→0.8→0.5) para el paper
- [ ] H2 transients BRIDGE / BL-1 / CL1; H3 doom-neuron (hecho en F2); H4 latencia