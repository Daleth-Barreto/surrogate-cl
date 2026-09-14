"""H1 (tarea portante): perturbacion + ablaciones del sustrato.

Fire lateral pushes on the pelvis (xfrc_applied in bridge_g1) at fixed times
and re-run the four decoder modes under the same disturbance schedule.

Expected: the neural hub (which decelerates when it reads tilt on channels
12/13) survives, while decoupled readouts (random/zero/mask0.5) fall. If the
hub's abstraction is not load-bearing, all modes survive and that negative
result motivates passing recovery torques through the same CL contract.

Phase A sweeps the lateral push amplitude over modes neural/random; the
strongest force where neural survives and random falls is selected. Phase B
runs zero/mask0.5 at that force.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablate_loop import TPS, DURATION_SEC, run_mode

PUSH_DUR = 0.15
PUSH_TIMES = [4.0, 8.0]
FORCES = [0.0, 50.0, 90.0, 140.0]
SEED = 7


def push_schedule(fy: float):
    return [{"t0": t, "dur": PUSH_DUR, "force": [0.0, fy, 0.0]}
            for t in PUSH_TIMES]


def main():
    RES = Path(__file__).resolve().parent.parent / "results"
    RES.mkdir(exist_ok=True)

    entries = []
    print("phase A: sweep lateral push force over neural/random")
    for fy in FORCES:
        for mode in ("neural", "random"):
            r = run_mode(mode, seed=SEED, perturb=push_schedule(fy))
            r["perturb_fy"] = fy
            entries.append(r)
            w = r["walker"]
            print("A F=%.0fN mode=%-8s fallen=%s h_last=%.2f vx_last=%.2f "
                  "cmd=%.2f bridge_n=%s wall=%.1fs"
                  % (fy, mode, w.get("fallen"), w.get("h_last") or -1.0,
                     w.get("vx_last") or -1.0, r["mean_cmd"],
                     r["bridge_read"].get("n_calls"), r["_wall_step"]))
            time.sleep(0.5)

    def fallen_by(mode):
        return {e["perturb_fy"]: bool(e["walker"].get("fallen"))
                for e in entries if e["mode"] == mode}

    fn = fallen_by("neural")
    fr = fallen_by("random")
    cands = [fy for fy in FORCES if fr.get(fy, True) and not fn.get(fy, False)]
    neural_safe = [fy for fy in FORCES if not fn.get(fy, False)]
    f_star = max(cands) if cands else (max(neural_safe) if neural_safe else max(FORCES))

    print("phase B: zero/mask0.5 at chosen force F=%.0fN" % f_star)
    for mode in ("zero", "mask0.5"):
        r = run_mode(mode, seed=SEED, perturb=push_schedule(f_star))
        r["perturb_fy"] = f_star
        entries.append(r)
        w = r["walker"]
        print("B F=%.0fN mode=%-8s fallen=%s h_last=%.2f vx_last=%.2f "
              "cmd=%.2f wall=%.1fs"
              % (f_star, mode, w.get("fallen"), w.get("h_last") or -1.0,
                 w.get("vx_last") or -1.0, r["mean_cmd"], r["_wall_step"]))
        time.sleep(0.5)

    for e in entries:
        e.pop("_latency_hist", None)

    interp = _interpret(entries, f_star)
    summary = {
        "block": "surrogate_cl / H1 load-bearing loop",
        "duration_sec": DURATION_SEC,
        "tps": TPS,
        "push_dur": PUSH_DUR,
        "push_times": PUSH_TIMES,
        "forces": FORCES,
        "chosen_f_star": f_star,
        "schedule": push_schedule(f_star),
        "random_range": [0.0, 2.0],
        "decide_ref": "demo_walk.decide: gate=30*(nspk-0.15); "
                      "slow=1-0.7*sig(8*(max(roll,pitch)-0.28))",
        "interpretation": interp,
        "matrix": {"neural_fallen": fn, "random_fallen": fr},
        "entries": [{k: v for k, v in e.items()
                     if k in ("mode", "perturb_fy", "mean_cmd", "mean_nspk",
                              "stim_events", "walker", "wall_time")}
                    for e in entries],
    }
    with open(RES / "perturb_sweep.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    fig = _figure(entries, f_star)
    fig.savefig(RES / "perturb_sweep.png", dpi=140)
    print("saved:", RES / "perturb_sweep.json", RES / "perturb_sweep.png")
    print("interpretation:", interp)


def _interpret(entries, f_star):
    by = {(e["mode"], e["perturb_fy"]): e for e in entries}
    n = by.get(("neural", f_star))
    r = by.get(("random", f_star))
    z = by.get(("zero", f_star))
    m = by.get(("mask0.5", f_star))
    nf = n["walker"].get("fallen") if n else None
    rf = r["walker"].get("fallen") if r else None
    zf = z["walker"].get("fallen") if z else None
    mf = m["walker"].get("fallen") if m else None
    if nf is False and (rf is True or zf is True or mf is True):
        head = ("H1 CONFIRMED lo siguiente: el hub spike del sustrato es tarea "
                "portante bajo perturbacion")
    else:
        head = ("H1 NO diferenciada en este punto de operacion: el lazo spike "
                "parece redundante aun bajo empuje lateral")
    return head + (", F*=%.0fN -> neural=%s random=%s zero=%s mask0.5=%s"
                   % (f_star, nf, rf, zf, mf))


def _figure(entries, f_star):
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    sel = [e for e in entries if e["perturb_fy"] in (0.0, f_star)]
    rows = list({(e["mode"], e["perturb_fy"]) for e in sel})[:3]
    for i, (mode, fy) in enumerate(rows or [("neural", 0.0)]):
        e = next(x for x in sel if x["mode"] == mode and x["perturb_fy"] == fy)
        ax = axes[i]
        ax[0].plot(e["cmd_series"], lw=0.8)
        ax[0].set_ylabel("cmd vx")
        ax[0].set_title("%s F=%.0fN fallen=%s" % (mode, fy, e["walker"].get("fallen")))
        ws = e["walker_series"]
        if ws:
            ax[1].plot([s[1] for s in ws], lw=0.8)
            ax[2].plot([s[2] for s in ws], lw=0.8)
        ax[1].set_ylabel("height (m)")
        ax[2].set_ylabel("vx (m/s)")
    fig.suptitle("G1 + CL-contract: lateral pelvis push at t=%.0f & t=%.0f s "
                 "(dur %.2f s)" % (PUSH_TIMES[0], PUSH_TIMES[1], PUSH_DUR))
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()