"""Terrain robustness probe (Fase 0.3).

Closes the same CL1-contract loop on non-flat MuJoCo scenes (ramp, curb, rough)
under a constant march duty (task = 0.5 m/s commanded on channel 62) and measures
whether the substrate-mediated loop preserves the reference policy's robustness.

Modes:
  reference  open-loop march (no stimulation at all; base policy only)
  neural     intact hub closes the contract
  zero       zero-command null
  random     uncoupled decoder null

Each (scene, mode, seed) combination runs in a FRESH worker subprocess to isolate
the cl producer lifecycle; orphaned cl-data-producer subprocesses are purged
before each attempt. Metrics per run: fallen flag, forward distance, final vx,
pelvis-height variability, and loop latency (deadline adherence).
Outputs: results/terrain_probe.json + results/terrain_probe.png
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
PROBE = Path(__file__).resolve()
RES = Path("C:/Proyectos/papers/surrogate_cl/results")
WDIR = Path("C:/Users/aland/AppData/Local/Temp/opencode")

SCENES = ["ramp", "curb", "rough"]
MODES = ["reference", "neural", "zero", "random"]
SEEDS = [7, 29]
DUTY = 0.5
SERIES_DT = 0.5  # telemetry series are downsampled by 25 ctrl steps (50 Hz)
PY = sys.executable


def _metrics(res):
    walk = res["walker"]
    series = res.get("walker_series", [])
    hh = np.asarray([s[1] for s in series]) if series else np.zeros(0)
    vv = np.asarray([s[2] for s in series]) if series else np.zeros(0)
    distance = float(np.sum(vv) * SERIES_DT) if len(vv) else 0.0
    lat = res.get("latency_loop_us", {})
    return {
        "mode": res["mode"],
        "scene": res.get("scene", "flat"),
        "seed": res.get("seed", None),
        "fallen": bool(walk["fallen"]),
        "distance_m": round(distance, 3),
        "vx_last": round(float(walk["vx_last"]) if walk.get("vx_last") else 0.0, 3),
        "h_std": round(float(np.std(hh)), 4) if len(hh) else None,
        "h_min": round(float(np.min(hh)), 3) if len(hh) else None,
        "t_end": round(float(walk["t_end"]) if walk.get("t_end") else 0.0, 2),
        "overruns_gt_25ms": lat.get("overruns_gt_25ms"),
        "latency_p95_us": lat.get("p95"),
        "mean_cmd": res.get("mean_cmd"),
        "stim_events": res.get("stim_events"),
    }


def _worker(scene, mode, seed, outfile):
    import cl._sim._base_producer as _bp

    _orig_start = _bp.BaseProducer.start

    def _start_long(self, timeout: float = 300.0):
        return _orig_start(self, timeout=timeout)

    _bp.BaseProducer.start = _start_long
    sys.path.insert(0, str(BASE))
    import ablate_loop as al

    task = [(0.0, DUTY)] if mode != "reference" else []
    res = al.run_mode(mode, seed=seed, task=task, scene=scene)
    res["scene"] = scene
    res["seed"] = seed
    Path(outfile).write_text(json.dumps(_metrics(res)), encoding="utf-8")


def _worker_main():
    _, scene, mode, seed, outfile = sys.argv
    _worker(scene, mode, int(seed), outfile)


def _purge_orphan_producers():
    script = ("$all = Get-CimInstance Win32_Process; "
              "$alive = @{}; "
              "foreach ($p in $all) { if ($p.ProcessId -gt 0) { $alive[$p.ProcessId] = $true } }; "
              "$all | Where-Object { $_.CommandLine -like '*cl-data-producer*' } | "
              "Where-Object { -not $alive.ContainsKey($_.ParentProcessId) } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, timeout=120)


def run_one(scene, mode, seed, idx=0, attempts: int = 3):
    outfile = WDIR / ("terrain_worker_%d_%s_%s_%d.json" % (idx, scene, mode, seed))
    if outfile.exists():
        outfile.unlink()
    for attempt in range(1, attempts + 1):
        _purge_orphan_producers()
        try:
            subp = subprocess.run(
                [PY, str(PROBE), scene, mode, str(seed), str(outfile)],
                capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            last = "timeout"
        else:
            if subp.returncode == 0 and outfile.exists():
                m = json.loads(outfile.read_text(encoding="utf-8"))
                print("  %-9s seed=%-2d fallen=%-5s dist=%.2fm "
                      "vx_last=%.2f h_std=%.4f" % (
                          mode, seed, m["fallen"], m["distance_m"],
                          m["vx_last"], m["h_std"]), flush=True)
                return m
            last = (subp.stderr or "")[-400:] or "exit %d" % subp.returncode
            if not subp.stderr and subp.stdout:
                last = (subp.stdout or "")[-400:]
        print("    [attempt %d/%d failed (%s)]" % (attempt, attempts, last[-160:]),
              flush=True)
        time.sleep(2.0)
    raise RuntimeError("run failed after %d attempts: %s %s seed=%d" % (
        attempts, scene, mode, seed))


def _parse_args():
    """--out FILE --scenes a,b --modes a,b --seeds a,b   (targeted battery)."""
    args = sys.argv[1:]
    if args and args[0].startswith("--"):
        kwargs = {"out": str(RES / "terrain_probe.json")}
        i = 0
        while i < len(args):
            key = args[i]
            val = args[i + 1] if i + 1 < len(args) else ""
            if key == "--out":
                kwargs["out"] = val
            elif key == "--scenes":
                kwargs["scenes"] = [s for s in val.split(",") if s]
            elif key == "--modes":
                kwargs["modes"] = [s for s in val.split(",") if s]
            elif key == "--seeds":
                kwargs["seeds"] = [int(s) for s in val.split(",") if s]
            else:
                raise SystemExit("unknown flag %s" % key)
            i += 2
        return kwargs
    kwargs = {"out": str(RES / "terrain_probe.json"),
              "scenes": list(SCENES), "modes": list(MODES),
              "seeds": list(SEEDS), "flat": True}
    return kwargs


def main():
    cfg = _parse_args()
    scenes = cfg["scenes"]
    modes = cfg["modes"]
    seeds = cfg["seeds"]
    outfile = Path(cfg["out"])
    t0 = time.monotonic()
    _purge_orphan_producers()
    print("Terrain probe: scenes=%s modes=%s seeds=%s duty=%.2f m/s -> %s" % (
        scenes, modes, seeds, DUTY, outfile.name), flush=True)
    all_rows = []
    idx = 0
    for scene in scenes:
        print("scene=%s" % scene, flush=True)
        for mode in modes:
            for seed in seeds:
                all_rows.append(run_one(scene, mode, seed, idx=idx))
                idx += 1
    flat_rows = []
    if cfg.get("flat"):
        for seed in seeds:
            print("scene=flat", flush=True)
            all_rows.append(run_one("flat", "reference", seed, idx=idx))
            flat_rows.append(all_rows[-1])
            idx += 1

    out = {"duty": DUTY, "scenes": scenes, "modes": modes,
           "seeds": list(seeds), "rows": all_rows,
           "flat_reference": flat_rows,
           "wall_min": round((time.monotonic() - t0) / 60.0, 1)}
    with open(outfile, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    _figure(all_rows, flat_rows, outfile)
    print("saved %s" % outfile.name, flush=True)


def _figure(rows, flat_rows, outfile=None):
    outfile = outfile or RES / "terrain_probe.json"
    scenes = sorted({r["scene"] for r in rows if r["scene"] != "flat"})
    modes = sorted({r["mode"] for r in rows}, key=lambda m: (
        "reference", "neural", "zero", "random").index(m))
    colors = {"reference": "#3498db", "neural": "#1e7d32", "zero": "#c0392b",
              "random": "#8e44ad"}
    fig, axes = plt.subplots(1, len(scenes), figsize=(5.2 * len(scenes), 4.6),
                             squeeze=False)
    for ax, scene in zip(axes.flat, scenes):
        srows = [r for r in rows if r["scene"] == scene]
        x = np.arange(len(modes))
        dist = [np.mean([r["distance_m"] for r in srows if r["mode"] == m])
                for m in modes]
        falls = [100.0 * np.mean([r["fallen"] for r in srows if r["mode"] == m])
                 for m in modes]
        fvar = [np.std([r["distance_m"] for r in srows if r["mode"] == m])
                for m in modes]
        ax.bar(x - 0.19, dist, width=0.38, yerr=fvar, capsize=3,
               color=[colors[m] for m in modes])
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_title("%s  (fall %s)" % (
            scene, "/".join("%d%%" % f for f in falls)))
        ax.set_ylabel("mean forward distance (m)")
        ax.grid(alpha=0.3)
        for xi, d in zip(x, dist):
            ax.annotate("%.1f" % d, (xi - 0.19, d), ha="center",
                        va="bottom", fontsize=8)
    fig.text(0.01, 0.965,
             "Terrain robustness under the CL1 contract (12 s runs, 0.5 m/s duty)  |  "
             "flat open-loop: %s m" % (
                 "/".join("%.1f" % r["distance_m"] for r in flat_rows)),
             fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat",
                                    alpha=0.9))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for m, c in colors.items()
               if m in modes]
    fig.legend(handles, [m for m in modes], loc="lower center", ncol=4,
               frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    png = outfile.with_suffix(".png")
    fig.savefig(png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        _worker_main()
    else:
        main()