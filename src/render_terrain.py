"""Replay video of Fase 0.3 terrain runs (package artefact).

Records the exact closed-loop physics of a terrain run (the datasource appends a
new key `record_path` -> npz of per-sim-step qpos) and replays it under the
offscreen MuJoCo renderer into an mp4 via ffmpeg.

Job = (scene, mode, seed). Each job runs in a FRESH worker subprocess (same
isolation/purge pattern as terrain_probe.py) because neural runs need the cl
producer; the replay+ffmpeg step happens inside the same worker.

Usage:
  python render_terrain.py                          # default battery
  python render_terrain.py --jobs rough:neural:13,curb:reference:1
  python render_terrain.py --out results/terrain_videos --width 960
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
RES = Path("C:/Proyectos/papers/surrogate_cl/results")
WDIR = Path("C:/Users/aland/AppData/Local/Temp/opencode")
CONF = Path("C:/Proyectos/papers/surrogate_cl/config")
PY = sys.executable
FFMPEG = shutil.which("ffmpeg")

DUTY = 0.5
JOBS = ["rough:reference:13", "rough:neural:13",
        "curb:reference:1", "curb:neural:1"]
WIDTH, HEIGHT, FPS = 960, 540, 60


def _compile_with_camera(scene_xml: str, width: int, height: int):
    """Return (model, camera_name, camera_rotation, camera_eye, meshdir_file) by
    composing the scene + embedded robot XML into one string with a fixed camera.

    The scene's `<include>` is resolved textually (MuJoCo merges the included
    document's elements, including its worldbody). Camera framing is explicit and
    deterministic; the returned transformation lets the caller verify ON-SCREEN
    presence numerically (see _proj_check)."""
    import re
    import numpy as np

    base = Path(scene_xml)
    inc = (base.parent / "g1_12dof.xml").read_text(encoding="utf-8")
    inc = re.sub(r"<mujoco[^>]*>", "", inc, count=1).replace("</mujoco>", "")
    body = base.read_text(encoding="utf-8").replace(
        '<include file="g1_12dof.xml"/>', inc)

    mdir = re.search(r'meshdir="([^"]+)"', inc).group(1).replace("\\", "/")
    cam_map = {"rough": ((3.0, -12.0, 4.6), (4.0, 0.0, 1.0)),
               "curb": ((1.8, -9.0, 3.8), (3.0, 0.0, 0.9)),
               "ramp": ((1.8, -9.0, 3.8), (3.0, 0.0, 0.9))}
    eye = np.asarray(cam_map[Path(scene_xml).stem.split("_")[-1]][0], dtype=float)
    target = np.asarray(cam_map[Path(scene_xml).stem.split("_")[-1]][1],
                        dtype=float)
    fwd = eye - target
    fwd /= np.linalg.norm(fwd)
    x = np.cross([0.0, 0.0, 1.0], fwd)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    cam = ('<camera name="vid" pos="%.3f %.3f %.3f" '
           'xyaxes="%.4f %.4f %.4f %.4f %.4f %.4f" fovy="45"/>'
           % (*eye, *x, *y))
    idx = body.rfind("</worldbody>")
    body = body[:idx] + cam + body[idx:]
    body = body.replace(
        "<mujoco",
        '<mujoco><visual><global offwidth="%d" offheight="%d"/></visual>'
        % (int(width), int(height)), 1)
    assets_dir = Path(mdir)
    assets = {p.name: p.read_bytes()
              for p in assets_dir.iterdir() if p.is_file()}
    import mujoco
    model = mujoco.MjModel.from_xml_string(body, assets=assets)
    return model, "vid", np.vstack([x, y, fwd]), eye


def _scene_xml(scene: str) -> str:
    if scene == "flat":
        cfg = CONF / "g1.yaml"
    else:
        cfg = CONF / ("g1_%s.yaml" % scene)
    with open(cfg, encoding="utf-8") as fp:
        return str(yaml.safe_load(fp)["xml_path"])


def _purge_orphan_producers():
    script = ("$all = Get-CimInstance Win32_Process; "
              "$alive = @{}; "
              "foreach ($p in $all) { if ($p.ProcessId -gt 0) { $alive[$p.ProcessId] = $true } }; "
              "$all | Where-Object { $_.CommandLine -like '*cl-data-producer*' } | "
              "Where-Object { -not $alive.ContainsKey($_.ParentProcessId) } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
    subprocess.run(["powershell", "-NoProfile", "-Command", script],
                   capture_output=True, timeout=120)


def _worker(scene, mode, seed, npz_path, tmp_dir, width, height, fps,
            reuse=False):
    import cl._sim._base_producer as _bp

    _orig_start = _bp.BaseProducer.start

    def _start_long(self, timeout: float = 300.0):
        return _orig_start(self, timeout=timeout)

    _bp.BaseProducer.start = _start_long
    if not (reuse and Path(npz_path).exists()):
        sys.path.insert(0, str(BASE))
        import ablate_loop as al

        task = [(0.0, DUTY)] if mode != "reference" else []
        al.run_mode(mode, seed=seed, task=task, scene=scene,
                    record_path=npz_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np

    z = _load_npz_retry(npz_path)
    qpos = z["qpos"]
    dt = float(z["dt"])
    model, cam_name, rot, eye = _compile_with_camera(_scene_xml(scene),
                                                     int(width), int(height))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    step = max(1, int(round(1.0 / dt / float(fps))))
    eff_fps = 1.0 / (dt * step)
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    proj_min = 1.0
    for i in range(0, qpos.shape[0], step):
        data.qpos[...] = qpos[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_name)
        img = renderer.render()
        plt.imsave(tmp / ("f%05d.png" % n), img)
        proj_min = min(proj_min, _proj_margin(model, data, rot, eye,
                                              aspect=width / float(height)))
        n += 1
    renderer.close()
    out = Path(npz_path).with_suffix(".mp4")
    subprocess.run([
        FFMPEG, "-y", "-framerate", "%.4f" % eff_fps,
        "-i", str(tmp / "f%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", str(out),
    ], check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"scene": scene, "mode": mode, "seed": seed,
            "frames": n, "fps_r": round(eff_fps, 3), "sim_end": float(z["t"][-1]),
            "proj_margin_min": round(float(proj_min), 3),
            "mp4": str(out)}


def _load_npz_retry(path):
    """The datasource subprocess saves the npz asynchronously; retry until a
    complete file is readable (mirrors read_telemetry's retry pattern). Arrays
    are copied into memory inside the retry so a replaced file cannot surface a
    BadZipFile on first access."""
    import time as _time
    import numpy as np
    last = None
    for _ in range(40):
        try:
            z = np.load(path)
            out = {"dt": float(z["dt"])}
            out["qpos"] = np.asarray(z["qpos"], dtype=np.float64).copy()
            out["t"] = np.asarray(z["t"], dtype=np.float64).copy()
            return out
        except Exception as e:
            last = e
            _time.sleep(0.5)
    raise RuntimeError("npz not readable after retries: %s (%s)" % (path, last))


def _proj_margin(model, data, rot, eye, fovy_deg=45.0, aspect=None):
    """Smallest distance (normalized NDC, 1 = frustum border) of the pelvis to
    the frustum border; 0 when behind the camera (cameras look down -z)."""
    import numpy as np
    p = np.asarray(data.qpos[:3], dtype=float)
    pc = rot @ (p - eye)
    if pc[2] >= 0.0:
        return 0.0
    z = -pc[2]
    t = float(np.tan(np.radians(fovy_deg) / 2.0))
    a = aspect if aspect else model.vis.global_.offwidth / float(model.vis.global_.offheight)
    ndc_x = pc[0] / (a * z * t)
    ndc_y = pc[1] / (z * t)
    return max(0.0, float(min(1.0 - abs(ndc_x), 1.0 - abs(ndc_y))))


def _worker_main():
    scene, mode, seed, npz_path, tmp_dir, width, height, fps = sys.argv[1:9]
    reuse = len(sys.argv) == 10 and sys.argv[9] == "reuse"
    if not (reuse and Path(npz_path).exists()):
        Path(npz_path).unlink(missing_ok=True)
    info = _worker(scene, mode, int(seed), npz_path, tmp_dir,
                   int(width), int(height), int(fps), reuse=reuse)
    Path(npz_path).with_suffix(".json").write_text(
        json.dumps(info), encoding="utf-8")


def run_one(scene, mode, seed, outdir, attempts=3, reuse=False):
    npz_path = outdir / ("%s_%s_s%d.npz" % (scene, mode, seed))
    if not (reuse and npz_path.exists()):
        npz_path.unlink(missing_ok=True)
        npz_path.with_suffix(".json").unlink(missing_ok=True)
    tmp = WDIR / ("vid_%s_%s_%d" % (scene, mode, seed))
    argv10 = [PY, str(Path(__file__).resolve()),
              scene, mode, str(seed), str(npz_path), str(tmp),
              str(WIDTH), str(HEIGHT), str(FPS)]
    if reuse:
        argv10.append("reuse")
    for attempt in range(1, attempts + 1):
        _purge_orphan_producers()
        try:
            subp = subprocess.run(argv10, capture_output=True, text=True,
                                  timeout=1800)
        except subprocess.TimeoutExpired:
            last = "timeout"
        else:
            meta = npz_path.with_suffix(".json")
            if subp.returncode == 0 and meta.exists():
                info = json.loads(meta.read_text(encoding="utf-8"))
                print("  %-9s s%-2d: %d frames @%.1f fps -> %s" % (
                    mode, seed, info["frames"], info["fps_r"],
                    Path(info["mp4"]).name), flush=True)
                return info
            last = (subp.stderr or "")[-400:] or "exit %d" % subp.returncode
            if not subp.stderr and subp.stdout:
                last = (subp.stdout or "")[-400:]
        print("    [attempt %d/%d failed (%s)]" % (attempt, attempts, last[-160:]),
              flush=True)
        time.sleep(2.0)
    raise RuntimeError("render failed: %s %s %d" % (scene, mode, seed))


def main():
    args = sys.argv[1:]
    outdir = RES / "terrain_videos"
    jobs = list(JOBS)
    i = 0
    while i < len(args):
        key = args[i]
        val = args[i + 1] if i + 1 < len(args) else ""
        if key == "--jobs":
            jobs = [j for j in val.split(",") if j]
        elif key == "--out":
            outdir = Path(val)
        elif key == "--width":
            global WIDTH
            WIDTH = int(val)
        elif key == "--reuse":
            pass
        else:
            raise SystemExit("unknown flag %s" % key)
        i += 2
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    _purge_orphan_producers()
    print("render_terrain: jobs=%s -> %s" % (jobs, outdir), flush=True)
    clips = []
    reuse = "--reuse" in args
    for job in jobs:
        scene, mode, seed = job.split(":")
        clips.append(run_one(scene, mode, int(seed), outdir, reuse=reuse))
    with open(outdir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump({"jobs": jobs, "clips": clips,
                   "wall_min": round((time.monotonic() - t0) / 60.0, 1)}, fp,
                  indent=2, ensure_ascii=False)
    print("saved %s (%.1f min)" % (outdir / "manifest.json",
                                   (time.monotonic() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    if not (len(sys.argv) >= 2 and sys.argv[1].startswith("--")):
        _worker_main()
    else:
        main()