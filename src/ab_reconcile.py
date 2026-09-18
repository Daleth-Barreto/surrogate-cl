"""A/B reconciliation probe: is baseline-neural tracking RMSE reproducible
today (current files), and which set of numbers is canonical?

Runs seed 7 and 91 twice in separate processes would be ideal; here we run
sequentially in one process, which is how the battery itself runs, then we also
re-run seed 7 a second time to check within-process reproducibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_tracking as T


def show(tag, seed):
    r = T.one_run("neural", seed, T.PROFILES["baseline"])
    print("%-22s seed=%-3d rmse=%.4f cmd_mean=%.3f nrmse=%s" % (
        tag, seed, r["rmse"], float(sum(r["cmd_series"]) / len(r["cmd_series"])),
        r["nrmse"]))


if __name__ == "__main__":
    show("AB seed7 run1", 7)
    show("AB seed7 run2", 7)
    show("AB seed91 run1", 91)
    show("AB seed91 run2", 91)