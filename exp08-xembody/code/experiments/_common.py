"""Shared helpers for the experiment scripts (makes `xembody` importable from anywhere)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # figures are saved to results/, never shown in a window

ROBOT_COLORS = {"kuka_iiwa": "#e07b00", "franka_panda": "#4a4a4a", "xarm6": "#1f6fb5", "kuka_mini": "#b04fa0"}

# --- my additions: name + timestamp log, and a personal seed offset -----------
REPO = os.path.dirname(os.path.dirname(ROOT))          # aisr-lab/
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from student import SEED, start_run  # noqa: E402

SEED_OFFSET = 100 * SEED          # shifts every episode seed, so the cubes/goals are my own draws
start_run("exp08-xembody/" + os.path.basename(sys.argv[0]))
