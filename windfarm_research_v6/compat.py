"""Load the frozen source adaptations without modifying or monkey-patching them."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "windfarm_terrain_v5"))
import bootstrap_v5  # noqa: E402,F401
import algorithms as legacy  # noqa: E402
from model_terrain import TerrainFarm  # noqa: E402
from model import METRIC_COLUMNS, repair_keys  # noqa: E402
from physical import physical_power  # noqa: E402
from geometry import tree_length  # noqa: E402

# The historical strongest CC comparator is retained, including its wake groups.
# This is an explicitly adapted implementation, not MATLAB equivalence.
BASELINES = {"CCGDE3": "CCGDE3_wake", "LMOCSO": "LMOCSO", "DGEA": "DGEA"}
METHODS = tuple(BASELINES) + tuple(name + "_problem" for name in BASELINES)
