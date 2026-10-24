"""Use the same frozen numerical/runtime dependencies as V4."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
V4=ROOT.parent/"windfarm_research_v4"
sys.path.insert(0,str(V4))
import bootstrap_v4
sys.path.insert(0,str(ROOT))
