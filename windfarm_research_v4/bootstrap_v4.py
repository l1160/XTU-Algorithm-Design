from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
V3=HERE.parent/"windfarm_study_v3"
sys.path.insert(0,str(V3))
import bootstrap
sys.path.insert(0,str(HERE))
