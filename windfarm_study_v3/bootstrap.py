from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parent
STANDARD = ROOT.parent / 'windflo_standard'
sys.path.insert(0, str(STANDARD))
sys.path.insert(0, str(ROOT.parent / 'windfarm_proto' / '.deps'))
# Local study modules take precedence over same-named files in the earlier
# integration project; only its evaluator/data are intentionally reused.
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
