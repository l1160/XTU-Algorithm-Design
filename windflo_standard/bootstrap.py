"""Standalone installs use normal site-packages; reuse existing local libs here."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parent
for deps in (ROOT.parent / 'windfarm_proto' / '.deps', ROOT / '.deps'):
    if deps.is_dir():
        sys.path.insert(0, str(deps))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
