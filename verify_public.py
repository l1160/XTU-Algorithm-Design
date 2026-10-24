"""Verify code/data relocation using fixed layouts and small interface checks."""
from pathlib import Path
import os
import sys
import json
import hashlib
import subprocess
import datetime
import importlib.metadata
import traceback
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"verification"
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg",PYTHONUTF8="1",PYTHONIOENCODING="utf-8")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def run():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/"PUBLIC_MANIFEST.json").read_text())
    for i,r in enumerate(manifest,1):
        path=(ROOT/r["path"]).resolve();assert path.is_relative_to(ROOT)
        assert sha(path)==r["sha256"],str(path)
        if i%1000==0:print("MANIFEST",i,len(manifest),flush=True)
    plans=[]
    for name in ("repair","terrain"):
        console_python=str(Path(sys.executable).with_name("python.exe"))
        response=subprocess.run([console_python,str(ROOT/"reproduce.py"),"--campaign",name,"--dry-run"],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW,check=True)
        plans.append(json.loads(response.stdout))
    sys.path.insert(0,str(ROOT/"windfarm_delivery"));sys.path.insert(0,str(ROOT/"windfarm_terrain_v5"))
    import bootstrap_v5
    import numpy as np
    from model_terrain import TerrainFarm
    from model_v4 import Farm
    from terrain import Terrain
    from fast_audit import independent_check
    from variants import Engine,DiagnosticFarm,configure
    fixed=[]
    for scenario in ("obs_00","obs_01"):
        folder=ROOT/"reference_results/terrain_layouts"/f"{scenario}_ridge_M1600_N400"
        representations=json.loads((folder/"representative_layouts.json").read_text(encoding="utf-8"))["layouts"]
        farm=TerrainFarm(scenario,1600,400,terrain_kind="ridge")
        for criterion,saved in representations.items():
            selected=np.array(saved["selected_candidate_indices"])
            keys=np.linspace(.6,.99,1600);keys[selected]=np.linspace(.01,.4,400)
            _,_,cv=farm.evaluate(keys);metrics=farm.records[-1][1]
            assert cv==0 and independent_check(farm.scene,farm.xy[selected])
            np.testing.assert_array_equal(np.frombuffer(farm.records[-1][0],dtype="<u2"),selected)
            expected=[saved["AEP_GWh"],saved["LCOE_USD2022_MWh"],saved["effective_area_km2"],saved["connection_length_m"],saved["gross_area_km2"]]
            np.testing.assert_allclose(metrics[:5],expected,rtol=1e-10,atol=1e-7)
            fixed.append(dict(scenario=scenario,terrain="ridge",criterion=criterion,matched=True))
        del farm
    for scenario in ("obs_00","obs_01"):
        for m in (800,1600):
            run_name=f"{scenario}_M{m}_N400_CCwake_base_s301"
            with np.load(ROOT/"verification_fixtures/flat"/run_name/"evaluations.npz") as z:
                ids=z["pareto_request_indices"];i=ids[np.argmax(z["metrics"][ids,0])]
                selected=z["selected"][i].copy();expected=z["metrics"][i].copy()
            farm=Farm(scenario,m,400)
            keys=np.linspace(.6,.99,m);keys[selected]=np.linspace(.01,.4,400)
            _,_,cv=farm.evaluate(keys)
            assert cv==0 and independent_check(farm.scene,farm.xy[selected])
            np.testing.assert_allclose(farm.records[-1][1],expected,rtol=1e-10,atol=1e-7)
            fixed.append(dict(scenario=scenario,terrain="flat_v4",M=m,criterion="CCwake_seed301_max_energy",matched=True))
            del farm
    for kind in ("flat","rolling","ridge"):
        for resolution in (25,50,100):
            regenerated=Terrain(kind,resolution,15.)
            with np.load(ROOT/"windfarm_terrain_v5/data"/f"{kind}_{resolution}m"/"dem.npz") as z:
                np.testing.assert_array_equal(regenerated.z,z["elevation_m"])
                np.testing.assert_array_equal(regenerated.forbidden_cells,z["forbidden_slope_cells"])
    # These 3 short executions test callable algorithms in the moved tree.
    # They are not extra performance replicates and receive no inferential analysis.
    smoke=[];reference=None
    for method in ("CCGDE3_wake","LMOCSO","DGEA"):
        configure({});farm=DiagnosticFarm("obs_00",800,400)
        engine=Engine(farm,method,913,budget=120,population=100)
        initial=[(r[0],r[1].tolist(),r[2]) for r in farm.records[:100]]
        if reference is None:reference=initial
        assert initial==reference
        while engine.remaining:engine.step()
        assert len(farm.records)==120
        assert all(r[2]==0 and np.isfinite(r[1]).all() for r in farm.records)
        smoke.append(dict(method=method,requests=120,scope="functional_only"))
        del farm,engine
    modules={}
    for name in ("bootstrap","bootstrap_v4","bootstrap_v5","evaluator","model","model_v4","model_terrain","terrain","geometry","physical","algorithms","variants","fast_audit"):
        path=Path(sys.modules[name].__file__).resolve()
        assert path.is_relative_to(ROOT),f"Imported original project source: {name}: {path}"
        modules[name]=str(path)
    packages={name:importlib.metadata.version(name) for name in ("numpy","scipy","shapely","pymoo","matplotlib","dill")}
    result=dict(status="passed",finished_at=datetime.datetime.now().astimezone().isoformat(),package_root=str(ROOT),
        manifest_files_verified=len(manifest),fixed_layouts_reproduced=len(fixed),fixed_layouts=fixed,
        synthetic_DEM_grids_reproduced=9,full_campaign_dry_plans=plans,algorithm_interface_checks=smoke,
        project_module_origins=modules,dependency_versions=packages,
        dependency_environment="Existing locally installed third-party runtime; not a clean-computer installation test",
        dependency_path_environment=os.environ.get("PYTHONPATH",""),new_performance_experiments=0)
    (OUT/"RELOCATION_VERIFIED.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("RELOCATION VERIFIED",len(fixed),"fixed layouts; nine DEM grids; three functional algorithm checks",flush=True)


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    sys.stdout=sys.stderr=(OUT/("verify_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".log")).open("a",encoding="utf-8",buffering=1)
    sys.stdin=open(os.devnull)
    try:run()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        (OUT/("ERROR_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")).write_text(json.dumps(dict(error=error),ensure_ascii=False,indent=2),encoding="utf-8")
        raise
