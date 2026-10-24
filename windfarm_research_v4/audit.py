"""Independent physics/economic/operator checks before new optimization."""
import bootstrap_v4
from pathlib import Path
import json,math,datetime,csv,hashlib,msvcrt,traceback
import numpy as np
from scipy.integrate import quad
from scipy.spatial.distance import cdist
from evaluator import Scenario,wake_squared
from physical import physical_power
from model import Economics,rooted_tree_length,repair_keys
from model_v4 import downstream_wake_squared
import algorithms as A
from pymoo.util.ref_dirs import get_reference_directions
HERE=Path(__file__).resolve().parent;OUT=HERE/"results/audit"
checks=[]
def save(name,obj):
    p=OUT/name;t=p.with_suffix(".tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");t.replace(p)
def check(name,passed,**details):
    checks.append(dict(name=name,passed=bool(passed),**details))
    save("checks.json",checks)
    if not passed:raise AssertionError(name)
    print("PASS",name,flush=True)
def scalar_wake(scene,xy):
    result=np.zeros((24,len(xy),len(xy)))
    for q in range(24):
        theta=math.radians(q*15+7.5);u=(math.cos(theta),math.sin(theta))
        for i,(x,y) in enumerate(xy):
            for j,(xx,yy) in enumerate(xy):
                t=(x-xx)*u[0]+(y-yy)*u[1]
                lateral=abs((x-xx)*u[1]-(y-yy)*u[0])
                if i!=j and t>0 and lateral<scene.radius+scene.decay*t:
                    result[q,i,j]=((1-math.sqrt(1-scene.thrust))/(1+scene.decay*t/scene.radius)**2)**2
    return result
def prim_length(xy,station):
    p=np.unique(np.vstack([xy,station]),axis=0);d=cdist(p,p)
    used=np.zeros(len(p),bool);best=np.full(len(p),np.inf);best[0]=0;total=0.
    for _ in p:
        i=int(np.argmin(np.where(used,np.inf,best)));total+=best[i];used[i]=True;best=np.minimum(best,d[i])
    return total
def scalar_apd(f,refs,theta):
    z=f-f.min(axis=0);norm=np.linalg.norm(z,axis=1)
    u=refs/np.linalg.norm(refs,axis=1)[:,None]
    gamma=[]
    for i in range(len(u)):
        vals=[math.acos(float(np.clip(np.dot(u[i],u[j]),-1,1))) if i!=j else math.pi/2 for j in range(len(u))]
        gamma.append(max(min(vals),1e-12))
    angle=np.array([[0. if norm[i]<=1e-14 else math.acos(float(np.clip(np.dot(z[i],v)/norm[i],-1,1))) for v in u] for i in range(len(f))])
    association=angle.argmin(axis=1);ids=[]
    for j in sorted(set(association)):
        cand=np.where(association==j)[0];score=[(1+f.shape[1]*theta*angle[i,j]/gamma[j])*norm[i] for i in cand]
        ids.append(int(cand[np.argmin(score)]))
    return np.array(ids)
def main():
    scene=Scenario.load(bootstrap_v4.V3.parent/"windflo_standard/data/Scenarios/obs_00.xml")
    u=np.array([math.cos(math.radians(7.5)),math.sin(math.radians(7.5))])
    xy=np.array([[1000.,1000.],[1000.,1000.]+308*u])
    old=wake_squared(scene,xy);new=downstream_wake_squared(scene,xy)
    reverse=dict(direction_center_deg=7.5,spacing_m=308.,legacy_upstream_deficit=float(np.sqrt(old[0,0,1])),
                 corrected_upstream_deficit=float(np.sqrt(new[0,0,1])),downstream_deficit=float(np.sqrt(new[0,1,0])))
    save("upstream_counterexample.json",reverse)
    check("Counterexample: legacy upstream wake exists at legal spacing",old[0,0,1]>0)
    check("Correction removes upstream while preserving downstream deficit",new[0,0,1]==0 and np.isclose(new[0,1,0],old[0,1,0]))
    rng=np.random.default_rng(20261020)
    random_xy=rng.uniform([0,0],[7000,14000],(50,2))
    corrected=downstream_wake_squared(scene,random_xy);scalar=scalar_wake(scene,random_xy)
    check("Independent scalar downstream wake matches vectorized",np.allclose(corrected,scalar,atol=1e-14,rtol=1e-12))
    legacy=wake_squared(scene,random_xy)
    angles=np.deg2rad(np.arange(24)*15+7.5)
    delta=random_xy[:,None,:]-random_xy[None,:,:]
    positive=np.einsum("ijd,qd->qij",delta,np.c_[np.cos(angles),np.sin(angles)])>0
    check("Downstream values unchanged relative to legacy",np.allclose(corrected[positive],legacy[positive],atol=1e-13))
    check("All upstream and self terms exactly zero",np.all(corrected[~positive]==0))
    # Independent quadrature of piecewise power times Weibull density.
    errors=[]
    for q in [0,5,13,23]:
        for loss in [0.,.15,.5,.9,1.]:
            scale=scene.c[q]*(1-loss);shape=scene.shape[q]
            if scale==0:expected=0.
            else:
                def density(v):return shape/scale*(v/scale)**(shape-1)*math.exp(-(v/scale)**shape)
                lo=max(scene.cut_in,500/140.86)
                expected=quad(lambda v:(140.86*v-500)*density(v),lo,scene.rated_speed,epsabs=1e-8)[0]
                expected+=quad(lambda v:scene.rated_kw*density(v),scene.rated_speed,scene.cut_out,epsabs=1e-8)[0]
                expected*=scene.omega[q]/scene.omega.sum()
            actual=physical_power(scene,np.full((24,1),loss))[q,0]
            errors.append(float(abs(actual-expected)))
    check("Independent quadrature agrees with exact power integral",max(errors)<1e-7,max_abs_kw=max(errors))
    # Algebra/selection operators compared to scalar reference equations.
    lx=rng.random((8,31));lv=rng.normal(size=(8,31));wx=rng.random((8,31));r1=rng.random((8,1));r2=rng.random((8,1))
    xx,vv=A.swarm_step(lx,lv,wx,r1,r2);expected=np.zeros_like(lx)
    for i in range(8):
        for j in range(31):
            velocity=r1[i,0]*lv[i,j]+r2[i,0]*(wx[i,j]-lx[i,j])
            expected[i,j]=lx[i,j]+velocity+r1[i,0]*(velocity-lv[i,j])
    check("LMOCSO particle equation versus independent scalar equation",np.allclose(xx,expected,rtol=0,atol=1e-14))
    f=rng.random((37,3));z=(f-f.min(0))/np.ptp(f,axis=0)
    expected=np.array([min(np.linalg.norm(np.maximum(z[j],z[i])-z[i]) for j in range(len(z)) if j!=i) for i in range(len(z))])
    check("LMOCSO shift-density scalar equivalence",np.allclose(A.sde(f),expected,atol=1e-14))
    refs=get_reference_directions("das-dennis",3,n_partitions=12)
    for trial in range(10):
        f=rng.random((55,3))*[1,3,9]
        pop=A.Pop(np.zeros((len(f),2)),f,np.zeros(len(f)),np.zeros((len(f),2)))
        check(f"APD scalar source-equation equivalence {trial}",np.array_equal(A.apd_ids(pop,refs,.36),scalar_apd(f,refs,.36)))
    # Large/zero-span corner guards are declared deviations, not source equality.
    check("SDE constant-column guard remains finite",np.isfinite(A.sde(np.c_[np.arange(10),np.ones(10),np.arange(10)[::-1]])).all())
    # Saved representative layouts: change only wake direction, recompute energy/cost.
    records=json.loads((bootstrap_v4.V3/"closeout/analysis/representative_layouts.json").read_text(encoding="utf-8"))
    impacts=[]
    for rec in records:
        case=rec["case"];scenario=case.split("_M")[0]
        scene=Scenario.load(bootstrap_v4.V3.parent/"windflo_standard/data/Scenarios"/(scenario+".xml"))
        pts=np.loadtxt(bootstrap_v4.V3/"closeout/analysis"/(case+"_layout.csv"),delimiter=",",skiprows=1)
        legacy=wake_squared(scene,pts);corrected=downstream_wake_squared(scene,pts)
        aep_old=float(physical_power(scene,np.sqrt(legacy.sum(2))).sum()*8760/1e6)
        aep_new=float(physical_power(scene,np.sqrt(corrected.sum(2))).sum()*8760/1e6)
        check(case+" archived AEP reproducible",np.isclose(aep_old,rec["energy_GWh"],rtol=1e-12))
        with np.load(bootstrap_v4.V3/"results/comparison_v3"/rec["run"]/"evaluations.npz") as d:
            metrics=d["metrics"][rec["request_index_zero_based"]]
        length=prim_length(pts,[3500,0])
        check(case+" independent Prim MST",np.isclose(length,metrics[3],rtol=1e-12))
        annual=600000*(.0673*(1750-64+64*length/metrics[9])+41)
        check(case+" independent economic equation",np.isclose(annual,metrics[7],rtol=1e-12))
        row=dict(case=case,removed_directional_links=int(np.count_nonzero((legacy>0)&(corrected==0))),
                 total_legacy_links=int(np.count_nonzero(legacy)),old_AEP_GWh=aep_old,new_AEP_GWh=aep_new,
                 relative_AEP_change=aep_new/aep_old-1,old_LCOE=rec["LCOE_USD2022_MWh"],
                 corrected_LCOE=annual/(1000*aep_new),area_unchanged_km2=rec["area_km2"])
        impacts.append(row);save("representative_impacts.json",impacts)
        print("IMPACT",case,row["relative_AEP_change"],flush=True)
    with (OUT/"representative_impacts.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,list(impacts[0]));w.writeheader();w.writerows(impacts)
    sig={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"audit.py",HERE/"model_v4.py",bootstrap_v4.V3/"algorithms.py",bootstrap_v4.V3/"model.py"]}
    save("DONE.json",dict(status="complete",checks_passed=True,checks=len(checks),new_optimization_runs=0,
                         findings=["V3 legacy cone admits upstream wake at legal minimum spacing",
                                   "Fixed-layout impact measured; old optimization ranking not validated under corrected model"],
                         matlab_octave_available=False,operator_check_scope="independent scalar Python equations, not MATLAB execution",
                         sources={"downstream_mask":"https://raw.githubusercontent.com/NatLabRockies/floris/main/floris/core/wake_velocity/jensen.py",
                                  "algorithm_reference_commit":"d25e65d1ffba58dbf4d7e1b5259786187d12968a"},
                         fingerprints=sig,finished_at=datetime.datetime.now().astimezone().isoformat()))
    print("AUDIT COMPLETE",flush=True)
if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b"0");lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit("already_running")
        try:main()
        except Exception:
            save("ERROR.json",dict(traceback=traceback.format_exc()))
            raise
