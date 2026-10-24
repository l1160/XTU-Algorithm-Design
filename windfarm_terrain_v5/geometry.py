"""Cached terrain-surface route distances, never chord-through-ground lengths."""
import bootstrap_v5
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from pilot import lock_file
ROOT=Path(__file__).resolve().parent
def pair_surface_lengths(terrain,start,end,order=64):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    return integrate_pairs(terrain,np.asarray(start),np.asarray(end),(nodes+1)/2,weights/2)
def integrate_pairs(terrain,start,end,t,w):
    delta=end-start
    points=start[:,None,:]+t[None,:,None]*delta[:,None,:]
    _,g=terrain.sample(points,gradient=True)
    vertical=np.sum(g*delta[:,None,:],axis=2)
    return (np.sqrt(np.sum(delta**2,axis=1)[:,None]+vertical**2)*w[None,:]).sum(axis=1)
def distance_matrix(terrain,points):
    points=np.asarray(points,float)
    report=ROOT/"results/terrain_audit/surface_quadrature_orders.json"
    if not report.exists():raise RuntimeError("Terrain geometry audit must pass first")
    order=json.loads(report.read_text())[terrain.kind]["order"]
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key=hashlib.sha256(points.astype("<f8").tobytes()+terrain.digest.encode()+str(order).encode()+code_hash.encode()).hexdigest()
    folder=ROOT/"cache";folder.mkdir(exist_ok=True)
    path=folder/(key+".npy")
    if path.exists():return np.load(path,mmap_mode="r"),dict(path=str(path),order=order,key=key)
    while True:
        try:handle=lock_file(folder/(key+".lock"));break
        except OSError:time.sleep(2)
    with handle:
        if not path.exists():
            n=len(points);matrix=np.zeros((n,n),float)
            if terrain.kind=="flat" and np.ptp(terrain.z)==0:
                matrix[:]=cdist(points,points)
            else:
                nodes,weights=np.polynomial.legendre.leggauss(order);t=(nodes+1)/2;w=weights/2
                ii,jj=np.triu_indices(n,1)
                for start in range(0,len(ii),1024):
                    a=ii[start:start+1024];b=jj[start:start+1024]
                    lengths=integrate_pairs(terrain,points[a],points[b],t,w)
                    assert np.isfinite(lengths).all()
                    matrix[a,b]=lengths;matrix[b,a]=lengths
                    if start%(1024*64)==0:print("SURFACE CACHE",terrain.kind,n,start,len(ii),flush=True)
            tmp=folder/(key+".npy.tmp")
            with tmp.open("wb") as f:np.save(f,matrix)
            tmp.replace(path)
            (folder/(key+".json")).write_text(json.dumps(dict(terrain=terrain.metadata(),nodes=n,order=order,
                source_sha256=code_hash,route="straight XY segment lifted to bilinear DEM surface",
                length_unit="metre",scope="No routing detours, electrical capacity or construction validation")),encoding="utf-8")
            print("SURFACE CACHE COMPLETE",terrain.kind,n,flush=True)
    return np.load(path,mmap_mode="r"),dict(path=str(path),order=order,key=key)
def tree_length(matrix,indices,points):
    unique=np.unique(points[indices],axis=0,return_index=True)[1]
    selected=np.asarray(indices)[unique]
    return float(minimum_spanning_tree(matrix[np.ix_(selected,selected)]).sum())
