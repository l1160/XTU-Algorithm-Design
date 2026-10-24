"""Synthetic-terrain geometric extension; no orographic wind-field speedup."""
import bootstrap_v5
import numpy as np
from shapely.geometry import MultiPoint
from model import Farm as V3Farm,Economics,repair_keys,independent_set_certificate,METRIC_COLUMNS
from model_v4 import Farm as V4Farm
from evaluator import Scenario
from physical import physical_power
from terrain import Terrain,terrain_candidates
from geometry import distance_matrix,tree_length

def wake_squared_3d(scene,xy,hub_z):
    xy=np.asarray(xy,float);hub_z=np.asarray(hub_z,float)
    dx=xy[:,0,None]-xy[None,:,0];dy=xy[:,1,None]-xy[None,:,1]
    dz=hub_z[:,None]-hub_z[None,:]
    out=np.empty((24,len(xy),len(xy)))
    for q in range(24):
        angle=np.deg2rad(q*15+7.5)
        downstream=dx*np.cos(angle)+dy*np.sin(angle)
        lateral=-dx*np.sin(angle)+dy*np.cos(angle)
        active=(downstream>0)&(lateral*lateral+dz*dz<(scene.radius+scene.decay*downstream)**2)
        distance=np.maximum(downstream,0)
        deficit=(1-np.sqrt(1-scene.thrust))/(1+scene.decay*distance/scene.radius)**2
        out[q]=np.where(active,deficit**2,0)
        np.fill_diagonal(out[q],0)
    return out

class TerrainFarm(V4Farm):
    def __init__(self,scenario="obs_00",m=800,n=400,seed=20261012,economics=None,
                 terrain_kind="flat",resolution=50.,slope_limit_deg=15.,options=None):
        self.scene=Scenario.load(bootstrap_v5.bootstrap_v4.V3.parent/"windflo_standard/data/Scenarios"/(scenario+".xml"))
        self.m,self.n=int(m),int(n)
        assert 2<=self.n<self.m<65536
        self.terrain=Terrain(terrain_kind,resolution,slope_limit_deg)
        self.options=dict(options or {});self.diagnostics=[]
        self.xy,self.candidate_metadata=terrain_candidates(self.scene,self.terrain,self.m,seed)
        self.economics=Economics(economics)
        self.records,self.cache=[],{}
        self.prepare()
        self.certificate=independent_set_certificate(self.conflict)
        if len(self.certificate)<self.n:raise ValueError("No sufficient terrain feasibility certificate")
    def prepare(self):
        # The cost anchor deliberately remains the V3/V4 flat reference.
        V3Farm.prepare(self)
        self.allowed=self.allowed.difference(self.terrain.excluded)
        self.scales[2]=self.allowed.area/1e6
        self.hub_height=80.  # Same relative hub height; cancels in pairwise differences.
        self.hub_z=self.terrain.sample(self.xy)+self.hub_height
        self.ground_dist=None;self.ground_cache_metadata=None
        if not self.terrain.allowed_mask(self.station[None])[0]:raise ValueError("Substation is on excluded slope")
    def ensure_wake(self):
        if self.wake is None:self.wake=wake_squared_3d(self.scene,self.xy,self.hub_z)
    def ensure_ground(self):
        if self.ground_dist is None:
            self.ground_points=np.vstack([self.xy,self.station])
            self.ground_dist,self.ground_cache_metadata=distance_matrix(self.terrain,self.ground_points)
    def __getstate__(self):
        state=super().__getstate__()
        for k in ["ground_dist","ground_points"]:state.pop(k,None)
        return state
    def evaluate(self,keys):
        original=np.clip(np.asarray(keys,float),0,1)
        raw=np.argsort(original,kind="stable")[:self.n]
        repaired,selected,raw_pairs,ok=repair_keys(keys,self.conflict,self.n)
        gaps=np.maximum(0,(self.scene.min_spacing-self.distance[np.ix_(selected,selected)])/self.scene.min_spacing)
        cv=float(gaps[np.triu_indices(self.n,1)].sum()/(self.n*(self.n-1)/2))
        packed=selected.astype("<u2").tobytes();hit=packed in self.cache
        if not hit:
            metrics=np.zeros(len(METRIC_COLUMNS))
            if cv==0:
                self.ensure_wake();self.ensure_ground()
                deficits=np.sqrt(self.wake[:,selected[:,None],selected[None,:]].sum(axis=2))
                energy=float(physical_power(self.scene,deficits).sum()*8760/1e6)
                length=tree_length(self.ground_dist,np.r_[selected,self.m],self.ground_points)
                cost=self.economics.compute(self.scene.rated_kw,self.n,length,self.reference_length,energy)
                hull=MultiPoint(self.xy[selected]).convex_hull
                gross=hull.area/1e6;area=hull.intersection(self.allowed).area/1e6
                metrics=np.array([energy,cost["lcoe"],area,length,gross,gross-area,
                                  cost["capex"],cost["annual_cost"],cost["electrical_capex"],self.reference_length])
                if not np.isfinite(metrics).all() or energy<=0:raise ValueError("Invalid terrain objectives")
            self.cache[packed]=(metrics,cv)
        metrics,cv=self.cache[packed]
        f=np.array([-metrics[0],metrics[1],metrics[2]])/self.scales if cv==0 else np.full(3,1e6)
        replacements=self.n-len(np.intersect1d(raw,selected))
        self.records.append((packed,metrics.copy(),cv,raw_pairs,ok,hit,replacements))
        self.diagnostics.append((float(np.mean(abs(repaired-original)>1e-12)),float(np.linalg.norm(repaired-original))))
        return (repaired if self.options.get("writeback",True) else original.copy()),f,cv
    def case_metadata(self):
        d=super().case_metadata()
        columns=list(METRIC_COLUMNS);columns[3]="terrain_surface_rooted_MST_m"
        d.update(model_version="v5.synthetic-terrain-geometry",terrain=self.terrain.metadata(),
                 elevation="Synthetic bilinear DEM supplied",metric_columns=columns,
                 wake_scope="3D hub-centre geometric Jensen proxy with horizontal imposed ambient wind; no terrain speedup/separation/flow-curvature solution",
                 nominal_relative_hub_height_m=self.hub_height,
                 wind_height_assumption="WindFLO distributions imposed uniformly at turbine hubs; measured reference height is not supplied",
                 connection_scope="MST of DEM-surface lengths along straight horizontal routes; no routing or electrical feasibility claim",
                 cost_anchor="Same flat reference length as V4; not recalibrated to cancel terrain-induced length changes",
                 surface_cache=self.ground_cache_metadata)
        return d
