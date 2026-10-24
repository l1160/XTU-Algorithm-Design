"""Deterministic synthetic local-coordinate DEM, not measured topography."""
import bootstrap_v5
from pathlib import Path
import hashlib,json
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

SCENARIOS={
    "flat":dict(base_m=100.,rolling_m=0.,ridge_m=0.),
    "rolling":dict(base_m=100.,rolling_m=40.,ridge_m=0.),
    "ridge":dict(base_m=100.,rolling_m=40.,ridge_m=350.)
}
def analytic_height(x,y,kind):
    p=SCENARIOS[kind]
    # Boundary-compatible long waves plus a broad, localized oblique ridge.
    rolling=p["rolling_m"]*np.sin(2*np.pi*x/7000.)*np.sin(2*np.pi*y/14000.)
    angle=np.deg2rad(25.);dx=x-3800.;dy=y-7600.
    across=dx*np.cos(angle)+dy*np.sin(angle)
    along=-dx*np.sin(angle)+dy*np.cos(angle)
    ridge=p["ridge_m"]*np.exp(-.5*((across/600.)**2+(along/2500.)**2))
    return p["base_m"]+rolling+ridge

class Terrain:
    def __init__(self,kind="flat",resolution=50.,slope_limit_deg=15.,elevation=None):
        if kind not in SCENARIOS:raise ValueError("Unknown synthetic terrain")
        self.kind=kind;self.resolution=float(resolution);self.slope_limit_deg=float(slope_limit_deg)
        self.x=np.arange(0,7000.+.1,self.resolution)
        self.y=np.arange(0,14000.+.1,self.resolution)
        assert self.x[-1]==7000 and self.y[-1]==14000
        xx,yy=np.meshgrid(self.x,self.y)
        self.z=analytic_height(xx,yy,kind) if elevation is None else np.asarray(elevation,float).copy()
        assert self.z.shape==xx.shape and np.isfinite(self.z).all()
        # Exact maximum gradient magnitude of the bilinear surface per cell.
        d=self.resolution
        dx0=(self.z[:-1,1:]-self.z[:-1,:-1])/d
        dx1=(self.z[1:,1:]-self.z[1:,:-1])/d
        dy0=(self.z[1:,:-1]-self.z[:-1,:-1])/d
        dy1=(self.z[1:,1:]-self.z[:-1,1:])/d
        magnitude=np.hypot(np.maximum(abs(dx0),abs(dx1)),np.maximum(abs(dy0),abs(dy1)))
        self.cell_slope_deg=np.rad2deg(np.arctan(magnitude))
        self.forbidden_cells=self.cell_slope_deg>self.slope_limit_deg
        self._excluded=None
        self.digest=hashlib.sha256(self.z.astype("<f8").tobytes()+json.dumps(
            [self.resolution,self.slope_limit_deg],separators=(",",":")).encode()).hexdigest()
    def indices(self,xy):
        a=np.asarray(xy,float)
        if a.shape[-1]!=2 or not np.isfinite(a).all():raise ValueError("Expected finite local x/y")
        if np.any(a[...,0]<0) or np.any(a[...,0]>7000) or np.any(a[...,1]<0) or np.any(a[...,1]>14000):
            raise ValueError("DEM extrapolation is forbidden")
        ix=np.minimum((a[...,0]/self.resolution).astype(int),len(self.x)-2)
        iy=np.minimum((a[...,1]/self.resolution).astype(int),len(self.y)-2)
        u=(a[...,0]-self.x[ix])/self.resolution;v=(a[...,1]-self.y[iy])/self.resolution
        return ix,iy,u,v
    def sample(self,xy,gradient=False):
        ix,iy,u,v=self.indices(xy)
        z00=self.z[iy,ix];z10=self.z[iy,ix+1]
        z01=self.z[iy+1,ix];z11=self.z[iy+1,ix+1]
        h=(1-u)*(1-v)*z00+u*(1-v)*z10+(1-u)*v*z01+u*v*z11
        if not gradient:return h
        gx=((1-v)*(z10-z00)+v*(z11-z01))/self.resolution
        gy=((1-u)*(z01-z00)+u*(z11-z10))/self.resolution
        return h,np.stack([gx,gy],axis=-1)
    def allowed_mask(self,xy):
        ix,iy,u,v=self.indices(xy)
        blocked=self.forbidden_cells[iy,ix].copy()
        left=np.isclose(u,0,rtol=0,atol=1e-12)&(ix>0)
        below=np.isclose(v,0,rtol=0,atol=1e-12)&(iy>0)
        blocked|=left&self.forbidden_cells[iy,np.maximum(ix-1,0)]
        blocked|=below&self.forbidden_cells[np.maximum(iy-1,0),ix]
        blocked|=left&below&self.forbidden_cells[np.maximum(iy-1,0),np.maximum(ix-1,0)]
        return ~blocked
    @property
    def excluded(self):
        if self._excluded is None:
            rectangles=[]
            for j,row in enumerate(self.forbidden_cells):
                # Fuse contiguous cells in each row before polygon union.
                edges=np.diff(np.r_[False,row,False].astype(int))
                starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
                rectangles.extend(box(self.x[a],self.y[j],self.x[b],self.y[j+1]) for a,b in zip(starts,ends))
            self._excluded=unary_union(rectangles)
        return self._excluded
    def metadata(self):
        return dict(source="synthetic analytic terrain sampled on a regular DEM; not observed data",
            coordinate_system="local Cartesian metres, no real geographic location or EPSG",
            kind=self.kind,generator_parameters=SCENARIOS[self.kind],
            formula="100 + A*sin(2*pi*x/7000)*sin(2*pi*y/14000) + B*exp(-0.5*((u/600)^2+(v/2500)^2)); u,v are 25-degree rotated offsets from (3800,7600)",
            extent_m=[0,0,7000,14000],resolution_m=self.resolution,node_shape=list(self.z.shape),
            minimum_height_m=float(self.z.min()),maximum_height_m=float(self.z.max()),
            maximum_cell_slope_deg=float(self.cell_slope_deg.max()),slope_limit_deg=self.slope_limit_deg,
            slope_limit_status="declared scenario assumption, not an engineering code or site approval threshold",
            excluded_slope_area_km2=float(self.forbidden_cells.sum()*self.resolution**2/1e6),
            slope_rule="A cell is excluded if the maximum gradient magnitude of its bilinear DEM surface exceeds tan(limit).",
            sha256=self.digest)
    def export(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/"dem.npz",x_m=self.x,y_m=self.y,elevation_m=self.z,
                            maximum_cell_slope_deg=self.cell_slope_deg,forbidden_slope_cells=self.forbidden_cells)
        (folder/"metadata.json").write_text(json.dumps(self.metadata(),ensure_ascii=False,indent=2),encoding="utf-8")
        # Node-centred ESRI ASCII grid; local coordinates only.
        with (folder/"elevation_local.asc").open("w",encoding="ascii") as f:
            f.write(f"ncols {len(self.x)}\nnrows {len(self.y)}\nxllcenter 0\nyllcenter 0\ncellsize {self.resolution:g}\nNODATA_value -9999\n")
            np.savetxt(f,self.z[::-1],fmt="%.10f")

def surface_distance(terrain,start,end,order=64):
    """Gauss-Legendre arc length of the DEM along a straight horizontal route."""
    a=np.atleast_2d(start).astype(float);b=np.atleast_2d(end).astype(float)
    a,b=np.broadcast_arrays(a,b);delta=b-a
    nodes,weights=np.polynomial.legendre.leggauss(order);t=(nodes+1)/2
    points=a[:,None,:]+t[None,:,None]*delta[:,None,:]
    _,g=terrain.sample(points,gradient=True)
    horizontal=np.sum(delta**2,axis=1)[:,None]
    vertical=np.sum(g*delta[:,None,:],axis=2)
    return (np.sqrt(horizontal+vertical**2)*weights[None,:]/2).sum(axis=1)

def exact_segment_arc(terrain,a,b):
    """Independent cell-by-cell closed-form arc integral for validation."""
    from math import sqrt,asinh
    a=np.asarray(a,float);b=np.asarray(b,float);delta=b-a;length=float(np.linalg.norm(delta))
    if length==0:return 0.
    events=[0.,1.]
    for dim,grid in enumerate([terrain.x,terrain.y]):
        if delta[dim]!=0:
            vals=(grid-a[dim])/delta[dim]
            events.extend(vals[(vals>0)&(vals<1)].tolist())
    events=sorted(set(events));total=0.
    for t0,t1 in zip(events[:-1],events[1:]):
        mid=(t0+t1)/2
        point=a+mid*delta
        ix=min(int(point[0]/terrain.resolution),len(terrain.x)-2)
        iy=min(int(point[1]/terrain.resolution),len(terrain.y)-2)
        h00,h10=terrain.z[iy,ix],terrain.z[iy,ix+1]
        h01,h11=terrain.z[iy+1,ix],terrain.z[iy+1,ix+1]
        cross=(h11-h10-h01+h00)/terrain.resolution**2
        def slope(t):
            x,y=a+t*delta
            gx=(h10-h00)/terrain.resolution+cross*(y-terrain.y[iy])
            gy=(h01-h00)/terrain.resolution+cross*(x-terrain.x[ix])
            return float(gx*delta[0]+gy*delta[1])
        q0=slope(t0);q1=slope(t1)
        change=q1-q0
        if abs(change)<1e-8*max(1.,abs(q0),abs(q1)):
            total+=(t1-t0)*sqrt(length**2+((q0+q1)/2)**2)
        else:
            def primitive(q):return .5*(q*sqrt(length**2+q*q)+length**2*asinh(q/length))
            total+=(t1-t0)*(primitive(q1)-primitive(q0))/change
    return total

def terrain_candidates(scene,terrain,m,seed=20261012):
    from scipy.stats import qmc
    exponent=int(np.ceil(np.log2(max(32,2*m))))
    while exponent<=20:
        raw=qmc.Sobol(2,scramble=True,seed=seed).random_base2(exponent)*[scene.width,scene.height]
        site=scene.site_mask(raw,closed_obstacles=True)
        slope=terrain.allowed_mask(raw)
        allowed=site&slope;indices=np.flatnonzero(allowed)
        if len(indices)>=m:
            end=int(indices[m-1])+1
            return raw[allowed][:m],dict(generator="scrambled_Sobol_2D",seed=seed,
                raw_points_inspected=end,original_forbidden_filtered=int((~site[:end]).sum()),
                slope_filtered_after_site=int((site[:end]&~slope[:end]).sum()),retained_points=m,
                installed_count_used=False,anchor_injection=False,terrain_sha256=terrain.digest)
        exponent+=1
    raise ValueError("Insufficient terrain-allowed candidate points")
