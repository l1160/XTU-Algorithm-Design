"""Downstream-only 2-D hub-centre Jensen proxy; V3 remains immutable."""
import bootstrap_v4
import numpy as np
from model import Farm as V3Farm

def downstream_wake_squared(scene,xy):
    xy=np.asarray(xy,float)
    dx=xy[:,0,None]-xy[None,:,0]
    dy=xy[:,1,None]-xy[None,:,1]
    result=np.empty((24,len(xy),len(xy)),float)
    for q in range(24):
        angle=np.deg2rad(q*15+7.5);co,si=np.cos(angle),np.sin(angle)
        downstream=dx*co+dy*si
        crosswind=np.abs(-dx*si+dy*co)
        active=(downstream>0)&(crosswind < scene.radius+scene.decay*downstream)
        distance=np.maximum(downstream,0)
        deficit=(1-np.sqrt(1-scene.thrust))/(1+scene.decay*distance/scene.radius)**2
        result[q]=np.where(active,deficit**2,0)
        np.fill_diagonal(result[q],0)
    return result

class Farm(V3Farm):
    def ensure_wake(self):
        if self.wake is None:self.wake=downstream_wake_squared(self.scene,self.xy)
    def case_metadata(self):
        d=super().case_metadata()
        d.update(model_version="v4.downstream-only",wake_scope="2-D hub-centre proxy, downstream>0; no rotor-area quadrature or terrain")
        return d
