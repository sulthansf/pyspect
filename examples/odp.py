"""
This is an example of how to implement pyspect using hj_reacahbility.
"""

import os
os.environ['PATH'] += ':' + f'{__import__("mlirlib").__path__[0]}/bin'

print(os.environ['PATH'])

import numpy as np
import optimized_dp as odp
import optimized_dp.shapes as shp

import pyspect as ps

class Set(ps.Set):

    vf: np.ndarray
    solver: odp.HJSolver
    
    def __init__(self, solver, vf, **kwargs):
        super().__init__(**kwargs)

        self.solver = solver
        self.vf = vf

    def copy(self):
        return Set(self.solver, 
                   self.vf.copy(), 
                   approx=self.approx)
    
    @classmethod
    def empty(self):
        return np.ones(self.solver.grid.shape)
    
    def is_empty(self):
        return np.all(0 < self.vf)
    
    def membership(self, point):
        return self.solver.grid.get_value(point) <= 0
    
    def complement(self):
        s = super().complement()
        s.vf = -s.vf
        return s
    
    def union(self, other):
        assert self.solver.grid is other.solver.grid, 'Grids must match'
        assert np.isclose(self.solver.tau, other.solver.tau).all(), 'Timelines must match'
        s = super().union(other)
        s.vf = np.minimum(self.vf, other.vf)
        return s
    
    def intersect(self, other):
        assert self.solver.grid is other.solver.grid, 'Grids must match'
        assert np.isclose(self.solver.tau, other.solver.tau).all(), 'Timelines must match'
        s = super().intersect(other)
        s.vf = np.maximum(self.vf, other.vf)
        return s
    
    def reach(self, constraints=None):
        s = super().reach(constraints)
        vf = self.solver(target=self.vf, target_mode='min',
                         constraint=constraints.vf, constraint_mode='max')
        s.vf = np.asarray(vf)
        return s

    def rci(self):
        s = super().rci()
        s._make_tube()
        target = np.ones_like(s.vf)
        target[-1, ...] = s.vf[-1, ...]
        constraint = s.vf
        vf = self.solver(target=target, target_mode='min',
                         constraint=constraint, constraint_mode='max')
        s.vf = np.asarray(vf)
        return s

    def _is_invariant(self):
        return len(self.vf.shape) != len(self.timeline.shape + self.grid.shape)

    def _make_tube(self):
        if self._is_invariant():    
            self.vf = np.concatenate([self.vf[np.newaxis, ...]] * len(self.timeline))

if __name__ == '__main__':

    from pyspect.rules import *
    from optimized_dp.systems import Bicycle5D

    ## Create computation grid ##

    grid = odp.Grid([   31,    31,     31,         7,    11],
                    [ +1.2,  +1.2,  +np.pi, +np.pi/5,    +1],
                    [ -1.2,  -1.2,  -np.pi, -np.pi/5,    +0],
                    [False, False, True, False, False])
    X, Y, YAW, DELTA, VEL = range(grid.ndims)

    ## Create time steps ##

    def new_timeline(target_time, start_time=0, time_step=0.2):
        assert time_step > 0
        is_forward = target_time >= start_time
        target_time += 1e-5 if is_forward else -1e-5
        time_step *= 1 if is_forward else -1
        return np.arange(start_time, target_time, time_step)

    timeline = new_timeline(3)

    ## Create system dynamics ##

    REACH_SCENARIO = {"uMode": "min", "dMode": "max"} # worst possible scenario
    AVOID_SCENARIO = {"uMode": "max", "dMode": "min"} # best possible scenario

    # Global model parameters
    model_settings = {**REACH_SCENARIO,
                      "uMin": [-3*np.pi, -5],
                      "uMax": [+3*np.pi, +5],
                      "dMin": [-0.0, -0.0, 0, 0, 0],
                      "dMax": [+0.0, +0.0, 0, 0, 0]}
    model = Bicycle5D(ctrl_range=[model_settings['uMin'],
                                  model_settings['uMax']],
                      dstb_range=[model_settings['dMin'],
                                  model_settings['dMax']],
                      mode='reach')

    ## Create solver ##

    solver = odp.HJSolver(grid, timeline, model, accuracy='low')

    ## Create static environment ##

    vf_r1 = shp.intersection(shp.upper_half_space(grid, VEL, +0.4),
                             shp.lower_half_space(grid, X, +0.0),
                             shp.upper_half_space(grid, Y, +0.2),
                             shp.lower_half_space(grid, Y, +1.1))
    vf_r2 = shp.intersection(shp.upper_half_space(grid, VEL, +0.4),
                             shp.lower_half_space(grid, Y, +0.5),
                             shp.upper_half_space(grid, X, -0.5),
                             shp.lower_half_space(grid, X, +0.5))
    vf_r3 = shp.intersection(shp.upper_half_space(grid, VEL, +0.4),
                             shp.upper_half_space(grid, X, +0.0),
                             shp.upper_half_space(grid, Y, +0.2),
                             shp.lower_half_space(grid, Y, +1.1))

    # Ports (in & out of region)
    vf_p1w = shp.intersection(vf_r1,  
                              shp.union(shp.upper_half_space(grid, YAW, +np.pi - np.pi/5),
                                        shp.lower_half_space(grid, YAW, -np.pi + np.pi/5)),   
                              shp.upper_half_space(grid, Y, +0.6))
    vf_p1e = shp.intersection(vf_r1,  
                              shp.intersection(shp.upper_half_space(grid, YAW, - np.pi/5),
                                               shp.lower_half_space(grid, YAW, + np.pi/5)),   
                              shp.lower_half_space(grid, Y, +0.6))
    vf_p2n = shp.intersection(vf_r2,  
                              shp.intersection(shp.upper_half_space(grid, YAW, +np.pi/2 - np.pi/5),
                                               shp.lower_half_space(grid, YAW, +np.pi/2 + np.pi/5)),
                              shp.upper_half_space(grid, X, 0.0))
    vf_p2s = shp.intersection(vf_r2,  
                              shp.intersection(shp.upper_half_space(grid, YAW, -np.pi/2 - np.pi/5),
                                               shp.lower_half_space(grid, YAW, -np.pi/2 + np.pi/5)),
                              shp.lower_half_space(grid, X, 0.0))
    vf_p3w = shp.intersection(vf_r3,  
                              shp.union(shp.upper_half_space(grid, YAW, +np.pi - np.pi/5),
                                        shp.lower_half_space(grid, YAW, -np.pi + np.pi/5)),   
                              shp.upper_half_space(grid, Y, +0.6))
    vf_p3e = shp.intersection(vf_r3,  
                              shp.intersection(shp.upper_half_space(grid, YAW, - np.pi/5),
                                               shp.lower_half_space(grid, YAW, + np.pi/5)),   
                              shp.lower_half_space(grid, Y, +0.6))

    # intersections 
    vf_i1 = shp.intersection(shp.upper_half_space(grid, VEL, +0.7),
                             shp.rectangle(grid,
                                           target_min=[-0.5, +0.2, *grid.min_bounds[Y+1:]],
                                           target_max=[+0.5, +1.1, *grid.max_bounds[Y+1:]]))

    # Regions
    vf_rg1 = shp.union(vf_i1, vf_p1w, vf_p1e, vf_p2n, vf_p2s, vf_p3w, vf_p3e)

    # Specific routes
    vf_p2n_p1w = shp.union(vf_p2n, vf_i1, vf_p1w)
    vf_p3w_p2s = shp.union(vf_p3w, vf_i1, vf_p2s)
    vf_p1e_p3e = shp.union(vf_p1e, vf_i1, vf_p3e)

    ## Run pyspect ##

    r1 = Until(p1=..., p2=...)

    out = r1(p1=Set(solver, shp.make_tube(vf_p1w, timeline)), 
             p2=Set(solver, shp.make_tube(vf_p2n_p1w, timeline)))
