"""
This is an example of how to implement pyspect using hj_reacahbility.
"""

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

import pyspect as ps

class Set(ps.Set):

    vf: np.ndarray
    grid: hj.Grid
    timeline: np.ndarray
    dynamics: hj.Dynamics

    solver_settings = hj.SolverSettings.with_accuracy("low")
    
    def __init__(self, dynamics, grid, timeline, vf, **kwargs):
        super().__init__(**kwargs)

        self.dynamics = dynamics
        self.timeline = timeline
        self.grid = grid
        self.vf = vf

    def copy(self):
        return Set(self.dynamics, 
                   self.grid, 
                   self.timeline.copy(), 
                   self.vf.copy(), 
                   approx=self.approx)
    
    @classmethod
    def empty(self):
        return np.ones(self.grid.shape)
    
    def is_empty(self):
        return np.all(0 < self.vf)
    
    def membership(self, point):
        idx = self.grid.nearest_index(point)
        return self.vf[idx] <= 0
    
    def complement(self):
        s = super().complement()
        s.vf = -s.vf
        return s
    
    def union(self, other):
        assert self.grid is other.grid, 'Grids must match'
        assert np.isclose(self.timeline, other.timeline).all(), 'Timelines must match'
        s = super().union(other)
        s.vf = np.minimum(self.vf, other.vf)
        return s
    
    def intersect(self, other):
        assert self.grid is other.grid, 'Grids must match'
        assert np.isclose(self.timeline, other.timeline).all(), 'Timelines must match'
        s = super().intersect(other)
        s.vf = np.maximum(self.vf, other.vf)
        return s
    
    def reach(self, constraints=None):
        s = super().reach(constraints)
        vf = hj.solve(self.solver_settings,
                      self.dynamics,
                      self.grid,
                      self.timeline,
                      self.vf,
                      constraints.vf)
        s.vf = np.asarray(vf)
        return s

    def rci(self):
        s = super().rci()
        s._make_tube()
        target = np.ones_like(s.vf)
        target[-1, ...] = s.vf[-1, ...]
        constraint = s.vf
        vf = hj.solver(self.solver_settings,
                       self.dynamics,
                       self.grid,
                       self.timeline,
                       target,
                       constraint)
        s.vf = np.asarray(vf)
        return s

    def _is_invariant(self):
        return len(self.vf.shape) != len(self.timeline.shape + self.grid.shape)

    def _make_tube(self):
        if self._is_invariant():    
            self.vf = np.concatenate([self.vf[np.newaxis, ...]] * len(self.timeline))

if __name__ == '__main__':

    from pyspect.rules import *
    import hj_reachability as hj
    import hj_reachability.shapes as shp

    reach_dynamics = hj.systems.SVEA5D(min_steer=-jnp.pi, 
                                       max_steer=+jnp.pi,
                                       min_accel=-0.5,
                                       max_accel=+0.5).with_mode('reach')
    
    min_bounds = np.array([-1.2, -1.2, -np.pi, -np.pi/5, +0])
    max_bounds = np.array([+1.2, +1.2, +np.pi, +np.pi/5, +1])
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                                (31, 31, 31, 7, 11),
                                                                periodic_dims=2)

    X, Y, YAW, DELTA, VEL = range(5)

    # Roads
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
                                           target_min=[-0.5, +0.2, *min_bounds[Y+1:]],
                                           target_max=[+0.5, +1.1, *max_bounds[Y+1:]]))

    # Regions
    vf_rg1 = shp.union(vf_i1, vf_p1w, vf_p1e, vf_p2n, vf_p2s, vf_p3w, vf_p3e)

    # Specific routes
    vf_p2n_p1w = shp.union(vf_p2n, vf_i1, vf_p1w)
    vf_p3w_p2s = shp.union(vf_p3w, vf_i1, vf_p2s)
    vf_p1e_p3e = shp.union(vf_p1e, vf_i1, vf_p3e)

    def new_timeline(target_time, start_time=0, time_step=0.2):
        assert time_step > 0
        is_forward = target_time >= start_time
        target_time += 1e-5 if is_forward else -1e-5
        time_step *= 1 if is_forward else -1
        return np.arange(start_time, target_time, time_step)

    timeline = new_timeline(3)

    r1 = Until(p1=..., p2=...)

    out = r1(p1=Set(reach_dynamics, grid, timeline, vf_p1w), 
             p2=Set(reach_dynamics, grid, timeline, vf_p2n_p1w))
