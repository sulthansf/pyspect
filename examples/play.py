from __future__ import annotations
import collections.abc
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import *

from typing_protocol_intersection import ProtocolIntersection as Has

######################################################################
## Language: Linear Temporal Logic

AP_TRUE = 'true'

OP_NOT = 'NOT'
OP_AND = 'AND'
OP_OR = 'OR'
OP_UNTIL = 'UNTIL'
OP_ALWAYS = 'ALWAYS'

type Atomic = str
type Operator = str
type Formula = (Atomic                                  # Proposition
                | tuple[Operator, Formula]              # Unary operation
                | tuple[Operator, Formula, Formula])    # Binary operation

RULES = {}

def __repr__(self) -> str:
        op = type(self).__name__
        next(it := iter(self)) # skip first
        body = ',\n'.join(map(repr, it))
        sep = '\n' + ' ' * (len(op)+1)
        body = sep.join(body.splitlines())
        return f'{op}({body})'

def iter_rule(rule: Formula):
    if isinstance(rule, str): # Atomic
        yield rule
    else:
        _, *args = rule
        yield rule
        for arg in args:
            yield from iter_rule(arg)

def repr_rule(rule: Formula):
    if isinstance(rule, str): # Atomic
        return f'<{rule}>'
    else:
        op = rule[0]
        body = ',\n'.join(map(repr_rule, iter_rule(rule)))
        sep = '\n' + ' ' * (len(op)+1)
        body = sep.join(body.splitlines())
        return f'{op}({body})'


def declare_rule(name, equiv):
    assert name not in RULES, 'Rule already exists'
    RULES[name] = equiv
    return lambda *args: (name, *args)

def expand_rule(rule):
    if isinstance(rule, str): # Atomic
        return rule
    else:
        op, *args = rule
        return RULES[op](*args) if op in RULES else rule

not_    : Callable[[Formula], Formula]          \
        = lambda arg: (OP_NOT, arg)
and_    : Callable[..., Formula]                \
        = lambda lhs, rhs, *rest: and_((OP_AND, lhs, rhs), *rest) if rest else (OP_AND, lhs, rhs)
or_     : Callable[..., Formula]                \
        = lambda lhs, rhs, *rest: or_((OP_OR, lhs, rhs), *rest) if rest else (OP_OR, lhs, rhs)
until_  : Callable[[Formula, Formula], Formula] \
        = lambda lhs, rhs: (OP_UNTIL, lhs, rhs)
always_ : Callable[[Formula], Formula]          \
        = lambda arg: (OP_ALWAYS, arg)

implies_ = declare_rule('IMPLIES', lambda lhs, rhs: or_(not_(lhs), rhs))
eventually_ = declare_rule('EVENTUALLY', lambda arg: until_(AP_TRUE, arg))


######################################################################
## Implementation

class ImplEmpty[R](Protocol):
    def empty(self) -> R: ...

class ImplComplement[R](Protocol):
    def complement(self, s: R) -> R: ...

class ImplIntersect[R](Protocol):
    def intersect(self, s1: R, s2: R) -> R: ...

class ImplUnion[R](Protocol):
    def union(self, s1: R, s2: R) -> R: ...

class ImplRCI[R](Protocol):
    def rci(self, s: R) -> R: ...

class ImplReachForw[R](Protocol):
    def reach_forw(self, target: R, constraints: R) -> R: ...

class ImplReachBack[R](Protocol):
    def reach_back(self, target: R, constraints: R) -> R: ...

class ImplAxes[R](Protocol):
    ndim: int
    def axis(self, name: str) -> int: ...
    def axis_name(self, i: int) -> str: ...
    def axis_is_periodic(self, i: int) -> bool: ...

class ImplPlaneCut[R](ImplAxes[R], Protocol):
    def plane_cut(self, normal: list[float], offset: Optional[list[float]] = None, axes: Optional[int|list[int]] = None) -> R: ...

######################################################################
## Sets

type SetBuilder[I, R] = Callable[[I], R]

def ABSURD[I, R](impl: I, **m: SetBuilder[I, R]) -> R:
    raise ValueError("Cannot realize the absurd set.")

def Empty[I, R]() -> SetBuilder[Has[I, ImplEmpty], R]:
    return lambda impl, **m: impl.empty()

def Set[I, R](arg: R) -> SetBuilder[I, R]:
    return lambda impl, **m: arg

def ReferredSet[I, R](name: str) -> SetBuilder[I, R]:
    return lambda impl, **m: m.pop(name)(impl, **m)

def HalfSpaceSet[I, R](*arg, **kwargs) -> SetBuilder[Has[I, ImplPlaneCut], R]:
    return lambda impl, **m: impl.plane_cut(*arg, **kwargs)

type _ImplBoundedSet = Has[ImplEmpty, ImplAxes, ImplPlaneCut, ImplComplement, ImplIntersect]
def BoundedSet[I, R](**bounds: tuple[float, float]) -> SetBuilder[Has[I, _ImplBoundedSet], R]:
    def sb(impl: _ImplBoundedSet, **m) -> R:
        s = impl.empty()
        _bounds = [(vmin, vmax, impl.axis(name))
                   for name, (vmin, vmax) in bounds.items()]
        for n, row in enumerate(_bounds):
            vmin, vmax, i = (row if len(row) == 3 else [*row, n])
            if vmax < vmin and impl.axis_is_periodic(i):
                upper_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                axis_range = impl.complement(impl.intersect(upper_bound, lower_bound))
            else:
                upper_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                axis_range = impl.intersect(upper_bound, lower_bound)
            s = impl.intersect(s, axis_range)
        return s
    return sb

######################################################################
## Temporal Logic Trees

from enum import IntEnum

class APPROX(IntEnum):
    UNDER = -1
    EXACT = 0
    OVER = +1

type ApproxType = Optional[APPROX]
type TLTLike[I, R] = Formula | SetBuilder[I, R] | TLT[I, R]
type LabellingMap[I, R] = dict[str, None | SetBuilder[I, R]]

@dataclass(slots=True, frozen=True)
class TLT[I, R]:
    formula: Formula            = field(default='_0')
    builder: SetBuilder[I, R]   = field(default=ABSURD)         # If constructed with the absurd set, then the TLT is also absurd, i.e. cannot be realized.
    approx: ApproxType          = field(default=APPROX.EXACT)
    lmap: LabellingMap[I, R]    = field(default_factory=dict)   # Realized sets associated by names for ReferredSets

    def realize(self, impl: I) -> R:
        assert self.is_realizable(), f'Cannot realize TLT, missing {list(self.iter_free())}'
        return self.builder(impl, **self.lmap)

    def is_realizable(self) -> bool:
        for sb in self.lmap.values():
            if sb is None: return False
        return True

    def iter_free(self):
        yield from filter(lambda p: self.lmap[p] is None, self.lmap)

    @classmethod
    def construct(cls, arg: TLTLike[I, R], **m: TLTLike[I, R]) -> TLT[I, R]:
        
        if isinstance(arg, TLT):
            # Create new TLT with updated information (we can bind new TLT-like objects to free variables)
            lmap = {p: sb if sb is not None else
                       (tlt := m.get(p)) and TLT.construct(tlt).builder
                    for p, sb in arg.lmap.items()}
            return cls(arg.formula, arg.builder, arg.approx, lmap)

        elif isinstance(arg, Callable): # Set Builder
            # Assume a formula "_0" where the set exactly represent the prop "_0".
            return cls(builder=arg)
        
        elif isinstance(arg, str): # Atomic Proposition
            return (cls.construct(m.pop(arg), **m) if arg in m else 
                    cls(arg, ReferredSet(arg), lmap={arg: None}))
        
        else:
            return cls._construct_from_formula(arg, **m)

    @classmethod
    def _construct_from_formula(cls, formula: Formula, **m: TLTLike[I, R]) -> TLT[I, R]:

        op_f, *args_f = expand_rule(formula)
        args = [cls.construct(subf, **m) for subf in args_f]
        
        if op_f == OP_NOT:
            (_1,) = args
            sb, a = _1.builder, _1.approx
            return cls((op_f, _1.formula),
                       lambda impl, **m: impl.complement(sb(impl, **m)),
                       None if a is None else APPROX(-1 * a),
                       {p: _1.lmap[p] for p in list(_1.lmap)})
        
        elif op_f == OP_AND:
            (_1, _2) = args
            sb1, a1 = _1.builder, _1.approx
            sb2, a2 = _2.builder, _2.approx
            return cls((op_f, _1.formula, _2.formula),
                       lambda impl, **m: impl.intersect(sb1(impl, **m), sb2(impl, **m)),
                       (None if None in (a1, a2) else
                        None if APPROX.UNDER in (a1, a2) else
                        APPROX.OVER),
                       {p: _1.lmap.get(p) or _2.lmap.get(p)
                        for p in list(_1.lmap) + list(_2.lmap)})
         
        elif op_f == OP_OR:
            (_1, _2) = args
            sb1, a1 = _1.builder, _1.approx
            sb2, a2 = _2.builder, _2.approx
            return cls((op_f, _1.formula, _2.formula),
                       lambda impl, **m: impl.union(sb1(impl, **m), sb2(impl, **m)),
                       (None if None in (a1, a2) else
                        a1 if a1 == a2 else
                        a2 if a1 == APPROX.EXACT else
                        a1 if a2 == APPROX.EXACT else
                        None), 
                       {p: _1.lmap.get(p) or _2.lmap.get(p)
                        for p in list(_1.lmap) + list(_2.lmap)})

        elif op_f == OP_UNTIL:
            (_1, _2) = args
            sb1, a1 = _1.builder, _1.approx # constraint
            sb2, a2 = _2.builder, _2.approx # target
            return cls((op_f, _1.formula, _2.formula),
                       lambda impl, **m: impl.reach_forw(sb2(impl, **m), sb1(impl, **m)),
                       APPROX.EXACT, # TODO
                       {p: _1.lmap.get(p) or _2.lmap.get(p)
                        for p in list(_1.lmap) + list(_2.lmap)})
        
        elif op_f == OP_ALWAYS:
            (_1,) = args
            sb, a = _1.builder, _1.approx # target
            return cls((op_f, _1.formula),
                       lambda impl, **m: impl.rci(sb(impl, **m)),
                       APPROX.EXACT, # TODO
                       {p: _1.lmap.get(p) for p in list(_1.lmap)})
        
        else:
            raise ValueError(f'Invalid operator: {op_f}')

# Pseudo-LTL/TLT operations

def Not(arg):
    return TLT.construct(not_('_1'), _1=arg)

def And(lhs, rhs, *rest):
    subtree = TLT.construct(and_('_1', '_2'), _1=lhs, _2=rhs)
    return And(subtree, *rest) if rest else subtree

def Or(lhs, rhs, *rest):
    subtree = TLT.construct(or_('_1', '_2'), _1=lhs, _2=rhs)
    return Or(subtree, *rest) if rest else subtree

def Minus(lhs, rhs):
    return TLT.construct(or_('_1', not_('_2')), _1=lhs, _2=rhs)

def Implies(lhs, rhs):
    return TLT.construct(implies_('_1', '_2'), _1=lhs, _2=rhs)

def Until(lhs, rhs):
    return TLT.construct(until_('_1', '_2'), _1=lhs, _2=rhs)

def Always(arg):
    return TLT.construct(always_('_1'), _1=arg)


######################################################################
#### User Space ####

######################################################################
## Environment

import numpy as np

print('Creating speed limits')
limit_30 = BoundedSet(v=(0.3, 0.6))
limit_50 = BoundedSet(v=(0.4, 1.0))

print('Creating road geometries')
intersection_geometry = And(
    BoundedSet(x=(4.20/8, 6.87/8), y=(1.20/6, 3.04/6)),
    HalfSpaceSet(normal=[-1, 1], offset=[4.60/8, 2.67/6]),
    HalfSpaceSet(normal=[1, 1],  offset=[6.40/8, 2.67/6]),
)
kyrkogatan_left_geometry = Or(
    intersection_geometry,
    BoundedSet(x=(0.00/8, 5.33/8), y=(1.85/6, 2.37/6), h=(+np.pi - np.pi/5, -np.pi + np.pi/5)),
    BoundedSet(x=(6.00/8, 8.00/8), y=(1.70/6, 2.24/6), h=(+np.pi - np.pi/5, -np.pi + np.pi/5)),
)
kyrkogatan_right_geometry = Or(
    intersection_geometry,
    BoundedSet(x=(0.00/8, 5.33/8), y=(1.20/6, 1.84/6), h=(-np.pi/5, +np.pi/5)),
    BoundedSet(x=(6.00/8, 8.00/8), y=(1.20/6, 1.73/6), h=(-np.pi/5, +np.pi/5)),
)
nygatan_down_geometry = Or(
    intersection_geometry,
    BoundedSet(x=(4.94/8, 5.53/8), y=(2.26/6, 6.00/6), h=(-np.pi/2 - np.pi/5, -np.pi/2 + np.pi/5)),
    BoundedSet(x=(5.07/8, 5.47/8), y=(0.00/6, 1.60/6), h=(-np.pi/2 - np.pi/5, -np.pi/2 + np.pi/5)),
)
nygatan_up_geometry = Or(
    intersection_geometry,
    BoundedSet(x=(5.60/8, 6.14/8), y=(2.27/6, 6.00/6), h=(+np.pi/2 - np.pi/5, +np.pi/2 + np.pi/5)),
    BoundedSet(x=(5.33/8, 5.74/8), y=(0.00/6, 1.60/6), h=(+np.pi/2 - np.pi/5, +np.pi/2 + np.pi/5)),
) 

print('Creating streets')
kyrkogatan_vel      = And(Implies(HalfSpaceSet(normal=[-1], offset=[3.01/8]), limit_50),
                          Implies(HalfSpaceSet(normal=[+1], offset=[3.00/8]), limit_30))
nygatan_vel         = And(Implies(HalfSpaceSet(normal=[-1], offset=[4.01/6]), limit_30),
                          Implies(HalfSpaceSet(normal=[+1], offset=[4.00/6]), limit_50))
kyrkogatan_left     = And(kyrkogatan_left_geometry, kyrkogatan_vel)
kyrkogatan_right    = And(kyrkogatan_right_geometry, kyrkogatan_vel)
nygatan_down        = And(nygatan_down_geometry, nygatan_vel)
nygatan_up          = And(nygatan_up_geometry, nygatan_vel) 
kyrkogatan          = Or(kyrkogatan_left, kyrkogatan_right)
nygatan             = Or(nygatan_down, nygatan_up)

print('Creating entry/exit zones')
exit_zone     = BoundedSet(x=(5.67/8, 6.13/8), y=(5.47/6, 5.93/6))
entry_zone    = BoundedSet(x=(1.50/8, 1.95/8), y=(1.87/6, 2.33/6))
parking_start = BoundedSet(x=(2.30/8, 2.75/8), y=(1.87/6, 2.33/6))

print('Creating parking lot')
parking_spot_1 = BoundedSet(x=(2.13/8, 2.40/8), y=(5.54/6, 6.00/6), h=(+np.pi/2 - np.pi/5, +np.pi/2 + np.pi/5))
parking_spot_2 = BoundedSet(x=(3.15/8, 3.47/8), y=(4.33/6, 4.80/6), h=(-np.pi/2 - np.pi/5, -np.pi/2 + np.pi/5))
parking_spot_entry_1 = BoundedSet(x=(2.00/8, 2.53/8), y=(5.33/6, 5.73/6))
parking_spot_entry_2 = BoundedSet(x=(3.02/8, 3.61/8), y=(4.67/6, 5.07/6))

parking_lot_geometry = Or(BoundedSet(x=(1.20/8, 2.10/8), y=(2.73/6, 6.00/6)),  # left side
                          # BoundedSet(x=(1.20/8, 4.40/8), y=(3.27/6, 3.74/6)),  # bottom
                          BoundedSet(x=(3.75/8, 4.60/8), y=(3.33/6, 5.47/6)),  # right side
                          BoundedSet(x=(1.20/8, 4.60/8), y=(4.87/6, 5.47/6)),  # top
                          BoundedSet(x=(1.20/8, 2.10/8), y=(2.73/6, 3.50/6)),  # entry inner
                          BoundedSet(x=(1.30/8, 1.95/8), y=(2.13/6, 3.00/6)))  # entry out
parking_spots = Or(parking_spot_1, parking_spot_2)
parking_spots_entry = Or(parking_spot_entry_1, parking_spot_entry_2)
parking_lot = Or(And(parking_lot_geometry, limit_30), parking_spots_entry, parking_spots)

print('Environment created!')

######################################################################
## HJ IMPLEMENTATION

import hj_reachability as hj

class StrImpl:

    def __init__(self, dynamics, grid, timeline):
        self.dynamics = dynamics
        self.timeline = timeline
        self.grid = grid
        self.ndim = grid.ndim

    def set_axes_names(self, *args):
        assert len(args) == self.ndim
        self._axes_names = args

    def axis(self, name: str) -> int:
        return self._axes_names.index(name)

    def axis_name(self, i: int) -> str:
        return self._axes_names[i]
    
    def axis_is_periodic(self, i: int) -> bool:
        return bool(self.grid._is_periodic_dim[i])

    def plane_cut(self, normal, offset, axes=None):
        return 'plane{...}'

    def empty(self):
        return 'empty{ }'
    
    def complement(self, vf):
        return f'({vf})^C'
    
    def intersect(self, vf1, vf2):
        return f'({vf1} ∩ {vf2})'

    def union(self, vf1, vf2):
        return f'({vf1} ∪ {vf2})'
    
    def rci(self, vf):
        return f'RCI({vf})'
    
    def reach_forw(self, target, constraints=None):
        return f'Rf({target}, {constraints})'

    def reach_back(self, target, constraints=None):
        return f'Rb({target}, {constraints})'


class HJImpl:

    solver_settings = hj.SolverSettings.with_accuracy("low")

    def __init__(self, dynamics, grid, timeline):
        self.dynamics = dynamics
        self.timeline = timeline
        self.grid = grid
        self.ndim = grid.ndim

    def set_axes_names(self, *args):
        assert len(args) == self.ndim
        self._axes_names = args

    def axis(self, name: str) -> int:
        return self._axes_names.index(name)

    def axis_name(self, i: int) -> str:
        return self._axes_names[i]
    
    def axis_is_periodic(self, i: int) -> bool:
        return bool(self.grid._is_periodic_dim[i])
    
    def plane_cut(self, normal, offset, axes=None):
        data = np.zeros(self.grid.shape)
        axes = axes or list(range(self.grid.ndim))
        x = lambda i: self.grid.states[..., i]
        for i, k, m in zip(axes, normal, offset):
            data += k*x(i) - k*m
        return data

    def empty(self):
        return np.ones(self.grid.shape)
    
    def complement(self, vf):
        return np.asarray(-vf)
    
    def intersect(self, vf1, vf2):
        return np.maximum(vf1, vf2)

    def union(self, vf1, vf2):
        return np.minimum(vf1, vf2)
    
    def rci(self, vf):
        vf = self._make_tube(vf)
        target = np.ones_like(vf)
        target[-1, ...] = vf[-1, ...]
        constraint = vf
        vf = hj.solver(self.solver_settings,
                       self.dynamics,
                       self.grid,
                       -self.timeline,
                       target,
                       constraint)
        return np.flip(np.asarray(vf), axis=0)
    
    def reach_forw(self, target, constraints=None):
        vf = hj.solve(self.solver_settings,
                      self.dynamics,
                      self.grid,
                      self.timeline,
                      target,
                      constraints)
        return np.asarray(vf)

    def reach_back(self, target, constraints=None):
        vf = hj.solve(self.solver_settings,
                      self.dynamics,
                      self.grid,
                      -self.timeline,
                      target,
                      constraints)
        return np.flip(np.asarray(vf), axis=0)

    def project_onto(self, vf, *idxs, keepdims=False):
        idxs = [len(vf.shape) + i if i < 0 else i for i in idxs]
        dims = [i for i in range(len(vf.shape)) if i not in idxs]
        return vf.min(axis=tuple(dims), keepdims=keepdims)

    def _is_invariant(self):
        return len(self.vf.shape) != len(self.timeline.shape + self.grid.shape)

    def _make_tube(self):
        if self._is_invariant():    
            self.vf = np.concatenate([self.vf[np.newaxis, ...]] * len(self.timeline))


######################################################################
## MAIN

import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import wraps

class Bicycle5D(hj.dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 min_steer, max_steer,
                 min_accel, max_accel,
                 min_disturbances=None, 
                 max_disturbances=None,
                 wheelbase=0.32,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        self.wheelbase = wheelbase

        if min_disturbances is None:
            min_disturbances = [0] * 5
        if max_disturbances is None:
            max_disturbances = [0] * 5

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([min_steer, min_accel]),
                                        jnp.array([max_steer, max_accel]))
        
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array(min_disturbances),
                                            jnp.array(max_disturbances))
        super().__init__(control_mode, 
                         disturbance_mode, 
                         control_space, 
                         disturbance_space)

    def with_mode(self, mode: str):
        assert mode in ["reach", "avoid"]
        if mode == "reach":
            self.control_mode = "min"
            self.disturbance_mode = "max"
        elif mode == "avoid":
            self.control_mode = "max"
            self.disturbance_mode = "min"
        return self

    ## Dynamics
    # Implements the affine dynamics 
    #   `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`.
    # 
    #   x_dot = v * cos(yaw) + d_1
    #   y_dot = v * sin(yaw) + d_2
    #   yaw_dot = (v * tan(delta))/L + d3
    #   delta_dot = u1 + d4
    #   v_dot = u2 + d5
        

    def open_loop_dynamics(self, state, time):
        x, y, yaw, delta, vel = state
        return jnp.array([
            vel * jnp.cos(yaw),
            vel * jnp.sin(yaw),
            (vel * jnp.tan(delta))/self.wheelbase,
            0.,
            0.,
        ])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.identity(5)

def new_timeline(target_time, start_time=0, time_step=0.2):
    assert time_step > 0
    is_forward = target_time >= start_time
    target_time += 1e-5 if is_forward else -1e-5
    time_step *= 1 if is_forward else -1
    return np.arange(start_time, target_time, time_step)

def auto_ax(f):
    @wraps(f)
    def wrapper(*args, ax: plt.Axes = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        kwargs.update(ax=ax)
        return f(*args, **kwargs)
    return wrapper

@auto_ax
def plot_im(im, *, ax, transpose=True, **kwargs):
    im = np.where(im, 0.5, np.nan)
    if transpose:
        im = np.transpose(im)
    kwargs.setdefault('cmap', 'Blues')
    kwargs.setdefault('aspect', 'auto')
    return [ax.imshow(im, vmin=0, vmax=1, origin='lower', **kwargs)]

@auto_ax
def plot_set(vf, **kwargs):
    vf = np.where(vf <= 0, 0.5, np.nan)
    kwargs.setdefault('aspect', 'equal')
    return plot_im(vf, **kwargs)

@auto_ax
def plot_set_many(*vfs, **kwargs):
    out = []
    f = lambda x: x if isinstance(x, tuple) else (x, {})
    for vf, kw in map(f, vfs):
        out += plot_set(vf, **kw, **kwargs)
    return out

def new_map(*pairs, **kwargs):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9*4/3, 9))
    extent=[min_bounds[0], max_bounds[0],
            min_bounds[1], max_bounds[1]]
    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")
    ax.invert_yaxis()
    background = plt.imread(BACKGROUND_PATH)
    ax.imshow(background, extent=extent)
    for cmap, vf in pairs:
        kw = dict(alpha=0.9, cmap=plt.get_cmap(cmap), extent=extent)
        kw.update(kwargs)
        plot_set(vf, ax=ax, **kw)
    fig.tight_layout()
    return fig

from pathlib import Path

BACKGROUND_PATH = Path(__file__).parent / 'Eskilstuna Intersection.png'

reach_dynamics = Bicycle5D(min_steer=-5*np.pi/4, 
                           max_steer=+5*np.pi/4,
                           min_accel=-0.4,
                           max_accel=+0.4).with_mode('reach')

min_bounds = np.array([0.0, 0.0, -np.pi, -np.pi/5, +0.1])
max_bounds = np.array([8.0, 6.0, +np.pi, +np.pi/5, +1.1])
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                               (61, 46, 31, 7, 11),
                                                               periodic_dims=2)

X, Y, A, D, V = range(grid.ndim)

print('dx =', grid.spacings[0], 
      'dy =', grid.spacings[1],
      'da =', grid.spacings[2],
      'dv =', grid.spacings[4])

timeline = new_timeline(30)

my_tlt = Until(Or('kyrkogatan', 'nygatan'), 'exit_zone')
print('my_tlt:', my_tlt.is_realizable(), list(my_tlt.iter_free()))


impl = HJImpl(reach_dynamics, grid, timeline)
impl.set_axes_names('x', 'y', 'h', 'd', 'v')

out = TLT.construct(my_tlt, 
                    kyrkogatan=kyrkogatan,
                    nygatan=nygatan,
                    exit_zone=exit_zone).realize(impl)

fig = new_map(
    ('Blues', impl.project_onto(out, 1, 2)),
)
fig.show()

breakpoint()