######################################################################
## Set Builder

from typing_protocol_intersection import ProtocolIntersection as Has
from typing import TypeVar, Callable, Tuple
from typing_extensions import TypeAlias
from .impl import *

__all__ = (
    'SetBuilder',
    'ABSURD',
    'Empty',
    'Set',
    'ReferredSet',
    'HalfSpaceSet',
    'BoundedSet',
)

I, R = TypeVar('I'), TypeVar('R')
SetBuilder: TypeAlias = Callable[[I], R] # Args of form: (impl: Impl, **m: SetBuilder)

I, R = TypeVar('I'), TypeVar('R')
def ABSURD(impl: I, **m: SetBuilder[I, R]) -> R:
    raise ValueError("Cannot realize the absurd set.")

I, R = TypeVar('I'), TypeVar('R')
def Empty() -> SetBuilder[ImplEmpty[R], R]:
    return lambda impl, **m: impl.empty()

I, R = TypeVar('I'), TypeVar('R')
def Set(arg: R) -> SetBuilder[I, R]:
    return lambda impl, **m: arg

I, R = TypeVar('I'), TypeVar('R')
def ReferredSet(name: str) -> SetBuilder[I, R]:
    return lambda impl, **m: m.pop(name)(impl, **m)

I, R = TypeVar('I'), TypeVar('R')
def HalfSpaceSet(*arg, **kwargs) -> SetBuilder[ImplPlaneCut[R], R]:
    return lambda impl, **m: impl.plane_cut(*arg, **kwargs)

I, R = TypeVar('I'), TypeVar('R')
_ImplBoundedSet: TypeAlias = Has[ImplEmpty[R], ImplAxes[R], ImplPlaneCut[R], ImplComplement[R], ImplIntersect[R]]
def BoundedSet(**bounds: Tuple[float, float]) -> SetBuilder[_ImplBoundedSet[R], R]:
    def sb(impl: _ImplBoundedSet[R], **m: SetBuilder[I, R]) -> R:
        s = impl.complement(impl.empty())
        _bounds = [(vmin, vmax, impl.axis(name))
                   for name, (vmin, vmax) in bounds.items()]
        for vmin, vmax, i in _bounds:
            if vmax < vmin and impl.axis_is_periodic(i):
                upper_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                axis_range = impl.complement(impl.intersect(upper_bound, lower_bound))
            else:
                upper_bound = impl.plane_cut(normal=[0 if i != j else -1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmax for j in range(impl.ndim)])
                lower_bound = impl.plane_cut(normal=[0 if i != j else +1 for j in range(impl.ndim)],
                                             offset=[0 if i != j else vmin for j in range(impl.ndim)])
                axis_range = impl.intersect(upper_bound, lower_bound)
            s = impl.intersect(s, axis_range)
        return s
    return sb
