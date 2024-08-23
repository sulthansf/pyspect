######################################################################
## Set Builder

from typing import TypeVar, Callable, Tuple, Dict
from .langs.base import ImplClientMeta

__all__ = (
    'SetBuilder',
    'ABSURD',
    'Empty',
    'Set',
    'AppliedSet',
    'ReferredSet',
    'HalfSpaceSet',
    'BoundedSet',
)

"""
from abc import ABC, abstractmethod

class _SetBuilder(metaclass=ImplClientMeta):

    @abstractmethod
    def __call__(self, impl: I, **m: '_SetBuilder') -> R: ... 


class EmptySet(_SetBuilder):
    class Impl(ABC):
        @abstractmethod
        def empty(self) -> R: ...

    def __call__(self, impl, *args): impl.empty()

class HalfSpaceSet(_SetBuilder):

    class Impl(ABC):
        @abstractmethod
        def plane_cut(self, normal, offset, axes): ...

    def __init__(self, *args, **kwds):
        self._args = args
        self._kwds = kwds

    def __call__(self, impl, **m):
        return impl.plane_cut(*self._args, **self._kwds)

class BoundedSet(_SetBuilder):

    def __init__(self, **bounds) -> None:
        self._bounds = bounds

    def __call__(self, impl, **m):
        s = impl.complement(impl.empty())
        _bounds = [(vmin, vmax, impl.axis(name))
                   for name, (vmin, vmax) in self._bounds.items()]
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
"""
        
I, R = TypeVar('I'), TypeVar('R')
SetBuilder = Callable[[I, Dict[str, 'SetBuilder[I, R]']], R]

I, R = TypeVar('I'), TypeVar('R')
def ABSURD(impl: I, m: Dict[str, 'SetBuilder[I, R]']) -> R:
    raise ValueError("Cannot realize the absurd set.")

I, R = TypeVar('I'), TypeVar('R')
def Set(arg: R) -> SetBuilder[I, R]:
    return lambda impl, **m: arg

I, R = TypeVar('I'), TypeVar('R')
def ReferredSet(name: str) -> SetBuilder[I, R]:
    return lambda impl, **m: m.pop(name)(impl, **m)

I, R = TypeVar('I'), TypeVar('R')
def AppliedSet(name: str, *builders: SetBuilder[I, R]) -> SetBuilder[I, R]:
    def sb(impl, **m):
        apply = getattr(impl, f'_apply__{name}')
        args = [sb(impl, **m) for sb in builders]
        return apply(*args)
    return sb

I, R = TypeVar('I'), TypeVar('R')
def Empty() -> SetBuilder[I, R]:
    return lambda impl, **m: impl.empty()

I, R = TypeVar('I'), TypeVar('R')
def HalfSpaceSet(*arg, **kwargs): # -> SetBuilder[ImplPlaneCut[R], R]:
    return lambda impl, **m: impl.plane_cut(*arg, **kwargs)

I, R = TypeVar('I', ), TypeVar('R')
def BoundedSet(**bounds: Tuple[float, float]): # -> SetBuilder[_ImplBoundedSet[R], R]:
    def sb(impl, **m: SetBuilder[I, R]) -> R:
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
