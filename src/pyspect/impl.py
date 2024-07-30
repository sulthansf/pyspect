######################################################################
## Implementation Interface

from typing import TypeVar, Protocol, Optional, runtime_checkable, Tuple, List, Union

__all__ = (
    'ImplProto',
    'ImplEmpty',
    'ImplComplement',
    'ImplIntersect',
    'ImplUnion',
    'ImplRCI',
    'ImplReachForw',
    'ImplReachBack',
    'ImplAxes',
    'ImplPlaneCut',
)

@runtime_checkable
class ImplProto(Protocol):

    @classmethod
    def check(cls, impl):
        assert isinstance(impl, cls), f'Implementation does not support {cls.__name__}'

R = TypeVar('R')
class ImplEmpty(ImplProto, Protocol[R]):
    def empty(self) -> R: ...

R = TypeVar('R')
class ImplComplement(ImplProto, Protocol[R]):
    def complement(self, s: R) -> R: ...

R = TypeVar('R')
class ImplIntersect(ImplProto, Protocol[R]):
    def intersect(self, s1: R, s2: R) -> R: ...

R = TypeVar('R')
class ImplUnion(ImplProto, Protocol[R]):
    def union(self, s1: R, s2: R) -> R: ...

R = TypeVar('R')
class ImplRCI(ImplProto, Protocol[R]):
    def rci(self, s: R) -> R: ...

R = TypeVar('R')
class ImplReachForw(ImplProto, Protocol[R]):
    def reach_forw(self, target: R, constraints: R) -> R: ...

R = TypeVar('R')
class ImplReachBack(ImplProto, Protocol[R]):
    def reach_back(self, target: R, constraints: R) -> R: ...

R = TypeVar('R')
class ImplAxes(ImplProto, Protocol[R]):
    ndim: int
    axes: Tuple[str, ...]
    def axis_is_periodic(self, i: int) -> bool: ...

R = TypeVar('R')
class ImplPlaneCut(ImplAxes[R], ImplProto, Protocol[R]):
    def plane_cut(self, 
                  normal: List[float], 
                  offset: Optional[List[float]] = None, 
                  axes: Optional[Union[int,List[int]]] = None) -> R: ...

