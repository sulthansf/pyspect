from abc import ABC, abstractmethod
from typing_extensions import Self, Optional

import numpy as np

__all__ = ('Set',)

class NaN:
    def __repr__(self):
        return 'NaN'
    def __eq__(self, other):
        return isinstance(other, NaN)
    def __ne__(self, other):
        return not isinstance(other, NaN)
    def __mul__(self, other):
        assert isinstance(other, int), 'Invalid operation'
        return NaN()
    def __rmul__(self, other):
        assert isinstance(other, int), 'Invalid operation'
        return NaN()
NAN = NaN()
U, E, O, I = -1, 0, 1, NAN
STR2APPROX = dict(U=U,     E=E,     O=O,    I=I,
                  under=U, exact=E, over=O, invalid=I)

class Set(ABC):

    approx: float

    def __init__(self, approx: str = 'exact'):
        self.approx = STR2APPROX[approx] if approx in STR2APPROX else approx
        assert self.approx in (U, E, O, I), 'Invalid approximation type'

    @classmethod
    def _overloads(cls, name: str):
        return getattr(cls, name) is not getattr(Set, name)

    @classmethod
    def _overloaded_methods(cls):
        ops = [name for name in dir(cls) 
               if not name.startswith('_') and callable(getattr(Set, name))]
        return list(filter(cls._overloads, ops))
    
    @abstractmethod
    def copy(self) -> Self: pass

    @classmethod
    @abstractmethod
    def empty(self) -> Self: pass

    @abstractmethod
    def is_empty(self) -> bool: pass

    @abstractmethod
    def membership(self, point: np.ndarray) -> bool: pass

    def complement(self) -> Self:
        s = self.copy()
        s.approx = None if self.approx is None else -1 * self.approx
        return s

    def union(self, other: Self) -> Self:
        s = self.copy()
        if self.approx == 'over':
            assert other.approx in ('over', 'exact'), 'Invalid approximation type'
            s.approx = self.approx
        elif self.approx == 'under':
            assert other.approx in ('under', 'exact'), 'Invalid approximation type'
            s.approx = self.approx
        else:
            s.approx = other.approx
        return s

    def intersect(self, other: Self) -> Self:
        s = self.copy()
        if self.approx == 'over':
            assert other.approx in ('over', 'exact'), 'Invalid approximation type'
            s.approx = self.approx
        elif self.approx == 'under':
            assert other.approx in ('under', 'exact'), 'Invalid approximation type'
            s.approx = self.approx
        else:
            s.approx = other.approx
        return s

    def reach(self, constraints: Self) -> Self:
        s = self.copy()
        s.approx = 'over'
        return s

    def rci(self) -> Self:
        s = self.copy()
        s.approx = 'under'
        return s
