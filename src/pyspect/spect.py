from abc import ABC, abstractmethod
from typing_extensions import Self

import numpy as np

class Set(ABC): 

    approx: str

    def __init__(self, approx: str = 'exact'):
        self.approx = approx
        assert self.approx in ('over', 'exact', 'under'), 'Invalid approximation type'

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
        s.approx = ('over' if self.approx == 'under' else
                    'under' if self.approx == 'over' else
                    'exact')
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
