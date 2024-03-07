from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import Self

from .spect import Set

__all__ = ("Rule", "Proposition", "Not", "And", "Or", "Until", "Always")

class Rule(ABC): 

    name: str
    children: list[Self]
    max_children: int = -1
    min_children: int = -1

    def __init__(self, *children, name='', **kwargs):
        children = list(children)
        props = [Proposition(k) for k, v in kwargs.items() if v is Ellipsis]
        kwargs = {k: v for k, v in kwargs.items() if v is not Ellipsis}

        self.name = name
        self.children = children + props

        if self.max_children > 0:
            assert len(self.children) <= self.max_children, f'Too many children for {type(self).__name__}'
        if self.min_children > 0:
            assert len(self.children) >= self.min_children, f'Too few children for {type(self).__name__}'

    def __repr__(self):
        op = type(self).__name__
        body = ',\n'.join(map(repr, self.children))
        sep = '\n' + ' ' * (len(op)+1)
        body = sep.join(body.splitlines())
        return f'{op}({body})'

    def __contains__(self, item):
        for child in self.children:
            if item in child:
                return True
        return False

    def __call__(self, *args, **kwargs):
        return self._eval(*args, **kwargs)
    
    @abstractmethod
    def _eval(self, **leafs: Set) -> dict[str, Set]: pass

    @abstractmethod
    def _check(self, **leafs: Set) -> str: pass

    def iter_props(self):
        for c in self.children:
            yield from c.iter_props()

class Proposition(Rule):
    """
    Replaced with a set S such that, for proposition p,

        p \in L(S).

    """

    max_children = 0

    def __init__(self, name): 
        self.name = name

    def __repr__(self):
        op = type(self).__name__
        name = self.name
        return f'{op}({name})'

    def __contains__(self, item):
        return item == self.name

    def iter_props(self):
        yield self

    def _eval(self, **leafs):
        # '...' represents the childrens' subtree.
        # a proposition is a leaf node is itself the subtree
        return {'...': leafs[self.name]}

    def _check(self, **leafs):
        return leafs[self.name].approx

class Not(Rule):
    """
    For the set S that is computed by child, apply the set complement S^C.
    """

    max_children = 1
    min_children = 1

    def _eval(self, **leafs: Set) -> dict[str, Set]:
        out = self.children[0]._eval(**leafs)
        out['...'] = out['...'].complement()
        return out
    
    def _check(self, **leafs: Set) -> str:
        child_approx = self.children[0]._check(**leafs)
        return ('over' if child_approx == 'under' else
                'under' if child_approx == 'over' else
                'exact')

class And(Rule):
    """
    For n >= 2 children there is corresponding sets S_i, i < n, which set intersections is applied on.
    This operation is left-associative.
    """

    min_children = 2

    def _eval(self, **leafs: Set) -> dict[str, Set]:

        # First child/subtree
        out = self.children[0]._eval(**leafs)
        s = out['...']

        # Subsequent children/subtrees
        for child in self.children[1:]:
            out.update(child._eval(**leafs))
            s = s.intersect(out['...'])

        out['...'] = s
        return out
    
    def _check(self, **leafs: Set) -> str:
        approx, *child_approxs = [child._check(**leafs) for child in self.children]
        for i, child_approx in enumerate(child_approxs):
            child = self.children[i]
            if approx == 'over':
                assert child_approx in ('over', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            elif approx == 'under':
                assert child_approx in ('under', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            else:
                approx = child_approx
        return approx

class Or(Rule):
    """
    For n >= 2 children there is corresponding sets S_i, i < n, which set union is applied on.
    This operation is left-associative.
    """

    min_children = 2

    def _eval(self, **leafs: Set) -> dict[str, Set]:
        out = {}

        # First child/subtree
        out = self.children[0]._eval(**leafs)
        s = out['...']

        # Subsequent children/subtrees
        for child in self.children[1:]:
            out.update(child._eval(**leafs))
            s = s.union(out['...'])

        out['...'] = s
        return out
    
    def _check(self, **leafs: Set) -> str:
        approx, *child_approxs = [child._check(**leafs) for child in self.children]
        for i, child_approx in enumerate(child_approxs):
            child = self.children[i]
            if approx == 'over':
                assert child_approx in ('over', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            elif approx == 'under':
                assert child_approx in ('under', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            else:
                approx = child_approx
        return approx

class Until(Rule):
    
    max_children = 2
    min_children = 2

    def _eval(self, **leafs: Set) -> dict[str, Set]:
        # left U right
        left = self.children[0]._eval(**leafs)
        right = self.children[1]._eval(**leafs)
        out = {**left, **right}
        out['...'] = right['...'].reach(left['...'])
        return out
    
    def _check(self, **leafs: Set) -> str:
        left_approx = self.children[0]._check(**leafs)
        right_approx = self.children[1]._check(**leafs)
        # TODO: Implement approximation type checking for Until
        return 'over'

class Always(Rule):
    
    max_children = 1

    def _eval(self, **leafs: Set) -> dict[str, Set]:
        out = self.children[0]._eval(**leafs)
        out['...'] = out['...'].rci()
        return out

    def _check(self, **leafs: Set) -> str:
        left_approx = self.children[0]._check(**leafs)
        right_approx = self.children[1]._check(**leafs)
        # TODO: Implement approximation type checking for Until
        return 'under'