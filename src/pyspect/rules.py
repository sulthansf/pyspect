from __future__ import annotations
from typing_extensions import Self, Callable, Optional

from .set import Set

__all__ = (
    "Spec",
    "Proposition",
    "Not",
    "And",
    "Or",
    "Until",
    "Always",
    "Implies",
)

U, E, O, I = -1, 0, 1, None
ApproxCheckRet = Optional[int]

class Spec:

    name: str
    children: list[Self]
    max_children: int = -1
    min_children: int = -1

    equiv: Callable[..., Spec] = None

    def __init__(self, *children, name='', early_eval=True):

        self.name = name
        self.children = [Proposition(child) if isinstance(child, str) else
                         Proposition(f'_{i}', child) if isinstance(child, Set) else 
                         child
                         for i, child in enumerate(children)]


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

    def __call__(self, **props):
        # assert self._approxCheck(**props) is not None, 'Invalid approximation'
        return self._eval(**props)
    
    def _eval(self, **props: Set) -> Set:
        assert self.equiv is not None, '"_eval" not implemented'
        return self.equiv(*self.children)._eval(**props)

    def _approxCheck(self, **props: Set) -> ApproxCheckRet:
        assert self.equiv is not None, '"_approxCheck" not implemented'
        return self.equiv(*self.children)._approxCheck(**props)

    def iter_props(self):
        for c in self.children:
            yield from c.iter_props()

class Proposition(Spec):
    """
    Replaced with a set S such that, for proposition p,

        p \in L(S).

    """

    max_children = 0

    def __init__(self, name: str, constant: Set = None):
        self.name = name
        self.constant = constant

    def __repr__(self):
        op = type(self).__name__
        name = self.name
        return f'{op}({name})'

    def __contains__(self, item):
        return item == self.name

    def iter_props(self):
        yield self

    def _eval(self, **props: Set) -> Set:
        # '...' represents the childrens' subtree.
        # a proposition is a leaf node is itself the subtree
        p = props[self.name] if self.name in props else self.constant
        assert p is not None, f'Proposition "{self.name}" not found'
        return p._eval() if isinstance(p, Spec) else p

    def _approxCheck(self, **props):
        return props[self.name].approx

class Not(Spec):
    """
    For the set S that is computed by child, apply the set complement S^C.
    """

    max_children = 1
    min_children = 1

    def _eval(self, **props: Set) -> Set:
        return self.children[0]._eval(**props).complement()
    
    def _approxCheck(self, **props: Set) -> str:
        child_approx = self.children[0]._approxCheck(**props)
        return ('over' if child_approx == 'under' else
                'under' if child_approx == 'over' else
                'exact')

class And(Spec):
    """
    For n >= 2 children there is corresponding sets S_i, i < n, which set intersections is applied on.
    This operation is left-associative.
    """

    min_children = 2

    def _eval(self, **props: Set) -> Set:
        out = self.children[0]._eval(**props)
        for child in self.children[1:]:
            out = out.intersect(child._eval(**props))
        return out
    
    def _approxCheck(self, **props: Set) -> str:
        approx, *child_approxs = [child._approxCheck(**props) for child in self.children]
        for i, child_approx in enumerate(child_approxs):
            child = self.children[i]
            if approx == 'over':
                assert child_approx in ('over', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            elif approx == 'under':
                assert child_approx in ('under', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            else:
                approx = child_approx
        return approx

class Or(Spec):
    """
    For n >= 2 children there is corresponding sets S_i, i < n, which set union is applied on.
    This operation is left-associative.
    """

    min_children = 2

    def _eval(self, **props: Set) -> Set:
        out = self.children[0]._eval(**props)
        for child in self.children[1:]:
            out = out.union(child._eval(**props))
        return out
    
    def _approxCheck(self, **props: Set) -> str:
        approx, *child_approxs = [child._approxCheck(**props) for child in self.children]
        for i, child_approx in enumerate(child_approxs):
            child = self.children[i]
            if approx == 'over':
                assert child_approx in ('over', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            elif approx == 'under':
                assert child_approx in ('under', 'exact'), f'Invalid approximation type "{child_approx}" for:\n{child!r}'
            else:
                approx = child_approx
        return approx

class Until(Spec):
    
    max_children = 2
    min_children = 2

    def _eval(self, **props: Set) -> dict[str, Set]:
        # left U right
        left = self.children[0]._eval(**props)
        right = self.children[1]._eval(**props)
        return right.reach(left)
    
    def _approxCheck(self, **props: Set) -> str:
        left_approx = self.children[0]._approxCheck(**props)
        right_approx = self.children[1]._approxCheck(**props)
        # TODO: Implement approximation type checking for Until
        return 'exact'

class Always(Spec):
    
    max_children = 1

    def _eval(self, **props: Set) -> Set:
        return self.children[0]._eval(**props).rci()

    def _approxCheck(self, **props: Set) -> str:
        left_approx = self.children[0]._approxCheck(**props)
        right_approx = self.children[1]._approxCheck(**props)
        # TODO: Implement approximation type checking for Until
        return 'exact'
    
## ## ## DERIVED OPERATORS ## ## ##

class Implies(Spec):

    max_children = 2
    min_children = 2

    @staticmethod
    def equiv(a, b):
        return Or(Not(a), b)
    