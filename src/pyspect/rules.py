
from abc import ABC, abstractmethod

class Set(ABC): 

    approximation: str

    def __init__(self, approximation: str):
        self.approximation = approximation

    def _overloads(self, name: str):
        return getattr(type(self), name) is getattr(Set, name)

    def _overloaded_methods(self):
        return list(filter(self._overloads, [
            'empty',
            'complement',
            'union',
            'intersect',
            'brs',
            'frs',
            'rci',
            'membership',
            'control_set',
        ]))

    @abstractmethod
    def empty(self): pass 

    def complement(self): pass

    def union(self, s: 'Set'): pass

    def intersect(self, s: 'Set'): pass

    def brs(self): pass

    def frs(self): pass

    def rci(self): pass

    def membership(self, point): pass

    def control_set(self, point): pass

class HJ(Set):

    array: np.ndarray

    def __init__(self, arr, approx):

        self.array = arr
        self.approximation = approx

    def union(self, s):
        return HJ(np.minimum(self.array, s),
                  approximation='under' if s.approximation == 'under' else 'exact')


class Rule(ABC): 

    name: str
    approximation: str

    def __init__(self, *children):
        self.children = children

        self._apply_approximation

    def _apply_approximation(self): 
        """
        1. If default approximation is 'exact', the first child decide type of approximation on parent
        2. If parent is under, no child can be over
        3. If parent is over, no child can be under
        """
        for child in self.children:

            if self.approximation == 'under':
                assert child.approximation != 'over' 
            elif self.approximation == 'over':
                assert child.approximation != 'under' 

            self.approximation = child.approximation

class Proposition(Rule):

    def __init__(self, name, approximation='exact'): 
        self.name = name
        self.approximation = approximation

    def __call__(self, g: dict[str, Set], r: None | dict[str, Set]) -> Set:

        s = g[self.name]

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

class Not(Rule):

    approximation = 'exact'

    def _apply_approximation(self):

        for child in self.children:

            if child.approximation == 'under':
                self.approximation = 'over'
            elif child.approximation == 'over':
                self.approximation = 'under'

class And(Rule):

    approximation = 'exact'

    children: list[Rule]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.children) > 1, 'Must have at least two children'

    def __call__(self, g: dict[str, Set], r: None | dict[str, Set]) -> Set:

        s = ...

        for child in self.children:
            s = s.intersect(child(g, r))

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

class Or(Rule): pass

class Eventually(Rule): 

    approximation = 'under'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.children) == 1, 'Can only have one child'

    def __call__(self, g: dict[str, Set], r: None | dict[str, Set]) -> Set:

        child = self.children[0]

        s = child(g, r).brs()

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

class Always(Rule):
    pass 

class Until(Rule):
    
    approximation = 'under' 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(self.children) == 2, 'Can only have one child'

    def __call__(self, g, r):

        precondition, postcondition = self.children

def solve(rule, save_intermediary=False, **props):

    g: dict[str, Set] = props

    r: None | dict[str, Set] = {} if save_intermediary else None

    s: Set = rule(g, r)

    return s, r

if __name__ == '__main__':

    rule1 = And(Always(Eventually(Proposition('p1'))),
                Always(Proposition('p2')),
                name='phi1')

    s, r = solve(rule1,
                 p1 = HJ([]), 
                 p2 = HJ([], approximation='over'))

    if s.empty():
        print('infeasible!')
    else:
        print('feasible!')

