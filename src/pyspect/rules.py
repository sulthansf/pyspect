
from abc import ABC, abstractmethod

class Set(ABC): 

    @classmethod
    def check(cls, rule):
        pass

    @abstractmethod
    def complement(self): pass

    @abstractmethod
    def union(self, s: 'Set'): pass

    @abstractmethod
    def intersect(self, s: 'Set'): pass

    @abstractmethod
    def brs(self): pass

    @abstractmethod
    def rci(self): pass


class Rule(ABC): 

    name: str

    @abstractmethod
    def __init__(self, *children: 'Rule'): pass


class Proposition(Rule):

    def __init__(self, name): 
        self.name = name

    def __call__(self, g: dict[str, Set], r: None | dict[str, Set]) -> Set:
        
        s = g[self.name]

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

class Not(Rule):
    pass

class And(Rule):

    children: list[Rule]

    def __call__(self, g: dict[str, Set], r: None | dict[str, Set]) -> Set:

        s = ...

        for child in self.children:
            s = s.intersect(child(g, r))

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

class Eventually(Rule): 

    child: Rule

    def __init__(self, child: Rule): 
        self.child = child 

    def __call__(self, g: dict[str, Set], r: None | dict[str, Set]) -> Set:

        s = self.child(g, r).brs()

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s


def solve(rule, save_intermediary=False, **props):
    
    ....

    g: dict[str, Set] = props

    r: None | dict[str, Set] = {} if save_intermediary else None

    s = rule(g, r)

    return ...

if __name__ == '__main__':


    rule1 = And(Always(Eventually(Proposition('p1'))),
                Always(Proposition('p2')),
                name='phi1')


    result = solve(rule1,
                   p1 = shape1, # np.ndarray 
                   p2 = shape2)


