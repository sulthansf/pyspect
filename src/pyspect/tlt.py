######################################################################
## Temporal Logic Trees

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional

from .lang import *
from .impl import *
from .set_builder import *

__all__ = (
    'TLT',
    'Not',
    'And',
    'Or',
    'Minus',
    'Implies',
    'Until',
    'Always',
)


class APPROX(IntEnum):
    UNDER = -1
    EXACT = 0
    OVER = +1


type ApproxType = Optional[APPROX]
type TLTLike[I, R] = Formula | SetBuilder[I, R] | TLT[I, R]
type LabellingMap[I, R] = dict[str, None | SetBuilder[I, R]]


@dataclass(slots=True, frozen=True)
class TLT[I, R]:
    _formula: Formula           = field(default='_0')
    _builder: SetBuilder[I, R]  = field(default=ABSURD)         # If constructed with the absurd set, then the TLT is also absurd, i.e. cannot be realized.
    _approx: ApproxType         = field(default=APPROX.EXACT)
    _lmap: LabellingMap[I, R]   = field(default_factory=dict)   # Realized sets associated by names for ReferredSets.
    _reqs: tuple[ImplProto]     = field(default_factory=tuple)  # TODO: Not supported yet. See intended use in `realize`.

    def realize(self, impl: I) -> R:
        for req in self._reqs:
            assert req.check(impl), f'Implementation does not support requirement {req}'
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
            return cls(arg.formula, arg.builder, arg.approx, lmap, arg._reqs)

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

        op_f, *args_f = expand_frml(formula)
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
