######################################################################
## Temporal Logic Trees

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, TypeVar, TypeAlias, Generic

from .lang import *
from .impl import *
from .idict import *
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


ApproxType: TypeAlias = Optional[APPROX]

I, R = TypeVar('I'), TypeVar('R')
TLTLike: TypeAlias = SyntaxTree | SetBuilder[I, R] | 'TLT[I, R]'

I, R = TypeVar('I'), TypeVar('R')
LabellingMap: TypeAlias = idict[str, None | SetBuilder[I, R]]


I, R = TypeVar('I'), TypeVar('R')
@dataclass(slots=True, frozen=True)
class TLT(Generic[I, R]):
    _formula: SyntaxTree        = field(default='_0')
    _builder: SetBuilder[I, R]  = field(default=ABSURD)         # If constructed with the absurd set, then the TLT is also absurd, i.e. cannot be realized.
    _approx: ApproxType         = field(default=APPROX.EXACT)
    _lmap: LabellingMap[I, R]   = field(default_factory=idict)  # Sets are associated with names using ReferredSets.
    _reqs: tuple[ImplProto]     = field(default_factory=tuple)  # TODO: Not supported yet. See intended use in `realize`.

    def realize(self, impl: I) -> R:
        for req in self._reqs:
            assert req.check(impl), f'Implementation does not support requirement {req}'
        assert self.is_realizable(), f'Cannot realize TLT, missing {list(self.iter_free())}'
        return self._builder(impl, **self._lmap)

    def is_realizable(self) -> bool:
        for sb in self._lmap.values():
            if sb is None: return False
        return True

    def iter_free(self):
        yield from filter(lambda p: self._lmap[p] is None, self._lmap)

    @classmethod
    def construct(cls, arg: TLTLike[I, R], **m: TLTLike[I, R]) -> 'TLT[I, R]':
        
        if isinstance(arg, TLT):
            # Create new TLT with updated information (we can bind new TLT-like objects to free variables/propositions)
            free = {p: cls.construct(m[p])
                    for p in arg.iter_free() 
                    if p in m and p not in arg._lmap}
            lmap = idict({p: tlt._builder for p, tlt in free.items()} | arg._lmap)
            reqs = tuple({tlt._reqs for tlt in free.values()}.union(arg._reqs))
            return cls(arg._formula, arg._builder, arg._approx, lmap, reqs)

        elif isinstance(arg, Callable): # SetBuilder
            # Assume a formula "_0" where the set exactly represent the prop "_0".
            # In reality, we define a unique ID `uid` instead of "_0". We add one
            # extra level of indirection with a ReferredSet for `uid` and 
            # letting `lmap` hold the actual set builder `arg`. This way, the root
            # TLT will hold a full formula that refers to even constant sets
            # (information is not lost as when internally binding constant sets to
            # builder functions). 
            uid = '_' + hash(arg).to_bytes(8).hex()
            return cls(uid, arg)
        
        elif isinstance(arg, str): # Terminal / Atomic Proposition
            return (cls.construct(m.pop(arg), **m) if arg in m else 
                    cls(arg, ReferredSet(arg), _lmap={arg: None}))
        
        else:
            return cls._construct_from_formula(arg, **m)

    @classmethod
    def _construct_from_formula(cls, formula: SyntaxTree, **m: TLTLike[I, R]) -> 'TLT[I, R]':

        op_f, *args_f = expand_frml(formula)
        args = [cls.construct(subf, **m) for subf in args_f]
        
        if op_f == OP_NOT:
            (_1,) = args
            sb, a = _1._builder, _1._approx
            return cls((op_f, _1._formula),
                       lambda impl, **m: impl.complement(sb(impl, **m)),
                       None if a is None else APPROX(-1 * a),
                       {p: _1._lmap[p] for p in list(_1._lmap)})
        
        elif op_f == OP_AND:
            (_1, _2) = args
            sb1, a1 = _1._builder, _1._approx
            sb2, a2 = _2._builder, _2._approx
            return cls((op_f, _1._formula, _2._formula),
                       lambda impl, **m: impl.intersect(sb1(impl, **m), sb2(impl, **m)),
                       (None if None in (a1, a2) else
                        None if APPROX.UNDER in (a1, a2) else
                        APPROX.OVER),
                       {p: _1._lmap.get(p) or _2._lmap.get(p)
                        for p in list(_1._lmap) + list(_2._lmap)})
         
        elif op_f == OP_OR:
            (_1, _2) = args
            sb1, a1 = _1._builder, _1._approx
            sb2, a2 = _2._builder, _2._approx
            return cls((op_f, _1._formula, _2._formula),
                       lambda impl, **m: impl.union(sb1(impl, **m), sb2(impl, **m)),
                       (None if None in (a1, a2) else
                        a1 if a1 == a2 else
                        a2 if a1 == APPROX.EXACT else
                        a1 if a2 == APPROX.EXACT else
                        None), 
                       {p: _1._lmap.get(p) or _2._lmap.get(p)
                        for p in list(_1._lmap) + list(_2._lmap)})

        elif op_f == OP_UNTIL:
            (_1, _2) = args
            sb1, a1 = _1._builder, _1._approx # constraint
            sb2, a2 = _2._builder, _2._approx # target
            return cls((op_f, _1._formula, _2._formula),
                       lambda impl, **m: impl.reach_forw(sb2(impl, **m), sb1(impl, **m)),
                       APPROX.EXACT, # TODO
                       {p: _1._lmap.get(p) or _2._lmap.get(p)
                        for p in list(_1._lmap) + list(_2._lmap)})
        
        elif op_f == OP_ALWAYS:
            (_1,) = args
            sb, a = _1._builder, _1._approx # target
            return cls((op_f, _1._formula),
                       lambda impl, **m: impl.rci(sb(impl, **m)),
                       APPROX.EXACT, # TODO
                       {p: _1._lmap.get(p) for p in list(_1._lmap)})
        
        else:
            raise ValueError(f'Invalid operator: {op_f}')

# Pseudo-LTL/TLT operations

I, R = TypeVar('I'), TypeVar('R')
def Not(arg: TLTLike[I, R]) -> TLT[I, R]:
    return TLT.construct(not_('_1'), _1=arg)

I, R = TypeVar('I'), TypeVar('R')
def And(lhs: TLTLike[I, R], rhs: TLTLike[I, R], *rest: TLTLike[I, R]) -> TLT[I, R]:
    subtree = TLT.construct(and_('_1', '_2'), _1=lhs, _2=rhs)
    return And(subtree, *rest) if rest else subtree

I, R = TypeVar('I'), TypeVar('R')
def Or(lhs: TLTLike[I, R], rhs: TLTLike[I, R], *rest: TLTLike[I, R]) -> TLT[I, R]:
    subtree = TLT.construct(or_('_1', '_2'), _1=lhs, _2=rhs)
    return Or(subtree, *rest) if rest else subtree

I, R = TypeVar('I'), TypeVar('R')
def Minus(lhs: TLTLike[I, R], rhs: TLTLike[I, R]) -> TLT[I, R]:
    return TLT.construct(or_('_1', not_('_2')), _1=lhs, _2=rhs)

I, R = TypeVar('I'), TypeVar('R')
def Implies(lhs: TLTLike[I, R], rhs: TLTLike[I, R]) -> TLT[I, R]:
    return TLT.construct(implies_('_1', '_2'), _1=lhs, _2=rhs)

I, R = TypeVar('I'), TypeVar('R')
def Until(lhs: TLTLike[I, R], rhs: TLTLike[I, R]) -> TLT[I, R]:
    return TLT.construct(until_('_1', '_2'), _1=lhs, _2=rhs)

I, R = TypeVar('I'), TypeVar('R')
def Always(arg: TLTLike[I, R]) -> TLT[I, R]:
    return TLT.construct(always_('_1'), _1=arg)
