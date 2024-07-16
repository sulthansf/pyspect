######################################################################
## Language: Linear Temporal Logic

from typing import Callable, Generator, TypeAlias, TypeVar


Atomic: TypeAlias   = str
Operator: TypeAlias = str
Formula: TypeAlias  = (Atomic                                   # Proposition
                       | tuple[Operator, 'Formula']             # Unary operation
                       | tuple[Operator, 'Formula', 'Formula']) # Binary operation
FormulaConstructor: TypeAlias = Callable[..., Formula]


AP_TRUE: Atomic     = 'true'

OP_NOT: Operator    = 'NOT'
OP_AND: Operator    = 'AND'
OP_OR: Operator     = 'OR'
OP_UNTIL: Operator  = 'UNTIL'
OP_ALWAYS: Operator = 'ALWAYS'


_EQUIVS: dict[str, FormulaConstructor] = {}


def iter_frml(frml: Formula) -> Generator[Formula]:
    if isinstance(frml, Atomic):
        yield frml
    else:
        _, *args = frml
        yield frml
        for arg in args:
            yield from iter_frml(arg)

def repr_frml(frml: Formula) -> str:
    if isinstance(frml, Atomic):
        return f'<{frml}>'
    else:
        op = frml[0]
        body = ',\n'.join(map(repr_frml, iter_frml(frml)))
        sep = '\n' + ' ' * (len(op)+1)
        body = sep.join(body.splitlines())
        return f'{op}({body})'

def decl_equiv(name: str, equiv: FormulaConstructor) -> FormulaConstructor:
    assert name not in _EQUIVS, 'Equivalent already exists'
    _EQUIVS[name] = equiv
    return lambda *args: (name, *args)

def expand_frml(frml: Formula) -> Formula:
    if isinstance(frml, Atomic):
        return frml
    else:
        op, *args = frml
        return _EQUIVS[op](*args) if op in _EQUIVS else frml

not_    : FormulaConstructor \
        = lambda arg: (OP_NOT, arg)
and_    : FormulaConstructor \
        = lambda lhs, rhs, *rest: and_((OP_AND, lhs, rhs), *rest) if rest else (OP_AND, lhs, rhs)
or_     : FormulaConstructor \
        = lambda lhs, rhs, *rest: or_((OP_OR, lhs, rhs), *rest) if rest else (OP_OR, lhs, rhs)
until_  : FormulaConstructor \
        = lambda lhs, rhs: (OP_UNTIL, lhs, rhs)
always_ : FormulaConstructor \
        = lambda arg: (OP_ALWAYS, arg)

implies_ = decl_equiv('IMPLIES', lambda lhs, rhs: or_(not_(lhs), rhs))
eventually_ = decl_equiv('EVENTUALLY', lambda arg: until_(AP_TRUE, arg))
