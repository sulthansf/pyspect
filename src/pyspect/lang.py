######################################################################
## Language: Linear Temporal Logic

from collections.abc import Generator
from typing import Callable, TypeAlias


Terminal: TypeAlias = str
Operator: TypeAlias = str
UnOperator: TypeAlias = tuple[Operator, 'SyntaxTree']
BiOperator: TypeAlias = tuple[Operator, 'SyntaxTree', 'SyntaxTree'] 
SyntaxTree: TypeAlias  = (Terminal | UnOperator | BiOperator)
Constructor: TypeAlias = Callable[..., SyntaxTree]


AP_TRUE: Terminal     = 'true'

OP_NOT: Operator    = 'NOT'
OP_AND: Operator    = 'AND'
OP_OR: Operator     = 'OR'
OP_UNTIL: Operator  = 'UNTIL'
OP_ALWAYS: Operator = 'ALWAYS'


_EQUIVS: dict[str, Constructor] = {}


def iter_frml(ast: SyntaxTree) -> Generator[SyntaxTree]:
    if isinstance(ast, Terminal):
        yield ast
    else:
        _, *args = ast
        yield ast
        for arg in args:
            yield from iter_frml(arg)

def decl_equiv(name: str, equiv: Constructor) -> Constructor:
    assert name not in _EQUIVS, 'Equivalent already exists'
    _EQUIVS[name] = equiv
    return lambda *args: (name, *args)

def expand_frml(ast: SyntaxTree) -> SyntaxTree:
    if isinstance(ast, Terminal):
        return ast
    else:
        op, *args = ast
        return _EQUIVS[op](*args) if op in _EQUIVS else ast

not_    : Constructor \
        = lambda arg: (OP_NOT, arg)
and_    : Constructor \
        = lambda lhs, rhs, *rest: and_((OP_AND, lhs, rhs), *rest) if rest else (OP_AND, lhs, rhs)
or_     : Constructor \
        = lambda lhs, rhs, *rest: or_((OP_OR, lhs, rhs), *rest) if rest else (OP_OR, lhs, rhs)
until_  : Constructor \
        = lambda lhs, rhs: (OP_UNTIL, lhs, rhs)
always_ : Constructor \
        = lambda arg: (OP_ALWAYS, arg)

implies_ = decl_equiv('IMPLIES', lambda lhs, rhs: or_(not_(lhs), rhs))
eventually_ = decl_equiv('EVENTUALLY', lambda arg: until_(AP_TRUE, arg))
