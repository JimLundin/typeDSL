"""Constraint-based type checker using unification."""

from typedsl.typechecker.core import (
    Bottom,
    Constraint,
    EqConstraint,
    SourceLocation,
    SubConstraint,
    Top,
    Type,
    TypeCon,
    TypeVar,
    TypeVarInfo,
)
from typedsl.typechecker.operations import (
    is_subtype,
    join,
    meet,
    occurs,
    satisfiable,
    satisfies_bounds,
)
from typedsl.typechecker.solver import (
    Solver,
    TypeError,
    solve,
    typecheck,
)

__all__ = [
    "Bottom",
    # Constraints
    "Constraint",
    "EqConstraint",
    "Solver",
    "SourceLocation",
    "SubConstraint",
    "Top",
    # Core types
    "Type",
    "TypeCon",
    # Errors
    "TypeError",
    "TypeVar",
    # Solver state
    "TypeVarInfo",
    "is_subtype",
    "join",
    "meet",
    # Operations
    "occurs",
    "satisfiable",
    "satisfies_bounds",
    # API
    "solve",
    "typecheck",
]
