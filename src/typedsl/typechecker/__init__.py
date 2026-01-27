"""Constraint-based type checker using unification."""

from typedsl.typechecker.core import (
    Bottom,
    Constraint,
    EqConstraint,
    Location,
    SubConstraint,
    Top,
    Type,
    TypeCon,
    TypeVar,
    TypeVarInfo,
)
from typedsl.typechecker.generate import (
    ConstraintGenerator,
    generate_constraints,
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
    "ConstraintGenerator",
    "EqConstraint",
    "Location",
    "Solver",
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
    # Constraint generation
    "generate_constraints",
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
