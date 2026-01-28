"""Constraint-based type checker using unification."""

from typedsl.typechecker.core import (
    Constraint,
    EqConstraint,
    Location,
    SubConstraint,
    TBot,
    TCon,
    TExp,
    TTop,
    TVar,
    TVarInfo,
)
from typedsl.typechecker.generate import (
    ConstraintGenerator,
    NodeConstraintGenerator,
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
    # Constraints
    "Constraint",
    "ConstraintGenerator",
    "EqConstraint",
    "Location",
    "NodeConstraintGenerator",
    "Solver",
    "SubConstraint",
    # Core types
    "TBot",
    "TCon",
    "TExp",
    "TTop",
    "TVar",
    # Solver state
    "TVarInfo",
    # Errors
    "TypeError",
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
