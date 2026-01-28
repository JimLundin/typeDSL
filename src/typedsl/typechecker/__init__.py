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
    FieldSchema,
    NodeConstraintGenerator,
    NodeSchema,
    bind_instance,
    extract_schema,
    generate_constraints,
    infer_value_type,
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
    # Schema types (two-phase approach)
    "FieldSchema",
    "Location",
    "NodeConstraintGenerator",
    "NodeSchema",
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
    # Instance binding (Phase 2)
    "bind_instance",
    # Schema extraction (Phase 1)
    "extract_schema",
    # Constraint generation
    "generate_constraints",
    "infer_value_type",
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
