"""Two-step type checker for typeDSL.

This module provides a constraint-based type checker that works in two phases:

1. Constraint Generation - Traverse the program and collect type equality constraints
2. Constraint Solving - Solve constraints via unification to detect type errors

Example:
    >>> from typedsl import Node, Ref, Program
    >>> from typedsl.checker import check_program

    >>> class Const(Node[int], tag="const"):
    ...     value: int

    >>> class Add(Node[int], tag="add"):
    ...     left: Ref[Node[int]]
    ...     right: Ref[Node[int]]

    >>> prog = Program(
    ...     root=Ref(id="result"),
    ...     nodes={
    ...         "a": Const(value=1),
    ...         "b": Const(value=2),
    ...         "result": Add(left=Ref(id="a"), right=Ref(id="b")),
    ...     },
    ... )
    >>> result = check_program(prog)
    >>> result.success
    True

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typedsl.ast import Program
from typedsl.checker.constraints import EqualityConstraint, Location, SubtypeConstraint
from typedsl.checker.generator import generate_constraints
from typedsl.checker.solver import SolverResult, Substitution, TypeCheckError, solve
from typedsl.checker.types import (
    TCon,
    TExpr,
    TVar,
    TVarFactory,
    from_hint,
    texpr_to_str,
)

if TYPE_CHECKING:
    from typedsl.nodes import Node


@dataclass
class CheckResult:
    """Result of type checking a program.

    Attributes:
        success: Whether the program is well-typed.
        errors: List of type errors (empty if success is True).
        substitution: The final substitution from constraint solving.
        constraints: The constraints that were generated.

    """

    success: bool
    errors: list[TypeCheckError] = field(default_factory=list)
    substitution: Substitution = field(default_factory=Substitution)
    constraints: list[EqualityConstraint | SubtypeConstraint] = field(
        default_factory=list,
    )

    def __str__(self) -> str:
        if self.success:
            return "Type check passed"
        lines = ["Type check failed:"]
        lines.extend(f"  - {error}" for error in self.errors)
        return "\n".join(lines)


def check_program(program: Program) -> CheckResult:
    """Type check a program.

    Generates type constraints from the program and solves them via
    unification to detect type errors.

    Args:
        program: The program to type check.

    Returns:
        A CheckResult indicating success or failure with error details.

    """
    # Step 1: Generate constraints (includes both equality and subtype constraints)
    constraints = generate_constraints(program)

    # Step 2: Solve constraints
    solver_result = solve(constraints)

    return CheckResult(
        success=solver_result.success,
        errors=solver_result.errors,
        substitution=solver_result.substitution,
        constraints=constraints,
    )


def check_node(node: Node[Any]) -> CheckResult:
    """Type check a standalone node tree.

    Wraps the node in a Program and type checks it. Use this for
    simple nested node trees without named references.

    Args:
        node: The root node to type check.

    Returns:
        A CheckResult indicating success or failure with error details.

    """
    program = Program(root=node)
    return check_program(program)


__all__ = [
    "CheckResult",
    # Constraints (for advanced use)
    "EqualityConstraint",
    "Location",
    # Solver (for advanced use)
    "SolverResult",
    "Substitution",
    "SubtypeConstraint",
    # Type expressions (for advanced use)
    "TCon",
    "TExpr",
    "TVar",
    "TVarFactory",
    "TypeCheckError",
    "check_node",
    # Public API
    "check_program",
    "from_hint",
    "generate_constraints",
    "solve",
    "texpr_to_str",
]
