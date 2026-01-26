"""Constraint-based type checker for typeDSL.

This module provides a two-phase type checker:
1. Phase 1 (Constraint Generation): Traverse the AST and emit type constraints
2. Phase 2 (Unification): Solve constraints to verify type correctness

Example usage:
    from typedsl import Program, Ref
    from typedsl.checker import check_program

    prog = Program(
        root=Ref(id="result"),
        nodes={
            "x": Const(value=42),
            "result": Add(left=Ref(id="x"), right=Const(value=3.14)),
        }
    )

    result = check_program(prog)
    if not result.success:
        print(result.format_errors())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typedsl.ast import Program
from typedsl.checker.errors import TypeCheckResult
from typedsl.checker.generator import ConstraintGenerator
from typedsl.checker.types import reset_var_counter
from typedsl.checker.unifier import Unifier

if TYPE_CHECKING:
    from typedsl.checker.constraints import Constraint
    from typedsl.nodes import Node

__all__ = [
    "TypeCheckResult",
    "check_node",
    "check_program",
]


def check_program(program: Program) -> TypeCheckResult:
    """Type check a complete program.

    Performs two-phase constraint-based type checking:
    1. Generates constraints by traversing the AST
    2. Solves constraints via unification

    Args:
        program: The program to type check

    Returns:
        TypeCheckResult with success status, errors, and resolved types

    Example:
        prog = Program(root=Ref(id="main"), nodes={"main": MyNode(...)})
        result = check_program(prog)
        if not result.success:
            print(result.format_errors())

    """
    # Reset variable counter for deterministic IDs (useful for testing)
    reset_var_counter()

    # Phase 1: Generate constraints
    generator = ConstraintGenerator(program)
    constraints, structural_errors = generator.generate()

    # If there are structural errors (unresolved references), fail early
    if structural_errors:
        return TypeCheckResult(
            success=False,
            errors=structural_errors,
            substitution={},
        )

    # Phase 2: Solve constraints via unification
    unifier = Unifier()
    success = unifier.solve(constraints)

    return TypeCheckResult(
        success=success,
        errors=unifier.errors,
        substitution=unifier.subst.mapping,
    )


def check_node(node: Node[Any]) -> TypeCheckResult:
    """Type check a single node tree.

    Convenience function that wraps the node in a Program
    and delegates to check_program.

    Args:
        node: Root node of the tree to check

    Returns:
        TypeCheckResult with success status, errors, and resolved types

    Example:
        tree = BinOp(left=Const(1), right=Const(2.0))
        result = check_node(tree)

    """
    return check_program(Program(root=node))


def get_constraints(program: Program) -> list[Constraint]:
    """Get the constraints generated for a program (for debugging).

    This is useful for understanding what constraints are being
    generated without running the full type checker.

    Args:
        program: The program to analyze

    Returns:
        List of generated constraints

    """
    reset_var_counter()
    generator = ConstraintGenerator(program)
    constraints, _ = generator.generate()
    return constraints
