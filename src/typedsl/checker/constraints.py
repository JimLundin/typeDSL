"""Type constraints for the constraint-based type checker.

Constraints express relationships between types that must hold for
a program to be well-typed. They are generated in Phase 1 and
solved in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typedsl.checker.types import TVar, TypeExpr


@dataclass(frozen=True)
class SourceLocation:
    """Location in the AST where a constraint originated.

    Used for error reporting to help users understand where
    type errors occur.

    Attributes:
        node_id: ID of the node (None for inline/root nodes)
        node_type: The Node subclass
        field_name: Name of the field (None for node-level constraints)

    """

    node_id: str | None
    node_type: type
    field_name: str | None = None

    def describe(self) -> str:
        """Human-readable description of the location."""
        if self.node_id:
            node_desc = f"node '{self.node_id}' ({self.node_type.__name__})"
        else:
            node_desc = f"inline {self.node_type.__name__}"

        if self.field_name:
            return f"{node_desc}.{self.field_name}"
        return node_desc


@dataclass(frozen=True)
class Constraint:
    """Base class for type constraints.

    All constraints have a source location for error reporting
    and a reason explaining why the constraint exists.
    """

    location: SourceLocation
    reason: str


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """Two types must be equal (unifiable).

    This is the primary constraint type, generated when:
    - A field value's type must match the field's declared type
    - A child node's return type must match the expected type
    - Type arguments must match during generic instantiation

    The unifier will attempt to find substitutions that make
    left and right equal.
    """

    left: TypeExpr
    right: TypeExpr


@dataclass(frozen=True)
class SubtypeConstraint(Constraint):
    """Subtype relationship: sub must be assignable to super_.

    Generated when:
    - A value is assigned to a union type (int assigned to int | str)
    - Covariant container relationships

    The sub type must be compatible with (a subtype of) super_.
    """

    sub: TypeExpr
    super_: TypeExpr


@dataclass(frozen=True)
class BoundConstraint(Constraint):
    """Type variable must satisfy its declared bound.

    Generated for bounded type parameters like T: int | float.
    When the variable is resolved to a concrete type, that type
    must be a subtype of the bound.

    Attributes:
        var: The type variable with a bound
        bound: The upper bound that var must satisfy

    """

    var: TVar
    bound: TypeExpr


def constraint_summary(constraint: Constraint) -> str:
    """Generate a one-line summary of a constraint for debugging."""
    # Import here to avoid circular dependency
    from typedsl.checker.convert import format_type_expr  # noqa: PLC0415

    if isinstance(constraint, EqualityConstraint):
        left = format_type_expr(constraint.left)
        right = format_type_expr(constraint.right)
        return f"{left} = {right}"

    if isinstance(constraint, SubtypeConstraint):
        sub = format_type_expr(constraint.sub)
        super_ = format_type_expr(constraint.super_)
        return f"{sub} <: {super_}"

    if isinstance(constraint, BoundConstraint):
        bound = format_type_expr(constraint.bound)
        return f"{constraint.var} <: {bound}"

    return str(constraint)
