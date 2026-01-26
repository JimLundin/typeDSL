"""Error types for the constraint-based type checker.

This module defines the hierarchy of type errors that can occur
during type checking, along with the TypeCheckResult that aggregates
all errors and provides formatting utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typedsl.checker.convert import format_type_expr

if TYPE_CHECKING:
    from typedsl.checker.constraints import (
        BoundConstraint,
        EqualityConstraint,
        SourceLocation,
        SubtypeConstraint,
    )
    from typedsl.checker.types import TVar, TypeExpr

# Constant for error message truncation
_MAX_AVAILABLE_SHOWN = 5


@dataclass(frozen=True)
class TypeCheckError:
    """Base class for type checking errors.

    All errors include a location and human-readable message.
    """

    location: SourceLocation
    message: str

    def format(self) -> str:
        """Format the error for display."""
        return f"{self.location.describe()}: {self.message}"


@dataclass(frozen=True)
class UnificationError(TypeCheckError):
    """Two types could not be unified.

    This occurs when an EqualityConstraint cannot be satisfied
    because the types are structurally incompatible.
    """

    left: TypeExpr
    right: TypeExpr
    constraint: EqualityConstraint

    def format(self) -> str:
        """Format the unification error for display."""
        left_str = format_type_expr(self.left)
        right_str = format_type_expr(self.right)
        return (
            f"{self.location.describe()}: Cannot unify types\n"
            f"  Expected: {right_str}\n"
            f"  Actual:   {left_str}\n"
            f"  Reason:   {self.constraint.reason}"
        )


@dataclass(frozen=True)
class SubtypeError(TypeCheckError):
    """Subtype relationship does not hold.

    This occurs when a SubtypeConstraint cannot be satisfied,
    typically when a value doesn't match a union type.
    """

    sub: TypeExpr
    super_: TypeExpr
    constraint: SubtypeConstraint

    def format(self) -> str:
        """Format the subtype error for display."""
        sub_str = format_type_expr(self.sub)
        super_str = format_type_expr(self.super_)
        return (
            f"{self.location.describe()}: Type is not assignable\n"
            f"  Type:     {sub_str}\n"
            f"  Expected: {super_str}\n"
            f"  Reason:   {self.constraint.reason}"
        )


@dataclass(frozen=True)
class BoundViolationError(TypeCheckError):
    """Resolved type violates its declared bound.

    This occurs when a type variable with a bound (e.g., T: int | float)
    is resolved to a type that doesn't satisfy the bound.
    """

    var: TVar
    resolved: TypeExpr
    bound: TypeExpr
    constraint: BoundConstraint

    def format(self) -> str:
        """Format the bound violation error for display."""
        resolved_str = format_type_expr(self.resolved)
        bound_str = format_type_expr(self.bound)
        return (
            f"{self.location.describe()}: Type parameter bound violated\n"
            f"  Parameter: {self.var.name}\n"
            f"  Resolved:  {resolved_str}\n"
            f"  Bound:     {bound_str}\n"
            f"  Reason:    {self.constraint.reason}"
        )


@dataclass(frozen=True)
class OccursCheckError(TypeCheckError):
    """Type variable occurs in its own definition (infinite type).

    This is detected during unification to prevent infinite types
    like T = list[T].
    """

    var: TVar
    in_type: TypeExpr

    def format(self) -> str:
        """Format the occurs check error for display."""
        type_str = format_type_expr(self.in_type)
        return (
            f"{self.location.describe()}: Infinite type detected\n"
            f"  Variable {self.var.name} occurs in {type_str}"
        )


@dataclass(frozen=True)
class UnresolvedReferenceError(TypeCheckError):
    """Reference points to a non-existent node.

    This is a structural error detected during constraint generation.
    """

    ref_id: str
    available: tuple[str, ...]

    def format(self) -> str:
        """Format the unresolved reference error for display."""
        available_str = ", ".join(self.available[:_MAX_AVAILABLE_SHOWN])
        if len(self.available) > _MAX_AVAILABLE_SHOWN:
            available_str += f" ... ({len(self.available)} total)"
        return (
            f"{self.location.describe()}: Unresolved reference\n"
            f"  Reference: '{self.ref_id}'\n"
            f"  Available: {available_str}"
        )


@dataclass
class TypeCheckResult:
    """Result of type checking a program.

    Contains success status, any errors encountered, and the
    final substitution mapping type variables to resolved types.
    """

    success: bool
    errors: list[TypeCheckError] = field(default_factory=list)
    substitution: dict[int, TypeExpr] = field(default_factory=dict)

    def format_errors(self) -> str:
        """Format all errors for display.

        Returns:
            A multi-line string with all errors formatted.

        """
        if self.success:
            return "Type check passed."

        lines = [f"Type check failed with {len(self.errors)} error(s):\n"]
        for i, error in enumerate(self.errors, 1):
            lines.append(f"[{i}] {error.format()}\n")

        return "\n".join(lines)

    def get_resolved_type(self, var_id: int) -> TypeExpr | None:
        """Get the resolved type for a variable ID.

        Args:
            var_id: The ID of the type variable

        Returns:
            The resolved type, or None if not resolved

        """
        return self.substitution.get(var_id)
