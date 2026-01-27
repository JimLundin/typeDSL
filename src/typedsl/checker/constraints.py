"""Constraint representation for the type checker.

A constraint represents a type equality requirement between two type
expressions, along with location information for error reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

from typedsl.checker.types import TExpr, texpr_to_str


@dataclass(frozen=True)
class Location:
    """Location information for error reporting.

    Tracks where in the program a constraint originated.

    Attributes:
        node_tag: The tag of the node class where the constraint originated.
        node_id: The ID of the node in the program (if it's a named node).
        field_name: The name of the field being checked.
        path: The path from the program root to this location.

    """

    node_tag: str
    node_id: str | None
    field_name: str | None
    path: tuple[str, ...]

    def __str__(self) -> str:
        parts: list[str] = []
        if self.path:
            parts.append("/".join(self.path))
        if self.node_id:
            parts.append(f"node '{self.node_id}'")
        else:
            parts.append(f"<{self.node_tag}>")
        if self.field_name:
            parts.append(f"field '{self.field_name}'")
        return " -> ".join(parts) if parts else "<unknown location>"


@dataclass(frozen=True)
class EqualityConstraint:
    """A type equality constraint.

    Represents the requirement that two type expressions must unify.

    Attributes:
        left: The expected type (from the declaration).
        right: The actual type (from the value).
        location: Where this constraint originated for error reporting.

    """

    left: TExpr
    right: TExpr
    location: Location

    def __str__(self) -> str:
        left_str = texpr_to_str(self.left)
        right_str = texpr_to_str(self.right)
        return f"{left_str} = {right_str} at {self.location}"


@dataclass(frozen=True)
class SubtypeConstraint:
    """A subtype constraint for bounded type variables.

    Represents the requirement that a type must be one of several allowed types.
    Used for PEP 695 bounded generics like `T: int | float`.

    Attributes:
        type_var: The type variable that must satisfy the bound.
        allowed_types: The concrete types that satisfy the bound.
        location: Where this constraint originated for error reporting.

    """

    type_var: TExpr
    allowed_types: tuple[type, ...]
    location: Location

    def __str__(self) -> str:
        var_str = texpr_to_str(self.type_var)
        types_str = " | ".join(t.__name__ for t in self.allowed_types)
        return f"{var_str} <: {types_str} at {self.location}"


# Type alias for any constraint type
Constraint = EqualityConstraint | SubtypeConstraint
