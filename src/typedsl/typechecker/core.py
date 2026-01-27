"""Core type representations for the constraint-based type checker."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Top:
    """Supertype of everything."""


@dataclass(frozen=True)
class Bottom:
    """Subtype of everything."""


@dataclass(frozen=True)
class TypeVar:
    """A type variable that can be unified with other types."""

    name: str
    default: Type | None = None


@dataclass(frozen=True)
class TypeCon:
    """A type constructor with optional type arguments."""

    constructor: type  # actual Python type: int, str, list, etc.
    args: tuple[Type, ...] = ()


# Union type for all type representations
Type = TypeVar | TypeCon | Top | Bottom


@dataclass(frozen=True)
class SourceLocation:
    """Location information for error reporting."""

    description: str


@dataclass(frozen=True)
class EqConstraint:
    """Equality constraint: left = right."""

    left: Type
    right: Type
    location: SourceLocation


@dataclass(frozen=True)
class SubConstraint:
    """Subtype constraint: sub <: sup."""

    sub: Type
    sup: Type
    location: SourceLocation


# Union type for all constraints
Constraint = EqConstraint | SubConstraint


@dataclass
class TypeVarInfo:
    """Tracks bounds for a type variable during solving."""

    lower: Type = field(default_factory=Bottom)  # T must be supertype of this
    upper: Type = field(default_factory=Top)  # T must be subtype of this
    default: Type | None = None
