"""Core type representations for the constraint-based type checker."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TTop:
    """Supertype of everything."""


@dataclass(frozen=True)
class TBot:
    """Subtype of everything."""


@dataclass(frozen=True)
class TVar:
    """A type variable that can be unified with other types."""

    name: str
    default: TExp | None = None


@dataclass(frozen=True)
class TCon:
    """A type constructor with optional type arguments."""

    con: type  # actual Python type: int, str, list, etc.
    args: tuple[TExp, ...] = ()


type TExp = TVar | TCon | TTop | TBot


@dataclass(frozen=True)
class Location:
    """Location information for error reporting."""

    path: str

    def child(self, segment: str) -> Location:
        """Create a child location by appending a path segment."""
        return Location(f"{self.path}.{segment}")

    def index(self, idx: int | str) -> Location:
        """Create a child location for an indexed access."""
        return Location(f"{self.path}[{idx!r}]")


@dataclass(frozen=True)
class EqConstraint:
    """Equality constraint: left = right."""

    left: TExp
    right: TExp
    location: Location


@dataclass(frozen=True)
class SubConstraint:
    """Subtype constraint: sub <: sup."""

    sub: TExp
    sup: TExp
    location: Location


type Constraint = EqConstraint | SubConstraint


@dataclass
class TVarInfo:
    """Tracks bounds for a type variable during solving."""

    lower: TExp = field(default_factory=TBot)
    upper: TExp = field(default_factory=TTop)
    default: TExp | None = None
