"""Type expressions for the type checker.

This module defines the type representation used by the constraint-based
type checker. It's independent of the schema module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, get_args, get_origin


@dataclass(frozen=True)
class TCon:
    """A type constructor with optional type arguments.

    Examples:
        int        -> TCon(int)
        list[int]  -> TCon(list, (TCon(int),))
        dict[str, float] -> TCon(dict, (TCon(str), TCon(float)))

    """

    con: type[Any]
    args: tuple[TExpr, ...] = ()

    def __repr__(self) -> str:
        if not self.args:
            return f"TCon({self.con.__name__})"
        args_str = ", ".join(repr(a) for a in self.args)
        return f"TCon({self.con.__name__}, ({args_str}))"


@dataclass(frozen=True)
class TVar:
    """A type variable for inference.

    Each TVar has a unique identifier. Fresh TVars are created during
    constraint generation when types are unknown.
    """

    id: int

    def __repr__(self) -> str:
        return f"TVar({self.id})"


type TExpr = TCon | TVar
"""Union type for type expressions."""


class TVarFactory:
    """Factory for generating fresh type variables with unique IDs."""

    def __init__(self, start: int = 0) -> None:
        self._next_id = start

    def fresh(self) -> TVar:
        """Create a fresh type variable with a unique ID."""
        var = TVar(self._next_id)
        self._next_id += 1
        return var


def from_hint(
    hint: Any,
    typevar_map: dict[TypeVar, TVar] | None = None,
    var_factory: TVarFactory | None = None,
) -> TExpr:
    """Convert a Python type hint to a TExpr.

    Uses get_origin() and get_args() to decompose generic types.
    Handles TypeVars by mapping them to TVars.

    Args:
        hint: A Python type hint (e.g., int, list[int], dict[str, float])
        typevar_map: Optional mapping from TypeVar to TVar for consistency.
        var_factory: Optional factory for creating fresh TVars.

    Returns:
        The corresponding TExpr representation.

    Examples:
        >>> from_hint(int)
        TCon(int)
        >>> from_hint(list[int])
        TCon(list, (TCon(int),))

    """
    origin = get_origin(hint)

    if origin is None:
        # Simple type like int, str, or a class
        if isinstance(hint, type):
            return TCon(hint)
        # Handle special cases like None
        if hint is None:
            return TCon(type(None))
        # Handle TypeVar
        if isinstance(hint, TypeVar):
            if typevar_map is not None and hint in typevar_map:
                return typevar_map[hint]
            if var_factory is not None:
                tvar = var_factory.fresh()
                if typevar_map is not None:
                    typevar_map[hint] = tvar
                return tvar
            # No factory provided - use a deterministic ID based on TypeVar name
            # This is a fallback; prefer using var_factory
            return TVar(hash(hint.__name__) % 10000)
        # For other cases (e.g., forward references)
        msg = f"Cannot convert hint to TExpr: {hint!r}"
        raise TypeError(msg)

    # Generic type like list[int], dict[str, float], etc.
    args = get_args(hint)
    converted_args = tuple(from_hint(arg, typevar_map, var_factory) for arg in args)
    return TCon(origin, converted_args)


def texpr_to_str(texpr: TExpr) -> str:
    """Convert a TExpr to a human-readable string.

    Args:
        texpr: A type expression.

    Returns:
        A human-readable string representation.

    """
    match texpr:
        case TVar(id=vid):
            return f"?T{vid}"
        case TCon(con=con, args=()):
            if con is type(None):
                return "None"
            return con.__name__
        case TCon(con=con, args=args):
            args_str = ", ".join(texpr_to_str(a) for a in args)
            return f"{con.__name__}[{args_str}]"
