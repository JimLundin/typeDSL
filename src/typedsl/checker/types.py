"""Type expressions for the type checker.

This module defines the type representation used by the constraint-based
type checker. It's independent of the schema module.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
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


@dataclass
class TVarFactory:
    """Factory for generating fresh type variables with unique IDs.

    Also tracks bounds for type variables created from bounded TypeVars.
    """

    _next_id: int = 0
    bounds: dict[int, tuple[type, ...]] = field(default_factory=dict)

    def fresh(self) -> TVar:
        """Create a fresh type variable with a unique ID."""
        var = TVar(self._next_id)
        self._next_id += 1
        return var

    def fresh_with_bound(self, bound_types: tuple[type, ...]) -> TVar:
        """Create a fresh type variable with a bound constraint."""
        var = self.fresh()
        self.bounds[var.id] = bound_types
        return var


def get_typevar_bound_types(tv: TypeVar) -> tuple[type, ...] | None:
    """Extract bound types from a TypeVar.

    For `T: int | float`, returns (int, float).
    For unbounded TypeVars, returns None.
    """
    match getattr(tv, "__bound__", None):
        case None:
            return None
        case types.UnionType() as bound:
            return get_args(bound)
        case type() as bound:
            return (bound,)
        case _:
            return None


def from_hint(
    hint: Any,
    typevar_map: dict[TypeVar, TVar] | None = None,
    var_factory: TVarFactory | None = None,
) -> TExpr:
    """Convert a Python type hint to a TExpr.

    Uses get_origin() and get_args() to decompose generic types.
    Handles TypeVars by mapping them to TVars, tracking bounds if present.

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
    # Generic types like list[int], dict[str, float]
    if origin := get_origin(hint):
        args = get_args(hint)
        converted_args = tuple(from_hint(arg, typevar_map, var_factory) for arg in args)
        return TCon(origin, converted_args)

    # Simple type like int, str, or a class
    if isinstance(hint, type):
        return TCon(hint)

    # None literal
    if hint is None:
        return TCon(type(None))

    # TypeVar handling
    if isinstance(hint, TypeVar):
        return _typevar_to_tvar(hint, typevar_map, var_factory)

    msg = f"Cannot convert hint to TExpr: {hint!r}"
    raise TypeError(msg)


def _typevar_to_tvar(
    hint: TypeVar,
    typevar_map: dict[TypeVar, TVar] | None,
    var_factory: TVarFactory | None,
) -> TVar:
    """Convert a TypeVar to a TVar, reusing existing mapping if present."""
    # Check if already mapped
    if typevar_map is not None and hint in typevar_map:
        return typevar_map[hint]

    # Create new TVar
    if var_factory is not None:
        bound_types = get_typevar_bound_types(hint)
        tvar = (
            var_factory.fresh_with_bound(bound_types)
            if bound_types
            else var_factory.fresh()
        )
        if typevar_map is not None:
            typevar_map[hint] = tvar
        return tvar

    # Fallback: deterministic ID based on TypeVar name
    return TVar(hash(hint.__name__) % 10000)


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
