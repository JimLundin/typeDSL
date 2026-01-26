"""Minimal type expressions for constraint-based type checking.

This module defines the constraint type language - a simple representation
used exclusively for type checking. It is separate from the TypeDef hierarchy
which exists for schema serialization.

The constraint type system has only two constructs:
- TVar: An unknown type variable to be solved for
- TCon: A type constructor applied to arguments (e.g., list[int])
"""

from __future__ import annotations

from dataclasses import dataclass


class _VarCounter:
    """Counter for generating fresh variable IDs."""

    def __init__(self) -> None:
        self._counter = 0

    def next_id(self) -> int:
        """Generate the next unique variable ID."""
        self._counter += 1
        return self._counter

    def reset(self) -> None:
        """Reset the counter (useful for testing)."""
        self._counter = 0


# Singleton counter instance
_var_counter = _VarCounter()


def reset_var_counter() -> None:
    """Reset the variable counter (useful for testing)."""
    _var_counter.reset()


@dataclass(frozen=True)
class TVar:
    """An unknown type variable.

    Type variables are created fresh for each generic node instance,
    allowing the same generic class to be instantiated with different
    types in the same program.

    Attributes:
        id: Unique identifier for this variable
        name: Human-readable name (from the TypeVar, e.g., "T")

    """

    id: int
    name: str

    @classmethod
    def fresh(cls, name: str) -> TVar:
        """Create a fresh type variable with a unique ID."""
        return cls(id=_var_counter.next_id(), name=name)

    def __repr__(self) -> str:
        """Return a debug representation of the type variable."""
        return f"?{self.name}_{self.id}"


@dataclass(frozen=True)
class TCon:
    """A type constructor applied to arguments.

    Represents concrete types (int, str) and parameterized types (list[int]).
    The constructor is the actual Python type object for direct equality checking.

    Attributes:
        con: The Python type (e.g., int, list, dict, Node)
        args: Type arguments (empty for non-generic types)

    Examples:
        int         -> TCon(int, ())
        list[int]   -> TCon(list, (TCon(int, ()),))
        dict[str,T] -> TCon(dict, (TCon(str, ()), TVar(1, "T")))

    """

    con: type
    args: tuple[TypeExpr, ...] = ()

    def __repr__(self) -> str:
        """Return a debug representation of the type constructor."""
        if not self.args:
            return self.con.__name__
        args_str = ", ".join(repr(arg) for arg in self.args)
        return f"{self.con.__name__}[{args_str}]"


# Type alias for the constraint type language
type TypeExpr = TVar | TCon
