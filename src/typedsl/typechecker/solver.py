"""Constraint solver using unification for the type checker."""

from __future__ import annotations

import builtins as _builtins
from dataclasses import dataclass

from typedsl.typechecker.core import (
    Bottom,
    Constraint,
    EqConstraint,
    Location,
    SubConstraint,
    Top,
    Type,
    TypeCon,
    TypeVar,
    TypeVarInfo,
)
from typedsl.typechecker.operations import (
    join,
    meet,
    occurs,
    satisfiable,
    satisfies_bounds,
)


@dataclass
class TypeError(Exception):
    """Type error with location information."""

    message: str
    location: Location

    def __str__(self) -> str:
        return f"{self.message} at {self.location.path}"


class Solver:
    """Constraint solver that tracks type variable bounds."""

    def __init__(self) -> None:
        self._vars: dict[str, TypeVarInfo] = {}

    def get(self, name: str, default: Type | None = None) -> TypeVarInfo:
        """Get or create TypeVarInfo for a variable.

        Args:
            name: The type variable name.
            default: Optional default type for the variable.

        Returns:
            The TypeVarInfo for this variable.

        """
        if name not in self._vars:
            self._vars[name] = TypeVarInfo(
                lower=Bottom(),
                upper=Top(),
                default=default,
            )
        return self._vars[name]

    def add_upper(self, name: str, bound: Type, loc: Location) -> None:
        """Add upper bound and validate immediately.

        Args:
            name: The type variable name.
            bound: The upper bound to add.
            loc: Source location for error reporting.

        Raises:
            TypeError: If the bounds become unsatisfiable.

        """
        info = self.get(name)
        info.upper = meet(info.upper, bound)
        self._validate(name, info, loc)

    def add_lower(self, name: str, bound: Type, loc: Location) -> None:
        """Add lower bound and validate immediately.

        Args:
            name: The type variable name.
            bound: The lower bound to add.
            loc: Source location for error reporting.

        Raises:
            TypeError: If the bounds become unsatisfiable.

        """
        info = self.get(name)
        info.lower = join(info.lower, bound)
        self._validate(name, info, loc)

    def bind(self, name: str, t: Type, loc: Location) -> list[Constraint]:
        """Bind variable to exact type (both bounds = t).

        Args:
            name: The type variable name.
            t: The type to bind to.
            loc: Source location for error reporting.

        Returns:
            New constraints generated from the binding.

        Raises:
            TypeError: If the binding conflicts with existing bounds.

        """
        info = self.get(name)

        # Check if already bound to something
        if info.lower == info.upper and not isinstance(info.lower, Bottom):
            # Already bound - generate equality constraint between old and new
            if info.lower != t:
                return [EqConstraint(info.lower, t, loc)]
            return []

        # Check that t is compatible with existing bounds
        if not satisfies_bounds(t, info.lower, info.upper):
            msg = (
                f"Cannot bind {name} to {t}: "
                f"violates bounds [{info.lower}, {info.upper}]"
            )
            raise TypeError(msg, loc)

        info.lower = t
        info.upper = t
        self._validate(name, info, loc)
        return []

    def _validate(self, name: str, info: TypeVarInfo, loc: Location) -> None:
        """Fail fast on invalid bounds.

        Args:
            name: The type variable name.
            info: The TypeVarInfo to validate.
            loc: Source location for error reporting.

        Raises:
            TypeError: If bounds are unsatisfiable or default violates bounds.

        """
        if not satisfiable(info.lower, info.upper):
            msg = (
                f"Bounds conflict for {name}: "
                f"{info.lower} is not subtype of {info.upper}"
            )
            raise TypeError(msg, loc)
        if info.default and not satisfies_bounds(info.default, info.lower, info.upper):
            msg = f"Default {info.default} violates bounds for {name}"
            raise TypeError(
                msg,
                loc,
            )

    def resolve(self, t: Type) -> Type:
        """Resolve a type by following TypeVar bindings.

        Args:
            t: The type to resolve.

        Returns:
            The fully resolved type.

        """
        match t:
            case TypeVar(name):
                if name in self._vars:
                    info = self._vars[name]
                    if info.lower == info.upper and not isinstance(info.lower, Bottom):
                        # Follow the binding chain
                        return self.resolve(info.lower)
                return t
            case TypeCon(constructor, args) if args:
                # Recursively resolve args
                resolved_args = tuple(self.resolve(arg) for arg in args)
                if resolved_args != args:
                    return TypeCon(constructor, resolved_args)
                return t
            case _:
                return t

    def get_type(self, name: str) -> Type | None:
        """Get the resolved type for a variable if fully bound.

        Args:
            name: The type variable name.

        Returns:
            The bound type if lower == upper, otherwise None.

        """
        if name not in self._vars:
            return None
        info = self._vars[name]
        if info.lower == info.upper and not isinstance(info.lower, Bottom):
            return self.resolve(info.lower)
        return None

    def get_bounds(self, name: str) -> tuple[Type, Type] | None:
        """Get the bounds for a variable.

        Args:
            name: The type variable name.

        Returns:
            Tuple of (lower, upper) bounds, or None if variable not tracked.

        """
        if name not in self._vars:
            return None
        info = self._vars[name]
        return (info.lower, info.upper)


def unify_eq(c: EqConstraint, solver: Solver) -> list[Constraint]:
    """Process equality constraint, return new constraints.

    Args:
        c: The equality constraint to process.
        solver: The solver state.

    Returns:
        List of new constraints generated.

    Raises:
        TypeError: If unification fails.

    """
    left, right, loc = c.left, c.right, c.location

    # Trivial case
    if left == right:
        return []

    match (left, right):
        # TypeVar on left: bind it
        case (TypeVar(name, default), _):
            if occurs(name, right):
                msg = f"Infinite type: {name} = {right}"
                raise TypeError(msg, loc)
            # Register with default if present
            solver.get(name, default)
            return solver.bind(name, right, loc)

        # TypeVar on right: bind it
        case (_, TypeVar(name, default)):
            if occurs(name, left):
                msg = f"Infinite type: {name} = {left}"
                raise TypeError(msg, loc)
            # Register with default if present
            solver.get(name, default)
            return solver.bind(name, left, loc)

        # TypeCon decomposition
        case (TypeCon(c1, args1), TypeCon(c2, args2)):
            if c1 != c2:
                msg = f"Type mismatch: {c1.__name__} vs {c2.__name__}"
                raise TypeError(msg, loc)
            if len(args1) != len(args2):
                msg = f"Arity mismatch for {c1.__name__}: {len(args1)} vs {len(args2)}"
                raise TypeError(
                    msg,
                    loc,
                )
            # Decompose into constraints on args
            return [
                EqConstraint(a1, a2, loc) for a1, a2 in zip(args1, args2, strict=False)
            ]

        # Top/Bottom cases
        case (Top(), Top()) | (Bottom(), Bottom()):
            return []

        case _:
            msg = f"Cannot unify {left} with {right}"
            raise TypeError(msg, loc)


def unify_sub(c: SubConstraint, solver: Solver) -> list[Constraint]:
    """Process subtype constraint, return new constraints.

    Args:
        c: The subtype constraint to process.
        solver: The solver state.

    Returns:
        List of new constraints generated.

    Raises:
        TypeError: If subtype relation cannot be established.

    """
    sub, sup, loc = c.sub, c.sup, c.location

    # Trivial case
    if sub == sup:
        return []

    match (sub, sup):
        # Bottom <: anything, anything <: Top
        case (Bottom(), _) | (_, Top()):
            return []

        # Both TypeVar: treat as equality (simplification)
        case (TypeVar(n1, d1), TypeVar(n2, d2)):
            solver.get(n1, d1)
            solver.get(n2, d2)
            return [EqConstraint(sub, sup, loc)]

        # TypeVar <: T: add upper bound
        case (TypeVar(name, default), _):
            solver.get(name, default)
            solver.add_upper(name, sup, loc)
            return []

        # T <: TypeVar: add lower bound
        case (_, TypeVar(name, default)):
            solver.get(name, default)
            solver.add_lower(name, sub, loc)
            return []

        # TypeCon subtyping
        case (TypeCon(c1, args1), TypeCon(c2, args2)):
            try:
                if not issubclass(c1, c2):
                    msg = f"{c1.__name__} is not subtype of {c2.__name__}"
                    raise TypeError(
                        msg,
                        loc,
                    )
            except _BuiltinTypeError:
                if c1 != c2:
                    msg = f"{c1.__name__} is not subtype of {c2.__name__}"
                    raise TypeError(
                        msg,
                        loc,
                    ) from None

            # If same constructor, check args invariantly
            if c1 == c2 and args1 and args2:
                if len(args1) != len(args2):
                    msg = (
                        f"Arity mismatch for {c1.__name__}: "
                        f"{len(args1)} vs {len(args2)}"
                    )
                    raise TypeError(msg, loc)
                # Invariant: generate equality constraints for args
                return [
                    EqConstraint(a1, a2, loc)
                    for a1, a2 in zip(args1, args2, strict=False)
                ]

            return []

        case _:
            msg = f"Cannot establish {sub} <: {sup}"
            raise TypeError(msg, loc)


# Alias for Python's built-in TypeError to distinguish from ours
_BuiltinTypeError = _builtins.TypeError


def solve(constraints: list[Constraint]) -> Solver:
    """Solve constraints, raising TypeError on failure.

    Args:
        constraints: List of constraints to solve.

    Returns:
        The solver with resolved type variable bounds.

    Raises:
        TypeError: If constraints cannot be satisfied.

    """
    solver = Solver()
    worklist = list(constraints)

    while worklist:
        c = worklist.pop()
        match c:
            case EqConstraint():
                new = unify_eq(c, solver)
            case SubConstraint():
                new = unify_sub(c, solver)
        worklist.extend(new)

    return solver


def typecheck(constraints: list[Constraint]) -> TypeError | None:
    """Type check constraints.

    Args:
        constraints: List of constraints to check.

    Returns:
        None on success, TypeError on failure.

    """
    try:
        solve(constraints)
        return None
    except TypeError as e:
        return e
