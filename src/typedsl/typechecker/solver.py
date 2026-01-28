"""Constraint solver using unification for the type checker."""

from __future__ import annotations

import builtins as _builtins
from dataclasses import dataclass

from typedsl.typechecker.core import (
    Constraint,
    EqConstraint,
    Location,
    SubConstraint,
    TBot,
    TCon,
    TExp,
    TTop,
    TVar,
    TVarInfo,
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
        self._vars: dict[str, TVarInfo] = {}

    def get(self, name: str, default: TExp | None = None) -> TVarInfo:
        """Get or create TVarInfo for a variable."""
        if name not in self._vars:
            self._vars[name] = TVarInfo(lower=TBot(), upper=TTop(), default=default)
        return self._vars[name]

    def add_upper(self, name: str, bound: TExp, loc: Location) -> None:
        """Add upper bound and validate immediately."""
        info = self.get(name)
        info.upper = meet(info.upper, bound)
        self._validate(name, info, loc)

    def add_lower(self, name: str, bound: TExp, loc: Location) -> None:
        """Add lower bound and validate immediately."""
        info = self.get(name)
        info.lower = join(info.lower, bound)
        self._validate(name, info, loc)

    def bind(self, name: str, t: TExp, loc: Location) -> list[Constraint]:
        """Bind variable to exact type (both bounds = t)."""
        info = self.get(name)

        if info.lower == info.upper and not isinstance(info.lower, TBot):
            if info.lower != t:
                return [EqConstraint(info.lower, t, loc)]
            return []

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

    def _validate(self, name: str, info: TVarInfo, loc: Location) -> None:
        """Fail fast on invalid bounds."""
        if not satisfiable(info.lower, info.upper):
            msg = (
                f"Bounds conflict for {name}: "
                f"{info.lower} is not subtype of {info.upper}"
            )
            raise TypeError(msg, loc)
        if info.default and not satisfies_bounds(info.default, info.lower, info.upper):
            msg = f"Default {info.default} violates bounds for {name}"
            raise TypeError(msg, loc)

    def resolve(self, t: TExp) -> TExp:
        """Resolve a type by following TVar bindings."""
        match t:
            case TVar(name):
                if name in self._vars:
                    info = self._vars[name]
                    if info.lower == info.upper and not isinstance(info.lower, TBot):
                        return self.resolve(info.lower)
                return t
            case TCon(con, args) if args:
                resolved_args = tuple(self.resolve(arg) for arg in args)
                if resolved_args != args:
                    return TCon(con, resolved_args)
                return t
            case _:
                return t

    def get_type(self, name: str) -> TExp | None:
        """Get the resolved type for a variable if fully bound."""
        if name not in self._vars:
            return None
        info = self._vars[name]
        if info.lower == info.upper and not isinstance(info.lower, TBot):
            return self.resolve(info.lower)
        return None

    def get_bounds(self, name: str) -> tuple[TExp, TExp] | None:
        """Get the bounds for a variable."""
        if name not in self._vars:
            return None
        info = self._vars[name]
        return (info.lower, info.upper)


def unify_eq(c: EqConstraint, solver: Solver) -> list[Constraint]:
    """Process equality constraint, return new constraints."""
    left, right, loc = c.left, c.right, c.location

    if left == right:
        return []

    match (left, right):
        case (TVar(name, default), _):
            if occurs(name, right):
                msg = f"Infinite type: {name} = {right}"
                raise TypeError(msg, loc)
            solver.get(name, default)
            return solver.bind(name, right, loc)

        case (_, TVar(name, default)):
            if occurs(name, left):
                msg = f"Infinite type: {name} = {left}"
                raise TypeError(msg, loc)
            solver.get(name, default)
            return solver.bind(name, left, loc)

        case (TCon(c1, args1), TCon(c2, args2)):
            if c1 != c2:
                msg = f"Type mismatch: {c1.__name__} vs {c2.__name__}"
                raise TypeError(msg, loc)
            if len(args1) != len(args2):
                msg = f"Arity mismatch for {c1.__name__}: {len(args1)} vs {len(args2)}"
                raise TypeError(msg, loc)
            return [
                EqConstraint(a1, a2, loc) for a1, a2 in zip(args1, args2, strict=False)
            ]

        case (TTop(), TTop()) | (TBot(), TBot()):
            return []

        case _:
            msg = f"Cannot unify {left} with {right}"
            raise TypeError(msg, loc)


def unify_sub(c: SubConstraint, solver: Solver) -> list[Constraint]:
    """Process subtype constraint, return new constraints."""
    sub, sup, loc = c.sub, c.sup, c.location

    if sub == sup:
        return []

    match (sub, sup):
        case (TBot(), _) | (_, TTop()):
            return []

        case (TVar(n1, d1), TVar(n2, d2)):
            solver.get(n1, d1)
            solver.get(n2, d2)
            return [EqConstraint(sub, sup, loc)]

        case (TVar(name, default), _):
            solver.get(name, default)
            solver.add_upper(name, sup, loc)
            return []

        case (_, TVar(name, default)):
            solver.get(name, default)
            solver.add_lower(name, sub, loc)
            return []

        case (TCon(c1, args1), TCon(c2, args2)):
            try:
                if not issubclass(c1, c2):
                    msg = f"{c1.__name__} is not subtype of {c2.__name__}"
                    raise TypeError(msg, loc)
            except _BuiltinTypeError:
                if c1 != c2:
                    msg = f"{c1.__name__} is not subtype of {c2.__name__}"
                    raise TypeError(msg, loc) from None

            if c1 == c2 and args1 and args2:
                if len(args1) != len(args2):
                    msg = (
                        f"Arity mismatch for {c1.__name__}: "
                        f"{len(args1)} vs {len(args2)}"
                    )
                    raise TypeError(msg, loc)
                return [
                    EqConstraint(a1, a2, loc)
                    for a1, a2 in zip(args1, args2, strict=False)
                ]

            return []

        case _:
            msg = f"Cannot establish {sub} <: {sup}"
            raise TypeError(msg, loc)


_BuiltinTypeError = _builtins.TypeError


def solve(constraints: list[Constraint]) -> Solver:
    """Solve constraints, raising TypeError on failure."""
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
    """Type check constraints. Returns None on success, TypeError on failure."""
    try:
        solve(constraints)
        return None
    except TypeError as e:
        return e
