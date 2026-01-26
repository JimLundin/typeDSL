"""Constraint solver using unification.

This module implements the substitution and unification algorithm for
solving type constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typedsl.checker.constraints import EqualityConstraint, SubtypeConstraint
from typedsl.checker.types import TCon, TExpr, TVar, texpr_to_str

if TYPE_CHECKING:
    from typedsl.checker.constraints import Location


@dataclass
class TypeCheckError:
    """A type error detected during constraint solving.

    Attributes:
        message: Human-readable error message.
        location: Where the error occurred.
        expected: The expected type (if applicable).
        actual: The actual type found (if applicable).

    """

    message: str
    location: Location
    expected: TExpr | None = None
    actual: TExpr | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.expected is not None and self.actual is not None:
            parts.append(
                f"  Expected: {texpr_to_str(self.expected)}, "
                f"got: {texpr_to_str(self.actual)}",
            )
        parts.append(f"  at {self.location}")
        return "\n".join(parts)


class Substitution:
    """A mapping from type variable IDs to type expressions.

    Supports application (replacing variables with their bindings) and
    composition (combining two substitutions).
    """

    def __init__(self, bindings: dict[int, TExpr] | None = None) -> None:
        self._bindings: dict[int, TExpr] = bindings.copy() if bindings else {}

    def __getitem__(self, var_id: int) -> TExpr | None:
        return self._bindings.get(var_id)

    def __contains__(self, var_id: int) -> bool:
        return var_id in self._bindings

    def __repr__(self) -> str:
        items = ", ".join(
            f"?T{k}: {texpr_to_str(v)}" for k, v in self._bindings.items()
        )
        return f"Substitution({{{items}}})"

    def bind(self, var_id: int, texpr: TExpr) -> None:
        """Bind a type variable to a type expression."""
        self._bindings[var_id] = texpr

    def apply(self, texpr: TExpr) -> TExpr:
        """Apply the substitution to a type expression.

        Replaces all type variables in the expression with their mapped
        values, recursively following chains of variable bindings.

        Args:
            texpr: The type expression to apply substitution to.

        Returns:
            The type expression with all known variables replaced.

        """
        match texpr:
            case TVar(id=vid):
                if vid in self._bindings:
                    # Recursively apply to handle chains: T1 -> T2 -> TCon(int)
                    return self.apply(self._bindings[vid])
                return texpr
            case TCon(con=con, args=args):
                if not args:
                    return texpr
                new_args = tuple(self.apply(arg) for arg in args)
                return TCon(con, new_args)

    def compose(self, other: Substitution) -> Substitution:
        """Compose this substitution with another.

        Creates a new substitution where:
        - All bindings from `other` have `self` applied to their values
        - All bindings from `self` that aren't in `other` are included

        Args:
            other: The substitution to compose with.

        Returns:
            A new substitution representing the composition.

        """
        # Apply self to all values in other
        new_bindings = {k: self.apply(v) for k, v in other._bindings.items()}
        # Add bindings from self that aren't in other
        for k, v in self._bindings.items():
            if k not in new_bindings:
                new_bindings[k] = v
        return Substitution(new_bindings)


def occurs_in(var: TVar, texpr: TExpr) -> bool:
    """Check if a type variable occurs within a type expression.

    This is the occurs check, which prevents infinite types like T = list[T].

    Args:
        var: The type variable to check for.
        texpr: The type expression to search in.

    Returns:
        True if the variable occurs in the expression, False otherwise.

    """
    match texpr:
        case TVar(id=vid):
            return var.id == vid
        case TCon(con=_, args=args):
            return any(occurs_in(var, arg) for arg in args)


# Numeric type hierarchy for subtype checking: int <: float <: complex
NUMERIC_SUBTYPES: dict[type, set[type]] = {
    int: {int, float, complex},
    float: {float, complex},
    complex: {complex},
}


def is_subtype(sub: type, sup: type) -> bool:
    """Check if sub is a subtype of sup.

    Handles:
    - Numeric tower: int <: float <: complex
    - Python class hierarchy (e.g., bool <: int)
    """
    if sub == sup:
        return True
    # Check numeric tower first
    if sub in NUMERIC_SUBTYPES:
        return sup in NUMERIC_SUBTYPES[sub]
    # Fall back to Python's class hierarchy
    try:
        return issubclass(sub, sup)
    except TypeError:
        return False


def unify(left: TExpr, right: TExpr) -> Substitution | str:
    """Unify two type expressions.

    Given two type expressions, find a substitution that makes them compatible,
    or return an error message if unification is impossible.

    Handles numeric subtyping: int is compatible with float, float with complex.

    Args:
        left: The expected type expression.
        right: The actual type expression.

    Returns:
        A Substitution that makes the types compatible, or an error string.

    """
    match (left, right):
        # Identical expressions
        case (l, r) if l == r:
            return Substitution()

        # TVar on the left
        case (TVar(id=vid), _):
            if occurs_in(TVar(vid), right):
                left_str = texpr_to_str(left)
                right_str = texpr_to_str(right)
                return f"Infinite type: {left_str} occurs in {right_str}"
            sub = Substitution()
            sub.bind(vid, right)
            return sub

        # TVar on the right
        case (_, TVar(id=vid)):
            if occurs_in(TVar(vid), left):
                left_str = texpr_to_str(left)
                right_str = texpr_to_str(right)
                return f"Infinite type: {right_str} occurs in {left_str}"
            sub = Substitution()
            sub.bind(vid, left)
            return sub

        # Both TCon
        case (TCon(con=lcon, args=largs), TCon(con=rcon, args=rargs)):
            # Check for numeric subtyping (right is subtype of left)
            if lcon != rcon:
                if is_subtype(rcon, lcon):
                    # Actual type is subtype of expected - compatible
                    return Substitution()
                return f"Type mismatch: {texpr_to_str(left)} vs {texpr_to_str(right)}"

            # Arities must match
            if len(largs) != len(rargs):
                return (
                    f"Arity mismatch: {texpr_to_str(left)} has {len(largs)} args, "
                    f"{texpr_to_str(right)} has {len(rargs)}"
                )

            # Recursively unify arguments left-to-right
            result = Substitution()
            for larg, rarg in zip(largs, rargs, strict=False):
                # Apply current substitution before unifying
                larg_sub = result.apply(larg)
                rarg_sub = result.apply(rarg)
                arg_result = unify(larg_sub, rarg_sub)
                if isinstance(arg_result, str):
                    return arg_result
                result = result.compose(arg_result)

            return result

    # This should be unreachable due to the union type
    return f"Cannot unify {texpr_to_str(left)} with {texpr_to_str(right)}"


@dataclass
class SolverResult:
    """Result of constraint solving.

    Attributes:
        success: Whether all constraints were solved successfully.
        substitution: The final substitution (on success).
        errors: List of type errors encountered.

    """

    success: bool
    substitution: Substitution = field(default_factory=Substitution)
    errors: list[TypeCheckError] = field(default_factory=list)


def check_subtype_constraint(
    constraint: SubtypeConstraint,
    substitution: Substitution,
) -> str | None:
    """Check if a subtype constraint is satisfied.

    Args:
        constraint: The subtype constraint to check.
        substitution: The current substitution.

    Returns:
        An error message if the constraint is violated, None otherwise.

    """
    resolved = substitution.apply(constraint.type_var)

    # If still a variable, constraint can't be checked yet
    if isinstance(resolved, TVar):
        return None

    # Must be a TCon - check if it's one of the allowed types
    if isinstance(resolved, TCon) and not resolved.args:
        actual_type = resolved.con
        # Check if actual_type is one of the bounds or a subtype of one
        is_valid = any(
            is_subtype(actual_type, bound_type)
            for bound_type in constraint.allowed_types
        )
        if not is_valid:
            bound_names = " | ".join(t.__name__ for t in constraint.allowed_types)
            return f"Type {actual_type.__name__} does not satisfy bound {bound_names}"

    return None


def solve(
    constraints: list[EqualityConstraint | SubtypeConstraint],
) -> SolverResult:
    """Solve a list of type constraints.

    Processes constraints in order, applying unification for equality constraints
    and checking bounds for subtype constraints. Records errors with location
    information when constraints are violated.

    Args:
        constraints: The list of constraints to solve. Can include both
            equality constraints (EqualityConstraint) and subtype constraints
            (SubtypeConstraint).

    Returns:
        A SolverResult indicating success/failure and any errors.

    """
    result = Substitution()
    errors: list[TypeCheckError] = []
    subtype_constraints: list[SubtypeConstraint] = []

    # First pass: process equality constraints and collect subtype constraints
    for constraint in constraints:
        if isinstance(constraint, SubtypeConstraint):
            subtype_constraints.append(constraint)
            continue

        # Handle equality constraint
        left = result.apply(constraint.left)
        right = result.apply(constraint.right)

        unify_result = unify(left, right)

        if isinstance(unify_result, str):
            errors.append(
                TypeCheckError(
                    message=unify_result,
                    location=constraint.location,
                    expected=left,
                    actual=right,
                ),
            )
        else:
            result = result.compose(unify_result)

    # Second pass: check subtype constraints with final substitution
    for constraint in subtype_constraints:
        error_msg = check_subtype_constraint(constraint, result)
        if error_msg:
            errors.append(
                TypeCheckError(
                    message=error_msg,
                    location=constraint.location,
                ),
            )

    return SolverResult(
        success=len(errors) == 0,
        substitution=result,
        errors=errors,
    )
