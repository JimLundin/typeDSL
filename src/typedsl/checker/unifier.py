"""Unification-based constraint solver.

This module implements Phase 2 of the type checker: solving the
constraints generated in Phase 1 via unification.

The unification algorithm finds substitutions (mappings from type
variables to types) that satisfy all equality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from typedsl.checker.constraints import (
    BoundConstraint,
    Constraint,
    EqualityConstraint,
    SourceLocation,
    SubtypeConstraint,
)
from typedsl.checker.errors import (
    BoundViolationError,
    OccursCheckError,
    SubtypeError,
    TypeCheckError,
    UnificationError,
)
from typedsl.checker.types import TCon, TVar, TypeExpr


def occurs(var_id: int, t: TypeExpr) -> bool:
    """Check if a variable occurs in a type expression.

    This is the "occurs check" that prevents infinite types.
    For example, we cannot unify T with list[T] because that
    would create an infinite type.

    Args:
        var_id: The variable ID to check for
        t: The type expression to search in

    Returns:
        True if var_id occurs in t, False otherwise

    """
    if isinstance(t, TVar):
        return t.id == var_id

    if isinstance(t, TCon):
        return any(occurs(var_id, arg) for arg in t.args)

    return False


@dataclass
class Substitution:
    """Maps type variable IDs to their resolved types.

    The substitution is built up during unification as we learn
    what types variables must be. It supports:
    - Following chains (if T -> U -> int, looking up T gives int)
    - Applying to type expressions (replacing variables with their values)
    """

    mapping: dict[int, TypeExpr] = field(default_factory=dict)

    def apply(self, t: TypeExpr) -> TypeExpr:
        """Apply the substitution to a type expression.

        Replaces all type variables with their resolved types,
        following chains as needed.

        Args:
            t: The type expression to apply substitution to

        Returns:
            The type with all known variables substituted

        """
        if isinstance(t, TVar):
            if t.id in self.mapping:
                # Follow the chain recursively
                return self.apply(self.mapping[t.id])
            return t

        if isinstance(t, TCon):
            # Recursively apply to all arguments
            new_args = tuple(self.apply(arg) for arg in t.args)
            return TCon(t.con, new_args)

        return t

    def extend(self, var_id: int, t: TypeExpr) -> None:
        """Extend the substitution with a new binding.

        Args:
            var_id: The variable ID to bind
            t: The type to bind it to

        """
        self.mapping[var_id] = t


@dataclass
class Unifier:
    """Constraint solver using unification.

    Processes a list of constraints and attempts to find a
    substitution that satisfies all of them. Collects errors
    for constraints that cannot be satisfied.
    """

    subst: Substitution = field(default_factory=Substitution)
    errors: list[TypeCheckError] = field(default_factory=list)

    def solve(self, constraints: list[Constraint]) -> bool:
        """Solve all constraints.

        Args:
            constraints: List of constraints to solve

        Returns:
            True if all constraints were satisfied, False otherwise

        """
        for constraint in constraints:
            if isinstance(constraint, EqualityConstraint):
                self._unify(constraint)
            elif isinstance(constraint, SubtypeConstraint):
                self._check_subtype(constraint)
            elif isinstance(constraint, BoundConstraint):
                self._check_bound(constraint)

        return len(self.errors) == 0

    def _unify(self, constraint: EqualityConstraint) -> None:  # noqa: PLR0911
        """Unify two types from an equality constraint.

        This is the core unification algorithm. It recursively
        decomposes types and extends the substitution as needed.
        """
        left = self.subst.apply(constraint.left)
        right = self.subst.apply(constraint.right)

        # Already equal
        if left == right:
            return

        # Left is a variable - bind it
        if isinstance(left, TVar):
            if occurs(left.id, right):
                self.errors.append(
                    OccursCheckError(
                        location=constraint.location,
                        message=f"Infinite type: {left.name} occurs in its definition",
                        var=left,
                        in_type=right,
                    ),
                )
                return
            self.subst.extend(left.id, right)
            return

        # Right is a variable - bind it
        if isinstance(right, TVar):
            if occurs(right.id, left):
                self.errors.append(
                    OccursCheckError(
                        location=constraint.location,
                        message=f"Infinite type: {right.name} occurs in its definition",
                        var=right,
                        in_type=left,
                    ),
                )
                return
            self.subst.extend(right.id, left)
            return

        # Both are type constructors
        if isinstance(left, TCon) and isinstance(right, TCon):
            # Constructors must match
            if left.con is not right.con:
                self._add_unification_error(left, right, constraint)
                return

            # Argument counts must match
            if len(left.args) != len(right.args):
                self._add_unification_error(left, right, constraint)
                return

            # Recursively unify arguments
            for l_arg, r_arg in zip(left.args, right.args, strict=False):
                self._unify(
                    EqualityConstraint(
                        location=constraint.location,
                        reason=constraint.reason,
                        left=l_arg,
                        right=r_arg,
                    ),
                )
            return

        # Cannot unify
        self._add_unification_error(left, right, constraint)

    def _check_subtype(self, constraint: SubtypeConstraint) -> None:
        """Check that sub is a subtype of super_.

        For union types, sub must be assignable to at least one option.
        For other types, we require equality.
        """
        sub = self.subst.apply(constraint.sub)
        super_ = self.subst.apply(constraint.super_)

        # If super is a Union, check if sub matches any option
        if isinstance(super_, TCon) and super_.con is Union:
            if self._is_subtype_of_union(sub, super_.args, constraint.location):
                return
            self.errors.append(
                SubtypeError(
                    location=constraint.location,
                    message="Type is not assignable to union",
                    sub=sub,
                    super_=super_,
                    constraint=constraint,
                ),
            )
            return

        # For non-union supertypes, require equality
        # Try to unify without adding errors first
        if not self._types_equal(sub, super_):
            self.errors.append(
                SubtypeError(
                    location=constraint.location,
                    message="Type is not a subtype",
                    sub=sub,
                    super_=super_,
                    constraint=constraint,
                ),
            )

    def _check_bound(self, constraint: BoundConstraint) -> None:
        """Check that a type variable satisfies its bound."""
        resolved = self.subst.apply(constraint.var)

        # If still a variable, we can't check yet (may be polymorphic)
        if isinstance(resolved, TVar):
            return

        bound = self.subst.apply(constraint.bound)

        # Check if resolved satisfies the bound
        if isinstance(bound, TCon) and bound.con is Union:
            if self._is_subtype_of_union(resolved, bound.args, constraint.location):
                return
        elif self._types_equal(resolved, bound):
            return

        self.errors.append(
            BoundViolationError(
                location=constraint.location,
                message="Type parameter bound violated",
                var=constraint.var,
                resolved=resolved,
                bound=bound,
                constraint=constraint,
            ),
        )

    def _is_subtype_of_union(
        self,
        sub: TypeExpr,
        options: tuple[TypeExpr, ...],
        location: SourceLocation,
    ) -> bool:
        """Check if sub is a subtype of any option in the union."""
        for option in options:
            if self._types_equal(sub, option):
                return True
            # If sub is also a variable, try to unify
            if isinstance(sub, TVar):
                # Save current state
                old_mapping = dict(self.subst.mapping)
                old_errors = len(self.errors)

                # Try unifying
                self._unify(
                    EqualityConstraint(
                        location=location,
                        reason="union member check",
                        left=sub,
                        right=option,
                    ),
                )

                # If no new errors, it worked
                if len(self.errors) == old_errors:
                    return True

                # Restore state and try next option
                self.subst.mapping = old_mapping
                self.errors = self.errors[:old_errors]

        return False

    def _types_equal(self, a: TypeExpr, b: TypeExpr) -> bool:
        """Check if two types are structurally equal."""
        a = self.subst.apply(a)
        b = self.subst.apply(b)
        return a == b

    def _add_unification_error(
        self,
        left: TypeExpr,
        right: TypeExpr,
        constraint: EqualityConstraint,
    ) -> None:
        """Add a unification error."""
        self.errors.append(
            UnificationError(
                location=constraint.location,
                message="Cannot unify types",
                left=left,
                right=right,
                constraint=constraint,
            ),
        )
