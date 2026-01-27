"""Type lattice operations for the constraint-based type checker."""

from __future__ import annotations

from typedsl.typechecker.core import Bottom, Top, Type, TypeCon, TypeVar


def occurs(name: str, t: Type) -> bool:
    """Check if type variable occurs in type (prevents infinite types).

    Args:
        name: The type variable name to search for.
        t: The type to search within.

    Returns:
        True if the type variable occurs in the type.

    """
    match t:
        case TypeVar(n):
            return n == name
        case TypeCon(_, args):
            return any(occurs(name, a) for a in args)
        case Top() | Bottom():
            return False
    # Should be unreachable, but satisfy type checker
    return False


def is_subtype(sub: Type, sup: Type) -> bool:
    """Check if sub is a subtype of sup.

    Args:
        sub: The potential subtype.
        sup: The potential supertype.

    Returns:
        True if sub is a subtype of sup.

    """
    match (sub, sup):
        # Bottom is subtype of everything
        case (Bottom(), _):
            return True
        # Everything is subtype of Top
        case (_, Top()):
            return True
        # Top is only subtype of Top (handled above)
        case (Top(), _):
            return False
        # Bottom is only supertype of Bottom
        case (_, Bottom()):
            return False
        # Type variables: only equal to themselves
        case (TypeVar(n1), TypeVar(n2)):
            return n1 == n2
        # TypeVar vs TypeCon: not subtypes
        case (TypeVar(_), TypeCon(_, _)) | (TypeCon(_, _), TypeVar(_)):
            return False
        # TypeCon subtyping
        case (TypeCon(c1, args1), TypeCon(c2, args2)):
            # Check constructor subtyping via issubclass
            try:
                if not issubclass(c1, c2):
                    return False
            except TypeError:
                # issubclass can raise for non-class types
                if c1 != c2:
                    return False

            # If same constructor, check args (invariant)
            if c1 == c2:
                if len(args1) != len(args2):
                    return False
                # Invariant: all args must be equal
                return all(
                    is_subtype(a1, a2) and is_subtype(a2, a1)
                    for a1, a2 in zip(args1, args2, strict=False)
                )

            # Different constructors but c1 <: c2 (e.g., bool <: int)
            # Only valid if no type args
            return len(args1) == 0 and len(args2) == 0

    # Should be unreachable
    return False


def meet(a: Type, b: Type) -> Type:
    """Compute greatest lower bound (most specific common subtype).

    Args:
        a: First type.
        b: Second type.

    Returns:
        The greatest lower bound of a and b.

    """
    match (a, b):
        # Top is identity for meet
        case (Top(), x) | (x, Top()):
            return x
        # Bottom absorbs
        case (Bottom(), _) | (_, Bottom()):
            return Bottom()
        # Same types
        case _ if a == b:
            return a
        # TypeCon with same constructor
        case (TypeCon(c1, args1), TypeCon(c2, args2)) if c1 == c2:
            if len(args1) != len(args2):
                return Bottom()
            # Meet the args (invariant means they must be equal)
            new_args = []
            for a1, a2 in zip(args1, args2, strict=False):
                m = meet(a1, a2)
                if isinstance(m, Bottom):
                    return Bottom()
                new_args.append(m)
            return TypeCon(c1, tuple(new_args))
        # Different TypeCon constructors
        case (TypeCon(c1, args1), TypeCon(c2, args2)):
            # Check if one is subtype of the other
            try:
                if issubclass(c1, c2) and len(args1) == 0 and len(args2) == 0:
                    return TypeCon(c1, ())
                if issubclass(c2, c1) and len(args1) == 0 and len(args2) == 0:
                    return TypeCon(c2, ())
            except TypeError:
                pass
            return Bottom()
        # TypeVar cases - can't compute meet without more info
        case (TypeVar(_), _) | (_, TypeVar(_)):
            # Conservative: if not equal, return Bottom
            return Bottom()

    return Bottom()


def join(a: Type, b: Type) -> Type:
    """Compute least upper bound (most general common supertype).

    Args:
        a: First type.
        b: Second type.

    Returns:
        The least upper bound of a and b.

    """
    match (a, b):
        # Bottom is identity for join
        case (Bottom(), x) | (x, Bottom()):
            return x
        # Top absorbs
        case (Top(), _) | (_, Top()):
            return Top()
        # Same types
        case _ if a == b:
            return a
        # TypeCon with same constructor
        case (TypeCon(c1, args1), TypeCon(c2, args2)) if c1 == c2:
            if len(args1) != len(args2):
                return Top()
            # Join the args (invariant means they must be equal)
            new_args = []
            for a1, a2 in zip(args1, args2, strict=False):
                j = join(a1, a2)
                if isinstance(j, Top):
                    return Top()
                new_args.append(j)
            return TypeCon(c1, tuple(new_args))
        # Different TypeCon constructors
        case (TypeCon(c1, args1), TypeCon(c2, args2)):
            # Check if one is subtype of the other
            try:
                if issubclass(c1, c2) and len(args1) == 0 and len(args2) == 0:
                    return TypeCon(c2, ())
                if issubclass(c2, c1) and len(args1) == 0 and len(args2) == 0:
                    return TypeCon(c1, ())
            except TypeError:
                pass
            return Top()
        # TypeVar cases - can't compute join without more info
        case (TypeVar(_), _) | (_, TypeVar(_)):
            # Conservative: if not equal, return Top
            return Top()

    return Top()


def satisfiable(lower: Type, upper: Type) -> bool:
    """Check if bounds are satisfiable (lower <: upper).

    Args:
        lower: The lower bound.
        upper: The upper bound.

    Returns:
        True if there exists a type T such that lower <: T <: upper.

    """
    return is_subtype(lower, upper)


def satisfies_bounds(t: Type, lower: Type, upper: Type) -> bool:
    """Check if t satisfies the bounds (lower <: t <: upper).

    Args:
        t: The type to check.
        lower: The lower bound.
        upper: The upper bound.

    Returns:
        True if lower <: t <: upper.

    """
    return is_subtype(lower, t) and is_subtype(t, upper)
