"""Type lattice operations for the constraint-based type checker."""

from __future__ import annotations

from typedsl.typechecker.core import TBot, TCon, TExp, TTop, TVar


def occurs(name: str, t: TExp) -> bool:
    """Check if type variable occurs in type (prevents infinite types)."""
    match t:
        case TVar(n):
            return n == name
        case TCon(_, args):
            return any(occurs(name, a) for a in args)
        case TTop() | TBot():
            return False
    return False


def is_subtype(sub: TExp, sup: TExp) -> bool:
    """Check if sub is a subtype of sup."""
    match (sub, sup):
        case (TBot(), _):
            return True
        case (_, TTop()):
            return True
        case (TTop(), _):
            return False
        case (_, TBot()):
            return False
        case (TVar(n1), TVar(n2)):
            return n1 == n2
        case (TVar(_), TCon(_, _)) | (TCon(_, _), TVar(_)):
            return False
        case (TCon(c1, args1), TCon(c2, args2)):
            try:
                if not issubclass(c1, c2):
                    return False
            except TypeError:
                if c1 != c2:
                    return False

            if c1 == c2:
                if len(args1) != len(args2):
                    return False
                return all(
                    is_subtype(a1, a2) and is_subtype(a2, a1)
                    for a1, a2 in zip(args1, args2, strict=False)
                )

            return len(args1) == 0 and len(args2) == 0

    return False


def meet(a: TExp, b: TExp) -> TExp:
    """Compute greatest lower bound (most specific common subtype)."""
    match (a, b):
        case (TTop(), x) | (x, TTop()):
            return x
        case (TBot(), _) | (_, TBot()):
            return TBot()
        case _ if a == b:
            return a
        case (TCon(c1, args1), TCon(c2, args2)) if c1 == c2:
            if len(args1) != len(args2):
                return TBot()
            new_args: list[TExp] = []
            for a1, a2 in zip(args1, args2, strict=False):
                m = meet(a1, a2)
                if isinstance(m, TBot):
                    return TBot()
                new_args.append(m)
            return TCon(c1, tuple(new_args))
        case (TCon(c1, args1), TCon(c2, args2)):
            try:
                if issubclass(c1, c2) and len(args1) == 0 and len(args2) == 0:
                    return TCon(c1, ())
                if issubclass(c2, c1) and len(args1) == 0 and len(args2) == 0:
                    return TCon(c2, ())
            except TypeError:
                pass
            return TBot()
        case (TVar(_), _) | (_, TVar(_)):
            return TBot()

    return TBot()


def join(a: TExp, b: TExp) -> TExp:
    """Compute least upper bound (most general common supertype)."""
    match (a, b):
        case (TBot(), x) | (x, TBot()):
            return x
        case (TTop(), _) | (_, TTop()):
            return TTop()
        case _ if a == b:
            return a
        case (TCon(c1, args1), TCon(c2, args2)) if c1 == c2:
            if len(args1) != len(args2):
                return TTop()
            new_args: list[TExp] = []
            for a1, a2 in zip(args1, args2, strict=False):
                j = join(a1, a2)
                if isinstance(j, TTop):
                    return TTop()
                new_args.append(j)
            return TCon(c1, tuple(new_args))
        case (TCon(c1, args1), TCon(c2, args2)):
            try:
                if issubclass(c1, c2) and len(args1) == 0 and len(args2) == 0:
                    return TCon(c2, ())
                if issubclass(c2, c1) and len(args1) == 0 and len(args2) == 0:
                    return TCon(c1, ())
            except TypeError:
                pass
            return TTop()
        case (TVar(_), _) | (_, TVar(_)):
            return TTop()

    return TTop()


def satisfiable(lower: TExp, upper: TExp) -> bool:
    """Check if bounds are satisfiable (lower <: upper)."""
    return is_subtype(lower, upper)


def satisfies_bounds(t: TExp, lower: TExp, upper: TExp) -> bool:
    """Check if t satisfies the bounds (lower <: t <: upper)."""
    return is_subtype(lower, t) and is_subtype(t, upper)
