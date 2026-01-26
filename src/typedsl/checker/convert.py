"""Convert Python types to constraint type expressions.

This module provides the bridge between Python's type system and
the constraint type language. It uses get_origin/get_args to
generically handle all parameterized types without hardcoding.
"""

from __future__ import annotations

import types
from typing import TypeVar, Union, get_args, get_origin

from typedsl.checker.types import TCon, TVar, TypeExpr


def to_type_expr(
    py_type: object,
    var_map: dict[str, TVar],
) -> TypeExpr:
    """Convert a Python type to a constraint TypeExpr.

    This function handles:
    - Simple types (int, str, bool, etc.)
    - Generic types (list[int], dict[str, T], etc.)
    - TypeVars (converted to TVar with fresh IDs)
    - Union types (int | str)
    - None type

    Args:
        py_type: A Python type annotation
        var_map: Mapping from TypeVar NAME to TVar.
                 This uses names instead of TypeVar identity because
                 get_type_hints() may return different TypeVar objects
                 than __type_params__.

    Returns:
        A TypeExpr representing the type in the constraint language.

    Examples:
        >>> var_map = {}
        >>> to_type_expr(int, var_map)
        int
        >>> to_type_expr(list[int], var_map)
        list[int]
        >>> T = TypeVar('T')
        >>> to_type_expr(list[T], var_map)
        list[?T_1]

    """
    # TypeVar -> TVar (create fresh if not seen, lookup by name)
    if isinstance(py_type, TypeVar):
        name = py_type.__name__
        if name not in var_map:
            var_map[name] = TVar.fresh(name)
        return var_map[name]

    # NoneType special case
    if py_type is type(None):
        return TCon(type(None), ())

    # Get origin for generic types
    origin = get_origin(py_type)

    # Simple type (int, str, etc.) - no origin
    if origin is None:
        # Must be a concrete type with __name__
        if isinstance(py_type, type):
            return TCon(py_type, ())
        # Could be a special form or forward reference
        # For now, try to handle it as-is
        msg = f"Cannot convert type: {py_type!r}"
        raise TypeError(msg)

    # Get type arguments
    args = get_args(py_type)
    converted_args = tuple(to_type_expr(arg, var_map) for arg in args)

    # Handle union types (T | U or Union[T, U])
    if origin is Union or isinstance(py_type, types.UnionType):
        # Use Union as the constructor for union types
        return TCon(Union, converted_args)

    # Generic type - use origin as constructor
    return TCon(origin, converted_args)


def format_type_expr(expr: TypeExpr) -> str:
    """Format a TypeExpr for human-readable display.

    Args:
        expr: The type expression to format

    Returns:
        A string representation suitable for error messages

    """
    if isinstance(expr, TVar):
        return f"?{expr.name}"

    if isinstance(expr, TCon):
        # Special case for NoneType
        if expr.con is type(None):
            return "None"

        # Special case for Union
        if expr.con is Union:
            return " | ".join(format_type_expr(arg) for arg in expr.args)

        # Regular type
        if not expr.args:
            return expr.con.__name__

        args_str = ", ".join(format_type_expr(arg) for arg in expr.args)
        return f"{expr.con.__name__}[{args_str}]"

    return str(expr)
