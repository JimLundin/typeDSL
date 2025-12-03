"""
Type system domain for runtime type representation.

This module defines the runtime type representation system used for schema
generation and type extraction. It uses concrete types rather than generic
wrappers to provide clear, self-documenting type definitions.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import dataclass_transform, get_args, get_origin, Any, ClassVar, Callable

# =============================================================================
# Type Registration Records
# =============================================================================


@dataclass(frozen=True)
class ExternalTypeRecord:
    """Record for external type registration."""

    python_type: type
    module: str  # Full module path, e.g., "pandas.core.frame"
    name: str  # Class name, e.g., "DataFrame"
    encode: Callable[[Any], dict]
    decode: Callable[[dict], Any]


# =============================================================================
# Type Definition Base
# =============================================================================


@dataclass_transform(frozen_default=True)
class TypeDef:
    """Base for type definitions."""

    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}
    _external_types: ClassVar[dict[type, ExternalTypeRecord]] = {}

    def __init_subclass__(cls, tag: str | None = None):
        # Always convert to frozen dataclass
        dataclass(frozen=True)(cls)

        # Determine tag
        cls._tag = tag or cls.__name__.lower().removesuffix("type")

        # Check for collisions
        if existing := TypeDef._registry.get(cls._tag):
            if existing is not cls:
                raise ValueError(
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    f"Choose a different tag."
                )

        TypeDef._registry[cls._tag] = cls

    @classmethod
    def register(
        cls,
        python_type: type,
        *,
        encode: Callable[[Any], dict],
        decode: Callable[[dict], Any],
    ) -> type:
        """
        Register an external type with the type system.

        Example:
            TypeDef.register(
                pd.DataFrame,
                encode=lambda df: {"data": df.to_dict()},
                decode=lambda d: pd.DataFrame(d["data"])
            )
            Creates ExternalType(module="pandas.core.frame", name="DataFrame")

        Args:
            python_type: The Python class to register.
            encode: Function to encode instances to dict.
            decode: Function to decode dict to instances.

        Returns:
            The original python_type (unchanged).
        """
        # Get type info
        module = python_type.__module__
        name = python_type.__name__

        # Check if already registered
        if python_type in cls._external_types:
            existing = cls._external_types[python_type]
            # Idempotent if same module/name
            if existing.module == module and existing.name == name:
                return python_type
            raise ValueError(
                f"Type {python_type} already registered as "
                f"{existing.module}.{existing.name}"
            )

        # Create and store record
        record = ExternalTypeRecord(
            python_type=python_type,
            module=module,
            name=name,
            encode=encode,
            decode=decode,
        )
        cls._external_types[python_type] = record
        return python_type

    @classmethod
    def get_registered_type(cls, python_type: type) -> TypeDef | None:
        """
        Get the registered TypeDef for a Python type.

        Args:
            python_type: The Python type to look up

        Returns:
            ExternalType instance for this type, or None if not registered
        """
        if python_type in cls._external_types:
            record = cls._external_types[python_type]
            # ExternalType is defined below in this same file
            return ExternalType(module=record.module, name=record.name)

        return None


# =============================================================================
# Primitive Types (Concrete)
# =============================================================================


class IntType(TypeDef, tag="int"):
    """Integer type."""


class FloatType(TypeDef, tag="float"):
    """Floating point type."""


class StrType(TypeDef, tag="str"):
    """String type."""


class BoolType(TypeDef, tag="bool"):
    """Boolean type."""


class NoneType(TypeDef, tag="none"):
    """None/null type."""


# =============================================================================
# Container Types (Concrete)
# =============================================================================


class ListType(TypeDef, tag="list"):
    """
    List type with element type.

    Example: list[int] → ListType(element=IntType())
    """

    element: TypeDef


class DictType(TypeDef, tag="dict"):
    """
    Dictionary type with key and value types.

    Example: dict[str, int] → DictType(key=StrType(), value=IntType())
    """

    key: TypeDef
    value: TypeDef


class SetType(TypeDef, tag="set"):
    """
    Set type with element type.

    Example: set[int] → SetType(element=IntType())
    """

    element: TypeDef


class TupleType(TypeDef, tag="tuple"):
    """
    Fixed-length heterogeneous tuple type.

    Unlike list (homogeneous), tuple types have:
    - Fixed length (known at schema time)
    - Heterogeneous element types (each position can have different type)

    Examples:
        tuple[int, str, float] → TupleType(elements=(IntType(), StrType(), FloatType()))
        tuple[str, str, str] → TupleType(elements=(StrType(), StrType(), StrType()))
    """

    elements: tuple[TypeDef, ...]


class LiteralType(TypeDef, tag="literal"):
    """
    Literal type representing enumeration of values.

    Maps Python's Literal[...] type to enumeration schema.

    Examples:
        Literal["red", "green", "blue"] → LiteralType(values=("red", "green", "blue"))
        Literal[1, 2, 3] → LiteralType(values=(1, 2, 3))
        Literal[True, False] → LiteralType(values=(True, False))

    Note: Does not support Python enum.Enum at this stage.
    """

    values: tuple[str | int | bool, ...]


# =============================================================================
# Domain Types
# =============================================================================


class NodeType(TypeDef, tag="node"):
    """
    AST Node type with return type.

    Example: Node[float] → NodeType(returns=FloatType())
    """

    returns: TypeDef


class RefType(TypeDef, tag="ref"):
    """
    Reference type pointing to another type.

    Example: Ref[Node[int]] → RefType(target=NodeType(returns=IntType()))
    """

    target: TypeDef


class UnionType(TypeDef, tag="union"):
    """
    Union of multiple types.

    Example: int | str → UnionType(options=(IntType(), StrType()))
    """

    options: tuple[TypeDef, ...]


class TypeParameter(TypeDef, tag="typeparam"):
    """
    Type parameter declaration in PEP 695 generic definitions.

    Represents the DECLARATION of a type parameter (e.g., in class Foo[T]).

    Examples:
        class Foo[T]: ... → TypeParameter(name="T", bound=None)
        class Foo[T: int | float]: ... → TypeParameter(name="T", bound=UnionType(...))

    This is the definition site of the type parameter.
    """

    name: str
    bound: TypeDef | None = None


class TypeParameterRef(TypeDef, tag="typeparamref"):
    """
    Reference to a type parameter within a type expression.

    Represents a USE of a type parameter (e.g., in field: T).

    Examples:
        In class Foo[T]:
            field: T → TypeParameterRef(name="T")
            field: list[T] → ListType(element=TypeParameterRef(name="T"))

    This is the use site that refers back to the TypeParameter declaration.
    """

    name: str


class ExternalType(TypeDef, tag="external"):
    """
    Reference to an externally registered type.

    Used for third-party types like pandas.DataFrame, polars.DataFrame, etc.
    Stores module and name to uniquely identify types across different libraries.

    Examples:
        pd.DataFrame → ExternalType(module="pandas.core.frame", name="DataFrame")
        pl.DataFrame → ExternalType(module="polars.dataframe.frame", name="DataFrame")
    """

    module: str  # Full module path
    name: str  # Class name


# =============================================================================
# Type Parameter Substitution
# =============================================================================


def _substitute_type_params(type_expr: Any, substitutions: dict[Any, Any]) -> Any:
    """
    Recursively substitute type parameters in a type expression.

    Args:
        type_expr: The type expression to substitute in
        substitutions: Mapping from type parameters to their concrete types

    Returns:
        The type expression with parameters substituted
    """
    # If this is a type parameter, substitute it
    if type_expr in substitutions:
        return substitutions[type_expr]

    # Get origin and args for generic types
    origin = get_origin(type_expr)
    args = get_args(type_expr)

    # If no origin, this is a simple type - return as-is
    if origin is None:
        return type_expr

    # If there are no args, return as-is
    if not args:
        return type_expr

    # Recursively substitute in the arguments
    new_args = tuple(_substitute_type_params(arg, substitutions) for arg in args)

    # Handle UnionType (created by | operator) specially
    if isinstance(type_expr, types.UnionType):
        # Reconstruct union using | operator
        result = new_args[0]
        for arg in new_args[1:]:
            result = result | arg
        return result

    # Reconstruct the type with substituted arguments
    return origin[new_args]
