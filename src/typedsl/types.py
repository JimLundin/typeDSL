"""Runtime type representation system for schema generation."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    dataclass_transform,
    get_args,
    get_origin,
)


@dataclass(frozen=True)
@dataclass_transform(frozen_default=True)
class TypeDef:
    """Base for type definitions."""

    tag: ClassVar[str]
    registry: ClassVar[dict[str, type[TypeDef]]] = {}

    def __init_subclass__(cls, tag: str | None = None) -> None:
        """Register typedef subclass with automatic tag derivation."""
        dataclass(frozen=True)(cls)
        cls.tag = tag if tag is not None else cls.__name__

        if (existing := TypeDef.registry.get(cls.tag)) and existing is not cls:
            msg = (
                f"Tag '{cls.tag}' already registered to {existing}. "
                "Choose a different tag."
            )
            raise ValueError(msg)

        TypeDef.registry[cls.tag] = cls


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


class BytesType(TypeDef, tag="bytes"):
    """Binary data type."""


class DecimalType(TypeDef, tag="decimal"):
    """Arbitrary precision decimal type."""


# Temporal types - abstract representations that serializers interpret
class DateType(TypeDef, tag="date"):
    """Date type (year, month, day)."""


class TimeType(TypeDef, tag="time"):
    """Time type (hour, minute, second, microsecond)."""


class DateTimeType(TypeDef, tag="datetime"):
    """DateTime type (combined date and time)."""


class DurationType(TypeDef, tag="duration"):
    """Duration/timedelta type."""


class ListType(TypeDef, tag="list"):
    """List type: list[int] → ListType(element=IntType())."""

    element: TypeDef


class DictType(TypeDef, tag="dict"):
    """Dict type: dict[str, int] → DictType(key=StrType(), value=IntType())."""

    key: TypeDef
    value: TypeDef


class SetType(TypeDef, tag="set"):
    """Set type: set[int] → SetType(element=IntType())."""

    element: TypeDef


class FrozenSetType(TypeDef, tag="frozenset"):
    """Immutable set type: frozenset[int] → FrozenSetType(element=IntType())."""

    element: TypeDef


class TupleType(TypeDef, tag="tuple"):
    """Fixed-length heterogeneous tuple: tuple[int, str] → TupleType(elements=(...))."""

    elements: tuple[TypeDef, ...]


# Generic container types - abstract containers that serializers interpret
class SequenceType(TypeDef, tag="sequence"):
    """Generic sequence type: Sequence[int] → SequenceType(element=IntType()).

    Abstract ordered collection - serializers determine concrete representation.
    """

    element: TypeDef


class MappingType(TypeDef, tag="mapping"):
    """Generic mapping type.

    Mapping[str, int] -> MappingType(key=StrType(), value=IntType()).
    Abstract key-value mapping - serializers determine concrete representation.
    """

    key: TypeDef
    value: TypeDef


class LiteralType(TypeDef, tag="literal"):
    """Literal enumeration: Literal["a", "b"] → LiteralType(values=("a", "b"))."""

    values: tuple[str | int | bool, ...]


class ReturnType(TypeDef, tag="return"):
    """Return type constraint: Node[float] → ReturnType(returns=FloatType()).

    Represents "any node that returns T" - a constraint on return type,
    not a reference to a specific node schema.
    """

    returns: TypeDef


class NodeType(TypeDef, tag="node"):
    """Specific node type reference.

    Example: Const[float] → NodeType(node_tag="Const", type_args=(...))

    References a specific node schema by tag. The return type is derived
    from the node's schema at runtime, not stored here.
    """

    node_tag: str
    type_args: tuple[TypeDef, ...] = ()


class RefType(TypeDef, tag="ref"):
    """Reference type: Ref[Node[int]] → RefType(target=NodeType(...))."""

    target: TypeDef


class UnionType(TypeDef, tag="union"):
    """Union type: int | str → UnionType(options=(IntType(), StrType()))."""

    options: tuple[TypeDef, ...]


class TypeParameter(TypeDef, tag="typeparam"):
    """Type parameter declaration (e.g., T in class Foo[T]).

    Attributes:
        name: The type parameter name (e.g., "T")
        bound: Optional upper bound constraint (e.g., int | float)
        default: Optional default type (PEP 696). Can be a concrete type or
            a TypeParameterRef when the default references another type parameter
            (e.g., class Foo[T, R = T] has R's default as TypeParameterRef("T"))

    """

    name: str
    bound: TypeDef | None = None
    default: TypeDef | None = None


class TypeParameterRef(TypeDef, tag="typeparamref"):
    """Reference to a type parameter (e.g., T used in a field annotation)."""

    name: str


class ExternalType(TypeDef, tag="external"):
    """Registered external type, identified by module path and class name."""

    module: str
    name: str


def substitute_type_params(type_expr: Any, substitutions: dict[Any, Any]) -> Any:
    """Recursively substitute type parameters in a type expression."""
    if type_expr in substitutions:
        return substitutions[type_expr]

    origin = get_origin(type_expr)
    args = get_args(type_expr)

    if origin is None or not args:
        return type_expr

    new_args = tuple(substitute_type_params(arg, substitutions) for arg in args)

    # UnionType (| operator) needs special reconstruction
    if isinstance(type_expr, types.UnionType):
        result = new_args[0]
        for arg in new_args[1:]:
            result = result | arg
        return result

    return origin[new_args]
