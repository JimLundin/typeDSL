"""Unified type codec registry and builtins conversion."""

from __future__ import annotations

import base64
import types
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import (
    Any,
    ClassVar,
    get_args,
    get_origin,
    get_type_hints,
)

from typedsl.nodes import Node, Ref
from typedsl.types import TypeDef

# Types already in schema.py's _TYPE_MAP (don't need ExternalType registration)
_SCHEMA_BUILTINS: set[type] = {
    int,
    float,
    str,
    bool,
    type(None),
    bytes,
    Decimal,
    date,
    time,
    datetime,
    timedelta,
    list,
    set,
    frozenset,
    dict,
}


@dataclass(frozen=True)
class ExternalTypeRecord:
    """Record for external type registration (for schema extraction).

    Only stores module and name strings - no Python type reference.
    This makes the record safe for export to non-Python systems.
    """

    module: str
    name: str


class TypeCodecs:
    """Unified registry for type serialization codecs.

    Handles both external types (DataFrame, ndarray) and format-varying
    builtins (bytes, datetime, set, etc.) with a single API.

    External types are automatically registered for schema extraction when
    encountered. Codec registration (encode/decode) is only needed for types
    that will be serialized.

    Usage:
        # Register codec for serializable external type
        TypeCodecs.register(
            DataFrame,
            encode=lambda df: df.to_dicts(),
            decode=DataFrame.from_dicts,
        )

        # Types used in Node fields are auto-registered for schema.
        # No explicit registration needed unless serialization is required.
    """

    _registry: ClassVar[
        dict[type, tuple[Callable[[Any], Any], Callable[[Any], Any]]]
    ] = {}
    _external_types: ClassVar[dict[type, ExternalTypeRecord]] = {}

    @classmethod
    def register[T](
        cls,
        typ: type[T],
        encode: Callable[[T], Any],
        decode: Callable[[Any], T],
    ) -> None:
        """Register encode/decode functions for a serializable type.

        Args:
            typ: The type to register (e.g., datetime, DataFrame)
            encode: Function to convert T → JSON-compatible builtins
            decode: Function to convert JSON-compatible builtins → T

        Example:
            TypeCodecs.register(
                datetime,
                encode=lambda dt: dt.isoformat(),
                decode=datetime.fromisoformat,
            )

        """
        cls._registry[typ] = (encode, decode)

        # Also register for schema extraction if it's an external type
        if typ not in _SCHEMA_BUILTINS:
            cls._ensure_external_registered(typ)

    @classmethod
    def _ensure_external_registered(cls, typ: type) -> ExternalTypeRecord:
        """Ensure type is registered for schema, creating record if needed."""
        if typ not in cls._external_types:
            cls._external_types[typ] = ExternalTypeRecord(
                module=typ.__module__,
                name=typ.__name__,
            )
        return cls._external_types[typ]

    @classmethod
    def get[T](
        cls,
        typ: type[T],
    ) -> tuple[Callable[[T], Any], Callable[[Any], T]] | None:
        """Get codec for type, or None if not registered."""
        return cls._registry.get(typ)

    @classmethod
    def get_external_type(cls, typ: type) -> ExternalTypeRecord:
        """Get external type record for schema extraction.

        Auto-registers the type if not already registered. This allows
        external types to be used in Node fields without explicit registration.
        """
        return cls._ensure_external_registered(typ)

    @classmethod
    def clear(cls) -> None:
        """Clear codec registry and re-register builtins."""
        cls._registry.clear()
        _register_builtins()


def _register_builtins() -> None:
    """Pre-register codecs for Python builtin types."""
    TypeCodecs.register(
        bytes,
        encode=lambda b: base64.b64encode(b).decode("ascii"),
        decode=base64.b64decode,
    )

    TypeCodecs.register(
        datetime,
        encode=lambda dt: dt.isoformat(),
        decode=datetime.fromisoformat,
    )

    TypeCodecs.register(
        date,
        encode=lambda d: d.isoformat(),
        decode=date.fromisoformat,
    )

    TypeCodecs.register(
        time,
        encode=lambda t: t.isoformat(),
        decode=time.fromisoformat,
    )

    TypeCodecs.register(
        timedelta,
        encode=lambda td: td.total_seconds(),
        decode=lambda s: timedelta(seconds=s),
    )

    TypeCodecs.register(
        Decimal,
        encode=str,
        decode=Decimal,
    )

    TypeCodecs.register(
        set,
        encode=list,
        decode=set,
    )

    TypeCodecs.register(
        frozenset,
        encode=list,
        decode=frozenset,
    )


# Register builtins on module load
_register_builtins()


def to_builtins(obj: Any) -> Any:
    """Convert object tree to JSON-compatible Python builtins.

    Handles:
    - Registered codecs (external types, datetime, bytes, etc.)
    - Node and Ref objects (adds tag field)
    - TypeDef objects (for schema serialization)
    - Containers (dict, list, tuple, set, frozenset)
    - Primitives (str, int, float, bool, None)

    Args:
        obj: Any Python object to serialize

    Returns:
        JSON-compatible Python value (dict, list, str, int, float, bool, None)

    """
    typ = type(obj)

    # 1. Registered codec (external types + builtins like bytes, datetime, set)
    if codec := TypeCodecs.get(typ):
        encode, _ = codec
        return to_builtins(encode(obj))  # Recurse on encoded result

    # 2. Node objects
    if isinstance(obj, Node):
        node_cls: type[Node[Any]] = type(obj)
        result: dict[str, Any] = {"tag": node_cls.tag}
        for f in fields(obj):
            if not f.name.startswith("_"):
                result[f.name] = to_builtins(getattr(obj, f.name))
        return result

    # 3. Ref objects
    if isinstance(obj, Ref):
        return {"tag": "ref", "id": obj.id}

    # 4. TypeDef objects (for schema serialization)
    if isinstance(obj, TypeDef):
        typedef_cls: type[TypeDef] = type(obj)
        result = {"tag": typedef_cls.tag}
        for f in fields(obj):
            if not f.name.startswith("_"):
                result[f.name] = to_builtins(getattr(obj, f.name))
        return result

    # 5. Sets and sequences (tuple becomes list, set becomes list)
    if isinstance(obj, AbstractSet):
        return [to_builtins(item) for item in obj]
    if isinstance(obj, Sequence) and not isinstance(obj, str | bytes):
        return [to_builtins(item) for item in obj]

    # 6. Mappings
    if isinstance(obj, Mapping):
        return {k: to_builtins(v) for k, v in obj.items()}

    # 7. Generic dataclass support (for NodeSchema, FieldSchema, etc.)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_builtins(getattr(obj, f.name))
            for f in fields(obj)
            if not f.name.startswith("_")
        }

    # 8. Primitives pass through
    return obj


def from_builtins(data: dict[str, Any]) -> Node[Any] | Ref[Any]:
    """Deserialize a tagged dict to a Node or Ref.

    Args:
        data: Dict with 'tag' field

    Returns:
        Deserialized Node or Ref

    Raises:
        KeyError: If 'tag' field is missing
        ValueError: If tag is unknown

    """
    if "tag" not in data:
        msg = "Missing required 'tag' field"
        raise KeyError(msg)
    tag = data["tag"]
    if tag == "ref":
        return Ref(id=data["id"])
    if tag in Node.registry:
        return _deserialize_node(data)
    msg = f"Unknown tag '{tag}'"
    raise ValueError(msg)


def _deserialize_node(data: dict[str, Any]) -> Node[Any]:
    """Deserialize a tagged dict to a Node."""
    tag = data["tag"]
    node_cls = Node.registry.get(tag)
    if node_cls is None:
        msg = f"Unknown tag '{tag}'"
        raise ValueError(msg)

    hints = get_type_hints(node_cls)

    field_values = {}
    for field in fields(node_cls):
        if field.name.startswith("_") or field.name not in data:
            continue
        python_type = hints[field.name]
        field_values[field.name] = _deserialize_value_with_type(
            data[field.name],
            python_type,
        )

    return node_cls(**field_values)


def _deserialize_value_with_type(value: Any, python_type: Any) -> Any:
    """Deserialize a value using its type hint to guide reconstruction."""
    if value is None:
        return None

    # Check for registered codec first (for non-generic types)
    origin = get_origin(python_type)
    if origin is None and (codec := TypeCodecs.get(python_type)):
        _, decode = codec
        return decode(value)

    # Tagged objects (nodes, refs)
    if isinstance(value, dict) and "tag" in value:
        tag = value["tag"]
        if tag == "ref":
            return Ref(id=value["id"])
        if tag in Node.registry:
            return _deserialize_node(value)
        msg = f"Unknown tag: {tag}"
        raise ValueError(msg)

    args = get_args(python_type)

    # Handle parameterized generic types
    if origin is not None:
        # Tuple - heterogeneous, each element has its own type
        if origin is tuple and isinstance(value, list):
            if not args or args == ((),):
                # Empty tuple: tuple[()] or tuple[()]
                return ()
            return tuple(
                _deserialize_value_with_type(item, arg)
                for item, arg in zip(value, args, strict=False)
            )

        # List - homogeneous
        if origin is list and isinstance(value, list):
            element_type = args[0] if args else Any
            return [_deserialize_value_with_type(item, element_type) for item in value]

        # Set - homogeneous (JSON array -> set)
        if origin is set and isinstance(value, list):
            element_type = args[0] if args else Any
            return {_deserialize_value_with_type(item, element_type) for item in value}

        # Frozenset - homogeneous (JSON array -> frozenset)
        if origin is frozenset and isinstance(value, list):
            element_type = args[0] if args else Any
            return frozenset(
                _deserialize_value_with_type(item, element_type) for item in value
            )

        # Sequence (abstract) -> list
        if origin is Sequence and isinstance(value, list):
            element_type = args[0] if args else Any
            return [_deserialize_value_with_type(item, element_type) for item in value]

        # Dict/Mapping
        if origin in (dict, Mapping) and isinstance(value, dict):
            value_type = args[1] if len(args) > 1 else Any
            return {
                k: _deserialize_value_with_type(v, value_type) for k, v in value.items()
            }

        # Union - try each option
        if origin is types.UnionType:
            for option in args:
                if option is type(None) and value is None:
                    return None
                if option is not type(None):
                    try:
                        return _deserialize_value_with_type(value, option)
                    except (TypeError, ValueError):
                        continue
            return value

    # Primitives and unknown types - return as-is
    return value
