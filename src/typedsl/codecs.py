"""Unified type codec registry and builtins conversion."""

from __future__ import annotations

import base64
import types
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import fields, is_dataclass
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


class TypeCodecs:
    """Unified registry for type serialization codecs.

    Handles both external types (DataFrame, ndarray) and format-varying
    builtins (bytes, datetime, set, etc.) with a single API.

    For external types (not in Python's standard library), registration
    also makes the type available for schema extraction via all_schemas().

    Usage:
        # Register external type - works for both serialization AND schema
        TypeCodecs.register(
            DataFrame, encode=lambda df: df.to_dicts(), decode=DataFrame
        )

        # In serialization
        if codec := TypeCodecs.get(type(obj)):
            encode, _ = codec
            return encode(obj)
    """

    _registry: ClassVar[
        dict[type, tuple[Callable[[Any], Any], Callable[[Any], Any]]]
    ] = {}
    _external_types: ClassVar[set[type]] = set()  # Track external types for cleanup

    @classmethod
    def register(
        cls,
        typ: type,
        encode: Callable[[Any], Any],
        decode: Callable[[Any], Any],
    ) -> None:
        """Register encode/decode functions for a type.

        For external types (not standard builtins), this also registers
        the type with the schema system so it can be used in node fields
        and extracted via all_schemas().

        Args:
            typ: The type to register (e.g., datetime, DataFrame)
            encode: Function to convert type → builtins (dict, list, str, etc.)
            decode: Function to convert builtins → type

        Example:
            TypeCodecs.register(
                datetime,
                encode=lambda dt: dt.isoformat(),
                decode=lambda s: datetime.fromisoformat(s),
            )

        """
        cls._registry[typ] = (encode, decode)

        # For external types, also register with TypeDef for schema support
        if typ not in _SCHEMA_BUILTINS:
            # Wrap encode to return dict (TypeDef expects dict output)
            def dict_encode(
                obj: Any,
                enc: Callable[[Any], Any] = encode,
            ) -> dict[str, Any]:
                result = enc(obj)
                if isinstance(result, dict):
                    return result
                return {"_data": result}

            def dict_decode(
                data: dict[str, Any],
                dec: Callable[[Any], Any] = decode,
            ) -> Any:
                return dec(data.get("_data", data))

            TypeDef.register(typ, encode=dict_encode, decode=dict_decode)
            cls._external_types.add(typ)

    @classmethod
    def get(cls, typ: type) -> tuple[Callable[[Any], Any], Callable[[Any], Any]] | None:
        """Get codec for type, or None if not registered."""
        return cls._registry.get(typ)

    @classmethod
    def clear(cls) -> None:
        """Clear codec registry and re-register builtins.

        Note: External types remain registered in TypeDef for schema extraction.
        This is intentional - schema registration is "permanent" for the module's
        lifetime, while codec registration can be reset for testing.
        """
        cls._registry.clear()
        _register_builtins()


def _register_builtins() -> None:
    """Pre-register codecs for Python builtin types."""
    TypeCodecs.register(
        bytes,
        encode=lambda b: base64.b64encode(b).decode("ascii"),
        decode=lambda s: base64.b64decode(s),
    )

    TypeCodecs.register(
        datetime,
        encode=lambda dt: dt.isoformat(),
        decode=lambda s: datetime.fromisoformat(s),
    )

    TypeCodecs.register(
        date,
        encode=lambda d: d.isoformat(),
        decode=lambda s: date.fromisoformat(s),
    )

    TypeCodecs.register(
        time,
        encode=lambda t: t.isoformat(),
        decode=lambda s: time.fromisoformat(s),
    )

    TypeCodecs.register(
        timedelta,
        encode=lambda td: td.total_seconds(),
        decode=lambda s: timedelta(seconds=s),
    )

    TypeCodecs.register(
        Decimal,
        encode=lambda d: str(d),
        decode=lambda s: Decimal(s),
    )

    TypeCodecs.register(
        set,
        encode=lambda s: list(s),
        decode=lambda lst: set(lst),
    )

    TypeCodecs.register(
        frozenset,
        encode=lambda s: list(s),
        decode=lambda lst: frozenset(lst),
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


def from_builtins(data: Any, type_hint: type | None = None) -> Any:
    """Convert JSON-compatible builtins back to typed Python objects.

    Uses type hints to properly reconstruct types like tuples, sets, and
    registered codec types that don't have native JSON representation.

    Args:
        data: JSON-compatible Python value
        type_hint: Optional type hint to guide reconstruction

    Returns:
        Reconstructed Python object

    """
    # Handle None
    if data is None:
        return None

    # Tagged objects (nodes, refs) - type hint not needed
    if isinstance(data, dict) and "tag" in data:
        tag = data["tag"]
        if tag == "ref":
            return Ref(id=data["id"])
        if tag in Node.registry:
            return _deserialize_node(data)
        msg = f"Unknown tag '{tag}' in data"
        raise ValueError(msg)

    # If we have a type hint, use it to guide deserialization
    if type_hint is not None:
        return _deserialize_value_with_type(data, type_hint)

    # Without type hint, recurse into containers
    if isinstance(data, dict):
        return {k: from_builtins(v) for k, v in data.items()}
    if isinstance(data, list):
        return [from_builtins(item) for item in data]

    # Primitives
    return data


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
