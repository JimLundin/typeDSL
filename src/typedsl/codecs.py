"""Unified type codec registry and builtins conversion."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, ClassVar

from typedsl.nodes import Node, Ref
from typedsl.types import TypeDef

# Type tag format: {"tag": "<type_name>", "val": <encoded_value>}
# Used for non-format-native types to enable unambiguous decoding.
_TAG_KEY = "tag"
_VAL_KEY = "val"

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
    tuple,
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

        Raises:
            ValueError: If a different type with the same __name__ is already
                registered. This ensures tag-based deserialization is unambiguous.

        Example:
            TypeCodecs.register(
                datetime,
                encode=lambda dt: dt.isoformat(),
                decode=datetime.fromisoformat,
            )

        """
        # Check for name collision with a DIFFERENT type
        type_name = typ.__name__
        for existing_type in cls._registry:
            if existing_type is not typ and existing_type.__name__ == type_name:
                msg = (
                    f"Cannot register {typ!r}: a different type with name "
                    f"'{type_name}' is already registered ({existing_type!r}). "
                    f"Type names must be unique for tag-based deserialization."
                )
                raise ValueError(msg)

        cls._registry[typ] = (encode, decode)

        # Also register for schema extraction if it's an external type
        if typ not in _SCHEMA_BUILTINS:
            cls._ensure_external_registered(typ)

    @classmethod
    def get[T](
        cls,
        typ: type[T],
    ) -> tuple[Callable[[T], Any], Callable[[Any], T]] | None:
        """Get codec for type, or None if not registered."""
        return cls._registry.get(typ)

    @classmethod
    def get_by_name(
        cls,
        type_name: str,
    ) -> tuple[type, Callable[[Any], Any]] | None:
        """Get type and decoder by type name (for tag-based deserialization)."""
        for typ, (_, decode) in cls._registry.items():
            if typ.__name__ == type_name:
                return typ, decode
        return None

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
    def get_external_type(cls, typ: type) -> ExternalTypeRecord:
        """Get external type record for schema extraction.

        Auto-registers the type if not already registered. This allows
        external types to be used in Node fields without explicit registration.
        """
        return cls._ensure_external_registered(typ)

    @classmethod
    def unregister(cls, typ: type) -> bool:
        """Unregister a type's codec.

        Args:
            typ: The type to unregister

        Returns:
            True if the type was registered and removed, False otherwise.

        """
        if typ in cls._registry:
            del cls._registry[typ]
            return True
        return False

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

    TypeCodecs.register(
        tuple,
        encode=list,
        decode=tuple,
    )


# Register builtins on module load
_register_builtins()


def _wrap_tagged(type_name: str, value: Any) -> dict[str, Any]:
    """Wrap a value with its type tag for unambiguous encoding."""
    return {_TAG_KEY: type_name, _VAL_KEY: value}


def to_builtins(obj: Any, *, extended_types: frozenset[type] = frozenset()) -> Any:
    """Convert object tree to JSON-compatible Python builtins.

    Types with codecs that are NOT in extended_types are wrapped with type tags
    for unambiguous decoding: {"tag": "<type_name>", "val": <encoded_value>}

    Args:
        obj: Any Python object to serialize
        extended_types: Types the target format natively supports (don't need tags).
            For JSON: frozenset() (no extended types)
            For YAML: frozenset({datetime, date}) (has native date support)

    Returns:
        JSON-compatible Python value (dict, list, str, int, float, bool, None)

    """
    typ = type(obj)

    # 1. Registered codec (builtins + external types)
    # Non-native types (tuple, set, frozenset, bytes, datetime, etc.) go through here
    if codec := TypeCodecs.get(typ):
        encode, _ = codec
        encoded = to_builtins(encode(obj), extended_types=extended_types)
        # Tag if this type is NOT natively supported by the format
        if typ not in extended_types:
            return _wrap_tagged(typ.__name__, encoded)
        return encoded

    # 2. Node objects
    if isinstance(obj, Node):
        node_cls: type[Node[Any]] = type(obj)
        result: dict[str, Any] = {_TAG_KEY: node_cls.tag}
        for f in fields(obj):
            if not f.name.startswith("_"):
                result[f.name] = to_builtins(
                    getattr(obj, f.name),
                    extended_types=extended_types,
                )
        return result

    # 3. Ref objects
    if isinstance(obj, Ref):
        return {_TAG_KEY: "ref", "id": obj.id}

    # 4. TypeDef objects (for schema serialization)
    if isinstance(obj, TypeDef):
        typedef_cls: type[TypeDef] = type(obj)
        result = {_TAG_KEY: typedef_cls.tag}
        for f in fields(obj):
            if not f.name.startswith("_"):
                result[f.name] = to_builtins(
                    getattr(obj, f.name),
                    extended_types=extended_types,
                )
        return result

    # 5. Sequences (tuples/sets handled by codec above, lists become JSON arrays)
    if isinstance(obj, AbstractSet):
        # Should not reach here if set/frozenset have registered codecs
        return [to_builtins(item, extended_types=extended_types) for item in obj]
    if isinstance(obj, Sequence) and not isinstance(obj, str | bytes):
        return [to_builtins(item, extended_types=extended_types) for item in obj]

    # 6. Mappings
    if isinstance(obj, Mapping):
        return {
            k: to_builtins(v, extended_types=extended_types) for k, v in obj.items()
        }

    # 7. Generic dataclass support (for NodeSchema, FieldSchema, etc.)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_builtins(getattr(obj, f.name), extended_types=extended_types)
            for f in fields(obj)
            if not f.name.startswith("_")
        }

    # 8. Primitives pass through
    return obj


def _is_type_tag(data: Any) -> bool:
    """Check if data is a type tag envelope.

    Type tags have exactly two keys: "tag" and "val".
    This distinguishes them from Nodes (which have "tag" + field names)
    and Refs (which have "tag" + "id").
    """
    if not isinstance(data, dict):
        return False
    keys = set(data.keys())
    return keys == {_TAG_KEY, _VAL_KEY}


def _unwrap_type_tag(data: dict[str, Any]) -> tuple[str, Any]:
    """Unwrap a type tag envelope, returning (type_name, value)."""
    return data[_TAG_KEY], data[_VAL_KEY]


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
    if _TAG_KEY not in data:
        msg = f"Missing required '{_TAG_KEY}' field"
        raise KeyError(msg)
    tag = data[_TAG_KEY]
    if tag == "ref":
        return Ref(id=data["id"])
    if tag in Node.registry:
        return _deserialize_node(data)
    msg = f"Unknown tag '{tag}'"
    raise ValueError(msg)


def _deserialize_node(data: dict[str, Any]) -> Node[Any]:
    """Deserialize a tagged dict to a Node."""
    tag = data[_TAG_KEY]
    node_cls = Node.registry.get(tag)
    if node_cls is None:
        msg = f"Unknown tag '{tag}'"
        raise ValueError(msg)

    field_values = {}
    for field in fields(node_cls):
        if field.name.startswith("_") or field.name not in data:
            continue
        field_values[field.name] = _deserialize_value(data[field.name])

    return node_cls(**field_values)


def _deserialize_value(value: Any) -> Any:
    """Deserialize a value using tags for type reconstruction.

    All non-native types are tagged, so we can deserialize without type hints.
    """
    if value is None:
        return None

    # Type tag envelope: {"tag": "<type>", "val": <value>}
    if _is_type_tag(value):
        tag_name, raw_value = _unwrap_type_tag(value)

        # Tuple: recursively deserialize elements
        if tag_name == "tuple":
            return tuple(_deserialize_value(item) for item in raw_value)

        # Look up registered decoder
        codec_entry = TypeCodecs.get_by_name(tag_name)
        if codec_entry is None:
            msg = f"Unknown type tag: {tag_name}"
            raise ValueError(msg)

        tagged_type, decode = codec_entry

        # For container types, recursively deserialize elements first
        if tagged_type in (set, frozenset):
            elements = [_deserialize_value(item) for item in raw_value]
            return decode(elements)

        # For other types (datetime, bytes, Decimal, etc.), decode directly
        return decode(raw_value)

    # Node/Ref objects: {"tag": "<node_type>", ...fields}
    if isinstance(value, dict) and _TAG_KEY in value:
        tag = value[_TAG_KEY]
        if tag == "ref":
            return Ref(id=value["id"])
        if tag in Node.registry:
            return _deserialize_node(value)
        msg = f"Unknown tag: {tag}"
        raise ValueError(msg)

    # Lists: recursively deserialize elements
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]

    # Dicts without tags: recursively deserialize values
    if isinstance(value, dict):
        return {k: _deserialize_value(v) for k, v in value.items()}

    # Primitives pass through
    return value
