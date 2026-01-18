"""Format adapters for serialization."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import fields
from typing import (
    TYPE_CHECKING,
    Any,
    get_args,
    get_origin,
    get_type_hints,
)

from typedsl.nodes import Node, Ref
from typedsl.types import TypeDef

if TYPE_CHECKING:
    from typedsl.schema import NodeSchema


class JSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Node, Ref, TypeDef, and sets."""

    def default(self, o: Any) -> Any:
        """Handle non-JSON-native types."""
        if isinstance(o, Node):
            result = {
                f.name: getattr(o, f.name)
                for f in fields(o)
                if not f.name.startswith("_")
            }
            result["tag"] = type(o).tag
            return result
        if isinstance(o, Ref):
            return {"tag": "ref", "id": o.id}
        if isinstance(o, TypeDef):
            result = {
                f.name: getattr(o, f.name)
                for f in fields(o)
                if not f.name.startswith("_")
            }
            result["tag"] = type(o).tag
            return result
        if isinstance(o, AbstractSet):
            return list(o)
        return super().default(o)


def deserialize_node(data: dict[str, Any]) -> Node[Any]:
    """Deserialize a JSON-compatible dictionary to a Node.

    Uses type hints to properly reconstruct types like tuples and sets
    that don't have native JSON representation.
    """
    tag = data["tag"]
    node_cls = Node.registry.get(tag)
    if node_cls is None:
        msg = f"Unknown node tag: {tag}"
        raise ValueError(msg)

    hints = get_type_hints(node_cls)

    field_values = {}
    for field in fields(node_cls):
        if field.name.startswith("_") or field.name not in data:
            continue
        python_type = hints[field.name]
        field_values[field.name] = _deserialize_value(data[field.name], python_type)

    return node_cls(**field_values)


def _deserialize_value(value: Any, python_type: Any) -> Any:
    """Deserialize a JSON value, coercing to the expected Python type."""
    if value is None:
        return None

    # Tagged objects (nodes, refs)
    if isinstance(value, dict) and "tag" in value:
        tag = value["tag"]
        if tag == "ref":
            return Ref(id=value["id"])
        if tag in Node.registry:
            return deserialize_node(value)
        msg = f"Unknown tag: {tag}"
        raise ValueError(msg)

    origin = get_origin(python_type)
    args = get_args(python_type)

    # Sequence types - JSON arrays -> tuple, list, set, frozenset
    if isinstance(value, list):
        if origin is tuple:
            # Heterogeneous - each element has its own type
            return tuple(
                _deserialize_value(item, arg)
                for item, arg in zip(value, args, strict=True)
            )
        if origin in (list, set, frozenset):
            # Homogeneous - single element type, use constructor
            element_type = args[0] if args else Any
            return origin(_deserialize_value(item, element_type) for item in value)
        if origin is Sequence:
            # Abstract Sequence -> concrete list
            element_type = args[0] if args else Any
            return [_deserialize_value(item, element_type) for item in value]

    # Mapping types - JSON objects -> dict
    if isinstance(value, dict) and origin in (dict, Mapping):
        value_type = args[1] if len(args) > 1 else Any
        return {k: _deserialize_value(v, value_type) for k, v in value.items()}

    # Union - try each option
    if origin is types.UnionType:
        for option in args:
            result = _deserialize_value(value, option)
            if result is not value:
                return result
        return value

    # Primitives and unknown types - return as-is
    return value


# Legacy adapter interface for backwards compatibility with existing code
class JSONAdapter:
    """JSON serialization adapter - thin wrapper around module functions."""

    def serialize_node(self, node: Node[Any]) -> dict[str, Any]:
        """Serialize a Node to a JSON-compatible dictionary."""
        return json.loads(json.dumps(node, cls=JSONEncoder))

    def deserialize_node(self, data: dict[str, Any]) -> Node[Any]:
        """Deserialize a JSON-compatible dictionary to a Node."""
        return deserialize_node(data)

    def serialize_typedef(self, typedef: TypeDef) -> dict[str, Any]:
        """Serialize a TypeDef to a JSON-compatible dictionary."""
        return json.loads(json.dumps(typedef, cls=JSONEncoder))

    def serialize_node_schema(self, schema: NodeSchema) -> dict[str, Any]:
        """Serialize a NodeSchema to a JSON-compatible dictionary."""
        return {
            "tag": schema.tag,
            "signature": schema.signature,
            "type_params": [self.serialize_typedef(tp) for tp in schema.type_params],
            "returns": self.serialize_typedef(schema.returns),
            "fields": [
                {"name": f.name, "type": self.serialize_typedef(f.type)}
                for f in schema.fields
            ],
        }
