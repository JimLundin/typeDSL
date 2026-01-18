"""Format adapters for serialization."""

from __future__ import annotations

import types
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typedsl.nodes import Node, Ref
from typedsl.types import TypeDef

if TYPE_CHECKING:
    from typedsl.schema import NodeSchema


class SerializedFieldSchema(TypedDict):
    """Serialized field schema structure."""

    name: str
    type: dict[str, Any]


class SerializedNodeSchema(TypedDict):
    """Serialized node schema structure."""

    tag: str
    signature: dict[str, Any]  # Signature kwargs
    type_params: list[dict[str, Any]]
    returns: dict[str, Any]
    fields: list[SerializedFieldSchema]


class FormatAdapter(ABC):
    """Base class for format-specific serialization."""

    @abstractmethod
    def serialize_node(self, node: Node[Any]) -> dict[str, Any]:
        """Serialize a Node to dictionary format."""
        ...

    @abstractmethod
    def deserialize_node(self, data: dict[str, Any]) -> Node[Any]:
        """Deserialize a dictionary to a Node."""
        ...

    @abstractmethod
    def serialize_typedef(self, typedef: TypeDef) -> dict[str, Any]:
        """Serialize a TypeDef to dictionary format (for schema export)."""
        ...

    @abstractmethod
    def serialize_node_schema(self, schema: NodeSchema) -> SerializedNodeSchema:
        """Serialize a NodeSchema to dictionary format."""
        ...


class JSONAdapter(FormatAdapter):
    """JSON serialization adapter.

    Uses Python type hints to properly serialize and deserialize
    types that don't have native JSON representation (tuples, sets, etc.).
    """

    def serialize_node(self, node: Node[Any]) -> dict[str, Any]:
        """Serialize a Node to a JSON-compatible dictionary."""
        result = {
            field.name: self._serialize_value(getattr(node, field.name))
            for field in fields(node)
            if not field.name.startswith("_")
        }
        result["tag"] = type(node).tag
        return result

    def deserialize_node(self, data: dict[str, Any]) -> Node[Any]:
        """Deserialize a JSON-compatible dictionary to a Node.

        Uses type hints to properly reconstruct types like tuples and sets
        that don't have native JSON representation.
        """
        tag = data["tag"]
        node_cls = Node.registry.get(tag)
        if node_cls is None:
            msg = f"Unknown node tag: {tag}"
            raise ValueError(msg)

        # Get type hints directly from the node class
        hints = get_type_hints(node_cls)

        field_values = {}
        for field in fields(node_cls):
            if field.name.startswith("_") or field.name not in data:
                continue
            python_type = hints[field.name]
            field_values[field.name] = self._deserialize_value(
                data[field.name],
                python_type,
            )

        return node_cls(**field_values)

    def serialize_typedef(self, typedef: TypeDef) -> dict[str, Any]:
        """Serialize a TypeDef to a JSON-compatible dictionary."""
        result = {
            field.name: self._serialize_value(getattr(typedef, field.name))
            for field in fields(typedef)
            if not field.name.startswith("_")
        }
        result["tag"] = type(typedef).tag
        return result

    def serialize_node_schema(self, schema: NodeSchema) -> SerializedNodeSchema:
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

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a Python value to JSON-compatible format."""
        if isinstance(value, Node):
            return self.serialize_node(value)
        if isinstance(value, Ref):
            return {"tag": "ref", "id": value.id}
        if isinstance(value, TypeDef):
            return self.serialize_typedef(value)
        if isinstance(value, list | tuple | set | frozenset):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return value

    def _deserialize_value(self, value: Any, python_type: Any) -> Any:
        """Deserialize a JSON value using Python type information."""
        if value is None:
            return None

        # Tagged objects (nodes, refs)
        if isinstance(value, dict) and "tag" in value:
            tag = value["tag"]
            if tag == "ref":
                return Ref(id=value["id"])
            if tag in Node.registry:
                return self.deserialize_node(value)
            msg = f"Unknown tag: {tag}"
            raise ValueError(msg)

        origin = get_origin(python_type)
        args = get_args(python_type)

        # Sequence types - all serialize to JSON arrays
        if isinstance(value, list):
            if origin is tuple:
                # Heterogeneous - each element has its own type
                return tuple(
                    self._deserialize_value(item, arg)
                    for item, arg in zip(value, args, strict=True)
                )
            if origin in (list, set, frozenset) or origin is Sequence:
                # Homogeneous - single element type
                element_type = args[0] if args else Any
                elements = (
                    self._deserialize_value(item, element_type) for item in value
                )
                # Sequence abstract type -> list, others use their constructor
                return list(elements) if origin is Sequence else origin(elements)

        # Dict/Mapping types
        if isinstance(value, dict) and origin in (dict, Mapping):
            value_type = args[1] if len(args) > 1 else Any
            return {k: self._deserialize_value(v, value_type) for k, v in value.items()}

        # Union - try each option
        if origin is Union or isinstance(python_type, types.UnionType):
            for option in args:
                result = self._deserialize_value(value, option)
                if result is not value:
                    return result
            return value

        # Primitives and unknown types - return as-is
        return value
