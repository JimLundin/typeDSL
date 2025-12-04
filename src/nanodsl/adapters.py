"""
Format adapters for serializing dataclass schemas to various formats.

This module provides the adapter interface and built-in adapters for converting
Node instances and schema dataclasses to/from different serialization formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields as dataclass_fields
from typing import Any

from nanodsl.nodes import Node, Ref
from nanodsl.types import TypeDef
from nanodsl.schema import NodeSchema, FieldSchema


# =============================================================================
# Adapter Interface
# =============================================================================


class FormatAdapter(ABC):
    """Base class for format-specific serialization."""

    @abstractmethod
    def serialize_node(self, node: Node) -> Any:
        """Serialize a node instance."""
        ...

    @abstractmethod
    def deserialize_node(self, data: Any) -> Node:
        """Deserialize to a node instance."""
        ...

    @abstractmethod
    def serialize_typedef(self, typedef: TypeDef) -> Any:
        """Serialize a TypeDef dataclass."""
        ...

    @abstractmethod
    def deserialize_typedef(self, data: Any) -> TypeDef:
        """Deserialize to a TypeDef instance."""
        ...

    @abstractmethod
    def serialize_node_schema(self, schema: NodeSchema) -> Any:
        """Serialize a NodeSchema dataclass."""
        ...


# =============================================================================
# JSON Adapter
# =============================================================================


class JSONAdapter(FormatAdapter):
    """JSON serialization adapter."""

    def serialize_node(self, node: Node) -> dict:
        """Serialize node to dict."""
        result = {"tag": type(node)._tag}

        for field in dataclass_fields(node):
            if not field.name.startswith("_"):
                value = getattr(node, field.name)
                result[field.name] = self._serialize_value(value)

        return result

    def deserialize_node(self, data: dict) -> Node:
        """Deserialize dict to node instance."""
        tag = data["tag"]

        # Handle Ref special case
        if tag == "ref":
            return Ref(id=data["id"])

        # Look up node class by tag
        node_cls = Node._registry.get(tag)
        if node_cls is None:
            raise ValueError(f"Unknown node tag: {tag}")

        # Deserialize fields
        field_values = {}
        for field in dataclass_fields(node_cls):
            if not field.name.startswith("_") and field.name in data:
                field_values[field.name] = self._deserialize_value(data[field.name])

        return node_cls(**field_values)

    def deserialize_typedef(self, data: dict) -> TypeDef:
        """Deserialize dict to TypeDef instance."""
        tag = data["tag"]

        # Look up TypeDef class by tag
        typedef_cls = TypeDef._registry.get(tag)
        if typedef_cls is None:
            raise ValueError(f"Unknown TypeDef tag: {tag}")

        # Deserialize fields
        field_values = {}
        for field in dataclass_fields(typedef_cls):
            if not field.name.startswith("_") and field.name in data:
                field_values[field.name] = self._deserialize_value(data[field.name])

        return typedef_cls(**field_values)

    def serialize_typedef(self, typedef: TypeDef) -> dict:
        """Serialize TypeDef to dict."""
        result = {"tag": type(typedef)._tag}

        for field in dataclass_fields(typedef):
            if not field.name.startswith("_"):
                value = getattr(typedef, field.name)
                result[field.name] = self._serialize_value(value)

        return result

    def serialize_node_schema(self, schema: NodeSchema) -> dict:
        """Serialize NodeSchema to dict."""
        return {
            "tag": schema.tag,
            "type_params": [
                self.serialize_typedef(tp) for tp in schema.type_params
            ],
            "returns": self.serialize_typedef(schema.returns),
            "fields": [
                {
                    "name": field.name,
                    "type": self.serialize_typedef(field.type),
                }
                for field in schema.fields
            ],
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a field value."""
        if isinstance(value, Node):
            return self.serialize_node(value)
        elif isinstance(value, Ref):
            return {"tag": "ref", "id": value.id}
        elif isinstance(value, TypeDef):
            return self.serialize_typedef(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            # Primitives pass through
            return value

    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a field value."""
        if isinstance(value, dict) and "tag" in value:
            # Could be a Node or Ref
            if value["tag"] == "ref":
                return Ref(id=value["id"])
            # Try to deserialize as node
            return self.deserialize_node(value)
        elif isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}
        else:
            # Primitives pass through
            return value
