"""Format adapters for serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import TYPE_CHECKING, Any, TypedDict

from typedsl.nodes import Node, Ref
from typedsl.types import (
    DictType,
    FrozenSetType,
    ListType,
    SetType,
    TupleType,
    TypeDef,
    UnionType,
)

if TYPE_CHECKING:
    from typedsl.schema import FieldSchema, NodeSchema


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
        """Serialize a TypeDef to dictionary format."""
        ...

    @abstractmethod
    def deserialize_typedef(self, data: dict[str, Any]) -> TypeDef:
        """Deserialize a dictionary to a TypeDef."""
        ...

    @abstractmethod
    def serialize_node_schema(self, schema: NodeSchema) -> SerializedNodeSchema:
        """Serialize a NodeSchema to dictionary format."""
        ...


class JSONAdapter(FormatAdapter):
    """JSON serialization adapter."""

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

        Uses schema information to properly reconstruct types like tuples and sets
        that don't have native JSON representation.
        """
        from typedsl.schema import node_schema

        tag = data["tag"]
        node_cls = Node.registry.get(tag)
        if node_cls is None:
            msg = f"Unknown node tag: {tag}"
            raise ValueError(msg)

        # Get schema for type-aware deserialization
        schema = node_schema(node_cls)
        field_schemas: dict[str, FieldSchema] = {f.name: f for f in schema.fields}

        field_values = {}
        for field in fields(node_cls):
            if field.name.startswith("_") or field.name not in data:
                continue
            field_schema = field_schemas.get(field.name)
            if field_schema is not None:
                field_values[field.name] = self._deserialize_value_with_type(
                    data[field.name], field_schema.type
                )
            else:
                field_values[field.name] = self._deserialize_value(data[field.name])

        return node_cls(**field_values)

    def deserialize_typedef(self, data: dict[str, Any]) -> TypeDef:
        """Deserialize a JSON-compatible dictionary to a TypeDef."""
        tag = data["tag"]
        typedef_cls = TypeDef.registry.get(tag)
        if typedef_cls is None:
            msg = f"Unknown TypeDef tag: {tag}"
            raise ValueError(msg)

        field_values = {
            field.name: self._deserialize_value(data[field.name])
            for field in fields(typedef_cls)
            if not field.name.startswith("_") and field.name in data
        }
        return typedef_cls(**field_values)

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

    def _deserialize_value(self, value: Any) -> Any:
        if isinstance(value, dict) and "tag" in value:
            tag = value["tag"]
            if tag == "ref":
                return Ref(id=value["id"])
            if tag in Node.registry:
                return self.deserialize_node(value)
            if tag in TypeDef.registry:
                return self.deserialize_typedef(value)
            msg = f"Unknown tag: {tag}"
            raise ValueError(msg)
        if isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}
        return value

    def _deserialize_value_with_type(self, value: Any, typedef: TypeDef) -> Any:
        """Deserialize a value using schema type information.

        This enables proper reconstruction of types like tuples and sets
        that are serialized as JSON arrays but need to be restored to
        their original Python types.
        """
        # Handle None values
        if value is None:
            return None

        # Handle tagged objects (nodes, refs, typedefs)
        if isinstance(value, dict) and "tag" in value:
            tag = value["tag"]
            if tag == "ref":
                return Ref(id=value["id"])
            if tag in Node.registry:
                return self.deserialize_node(value)
            if tag in TypeDef.registry:
                return self.deserialize_typedef(value)
            msg = f"Unknown tag: {tag}"
            raise ValueError(msg)

        # Handle tuple type - convert list to tuple
        if isinstance(typedef, TupleType) and isinstance(value, list):
            elements = typedef.elements
            if len(value) != len(elements):
                # Variable-length tuple or mismatch - just convert to tuple
                return tuple(self._deserialize_value(item) for item in value)
            # Deserialize each element with its specific type
            return tuple(
                self._deserialize_value_with_type(item, elem_type)
                for item, elem_type in zip(value, elements, strict=True)
            )

        # Handle set type - convert list to set
        if isinstance(typedef, SetType) and isinstance(value, list):
            return {
                self._deserialize_value_with_type(item, typedef.element)
                for item in value
            }

        # Handle frozenset type - convert list to frozenset
        if isinstance(typedef, FrozenSetType) and isinstance(value, list):
            return frozenset(
                self._deserialize_value_with_type(item, typedef.element)
                for item in value
            )

        # Handle list type - recursively deserialize elements
        if isinstance(typedef, ListType) and isinstance(value, list):
            return [
                self._deserialize_value_with_type(item, typedef.element)
                for item in value
            ]

        # Handle dict type - recursively deserialize values
        if isinstance(typedef, DictType) and isinstance(value, dict):
            return {
                k: self._deserialize_value_with_type(v, typedef.value)
                for k, v in value.items()
            }

        # Handle union type - try each option
        if isinstance(typedef, UnionType) and isinstance(value, list):
            # For unions, we need to check if any option is a tuple/set/frozenset
            for option in typedef.options:
                if isinstance(option, TupleType | SetType | FrozenSetType):
                    return self._deserialize_value_with_type(value, option)
            # No special handling needed, fall through to default
            return [self._deserialize_value(item) for item in value]

        # Default handling for non-list values or unrecognized types
        if isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}
        return value
