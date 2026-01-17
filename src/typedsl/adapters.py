"""Format adapters for serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import TYPE_CHECKING, Any, TypedDict

from typedsl.nodes import Node, Ref
from typedsl.schema import node_schema
from typedsl.types import (
    DictType,
    FrozenSetType,
    ListType,
    MappingType,
    RefType,
    ReturnType,
    SequenceType,
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
    """JSON serialization adapter.

    Uses schema type information to properly serialize and deserialize
    Python types that don't have native JSON representation (tuples, sets, etc.).
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

        Uses schema information to properly reconstruct types like tuples and sets
        that don't have native JSON representation.
        """
        tag = data["tag"]
        node_cls = Node.registry.get(tag)
        if node_cls is None:
            msg = f"Unknown node tag: {tag}"
            raise ValueError(msg)

        schema = node_schema(node_cls)
        field_schemas: dict[str, FieldSchema] = {f.name: f for f in schema.fields}

        field_values = {}
        for field in fields(node_cls):
            if field.name.startswith("_") or field.name not in data:
                continue
            field_schema = field_schemas[field.name]
            field_values[field.name] = self._deserialize_value(
                data[field.name],
                field_schema.type,
            )

        return node_cls(**field_values)

    def deserialize_typedef(self, data: dict[str, Any]) -> TypeDef:
        """Deserialize a JSON-compatible dictionary to a TypeDef."""
        tag = data["tag"]
        typedef_cls = TypeDef.registry.get(tag)
        if typedef_cls is None:
            msg = f"Unknown TypeDef tag: {tag}"
            raise ValueError(msg)

        field_values = {}
        for field in fields(typedef_cls):
            if field.name.startswith("_") or field.name not in data:
                continue
            raw_value = data[field.name]
            # TypeDef fields are either primitives, tuples of TypeDefs, or TypeDefs
            if isinstance(raw_value, dict) and "tag" in raw_value:
                field_values[field.name] = self.deserialize_typedef(raw_value)
            elif isinstance(raw_value, list):
                field_values[field.name] = tuple(
                    self.deserialize_typedef(item) if isinstance(item, dict) else item
                    for item in raw_value
                )
            else:
                field_values[field.name] = raw_value

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

    def _deserialize_value(self, value: Any, typedef: TypeDef) -> Any:
        """Deserialize a JSON value using schema type information.

        Uses the TypeDef to determine how to reconstruct Python types
        that don't have native JSON representation (tuples, sets, etc.).
        """
        if value is None:
            return None

        # Tagged objects (nodes, refs)
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

        # Tuple - convert list to tuple with per-element types
        if isinstance(typedef, TupleType) and isinstance(value, list):
            elements = typedef.elements
            return tuple(
                self._deserialize_value(item, elem_type)
                for item, elem_type in zip(value, elements, strict=True)
            )

        # Set - convert list to set
        if isinstance(typedef, SetType) and isinstance(value, list):
            return {self._deserialize_value(item, typedef.element) for item in value}

        # FrozenSet - convert list to frozenset
        if isinstance(typedef, FrozenSetType) and isinstance(value, list):
            return frozenset(
                self._deserialize_value(item, typedef.element) for item in value
            )

        # List - recursively deserialize elements
        if isinstance(typedef, ListType) and isinstance(value, list):
            return [self._deserialize_value(item, typedef.element) for item in value]

        # Sequence (abstract) - deserialize as list
        if isinstance(typedef, SequenceType) and isinstance(value, list):
            return [self._deserialize_value(item, typedef.element) for item in value]

        # Dict - recursively deserialize values
        if isinstance(typedef, DictType) and isinstance(value, dict):
            return {
                k: self._deserialize_value(v, typedef.value) for k, v in value.items()
            }

        # Mapping (abstract) - deserialize as dict
        if isinstance(typedef, MappingType) and isinstance(value, dict):
            return {
                k: self._deserialize_value(v, typedef.value) for k, v in value.items()
            }

        # Union - find matching type for the value
        if isinstance(typedef, UnionType):
            return self._deserialize_union_value(value, typedef)

        # RefType - the value should be a tagged ref dict
        if isinstance(typedef, RefType):
            if isinstance(value, dict) and value.get("tag") == "ref":
                return Ref(id=value["id"])
            return value

        # ReturnType - unwrap and deserialize the inner type
        if isinstance(typedef, ReturnType):
            return self._deserialize_value(value, typedef.returns)

        # Primitives and unknown types - return as-is
        return value

    def _deserialize_union_value(self, value: Any, typedef: UnionType) -> Any:
        """Deserialize a value that could be one of several union types."""
        # For lists, check if any option expects a tuple/set/frozenset/list
        if isinstance(value, list):
            for option in typedef.options:
                if isinstance(option, TupleType):
                    if len(value) == len(option.elements):
                        return self._deserialize_value(value, option)
                elif isinstance(
                    option,
                    SetType | FrozenSetType | ListType | SequenceType,
                ):
                    return self._deserialize_value(value, option)
            # Default to list if no specific match
            return value

        # For dicts, check if any option expects a dict type
        if isinstance(value, dict) and "tag" not in value:
            for option in typedef.options:
                if isinstance(option, DictType | MappingType):
                    return self._deserialize_value(value, option)
            return value

        # For tagged values, let the normal flow handle it
        if isinstance(value, dict) and "tag" in value:
            tag = value["tag"]
            if tag == "ref":
                return Ref(id=value["id"])
            if tag in Node.registry:
                return self.deserialize_node(value)

        # Primitives - return as-is
        return value
