"""Tests for typedsl serialization module."""

import json

import pytest

from typedsl.adapters import JSONEncoder
from typedsl.nodes import Node, Ref
from typedsl.schema import node_schema
from typedsl.serialization import from_dict, from_json, to_dict, to_json
from typedsl.types import (
    DictType,
    FrozenSetType,
    IntType,
    ListType,
    NodeType,
    RefType,
    ReturnType,
    SetType,
    StrType,
    TupleType,
    UnionType,
)


class TestSerializeNode:
    """Test to_dict() for Node serialization."""

    def test_serialize_simple_node(self) -> None:
        """Test serializing a simple node."""

        class Value(Node[int], tag="value_adapter"):
            num: int

        node = Value(num=42)
        result = to_dict(node)

        assert result == {"tag": "value_adapter", "num": 42}

    def test_serialize_node_with_multiple_fields(self) -> None:
        """Test serializing node with multiple fields."""

        class Record(Node[str], tag="record_adapter"):
            name: str
            age: int
            active: bool

        node = Record(name="Bob", age=25, active=True)
        result = to_dict(node)

        assert result == {
            "tag": "record_adapter",
            "name": "Bob",
            "age": 25,
            "active": True,
        }

    def test_serialize_nested_nodes(self) -> None:
        """Test serializing nested nodes."""

        class Leaf(Node[int], tag="leaf_adapter"):
            value: int

        class Branch(Node[int], tag="branch_adapter"):
            left: Node[int]
            right: Node[int]

        tree = Branch(left=Leaf(value=1), right=Leaf(value=2))
        result = to_dict(tree)

        assert result == {
            "tag": "branch_adapter",
            "left": {"tag": "leaf_adapter", "value": 1},
            "right": {"tag": "leaf_adapter", "value": 2},
        }

    def test_serialize_node_with_ref(self) -> None:
        """Test serializing node containing a Ref."""

        class RefNode(Node[int], tag="ref_node_adapter"):
            target: Ref[Node[int]]

        node = RefNode(target=Ref(id="node-123"))
        result = to_dict(node)

        assert result == {
            "tag": "ref_node_adapter",
            "target": {"tag": "ref", "id": "node-123"},
        }

    def test_serialize_node_with_list(self) -> None:
        """Test serializing node with list field."""

        class ListNode(Node[list[int]], tag="list_node_adapter"):
            items: list[int]

        node = ListNode(items=[1, 2, 3])
        result = to_dict(node)

        assert result == {"tag": "list_node_adapter", "items": [1, 2, 3]}

    def test_serialize_node_with_dict(self) -> None:
        """Test serializing node with dict field."""

        class DictNode(Node[dict[str, int]], tag="dict_node_adapter"):
            data: dict[str, int]

        node = DictNode(data={"a": 1, "b": 2})
        result = to_dict(node)

        assert result == {"tag": "dict_node_adapter", "data": {"a": 1, "b": 2}}

    def test_serialize_node_with_optional(self) -> None:
        """Test serializing node with optional field (None value)."""

        class OptionalNode(Node[int | None], tag="optional_node_adapter"):
            value: int | None

        node = OptionalNode(value=None)
        result = to_dict(node)

        assert result == {"tag": "optional_node_adapter", "value": None}

    def test_serialize_deeply_nested(self) -> None:
        """Test serializing deeply nested structure."""

        class DeepLeaf(Node[int], tag="deep_leaf_adapter"):
            value: int

        class DeepWrapper(Node[int], tag="deep_wrapper_adapter"):
            child: Node[int]

        deep = DeepWrapper(
            child=DeepWrapper(child=DeepWrapper(child=DeepLeaf(value=42))),
        )
        result = to_dict(deep)

        assert result == {
            "tag": "deep_wrapper_adapter",
            "child": {
                "tag": "deep_wrapper_adapter",
                "child": {
                    "tag": "deep_wrapper_adapter",
                    "child": {"tag": "deep_leaf_adapter", "value": 42},
                },
            },
        }


class TestDeserializeNode:
    """Test from_dict() for Node deserialization."""

    def test_deserialize_simple_node(self) -> None:
        """Test deserializing a simple node."""

        class SimpleValue(Node[int], tag="simple_value_deser"):
            num: int

        data = {"tag": "simple_value_deser", "num": 42}
        result = from_dict(data)

        assert isinstance(result, SimpleValue)
        assert result.num == 42

    def test_deserialize_node_with_multiple_fields(self) -> None:
        """Test deserializing node with multiple fields."""

        class Person(Node[str], tag="person_deser"):
            name: str
            age: int
            active: bool

        data = {"tag": "person_deser", "name": "Alice", "age": 30, "active": True}
        result = from_dict(data)

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30
        assert result.active is True

    def test_deserialize_nested_nodes(self) -> None:
        """Test deserializing nested nodes."""

        class LeafDeser(Node[int], tag="leaf_deser"):
            value: int

        class BranchDeser(Node[int], tag="branch_deser"):
            left: Node[int]
            right: Node[int]

        data = {
            "tag": "branch_deser",
            "left": {"tag": "leaf_deser", "value": 10},
            "right": {"tag": "leaf_deser", "value": 20},
        }
        result = from_dict(data)

        assert isinstance(result, BranchDeser)
        assert isinstance(result.left, LeafDeser)
        assert isinstance(result.right, LeafDeser)
        assert result.left.value == 10
        assert result.right.value == 20

    def test_deserialize_node_with_ref(self) -> None:
        """Test deserializing node containing a Ref."""

        class RefNodeDeser(Node[int], tag="ref_node_deser"):
            target: Ref[Node[int]]

        data = {"tag": "ref_node_deser", "target": {"tag": "ref", "id": "ref-456"}}
        result = from_dict(data)

        assert isinstance(result, RefNodeDeser)
        assert isinstance(result.target, Ref)
        assert result.target.id == "ref-456"

    def test_deserialize_node_with_list(self) -> None:
        """Test deserializing node with list field."""

        class ListNodeDeser(Node[list[int]], tag="list_node_deser"):
            items: list[int]

        data = {"tag": "list_node_deser", "items": [10, 20, 30]}
        result = from_dict(data)

        assert isinstance(result, ListNodeDeser)
        assert result.items == [10, 20, 30]

    def test_deserialize_unknown_tag_raises(self) -> None:
        """Test that deserializing unknown tag raises ValueError."""
        data = {"tag": "unknown_tag_xyz"}

        with pytest.raises(ValueError, match="Unknown tag 'unknown_tag_xyz'"):
            from_dict(data)

    def test_deserialize_missing_tag_raises(self) -> None:
        """Test that deserializing without tag raises KeyError."""
        data = {"value": 42}

        with pytest.raises(KeyError, match="Missing required 'tag' field"):
            from_dict(data)


class TestSerializeTypeDef:
    """Test to_dict() for TypeDef serialization."""

    def test_serialize_int_type(self) -> None:
        """Test serializing IntType."""
        result = to_dict(IntType())
        assert result == {"tag": "int"}

    def test_serialize_str_type(self) -> None:
        """Test serializing StrType."""
        result = to_dict(StrType())
        assert result == {"tag": "str"}

    def test_serialize_list_type(self) -> None:
        """Test serializing ListType."""
        list_type = ListType(element=IntType())
        result = to_dict(list_type)

        assert result == {"tag": "list", "element": {"tag": "int"}}

    def test_serialize_dict_type(self) -> None:
        """Test serializing DictType."""
        dict_type = DictType(key=StrType(), value=IntType())
        result = to_dict(dict_type)

        assert result == {
            "tag": "dict",
            "key": {"tag": "str"},
            "value": {"tag": "int"},
        }

    def test_serialize_union_type(self) -> None:
        """Test serializing UnionType."""
        union_type = UnionType(options=(IntType(), StrType()))
        result = to_dict(union_type)

        assert result == {
            "tag": "union",
            "options": [{"tag": "int"}, {"tag": "str"}],
        }

    def test_serialize_node_type(self) -> None:
        """Test serializing NodeType."""
        node_type = NodeType(node_tag="some_tag", type_args=(IntType(),))
        result = to_dict(node_type)

        assert result == {
            "tag": "node",
            "node_tag": "some_tag",
            "type_args": [{"tag": "int"}],
        }

    def test_serialize_ref_type(self) -> None:
        """Test serializing RefType."""
        ref_type = RefType(target=ReturnType(returns=IntType()))
        result = to_dict(ref_type)

        assert result == {
            "tag": "ref",
            "target": {"tag": "return", "returns": {"tag": "int"}},
        }


class TestSerializeNodeSchema:
    """Test schema serialization via JSONEncoder."""

    def test_serialize_simple_node_schema(self) -> None:
        """Test serializing schema for simple node."""

        class SimpleNode(Node[int], tag="simple_schema"):
            value: int

        schema = node_schema(SimpleNode)
        result = json.loads(json.dumps(schema, cls=JSONEncoder))

        assert result["tag"] == "simple_schema"
        assert result["returns"] == {"tag": "int"}
        assert len(result["fields"]) == 1
        assert result["fields"][0]["name"] == "value"
        assert result["fields"][0]["type"] == {"tag": "int"}

    def test_serialize_schema_with_multiple_fields(self) -> None:
        """Test serializing schema with multiple fields."""

        class MultiFieldNode(Node[str], tag="multi_field_schema"):
            name: str
            count: int
            active: bool

        schema = node_schema(MultiFieldNode)
        result = json.loads(json.dumps(schema, cls=JSONEncoder))

        assert result["tag"] == "multi_field_schema"
        field_names = [f["name"] for f in result["fields"]]
        assert field_names == ["name", "count", "active"]


class TestRoundTrip:
    """Test serialization round-trip (serialize then deserialize)."""

    def test_simple_node_round_trip(self) -> None:
        """Test simple node round-trip."""

        class SimpleRT(Node[int], tag="simple_rt"):
            value: int

        original = SimpleRT(value=99)
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_nested_node_round_trip(self) -> None:
        """Test nested node round-trip."""

        class Inner(Node[int], tag="inner_adapter_rt"):
            value: int

        class Outer(Node[int], tag="outer_adapter_rt"):
            child: Node[int]

        original = Outer(child=Inner(value=100))

        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_complex_structure_round_trip(self) -> None:
        """Test complex nested structure round-trip."""

        class Const(Node[float], tag="const_adapter_rt"):
            value: float

        class BinOp(Node[float], tag="binop_adapter_rt"):
            op: str
            left: Node[float]
            right: Node[float]

        original = BinOp(
            op="+",
            left=Const(value=1.5),
            right=BinOp(op="*", left=Const(value=2.0), right=Const(value=3.0)),
        )

        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_json_round_trip(self) -> None:
        """Test JSON string round-trip."""

        class JsonRT(Node[int], tag="json_rt"):
            value: int

        original = JsonRT(value=42)
        json_str = to_json(original)
        deserialized = from_json(json_str)

        assert deserialized == original


class TestEdgeCases:
    """Test edge cases for serialization."""

    def test_serialize_ref_directly(self) -> None:
        """Test serializing a Ref directly."""
        ref = Ref[int](id="direct-ref")
        result = to_dict(ref)

        assert result == {"tag": "ref", "id": "direct-ref"}

    def test_deserialize_ref_directly(self) -> None:
        """Test deserializing a Ref directly."""
        data = {"tag": "ref", "id": "direct-ref-deser"}
        result = from_dict(data)

        assert isinstance(result, Ref)
        assert result.id == "direct-ref-deser"

    def test_serialize_empty_list(self) -> None:
        """Test serializing node with empty list."""

        class EmptyListNode(Node[list[int]], tag="empty_list_adapter"):
            items: list[int]

        node = EmptyListNode(items=[])
        result = to_dict(node)

        assert result == {"tag": "empty_list_adapter", "items": []}

    def test_serialize_empty_dict(self) -> None:
        """Test serializing node with empty dict."""

        class EmptyDictNode(Node[dict[str, int]], tag="empty_dict_adapter"):
            data: dict[str, int]

        node = EmptyDictNode(data={})
        result = to_dict(node)

        assert result == {"tag": "empty_dict_adapter", "data": {}}


class TestTupleSupport:
    """Test tuple serialization and deserialization."""

    def test_serialize_tuple_field(self) -> None:
        """Test serializing node with tuple field."""

        class TupleNode(Node[None], tag="tuple_node_adapter"):
            coords: tuple[int, int]

        node = TupleNode(coords=(10, 20))
        result = to_dict(node)

        assert result == {"tag": "tuple_node_adapter", "coords": [10, 20]}

    def test_deserialize_tuple_field(self) -> None:
        """Test deserializing node with tuple field."""

        class TupleNodeDeser(Node[None], tag="tuple_node_deser"):
            coords: tuple[int, int]

        data = {"tag": "tuple_node_deser", "coords": [10, 20]}
        result = from_dict(data)

        assert isinstance(result, TupleNodeDeser)
        assert result.coords == (10, 20)
        assert isinstance(result.coords, tuple)

    def test_tuple_round_trip(self) -> None:
        """Test tuple field round-trip."""

        class TupleRT(Node[None], tag="tuple_rt"):
            point: tuple[float, float]

        original = TupleRT(point=(1.5, 2.5))
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert isinstance(deserialized.point, tuple)

    def test_nested_tuple_round_trip(self) -> None:
        """Test nested tuple field round-trip."""

        class NestedTupleRT(Node[None], tag="nested_tuple_rt"):
            nested: tuple[tuple[int, int], str]

        original = NestedTupleRT(nested=((1, 2), "hello"))
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert isinstance(deserialized.nested, tuple)
        assert isinstance(deserialized.nested[0], tuple)


class TestSetSupport:
    """Test set serialization and deserialization."""

    def test_serialize_set_field(self) -> None:
        """Test serializing node with set field."""

        class SetNode(Node[None], tag="set_node_adapter"):
            tags: set[str]

        node = SetNode(tags={"a", "b", "c"})
        result = to_dict(node)

        assert result["tag"] == "set_node_adapter"
        assert set(result["tags"]) == {"a", "b", "c"}

    def test_deserialize_set_field(self) -> None:
        """Test deserializing node with set field."""

        class SetNodeDeser(Node[None], tag="set_node_deser"):
            tags: set[str]

        data = {"tag": "set_node_deser", "tags": ["x", "y", "z"]}
        result = from_dict(data)

        assert isinstance(result, SetNodeDeser)
        assert result.tags == {"x", "y", "z"}
        assert isinstance(result.tags, set)

    def test_set_round_trip(self) -> None:
        """Test set field round-trip."""

        class SetRT(Node[None], tag="set_rt"):
            items: set[int]

        original = SetRT(items={1, 2, 3})
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert isinstance(deserialized.items, set)

    def test_empty_set_round_trip(self) -> None:
        """Test empty set field round-trip."""

        class EmptySetRT(Node[None], tag="empty_set_rt"):
            items: set[str]

        original = EmptySetRT(items=set())
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert deserialized.items == set()


class TestFrozenSetSupport:
    """Test frozenset serialization and deserialization."""

    def test_serialize_frozenset_field(self) -> None:
        """Test serializing node with frozenset field."""

        class FrozenSetNode(Node[None], tag="frozenset_node_adapter"):
            ids: frozenset[int]

        node = FrozenSetNode(ids=frozenset({1, 2, 3}))
        result = to_dict(node)

        assert result["tag"] == "frozenset_node_adapter"
        assert set(result["ids"]) == {1, 2, 3}

    def test_deserialize_frozenset_field(self) -> None:
        """Test deserializing node with frozenset field."""

        class FrozenSetNodeDeser(Node[None], tag="frozenset_node_deser"):
            ids: frozenset[int]

        data = {"tag": "frozenset_node_deser", "ids": [10, 20, 30]}
        result = from_dict(data)

        assert isinstance(result, FrozenSetNodeDeser)
        assert result.ids == frozenset({10, 20, 30})
        assert isinstance(result.ids, frozenset)

    def test_frozenset_round_trip(self) -> None:
        """Test frozenset field round-trip."""

        class FrozenSetRT(Node[None], tag="frozenset_rt"):
            items: frozenset[str]

        original = FrozenSetRT(items=frozenset({"a", "b"}))
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert isinstance(deserialized.items, frozenset)


class TestComplexTypeRoundTrip:
    """Test complex type round-trip scenarios."""

    def test_list_of_tuples_round_trip(self) -> None:
        """Test list of tuples round-trip."""

        class ListTupleRT(Node[None], tag="list_tuple_rt"):
            points: list[tuple[int, int]]

        original = ListTupleRT(points=[(1, 2), (3, 4), (5, 6)])
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert all(isinstance(p, tuple) for p in deserialized.points)

    def test_tuple_of_sets_round_trip(self) -> None:
        """Test tuple containing sets round-trip."""

        class TupleSetRT(Node[None], tag="tuple_set_rt"):
            data: tuple[set[int], set[str]]

        original = TupleSetRT(data=({1, 2}, {"a", "b"}))
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert isinstance(deserialized.data, tuple)
        assert isinstance(deserialized.data[0], set)
        assert isinstance(deserialized.data[1], set)

    def test_nested_node_with_tuple_round_trip(self) -> None:
        """Test nested node with tuple field round-trip."""

        class InnerTuple(Node[None], tag="inner_tuple_rt"):
            pair: tuple[int, str]

        class OuterTuple(Node[None], tag="outer_tuple_rt"):
            child: Node[None]

        original = OuterTuple(child=InnerTuple(pair=(42, "hello")))
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original
        assert isinstance(deserialized.child, InnerTuple)
        assert isinstance(deserialized.child.pair, tuple)


class TestSerializeTypeDefTupleSet:
    """Test TypeDef serialization for tuple and set types."""

    def test_serialize_tuple_type(self) -> None:
        """Test serializing TupleType."""
        tuple_type = TupleType(elements=(IntType(), StrType()))
        result = to_dict(tuple_type)

        assert result == {
            "tag": "tuple",
            "elements": [{"tag": "int"}, {"tag": "str"}],
        }

    def test_serialize_set_type(self) -> None:
        """Test serializing SetType."""
        set_type = SetType(element=IntType())
        result = to_dict(set_type)

        assert result == {"tag": "set", "element": {"tag": "int"}}

    def test_serialize_frozenset_typedef(self) -> None:
        """Test serializing FrozenSetType."""
        frozenset_type = FrozenSetType(element=StrType())
        result = to_dict(frozenset_type)

        assert result == {"tag": "frozenset", "element": {"tag": "str"}}
