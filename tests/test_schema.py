"""Tests for typedsl.schema module."""

import datetime
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import TypeVar

import pytest

from typedsl.nodes import Node
from typedsl.schema import extract_type, node_schema
from typedsl.types import (
    BoolType,
    BytesType,
    DateTimeType,
    DateType,
    DecimalType,
    DictType,
    DurationType,
    FloatType,
    FrozenSetType,
    IntType,
    ListType,
    MappingType,
    NodeType,
    NoneType,
    ReturnType,
    SequenceType,
    SetType,
    StrType,
    TimeType,
    TupleType,
    TypeParameter,
    UnionType,
)

# PEP 695 type alias for testing generic type alias extraction
type StrKeyDict[V] = dict[str, V]

# Simple type extraction test cases: (python_type, expected_typedef_class)
SIMPLE_TYPE_CASES = [
    (int, IntType),
    (float, FloatType),
    (str, StrType),
    (bool, BoolType),
    (type(None), NoneType),
    (bytes, BytesType),
    (Decimal, DecimalType),
    (datetime.date, DateType),
    (datetime.time, TimeType),
    (datetime.datetime, DateTimeType),
    (datetime.timedelta, DurationType),
]


class TestExtractSimpleTypes:
    """Test extracting simple types (primitives, temporal, binary)."""

    @pytest.mark.parametrize(("py_type", "expected_cls"), SIMPLE_TYPE_CASES)
    def test_extract_simple_type(self, py_type: type, expected_cls: type) -> None:
        """Test that simple Python types extract to correct TypeDef."""
        result = extract_type(py_type)
        assert isinstance(result, expected_cls)


class TestExtractTypeParameters:
    """Test extracting TypeVar (PEP 695 type parameters)."""

    def test_unbounded_type_parameter(self) -> None:
        """Test extracting an unbounded TypeVar."""
        T = TypeVar("T")
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert result.bound is None

    def test_bounded_type_parameter(self) -> None:
        """Test extracting a TypeVar with bound."""
        T = TypeVar("T", bound=int)
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert isinstance(result.bound, IntType)


class TestExtractUnion:
    """Test extracting Union types."""

    def test_two_type_union(self) -> None:
        """Test extracting int | str."""
        result = extract_type(int | str)
        assert isinstance(result, UnionType)
        assert len(result.options) == 2
        assert isinstance(result.options[0], IntType)
        assert isinstance(result.options[1], StrType)

    def test_multi_type_union(self) -> None:
        """Test extracting int | str | float."""
        result = extract_type(int | str | float)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3

    def test_union_with_containers(self) -> None:
        """Test extracting list[int] | dict[str, float] | None."""
        result = extract_type(list[int] | dict[str, float] | None)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3
        assert isinstance(result.options[0], ListType)
        assert isinstance(result.options[1], DictType)
        assert isinstance(result.options[2], NoneType)


class TestExtractContainers:
    """Test extracting container types."""

    def test_list(self) -> None:
        """Test extracting list[int]."""
        result = extract_type(list[int])
        assert isinstance(result, ListType)
        assert isinstance(result.element, IntType)

    def test_dict(self) -> None:
        """Test extracting dict[str, int]."""
        result = extract_type(dict[str, int])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)

    def test_set(self) -> None:
        """Test extracting set[int]."""
        result = extract_type(set[int])
        assert isinstance(result, SetType)
        assert isinstance(result.element, IntType)

    def test_frozenset(self) -> None:
        """Test extracting frozenset[int]."""
        result = extract_type(frozenset[int])
        assert isinstance(result, FrozenSetType)
        assert isinstance(result.element, IntType)

    def test_nested_containers(self) -> None:
        """Test extracting list[dict[str, list[int]]]."""
        result = extract_type(list[dict[str, list[int]]])
        assert isinstance(result, ListType)
        assert isinstance(result.element, DictType)
        assert isinstance(result.element.value, ListType)
        assert isinstance(result.element.value.element, IntType)


class TestExtractGenericContainers:
    """Test extracting generic container types (Sequence, Mapping)."""

    def test_sequence(self) -> None:
        """Test extracting Sequence[int]."""
        result = extract_type(Sequence[int])
        assert isinstance(result, SequenceType)
        assert isinstance(result.element, IntType)

    def test_mapping(self) -> None:
        """Test extracting Mapping[str, int]."""
        result = extract_type(Mapping[str, int])
        assert isinstance(result, MappingType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)

    def test_sequence_with_type_parameter(self) -> None:
        """Test extracting Sequence[T] where T is a type parameter."""
        T = TypeVar("T")
        result = extract_type(Sequence[T])
        assert isinstance(result, SequenceType)
        assert isinstance(result.element, TypeParameter)
        assert result.element.name == "T"

    def test_mapping_with_type_parameters(self) -> None:
        """Test extracting Mapping[K, V] where K, V are type parameters."""
        K = TypeVar("K")
        V = TypeVar("V")
        result = extract_type(Mapping[K, V])
        assert isinstance(result, MappingType)
        assert isinstance(result.key, TypeParameter)
        assert isinstance(result.value, TypeParameter)


class TestExtractTuples:
    """Test extracting tuple types."""

    def test_tuple_two_elements(self) -> None:
        """Test extracting tuple[int, str]."""
        result = extract_type(tuple[int, str])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 2
        assert isinstance(result.elements[0], IntType)
        assert isinstance(result.elements[1], StrType)

    def test_tuple_single_element(self) -> None:
        """Test extracting tuple[int]."""
        result = extract_type(tuple[int])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 1

    def test_empty_tuple(self) -> None:
        """Test that tuple[()] is valid as an empty tuple type."""
        result = extract_type(tuple[()])
        assert isinstance(result, TupleType)
        assert result.elements == ()

    def test_nested_tuple(self) -> None:
        """Test extracting tuple[tuple[int, int], str]."""
        result = extract_type(tuple[tuple[int, int], str])
        assert isinstance(result, TupleType)
        assert isinstance(result.elements[0], TupleType)
        assert len(result.elements[0].elements) == 2


class TestExtractWithTypeParameters:
    """Test extracting types with type parameters."""

    def test_list_with_type_parameter(self) -> None:
        """Test extracting list[T]."""
        T = TypeVar("T")
        result = extract_type(list[T])
        assert isinstance(result, ListType)
        assert isinstance(result.element, TypeParameter)

    def test_dict_with_type_parameter(self) -> None:
        """Test extracting dict[str, T]."""
        T = TypeVar("T")
        result = extract_type(dict[str, T])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, TypeParameter)


class TestPEP695TypeAlias:
    """Test PEP 695 type alias support."""

    def test_generic_type_alias(self) -> None:
        """Test extracting StrKeyDict[int] -> dict[str, int]."""
        result = extract_type(StrKeyDict[int])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)


class TestExtractNodeTypes:
    """Test extracting Node types as field types."""

    def test_generic_node_type(self) -> None:
        """Test that Node[T] extracts as ReturnType."""
        result = extract_type(Node[float])
        assert isinstance(result, ReturnType)
        assert isinstance(result.returns, FloatType)

    def test_simple_specific_node(self) -> None:
        """Test extracting a simple specific node class."""

        class SimpleConst(Node[int], tag="simple_const_test"):
            value: int

        result = extract_type(SimpleConst)
        assert isinstance(result, NodeType)
        assert result.node_tag == "simple_const_test"
        assert result.type_args == ()

    def test_parameterized_specific_node(self) -> None:
        """Test extracting a parameterized node like Const[float]."""

        class GenericConst[T](Node[T], tag="generic_const_test"):
            value: T

        result = extract_type(GenericConst[float])
        assert isinstance(result, NodeType)
        assert result.node_tag == "generic_const_test"
        assert len(result.type_args) == 1
        assert isinstance(result.type_args[0], FloatType)

    def test_node_with_transformed_return_type(self) -> None:
        """Test node where return type differs from type parameter."""

        class ArrayNode[T](Node[list[T]], tag="array_node_test"):
            items: list[T]

        result = extract_type(ArrayNode[str])
        assert isinstance(result, NodeType)
        assert result.node_tag == "array_node_test"
        assert isinstance(result.type_args[0], StrType)


class TestNodeSchema:
    """Test node_schema function for field extraction."""

    def test_specific_node_as_field(self) -> None:
        """Test that specific nodes used as fields are captured correctly."""

        class ValueNode[T](Node[T], tag="value_node_schema_test"):
            value: T

        class ContainerNode(Node[None], tag="container_node_schema_test"):
            child: ValueNode[float]

        schema = node_schema(ContainerNode)
        assert len(schema.fields) == 1
        child_type = schema.fields[0].type
        assert isinstance(child_type, NodeType)
        assert child_type.node_tag == "value_node_schema_test"
        assert isinstance(child_type.type_args[0], FloatType)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_type_raises(self) -> None:
        """Test that extracting an invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract type from"):
            extract_type(object())
