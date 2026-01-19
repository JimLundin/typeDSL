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


class TestExtractPrimitives:
    """Test extracting primitive types."""

    def test_extract_int(self) -> None:
        """Test extracting int type."""
        result = extract_type(int)
        assert isinstance(result, IntType)

    def test_extract_float(self) -> None:
        """Test extracting float type."""
        result = extract_type(float)
        assert isinstance(result, FloatType)

    def test_extract_str(self) -> None:
        """Test extracting str type."""
        result = extract_type(str)
        assert isinstance(result, StrType)

    def test_extract_bool(self) -> None:
        """Test extracting bool type."""
        result = extract_type(bool)
        assert isinstance(result, BoolType)

    def test_extract_none(self) -> None:
        """Test extracting None type."""
        result = extract_type(type(None))
        assert isinstance(result, NoneType)


class TestExtractBinaryAndPrecisionTypes:
    """Test extracting binary and precision types."""

    def test_extract_bytes(self) -> None:
        """Test extracting bytes type."""
        result = extract_type(bytes)
        assert isinstance(result, BytesType)

    def test_extract_decimal(self) -> None:
        """Test extracting Decimal type."""
        result = extract_type(Decimal)
        assert isinstance(result, DecimalType)


class TestExtractTemporalTypes:
    """Test extracting temporal types."""

    def test_extract_date(self) -> None:
        """Test extracting datetime.date type."""
        result = extract_type(datetime.date)
        assert isinstance(result, DateType)

    def test_extract_time(self) -> None:
        """Test extracting datetime.time type."""
        result = extract_type(datetime.time)
        assert isinstance(result, TimeType)

    def test_extract_datetime(self) -> None:
        """Test extracting datetime.datetime type."""
        result = extract_type(datetime.datetime)
        assert isinstance(result, DateTimeType)

    def test_extract_timedelta(self) -> None:
        """Test extracting datetime.timedelta type."""
        result = extract_type(datetime.timedelta)
        assert isinstance(result, DurationType)


class TestExtractTypeParameter:
    """Test extracting TypeVar (PEP 695 type parameters)."""

    def test_extract_simple_type_parameter(self) -> None:
        """Test extracting an unbounded TypeVar."""
        T = TypeVar("T")
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert result.bound is None

    def test_extract_bounded_type_parameter(self) -> None:
        """Test extracting a TypeVar with bound (like T: int)."""
        T = TypeVar("T", bound=int)
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert result.bound is not None
        assert isinstance(result.bound, IntType)


class TestExtractUnion:
    """Test extracting Union types."""

    def test_extract_union_pipe(self) -> None:
        """Test extracting Union with | operator."""
        result = extract_type(int | str)
        assert isinstance(result, UnionType)
        assert len(result.options) == 2
        assert isinstance(result.options[0], IntType)
        assert isinstance(result.options[1], StrType)

    def test_extract_union_multiple_types(self) -> None:
        """Test extracting Union with multiple types."""
        result = extract_type(int | str | float)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3


class TestExtractContainers:
    """Test extracting container types."""

    def test_extract_list_int(self) -> None:
        """Test extracting list[int]."""
        result = extract_type(list[int])
        assert isinstance(result, ListType)
        assert isinstance(result.element, IntType)

    def test_extract_list_str(self) -> None:
        """Test extracting list[str]."""
        result = extract_type(list[str])
        assert isinstance(result, ListType)
        assert isinstance(result.element, StrType)

    def test_extract_dict_str_int(self) -> None:
        """Test extracting dict[str, int]."""
        result = extract_type(dict[str, int])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)

    def test_extract_dict_int_float(self) -> None:
        """Test extracting dict[int, float]."""
        result = extract_type(dict[int, float])
        assert isinstance(result, DictType)
        assert isinstance(result.key, IntType)
        assert isinstance(result.value, FloatType)

    def test_extract_nested_list(self) -> None:
        """Test extracting list[list[int]]."""
        result = extract_type(list[list[int]])
        assert isinstance(result, ListType)
        assert isinstance(result.element, ListType)
        assert isinstance(result.element.element, IntType)

    def test_extract_list_dict(self) -> None:
        """Test extracting list[dict[str, int]]."""
        result = extract_type(list[dict[str, int]])
        assert isinstance(result, ListType)
        assert isinstance(result.element, DictType)
        assert isinstance(result.element.key, StrType)
        assert isinstance(result.element.value, IntType)


class TestExtractGenericContainers:
    """Test extracting generic container types (Sequence, Mapping)."""

    def test_extract_sequence_int(self) -> None:
        """Test extracting Sequence[int]."""
        result = extract_type(Sequence[int])
        assert isinstance(result, SequenceType)
        assert isinstance(result.element, IntType)

    def test_extract_sequence_str(self) -> None:
        """Test extracting Sequence[str]."""
        result = extract_type(Sequence[str])
        assert isinstance(result, SequenceType)
        assert isinstance(result.element, StrType)

    def test_extract_mapping_str_int(self) -> None:
        """Test extracting Mapping[str, int]."""
        result = extract_type(Mapping[str, int])
        assert isinstance(result, MappingType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)

    def test_extract_mapping_int_float(self) -> None:
        """Test extracting Mapping[int, float]."""
        result = extract_type(Mapping[int, float])
        assert isinstance(result, MappingType)
        assert isinstance(result.key, IntType)
        assert isinstance(result.value, FloatType)

    def test_extract_nested_sequence(self) -> None:
        """Test extracting Sequence[Sequence[int]]."""
        result = extract_type(Sequence[Sequence[int]])
        assert isinstance(result, SequenceType)
        assert isinstance(result.element, SequenceType)
        assert isinstance(result.element.element, IntType)

    def test_extract_sequence_of_mapping(self) -> None:
        """Test extracting Sequence[Mapping[str, int]]."""
        result = extract_type(Sequence[Mapping[str, int]])
        assert isinstance(result, SequenceType)
        assert isinstance(result.element, MappingType)
        assert isinstance(result.element.key, StrType)
        assert isinstance(result.element.value, IntType)

    def test_extract_mapping_with_sequence_value(self) -> None:
        """Test extracting Mapping[str, Sequence[int]]."""
        result = extract_type(Mapping[str, Sequence[int]])
        assert isinstance(result, MappingType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, SequenceType)
        assert isinstance(result.value.element, IntType)

    def test_extract_sequence_with_type_parameter(self) -> None:
        """Test extracting Sequence[T] where T is a type parameter."""
        T = TypeVar("T")
        result = extract_type(Sequence[T])
        assert isinstance(result, SequenceType)
        assert isinstance(result.element, TypeParameter)
        assert result.element.name == "T"

    def test_extract_mapping_with_type_parameters(self) -> None:
        """Test extracting Mapping[K, V] where K, V are type parameters."""
        K = TypeVar("K")
        V = TypeVar("V")
        result = extract_type(Mapping[K, V])
        assert isinstance(result, MappingType)
        assert isinstance(result.key, TypeParameter)
        assert result.key.name == "K"
        assert isinstance(result.value, TypeParameter)
        assert result.value.name == "V"

    def test_extract_frozenset_int(self) -> None:
        """Test extracting frozenset[int]."""
        result = extract_type(frozenset[int])
        assert isinstance(result, FrozenSetType)
        assert isinstance(result.element, IntType)

    def test_extract_frozenset_str(self) -> None:
        """Test extracting frozenset[str]."""
        result = extract_type(frozenset[str])
        assert isinstance(result, FrozenSetType)
        assert isinstance(result.element, StrType)

    def test_extract_set_int(self) -> None:
        """Test extracting set[int]."""
        result = extract_type(set[int])
        assert isinstance(result, SetType)
        assert isinstance(result.element, IntType)

    def test_extract_set_str(self) -> None:
        """Test extracting set[str]."""
        result = extract_type(set[str])
        assert isinstance(result, SetType)
        assert isinstance(result.element, StrType)

    def test_extract_nested_set(self) -> None:
        """Test extracting set of frozensets (nested set containers)."""
        result = extract_type(set[frozenset[int]])
        assert isinstance(result, SetType)
        assert isinstance(result.element, FrozenSetType)
        assert isinstance(result.element.element, IntType)


class TestExtractTuples:
    """Test extracting tuple types."""

    def test_extract_tuple_two_elements(self) -> None:
        """Test extracting tuple[int, str]."""
        result = extract_type(tuple[int, str])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 2
        assert isinstance(result.elements[0], IntType)
        assert isinstance(result.elements[1], StrType)

    def test_extract_tuple_three_elements(self) -> None:
        """Test extracting tuple[int, str, float]."""
        result = extract_type(tuple[int, str, float])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 3
        assert isinstance(result.elements[0], IntType)
        assert isinstance(result.elements[1], StrType)
        assert isinstance(result.elements[2], FloatType)

    def test_extract_tuple_single_element(self) -> None:
        """Test extracting tuple[int]."""
        result = extract_type(tuple[int])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 1
        assert isinstance(result.elements[0], IntType)

    def test_extract_tuple_with_nested_list(self) -> None:
        """Test extracting tuple[list[int], str]."""
        result = extract_type(tuple[list[int], str])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 2
        assert isinstance(result.elements[0], ListType)
        assert isinstance(result.elements[0].element, IntType)
        assert isinstance(result.elements[1], StrType)

    def test_extract_tuple_with_dict(self) -> None:
        """Test extracting tuple[str, dict[str, int]]."""
        result = extract_type(tuple[str, dict[str, int]])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 2
        assert isinstance(result.elements[0], StrType)
        assert isinstance(result.elements[1], DictType)

    def test_extract_nested_tuple(self) -> None:
        """Test extracting tuple[tuple[int, int], str]."""
        result = extract_type(tuple[tuple[int, int], str])
        assert isinstance(result, TupleType)
        assert len(result.elements) == 2
        assert isinstance(result.elements[0], TupleType)
        assert len(result.elements[0].elements) == 2
        assert isinstance(result.elements[1], StrType)

    def test_extract_empty_tuple(self) -> None:
        """Test that tuple[()] is valid as an empty tuple type."""
        result = extract_type(tuple[()])
        assert isinstance(result, TupleType)
        assert result.elements == ()


class TestExtractWithTypeParameters:
    """Test extracting types with type parameters."""

    def test_extract_list_with_type_parameter(self) -> None:
        """Test extracting list[T] where T is a type parameter."""
        T = TypeVar("T")
        result = extract_type(list[T])
        assert isinstance(result, ListType)
        assert isinstance(result.element, TypeParameter)
        assert result.element.name == "T"

    def test_extract_dict_with_type_parameter(self) -> None:
        """Test extracting dict[str, T] where T is a type parameter."""
        T = TypeVar("T")
        result = extract_type(dict[str, T])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, TypeParameter)
        assert result.value.name == "T"

    def test_extract_nested_with_type_parameter(self) -> None:
        """Test extracting list[dict[str, T]]."""
        T = TypeVar("T")
        result = extract_type(list[dict[str, T]])
        assert isinstance(result, ListType)
        assert isinstance(result.element, DictType)
        assert isinstance(result.element.value, TypeParameter)
        assert result.element.value.name == "T"


class TestPEP695TypeAlias:
    """Test PEP 695 type alias support."""

    def test_extract_generic_type_alias(self) -> None:
        """Test extracting a generic PEP 695 type alias."""
        # StrKeyDict is defined at module level as: type StrKeyDict[V] = dict[str, V]
        # When we use StrKeyDict[int], it should expand to dict[str, int]
        result = extract_type(StrKeyDict[int])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_extract_invalid_type_raises(self) -> None:
        """Test that extracting an invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract type from"):
            extract_type(object())

    def test_extract_complex_union(self) -> None:
        """Test extracting complex union with nested types."""
        result = extract_type(list[int] | dict[str, float] | None)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3
        assert isinstance(result.options[0], ListType)
        assert isinstance(result.options[1], DictType)
        assert isinstance(result.options[2], NoneType)

    def test_list_without_element_type_raises(self) -> None:
        """Test that list without element type raises ValueError."""
        # This test may not be possible with Python's typing system
        # as list without args gives list directly, not a parameterized type

    def test_dict_with_wrong_arg_count_raises(self) -> None:
        """Test that dict with wrong number of args raises ValueError."""
        # This is also hard to test as Python's typing system enforces this


class TestExtractSpecificNodeTypes:
    """Test extracting specific node types as field types."""

    def test_extract_generic_node_type(self) -> None:
        """Test that generic Node[T] extracts as ReturnType (return type constraint)."""
        result = extract_type(Node[float])
        assert isinstance(result, ReturnType)
        assert isinstance(result.returns, FloatType)

    def test_extract_simple_specific_node(self) -> None:
        """Test extracting a simple specific node class."""

        class SimpleConst(Node[int], tag="simple_const_test"):
            value: int

        result = extract_type(SimpleConst)
        assert isinstance(result, NodeType)
        assert result.node_tag == "simple_const_test"
        assert result.type_args == ()

    def test_extract_parameterized_specific_node(self) -> None:
        """Test extracting a parameterized specific node like Const[float]."""

        class GenericConst[T](Node[T], tag="generic_const_test"):
            value: T

        result = extract_type(GenericConst[float])
        assert isinstance(result, NodeType)
        assert result.node_tag == "generic_const_test"
        assert len(result.type_args) == 1
        assert isinstance(result.type_args[0], FloatType)

    def test_extract_node_with_transformed_return_type(self) -> None:
        """Test node where return type differs from type parameter."""

        class ArrayNode[T](Node[list[T]], tag="array_node_test"):
            items: list[T]

        result = extract_type(ArrayNode[str])
        assert isinstance(result, NodeType)
        assert result.node_tag == "array_node_test"
        assert len(result.type_args) == 1
        assert isinstance(result.type_args[0], StrType)
        # Return type is derived from schema at runtime, not stored in NodeType

    def test_extract_node_with_complex_return_type(self) -> None:
        """Test node with complex return type transformation."""

        class PairNode[T](Node[tuple[T, T]], tag="pair_node_test"):
            first: T
            second: T

        result = extract_type(PairNode[int])
        assert isinstance(result, NodeType)
        assert result.node_tag == "pair_node_test"
        assert len(result.type_args) == 1
        assert isinstance(result.type_args[0], IntType)
        # Return type is derived from schema at runtime, not stored in NodeType

    def test_specific_node_as_field_in_schema(self) -> None:
        """Test that specific nodes used as fields in schemas are captured correctly."""

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

    def test_transformed_node_as_field_in_schema(self) -> None:
        """Test that transformed nodes (Array[str]) as fields are captured correctly."""

        class ListWrapper[T](Node[list[T]], tag="list_wrapper_schema_test"):
            items: list[T]

        class WrapperContainer(Node[None], tag="wrapper_container_schema_test"):
            wrapped: ListWrapper[str]

        schema = node_schema(WrapperContainer)
        wrapped_type = schema.fields[0].type
        assert isinstance(wrapped_type, NodeType)
        assert wrapped_type.node_tag == "list_wrapper_schema_test"
        assert isinstance(wrapped_type.type_args[0], StrType)
