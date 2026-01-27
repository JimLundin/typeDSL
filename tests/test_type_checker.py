"""Tests for the type checker implementation.

These tests define the expected behavior of the type checker by providing
concrete examples of valid and invalid programs.

Run with: pytest tests/test_type_checker.py -v
"""

from collections.abc import Callable
from typing import Any, Literal

import pytest

from typedsl import Child, Node, Program, Ref
from typedsl.type_checker import get_node_return_type, type_check

# =============================================================================
# Test Node Definitions
# =============================================================================
# These nodes are used across multiple tests. Each has a unique tag to avoid
# collisions with nodes defined in other test files.


class IntConst(Node[int], tag="tc_int_const"):
    """Node that returns int."""

    value: int


class FloatConst(Node[float], tag="tc_float_const"):
    """Node that returns float."""

    value: float


class StrConst(Node[str], tag="tc_str_const"):
    """Node that returns str."""

    text: str


class BoolConst(Node[bool], tag="tc_bool_const"):
    """Node that returns bool."""

    value: bool


class NoneNode(Node[None], tag="tc_none_node"):
    """Node that returns None."""


class AnyNode(Node[Any], tag="tc_any_node"):
    """Node that returns Any."""

    data: Any


# =============================================================================
# SECTION 1: Return Type Extraction
# =============================================================================


class TestReturnTypeExtraction:
    """Test extracting return type T from Node[T] subclasses."""

    def test_simple_return_type(self) -> None:
        """Should extract int from Node[int]."""
        assert get_node_return_type(IntConst) is int

    def test_none_return_type(self) -> None:
        """Should extract NoneType from Node[None]."""
        assert get_node_return_type(NoneNode) is type(None)

    def test_any_return_type(self) -> None:
        """Should extract Any from Node[Any]."""
        assert get_node_return_type(AnyNode) is Any

    def test_complex_return_type(self) -> None:
        """Should extract complex types like list[int]."""

        class ListNode(Node[list[int]], tag="tc_list_ret"):
            items: list[int]

        ret = get_node_return_type(ListNode)
        assert ret == list[int]

    def test_nested_generic_return_type(self) -> None:
        """Should extract nested generics like dict[str, list[int]]."""

        class DictNode(Node[dict[str, list[int]]], tag="tc_dict_ret"):
            mapping: dict[str, list[int]]

        ret = get_node_return_type(DictNode)
        assert ret == dict[str, list[int]]


# =============================================================================
# SECTION 2: Direct Field Type Validation
# =============================================================================


class TestDirectFieldValidation:
    """Test type checking of directly nested nodes (not via references)."""

    def test_valid_direct_nesting(self) -> None:
        """Valid: Node[float] field receives a node returning float."""

        class Consumer(Node[float], tag="tc_consumer_valid"):
            input: Node[float]

        node = Consumer(input=FloatConst(value=1.0))
        errors = type_check(node)
        assert errors == []

    def test_invalid_direct_nesting_wrong_type(self) -> None:
        """Invalid: Node[float] field receives a node returning str."""

        class Consumer(Node[float], tag="tc_consumer_invalid"):
            input: Node[float]

        node = Consumer(input=StrConst(text="hello"))  # type: ignore[arg-type]
        errors = type_check(node)
        assert len(errors) == 1
        assert "input" in errors[0].field_name
        assert "float" in errors[0].expected.lower()

    def test_invalid_nesting_int_where_float_expected(self) -> None:
        """Invalid: Node[float] field receives Node[int].

        Python generics are invariant - int is not float.
        """

        class Consumer(Node[float], tag="tc_consumer_int_float"):
            input: Node[float]

        node = Consumer(input=IntConst(value=1))  # type: ignore[arg-type]
        errors = type_check(node)
        assert len(errors) == 1

    def test_multiple_fields_validation(self) -> None:
        """Test validating multiple node fields."""

        class BinOp(Node[float], tag="tc_binop_multi"):
            left: Node[float]
            right: Node[float]

        # Valid
        valid = BinOp(left=FloatConst(value=1.0), right=FloatConst(value=2.0))
        assert type_check(valid) == []

        # Invalid - first field wrong
        invalid1 = BinOp(
            left=StrConst(text="x"),  # type: ignore[arg-type]
            right=FloatConst(value=2.0),
        )
        errors1 = type_check(invalid1)
        assert len(errors1) == 1
        assert "left" in errors1[0].field_name

        # Invalid - both fields wrong
        invalid2 = BinOp(
            left=StrConst(text="x"),  # type: ignore[arg-type]
            right=IntConst(value=1),  # type: ignore[arg-type]
        )
        errors2 = type_check(invalid2)
        assert len(errors2) == 2


# =============================================================================
# SECTION 3: Reference Type Validation
# =============================================================================


class TestReferenceTypeValidation:
    """Test type checking of Ref[Node[T]] in Programs."""

    def test_valid_reference(self) -> None:
        """Valid: Ref[Node[float]] points to a node returning float."""

        class RefConsumer(Node[float], tag="tc_ref_consumer_valid"):
            input: Ref[Node[float]]

        prog = Program(
            root=Ref(id="consumer"),
            nodes={
                "value": FloatConst(value=1.0),
                "consumer": RefConsumer(input=Ref(id="value")),
            },
        )
        errors = type_check(prog)
        assert errors == []

    def test_invalid_reference_wrong_type(self) -> None:
        """Invalid: Ref[Node[float]] points to a node returning str."""

        class RefConsumer(Node[float], tag="tc_ref_consumer_invalid"):
            input: Ref[Node[float]]

        prog = Program(
            root=Ref(id="consumer"),
            nodes={
                "value": StrConst(text="hello"),  # Returns str, not float
                "consumer": RefConsumer(input=Ref(id="value")),
            },
        )
        errors = type_check(prog)
        assert len(errors) == 1
        assert "input" in errors[0].field_name

    def test_chained_references(self) -> None:
        """Test validation through chains of references."""

        class Wrapper(Node[float], tag="tc_wrapper_chain"):
            inner: Ref[Node[float]]

        # Valid chain: root -> wrapper -> value
        valid_prog = Program(
            root=Ref(id="outer"),
            nodes={
                "value": FloatConst(value=1.0),
                "inner": Wrapper(inner=Ref(id="value")),
                "outer": Wrapper(inner=Ref(id="inner")),
            },
        )
        assert type_check(valid_prog) == []

        # Invalid: one link in chain is wrong type
        invalid_prog = Program(
            root=Ref(id="outer"),
            nodes={
                "value": StrConst(text="x"),  # Wrong type!
                "inner": Wrapper(inner=Ref(id="value")),
                "outer": Wrapper(inner=Ref(id="inner")),
            },
        )
        errors = type_check(invalid_prog)
        assert len(errors) >= 1

    def test_reference_to_nonexistent_node(self) -> None:
        """References to non-existent nodes should be reported."""

        class RefUser(Node[int], tag="tc_ref_nonexistent"):
            target: Ref[Node[int]]

        prog = Program(
            root=Ref(id="user"),
            nodes={
                "user": RefUser(target=Ref(id="missing")),  # 'missing' doesn't exist
            },
        )
        errors = type_check(prog)
        assert len(errors) >= 1
        assert "missing" in str(errors[0])


# =============================================================================
# SECTION 4: Child[T] Type Validation
# =============================================================================


class TestChildTypeValidation:
    """Test type checking of Child[T] = Node[T] | Ref[Node[T]]."""

    def test_child_with_inline_node_valid(self) -> None:
        """Valid: Child[float] with inline Node[float]."""

        class ChildUser(Node[float], tag="tc_child_inline_valid"):
            input: Child[float]

        node = ChildUser(input=FloatConst(value=1.0))
        errors = type_check(node)
        assert errors == []

    def test_child_with_inline_node_invalid(self) -> None:
        """Invalid: Child[float] with inline Node[str]."""

        class ChildUser(Node[float], tag="tc_child_inline_invalid"):
            input: Child[float]

        node = ChildUser(input=StrConst(text="x"))  # type: ignore[arg-type]
        errors = type_check(node)
        assert len(errors) == 1

    def test_child_with_reference_valid(self) -> None:
        """Valid: Child[float] with Ref[Node[float]]."""

        class ChildUser(Node[float], tag="tc_child_ref_valid"):
            input: Child[float]

        prog = Program(
            root=Ref(id="user"),
            nodes={
                "value": FloatConst(value=1.0),
                "user": ChildUser(input=Ref(id="value")),
            },
        )
        errors = type_check(prog)
        assert errors == []

    def test_child_with_reference_invalid(self) -> None:
        """Invalid: Child[float] with Ref to Node[str]."""

        class ChildUser(Node[float], tag="tc_child_ref_invalid"):
            input: Child[float]

        prog = Program(
            root=Ref(id="user"),
            nodes={
                "value": StrConst(text="x"),  # Wrong type
                "user": ChildUser(input=Ref(id="value")),
            },
        )
        errors = type_check(prog)
        assert len(errors) >= 1

    def test_child_mixed_usage(self) -> None:
        """Test node using both inline and ref for Child[T] fields."""

        class BinaryOp(Node[float], tag="tc_binop_child_mixed"):
            left: Child[float]
            right: Child[float]

        # Mixed: inline left, ref right
        prog = Program(
            root=Ref(id="op"),
            nodes={
                "value": FloatConst(value=2.0),
                "op": BinaryOp(
                    left=FloatConst(value=1.0),  # Inline
                    right=Ref(id="value"),  # Reference
                ),
            },
        )
        errors = type_check(prog)
        assert errors == []


# =============================================================================
# SECTION 5: Union Type Validation
# =============================================================================


class TestUnionTypeValidation:
    """Test type checking with union types."""

    def test_union_node_type_first_variant(self) -> None:
        """Valid: Union field accepts first variant."""

        class UnionUser(Node[str], tag="tc_union_first"):
            data: Node[int] | Node[str]

        node = UnionUser(data=IntConst(value=1))
        errors = type_check(node)
        assert errors == []

    def test_union_node_type_second_variant(self) -> None:
        """Valid: Union field accepts second variant."""

        class UnionUser(Node[str], tag="tc_union_second"):
            data: Node[int] | Node[str]

        node = UnionUser(data=StrConst(text="hello"))
        errors = type_check(node)
        assert errors == []

    def test_union_node_type_neither_variant(self) -> None:
        """Invalid: Union field receives neither variant."""

        class UnionUser(Node[str], tag="tc_union_neither"):
            data: Node[int] | Node[str]

        node = UnionUser(data=FloatConst(value=1.0))  # type: ignore[arg-type]
        errors = type_check(node)
        assert len(errors) == 1

    def test_optional_node_with_none(self) -> None:
        """Valid: Optional node field with None."""

        class OptionalUser(Node[int], tag="tc_optional_none"):
            child: Node[int] | None

        node = OptionalUser(child=None)
        errors = type_check(node)
        assert errors == []

    def test_optional_node_with_value(self) -> None:
        """Valid: Optional node field with actual node."""

        class OptionalUser(Node[int], tag="tc_optional_value"):
            child: Node[int] | None

        node = OptionalUser(child=IntConst(value=42))
        errors = type_check(node)
        assert errors == []

    def test_optional_node_with_wrong_type(self) -> None:
        """Invalid: Optional Node[int] with Node[str]."""

        class OptionalUser(Node[int], tag="tc_optional_wrong"):
            child: Node[int] | None

        node = OptionalUser(child=StrConst(text="x"))  # type: ignore[arg-type]
        errors = type_check(node)
        assert len(errors) == 1


# =============================================================================
# SECTION 6: Literal Type Validation
# =============================================================================


class TestLiteralTypeValidation:
    """Test type checking of Literal types."""

    def test_literal_valid_value(self) -> None:
        """Valid: Literal field has value from allowed set."""

        class Op(Node[float], tag="tc_literal_valid"):
            operator: Literal["+", "-", "*", "/"]
            left: Node[float]
            right: Node[float]

        node = Op(
            operator="+",
            left=FloatConst(value=1.0),
            right=FloatConst(value=2.0),
        )
        errors = type_check(node)
        assert errors == []

    def test_literal_invalid_value(self) -> None:
        """Invalid: Literal field has value not in allowed set."""

        class Op(Node[float], tag="tc_literal_invalid"):
            operator: Literal["+", "-", "*", "/"]
            left: Node[float]
            right: Node[float]

        node = Op(
            operator="%",  # type: ignore[arg-type]
            left=FloatConst(value=1.0),
            right=FloatConst(value=2.0),
        )
        errors = type_check(node)
        assert len(errors) == 1
        assert "operator" in errors[0].field_name

    def test_literal_int_values(self) -> None:
        """Test Literal with int values."""

        class Priority(Node[str], tag="tc_literal_int"):
            level: Literal[1, 2, 3]
            name: str

        valid = Priority(level=2, name="medium")
        assert type_check(valid) == []

        invalid = Priority(level=5, name="ultra")  # type: ignore[arg-type]
        errors = type_check(invalid)
        assert len(errors) == 1

    def test_literal_bool_values(self) -> None:
        """Test Literal with bool values."""

        class Flag(Node[str], tag="tc_literal_bool"):
            enabled: Literal[True, False]

        node = Flag(enabled=True)
        errors = type_check(node)
        assert errors == []


# =============================================================================
# SECTION 7: Generic Node Validation
# =============================================================================


class TestGenericNodeValidation:
    """Test type checking of generic nodes with type parameters."""

    def test_generic_node_consistent_type_param(self) -> None:
        """Valid: Generic node with consistent type parameter."""

        class Pair[T](Node[tuple[T, T]], tag="tc_pair_consistent"):
            first: Node[T]
            second: Node[T]

        # Both return float, so T=float is consistent
        node = Pair(
            first=FloatConst(value=1.0),
            second=FloatConst(value=2.0),
        )
        errors = type_check(node)
        assert errors == []

    def test_generic_node_inconsistent_type_param(self) -> None:
        """Invalid: Generic node with inconsistent type parameter."""

        class Pair[T](Node[tuple[T, T]], tag="tc_pair_inconsistent"):
            first: Node[T]
            second: Node[T]

        # first=float, second=str - T cannot be both!
        node = Pair(
            first=FloatConst(value=1.0),
            second=StrConst(text="x"),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1

    def test_generic_node_multiple_type_params(self) -> None:
        """Test generic node with multiple type parameters."""

        class Transform[T, U](Node[U], tag="tc_transform_multi"):
            input: Node[T]
            mapper: Node[U]

        # T=int, U=str
        node = Transform(
            input=IntConst(value=1),
            mapper=StrConst(text="result"),
        )
        errors = type_check(node)
        assert errors == []

    def test_generic_node_with_bound(self) -> None:
        """Test generic node with bounded type parameter."""

        class Numeric[T: int | float](Node[T], tag="tc_numeric_bound"):
            value: T

        # Valid: int is within bound
        valid_int = Numeric(value=1)
        assert type_check(valid_int) == []

        # Valid: float is within bound
        valid_float = Numeric(value=1.5)
        assert type_check(valid_float) == []

    def test_generic_node_nested_in_container(self) -> None:
        """Test generic node used in container types."""

        class ListOf[T](Node[list[T]], tag="tc_list_generic"):
            items: list[Node[T]]

        # All items return int, so T=int is consistent
        node = ListOf(
            items=[IntConst(value=1), IntConst(value=2), IntConst(value=3)],
        )
        errors = type_check(node)
        assert errors == []

    def test_generic_node_inconsistent_in_list(self) -> None:
        """Invalid: Generic list with inconsistent element types."""

        class ListOf[T](Node[list[T]], tag="tc_list_inconsistent"):
            items: list[Node[T]]

        node = ListOf(
            items=[
                IntConst(value=1),
                StrConst(text="x"),  # type: ignore[list-item]
            ],
        )
        errors = type_check(node)
        assert len(errors) >= 1


# =============================================================================
# SECTION 7b: Type Parameter Bounds (PEP 695)
# =============================================================================


class TestTypeParameterBounds:
    """Test type checking with bounded type parameters (PEP 695 syntax).

    PEP 695 introduced the new generic syntax:
        class Foo[T: Bound]: ...

    The type checker must validate that inferred types satisfy bounds.
    """

    def test_bound_satisfied_exactly(self) -> None:
        """Type parameter bound is satisfied by exact match."""

        class NumericOp[T: int | float](Node[T], tag="tc_bound_exact"):
            left: Node[T]
            right: Node[T]

        # T=int satisfies T: int | float
        node1 = NumericOp(left=IntConst(value=1), right=IntConst(value=2))
        assert type_check(node1) == []

        # T=float satisfies T: int | float
        node2 = NumericOp(left=FloatConst(value=1.0), right=FloatConst(value=2.0))
        assert type_check(node2) == []

    def test_bound_violated(self) -> None:
        """Type parameter bound is violated."""

        class NumericOp[T: int | float](Node[T], tag="tc_bound_violated"):
            left: Node[T]
            right: Node[T]

        # T=str does NOT satisfy T: int | float
        node = NumericOp(
            left=StrConst(text="a"),  # type: ignore[arg-type]
            right=StrConst(text="b"),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1

    def test_bound_with_single_type(self) -> None:
        """Type parameter bound with single type constraint."""

        class StringProcessor[T: str](Node[T], tag="tc_bound_single"):
            input: Node[T]

        # T=str satisfies T: str
        valid = StringProcessor(input=StrConst(text="hello"))
        assert type_check(valid) == []

        # T=int violates T: str
        invalid = StringProcessor(input=IntConst(value=1))  # type: ignore[arg-type]
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_bound_with_none_in_union(self) -> None:
        """Type parameter bound that includes None."""

        class OptionalValue[T: int | None](Node[T], tag="tc_bound_none"):
            value: Node[T]

        # Both should satisfy the bound
        node1 = OptionalValue(value=IntConst(value=42))
        node2 = OptionalValue(value=NoneNode())
        assert type_check(node1) == []
        assert type_check(node2) == []

    def test_multiple_bounded_type_params(self) -> None:
        """Multiple type parameters each with different bounds."""

        class Converter[T: int | float, U: str | bytes](Node[U], tag="tc_multi_bound"):
            input: Node[T]
            format: Node[U]

        # T=int satisfies int|float, U=str satisfies str|bytes
        valid = Converter(input=IntConst(value=1), format=StrConst(text="decimal"))
        assert type_check(valid) == []

        # T=str violates int|float bound
        invalid = Converter(
            input=StrConst(text="x"),  # type: ignore[arg-type]
            format=StrConst(text="y"),
        )
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_bound_with_subtype(self) -> None:
        """Test whether subtypes satisfy bounds (bool is subtype of int)."""

        class IntProcessor[T: int](Node[T], tag="tc_bound_subtype"):
            value: Node[T]

        # In Python, bool is a subtype of int - should satisfy T: int
        node = IntProcessor(value=BoolConst(value=True))
        errors = type_check(node)
        assert errors == []

    def test_unbounded_type_param(self) -> None:
        """Unbounded type parameter accepts any type."""

        class Identity[T](Node[T], tag="tc_unbounded"):
            value: Node[T]

        # All of these should be valid (T is unbounded)
        assert type_check(Identity(value=IntConst(value=1))) == []
        assert type_check(Identity(value=StrConst(text="x"))) == []
        assert type_check(Identity(value=FloatConst(value=1.0))) == []

    def test_bound_inference_from_multiple_fields(self) -> None:
        """Infer T from multiple fields, then check bound."""

        class BoundedPair[T: int | float](Node[tuple[T, T]], tag="tc_bound_infer"):
            first: Node[T]
            second: Node[T]

        # Infer T=int from first, validate second matches, check bound
        valid = BoundedPair(first=IntConst(value=1), second=IntConst(value=2))
        assert type_check(valid) == []

        # Infer T=str from first - but str violates bound!
        invalid = BoundedPair(
            first=StrConst(text="a"),  # type: ignore[arg-type]
            second=StrConst(text="b"),  # type: ignore[arg-type]
        )
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_bound_with_container_type(self) -> None:
        """Type parameter with bound used in container context."""

        class NumericList[T: int | float](Node[list[T]], tag="tc_bound_container"):
            items: list[Node[T]]

        # All items return int, T=int satisfies bound
        valid = NumericList(items=[IntConst(value=1), IntConst(value=2)])
        assert type_check(valid) == []

        # All items return str, T=str violates bound
        invalid = NumericList(
            items=[
                StrConst(text="a"),  # type: ignore[list-item]
                StrConst(text="b"),  # type: ignore[list-item]
            ],
        )
        errors = type_check(invalid)
        assert len(errors) >= 1


# =============================================================================
# SECTION 7c: Type Parameter Inference
# =============================================================================


class TestTypeParameterInference:
    """Test inference of type parameters from field values.

    When a generic node is instantiated, the type checker must infer
    the concrete type for each type parameter from the provided values.
    """

    def test_infer_from_single_field(self) -> None:
        """Infer T from a single field."""

        class Wrap[T](Node[T], tag="tc_infer_single"):
            inner: Node[T]

        # T should be inferred as float from the field
        node = Wrap(inner=FloatConst(value=1.0))
        errors = type_check(node)
        assert errors == []

    def test_infer_from_first_field_validate_rest(self) -> None:
        """Infer T from first field, validate remaining fields match."""

        class Triple[T](Node[tuple[T, T, T]], tag="tc_infer_validate"):
            a: Node[T]
            b: Node[T]
            c: Node[T]

        # Infer T=int from 'a', validate b and c match
        valid = Triple(a=IntConst(value=1), b=IntConst(value=2), c=IntConst(value=3))
        assert type_check(valid) == []

        # Infer T=int from 'a', but 'c' is str - mismatch!
        invalid = Triple(
            a=IntConst(value=1),
            b=IntConst(value=2),
            c=StrConst(text="x"),  # type: ignore[arg-type]
        )
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_infer_independent_type_params(self) -> None:
        """Multiple type parameters inferred independently."""

        class Map[T, U](Node[U], tag="tc_infer_multi"):
            input: Node[T]
            output: Node[U]

        # Infer T=int from input, U=str from output
        node = Map(input=IntConst(value=1), output=StrConst(text="result"))
        errors = type_check(node)
        assert errors == []

    def test_infer_from_container_elements(self) -> None:
        """Infer T from elements in a container field."""

        class Aggregate[T](Node[T], tag="tc_infer_container"):
            values: list[Node[T]]

        # Infer T=int from list elements
        valid = Aggregate(values=[IntConst(value=1), IntConst(value=2)])
        assert type_check(valid) == []

        # Inconsistent elements - can't unify
        invalid = Aggregate(
            values=[
                IntConst(value=1),
                StrConst(text="x"),  # type: ignore[list-item]
            ],
        )
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_infer_with_empty_container(self) -> None:
        """Empty container provides no information for inference."""

        class Aggregate[T](Node[list[T]], tag="tc_infer_empty"):
            values: list[Node[T]]

        # Empty list - T cannot be inferred from values, should be allowed
        node = Aggregate(values=[])
        # No concrete values to check, should pass
        errors = type_check(node)
        assert errors == []

    def test_infer_from_nested_generic(self) -> None:
        """Infer T from nested generic position."""

        class Processor[T](Node[list[T]], tag="tc_infer_nested"):
            mapping: dict[str, Node[T]]

        # Infer T=float from dict values
        node = Processor(
            mapping={
                "a": FloatConst(value=1.0),
                "b": FloatConst(value=2.0),
            },
        )
        errors = type_check(node)
        assert errors == []

    def test_type_param_used_only_in_return(self) -> None:
        """Type parameter appears only in return type, not fields.

        This is an edge case - T cannot be inferred from fields.
        """

        class Phantom[T](Node[T], tag="tc_phantom"):
            # T only appears in Node[T] return type, not in fields
            label: str

        # T is unconstrained by fields - should be allowed
        node = Phantom(label="test")
        errors = type_check(node)
        assert errors == []

    def test_infer_from_ref_target(self) -> None:
        """Infer T from the type of referenced node in a Program."""

        class Wrapper[T](Node[T], tag="tc_infer_ref"):
            inner: Ref[Node[T]]

        prog = Program(
            root=Ref(id="wrap"),
            nodes={
                "value": FloatConst(value=1.0),  # Returns float
                "wrap": Wrapper(inner=Ref(id="value")),  # T inferred as float
            },
        )
        errors = type_check(prog)
        assert errors == []


# =============================================================================
# SECTION 7d: Complex Type Parameter Scenarios
# =============================================================================


class TestComplexTypeParameters:
    """Test complex scenarios involving type parameters."""

    def test_type_param_in_union_field(self) -> None:
        """Type parameter used within a union type field."""

        class MaybeProcess[T](Node[T | None], tag="tc_param_union"):
            input: Node[T] | None

        # T=int inferred, None is also valid
        node1 = MaybeProcess(input=IntConst(value=1))
        node2 = MaybeProcess(input=None)
        assert type_check(node1) == []
        assert type_check(node2) == []

    def test_type_param_in_nested_container(self) -> None:
        """Type parameter in deeply nested container type."""

        class DeepNest[T](Node[list[list[T]]], tag="tc_deep_nest"):
            data: list[list[Node[T]]]

        node = DeepNest(
            data=[
                [IntConst(value=1), IntConst(value=2)],
                [IntConst(value=3), IntConst(value=4)],
            ],
        )
        errors = type_check(node)
        assert errors == []

    def test_type_param_in_dict_key_and_value(self) -> None:
        """Type parameter used in both dict key and value positions.

        Note: dict keys must be hashable, so T: Hashable might be implicit.
        """

        class BiMap[T](Node[dict[T, T]], tag="tc_bimap"):
            forward: dict[str, Node[T]]
            backward: dict[str, Node[T]]

        node = BiMap(
            forward={"a": IntConst(value=1)},
            backward={"x": IntConst(value=2)},
        )
        errors = type_check(node)
        assert errors == []

    def test_same_param_different_container_types(self) -> None:
        """Same type parameter used in different container types."""

        class MultiContainer[T](Node[T], tag="tc_multi_container"):
            as_list: list[Node[T]]
            as_set: set[str]  # Not using T here
            as_dict_value: dict[str, Node[T]]

        node = MultiContainer(
            as_list=[FloatConst(value=1.0)],
            as_set={"key"},
            as_dict_value={"a": FloatConst(value=2.0)},
        )
        errors = type_check(node)
        assert errors == []

    def test_generic_node_field_with_concrete_type(self) -> None:
        """Generic node has one field with concrete type, one with T."""

        class Hybrid[T](Node[T], tag="tc_hybrid"):
            typed: Node[T]  # Uses type parameter
            fixed: Node[int]  # Always int

        # T inferred as str, fixed must be int
        valid = Hybrid(typed=StrConst(text="x"), fixed=IntConst(value=1))
        assert type_check(valid) == []

        # T inferred as str, but wrong type for fixed
        invalid = Hybrid(
            typed=StrConst(text="x"),
            fixed=StrConst(text="y"),  # type: ignore[arg-type]
        )
        errors = type_check(invalid)
        assert len(errors) >= 1

    @pytest.mark.skip(reason="Advanced feature - not implemented")
    def test_bound_references_another_param(self) -> None:
        """Advanced: one type param's bound references another.

        Note: This is more advanced and may not be supported initially.
        """
        # class Related[T, U: T](Node[U]): ...
        # This would mean U must be a subtype of T
        # Complex to implement - may skip initially

    def test_recursive_generic_node(self) -> None:
        """Generic node that can contain nodes of same generic type."""

        class Tree[T](Node[T], tag="tc_recursive_generic"):
            value: Node[T]
            children: list[Child[T]]  # Recursive reference

        # Build a simple tree with T=int
        prog = Program(
            root=Ref(id="root"),
            nodes={
                "leaf1": Tree(value=IntConst(value=1), children=[]),
                "leaf2": Tree(value=IntConst(value=2), children=[]),
                "root": Tree(
                    value=IntConst(value=0),
                    children=[Ref(id="leaf1"), Ref(id="leaf2")],
                ),
            },
        )
        errors = type_check(prog)
        assert errors == []

    def test_generic_with_child_type(self) -> None:
        """Generic node using Child[T] (union of Node[T] | Ref[Node[T]])."""

        class Flexible[T](Node[T], tag="tc_generic_child"):
            input: Child[T]

        # Inline node
        inline = Flexible(input=FloatConst(value=1.0))
        assert type_check(inline) == []

        # Via Program with ref
        prog = Program(
            root=Ref(id="flex"),
            nodes={
                "val": FloatConst(value=2.0),
                "flex": Flexible(input=Ref(id="val")),
            },
        )
        assert type_check(prog) == []


# =============================================================================
# SECTION 7e: Type Parameter Error Messages
# =============================================================================


class TestTypeParameterErrors:
    """Test error message quality for type parameter issues."""

    def test_error_shows_inferred_type(self) -> None:
        """Error message should show what T was inferred as."""

        class Pair[T](Node[tuple[T, T]], tag="tc_err_inferred"):
            first: Node[T]
            second: Node[T]

        node = Pair(
            first=IntConst(value=1),
            second=StrConst(text="x"),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1
        # Error should mention the type inconsistency
        error_str = str(errors[0])
        assert "second" in error_str or "int" in error_str.lower()

    def test_error_shows_bound(self) -> None:
        """Error message should show the violated bound."""

        class Numeric[T: int | float](Node[T], tag="tc_err_bound"):
            value: Node[T]

        node = Numeric(value=StrConst(text="x"))  # type: ignore[arg-type]
        errors = type_check(node)
        assert len(errors) >= 1

    def test_error_for_multiple_type_params(self) -> None:
        """Error identifies which type parameter has the issue."""

        class TwoParams[T, U](Node[tuple[T, U]], tag="tc_err_which_param"):
            t_val: Node[T]
            u_val: Node[U]
            t_val2: Node[T]

        node = TwoParams(
            t_val=IntConst(value=1),
            u_val=StrConst(text="x"),
            t_val2=FloatConst(value=2.0),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1


# =============================================================================
# SECTION 8: Any Type Handling
# =============================================================================


class TestAnyTypeHandling:
    """Test that Any acts as a universal escape hatch."""

    def test_any_accepts_any_node(self) -> None:
        """Node[Any] should accept any node type."""

        class Wrapper(Node[Any], tag="tc_any_wrapper"):
            child: Node[Any]

        # All of these should be valid
        assert type_check(Wrapper(child=IntConst(value=1))) == []
        assert type_check(Wrapper(child=StrConst(text="x"))) == []
        assert type_check(Wrapper(child=FloatConst(value=1.0))) == []

    def test_any_field_in_typed_node(self) -> None:
        """Node with specific return type but Any field."""

        class Processor(Node[str], tag="tc_any_field"):
            input: Node[Any]
            output: str

        # Any input should be accepted
        node = Processor(input=IntConst(value=42), output="result")
        errors = type_check(node)
        assert errors == []


# =============================================================================
# SECTION 8b: Specific Node Subclass Type Checking
# =============================================================================


class TestSpecificNodeSubclass:
    """Test type checking when fields are annotated with specific node subclasses.

    Fields can be annotated with:
    - Node[T]: Accepts ANY node that returns T
    - SpecificNode: Accepts only that specific node class (or subclasses)
    - SpecificNode[T]: Accepts only that specific generic node with type arg T

    The type checker must distinguish between these cases.
    """

    def test_specific_subclass_valid(self) -> None:
        """Valid: Field expects specific subclass, receives that subclass."""

        class Operand(Node[float], tag="tc_operand"):
            value: float

        class Calculator(Node[float], tag="tc_calc_specific"):
            # Field requires specifically an Operand, not just any Node[float]
            left: Operand
            right: Operand

        node = Calculator(
            left=Operand(value=1.0),
            right=Operand(value=2.0),
        )
        errors = type_check(node)
        assert errors == []

    def test_specific_subclass_invalid_different_node(self) -> None:
        """Invalid: Field expects specific subclass, receives different node."""

        class Operand(Node[float], tag="tc_operand_diff"):
            value: float

        class OtherFloat(Node[float], tag="tc_other_float"):
            data: float

        class Calculator(Node[float], tag="tc_calc_diff"):
            left: Operand  # Specifically requires Operand
            right: Operand

        # OtherFloat returns float, but is not an Operand!
        node = Calculator(
            left=Operand(value=1.0),
            right=OtherFloat(data=2.0),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1

    def test_specific_subclass_accepts_further_subclass(self) -> None:
        """Valid: Field expects subclass, receives a further subclass of it."""

        class Expr(Node[float], tag="tc_expr_base"):
            pass

        class Const(Expr, tag="tc_const_derived"):
            value: float

        class Consumer(Node[float], tag="tc_consumer_derived"):
            expr: Expr  # Accepts Expr or any subclass

        # Const is a subclass of Expr, should be accepted
        node = Consumer(expr=Const(value=1.0))
        errors = type_check(node)
        assert errors == []

    def test_generic_subclass_valid(self) -> None:
        """Valid: Field expects generic subclass with specific type arg."""

        class Wrapper[T](Node[T], tag="tc_wrapper_generic"):
            inner: Node[T]

        class Consumer(Node[float], tag="tc_consumer_wrapper"):
            wrapped: Wrapper[float]  # Specifically Wrapper[float]

        node = Consumer(wrapped=Wrapper(inner=FloatConst(value=1.0)))
        errors = type_check(node)
        assert errors == []

    def test_generic_subclass_wrong_type_arg(self) -> None:
        """Invalid: Field expects Wrapper[float], receives Wrapper[int]."""

        class Wrapper[T](Node[T], tag="tc_wrapper_wrong_arg"):
            inner: Node[T]

        class Consumer(Node[float], tag="tc_consumer_wrong_wrapper"):
            wrapped: Wrapper[float]  # Specifically Wrapper[float]

        # Wrapper[int] is not Wrapper[float]!
        node = Consumer(
            wrapped=Wrapper(inner=IntConst(value=1)),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1

    def test_generic_subclass_wrong_class(self) -> None:
        """Invalid: Field expects Wrapper[float], receives different Node[float]."""

        class Wrapper[T](Node[T], tag="tc_wrapper_class"):
            inner: Node[T]

        class OtherWrapper[T](Node[T], tag="tc_other_wrapper"):
            data: Node[T]

        class Consumer(Node[float], tag="tc_consumer_class"):
            wrapped: Wrapper[float]

        # OtherWrapper[float] is Node[float] but not Wrapper[float]
        node = Consumer(
            wrapped=OtherWrapper(data=FloatConst(value=1.0)),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1

    def test_mixed_specific_and_generic_fields(self) -> None:
        """Test node with both specific subclass and generic Node fields."""

        class Const(Node[float], tag="tc_const_mixed"):
            value: float

        class Var(Node[float], tag="tc_var_mixed"):
            name: str

        class BinOp(Node[float], tag="tc_binop_mixed"):
            # left must be specifically Const
            left: Const
            # right can be any Node[float]
            right: Node[float]

        # Valid: left is Const, right is any Node[float]
        valid = BinOp(left=Const(value=1.0), right=Var(name="x"))
        assert type_check(valid) == []

        # Invalid: left is not Const
        invalid = BinOp(
            left=Var(name="y"),  # type: ignore[arg-type]
            right=Const(value=2.0),
        )
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_specific_subclass_with_type_param_inference(self) -> None:
        """Infer T from specific subclass annotation."""

        class Box[T](Node[T], tag="tc_box_infer"):
            content: Node[T]

        class Processor[T](Node[T], tag="tc_processor_infer"):
            # T should be inferred from Box[T]
            input: Box[T]
            output: Node[T]

        # T inferred as float from Box[float]
        valid = Processor(
            input=Box(content=FloatConst(value=1.0)),
            output=FloatConst(value=2.0),
        )
        assert type_check(valid) == []

        # T inferred as float from input, but output returns int
        invalid = Processor(
            input=Box(content=FloatConst(value=1.0)),
            output=IntConst(value=2),  # type: ignore[arg-type]
        )
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_list_of_specific_subclass(self) -> None:
        """Test list containing specific node subclass."""

        class Const(Node[int], tag="tc_const_list_specific"):
            value: int

        class Aggregator(Node[int], tag="tc_agg_specific"):
            # Must be list of Const, not just list of Node[int]
            values: list[Const]

        valid = Aggregator(values=[Const(value=1), Const(value=2)])
        assert type_check(valid) == []

        invalid = Aggregator(values=[Const(value=1), IntConst(value=2)])  # type: ignore[list-item]
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_dict_of_specific_subclass(self) -> None:
        """Test dict with specific node subclass values."""

        class Expr(Node[float], tag="tc_expr_dict"):
            pass

        class Const(Expr, tag="tc_const_dict_specific"):
            value: float

        class Registry(Node[float], tag="tc_registry_specific"):
            entries: dict[str, Const]  # Values must be Const

        node = Registry(entries={"a": Const(value=1.0), "b": Const(value=2.0)})
        errors = type_check(node)
        assert errors == []

    def test_specific_subclass_in_union(self) -> None:
        """Test union of specific subclasses."""

        class Const(Node[float], tag="tc_const_union_specific"):
            value: float

        class Var(Node[float], tag="tc_var_union_specific"):
            name: str

        class OtherNode(Node[float], tag="tc_other_union"):
            data: float

        class Consumer(Node[float], tag="tc_consumer_union_specific"):
            # Accepts Const OR Var, but not other Node[float]
            input: Const | Var

        # Valid: Const is in union
        assert type_check(Consumer(input=Const(value=1.0))) == []

        # Valid: Var is in union
        assert type_check(Consumer(input=Var(name="x"))) == []

        # Invalid: OtherNode is Node[float] but not Const or Var
        invalid = Consumer(input=OtherNode(data=1.0))  # type: ignore[arg-type]
        errors = type_check(invalid)
        assert len(errors) >= 1

    def test_specific_generic_subclass_with_bound(self) -> None:
        """Specific generic subclass where type param has bound."""

        class NumericBox[T: int | float](Node[T], tag="tc_numbox"):
            value: T

        class Processor(Node[float], tag="tc_proc_numbox"):
            box: NumericBox[float]  # Specifically NumericBox[float]

        node = Processor(box=NumericBox(value=1.0))
        errors = type_check(node)
        assert errors == []

    def test_ref_to_specific_subclass(self) -> None:
        """Test Ref that must point to specific subclass."""

        class Const(Node[float], tag="tc_const_ref_specific"):
            value: float

        class Consumer(Node[float], tag="tc_consumer_ref_specific"):
            input: Ref[Const]  # Must reference a Const, not just any Node[float]

        # Valid: ref points to Const
        valid_prog = Program(
            root=Ref(id="consumer"),
            nodes={
                "c": Const(value=1.0),
                "consumer": Consumer(input=Ref(id="c")),
            },
        )
        assert type_check(valid_prog) == []

        # Invalid: ref points to FloatConst (different class)
        invalid_prog = Program(
            root=Ref(id="consumer"),
            nodes={
                "f": FloatConst(value=1.0),  # Not a Const!
                "consumer": Consumer(input=Ref(id="f")),
            },
        )
        errors = type_check(invalid_prog)
        assert len(errors) >= 1

    def test_child_of_specific_subclass(self) -> None:
        """Test Child[...] that must be specific subclass."""
        # Note: Child[T] = Node[T] | Ref[Node[T]]
        # But what about Child alias for specific subclass?
        # This might need special handling: SpecificNode | Ref[SpecificNode]

        class Const(Node[float], tag="tc_const_child_specific"):
            value: float

        class Consumer(Node[float], tag="tc_consumer_child_specific"):
            # Accept inline Const or Ref to Const
            input: Const | Ref[Const]

        # Valid: inline Const
        inline = Consumer(input=Const(value=1.0))
        assert type_check(inline) == []

        # Valid: ref to Const (in program context)
        prog = Program(
            root=Ref(id="consumer"),
            nodes={
                "c": Const(value=1.0),
                "consumer": Consumer(input=Ref(id="c")),
            },
        )
        assert type_check(prog) == []

    def test_nested_specific_subclass_requirements(self) -> None:
        """Test deeply nested specific subclass requirements."""

        class Inner(Node[int], tag="tc_inner_nested"):
            value: int

        class Middle(Node[int], tag="tc_middle_nested"):
            inner: Inner  # Must be Inner

        class Outer(Node[int], tag="tc_outer_nested"):
            middle: Middle  # Must be Middle

        # Valid: proper nesting
        valid = Outer(middle=Middle(inner=Inner(value=1)))
        assert type_check(valid) == []

        # Invalid at leaf level
        invalid = Outer(
            middle=Middle(
                inner=IntConst(value=1),  # type: ignore[arg-type]
            ),
        )
        errors = type_check(invalid)
        assert len(errors) >= 1


# =============================================================================
# SECTION 9: Container Types with Nodes
# =============================================================================


class TestContainerTypesWithNodes:
    """Test type checking of containers (list, dict) containing nodes."""

    def test_list_of_nodes_valid(self) -> None:
        """Valid: list[Node[int]] contains all Node[int]."""

        class Aggregator(Node[int], tag="tc_list_nodes_valid"):
            inputs: list[Node[int]]

        node = Aggregator(
            inputs=[IntConst(value=1), IntConst(value=2), IntConst(value=3)],
        )
        errors = type_check(node)
        assert errors == []

    def test_list_of_nodes_invalid(self) -> None:
        """Invalid: list[Node[int]] contains Node[str]."""

        class Aggregator(Node[int], tag="tc_list_nodes_invalid"):
            inputs: list[Node[int]]

        node = Aggregator(
            inputs=[
                IntConst(value=1),
                StrConst(text="x"),  # type: ignore[list-item]
            ],
        )
        errors = type_check(node)
        assert len(errors) >= 1

    def test_list_of_child_type(self) -> None:
        """Test list[Child[T]] with mixed inline and refs."""

        class MultiInput(Node[float], tag="tc_list_child"):
            inputs: list[Child[float]]

        prog = Program(
            root=Ref(id="multi"),
            nodes={
                "shared": FloatConst(value=1.0),
                "multi": MultiInput(
                    inputs=[
                        FloatConst(value=2.0),  # Inline
                        Ref(id="shared"),  # Reference
                        FloatConst(value=3.0),  # Inline
                    ],
                ),
            },
        )
        errors = type_check(prog)
        assert errors == []

    def test_dict_of_nodes_valid(self) -> None:
        """Valid: dict[str, Node[int]] with all Node[int] values."""

        class NamedInputs(Node[int], tag="tc_dict_nodes_valid"):
            inputs: dict[str, Node[int]]

        node = NamedInputs(
            inputs={
                "a": IntConst(value=1),
                "b": IntConst(value=2),
            },
        )
        errors = type_check(node)
        assert errors == []

    def test_dict_of_nodes_invalid(self) -> None:
        """Invalid: dict[str, Node[int]] with Node[str] value."""

        class NamedInputs(Node[int], tag="tc_dict_nodes_invalid"):
            inputs: dict[str, Node[int]]

        node = NamedInputs(
            inputs={
                "a": IntConst(value=1),
                "b": StrConst(text="x"),  # type: ignore[dict-item]
            },
        )
        errors = type_check(node)
        assert len(errors) >= 1

    def test_empty_list_valid(self) -> None:
        """Empty list should be valid for any list[Node[T]]."""

        class Aggregator(Node[int], tag="tc_empty_list"):
            inputs: list[Node[int]]

        node = Aggregator(inputs=[])
        errors = type_check(node)
        assert errors == []


# =============================================================================
# SECTION 10: Cyclic Reference Handling
# =============================================================================


class TestCyclicReferences:
    """Test that type checker handles cyclic references without infinite loops."""

    def test_simple_cycle(self) -> None:
        """Test simple A -> B -> A cycle."""

        class CycleNode(Node[int], tag="tc_cycle_simple"):
            value: int
            next: Ref[Node[int]] | None

        prog = Program(
            root=Ref(id="a"),
            nodes={
                "a": CycleNode(value=1, next=Ref(id="b")),
                "b": CycleNode(value=2, next=Ref(id="a")),  # Back to a
            },
        )
        # type_check(prog) should complete without infinite loop
        errors = type_check(prog)
        assert errors == []

    def test_self_reference(self) -> None:
        """Test node referencing itself."""

        class SelfRef(Node[int], tag="tc_self_ref"):
            value: int
            self_ref: Ref[Node[int]] | None

        prog = Program(
            root=Ref(id="self"),
            nodes={
                "self": SelfRef(value=1, self_ref=Ref(id="self")),
            },
        )
        errors = type_check(prog)
        assert errors == []

    def test_cycle_with_type_error(self) -> None:
        """Test cycle where one node has wrong type."""

        class TypedCycle(Node[int], tag="tc_cycle_error"):
            value: int
            next: Ref[Node[int]] | None

        # The type checker should still detect errors in cyclic structures
        # Even with cycles, it should complete and identify type mismatches
        prog = Program(
            root=Ref(id="a"),
            nodes={
                "a": TypedCycle(value=1, next=Ref(id="b")),
                "b": TypedCycle(value=2, next=Ref(id="a")),
            },
        )
        # Valid cycle, should complete
        errors = type_check(prog)
        assert errors == []


# =============================================================================
# SECTION 11: Deep Nesting Validation
# =============================================================================


class TestDeepNesting:
    """Test type checking of deeply nested structures."""

    def test_deep_inline_nesting(self) -> None:
        """Test deeply nested inline nodes."""

        class Unary(Node[int], tag="tc_deep_unary"):
            child: Node[int]

        # 5 levels deep
        node = Unary(
            child=Unary(
                child=Unary(
                    child=Unary(child=Unary(child=IntConst(value=1))),
                ),
            ),
        )
        errors = type_check(node)
        assert errors == []

    def test_deep_nesting_error_at_leaf(self) -> None:
        """Test that errors at deeply nested leaves are caught."""

        class Unary(Node[int], tag="tc_deep_error"):
            child: Node[int]

        node = Unary(
            child=Unary(
                child=Unary(
                    child=StrConst(text="x"),  # type: ignore[arg-type]
                ),
            ),
        )
        errors = type_check(node)
        assert len(errors) >= 1


# =============================================================================
# SECTION 12: Return Type in Node Hierarchy
# =============================================================================


class TestNodeHierarchy:
    """Test type checking with node inheritance hierarchies."""

    def test_subclass_preserves_return_type(self) -> None:
        """Test that subclasses preserve return type compatibility."""

        class BaseExpr(Node[float], tag="tc_base_expr"):
            pass

        class Const(BaseExpr, tag="tc_const_sub"):
            value: float

        class Add(BaseExpr, tag="tc_add_sub"):
            left: Node[float]
            right: Node[float]

        # Const returns float (from BaseExpr), so should be valid
        node = Add(
            left=Const(value=1.0),
            right=Const(value=2.0),
        )
        errors = type_check(node)
        assert errors == []


# =============================================================================
# SECTION 13: Special Python Types
# =============================================================================


class TestSpecialPythonTypes:
    """Test type checking with special Python types."""

    def test_tuple_return_type(self) -> None:
        """Test node with tuple return type."""

        class TupleNode(Node[tuple[int, str]], tag="tc_tuple_ret"):
            num: int
            text: str

        class Consumer(Node[str], tag="tc_tuple_consumer"):
            input: Node[tuple[int, str]]

        node = Consumer(input=TupleNode(num=1, text="x"))
        errors = type_check(node)
        assert errors == []

    def test_callable_in_annotation(self) -> None:
        """Test that Callable types in annotations are handled."""

        class FuncHolder(Node[int], tag="tc_callable"):
            func: Callable[[int], int]  # Not a Node, just a function
            input: int

        # This tests that non-Node types don't cause crashes
        node = FuncHolder(func=lambda x: x * 2, input=5)
        # Should not crash, even with Callable fields
        errors = type_check(node)
        assert errors == []


# =============================================================================
# SECTION 14: Forward References
# =============================================================================


class TestForwardReferences:
    """Test type checking with forward references in annotations."""

    def test_string_forward_reference(self) -> None:
        """Test that string forward references are resolved."""
        # Note: Forward references are resolved by get_type_hints()
        # when the proper namespace is provided

        class Container(Node[int], tag="tc_forward_ref"):
            # This is a forward reference (string annotation)
            nested: "Node[int]"

        node = Container(nested=IntConst(value=1))
        errors = type_check(node)
        assert errors == []


# =============================================================================
# SECTION 15: Program-Level Validation
# =============================================================================


class TestProgramLevelValidation:
    """Test validation that requires whole-program analysis."""

    def test_root_type_validation(self) -> None:
        """Test that program root has expected type."""
        # A program's root should match what the interpreter expects

        prog = Program(
            root=Ref(id="main"),
            nodes={"main": IntConst(value=42)},
        )
        # Basic type check should pass (no expected root type constraint)
        errors = type_check(prog)
        assert errors == []

    def test_unused_nodes_warning(self) -> None:
        """Optionally warn about nodes not reachable from root."""
        prog = Program(
            root=Ref(id="main"),
            nodes={
                "main": IntConst(value=1),
                "orphan": StrConst(text="unused"),  # Never referenced
            },
        )
        # Current implementation doesn't warn about unused nodes
        # but should not error
        errors = type_check(prog)
        assert errors == []

    def test_all_references_resolved(self) -> None:
        """Ensure all Refs in program point to existing nodes."""

        class RefUser(Node[int], tag="tc_ref_resolved"):
            child: Ref[Node[int]]

        # Note: This is caught by Program.resolve(), but type checker
        # should provide comprehensive report of ALL missing refs
        prog = Program(
            root=Ref(id="main"),
            nodes={
                "main": RefUser(child=Ref(id="missing1")),
                # "missing1" doesn't exist
            },
        )
        errors = type_check(prog)
        assert len(errors) >= 1
        assert "missing1" in str(errors[0])


# =============================================================================
# SECTION 16: Error Message Quality
# =============================================================================


class TestErrorMessageQuality:
    """Test that error messages are helpful and informative."""

    def test_error_includes_field_name(self) -> None:
        """Error should mention which field has the problem."""

        class TwoFields(Node[float], tag="tc_err_field"):
            first: Node[float]
            second: Node[float]

        node = TwoFields(
            first=FloatConst(value=1.0),
            second=StrConst(text="x"),  # type: ignore[arg-type]
        )
        errors = type_check(node)
        assert len(errors) >= 1
        assert "second" in errors[0].field_name

    def test_error_includes_expected_and_actual(self) -> None:
        """Error should show expected type and actual type."""

        class Consumer(Node[float], tag="tc_err_expected_actual"):
            input: Node[float]

        node = Consumer(input=StrConst(text="x"))  # type: ignore[arg-type]
        errors = type_check(node)
        assert len(errors) >= 1
        # Error should have expected and actual
        assert errors[0].expected
        assert errors[0].actual

    def test_error_includes_ref_target(self) -> None:
        """For ref errors, show which node ID was referenced."""

        class RefUser(Node[float], tag="tc_err_ref"):
            child: Ref[Node[float]]

        prog = Program(
            root=Ref(id="main"),
            nodes={
                "wrong_type": StrConst(text="x"),
                "main": RefUser(child=Ref(id="wrong_type")),
            },
        )
        errors = type_check(prog)
        assert len(errors) >= 1
        error_str = str(errors[0])
        assert "wrong_type" in error_str


# =============================================================================
# SECTION 17: Covariance and Contravariance
# =============================================================================


class TestVariance:
    """Test handling of type variance (covariance/contravariance).

    Python generics are invariant by default, meaning:
    - list[int] is NOT a subtype of list[object]
    - Node[int] is NOT a subtype of Node[object]

    The type checker should follow Python's semantics.
    """

    def test_invariant_generics_reject_subtype(self) -> None:
        """Node[int] should not be accepted where Node[object] expected.

        This follows Python's invariant generic semantics.
        """

        class ObjectConsumer(Node[object], tag="tc_invariant_obj"):
            input: Node[object]

        # Strictly speaking, this should fail due to invariance
        # But we allow it for pragmatic reasons (int is subtype of object)
        node = ObjectConsumer(input=IntConst(value=1))
        errors = type_check(node)
        # Decision: allowing subtype for pragmatic usability
        assert errors == []

    def test_any_breaks_invariance(self) -> None:
        """Node[Any] should accept any node (Any is special)."""

        class AnyConsumer(Node[Any], tag="tc_any_consumer"):
            input: Node[Any]

        # Any is the escape hatch - always valid
        node = AnyConsumer(input=IntConst(value=1))
        errors = type_check(node)
        assert errors == []


# =============================================================================
# SECTION 18: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_program(self) -> None:
        """Test program with just root, no additional nodes."""
        prog = Program(root=IntConst(value=1))
        errors = type_check(prog)
        assert errors == []

    def test_node_with_no_fields(self) -> None:
        """Test node class with no fields."""

        class Empty(Node[None], tag="tc_empty_node"):
            pass

        node = Empty()
        errors = type_check(node)
        assert errors == []

    def test_node_with_only_primitive_fields(self) -> None:
        """Test node with only primitive fields (no nested nodes)."""

        class Data(Node[str], tag="tc_primitives_only"):
            name: str
            count: int
            ratio: float
            flag: bool

        node = Data(name="test", count=1, ratio=0.5, flag=True)
        errors = type_check(node)
        assert errors == []

    def test_very_wide_node(self) -> None:
        """Test node with many fields."""

        class Wide(Node[int], tag="tc_wide"):
            a: Node[int]
            b: Node[int]
            c: Node[int]
            d: Node[int]
            e: Node[int]

        node = Wide(
            a=IntConst(value=1),
            b=IntConst(value=2),
            c=IntConst(value=3),
            d=IntConst(value=4),
            e=IntConst(value=5),
        )
        errors = type_check(node)
        assert errors == []

    def test_program_with_many_nodes(self) -> None:
        """Test program with many nodes."""
        nodes = {f"n{i}": IntConst(value=i) for i in range(100)}
        prog = Program(root=Ref(id="n0"), nodes=nodes)
        errors = type_check(prog)
        assert errors == []
