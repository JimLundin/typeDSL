"""Tests for the type checker implementation.

These tests define the expected behavior of the type checker by providing
concrete examples of valid and invalid programs. The tests use pytest.skip()
since the type checker is not yet implemented, but they document all the
edge cases and scenarios the implementation must handle.

Run with: pytest tests/test_type_checker.py -v
"""

from collections.abc import Callable
from typing import Any, Literal

import pytest

from typedsl import Child, Node, Program, Ref

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

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_simple_return_type(self) -> None:
        """Should extract int from Node[int]."""
        # get_node_return_type(IntConst) should return int

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_none_return_type(self) -> None:
        """Should extract NoneType from Node[None]."""
        # get_node_return_type(NoneNode) should return type(None)

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_any_return_type(self) -> None:
        """Should extract Any from Node[Any]."""
        # get_node_return_type(AnyNode) should return Any

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_complex_return_type(self) -> None:
        """Should extract complex types like list[int]."""

        class ListNode(Node[list[int]], tag="tc_list_ret"):
            items: list[int]

        # get_node_return_type(ListNode) should return list[int]

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_nested_generic_return_type(self) -> None:
        """Should extract nested generics like dict[str, list[int]]."""

        class DictNode(Node[dict[str, list[int]]], tag="tc_dict_ret"):
            mapping: dict[str, list[int]]

        # get_node_return_type(DictNode) should return dict[str, list[int]]


# =============================================================================
# SECTION 2: Direct Field Type Validation
# =============================================================================


class TestDirectFieldValidation:
    """Test type checking of directly nested nodes (not via references)."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_valid_direct_nesting(self) -> None:
        """Valid: Node[float] field receives a node returning float."""

        class Consumer(Node[float], tag="tc_consumer_valid"):
            input: Node[float]

        # This should pass type checking
        Consumer(input=FloatConst(value=1.0))
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_invalid_direct_nesting_wrong_type(self) -> None:
        """Invalid: Node[float] field receives a node returning str."""

        class Consumer(Node[float], tag="tc_consumer_invalid"):
            input: Node[float]

        # This should fail type checking
        Consumer(input=StrConst(text="hello"))  # type: ignore[arg-type]
        # type_check(node) should return error:
        # "Consumer.input: expected Node[float], got Node[str]"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_invalid_nesting_int_where_float_expected(self) -> None:
        """Invalid: Node[float] field receives Node[int].

        Python generics are invariant - int is not float.
        """

        class Consumer(Node[float], tag="tc_consumer_int_float"):
            input: Node[float]

        Consumer(input=IntConst(value=1))  # type: ignore[arg-type]
        # Should fail: int is not float (invariant generics)

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_multiple_fields_validation(self) -> None:
        """Test validating multiple node fields."""

        class BinOp(Node[float], tag="tc_binop_multi"):
            left: Node[float]
            right: Node[float]

        # Valid
        BinOp(left=FloatConst(value=1.0), right=FloatConst(value=2.0))
        # type_check(valid) should return []

        # Invalid - first field wrong
        BinOp(
            left=StrConst(text="x"),  # type: ignore[arg-type]
            right=FloatConst(value=2.0),
        )
        # type_check(invalid1) should return 1 error for 'left'

        # Invalid - both fields wrong
        BinOp(
            left=StrConst(text="x"),  # type: ignore[arg-type]
            right=IntConst(value=1),  # type: ignore[arg-type]
        )
        # type_check(invalid2) should return 2 errors


# =============================================================================
# SECTION 3: Reference Type Validation
# =============================================================================


class TestReferenceTypeValidation:
    """Test type checking of Ref[Node[T]] in Programs."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_valid_reference(self) -> None:
        """Valid: Ref[Node[float]] points to a node returning float."""

        class RefConsumer(Node[float], tag="tc_ref_consumer_valid"):
            input: Ref[Node[float]]

        Program(
            root=Ref(id="consumer"),
            nodes={
                "value": FloatConst(value=1.0),
                "consumer": RefConsumer(input=Ref(id="value")),
            },
        )
        # type_check(prog) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_invalid_reference_wrong_type(self) -> None:
        """Invalid: Ref[Node[float]] points to a node returning str."""

        class RefConsumer(Node[float], tag="tc_ref_consumer_invalid"):
            input: Ref[Node[float]]

        Program(
            root=Ref(id="consumer"),
            nodes={
                "value": StrConst(text="hello"),  # Returns str, not float
                "consumer": RefConsumer(input=Ref(id="value")),
            },
        )
        # type_check(prog) should return error:
        # "RefConsumer.input: Ref[Node[float]] references 'value' returning str"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_chained_references(self) -> None:
        """Test validation through chains of references."""

        class Wrapper(Node[float], tag="tc_wrapper_chain"):
            inner: Ref[Node[float]]

        # Valid chain: root -> wrapper -> value
        Program(
            root=Ref(id="outer"),
            nodes={
                "value": FloatConst(value=1.0),
                "inner": Wrapper(inner=Ref(id="value")),
                "outer": Wrapper(inner=Ref(id="inner")),
            },
        )
        # type_check(valid_prog) should return []

        # Invalid: one link in chain is wrong type
        Program(
            root=Ref(id="outer"),
            nodes={
                "value": StrConst(text="x"),  # Wrong type!
                "inner": Wrapper(inner=Ref(id="value")),
                "outer": Wrapper(inner=Ref(id="inner")),
            },
        )
        # type_check(invalid_prog) should catch the error

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_reference_to_nonexistent_node(self) -> None:
        """References to non-existent nodes should be reported."""

        class RefUser(Node[int], tag="tc_ref_nonexistent"):
            target: Ref[Node[int]]

        Program(
            root=Ref(id="user"),
            nodes={
                "user": RefUser(target=Ref(id="missing")),  # 'missing' doesn't exist
            },
        )
        # type_check(prog) should report missing reference
        # (This is already caught by Program.resolve(), but type checker should too)


# =============================================================================
# SECTION 4: Child[T] Type Validation
# =============================================================================


class TestChildTypeValidation:
    """Test type checking of Child[T] = Node[T] | Ref[Node[T]]."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_child_with_inline_node_valid(self) -> None:
        """Valid: Child[float] with inline Node[float]."""

        class ChildUser(Node[float], tag="tc_child_inline_valid"):
            input: Child[float]

        ChildUser(input=FloatConst(value=1.0))
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_child_with_inline_node_invalid(self) -> None:
        """Invalid: Child[float] with inline Node[str]."""

        class ChildUser(Node[float], tag="tc_child_inline_invalid"):
            input: Child[float]

        ChildUser(input=StrConst(text="x"))  # type: ignore[arg-type]
        # type_check(node) should return error

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_child_with_reference_valid(self) -> None:
        """Valid: Child[float] with Ref[Node[float]]."""

        class ChildUser(Node[float], tag="tc_child_ref_valid"):
            input: Child[float]

        Program(
            root=Ref(id="user"),
            nodes={
                "value": FloatConst(value=1.0),
                "user": ChildUser(input=Ref(id="value")),
            },
        )
        # type_check(prog) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_child_with_reference_invalid(self) -> None:
        """Invalid: Child[float] with Ref to Node[str]."""

        class ChildUser(Node[float], tag="tc_child_ref_invalid"):
            input: Child[float]

        Program(
            root=Ref(id="user"),
            nodes={
                "value": StrConst(text="x"),  # Wrong type
                "user": ChildUser(input=Ref(id="value")),
            },
        )
        # type_check(prog) should return error

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_child_mixed_usage(self) -> None:
        """Test node using both inline and ref for Child[T] fields."""

        class BinaryOp(Node[float], tag="tc_binop_child_mixed"):
            left: Child[float]
            right: Child[float]

        # Mixed: inline left, ref right
        Program(
            root=Ref(id="op"),
            nodes={
                "value": FloatConst(value=2.0),
                "op": BinaryOp(
                    left=FloatConst(value=1.0),  # Inline
                    right=Ref(id="value"),  # Reference
                ),
            },
        )
        # type_check(prog) should return []


# =============================================================================
# SECTION 5: Union Type Validation
# =============================================================================


class TestUnionTypeValidation:
    """Test type checking with union types."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_union_node_type_first_variant(self) -> None:
        """Valid: Union field accepts first variant."""

        class UnionUser(Node[str], tag="tc_union_first"):
            data: Node[int] | Node[str]

        UnionUser(data=IntConst(value=1))
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_union_node_type_second_variant(self) -> None:
        """Valid: Union field accepts second variant."""

        class UnionUser(Node[str], tag="tc_union_second"):
            data: Node[int] | Node[str]

        UnionUser(data=StrConst(text="hello"))
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_union_node_type_neither_variant(self) -> None:
        """Invalid: Union field receives neither variant."""

        class UnionUser(Node[str], tag="tc_union_neither"):
            data: Node[int] | Node[str]

        UnionUser(data=FloatConst(value=1.0))  # type: ignore[arg-type]
        # type_check(node) should return error:
        # "expected Node[int] | Node[str], got Node[float]"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_optional_node_with_none(self) -> None:
        """Valid: Optional node field with None."""

        class OptionalUser(Node[int], tag="tc_optional_none"):
            child: Node[int] | None

        OptionalUser(child=None)
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_optional_node_with_value(self) -> None:
        """Valid: Optional node field with actual node."""

        class OptionalUser(Node[int], tag="tc_optional_value"):
            child: Node[int] | None

        OptionalUser(child=IntConst(value=42))
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_optional_node_with_wrong_type(self) -> None:
        """Invalid: Optional Node[int] with Node[str]."""

        class OptionalUser(Node[int], tag="tc_optional_wrong"):
            child: Node[int] | None

        OptionalUser(child=StrConst(text="x"))  # type: ignore[arg-type]
        # type_check(node) should return error


# =============================================================================
# SECTION 6: Literal Type Validation
# =============================================================================


class TestLiteralTypeValidation:
    """Test type checking of Literal types."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_literal_valid_value(self) -> None:
        """Valid: Literal field has value from allowed set."""

        class Op(Node[float], tag="tc_literal_valid"):
            operator: Literal["+", "-", "*", "/"]
            left: Node[float]
            right: Node[float]

        Op(
            operator="+",
            left=FloatConst(value=1.0),
            right=FloatConst(value=2.0),
        )
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_literal_invalid_value(self) -> None:
        """Invalid: Literal field has value not in allowed set."""

        class Op(Node[float], tag="tc_literal_invalid"):
            operator: Literal["+", "-", "*", "/"]
            left: Node[float]
            right: Node[float]

        Op(
            operator="%",  # type: ignore[arg-type]
            left=FloatConst(value=1.0),
            right=FloatConst(value=2.0),
        )
        # type_check(node) should return error:
        # "Op.operator: '%' not in Literal['+', '-', '*', '/']"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_literal_int_values(self) -> None:
        """Test Literal with int values."""

        class Priority(Node[str], tag="tc_literal_int"):
            level: Literal[1, 2, 3]
            name: str

        Priority(level=2, name="medium")
        # type_check(valid) should return []

        Priority(level=5, name="ultra")  # type: ignore[arg-type]
        # type_check(invalid) should return error

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_literal_bool_values(self) -> None:
        """Test Literal with bool values."""

        class Flag(Node[str], tag="tc_literal_bool"):
            enabled: Literal[True, False]

        Flag(enabled=True)
        # type_check(valid) should return []


# =============================================================================
# SECTION 7: Generic Node Validation
# =============================================================================


class TestGenericNodeValidation:
    """Test type checking of generic nodes with type parameters."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_node_consistent_type_param(self) -> None:
        """Valid: Generic node with consistent type parameter."""

        class Pair[T](Node[tuple[T, T]], tag="tc_pair_consistent"):
            first: Node[T]
            second: Node[T]

        # Both return float, so T=float is consistent
        Pair(
            first=FloatConst(value=1.0),
            second=FloatConst(value=2.0),
        )
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_node_inconsistent_type_param(self) -> None:
        """Invalid: Generic node with inconsistent type parameter."""

        class Pair[T](Node[tuple[T, T]], tag="tc_pair_inconsistent"):
            first: Node[T]
            second: Node[T]

        # first=float, second=str - T cannot be both!
        Pair(
            first=FloatConst(value=1.0),
            second=StrConst(text="x"),  # type: ignore[arg-type]
        )
        # type_check(node) should return error:
        # "Pair: type parameter T is inconsistent (float vs str)"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_node_multiple_type_params(self) -> None:
        """Test generic node with multiple type parameters."""

        class Transform[T, U](Node[U], tag="tc_transform_multi"):
            input: Node[T]
            mapper: Node[U]

        # T=int, U=str
        Transform(
            input=IntConst(value=1),
            mapper=StrConst(text="result"),
        )
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_node_with_bound(self) -> None:
        """Test generic node with bounded type parameter."""

        class Numeric[T: int | float](Node[T], tag="tc_numeric_bound"):
            value: T

        # Valid: int is within bound
        Numeric(value=1)
        # type_check(valid_int) should return []

        # Valid: float is within bound
        Numeric(value=1.5)
        # type_check(valid_float) should return []

        # Note: Checking if str violates the bound requires runtime value inspection
        # which may be beyond pure type checking scope

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_node_nested_in_container(self) -> None:
        """Test generic node used in container types."""

        class ListOf[T](Node[list[T]], tag="tc_list_generic"):
            items: list[Node[T]]

        # All items return int, so T=int is consistent
        ListOf(
            items=[IntConst(value=1), IntConst(value=2), IntConst(value=3)],
        )
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_node_inconsistent_in_list(self) -> None:
        """Invalid: Generic list with inconsistent element types."""

        class ListOf[T](Node[list[T]], tag="tc_list_inconsistent"):
            items: list[Node[T]]

        ListOf(
            items=[
                IntConst(value=1),
                StrConst(text="x"),  # type: ignore[list-item]
            ],
        )
        # type_check(node) should return error:
        # "ListOf: type parameter T is inconsistent in items"


# =============================================================================
# SECTION 7b: Type Parameter Bounds (PEP 695)
# =============================================================================


class TestTypeParameterBounds:
    """Test type checking with bounded type parameters (PEP 695 syntax).

    PEP 695 introduced the new generic syntax:
        class Foo[T: Bound]: ...

    The type checker must validate that inferred types satisfy bounds.
    """

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_satisfied_exactly(self) -> None:
        """Type parameter bound is satisfied by exact match."""

        class NumericOp[T: int | float](Node[T], tag="tc_bound_exact"):
            left: Node[T]
            right: Node[T]

        # T=int satisfies T: int | float
        NumericOp(left=IntConst(value=1), right=IntConst(value=2))
        # type_check should return []

        # T=float satisfies T: int | float
        NumericOp(left=FloatConst(value=1.0), right=FloatConst(value=2.0))
        # type_check should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_violated(self) -> None:
        """Type parameter bound is violated."""

        class NumericOp[T: int | float](Node[T], tag="tc_bound_violated"):
            left: Node[T]
            right: Node[T]

        # T=str does NOT satisfy T: int | float
        NumericOp(
            left=StrConst(text="a"),  # type: ignore[arg-type]
            right=StrConst(text="b"),  # type: ignore[arg-type]
        )
        # type_check should return error:
        # "NumericOp: type parameter T bound violation - str is not int | float"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_with_single_type(self) -> None:
        """Type parameter bound with single type constraint."""

        class StringProcessor[T: str](Node[T], tag="tc_bound_single"):
            input: Node[T]

        # T=str satisfies T: str
        StringProcessor(input=StrConst(text="hello"))
        # type_check should return []

        # T=int violates T: str
        StringProcessor(input=IntConst(value=1))  # type: ignore[arg-type]
        # type_check should return error

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_with_none_in_union(self) -> None:
        """Type parameter bound that includes None."""

        class OptionalValue[T: int | None](Node[T], tag="tc_bound_none"):
            value: Node[T]

        # Both should satisfy the bound
        OptionalValue(value=IntConst(value=42))
        OptionalValue(value=NoneNode())
        # type_check should return [] for both

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_multiple_bounded_type_params(self) -> None:
        """Multiple type parameters each with different bounds."""

        class Converter[T: int | float, U: str | bytes](Node[U], tag="tc_multi_bound"):
            input: Node[T]
            format: Node[U]

        # T=int satisfies int|float, U=str satisfies str|bytes
        Converter(input=IntConst(value=1), format=StrConst(text="decimal"))
        # type_check should return []

        # T=str violates int|float bound
        Converter(
            input=StrConst(text="x"),  # type: ignore[arg-type]
            format=StrConst(text="y"),
        )
        # type_check should return error for T bound

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_with_subtype(self) -> None:
        """Test whether subtypes satisfy bounds (bool is subtype of int)."""

        class IntProcessor[T: int](Node[T], tag="tc_bound_subtype"):
            value: Node[T]

        # In Python, bool is a subtype of int
        # Should this satisfy T: int?
        IntProcessor(value=BoolConst(value=True))
        # Decision point: strict bound matching or allow subtypes?

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_unbounded_type_param(self) -> None:
        """Unbounded type parameter accepts any type."""

        class Identity[T](Node[T], tag="tc_unbounded"):
            value: Node[T]

        # All of these should be valid (T is unbounded)
        Identity(value=IntConst(value=1))
        Identity(value=StrConst(text="x"))
        Identity(value=FloatConst(value=1.0))
        # type_check should return [] for all

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_inference_from_multiple_fields(self) -> None:
        """Infer T from multiple fields, then check bound."""

        class BoundedPair[T: int | float](Node[tuple[T, T]], tag="tc_bound_infer"):
            first: Node[T]
            second: Node[T]

        # Infer T=int from first, validate second matches, check bound
        BoundedPair(first=IntConst(value=1), second=IntConst(value=2))
        # type_check should return []

        # Infer T=str from first - but str violates bound!
        BoundedPair(
            first=StrConst(text="a"),  # type: ignore[arg-type]
            second=StrConst(text="b"),  # type: ignore[arg-type]
        )
        # type_check should return bound violation error

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_with_container_type(self) -> None:
        """Type parameter with bound used in container context."""

        class NumericList[T: int | float](Node[list[T]], tag="tc_bound_container"):
            items: list[Node[T]]

        # All items return int, T=int satisfies bound
        NumericList(items=[IntConst(value=1), IntConst(value=2)])
        # type_check should return []

        # All items return str, T=str violates bound
        NumericList(
            items=[
                StrConst(text="a"),  # type: ignore[list-item]
                StrConst(text="b"),  # type: ignore[list-item]
            ],
        )
        # type_check should return bound violation error


# =============================================================================
# SECTION 7c: Type Parameter Inference
# =============================================================================


class TestTypeParameterInference:
    """Test inference of type parameters from field values.

    When a generic node is instantiated, the type checker must infer
    the concrete type for each type parameter from the provided values.
    """

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_infer_from_single_field(self) -> None:
        """Infer T from a single field."""

        class Wrap[T](Node[T], tag="tc_infer_single"):
            inner: Node[T]

        # T should be inferred as float from the field
        Wrap(inner=FloatConst(value=1.0))
        # After inference: T=float
        # type_check should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_infer_from_first_field_validate_rest(self) -> None:
        """Infer T from first field, validate remaining fields match."""

        class Triple[T](Node[tuple[T, T, T]], tag="tc_infer_validate"):
            a: Node[T]
            b: Node[T]
            c: Node[T]

        # Infer T=int from 'a', validate b and c match
        Triple(a=IntConst(value=1), b=IntConst(value=2), c=IntConst(value=3))
        # type_check should return []

        # Infer T=int from 'a', but 'c' is str - mismatch!
        Triple(
            a=IntConst(value=1),
            b=IntConst(value=2),
            c=StrConst(text="x"),  # type: ignore[arg-type]
        )
        # type_check should return error: T inferred as int but c returns str

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_infer_independent_type_params(self) -> None:
        """Multiple type parameters inferred independently."""

        class Map[T, U](Node[U], tag="tc_infer_multi"):
            input: Node[T]
            output: Node[U]

        # Infer T=int from input, U=str from output
        Map(input=IntConst(value=1), output=StrConst(text="result"))
        # type_check should return [] - T and U are independent

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_infer_from_container_elements(self) -> None:
        """Infer T from elements in a container field."""

        class Aggregate[T](Node[T], tag="tc_infer_container"):
            values: list[Node[T]]

        # Infer T=int from list elements
        Aggregate(values=[IntConst(value=1), IntConst(value=2)])
        # type_check should return []

        # Inconsistent elements - can't unify
        Aggregate(
            values=[
                IntConst(value=1),
                StrConst(text="x"),  # type: ignore[list-item]
            ],
        )
        # type_check should return error: cannot infer single T

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_infer_with_empty_container(self) -> None:
        """Empty container provides no information for inference."""

        class Aggregate[T](Node[list[T]], tag="tc_infer_empty"):
            values: list[Node[T]]

        # Empty list - T cannot be inferred from values
        Aggregate(values=[])
        # Options: error (can't infer), or defer checking
        # type_check behavior TBD

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_infer_from_nested_generic(self) -> None:
        """Infer T from nested generic position."""

        class Processor[T](Node[list[T]], tag="tc_infer_nested"):
            mapping: dict[str, Node[T]]

        # Infer T=float from dict values
        Processor(
            mapping={
                "a": FloatConst(value=1.0),
                "b": FloatConst(value=2.0),
            },
        )
        # type_check should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_type_param_used_only_in_return(self) -> None:
        """Type parameter appears only in return type, not fields.

        This is an interesting edge case - how do we infer T?
        """

        class Phantom[T](Node[T], tag="tc_phantom"):
            # T only appears in Node[T] return type, not in fields
            label: str

        # T is unconstrained by fields
        # type_check behavior TBD - error or allow?

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_infer_from_ref_target(self) -> None:
        """Infer T from the type of referenced node in a Program."""

        class Wrapper[T](Node[T], tag="tc_infer_ref"):
            inner: Ref[Node[T]]

        Program(
            root=Ref(id="wrap"),
            nodes={
                "value": FloatConst(value=1.0),  # Returns float
                "wrap": Wrapper(inner=Ref(id="value")),  # T should be inferred as float
            },
        )
        # type_check(prog) should infer T=float and return []


# =============================================================================
# SECTION 7d: Complex Type Parameter Scenarios
# =============================================================================


class TestComplexTypeParameters:
    """Test complex scenarios involving type parameters."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_type_param_in_union_field(self) -> None:
        """Type parameter used within a union type field."""

        class MaybeProcess[T](Node[T | None], tag="tc_param_union"):
            input: Node[T] | None

        # T=int inferred, None is also valid
        MaybeProcess(input=IntConst(value=1))
        MaybeProcess(input=None)
        # type_check should return [] for both

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_type_param_in_nested_container(self) -> None:
        """Type parameter in deeply nested container type."""

        class DeepNest[T](Node[list[list[T]]], tag="tc_deep_nest"):
            data: list[list[Node[T]]]

        DeepNest(
            data=[
                [IntConst(value=1), IntConst(value=2)],
                [IntConst(value=3), IntConst(value=4)],
            ],
        )
        # type_check should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_type_param_in_dict_key_and_value(self) -> None:
        """Type parameter used in both dict key and value positions.

        Note: dict keys must be hashable, so T: Hashable might be implicit.
        """

        class BiMap[T](Node[dict[T, T]], tag="tc_bimap"):
            forward: dict[str, Node[T]]
            backward: dict[str, Node[T]]

        BiMap(
            forward={"a": IntConst(value=1)},
            backward={"x": IntConst(value=2)},
        )
        # type_check should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_same_param_different_container_types(self) -> None:
        """Same type parameter used in different container types."""

        class MultiContainer[T](Node[T], tag="tc_multi_container"):
            as_list: list[Node[T]]
            as_set: set[str]  # Not using T here
            as_dict_value: dict[str, Node[T]]

        MultiContainer(
            as_list=[FloatConst(value=1.0)],
            as_set={"key"},
            as_dict_value={"a": FloatConst(value=2.0)},
        )
        # type_check should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_node_field_with_concrete_type(self) -> None:
        """Generic node has one field with concrete type, one with T."""

        class Hybrid[T](Node[T], tag="tc_hybrid"):
            typed: Node[T]  # Uses type parameter
            fixed: Node[int]  # Always int

        # T inferred as str, fixed must be int
        Hybrid(typed=StrConst(text="x"), fixed=IntConst(value=1))
        # type_check should return []

        # T inferred as str, but wrong type for fixed
        Hybrid(
            typed=StrConst(text="x"),
            fixed=StrConst(text="y"),  # type: ignore[arg-type]
        )
        # type_check should return error for 'fixed' field

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_bound_references_another_param(self) -> None:
        """Advanced: one type param's bound references another.

        Note: This is more advanced and may not be supported initially.
        """
        # class Related[T, U: T](Node[U]): ...
        # This would mean U must be a subtype of T
        # Complex to implement - may skip initially

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_recursive_generic_node(self) -> None:
        """Generic node that can contain nodes of same generic type."""

        class Tree[T](Node[T], tag="tc_recursive_generic"):
            value: Node[T]
            children: list[Child[T]]  # Recursive reference

        # Build a simple tree with T=int
        Program(
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
        # type_check(prog) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_generic_with_child_type(self) -> None:
        """Generic node using Child[T] (union of Node[T] | Ref[Node[T]])."""

        class Flexible[T](Node[T], tag="tc_generic_child"):
            input: Child[T]

        # Inline node
        Flexible(input=FloatConst(value=1.0))

        # Via Program with ref
        Program(
            root=Ref(id="flex"),
            nodes={
                "val": FloatConst(value=2.0),
                "flex": Flexible(input=Ref(id="val")),
            },
        )
        # type_check for both should return []


# =============================================================================
# SECTION 7e: Type Parameter Error Messages
# =============================================================================


class TestTypeParameterErrors:
    """Test error message quality for type parameter issues."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_error_shows_inferred_type(self) -> None:
        """Error message should show what T was inferred as."""

        class Pair[T](Node[tuple[T, T]], tag="tc_err_inferred"):
            first: Node[T]
            second: Node[T]

        Pair(
            first=IntConst(value=1),
            second=StrConst(text="x"),  # type: ignore[arg-type]
        )
        # Error should say: "T inferred as int from 'first', but 'second' returns str"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_error_shows_bound(self) -> None:
        """Error message should show the violated bound."""

        class Numeric[T: int | float](Node[T], tag="tc_err_bound"):
            value: Node[T]

        Numeric(value=StrConst(text="x"))  # type: ignore[arg-type]
        # Error should say: "T bound violation: str does not satisfy int | float"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_error_for_multiple_type_params(self) -> None:
        """Error identifies which type parameter has the issue."""

        class TwoParams[T, U](Node[tuple[T, U]], tag="tc_err_which_param"):
            t_val: Node[T]
            u_val: Node[U]
            t_val2: Node[T]

        TwoParams(
            t_val=IntConst(value=1),
            u_val=StrConst(text="x"),
            t_val2=FloatConst(value=2.0),  # type: ignore[arg-type]
        )
        # Error should identify T as inconsistent (int vs float), not U


# =============================================================================
# SECTION 8: Any Type Handling
# =============================================================================


class TestAnyTypeHandling:
    """Test that Any acts as a universal escape hatch."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_any_accepts_any_node(self) -> None:
        """Node[Any] should accept any node type."""

        class Wrapper(Node[Any], tag="tc_any_wrapper"):
            child: Node[Any]

        # All of these should be valid
        Wrapper(child=IntConst(value=1))
        Wrapper(child=StrConst(text="x"))
        Wrapper(child=FloatConst(value=1.0))

        # type_check should return [] for all

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_any_field_in_typed_node(self) -> None:
        """Node with specific return type but Any field."""

        class Processor(Node[str], tag="tc_any_field"):
            input: Node[Any]
            output: str

        # Any input should be accepted
        Processor(input=IntConst(value=42), output="result")
        # type_check(node) should return []


# =============================================================================
# SECTION 9: Container Types with Nodes
# =============================================================================


class TestContainerTypesWithNodes:
    """Test type checking of containers (list, dict) containing nodes."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_list_of_nodes_valid(self) -> None:
        """Valid: list[Node[int]] contains all Node[int]."""

        class Aggregator(Node[int], tag="tc_list_nodes_valid"):
            inputs: list[Node[int]]

        Aggregator(
            inputs=[IntConst(value=1), IntConst(value=2), IntConst(value=3)],
        )
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_list_of_nodes_invalid(self) -> None:
        """Invalid: list[Node[int]] contains Node[str]."""

        class Aggregator(Node[int], tag="tc_list_nodes_invalid"):
            inputs: list[Node[int]]

        Aggregator(
            inputs=[
                IntConst(value=1),
                StrConst(text="x"),  # type: ignore[list-item]
            ],
        )
        # type_check(node) should return error for the str item

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_list_of_child_type(self) -> None:
        """Test list[Child[T]] with mixed inline and refs."""

        class MultiInput(Node[float], tag="tc_list_child"):
            inputs: list[Child[float]]

        Program(
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
        # type_check(prog) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_dict_of_nodes_valid(self) -> None:
        """Valid: dict[str, Node[int]] with all Node[int] values."""

        class NamedInputs(Node[int], tag="tc_dict_nodes_valid"):
            inputs: dict[str, Node[int]]

        NamedInputs(
            inputs={
                "a": IntConst(value=1),
                "b": IntConst(value=2),
            },
        )
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_dict_of_nodes_invalid(self) -> None:
        """Invalid: dict[str, Node[int]] with Node[str] value."""

        class NamedInputs(Node[int], tag="tc_dict_nodes_invalid"):
            inputs: dict[str, Node[int]]

        NamedInputs(
            inputs={
                "a": IntConst(value=1),
                "b": StrConst(text="x"),  # type: ignore[dict-item]
            },
        )
        # type_check(node) should return error for key "b"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_empty_list_valid(self) -> None:
        """Empty list should be valid for any list[Node[T]]."""

        class Aggregator(Node[int], tag="tc_empty_list"):
            inputs: list[Node[int]]

        Aggregator(inputs=[])
        # type_check(node) should return []


# =============================================================================
# SECTION 10: Cyclic Reference Handling
# =============================================================================


class TestCyclicReferences:
    """Test that type checker handles cyclic references without infinite loops."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_simple_cycle(self) -> None:
        """Test simple A -> B -> A cycle."""

        class CycleNode(Node[int], tag="tc_cycle_simple"):
            value: int
            next: Ref[Node[int]] | None

        Program(
            root=Ref(id="a"),
            nodes={
                "a": CycleNode(value=1, next=Ref(id="b")),
                "b": CycleNode(value=2, next=Ref(id="a")),  # Back to a
            },
        )
        # type_check(prog) should complete without infinite loop
        # and return [] (valid types)

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_self_reference(self) -> None:
        """Test node referencing itself."""

        class SelfRef(Node[int], tag="tc_self_ref"):
            value: int
            self_ref: Ref[Node[int]] | None

        Program(
            root=Ref(id="self"),
            nodes={
                "self": SelfRef(value=1, self_ref=Ref(id="self")),
            },
        )
        # type_check(prog) should complete and return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_cycle_with_type_error(self) -> None:
        """Test cycle where one node has wrong type."""

        class TypedCycle(Node[int], tag="tc_cycle_error"):
            value: int
            next: Ref[Node[int]] | None

        # Note: This actually can't be expressed since nodes dict
        # only accepts Node[Any], but demonstrates the concept
        # that cycles shouldn't prevent error detection


# =============================================================================
# SECTION 11: Deep Nesting Validation
# =============================================================================


class TestDeepNesting:
    """Test type checking of deeply nested structures."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_deep_inline_nesting(self) -> None:
        """Test deeply nested inline nodes."""

        class Unary(Node[int], tag="tc_deep_unary"):
            child: Node[int]

        # 5 levels deep
        Unary(
            child=Unary(
                child=Unary(
                    child=Unary(child=Unary(child=IntConst(value=1))),
                ),
            ),
        )
        # type_check(deep) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_deep_nesting_error_at_leaf(self) -> None:
        """Test that errors at deeply nested leaves are caught."""

        class Unary(Node[int], tag="tc_deep_error"):
            child: Node[int]

        Unary(
            child=Unary(
                child=Unary(
                    child=StrConst(text="x"),  # type: ignore[arg-type]
                ),
            ),
        )
        # type_check(deep) should return error with path information


# =============================================================================
# SECTION 12: Return Type in Node Hierarchy
# =============================================================================


class TestNodeHierarchy:
    """Test type checking with node inheritance hierarchies."""

    @pytest.mark.skip(reason="Type checker not implemented")
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
        Add(
            left=Const(value=1.0),
            right=Const(value=2.0),
        )
        # type_check(node) should return []


# =============================================================================
# SECTION 13: Special Python Types
# =============================================================================


class TestSpecialPythonTypes:
    """Test type checking with special Python types."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_tuple_return_type(self) -> None:
        """Test node with tuple return type."""

        class TupleNode(Node[tuple[int, str]], tag="tc_tuple_ret"):
            num: int
            text: str

        class Consumer(Node[str], tag="tc_tuple_consumer"):
            input: Node[tuple[int, str]]

        Consumer(input=TupleNode(num=1, text="x"))
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_callable_in_annotation(self) -> None:
        """Test that Callable types in annotations are handled."""

        class FuncHolder(Node[int], tag="tc_callable"):
            func: Callable[[int], int]  # Not a Node, just a function
            input: int

        # This tests that non-Node types don't cause crashes


# =============================================================================
# SECTION 14: Forward References
# =============================================================================


class TestForwardReferences:
    """Test type checking with forward references in annotations."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_string_forward_reference(self) -> None:
        """Test that string forward references are resolved."""
        # Note: Forward references are resolved by get_type_hints()
        # when the proper namespace is provided

        class Container(Node[int], tag="tc_forward_ref"):
            # This is a forward reference (string annotation)
            nested: "Node[int]"

        Container(nested=IntConst(value=1))
        # type_check(node) should properly resolve "Node[int]" and validate


# =============================================================================
# SECTION 15: Program-Level Validation
# =============================================================================


class TestProgramLevelValidation:
    """Test validation that requires whole-program analysis."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_root_type_validation(self) -> None:
        """Test that program root has expected type."""
        # A program's root should match what the interpreter expects

        Program(
            root=Ref(id="main"),
            nodes={"main": IntConst(value=42)},
        )
        # type_check(prog, expected_root_type=int) should return []
        # type_check(prog, expected_root_type=str) should return error

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_unused_nodes_warning(self) -> None:
        """Optionally warn about nodes not reachable from root."""
        Program(
            root=Ref(id="main"),
            nodes={
                "main": IntConst(value=1),
                "orphan": StrConst(text="unused"),  # Never referenced
            },
        )
        # type_check(prog, warn_unused=True) might return warning

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_all_references_resolved(self) -> None:
        """Ensure all Refs in program point to existing nodes."""

        class RefUser(Node[int], tag="tc_ref_resolved"):
            child: Ref[Node[int]]

        # Note: This is caught by Program.resolve(), but type checker
        # should provide comprehensive report of ALL missing refs
        Program(
            root=Ref(id="main"),
            nodes={
                "main": RefUser(child=Ref(id="missing1")),
                # "missing1" doesn't exist
            },
        )
        # type_check(prog) should report missing "missing1"


# =============================================================================
# SECTION 16: Error Message Quality
# =============================================================================


class TestErrorMessageQuality:
    """Test that error messages are helpful and informative."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_error_includes_field_name(self) -> None:
        """Error should mention which field has the problem."""

        class TwoFields(Node[float], tag="tc_err_field"):
            first: Node[float]
            second: Node[float]

        TwoFields(
            first=FloatConst(value=1.0),
            second=StrConst(text="x"),  # type: ignore[arg-type]
        )
        # Error should mention "second" field specifically

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_error_includes_expected_and_actual(self) -> None:
        """Error should show expected type and actual type."""
        # Error format: "Field.name: expected Node[float], got Node[str]"

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_error_includes_ref_target(self) -> None:
        """For ref errors, show which node ID was referenced."""

        class RefUser(Node[float], tag="tc_err_ref"):
            child: Ref[Node[float]]

        Program(
            root=Ref(id="main"),
            nodes={
                "wrong_type": StrConst(text="x"),
                "main": RefUser(child=Ref(id="wrong_type")),
            },
        )
        # Error should mention: "references node 'wrong_type' which returns str"


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

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_invariant_generics_reject_subtype(self) -> None:
        """Node[int] should not be accepted where Node[object] expected.

        This follows Python's invariant generic semantics.
        """

        class ObjectConsumer(Node[object], tag="tc_invariant_obj"):
            input: Node[object]

        # Strictly speaking, this should fail due to invariance
        # But we may choose to allow it for pragmatic reasons
        ObjectConsumer(input=IntConst(value=1))

        # Decision point: strict (error) or pragmatic (allow)?

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_any_breaks_invariance(self) -> None:
        """Node[Any] should accept any node (Any is special)."""

        class AnyConsumer(Node[Any], tag="tc_any_consumer"):
            input: Node[Any]

        # Any is the escape hatch - always valid
        AnyConsumer(input=IntConst(value=1))
        # type_check(node) should return []


# =============================================================================
# SECTION 18: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_empty_program(self) -> None:
        """Test program with just root, no additional nodes."""
        Program(root=IntConst(value=1))
        # type_check(prog) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_node_with_no_fields(self) -> None:
        """Test node class with no fields."""

        class Empty(Node[None], tag="tc_empty_node"):
            pass

        Empty()
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_node_with_only_primitive_fields(self) -> None:
        """Test node with only primitive fields (no nested nodes)."""

        class Data(Node[str], tag="tc_primitives_only"):
            name: str
            count: int
            ratio: float
            flag: bool

        Data(name="test", count=1, ratio=0.5, flag=True)
        # type_check(node) should return [] (nothing to check)

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_very_wide_node(self) -> None:
        """Test node with many fields."""

        class Wide(Node[int], tag="tc_wide"):
            a: Node[int]
            b: Node[int]
            c: Node[int]
            d: Node[int]
            e: Node[int]

        Wide(
            a=IntConst(value=1),
            b=IntConst(value=2),
            c=IntConst(value=3),
            d=IntConst(value=4),
            e=IntConst(value=5),
        )
        # type_check(node) should return []

    @pytest.mark.skip(reason="Type checker not implemented")
    def test_program_with_many_nodes(self) -> None:
        """Test program with many nodes."""
        nodes = {f"n{i}": IntConst(value=i) for i in range(100)}
        Program(root=Ref(id="n0"), nodes=nodes)
        # type_check(prog) should complete in reasonable time
