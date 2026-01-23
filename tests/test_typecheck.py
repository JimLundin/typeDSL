"""Tests for typedsl.typecheck module."""

import datetime
from decimal import Decimal
from typing import Literal, TypeVar

from typedsl import (
    Child,
    Node,
    Program,
    Ref,
    TypeChecker,
    TypeCheckResult,
    typecheck,
    typecheck_program,
)
from typedsl.typecheck import TypeCheckError


class TestTypeCheckResult:
    """Test TypeCheckResult dataclass."""

    def test_valid_result(self) -> None:
        """Test that empty errors means valid."""
        result = TypeCheckResult(errors=())
        assert result.is_valid
        assert result
        assert "valid" in str(result)

    def test_invalid_result(self) -> None:
        """Test that errors means invalid."""
        error = TypeCheckError(
            path="root.value",
            expected="int",
            actual="str",
            message="Expected integer",
        )
        result = TypeCheckResult(errors=(error,))
        assert not result.is_valid
        assert not result
        assert "1 error" in str(result)
        assert "root.value" in str(result)


class TestPrimitiveTypeChecking:
    """Test type checking of primitive types."""

    def test_int_field_valid(self) -> None:
        """Test valid int field."""

        class IntNode(Node[int], tag="int_node_tc"):
            value: int

        node = IntNode(value=42)
        result = typecheck(node)
        assert result.is_valid

    def test_int_field_invalid(self) -> None:
        """Test invalid int field with string value."""

        class IntNode(Node[int], tag="int_node_tc_invalid"):
            value: int

        # Force wrong type through object.__setattr__ or construction
        node = IntNode.__new__(IntNode)
        object.__setattr__(node, "value", "not an int")

        result = typecheck(node)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "int" in result.errors[0].expected

    def test_int_rejects_bool(self) -> None:
        """Test that int field rejects bool (even though bool is subclass of int)."""

        class IntNode(Node[int], tag="int_node_tc_bool"):
            value: int

        node = IntNode.__new__(IntNode)
        object.__setattr__(node, "value", True)

        result = typecheck(node)
        assert not result.is_valid
        assert "bool" in result.errors[0].actual

    def test_float_field_valid(self) -> None:
        """Test valid float field."""

        class FloatNode(Node[float], tag="float_node_tc"):
            value: float

        node = FloatNode(value=3.14)
        result = typecheck(node)
        assert result.is_valid

    def test_float_accepts_int(self) -> None:
        """Test that float field accepts int."""

        class FloatNode(Node[float], tag="float_node_tc_int"):
            value: float

        node = FloatNode(value=42)  # int is promotable to float
        result = typecheck(node)
        assert result.is_valid

    def test_str_field_valid(self) -> None:
        """Test valid str field."""

        class StrNode(Node[str], tag="str_node_tc"):
            value: str

        node = StrNode(value="hello")
        result = typecheck(node)
        assert result.is_valid

    def test_bool_field_valid(self) -> None:
        """Test valid bool field."""

        class BoolNode(Node[bool], tag="bool_node_tc"):
            value: bool

        node = BoolNode(value=True)
        result = typecheck(node)
        assert result.is_valid

    def test_none_field_valid(self) -> None:
        """Test valid None field."""

        class NoneNode(Node[None], tag="none_node_tc"):
            value: None

        node = NoneNode(value=None)
        result = typecheck(node)
        assert result.is_valid

    def test_bytes_field_valid(self) -> None:
        """Test valid bytes field."""

        class BytesNode(Node[bytes], tag="bytes_node_tc"):
            value: bytes

        node = BytesNode(value=b"hello")
        result = typecheck(node)
        assert result.is_valid

    def test_decimal_field_valid(self) -> None:
        """Test valid Decimal field."""

        class DecimalNode(Node[Decimal], tag="decimal_node_tc"):
            value: Decimal

        node = DecimalNode(value=Decimal("3.14"))
        result = typecheck(node)
        assert result.is_valid


class TestTemporalTypeChecking:
    """Test type checking of temporal types."""

    def test_date_field_valid(self) -> None:
        """Test valid date field."""

        class DateNode(Node[datetime.date], tag="date_node_tc"):
            value: datetime.date

        node = DateNode(value=datetime.date(2024, 1, 15))
        result = typecheck(node)
        assert result.is_valid

    def test_date_rejects_datetime(self) -> None:
        """Test that date field rejects datetime."""

        class DateNode(Node[datetime.date], tag="date_node_tc_dt"):
            value: datetime.date

        node = DateNode.__new__(DateNode)
        object.__setattr__(node, "value", datetime.datetime(2024, 1, 15, 12, 0))

        result = typecheck(node)
        assert not result.is_valid
        assert "datetime" in result.errors[0].actual

    def test_time_field_valid(self) -> None:
        """Test valid time field."""

        class TimeNode(Node[datetime.time], tag="time_node_tc"):
            value: datetime.time

        node = TimeNode(value=datetime.time(12, 30, 45))
        result = typecheck(node)
        assert result.is_valid

    def test_datetime_field_valid(self) -> None:
        """Test valid datetime field."""

        class DateTimeNode(Node[datetime.datetime], tag="datetime_node_tc"):
            value: datetime.datetime

        node = DateTimeNode(value=datetime.datetime(2024, 1, 15, 12, 30))
        result = typecheck(node)
        assert result.is_valid

    def test_timedelta_field_valid(self) -> None:
        """Test valid timedelta field."""

        class DurationNode(Node[datetime.timedelta], tag="duration_node_tc"):
            value: datetime.timedelta

        node = DurationNode(value=datetime.timedelta(days=5, hours=3))
        result = typecheck(node)
        assert result.is_valid


class TestContainerTypeChecking:
    """Test type checking of container types."""

    def test_list_valid(self) -> None:
        """Test valid list field."""

        class ListNode(Node[list[int]], tag="list_node_tc"):
            items: list[int]

        node = ListNode(items=[1, 2, 3])
        result = typecheck(node)
        assert result.is_valid

    def test_list_invalid_element(self) -> None:
        """Test list with invalid element type."""

        class ListNode(Node[list[int]], tag="list_node_tc_inv"):
            items: list[int]

        node = ListNode.__new__(ListNode)
        object.__setattr__(node, "items", [1, "two", 3])

        result = typecheck(node)
        assert not result.is_valid
        assert "items[1]" in result.errors[0].path

    def test_set_valid(self) -> None:
        """Test valid set field."""

        class SetNode(Node[set[str]], tag="set_node_tc"):
            items: set[str]

        node = SetNode(items={"a", "b", "c"})
        result = typecheck(node)
        assert result.is_valid

    def test_frozenset_valid(self) -> None:
        """Test valid frozenset field."""

        class FrozenSetNode(Node[frozenset[int]], tag="frozenset_node_tc"):
            items: frozenset[int]

        node = FrozenSetNode(items=frozenset([1, 2, 3]))
        result = typecheck(node)
        assert result.is_valid

    def test_tuple_valid(self) -> None:
        """Test valid tuple field."""

        class TupleNode(Node[tuple[int, str]], tag="tuple_node_tc"):
            pair: tuple[int, str]

        node = TupleNode(pair=(42, "hello"))
        result = typecheck(node)
        assert result.is_valid

    def test_tuple_wrong_length(self) -> None:
        """Test tuple with wrong number of elements."""

        class TupleNode(Node[tuple[int, str]], tag="tuple_node_tc_len"):
            pair: tuple[int, str]

        node = TupleNode.__new__(TupleNode)
        object.__setattr__(node, "pair", (1, "a", "extra"))

        result = typecheck(node)
        assert not result.is_valid
        assert "2 elements" in result.errors[0].message

    def test_dict_valid(self) -> None:
        """Test valid dict field."""

        class DictNode(Node[dict[str, int]], tag="dict_node_tc"):
            data: dict[str, int]

        node = DictNode(data={"a": 1, "b": 2})
        result = typecheck(node)
        assert result.is_valid

    def test_dict_invalid_value(self) -> None:
        """Test dict with invalid value type."""

        class DictNode(Node[dict[str, int]], tag="dict_node_tc_inv"):
            data: dict[str, int]

        node = DictNode.__new__(DictNode)
        object.__setattr__(node, "data", {"a": 1, "b": "two"})

        result = typecheck(node)
        assert not result.is_valid
        assert "data['b']" in result.errors[0].path


class TestLiteralTypeChecking:
    """Test type checking of Literal types."""

    def test_literal_valid(self) -> None:
        """Test valid literal value."""

        class OpNode(Node[str], tag="op_node_tc"):
            op: Literal["+", "-", "*", "/"]

        node = OpNode(op="+")
        result = typecheck(node)
        assert result.is_valid

    def test_literal_invalid(self) -> None:
        """Test invalid literal value."""

        class OpNode(Node[str], tag="op_node_tc_inv"):
            op: Literal["+", "-", "*", "/"]

        node = OpNode.__new__(OpNode)
        object.__setattr__(node, "op", "%")

        result = typecheck(node)
        assert not result.is_valid
        assert "Literal" in result.errors[0].expected


class TestUnionTypeChecking:
    """Test type checking of union types."""

    def test_union_first_option(self) -> None:
        """Test union matching first option."""

        class UnionNode(Node[int | str], tag="union_node_tc1"):
            value: int | str

        node = UnionNode(value=42)
        result = typecheck(node)
        assert result.is_valid

    def test_union_second_option(self) -> None:
        """Test union matching second option."""

        class UnionNode(Node[int | str], tag="union_node_tc2"):
            value: int | str

        node = UnionNode(value="hello")
        result = typecheck(node)
        assert result.is_valid

    def test_union_no_match(self) -> None:
        """Test union with no matching option."""

        class UnionNode(Node[int | str], tag="union_node_tc3"):
            value: int | str

        node = UnionNode.__new__(UnionNode)
        object.__setattr__(node, "value", [1, 2, 3])

        result = typecheck(node)
        assert not result.is_valid
        assert "union" in result.errors[0].message.lower()

    def test_optional_none(self) -> None:
        """Test optional (union with None) with None value."""

        class OptionalNode(Node[int | None], tag="optional_node_tc"):
            value: int | None

        node = OptionalNode(value=None)
        result = typecheck(node)
        assert result.is_valid

    def test_optional_value(self) -> None:
        """Test optional with actual value."""

        class OptionalNode(Node[int | None], tag="optional_node_tc2"):
            value: int | None

        node = OptionalNode(value=42)
        result = typecheck(node)
        assert result.is_valid


class TestNestedNodeTypeChecking:
    """Test type checking of nested nodes."""

    def test_nested_node_valid(self) -> None:
        """Test valid nested node."""

        class Leaf(Node[int], tag="leaf_tc"):
            value: int

        class Parent(Node[int], tag="parent_tc"):
            child: Node[int]

        node = Parent(child=Leaf(value=42))
        result = typecheck(node)
        assert result.is_valid

    def test_nested_node_wrong_return_type(self) -> None:
        """Test nested node with wrong return type."""

        class IntLeaf(Node[int], tag="int_leaf_tc"):
            value: int

        class StrLeaf(Node[str], tag="str_leaf_tc"):
            value: str

        class Parent(Node[int], tag="parent_tc_wrong"):
            child: Node[int]

        # Create parent with wrong child type
        node = Parent.__new__(Parent)
        object.__setattr__(node, "child", StrLeaf(value="hello"))

        result = typecheck(node)
        assert not result.is_valid
        assert "Node[int]" in result.errors[0].expected

    def test_deeply_nested_nodes(self) -> None:
        """Test deeply nested valid nodes."""

        class Const(Node[float], tag="const_tc_deep"):
            value: float

        class BinOp(Node[float], tag="binop_tc_deep"):
            left: Node[float]
            right: Node[float]

        # (1 + 2) * 3
        node = BinOp(
            left=BinOp(left=Const(value=1.0), right=Const(value=2.0)),
            right=Const(value=3.0),
        )
        result = typecheck(node)
        assert result.is_valid


class TestSpecificNodeTypeChecking:
    """Test type checking of specific node types."""

    def test_specific_node_type_valid(self) -> None:
        """Test valid specific node type field."""

        class Leaf(Node[int], tag="specific_leaf_tc"):
            value: int

        class Wrapper(Node[int], tag="specific_wrapper_tc"):
            # Using the Leaf type directly (not Node[int])
            # This will be extracted as NodeType
            pass

        # For now, we test using Node[int] which is ReturnType
        class Container(Node[int], tag="container_tc"):
            child: Node[int]

        node = Container(child=Leaf(value=42))
        result = typecheck(node)
        assert result.is_valid


class TestReferenceTypeChecking:
    """Test type checking of Ref types."""

    def test_ref_valid(self) -> None:
        """Test valid reference in Program context."""

        class Value(Node[int], tag="value_tc_ref"):
            num: int

        class Wrapper(Node[int], tag="wrapper_tc_ref"):
            child: Ref[Node[int]]

        prog = Program(
            root=Ref(id="wrapper"),
            nodes={
                "value": Value(num=42),
                "wrapper": Wrapper(child=Ref(id="value")),
            },
        )

        result = typecheck_program(prog)
        assert result.is_valid

    def test_ref_missing_target(self) -> None:
        """Test reference to non-existent node."""

        class Value(Node[int], tag="value_tc_ref_miss"):
            num: int

        class Wrapper(Node[int], tag="wrapper_tc_ref_miss"):
            child: Ref[Node[int]]

        prog = Program(
            root=Ref(id="wrapper"),
            nodes={
                "wrapper": Wrapper(child=Ref(id="nonexistent")),
            },
        )

        result = typecheck_program(prog)
        assert not result.is_valid
        assert "nonexistent" in result.errors[0].message

    def test_ref_wrong_return_type(self) -> None:
        """Test reference to node with wrong return type."""

        class IntValue(Node[int], tag="int_value_tc_ref"):
            num: int

        class StrValue(Node[str], tag="str_value_tc_ref"):
            text: str

        class Wrapper(Node[int], tag="wrapper_tc_ref_type"):
            child: Ref[Node[int]]

        prog = Program(
            root=Ref(id="wrapper"),
            nodes={
                "str_val": StrValue(text="hello"),
                "wrapper": Wrapper(child=Ref(id="str_val")),
            },
        )

        result = typecheck_program(prog)
        assert not result.is_valid
        assert "returns str" in str(result) or "Node[str]" in str(result)


class TestProgramTypeChecking:
    """Test type checking of full Programs."""

    def test_program_all_nodes_checked(self) -> None:
        """Test that all nodes in program are checked."""

        class Num(Node[int], tag="num_tc_prog"):
            value: int

        prog = Program(
            root=Ref(id="a"),
            nodes={
                "a": Num(value=1),
                "b": Num(value=2),
                "c": Num(value=3),
            },
        )

        result = typecheck_program(prog)
        assert result.is_valid

    def test_program_with_inline_root(self) -> None:
        """Test program with inline root node."""

        class Num(Node[int], tag="num_tc_prog_inline"):
            value: int

        prog = Program(root=Num(value=42))

        result = typecheck_program(prog)
        assert result.is_valid

    def test_program_invalid_node_in_mapping(self) -> None:
        """Test that invalid node in mapping is caught."""

        class Num(Node[int], tag="num_tc_prog_inv"):
            value: int

        # Create a node with wrong type
        bad_node = Num.__new__(Num)
        object.__setattr__(bad_node, "value", "not an int")

        prog = Program(
            root=Ref(id="good"),
            nodes={
                "good": Num(value=42),
                "bad": bad_node,
            },
        )

        result = typecheck_program(prog)
        assert not result.is_valid
        assert "nodes['bad']" in result.errors[0].path

    def test_complex_program(self) -> None:
        """Test complex program with multiple node types and references."""

        class Const(Node[float], tag="const_tc_complex"):
            value: float

        class Var(Node[float], tag="var_tc_complex"):
            name: str

        class BinOp(Node[float], tag="binop_tc_complex"):
            op: Literal["+", "-", "*", "/"]
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        # Expression: (x + 2) * y
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Var(name="x"),
                "y": Var(name="y"),
                "two": Const(value=2.0),
                "sum": BinOp(op="+", left=Ref(id="x"), right=Ref(id="two")),
                "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="y")),
            },
        )

        result = typecheck_program(prog)
        assert result.is_valid


class TestChildTypeAlias:
    """Test type checking with Child[T] type alias."""

    def test_child_inline_node(self) -> None:
        """Test Child field with inline node."""

        class Leaf(Node[int], tag="leaf_tc_child"):
            value: int

        class Parent(Node[int], tag="parent_tc_child"):
            child: Child[int]

        node = Parent(child=Leaf(value=42))
        result = typecheck(node)
        assert result.is_valid

    def test_child_ref(self) -> None:
        """Test Child field with reference."""

        class Leaf(Node[int], tag="leaf_tc_child_ref"):
            value: int

        class Parent(Node[int], tag="parent_tc_child_ref"):
            child: Child[int]

        prog = Program(
            root=Ref(id="parent"),
            nodes={
                "leaf": Leaf(value=42),
                "parent": Parent(child=Ref(id="leaf")),
            },
        )

        result = typecheck_program(prog)
        assert result.is_valid


class TestGenericNodes:
    """Test type checking of generic nodes."""

    def test_generic_node_instantiation(self) -> None:
        """Test generic node with type parameter."""

        class Container[T](Node[T], tag="container_tc_generic"):
            value: T

        # With int
        int_container = Container(value=42)
        result = typecheck(int_container)
        assert result.is_valid

        # With str
        str_container = Container(value="hello")
        result = typecheck(str_container)
        assert result.is_valid

    def test_generic_node_list(self) -> None:
        """Test generic node with list of type parameter."""

        class ListContainer[T](Node[list[T]], tag="list_container_tc"):
            items: list[T]

        node = ListContainer(items=[1, 2, 3])
        result = typecheck(node)
        assert result.is_valid


class TestTypeParameterBounds:
    """Test type checking of type parameters with bounds."""

    def test_bounded_type_parameter_valid(self) -> None:
        """Test that bounded type parameter accepts values matching the bound."""
        T = TypeVar("T", bound=int)

        class BoundedContainer(Node[int], tag="bounded_container_tc"):
            __type_params__ = (T,)
            value: T  # type: ignore[valid-type]

        node = BoundedContainer(value=42)
        result = typecheck(node)
        assert result.is_valid

    def test_bounded_type_parameter_invalid(self) -> None:
        """Test that bounded type parameter rejects values not matching bound."""
        T = TypeVar("T", bound=int)

        class BoundedContainer2(Node[int], tag="bounded_container_tc2"):
            __type_params__ = (T,)
            value: T  # type: ignore[valid-type]

        node = BoundedContainer2.__new__(BoundedContainer2)
        object.__setattr__(node, "value", "not an int")

        result = typecheck(node)
        assert not result.is_valid
        assert "int" in result.errors[0].expected

    def test_bounded_type_parameter_with_node_bound(self) -> None:
        """Test bounded type parameter where bound is Node[int]."""

        class IntLeaf(Node[int], tag="int_leaf_bound_tc"):
            value: int

        T = TypeVar("T", bound=Node[int])

        class NodeContainer(Node[int], tag="node_container_bound_tc"):
            __type_params__ = (T,)
            child: T  # type: ignore[valid-type]

        node = NodeContainer(child=IntLeaf(value=42))
        result = typecheck(node)
        assert result.is_valid

    def test_bounded_type_parameter_rejects_wrong_node(self) -> None:
        """Test bounded type parameter rejects node with wrong return type."""

        class StrLeaf(Node[str], tag="str_leaf_bound_tc"):
            text: str

        T = TypeVar("T", bound=Node[int])

        class NodeContainer2(Node[int], tag="node_container_bound_tc2"):
            __type_params__ = (T,)
            child: T  # type: ignore[valid-type]

        node = NodeContainer2.__new__(NodeContainer2)
        object.__setattr__(node, "child", StrLeaf(text="hello"))

        result = typecheck(node)
        assert not result.is_valid

    def test_unbounded_type_parameter_accepts_any(self) -> None:
        """Test that unbounded type parameter accepts any value."""

        class AnyContainer[T](Node[T], tag="any_container_tc"):
            value: T

        # int
        result = typecheck(AnyContainer(value=42))
        assert result.is_valid

        # str
        result = typecheck(AnyContainer(value="hello"))
        assert result.is_valid

        # list
        result = typecheck(AnyContainer(value=[1, 2, 3]))
        assert result.is_valid

        # None
        result = typecheck(AnyContainer(value=None))
        assert result.is_valid


class TestRefTypeExpectedReturnType:
    """Test that RefType correctly validates expected return types."""

    def test_ref_with_return_type_valid(self) -> None:
        """Test Ref[Node[int]] accepts ref to Node[int]."""

        class IntNode(Node[int], tag="int_node_ref_ret"):
            value: int

        class Wrapper(Node[int], tag="wrapper_ref_ret"):
            child: Ref[Node[int]]

        prog = Program(
            root=Ref(id="wrapper"),
            nodes={
                "int_val": IntNode(value=42),
                "wrapper": Wrapper(child=Ref(id="int_val")),
            },
        )

        result = typecheck_program(prog)
        assert result.is_valid

    def test_ref_with_return_type_invalid(self) -> None:
        """Test Ref[Node[int]] rejects ref to Node[str]."""

        class IntNode(Node[int], tag="int_node_ref_ret2"):
            value: int

        class StrNode(Node[str], tag="str_node_ref_ret"):
            text: str

        class Wrapper(Node[int], tag="wrapper_ref_ret2"):
            child: Ref[Node[int]]

        prog = Program(
            root=Ref(id="wrapper"),
            nodes={
                "str_val": StrNode(text="hello"),
                "wrapper": Wrapper(child=Ref(id="str_val")),
            },
        )

        result = typecheck_program(prog)
        assert not result.is_valid
        # Should mention str and int return types
        error_str = str(result)
        assert "str" in error_str
        assert "int" in error_str

    def test_ref_with_covariant_return_type(self) -> None:
        """Test Ref[Node[float]] accepts ref to Node[int] (covariance)."""

        class IntNode(Node[int], tag="int_node_ref_covar"):
            value: int

        class Wrapper(Node[float], tag="wrapper_ref_covar"):
            child: Ref[Node[float]]

        prog = Program(
            root=Ref(id="wrapper"),
            nodes={
                "int_val": IntNode(value=42),
                "wrapper": Wrapper(child=Ref(id="int_val")),
            },
        )

        result = typecheck_program(prog)
        assert result.is_valid

    def test_ref_with_specific_node_type(self) -> None:
        """Test Ref[SpecificNode] checks node tag, not just return type."""

        class LeafA(Node[int], tag="leaf_a_ref_tc"):
            value: int

        class LeafB(Node[int], tag="leaf_b_ref_tc"):
            value: int

        class Wrapper(Node[int], tag="wrapper_specific_ref"):
            child: Ref[LeafA]

        # Valid: ref to LeafA
        prog_valid = Program(
            root=Ref(id="wrapper"),
            nodes={
                "leaf": LeafA(value=42),
                "wrapper": Wrapper(child=Ref(id="leaf")),
            },
        )
        result = typecheck_program(prog_valid)
        assert result.is_valid

        # Invalid: ref to LeafB (same return type, but wrong node type)
        prog_invalid = Program(
            root=Ref(id="wrapper"),
            nodes={
                "leaf": LeafB(value=42),
                "wrapper": Wrapper(child=Ref(id="leaf")),
            },
        )
        result = typecheck_program(prog_invalid)
        assert not result.is_valid
        assert "leaf_a_ref_tc" in str(result) or "leaf_b_ref_tc" in str(result)

    def test_ref_without_program_context_skips_validation(self) -> None:
        """Test that refs without program context don't error."""

        class IntNode(Node[int], tag="int_node_no_ctx"):
            value: int

        class Wrapper(Node[int], tag="wrapper_no_ctx"):
            child: Ref[Node[int]]

        # Check node without program - ref target cannot be validated
        wrapper = Wrapper(child=Ref(id="nonexistent"))
        result = typecheck(wrapper)
        # Should be valid because we can't check refs without program
        assert result.is_valid

    def test_ref_to_node_with_union_return_type(self) -> None:
        """Test Ref[Node[int | str]] accepts refs to Node[int] or Node[str]."""

        class IntNode(Node[int], tag="int_node_union_ref"):
            value: int

        class StrNode(Node[str], tag="str_node_union_ref"):
            text: str

        class Wrapper(Node[int | str], tag="wrapper_union_ref"):
            child: Ref[Node[int | str]]

        # Int node is valid
        prog_int = Program(
            root=Ref(id="wrapper"),
            nodes={
                "val": IntNode(value=42),
                "wrapper": Wrapper(child=Ref(id="val")),
            },
        )
        result = typecheck_program(prog_int)
        assert result.is_valid

        # Str node is valid
        prog_str = Program(
            root=Ref(id="wrapper"),
            nodes={
                "val": StrNode(text="hello"),
                "wrapper": Wrapper(child=Ref(id="val")),
            },
        )
        result = typecheck_program(prog_str)
        assert result.is_valid


class TestTypeCheckerAPI:
    """Test TypeChecker class API."""

    def test_checker_reusable(self) -> None:
        """Test that TypeChecker can be reused."""

        class Num(Node[int], tag="num_tc_reuse"):
            value: int

        checker = TypeChecker()

        node1 = Num(value=1)
        result1 = checker.check_node(node1)
        assert result1.is_valid

        node2 = Num(value=2)
        result2 = checker.check_node(node2)
        assert result2.is_valid

    def test_check_node_with_program_context(self) -> None:
        """Test checking node with program context for refs."""

        class Value(Node[int], tag="value_tc_ctx"):
            num: int

        class Wrapper(Node[int], tag="wrapper_tc_ctx"):
            child: Ref[Node[int]]

        prog = Program(
            root=Ref(id="value"),
            nodes={"value": Value(num=42)},
        )

        wrapper = Wrapper(child=Ref(id="value"))

        checker = TypeChecker()
        result = checker.check_node(wrapper, program=prog)
        assert result.is_valid


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_typecheck_function(self) -> None:
        """Test typecheck() convenience function."""

        class Num(Node[int], tag="num_tc_conv"):
            value: int

        result = typecheck(Num(value=42))
        assert result.is_valid

    def test_typecheck_program_function(self) -> None:
        """Test typecheck_program() convenience function."""

        class Num(Node[int], tag="num_tc_conv_prog"):
            value: int

        prog = Program(root=Num(value=42))
        result = typecheck_program(prog)
        assert result.is_valid


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_node(self) -> None:
        """Test node with no fields."""

        class Empty(Node[None], tag="empty_tc"):
            pass

        node = Empty()
        result = typecheck(node)
        assert result.is_valid

    def test_multiple_errors(self) -> None:
        """Test that multiple errors are collected."""

        class Multi(Node[int], tag="multi_tc"):
            a: int
            b: str
            c: bool

        node = Multi.__new__(Multi)
        object.__setattr__(node, "a", "not int")
        object.__setattr__(node, "b", 123)
        object.__setattr__(node, "c", "not bool")

        result = typecheck(node)
        assert not result.is_valid
        assert len(result.errors) == 3

    def test_error_path_tracking(self) -> None:
        """Test that error paths are correctly tracked."""

        class Inner(Node[int], tag="inner_tc_path"):
            value: int

        class Outer(Node[int], tag="outer_tc_path"):
            child: Node[int]

        inner = Inner.__new__(Inner)
        object.__setattr__(inner, "value", "wrong")

        outer = Outer(child=inner)
        result = typecheck(outer)

        assert not result.is_valid
        assert "root.child.value" in result.errors[0].path

    def test_nested_list_errors(self) -> None:
        """Test error tracking in nested lists."""

        class ListNode(Node[list[int]], tag="list_tc_nested"):
            items: list[list[int]]

        node = ListNode.__new__(ListNode)
        object.__setattr__(node, "items", [[1, 2], [3, "wrong"]])

        result = typecheck(node)
        assert not result.is_valid
        assert "items[1][1]" in result.errors[0].path
