"""Tests for the constraint-based type checker."""

from __future__ import annotations

from typing import TypeVar

from typedsl import Node, Program, Ref
from typedsl.checker import check_node, check_program
from typedsl.checker.convert import format_type_expr, to_type_expr
from typedsl.checker.types import TCon, TVar, reset_var_counter

# =============================================================================
# Test Node Definitions (prefixed with Tc to avoid tag collisions)
# =============================================================================


class TcConst[T](Node[T]):
    """A constant value node."""

    value: T


class TcAdd(Node[float]):
    """Addition of two float-returning nodes."""

    left: Node[float] | Ref[Node[float]]
    right: Node[float] | Ref[Node[float]]


class TcIntAdd(Node[int]):
    """Addition of two int-returning nodes."""

    left: Node[int] | Ref[Node[int]]
    right: Node[int] | Ref[Node[int]]


class TcContainer[T](Node[list[T]]):
    """A container holding a list of T-returning nodes."""

    items: list[Node[T] | Ref[Node[T]]]


class TcWrapper[T](Node[T]):
    """A wrapper that passes through the inner node's type."""

    inner: Node[T] | Ref[Node[T]]


class TcTransform[T, U](Node[U]):
    """Transforms from T to U."""

    input: Node[T] | Ref[Node[T]]
    output_type: type


# For bounded type parameter tests
IntOrFloat = TypeVar("IntOrFloat", bound=int | float)


class TcNumericOp[T: int | float](Node[T]):
    """Operation on numeric types with a bound."""

    value: T


# Aliases for cleaner test code
Const = TcConst
Add = TcAdd
IntAdd = TcIntAdd
Container = TcContainer
Wrapper = TcWrapper
Transform = TcTransform
NumericOp = TcNumericOp


# =============================================================================
# Test: Type Expression Conversion
# =============================================================================


class TestTypeExprConversion:
    """Tests for converting Python types to TypeExpr."""

    def test_simple_types(self) -> None:
        """Test conversion of simple types."""
        var_map: dict[str, TVar] = {}

        assert to_type_expr(int, var_map) == TCon(int, ())
        assert to_type_expr(str, var_map) == TCon(str, ())
        assert to_type_expr(float, var_map) == TCon(float, ())
        assert to_type_expr(bool, var_map) == TCon(bool, ())

    def test_generic_types(self) -> None:
        """Test conversion of generic types."""
        var_map: dict[str, TVar] = {}

        result = to_type_expr(list[int], var_map)
        assert result == TCon(list, (TCon(int, ()),))

        result = to_type_expr(dict[str, int], var_map)
        assert result == TCon(dict, (TCon(str, ()), TCon(int, ())))

    def test_type_variables(self) -> None:
        """Test conversion of TypeVars."""
        reset_var_counter()
        var_map: dict[str, TVar] = {}
        T = TypeVar("T")

        result = to_type_expr(T, var_map)
        assert isinstance(result, TVar)
        assert result.name == "T"

        # Same TypeVar name should return same TVar
        result2 = to_type_expr(T, var_map)
        assert result2 is var_map["T"]

    def test_nested_type_variables(self) -> None:
        """Test conversion of nested types with variables."""
        reset_var_counter()
        var_map: dict[str, TVar] = {}
        T = TypeVar("T")

        result = to_type_expr(list[T], var_map)
        assert isinstance(result, TCon)
        assert result.con is list
        assert len(result.args) == 1
        assert isinstance(result.args[0], TVar)


class TestFormatTypeExpr:
    """Tests for formatting type expressions."""

    def test_format_simple(self) -> None:
        """Test formatting simple types."""
        assert format_type_expr(TCon(int, ())) == "int"
        assert format_type_expr(TCon(str, ())) == "str"

    def test_format_generic(self) -> None:
        """Test formatting generic types."""
        expr = TCon(list, (TCon(int, ()),))
        assert format_type_expr(expr) == "list[int]"

    def test_format_variable(self) -> None:
        """Test formatting type variables."""
        var = TVar(id=1, name="T")
        assert format_type_expr(var) == "?T"


# =============================================================================
# Test: Basic Type Checking
# =============================================================================


class TestBasicTypeChecking:
    """Tests for basic type checking scenarios."""

    def test_simple_const(self) -> None:
        """Test type checking a simple constant node."""
        node = Const[int](value=42)
        result = check_node(node)
        assert result.success

    def test_simple_addition(self) -> None:
        """Test type checking a simple addition."""
        prog = Program(
            root=Add(
                left=Const[float](value=1.0),
                right=Const[float](value=2.0),
            ),
        )
        result = check_program(prog)
        assert result.success

    def test_program_with_references(self) -> None:
        """Test type checking a program with references."""
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Const[float](value=1.0),
                "y": Const[float](value=2.0),
                "result": Add(left=Ref(id="x"), right=Ref(id="y")),
            },
        )
        result = check_program(prog)
        assert result.success


# =============================================================================
# Test: Type Errors
# =============================================================================


class TestTypeErrors:
    """Tests for detecting type errors."""

    def test_type_mismatch_in_field(self) -> None:
        """Test detection of type mismatch in a field."""
        # Add expects float nodes, but we give it int nodes
        prog = Program(
            root=Add(
                left=Const[int](value=1),  # Should be float
                right=Const[float](value=2.0),
            ),
        )
        result = check_program(prog)
        assert not result.success
        assert len(result.errors) > 0

    def test_type_mismatch_via_reference(self) -> None:
        """Test detection of type mismatch through reference."""
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Const[int](value=1),  # int, not float
                "result": Add(left=Ref(id="x"), right=Const[float](value=2.0)),
            },
        )
        result = check_program(prog)
        assert not result.success

    def test_unresolved_reference(self) -> None:
        """Test detection of unresolved reference."""
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "result": Add(
                    left=Ref(id="nonexistent"),  # Does not exist
                    right=Const[float](value=2.0),
                ),
            },
        )
        result = check_program(prog)
        assert not result.success
        # Should have unresolved reference error
        assert any("nonexistent" in err.message for err in result.errors)


# =============================================================================
# Test: Generic Nodes
# =============================================================================


class TestGenericNodes:
    """Tests for generic node type checking."""

    def test_generic_const(self) -> None:
        """Test generic Const with different types."""
        # All these should pass
        assert check_node(Const[int](value=42)).success
        assert check_node(Const[str](value="hello")).success
        assert check_node(Const[float](value=3.14)).success

    def test_container_with_consistent_types(self) -> None:
        """Test container with consistently typed items."""
        prog = Program(
            root=Container[int](
                items=[
                    Const[int](value=1),
                    Const[int](value=2),
                    Const[int](value=3),
                ],
            ),
        )
        result = check_program(prog)
        assert result.success

    def test_container_with_inconsistent_types(self) -> None:
        """Test container with inconsistently typed items."""
        prog = Program(
            root=Container[int](
                items=[
                    Const[int](value=1),
                    Const[str](value="oops"),  # Wrong type!
                ],
            ),
        )
        result = check_program(prog)
        assert not result.success

    def test_wrapper_propagates_type(self) -> None:
        """Test that Wrapper correctly propagates inner type."""
        prog = Program(
            root=Wrapper[int](inner=Const[int](value=42)),
        )
        result = check_program(prog)
        assert result.success

    def test_wrapper_type_mismatch(self) -> None:
        """Test Wrapper with mismatched inner type.

        Note: Python erases type arguments at runtime, so Wrapper[int]
        doesn't directly constrain T. To detect the mismatch, we need
        to place Wrapper inside a node that expects a specific return type.
        """
        # IntAdd expects Node[int] children, but Wrapper contains Const[str]
        prog = Program(
            root=IntAdd(
                left=Wrapper[int](inner=Const[str](value="wrong")),  # str != int
                right=Const[int](value=1),
            ),
        )
        result = check_program(prog)
        assert not result.success


# =============================================================================
# Test: Deep Generic Chaining
# =============================================================================


class TestDeepGenericChaining:
    """Tests for deep generic type parameter chaining."""

    def test_nested_wrappers(self) -> None:
        """Test nested wrappers maintain type consistency."""
        prog = Program(
            root=Wrapper[int](
                inner=Wrapper[int](
                    inner=Wrapper[int](inner=Const[int](value=42)),
                ),
            ),
        )
        result = check_program(prog)
        assert result.success

    def test_nested_wrappers_type_error(self) -> None:
        """Test type error in deeply nested structure.

        The deeply nested Const[str] conflicts with IntAdd expecting int.
        """
        prog = Program(
            root=IntAdd(
                left=Wrapper[int](
                    inner=Wrapper[int](
                        inner=Wrapper[str](  # Type changes here!
                            inner=Const[str](value="wrong"),
                        ),
                    ),
                ),
                right=Const[int](value=1),
            ),
        )
        result = check_program(prog)
        assert not result.success

    def test_container_of_wrappers(self) -> None:
        """Test Container of Wrapper nodes."""
        prog = Program(
            root=Container[int](
                items=[
                    Wrapper[int](inner=Const[int](value=1)),
                    Wrapper[int](inner=Const[int](value=2)),
                ],
            ),
        )
        result = check_program(prog)
        assert result.success

    def test_wrapper_of_container(self) -> None:
        """Test Wrapper containing Container."""
        prog = Program(
            root=Wrapper[list[int]](
                inner=Container[int](
                    items=[
                        Const[int](value=1),
                        Const[int](value=2),
                    ],
                ),
            ),
        )
        result = check_program(prog)
        assert result.success


# =============================================================================
# Test: References with Generics
# =============================================================================


class TestReferencesWithGenerics:
    """Tests for references with generic nodes."""

    def test_shared_generic_node(self) -> None:
        """Test sharing a generic node via reference."""
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "shared": Const[int](value=42),
                "wrap1": Wrapper[int](inner=Ref(id="shared")),
                "wrap2": Wrapper[int](inner=Ref(id="shared")),
                "result": Container[int](
                    items=[Ref(id="wrap1"), Ref(id="wrap2")],
                ),
            },
        )
        result = check_program(prog)
        assert result.success

    def test_reference_type_mismatch(self) -> None:
        """Test type mismatch when using reference.

        The str_const returns str, but IntAdd expects int children.
        """
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "str_const": Const[str](value="hello"),
                "int_const": Const[int](value=1),
                "result": IntAdd(
                    left=Wrapper[int](inner=Ref(id="str_const")),  # Wrong! str != int
                    right=Ref(id="int_const"),
                ),
            },
        )
        result = check_program(prog)
        assert not result.success


# =============================================================================
# Test: Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests for error message quality."""

    def test_error_includes_location(self) -> None:
        """Test that errors include source location."""
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "result": Add(
                    left=Const[int](value=1),
                    right=Const[float](value=2.0),
                ),
            },
        )
        result = check_program(prog)
        assert not result.success

        formatted = result.format_errors()
        assert "result" in formatted  # Should mention the node

    def test_unresolved_reference_lists_available(self) -> None:
        """Test that unresolved reference error lists available nodes."""
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "a": Const[int](value=1),
                "b": Const[int](value=2),
                "result": IntAdd(left=Ref(id="missing"), right=Ref(id="b")),
            },
        )
        result = check_program(prog)
        assert not result.success

        formatted = result.format_errors()
        assert "missing" in formatted


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_container(self) -> None:
        """Test container with no items."""
        prog = Program(root=Container[int](items=[]))
        result = check_program(prog)
        assert result.success

    def test_self_referential_program(self) -> None:
        """Test program where nodes reference each other."""
        prog = Program(
            root=Ref(id="a"),
            nodes={
                "a": IntAdd(left=Ref(id="b"), right=Const[int](value=1)),
                "b": IntAdd(left=Ref(id="c"), right=Const[int](value=2)),
                "c": Const[int](value=3),
            },
        )
        result = check_program(prog)
        assert result.success
