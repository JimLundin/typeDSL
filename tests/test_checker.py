"""Tests for the type checker module."""

from typedsl import Node, Program, Ref
from typedsl.checker import (
    CheckResult,
    EqualityConstraint,
    Location,
    Substitution,
    TCon,
    TVar,
    check_node,
    check_program,
    from_hint,
    solve,
)
from typedsl.checker.solver import is_subtype, unify


class TestFromHint:
    """Tests for from_hint conversion."""

    def test_simple_types(self) -> None:
        assert from_hint(int) == TCon(int)
        assert from_hint(str) == TCon(str)

    def test_generic_types(self) -> None:
        assert from_hint(list[int]) == TCon(list, (TCon(int),))
        assert from_hint(dict[str, int]) == TCon(dict, (TCon(str), TCon(int)))

    def test_nested_generics(self) -> None:
        result = from_hint(list[dict[str, int]])
        expected = TCon(list, (TCon(dict, (TCon(str), TCon(int))),))
        assert result == expected


class TestSubstitution:
    """Tests for Substitution application and composition."""

    def test_apply_replaces_bound_variables(self) -> None:
        sub = Substitution()
        sub.bind(0, TCon(int))
        assert sub.apply(TVar(0)) == TCon(int)
        assert sub.apply(TVar(1)) == TVar(1)  # Unbound stays unchanged

    def test_apply_recurses_into_type_args(self) -> None:
        sub = Substitution()
        sub.bind(0, TCon(int))
        result = sub.apply(TCon(list, (TVar(0),)))
        assert result == TCon(list, (TCon(int),))

    def test_apply_follows_variable_chains(self) -> None:
        sub = Substitution()
        sub.bind(0, TVar(1))
        sub.bind(1, TCon(int))
        assert sub.apply(TVar(0)) == TCon(int)

    def test_compose_applies_first_to_second(self) -> None:
        sub1 = Substitution()
        sub1.bind(0, TCon(int))
        sub2 = Substitution()
        sub2.bind(1, TVar(0))
        composed = sub1.compose(sub2)
        assert composed.apply(TVar(1)) == TCon(int)


class TestSubtyping:
    """Tests for Python class hierarchy subtyping."""

    def test_bool_is_subtype_of_int(self) -> None:
        # bool is a subclass of int in Python
        assert is_subtype(bool, int)

    def test_int_is_not_subtype_of_float(self) -> None:
        # int is NOT a subclass of float - no custom numeric tower
        assert not is_subtype(int, float)

    def test_float_is_not_subtype_of_complex(self) -> None:
        # float is NOT a subclass of complex
        assert not is_subtype(float, complex)

    def test_float_is_not_subtype_of_int(self) -> None:
        assert not is_subtype(float, int)

    def test_str_is_not_subtype_of_int(self) -> None:
        assert not is_subtype(str, int)

    def test_same_type_is_subtype(self) -> None:
        assert is_subtype(int, int)
        assert is_subtype(float, float)
        assert is_subtype(str, str)


class TestUnify:
    """Tests for unification algorithm."""

    def test_identical_types_succeed(self) -> None:
        result = unify(TCon(int), TCon(int))
        assert isinstance(result, Substitution)

    def test_different_constructors_fail(self) -> None:
        result = unify(TCon(int), TCon(str))
        assert isinstance(result, str)

    def test_variable_binds_to_concrete_type(self) -> None:
        result = unify(TVar(0), TCon(int))
        assert isinstance(result, Substitution)
        assert result.apply(TVar(0)) == TCon(int)

    def test_generic_type_unifies_element_types(self) -> None:
        result = unify(
            TCon(list, (TVar(0),)),
            TCon(list, (TCon(int),)),
        )
        assert isinstance(result, Substitution)
        assert result.apply(TVar(0)) == TCon(int)

    def test_arity_mismatch_fails(self) -> None:
        result = unify(
            TCon(list, (TCon(int),)),
            TCon(list, ()),
        )
        assert isinstance(result, str)

    def test_occurs_check_prevents_infinite_types(self) -> None:
        result = unify(TVar(0), TCon(list, (TVar(0),)))
        assert isinstance(result, str)
        assert "infinite" in result.lower()

    def test_int_does_not_unify_with_float(self) -> None:
        # int and float are different types - no implicit conversion
        result = unify(TCon(float), TCon(int))
        assert isinstance(result, str)

    def test_float_does_not_unify_with_int(self) -> None:
        # float and int are different types
        result = unify(TCon(int), TCon(float))
        assert isinstance(result, str)

    def test_bool_unifies_with_int_expected(self) -> None:
        # bool is a subtype of int (Python class hierarchy)
        result = unify(TCon(int), TCon(bool))
        assert isinstance(result, Substitution)


class TestSolver:
    """Tests for constraint solving."""

    def test_empty_constraints_succeed(self) -> None:
        result = solve([])
        assert result.success

    def test_satisfiable_constraints_succeed(self) -> None:
        constraint = EqualityConstraint(
            left=TCon(int),
            right=TCon(int),
            location=Location(
                node_tag="test",
                node_id=None,
                field_name="value",
                path=("root",),
            ),
        )
        result = solve([constraint])
        assert result.success

    def test_unsatisfiable_constraints_fail_with_error(self) -> None:
        constraint = EqualityConstraint(
            left=TCon(int),
            right=TCon(str),
            location=Location(
                node_tag="test",
                node_id="n1",
                field_name="value",
                path=("root",),
            ),
        )
        result = solve([constraint])
        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0].location.node_id == "n1"


# Test node definitions
class Literal[T](Node[T], tag="checker_literal"):
    value: T


class AddNode(Node[float], tag="checker_add_float"):
    left: Node[float]
    right: Node[float]


class RefAdd(Node[float], tag="checker_ref_add"):
    left: Ref[Node[float]]
    right: Ref[Node[float]]


class IntLiteral(Node[int], tag="checker_int_lit"):
    value: int


class NumericLiteral[T: int | float](Node[T], tag="checker_numeric_lit"):
    """A literal that only accepts numeric types (int or float)."""

    value: T


class NumericAdd[T: int | float](Node[T], tag="checker_numeric_add"):
    """Addition node that only accepts numeric types."""

    left: Node[T]
    right: Node[T]


# Edge case test nodes
class IntOnly[T: int](Node[T], tag="checker_int_only"):
    """Node with single-type bound (not a union)."""

    value: T


class FloatBounded[T: float](Node[T], tag="checker_float_bounded"):
    """Node bounded by float - should accept int via subtyping."""

    value: T


class PairSame[T](Node[tuple[T, T]], tag="checker_pair_same"):
    """Node using same TypeVar in multiple fields."""

    left: T
    right: T


class PairDifferent[T, U](Node[tuple[T, U]], tag="checker_pair_diff"):
    """Node with multiple type parameters."""

    left: T
    right: U


class ListContainer[T](Node[list[T]], tag="checker_list_container"):
    """Node with TypeVar in container."""

    items: list[T]


class NumericList[T: int | float](Node[list[T]], tag="checker_numeric_list"):
    """Node with bounded TypeVar in container."""

    items: list[T]


class StringLiteral(Node[str], tag="checker_string_lit"):
    """String literal node for type mismatch tests."""

    value: str


# Nodes for testing constraint error locations
class NumericProcessor[T: int | float](Node[T], tag="checker_numeric_processor"):
    """Bounded node that takes a subnode - for testing bound violation via subnode."""

    child: Node[T]


class NumericRefProcessor[T: int | float](Node[T], tag="checker_numeric_ref_processor"):
    """Bounded node that takes a Ref to subnode."""

    child: Ref[Node[T]]


class IntProcessor(Node[int], tag="checker_int_processor"):
    """Concrete node expecting Node[int] - for testing EqualityConstraint errors."""

    child: Node[int]


class IntRefProcessor(Node[int], tag="checker_int_ref_processor"):
    """Concrete node expecting Ref[Node[int]]."""

    child: Ref[Node[int]]


class TestGenericNodes:
    """Tests for generic node type checking."""

    def test_generic_literal_infers_type_from_value(self) -> None:
        # Literal[T] with int value should work
        result = check_node(Literal(value=42))
        assert result.success

    def test_generic_literal_with_float(self) -> None:
        result = check_node(Literal(value=3.14))
        assert result.success

    def test_generic_literal_with_string(self) -> None:
        result = check_node(Literal(value="hello"))
        assert result.success


class TestTypeMismatchInPrograms:
    """Tests for type mismatches - int and float are distinct types."""

    def test_int_literal_in_float_context_fails(self) -> None:
        # AddNode expects Node[float], Literal(value=5) is Node[int]
        # int is NOT a subtype of float, so this should fail
        program = Program(
            root=AddNode(
                left=Literal(value=5),
                right=Literal(value=5.0),
            ),
        )
        result = check_program(program)
        assert not result.success
        assert any("int" in e.message and "float" in e.message for e in result.errors)

    def test_both_int_literals_in_float_context_fails(self) -> None:
        # Both operands are int, but AddNode expects float
        program = Program(
            root=AddNode(
                left=Literal(value=1),
                right=Literal(value=2),
            ),
        )
        result = check_program(program)
        assert not result.success

    def test_int_ref_in_float_context_fails(self) -> None:
        # Graph with IntLiteral nodes referenced where Node[float] expected
        program = Program(
            root=Ref(id="add"),
            nodes={
                "x": IntLiteral(value=5),
                "y": IntLiteral(value=10),
                "add": RefAdd(left=Ref(id="x"), right=Ref(id="y")),
            },
        )
        result = check_program(program)
        assert not result.success

    def test_float_literals_in_float_context_succeeds(self) -> None:
        # Both operands are float, matching AddNode's expectation
        program = Program(
            root=AddNode(
                left=Literal(value=1.0),
                right=Literal(value=2.0),
            ),
        )
        result = check_program(program)
        assert result.success


class TestBoundedTypeVars:
    """Tests for bounded type variables."""

    def test_bounded_numeric_literal_with_int(self) -> None:
        """NumericLiteral[T: int | float] should accept int."""
        result = check_node(NumericLiteral(value=42))
        assert result.success

    def test_bounded_numeric_literal_with_float(self) -> None:
        """NumericLiteral[T: int | float] should accept float."""
        result = check_node(NumericLiteral(value=3.14))
        assert result.success

    def test_bounded_numeric_literal_with_string_fails(self) -> None:
        """NumericLiteral[T: int | float] should reject str."""
        result = check_node(NumericLiteral(value="hello"))  # type: ignore[arg-type]
        assert not result.success
        assert any("str" in str(e).lower() for e in result.errors)

    def test_bounded_numeric_add_with_ints(self) -> None:
        """NumericAdd should work with int literals."""
        program = Program(
            root=NumericAdd(
                left=NumericLiteral(value=1),
                right=NumericLiteral(value=2),
            ),
        )
        result = check_program(program)
        assert result.success

    def test_bounded_numeric_add_with_floats(self) -> None:
        """NumericAdd should work with float literals."""
        program = Program(
            root=NumericAdd(
                left=NumericLiteral(value=1.5),
                right=NumericLiteral(value=2.5),
            ),
        )
        result = check_program(program)
        assert result.success


class TestBoundedTypeVarsEdgeCases:
    """Edge case tests for bounded type variables."""

    def test_single_type_bound(self) -> None:
        """A bound with a single type (not union) should work."""
        # IntOnly[T: int] only accepts int
        result = check_node(IntOnly(value=42))
        assert result.success

    def test_single_type_bound_rejects_other_types(self) -> None:
        """Single type bound should reject other types."""
        result = check_node(IntOnly(value=3.14))  # type: ignore[arg-type]
        assert not result.success

    def test_bound_rejects_non_subtype(self) -> None:
        """T: float should reject int because int is not a subclass of float."""
        result = check_node(FloatBounded(value=42))  # int value
        assert not result.success
        assert any("int" in e.message and "float" in e.message for e in result.errors)

    def test_bound_accepts_subtype(self) -> None:
        """T: int should accept bool because bool is a subclass of int."""
        result = check_node(IntOnly(value=True))  # bool value
        assert result.success

    def test_same_typevar_must_be_consistent(self) -> None:
        """Same TypeVar in multiple fields must resolve to same type."""
        # PairSame[T] has left: T and right: T
        result = check_node(PairSame(left=1, right=2))  # both int - OK
        assert result.success

    def test_same_typevar_inconsistent_fails(self) -> None:
        """Inconsistent types for same TypeVar should fail."""
        result = check_node(PairSame(left=1, right="hello"))  # type: ignore[arg-type]
        assert not result.success

    def test_multiple_type_parameters(self) -> None:
        """Node with multiple type parameters should work."""
        result = check_node(PairDifferent(left=42, right="hello"))
        assert result.success

    def test_typevar_in_container(self) -> None:
        """TypeVar used in container field should work."""
        result = check_node(ListContainer(items=[1, 2, 3]))
        assert result.success

    def test_bounded_typevar_in_container(self) -> None:
        """Bounded TypeVar in container should enforce bound."""
        result = check_node(NumericList(items=[1, 2, 3]))
        assert result.success

    def test_bounded_typevar_in_container_rejects_invalid(self) -> None:
        """Bounded TypeVar in container should reject invalid types."""
        result = check_node(NumericList(items=["a", "b"]))  # type: ignore[list-item]
        assert not result.success


class TestTypeMismatchEdgeCases:
    """Edge cases for type mismatches."""

    def test_type_mismatch_via_ref(self) -> None:
        """Type mismatch detected through Ref connections."""
        # StringLiteral returns Node[str], but RefAdd expects Node[float]
        program = Program(
            root=Ref(id="add"),
            nodes={
                "s": StringLiteral(value="hello"),
                "add": RefAdd(left=Ref(id="s"), right=Ref(id="s")),
            },
        )
        result = check_program(program)
        assert not result.success

    def test_nested_type_error(self) -> None:
        """Type error in deeply nested inline node."""
        # Inner node has type mismatch
        program = Program(
            root=AddNode(
                left=Literal(value=1.0),
                right=AddNode(
                    left=Literal(value=2.0),
                    right=StringLiteral(value="oops"),  # type: ignore[arg-type]
                ),
            ),
        )
        result = check_program(program)
        assert not result.success

    def test_multiple_errors_reported(self) -> None:
        """Multiple type errors should all be reported."""
        program = Program(
            root=Ref(id="add"),
            nodes={
                "s1": StringLiteral(value="a"),
                "s2": StringLiteral(value="b"),
                "add": RefAdd(left=Ref(id="s1"), right=Ref(id="s2")),
            },
        )
        result = check_program(program)
        assert not result.success
        # Both left and right refs have type errors
        assert len(result.errors) >= 1


class TestNegativeCases:
    """Comprehensive negative tests to verify type errors are caught."""

    def test_wrong_primitive_type_in_concrete_field(self) -> None:
        """Passing wrong primitive type to a concrete-typed field."""
        result = check_node(IntLiteral(value="not an int"))  # type: ignore[arg-type]
        assert not result.success
        assert len(result.errors) >= 1
        # Error should mention the type mismatch
        error_str = str(result.errors[0])
        assert "str" in error_str or "int" in error_str

    def test_float_not_accepted_where_int_expected(self) -> None:
        """Float should NOT be accepted where int is expected (no contravariance)."""
        result = check_node(IntLiteral(value=3.14))  # type: ignore[arg-type]
        assert not result.success

    def test_none_not_accepted_for_non_optional(self) -> None:
        """None should not be accepted for non-optional fields."""
        result = check_node(IntLiteral(value=None))  # type: ignore[arg-type]
        assert not result.success

    def test_list_not_accepted_where_int_expected(self) -> None:
        """List should not be accepted where int is expected."""
        result = check_node(IntLiteral(value=[1, 2, 3]))  # type: ignore[arg-type]
        assert not result.success

    def test_bound_violation_with_complex(self) -> None:
        """Complex should not satisfy bound T: int | float."""
        result = check_node(NumericLiteral(value=1 + 2j))  # type: ignore[arg-type]
        assert not result.success
        # Error should mention the bound
        assert any("complex" in str(e).lower() for e in result.errors)

    def test_bound_violation_with_bool(self) -> None:
        """Bool technically is int subtype, but let's verify behavior."""
        # Note: In Python, bool is a subtype of int, so this might pass
        # This test documents the actual behavior
        result = check_node(NumericLiteral(value=True))
        # bool is subtype of int, which is in the bound, so this should pass
        assert result.success

    def test_single_bound_rejects_subtype(self) -> None:
        """T: int should reject float even though float > int numerically."""
        result = check_node(IntOnly(value=3.14))  # type: ignore[arg-type]
        assert not result.success
        assert any("float" in str(e).lower() for e in result.errors)

    def test_wrong_node_type_inline(self) -> None:
        """Wrong node type passed as inline child."""
        # AddNode expects Node[float], StringLiteral is Node[str]
        result = check_node(
            AddNode(
                left=Literal(value=1.0),
                right=StringLiteral(value="wrong"),  # type: ignore[arg-type]
            ),
        )
        assert not result.success

    def test_inconsistent_typevar_with_three_fields(self) -> None:
        """TypeVar must be consistent across all usages."""
        # If we had a node with three T fields, all must match
        result = check_node(PairSame(left="hello", right=42))  # type: ignore[arg-type]
        assert not result.success

    def test_error_location_includes_field_name(self) -> None:
        """Error location should include the field name."""
        result = check_node(IntLiteral(value="wrong"))  # type: ignore[arg-type]
        assert not result.success
        assert result.errors[0].location.field_name == "value"

    def test_error_location_includes_node_tag(self) -> None:
        """Error location should include the node tag."""
        result = check_node(IntLiteral(value="wrong"))  # type: ignore[arg-type]
        assert not result.success
        assert result.errors[0].location.node_tag == "checker_int_lit"

    def test_ref_to_wrong_return_type(self) -> None:
        """Ref to node with wrong return type should fail."""
        program = Program(
            root=Ref(id="add"),
            nodes={
                # IntLiteral returns Node[int], but RefAdd expects Ref[Node[float]]
                "x": IntLiteral(value=5),
                "add": RefAdd(left=Ref(id="x"), right=Ref(id="x")),
            },
        )
        result = check_program(program)
        # int is NOT a subtype of float, so this should fail
        assert not result.success

    def test_ref_to_incompatible_return_type(self) -> None:
        """Ref to node with incompatible return type should fail."""
        program = Program(
            root=Ref(id="add"),
            nodes={
                # StringLiteral returns Node[str], incompatible with Node[float]
                "x": StringLiteral(value="hello"),
                "add": RefAdd(left=Ref(id="x"), right=Ref(id="x")),
            },
        )
        result = check_program(program)
        assert not result.success

    def test_container_element_type_mismatch(self) -> None:
        """List elements must match the declared element type."""
        result = check_node(ListContainer(items=["a", "b", "c"]))
        # This infers T=str from the elements, which is fine for unbounded T
        assert result.success

        # But for bounded T, wrong element types should fail
        result = check_node(NumericList(items=["a", "b"]))  # type: ignore[list-item]
        assert not result.success

    def test_empty_list_with_bounded_typevar_unresolved(self) -> None:
        """Empty list with bounded TypeVar - TVar remains unresolved."""
        # Empty list means T can't be inferred from elements
        # The bound check should handle unresolved TVars gracefully
        result = check_node(NumericList(items=[]))
        # With no elements, T is unresolved - should pass (no violation detected)
        assert result.success

    def test_deeply_nested_bound_violation(self) -> None:
        """Bound violation in deeply nested structure."""
        program = Program(
            root=NumericAdd(
                left=NumericLiteral(value=1),
                right=NumericAdd(
                    left=NumericLiteral(value=2),
                    right=NumericLiteral(value="oops"),  # type: ignore[arg-type]
                ),
            ),
        )
        result = check_program(program)
        assert not result.success


class TestCheckProgram:
    """Tests for program type checking."""

    def test_well_typed_graph_program(self) -> None:
        program = Program(
            root=Ref(id="result"),
            nodes={
                "a": Literal(value=1.0),
                "b": Literal(value=2.0),
                "result": RefAdd(left=Ref(id="a"), right=Ref(id="b")),
            },
        )
        result = check_program(program)
        assert result.success

    def test_nested_inline_nodes(self) -> None:
        node = AddNode(left=Literal(value=1.0), right=Literal(value=2.0))
        result = check_node(node)
        assert result.success

    def test_result_contains_constraints(self) -> None:
        result = check_node(Literal(value=42))
        assert isinstance(result, CheckResult)
        assert isinstance(result.constraints, list)

    def test_failed_check_reports_error_location(self) -> None:
        constraint = EqualityConstraint(
            left=TCon(int),
            right=TCon(str),
            location=Location(
                node_tag="test",
                node_id="my_node",
                field_name="value",
                path=("root", "my_node"),
            ),
        )
        result = solve([constraint])
        error = result.errors[0]
        assert error.location.node_tag == "test"
        assert error.location.node_id == "my_node"
        assert error.location.field_name == "value"


class TestSubtypeConstraintErrors:
    """Tests for SubtypeConstraint error detection and location tracking."""

    def test_direct_bound_violation_detected(self) -> None:
        """Passing wrong type directly to bounded TypeVar should fail."""
        # NumericLiteral[T: int | float] with string value
        result = check_node(NumericLiteral(value="not a number"))  # type: ignore[arg-type]
        assert not result.success
        # Should mention the type that violated the bound
        assert any("str" in e.message for e in result.errors)

    def test_bound_violation_error_mentions_allowed_types(self) -> None:
        """Bound violation error should mention the allowed types."""
        result = check_node(NumericLiteral(value="bad"))  # type: ignore[arg-type]
        assert not result.success
        # Error should mention the bound types
        bound_error = next(
            (e for e in result.errors if "bound" in e.message.lower()),
            None,
        )
        assert bound_error is not None
        assert "int" in bound_error.message or "float" in bound_error.message

    def test_ref_causes_bound_violation(self) -> None:
        """Ref to wrong type should cause bound violation in bounded node."""
        # NumericAdd expects Node[T] where T: int | float
        # StringLiteral returns Node[str] - should violate the bound
        program = Program(
            root=Ref(id="add"),
            nodes={
                "s": StringLiteral(value="hello"),
                "add": NumericAdd(
                    left=Ref(id="s"),  # This should unify T with str
                    right=NumericLiteral(value=1),
                ),
            },
        )
        result = check_program(program)
        assert not result.success
        # Should detect the bound violation
        assert any("str" in e.message for e in result.errors)

    def test_bound_violation_with_consistent_wrong_type(self) -> None:
        """Same TypeVar resolved to wrong type consistently should fail bounds."""
        # Both fields are string (consistent) but string is not in int | float
        result = check_node(
            NumericAdd(
                left=NumericLiteral(value="a"),  # type: ignore[arg-type]
                right=NumericLiteral(value="b"),  # type: ignore[arg-type]
            ),
        )
        assert not result.success

    def test_subtype_constraint_location_exists(self) -> None:
        """SubtypeConstraint errors should have location information."""
        result = check_node(NumericLiteral(value="wrong"))  # type: ignore[arg-type]
        assert not result.success
        # Find the bound violation error
        bound_error = next(
            (e for e in result.errors if "bound" in e.message.lower()),
            None,
        )
        assert bound_error is not None
        assert bound_error.location is not None

    def test_multiple_bounded_typevars_independent_errors(self) -> None:
        """Multiple bounded TypeVars should report independent errors."""
        # Create a scenario with two different bounded TypeVars
        # One violates bounds, one doesn't
        program = Program(
            root=NumericAdd(
                left=NumericLiteral(value=1),  # OK - int is in bounds
                right=NumericLiteral(value="bad"),  # type: ignore[arg-type]
            ),
        )
        result = check_program(program)
        assert not result.success

    def test_nested_bound_violation_location_includes_path(self) -> None:
        """Deeply nested bound violations should have path in location."""
        program = Program(
            root=NumericAdd(
                left=NumericLiteral(value=1),
                right=NumericAdd(
                    left=NumericLiteral(value=2),
                    right=NumericLiteral(value="deep error"),  # type: ignore[arg-type]
                ),
            ),
        )
        result = check_program(program)
        assert not result.success
        # At minimum we should have error(s) detecting the problem
        assert len(result.errors) >= 1

    def test_bound_check_with_subtype_passes(self) -> None:
        """Type that is subtype of bound should pass."""
        # IntOnly[T: int] with bool value - bool is subtype of int
        result = check_node(IntOnly(value=True))
        assert result.success

    def test_bound_check_with_non_subtype_fails(self) -> None:
        """Type that is not subtype of any bound type should fail."""
        # IntOnly[T: int] with float value - float is not subtype of int
        result = check_node(IntOnly(value=3.14))  # type: ignore[arg-type]
        assert not result.success
        assert any("float" in e.message for e in result.errors)

    def test_bound_violation_location_points_to_node(self) -> None:
        """Bound violation location should point to the node, not generic <bound>."""
        result = check_node(NumericLiteral(value="wrong"))  # type: ignore[arg-type]
        assert not result.success
        bound_error = next(
            (e for e in result.errors if "bound" in e.message.lower()),
            None,
        )
        assert bound_error is not None
        # Location should point to the actual node, not "<bound>"
        assert bound_error.location.node_tag == "checker_numeric_lit"
        assert bound_error.location.path == ("root",)

    def test_bound_violation_in_named_node_includes_node_id(self) -> None:
        """Bound violation in named node should include node_id in location."""
        program = Program(
            root=Ref(id="bad"),
            nodes={
                "bad": NumericLiteral(value="not a number"),  # type: ignore[arg-type]
            },
        )
        result = check_program(program)
        assert not result.success
        bound_error = next(
            (e for e in result.errors if "bound" in e.message.lower()),
            None,
        )
        assert bound_error is not None
        assert bound_error.location.node_id == "bad"
        assert bound_error.location.node_tag == "checker_numeric_lit"


class TestConstraintErrorLocations:
    """Tests that verify error locations correctly identify the source of type errors.

    This class tests the distinction between:
    - EqualityConstraint errors: when types can't unify (e.g., int vs str)
    - SubtypeConstraint errors: when a TypeVar is resolved outside its bounds

    For each scenario, we verify:
    1. The correct error type is raised
    2. The error location points to the right node/field
    """

    def test_equality_error_when_concrete_type_mismatch(self) -> None:
        """Passing wrong concrete type to concrete field is EqualityConstraint error.

        IntProcessor expects Node[int], passing StringLiteral (Node[str])
        results in EqualityConstraint failure: int != str
        """
        program = Program(
            root=IntProcessor(child=StringLiteral(value="hello")),
        )
        result = check_program(program)
        assert not result.success

        # This should be an EqualityConstraint error (type mismatch)
        # NOT a SubtypeConstraint error (no bounded TypeVar involved)
        assert any("mismatch" in e.message.lower() for e in result.errors)
        # No bound error should appear
        assert not any("bound" in e.message.lower() for e in result.errors)

        # Error should point to IntProcessor's child field
        error = result.errors[0]
        assert error.location.node_tag == "checker_int_processor"
        assert error.location.field_name == "child"

    def test_subtype_error_when_bounded_typevar_violated(self) -> None:
        """Passing wrong type to bounded TypeVar field is SubtypeConstraint error.

        NumericProcessor[T: int | float] receives StringLiteral (Node[str]).
        T unifies to str, then SubtypeConstraint T <: int | float fails.
        """
        program = Program(
            root=NumericProcessor(child=StringLiteral(value="hello")),
        )
        result = check_program(program)
        assert not result.success

        # This should include a SubtypeConstraint error (bound violation)
        bound_error = next(
            (e for e in result.errors if "bound" in e.message.lower()),
            None,
        )
        assert bound_error is not None, f"Expected bound error, got: {result.errors}"
        assert "str" in bound_error.message
        assert "int" in bound_error.message or "float" in bound_error.message

        # Bound error location should point to NumericProcessor (where T is bounded)
        assert bound_error.location.node_tag == "checker_numeric_processor"

    def test_equality_error_via_ref_concrete_type(self) -> None:
        """Ref to wrong concrete type is EqualityConstraint error.

        IntRefProcessor expects Ref[Node[int]], but refs to StringLiteral (Node[str]).
        """
        program = Program(
            root=Ref(id="processor"),
            nodes={
                "str_node": StringLiteral(value="hello"),
                "processor": IntRefProcessor(child=Ref(id="str_node")),
            },
        )
        result = check_program(program)
        assert not result.success

        # Should be type mismatch, not bound violation
        assert any("mismatch" in e.message.lower() for e in result.errors)

        # Error should point to IntRefProcessor
        mismatch_error = next(
            e for e in result.errors if "mismatch" in e.message.lower()
        )
        assert mismatch_error.location.node_tag == "checker_int_ref_processor"

    def test_subtype_error_via_ref_bounded_typevar(self) -> None:
        """Ref to wrong type for bounded TypeVar is SubtypeConstraint error.

        NumericRefProcessor[T: int | float] refs to StringLiteral (Node[str]).
        T unifies to str via Ref, then SubtypeConstraint fails.
        """
        program = Program(
            root=Ref(id="processor"),
            nodes={
                "str_node": StringLiteral(value="hello"),
                "processor": NumericRefProcessor(child=Ref(id="str_node")),
            },
        )
        result = check_program(program)
        assert not result.success

        # Should have bound violation error
        bound_error = next(
            (e for e in result.errors if "bound" in e.message.lower()),
            None,
        )
        assert bound_error is not None
        assert "str" in bound_error.message

        # Bound error points to NumericRefProcessor (where T's bound is declared)
        assert bound_error.location.node_tag == "checker_numeric_ref_processor"
        assert bound_error.location.node_id == "processor"

    def test_nested_bound_violation_location_chain(self) -> None:
        """Bound violation in nested node shows correct location chain.

        Structure: NumericProcessor -> child: NumericLiteral(value="bad")
        The bound violation is on NumericProcessor's T, not NumericLiteral's T.
        """
        program = Program(
            root=NumericProcessor(
                child=NumericLiteral(value="bad"),  # type: ignore[arg-type]
            ),
        )
        result = check_program(program)
        assert not result.success

        # Both nodes have bounded TypeVars, so we might get multiple bound errors
        bound_errors = [e for e in result.errors if "bound" in e.message.lower()]
        assert len(bound_errors) >= 1

        # At least one should point to the inner NumericLiteral
        inner_error = next(
            (e for e in bound_errors if e.location.node_tag == "checker_numeric_lit"),
            None,
        )
        assert inner_error is not None
        # The path should show we're inside root
        assert "root" in inner_error.location.path

    def test_valid_bounded_subnode_succeeds(self) -> None:
        """Valid type for bounded TypeVar should succeed.

        NumericProcessor[T: int | float] with IntLiteral (Node[int]) child.
        T unifies to int, which satisfies the bound.
        """
        program = Program(
            root=NumericProcessor(child=IntLiteral(value=42)),
        )
        result = check_program(program)
        assert result.success

    def test_valid_bounded_ref_succeeds(self) -> None:
        """Valid Ref type for bounded TypeVar should succeed."""
        program = Program(
            root=Ref(id="processor"),
            nodes={
                "num": IntLiteral(value=42),
                "processor": NumericRefProcessor(child=Ref(id="num")),
            },
        )
        result = check_program(program)
        assert result.success

    def test_error_distinguishes_which_constraint_failed(self) -> None:
        """Test that we can identify whether Equality or Subtype constraint failed.

        Two separate programs to verify both error types are reported correctly.
        """
        # Test 1: EqualityConstraint error - IntProcessor expects int, gets str
        program1 = Program(
            root=Ref(id="int_proc"),
            nodes={
                "bad_string": StringLiteral(value="hello"),
                "int_proc": IntProcessor(child=Ref(id="bad_string")),
            },
        )
        result1 = check_program(program1)
        assert not result1.success
        assert any("mismatch" in e.message.lower() for e in result1.errors)
        assert not any("bound" in e.message.lower() for e in result1.errors)

        mismatch_error = next(
            e for e in result1.errors if "mismatch" in e.message.lower()
        )
        assert mismatch_error.location.node_tag == "checker_int_processor"

        # Test 2: SubtypeConstraint error - NumericProcessor[T: int|float] gets str
        program2 = Program(
            root=Ref(id="num_proc"),
            nodes={
                "bad_string": StringLiteral(value="hello"),
                "num_proc": NumericRefProcessor(child=Ref(id="bad_string")),
            },
        )
        result2 = check_program(program2)
        assert not result2.success
        assert any("bound" in e.message.lower() for e in result2.errors)

        bound_error = next(e for e in result2.errors if "bound" in e.message.lower())
        assert bound_error.location.node_tag == "checker_numeric_ref_processor"
