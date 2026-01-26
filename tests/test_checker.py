"""Tests for the type checker module."""

from typedsl import Node, Program, Ref
from typedsl.checker import (
    CheckResult,
    Constraint,
    Location,
    Substitution,
    TCon,
    TVar,
    check_node,
    check_program,
    from_hint,
    solve,
)
from typedsl.checker.solver import unify


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
        # T0 = list[T0] would be infinite
        result = unify(TVar(0), TCon(list, (TVar(0),)))
        assert isinstance(result, str)
        assert "infinite" in result.lower()


class TestSolver:
    """Tests for constraint solving."""

    def test_empty_constraints_succeed(self) -> None:
        result = solve([])
        assert result.success

    def test_satisfiable_constraints_succeed(self) -> None:
        constraint = Constraint(
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
        constraint = Constraint(
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
class Const(Node[int], tag="checker_const"):
    value: int


class FloatConst(Node[float], tag="checker_float_const"):
    value: float


class Add(Node[int], tag="checker_add"):
    left: Ref[Node[int]]
    right: Ref[Node[int]]


class InlineAdd(Node[int], tag="checker_inline_add"):
    left: Node[int]
    right: Node[int]


class ListNode(Node[list[int]], tag="checker_list_node"):
    items: list[int]


class DictNode(Node[dict[str, int]], tag="checker_dict_node"):
    mapping: dict[str, int]


class TestCheckProgram:
    """Tests for program type checking."""

    def test_well_typed_graph_program(self) -> None:
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "a": Const(value=1),
                "b": Const(value=2),
                "result": Add(left=Ref(id="a"), right=Ref(id="b")),
            },
        )
        result = check_program(prog)
        assert result.success, f"Expected success: {result}"

    def test_simple_root_node(self) -> None:
        prog = Program(root=Const(value=42))
        result = check_program(prog)
        assert result.success

    def test_nested_inline_nodes(self) -> None:
        node = InlineAdd(left=Const(value=1), right=Const(value=2))
        result = check_node(node)
        assert result.success

    def test_list_fields(self) -> None:
        result = check_node(ListNode(items=[1, 2, 3]))
        assert result.success

    def test_empty_list_fields(self) -> None:
        result = check_node(ListNode(items=[]))
        assert result.success

    def test_dict_fields(self) -> None:
        result = check_node(DictNode(mapping={"a": 1}))
        assert result.success

    def test_empty_dict_fields(self) -> None:
        result = check_node(DictNode(mapping={}))
        assert result.success

    def test_result_contains_constraints(self) -> None:
        result = check_node(Const(value=42))
        assert isinstance(result, CheckResult)
        assert isinstance(result.constraints, list)

    def test_failed_check_reports_error_location(self) -> None:
        constraint = Constraint(
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
