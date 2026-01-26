"""Tests for the type checker module."""

from typedsl import Node, Program, Ref
from typedsl.checker import (
    CheckResult,
    Constraint,
    Substitution,
    TCon,
    TVar,
    TVarFactory,
    check_node,
    check_program,
    from_hint,
    solve,
    texpr_to_str,
)
from typedsl.checker.solver import TypeCheckError, unify


class TestTCon:
    """Tests for TCon type constructor."""

    def test_simple_type(self) -> None:
        t = TCon(int)
        assert t.con is int
        assert t.args == ()

    def test_generic_type(self) -> None:
        t = TCon(list, (TCon(int),))
        assert t.con is list
        assert len(t.args) == 1
        assert t.args[0] == TCon(int)

    def test_nested_generic(self) -> None:
        t = TCon(dict, (TCon(str), TCon(list, (TCon(int),))))
        assert t.con is dict
        assert len(t.args) == 2

    def test_repr(self) -> None:
        assert "TCon(int)" in repr(TCon(int))
        assert "TCon(list" in repr(TCon(list, (TCon(int),)))


class TestTVar:
    """Tests for TVar type variable."""

    def test_creation(self) -> None:
        t = TVar(0)
        assert t.id == 0

    def test_equality(self) -> None:
        assert TVar(0) == TVar(0)
        assert TVar(0) != TVar(1)

    def test_repr(self) -> None:
        assert "TVar(0)" in repr(TVar(0))


class TestTVarFactory:
    """Tests for TVarFactory."""

    def test_fresh_increments(self) -> None:
        factory = TVarFactory()
        t0 = factory.fresh()
        t1 = factory.fresh()
        t2 = factory.fresh()
        assert t0.id == 0
        assert t1.id == 1
        assert t2.id == 2

    def test_start_value(self) -> None:
        factory = TVarFactory(start=10)
        t = factory.fresh()
        assert t.id == 10


class TestFromHint:
    """Tests for from_hint conversion."""

    def test_simple_int(self) -> None:
        result = from_hint(int)
        assert result == TCon(int)

    def test_simple_str(self) -> None:
        result = from_hint(str)
        assert result == TCon(str)

    def test_list_int(self) -> None:
        result = from_hint(list[int])
        assert result == TCon(list, (TCon(int),))

    def test_dict_str_int(self) -> None:
        result = from_hint(dict[str, int])
        assert result == TCon(dict, (TCon(str), TCon(int)))

    def test_nested_generic(self) -> None:
        result = from_hint(list[dict[str, int]])
        expected = TCon(list, (TCon(dict, (TCon(str), TCon(int))),))
        assert result == expected


class TestTexprToStr:
    """Tests for texpr_to_str."""

    def test_simple_type(self) -> None:
        assert texpr_to_str(TCon(int)) == "int"
        assert texpr_to_str(TCon(str)) == "str"

    def test_type_var(self) -> None:
        assert texpr_to_str(TVar(0)) == "?T0"
        assert texpr_to_str(TVar(5)) == "?T5"

    def test_generic_type(self) -> None:
        assert texpr_to_str(TCon(list, (TCon(int),))) == "list[int]"

    def test_none_type(self) -> None:
        assert texpr_to_str(TCon(type(None))) == "None"


class TestSubstitution:
    """Tests for Substitution."""

    def test_empty_substitution(self) -> None:
        sub = Substitution()
        assert TVar(0).id not in sub
        result = sub.apply(TVar(0))
        assert result == TVar(0)

    def test_bind_and_apply(self) -> None:
        sub = Substitution()
        sub.bind(0, TCon(int))
        result = sub.apply(TVar(0))
        assert result == TCon(int)

    def test_apply_to_tcon(self) -> None:
        sub = Substitution()
        sub.bind(0, TCon(int))
        result = sub.apply(TCon(list, (TVar(0),)))
        assert result == TCon(list, (TCon(int),))

    def test_chained_application(self) -> None:
        sub = Substitution()
        sub.bind(0, TVar(1))
        sub.bind(1, TCon(int))
        result = sub.apply(TVar(0))
        assert result == TCon(int)

    def test_compose(self) -> None:
        sub1 = Substitution()
        sub1.bind(0, TCon(int))
        sub2 = Substitution()
        sub2.bind(1, TVar(0))
        composed = sub1.compose(sub2)
        # sub2[1] = TVar(0), after applying sub1: TVar(0) -> TCon(int)
        assert composed.apply(TVar(1)) == TCon(int)


class TestUnify:
    """Tests for unification."""

    def test_identical_types(self) -> None:
        result = unify(TCon(int), TCon(int))
        assert isinstance(result, Substitution)

    def test_different_types_fail(self) -> None:
        result = unify(TCon(int), TCon(str))
        assert isinstance(result, str)
        assert "mismatch" in result.lower()

    def test_tvar_unifies_with_tcon(self) -> None:
        result = unify(TVar(0), TCon(int))
        assert isinstance(result, Substitution)
        assert result.apply(TVar(0)) == TCon(int)

    def test_tcon_unifies_with_tvar(self) -> None:
        result = unify(TCon(int), TVar(0))
        assert isinstance(result, Substitution)
        assert result.apply(TVar(0)) == TCon(int)

    def test_generic_unification(self) -> None:
        result = unify(
            TCon(list, (TVar(0),)),
            TCon(list, (TCon(int),)),
        )
        assert isinstance(result, Substitution)
        assert result.apply(TVar(0)) == TCon(int)

    def test_arity_mismatch(self) -> None:
        result = unify(
            TCon(list, (TCon(int),)),
            TCon(list, ()),
        )
        assert isinstance(result, str)
        assert "arity" in result.lower()


class TestSolver:
    """Tests for constraint solving."""

    def test_empty_constraints(self) -> None:
        result = solve([])
        assert result.success
        assert len(result.errors) == 0

    def test_single_satisfiable_constraint(self) -> None:
        from typedsl.checker.constraints import Location

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

    def test_single_unsatisfiable_constraint(self) -> None:
        from typedsl.checker.constraints import Location

        constraint = Constraint(
            left=TCon(int),
            right=TCon(str),
            location=Location(
                node_tag="test",
                node_id=None,
                field_name="value",
                path=("root",),
            ),
        )
        result = solve([constraint])
        assert not result.success
        assert len(result.errors) == 1


# Define test node classes for type checker tests
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


class MixedTypeAdd(Node[int], tag="checker_mixed_add"):
    left: Ref[Node[int]]
    right: Ref[Node[float]]  # Type mismatch!


class TestCheckProgram:
    """Tests for check_program."""

    def test_well_typed_program(self) -> None:
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "a": Const(value=1),
                "b": Const(value=2),
                "result": Add(left=Ref(id="a"), right=Ref(id="b")),
            },
        )
        result = check_program(prog)
        assert result.success, f"Expected success but got: {result}"

    def test_simple_node(self) -> None:
        prog = Program(root=Const(value=42))
        result = check_program(prog)
        assert result.success

    def test_result_is_check_result(self) -> None:
        prog = Program(root=Const(value=42))
        result = check_program(prog)
        assert isinstance(result, CheckResult)
        assert hasattr(result, "success")
        assert hasattr(result, "errors")
        assert hasattr(result, "substitution")
        assert hasattr(result, "constraints")


class TestCheckNode:
    """Tests for check_node."""

    def test_simple_node(self) -> None:
        node = Const(value=42)
        result = check_node(node)
        assert result.success

    def test_nested_inline_nodes(self) -> None:
        node = InlineAdd(
            left=Const(value=1),
            right=Const(value=2),
        )
        result = check_node(node)
        assert result.success


class ListNode(Node[list[int]], tag="checker_list_node"):
    items: list[int]


class DictNode(Node[dict[str, int]], tag="checker_dict_node"):
    mapping: dict[str, int]


class TestContainerTypes:
    """Tests for container type checking."""

    def test_list_type(self) -> None:
        node = ListNode(items=[1, 2, 3])
        result = check_node(node)
        assert result.success

    def test_empty_list(self) -> None:
        node = ListNode(items=[])
        result = check_node(node)
        assert result.success

    def test_dict_type(self) -> None:
        node = DictNode(mapping={"a": 1, "b": 2})
        result = check_node(node)
        assert result.success

    def test_empty_dict(self) -> None:
        node = DictNode(mapping={})
        result = check_node(node)
        assert result.success


class TestErrorReporting:
    """Tests for error reporting."""

    def test_error_has_location(self) -> None:
        from typedsl.checker.constraints import Location

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
        assert not result.success
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.location.node_tag == "test"
        assert error.location.node_id == "my_node"
        assert error.location.field_name == "value"

    def test_check_result_str(self) -> None:
        prog = Program(root=Const(value=42))
        result = check_program(prog)
        assert "passed" in str(result).lower()


class TestTypeCheckError:
    """Tests for TypeCheckError."""

    def test_error_message(self) -> None:
        from typedsl.checker.constraints import Location

        error = TypeCheckError(
            message="Type mismatch",
            location=Location(
                node_tag="test",
                node_id=None,
                field_name="value",
                path=("root",),
            ),
            expected=TCon(int),
            actual=TCon(str),
        )
        error_str = str(error)
        assert "Type mismatch" in error_str
        assert "int" in error_str
        assert "str" in error_str
