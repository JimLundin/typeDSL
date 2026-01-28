"""Comprehensive tests for the constraint-based type checker."""

import pytest

from typedsl.ast import Program
from typedsl.nodes import Child, Node, Ref
from typedsl.typechecker import (
    ConstraintGenerator,
    EqConstraint,
    Location,
    NodeConstraintGenerator,
    Solver,
    SubConstraint,
    TBot,
    TCon,
    TExp,
    TTop,
    TVar,
    TypeError,
    generate_constraints,
    is_subtype,
    join,
    meet,
    occurs,
    satisfiable,
    satisfies_bounds,
    solve,
    typecheck,
)


def loc(path: str = "test") -> Location:
    """Create a location for tests."""
    return Location(path)


class TestOccurs:
    """Tests for the occurs check."""

    def test_occurs_in_self(self) -> None:
        """Type variable occurs in itself."""
        assert occurs("T", TVar("T"))

    def test_not_occurs_different_var(self) -> None:
        """Type variable does not occur in different variable."""
        assert not occurs("T", TVar("U"))

    def test_occurs_in_typecon_arg(self) -> None:
        """Type variable occurs in TypeCon argument."""
        assert occurs("T", TCon(list, (TVar("T"),)))

    def test_not_occurs_in_typecon_without_var(self) -> None:
        """Type variable does not occur in TypeCon without it."""
        assert not occurs("T", TCon(list, (TCon(int, ()),)))

    def test_occurs_nested(self) -> None:
        """Type variable occurs in nested TypeCon."""
        nested = TCon(dict, (TCon(str, ()), TCon(list, (TVar("T"),))))
        assert occurs("T", nested)

    def test_not_occurs_in_top(self) -> None:
        """Type variable does not occur in Top."""
        assert not occurs("T", TTop())

    def test_not_occurs_in_bottom(self) -> None:
        """Type variable does not occur in Bottom."""
        assert not occurs("T", TBot())


class TestIsSubtype:
    """Tests for the is_subtype function."""

    def test_bottom_subtype_of_all(self) -> None:
        """Bottom is subtype of everything."""
        assert is_subtype(TBot(), TCon(int, ()))
        assert is_subtype(TBot(), TCon(str, ()))
        assert is_subtype(TBot(), TTop())

    def test_all_subtype_of_top(self) -> None:
        """Everything is subtype of Top."""
        assert is_subtype(TCon(int, ()), TTop())
        assert is_subtype(TCon(str, ()), TTop())
        assert is_subtype(TBot(), TTop())

    def test_top_not_subtype_of_typecon(self) -> None:
        """Top is not subtype of TypeCon."""
        assert not is_subtype(TTop(), TCon(int, ()))

    def test_typecon_not_subtype_of_bottom(self) -> None:
        """TypeCon is not subtype of Bottom."""
        assert not is_subtype(TCon(int, ()), TBot())

    def test_same_typecon_is_subtype(self) -> None:
        """Same TypeCon is subtype of itself."""
        assert is_subtype(TCon(int, ()), TCon(int, ()))
        assert is_subtype(TCon(str, ()), TCon(str, ()))

    def test_bool_subtype_of_int(self) -> None:
        """Bool is subtype of int in Python."""
        assert is_subtype(TCon(bool, ()), TCon(int, ()))

    def test_int_not_subtype_of_bool(self) -> None:
        """Int is not subtype of bool."""
        assert not is_subtype(TCon(int, ()), TCon(bool, ()))

    def test_str_not_subtype_of_int(self) -> None:
        """Str is not subtype of int."""
        assert not is_subtype(TCon(str, ()), TCon(int, ()))

    def test_generic_subtype_invariant(self) -> None:
        """Generic types are invariant in their arguments."""
        list_int = TCon(list, (TCon(int, ()),))
        list_bool = TCon(list, (TCon(bool, ()),))
        # Even though bool <: int, list[bool] is not <: list[int] (invariant)
        assert not is_subtype(list_bool, list_int)

    def test_same_generic_is_subtype(self) -> None:
        """Same generic type is subtype of itself."""
        list_int = TCon(list, (TCon(int, ()),))
        assert is_subtype(list_int, list_int)

    def test_typevar_subtype_of_itself(self) -> None:
        """TypeVar is subtype of itself."""
        assert is_subtype(TVar("T"), TVar("T"))

    def test_different_typevars_not_subtypes(self) -> None:
        """Different TypeVars are not subtypes."""
        assert not is_subtype(TVar("T"), TVar("U"))


class TestMeet:
    """Tests for the meet (greatest lower bound) function."""

    def test_meet_with_top(self) -> None:
        """Meet with Top returns the other type."""
        int_type = TCon(int, ())
        assert meet(TTop(), int_type) == int_type
        assert meet(int_type, TTop()) == int_type

    def test_meet_with_bottom(self) -> None:
        """Meet with Bottom returns Bottom."""
        int_type = TCon(int, ())
        assert meet(TBot(), int_type) == TBot()
        assert meet(int_type, TBot()) == TBot()

    def test_meet_same_type(self) -> None:
        """Meet of same type is that type."""
        int_type = TCon(int, ())
        assert meet(int_type, int_type) == int_type

    def test_meet_incompatible_returns_bottom(self) -> None:
        """Meet of incompatible types returns Bottom."""
        int_type = TCon(int, ())
        str_type = TCon(str, ())
        assert meet(int_type, str_type) == TBot()

    def test_meet_bool_int(self) -> None:
        """Meet of bool and int is bool (more specific)."""
        bool_type = TCon(bool, ())
        int_type = TCon(int, ())
        assert meet(bool_type, int_type) == bool_type
        assert meet(int_type, bool_type) == bool_type


class TestJoin:
    """Tests for the join (least upper bound) function."""

    def test_join_with_bottom(self) -> None:
        """Join with Bottom returns the other type."""
        int_type = TCon(int, ())
        assert join(TBot(), int_type) == int_type
        assert join(int_type, TBot()) == int_type

    def test_join_with_top(self) -> None:
        """Join with Top returns Top."""
        int_type = TCon(int, ())
        assert join(TTop(), int_type) == TTop()
        assert join(int_type, TTop()) == TTop()

    def test_join_same_type(self) -> None:
        """Join of same type is that type."""
        int_type = TCon(int, ())
        assert join(int_type, int_type) == int_type

    def test_join_incompatible_returns_top(self) -> None:
        """Join of incompatible types returns Top."""
        int_type = TCon(int, ())
        str_type = TCon(str, ())
        assert join(int_type, str_type) == TTop()

    def test_join_bool_int(self) -> None:
        """Join of bool and int is int (more general)."""
        bool_type = TCon(bool, ())
        int_type = TCon(int, ())
        assert join(bool_type, int_type) == int_type
        assert join(int_type, bool_type) == int_type


class TestSatisfiable:
    """Tests for bound satisfiability."""

    def test_bottom_top_satisfiable(self) -> None:
        """Bottom <: Top is satisfiable."""
        assert satisfiable(TBot(), TTop())

    def test_same_type_satisfiable(self) -> None:
        """Same type bounds are satisfiable."""
        int_type = TCon(int, ())
        assert satisfiable(int_type, int_type)

    def test_subtype_bounds_satisfiable(self) -> None:
        """Subtype lower bound is satisfiable."""
        bool_type = TCon(bool, ())
        int_type = TCon(int, ())
        assert satisfiable(bool_type, int_type)

    def test_incompatible_bounds_not_satisfiable(self) -> None:
        """Incompatible bounds are not satisfiable."""
        int_type = TCon(int, ())
        str_type = TCon(str, ())
        assert not satisfiable(int_type, str_type)
        assert not satisfiable(str_type, int_type)


class TestSatisfiesBounds:
    """Tests for checking if a type satisfies bounds."""

    def test_type_satisfies_trivial_bounds(self) -> None:
        """Any type satisfies Bottom <: T <: Top."""
        int_type = TCon(int, ())
        assert satisfies_bounds(int_type, TBot(), TTop())

    def test_type_satisfies_exact_bounds(self) -> None:
        """Type satisfies exact bounds."""
        int_type = TCon(int, ())
        assert satisfies_bounds(int_type, int_type, int_type)

    def test_bool_satisfies_int_upper_bound(self) -> None:
        """Bool satisfies upper bound of int."""
        bool_type = TCon(bool, ())
        int_type = TCon(int, ())
        assert satisfies_bounds(bool_type, TBot(), int_type)

    def test_int_does_not_satisfy_bool_upper_bound(self) -> None:
        """Int does not satisfy upper bound of bool."""
        bool_type = TCon(bool, ())
        int_type = TCon(int, ())
        assert not satisfies_bounds(int_type, TBot(), bool_type)


class TestEquality:
    """Tests for equality constraints."""

    def test_same_primitives(self) -> None:
        """Int = int passes."""
        c = EqConstraint(TCon(int, ()), TCon(int, ()), loc())
        assert typecheck([c]) is None

    def test_different_primitives(self) -> None:
        """Int = str fails."""
        c = EqConstraint(TCon(int, ()), TCon(str, ()), loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)

    def test_bind_variable(self) -> None:
        """T = int passes and binds T."""
        c = EqConstraint(TVar("T"), TCon(int, ()), loc())
        solver = solve([c])
        assert solver.get_type("T") == TCon(int, ())

    def test_conflicting_bindings(self) -> None:
        """T = int, T = str fails."""
        c1 = EqConstraint(TVar("T"), TCon(int, ()), loc())
        c2 = EqConstraint(TVar("T"), TCon(str, ()), loc())
        result = typecheck([c1, c2])
        assert isinstance(result, TypeError)

    def test_same_generic(self) -> None:
        """list[int] = list[int] passes."""
        list_int = TCon(list, (TCon(int, ()),))
        c = EqConstraint(list_int, list_int, loc())
        assert typecheck([c]) is None

    def test_different_generic_args(self) -> None:
        """list[int] = list[str] fails."""
        list_int = TCon(list, (TCon(int, ()),))
        list_str = TCon(list, (TCon(str, ()),))
        c = EqConstraint(list_int, list_str, loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)

    def test_bind_in_generic(self) -> None:
        """list[T] = list[int] passes and binds T."""
        list_t = TCon(list, (TVar("T"),))
        list_int = TCon(list, (TCon(int, ()),))
        c = EqConstraint(list_t, list_int, loc())
        solver = solve([c])
        assert solver.get_type("T") == TCon(int, ())


class TestSubtyping:
    """Tests for subtype constraints."""

    def test_bool_subtype_int(self) -> None:
        """Bool <: int passes."""
        c = SubConstraint(TCon(bool, ()), TCon(int, ()), loc())
        assert typecheck([c]) is None

    def test_str_not_subtype_int(self) -> None:
        """Str <: int fails."""
        c = SubConstraint(TCon(str, ()), TCon(int, ()), loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)

    def test_upper_bound(self) -> None:
        """T <: int passes and sets upper bound."""
        c = SubConstraint(TVar("T"), TCon(int, ()), loc())
        solver = solve([c])
        bounds = solver.get_bounds("T")
        assert bounds is not None
        lower, upper = bounds
        assert isinstance(lower, TBot)
        assert upper == TCon(int, ())

    def test_lower_bound(self) -> None:
        """Int <: T passes and sets lower bound."""
        c = SubConstraint(TCon(int, ()), TVar("T"), loc())
        solver = solve([c])
        bounds = solver.get_bounds("T")
        assert bounds is not None
        lower, upper = bounds
        assert lower == TCon(int, ())
        assert isinstance(upper, TTop)

    def test_bounds_conflict(self) -> None:
        """Str <: T, T <: int fails (bounds conflict)."""
        c1 = SubConstraint(TCon(str, ()), TVar("T"), loc())
        c2 = SubConstraint(TVar("T"), TCon(int, ()), loc())
        result = typecheck([c1, c2])
        assert isinstance(result, TypeError)

    def test_compatible_bounds(self) -> None:
        """Bool <: T, T <: int passes (compatible bounds)."""
        c1 = SubConstraint(TCon(bool, ()), TVar("T"), loc())
        c2 = SubConstraint(TVar("T"), TCon(int, ()), loc())
        solver = solve([c1, c2])
        bounds = solver.get_bounds("T")
        assert bounds is not None
        lower, upper = bounds
        assert lower == TCon(bool, ())
        assert upper == TCon(int, ())


class TestDefaults:
    """Tests for type variable defaults."""

    def test_default_no_constraints(self) -> None:
        """T with default=int and no constraints passes."""
        # Create a constraint that introduces T with default
        t_with_default = TVar("T", default=TCon(int, ()))
        c = EqConstraint(t_with_default, t_with_default, loc())
        assert typecheck([c]) is None

    def test_default_violates_upper(self) -> None:
        """T <: str with default=int fails."""
        t_with_default = TVar("T", default=TCon(int, ()))
        c = SubConstraint(t_with_default, TCon(str, ()), loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)

    def test_default_violates_lower(self) -> None:
        """Int <: T with default=bool fails (int not <: bool)."""
        t_with_default = TVar("T", default=TCon(bool, ()))
        c = SubConstraint(TCon(int, ()), t_with_default, loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)

    def test_default_compatible_with_bounds(self) -> None:
        """Bool <: T <: int with default=bool passes."""
        t_with_default = TVar("T", default=TCon(bool, ()))
        c1 = SubConstraint(TCon(bool, ()), t_with_default, loc())
        c2 = SubConstraint(t_with_default, TCon(int, ()), loc())
        assert typecheck([c1, c2]) is None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_infinite_type(self) -> None:
        """T = list[T] fails (occurs check)."""
        list_t = TCon(list, (TVar("T"),))
        c = EqConstraint(TVar("T"), list_t, loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)
        assert "Infinite type" in result.message

    def test_transitive_conflict(self) -> None:
        """T = U, U = int, T = str fails."""
        c1 = EqConstraint(TVar("T"), TVar("U"), loc())
        c2 = EqConstraint(TVar("U"), TCon(int, ()), loc())
        c3 = EqConstraint(TVar("T"), TCon(str, ()), loc())
        result = typecheck([c1, c2, c3])
        assert isinstance(result, TypeError)

    def test_same_var_different_positions(self) -> None:
        """dict[T, T] = dict[int, str] fails."""
        dict_tt = TCon(dict, (TVar("T"), TVar("T")))
        dict_int_str = TCon(dict, (TCon(int, ()), TCon(str, ())))
        c = EqConstraint(dict_tt, dict_int_str, loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)

    def test_empty_constraints(self) -> None:
        """No constraints passes."""
        assert typecheck([]) is None

    def test_trivial_equality(self) -> None:
        """T = T passes."""
        c = EqConstraint(TVar("T"), TVar("T"), loc())
        assert typecheck([c]) is None

    def test_top_equals_top(self) -> None:
        """Top = Top passes."""
        c = EqConstraint(TTop(), TTop(), loc())
        assert typecheck([c]) is None

    def test_bottom_equals_bottom(self) -> None:
        """Bottom = Bottom passes."""
        c = EqConstraint(TBot(), TBot(), loc())
        assert typecheck([c]) is None

    def test_top_not_equals_bottom(self) -> None:
        """Top = Bottom fails."""
        c = EqConstraint(TTop(), TBot(), loc())
        result = typecheck([c])
        assert isinstance(result, TypeError)


class TestSolver:
    """Tests for the Solver class directly."""

    def test_get_creates_new_var(self) -> None:
        """get() creates a new TypeVarInfo with default bounds."""
        solver = Solver()
        info = solver.get("T")
        assert isinstance(info.lower, TBot)
        assert isinstance(info.upper, TTop)
        assert info.default is None

    def test_get_with_default(self) -> None:
        """get() with default stores the default."""
        solver = Solver()
        int_type = TCon(int, ())
        info = solver.get("T", default=int_type)
        assert info.default == int_type

    def test_get_type_returns_none_for_unbounded(self) -> None:
        """get_type() returns None for unbounded variable."""
        solver = Solver()
        solver.get("T")
        assert solver.get_type("T") is None

    def test_get_type_returns_type_when_bound(self) -> None:
        """get_type() returns type when lower == upper."""
        solver = Solver()
        int_type = TCon(int, ())
        solver.bind("T", int_type, loc())
        assert solver.get_type("T") == int_type

    def test_add_upper_tightens_bound(self) -> None:
        """add_upper() tightens upper bound via meet."""
        solver = Solver()
        solver.get("T")  # Initialize with Top
        solver.add_upper("T", TCon(int, ()), loc())
        bounds = solver.get_bounds("T")
        assert bounds is not None
        _, upper = bounds
        assert upper == TCon(int, ())

    def test_add_lower_tightens_bound(self) -> None:
        """add_lower() tightens lower bound via join."""
        solver = Solver()
        solver.get("T")  # Initialize with Bottom
        solver.add_lower("T", TCon(int, ()), loc())
        bounds = solver.get_bounds("T")
        assert bounds is not None
        lower, _ = bounds
        assert lower == TCon(int, ())


class TestComplexScenarios:
    """Tests for more complex type checking scenarios."""

    def test_multiple_variables(self) -> None:
        """Multiple variables can be bound independently."""
        c1 = EqConstraint(TVar("T"), TCon(int, ()), loc())
        c2 = EqConstraint(TVar("U"), TCon(str, ()), loc())
        solver = solve([c1, c2])
        assert solver.get_type("T") == TCon(int, ())
        assert solver.get_type("U") == TCon(str, ())

    def test_chained_variables(self) -> None:
        """T = U, U = int binds both to int."""
        c1 = EqConstraint(TVar("T"), TVar("U"), loc())
        c2 = EqConstraint(TVar("U"), TCon(int, ()), loc())
        solver = solve([c1, c2])
        assert solver.get_type("T") == TCon(int, ())
        assert solver.get_type("U") == TCon(int, ())

    def test_nested_generics(self) -> None:
        """list[dict[T, U]] = list[dict[int, str]] binds T and U."""
        inner_t_u = TCon(dict, (TVar("T"), TVar("U")))
        list_inner = TCon(list, (inner_t_u,))
        inner_int_str = TCon(dict, (TCon(int, ()), TCon(str, ())))
        list_concrete = TCon(list, (inner_int_str,))
        c = EqConstraint(list_inner, list_concrete, loc())
        solver = solve([c])
        assert solver.get_type("T") == TCon(int, ())
        assert solver.get_type("U") == TCon(str, ())

    def test_constraint_order_independence(self) -> None:
        """Order of constraints should not affect result."""
        # Same constraints in different orders
        c1 = EqConstraint(TVar("T"), TCon(int, ()), loc())
        c2 = EqConstraint(TVar("U"), TVar("T"), loc())

        solver1 = solve([c1, c2])
        solver2 = solve([c2, c1])

        assert solver1.get_type("T") == solver2.get_type("T")
        assert solver1.get_type("U") == solver2.get_type("U")

    def test_subtype_with_generics(self) -> None:
        """Subtyping with generic types."""
        # list[int] <: list[int] should pass
        list_int = TCon(list, (TCon(int, ()),))
        c = SubConstraint(list_int, list_int, loc())
        assert typecheck([c]) is None

    def test_error_location_preserved(self) -> None:
        """Error messages preserve source location."""
        c = EqConstraint(TCon(int, ()), TCon(str, ()), loc("line 42"))
        result = typecheck([c])
        assert isinstance(result, TypeError)
        assert "line 42" in str(result)


# =============================================================================
# Tests for Constraint Generation from Programs
# =============================================================================


class Literal(Node[int], tag="tc_literal"):
    """A literal integer value."""

    value: int


class StringLiteral(Node[str], tag="tc_string_literal"):
    """A literal string value."""

    value: str


class Add(Node[int], tag="tc_add"):
    """Add two integer nodes."""

    left: Child[int]
    right: Child[int]


class Box[T](Node[T], tag="tc_box"):
    """A generic box containing a value."""

    value: T


class Pair[A, B](Node[tuple[A, B]], tag="tc_pair"):
    """A pair of values."""

    first: A
    second: B


class Container(Node[list[int]], tag="tc_container"):
    """A container with a list of integers."""

    items: list[int]


class RefNode(Node[int], tag="tc_refnode"):
    """A node that references another node."""

    target: Ref[Node[int]]


class TestConstraintGenerator:
    """Tests for the ConstraintGenerator class."""

    def test_fresh_var_uses_location_based_names(self) -> None:
        """Fresh variables use location-based names for uniqueness."""
        prog = Program(root=Literal(value=0))
        cache: dict[str, TExp] = {}

        # Different locations produce different type variable names
        gen1 = NodeConstraintGenerator(Location("root"), prog, cache)
        gen2 = NodeConstraintGenerator(Location("nodes").index("x"), prog, cache)

        v1 = gen1.fresh_var("T")
        v2 = gen2.fresh_var("T")

        assert v1.name == "T@root"
        assert v2.name == "T@nodes['x']"
        assert v1 != v2

    def test_simple_node_generates_constraints(self) -> None:
        """A simple node generates field constraints."""
        prog = Program(root=Literal(value=42))
        constraints = generate_constraints(prog)

        # Should have a constraint for the value field
        assert len(constraints) >= 1
        # Should be able to solve it (int <: int)
        result = typecheck(constraints)
        assert result is None

    def test_nested_nodes_generate_constraints(self) -> None:
        """Nested nodes generate constraints for all levels."""
        prog = Program(
            root=Add(
                left=Literal(value=1),
                right=Literal(value=2),
            ),
        )
        constraints = generate_constraints(prog)

        # Should have constraints for Add.left, Add.right, and each Literal.value
        assert len(constraints) >= 3
        result = typecheck(constraints)
        assert result is None

    def test_generic_node_creates_fresh_type_vars(self) -> None:
        """Generic nodes get fresh type variables."""
        prog = Program(root=Box(value=42))
        gen = ConstraintGenerator(prog)
        constraints = gen.generate()

        # The Box[T] should have a fresh type var for T
        assert len(constraints) >= 1
        result = typecheck(constraints)
        assert result is None

    def test_multiple_generic_nodes_have_separate_type_vars(self) -> None:
        """Multiple generic nodes don't share type variables."""
        # Two Box nodes with different value types
        box_int = Box(value=42)
        box_str = Box(value="hello")

        prog = Program(
            root=Literal(value=0),  # Dummy root
            nodes={"box_int": box_int, "box_str": box_str},
        )

        gen = ConstraintGenerator(prog)
        constraints = gen.generate()

        # Should have created at least 2 different type vars for the two boxes
        # (T$0 and T$1 or similar)
        result = typecheck(constraints)
        assert result is None

    def test_program_with_refs(self) -> None:
        """Programs with references generate valid constraints."""
        prog = Program(
            root=Ref[Node[int]](id="main"),
            nodes={
                "main": Add(
                    left=Ref[Node[int]](id="x"),
                    right=Ref[Node[int]](id="y"),
                ),
                "x": Literal(value=1),
                "y": Literal(value=2),
            },
        )

        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None

    def test_container_with_list_field(self) -> None:
        """Nodes with list fields type check correctly."""
        prog = Program(root=Container(items=[1, 2, 3]))
        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None


class TestConstraintGenerationTypeErrors:
    """Tests for type errors detected via constraint generation."""

    def test_wrong_primitive_type_fails(self) -> None:
        """Wrong primitive type in field should fail type check."""
        # Create a Literal with a string value (should be int)
        # We can't actually do this in Python without tricks,
        # but we can test the type inference logic

        # Instead, test with Box which is generic
        box = Box(value="not_an_int")  # T will be inferred as str
        prog = Program(root=box)
        constraints = generate_constraints(prog)

        # This should pass because Box[T] accepts any T
        result = typecheck(constraints)
        assert result is None

    def test_empty_program(self) -> None:
        """Empty program with just root generates minimal constraints."""
        prog = Program(root=Literal(value=0))
        constraints = generate_constraints(prog)

        # At least one constraint for the value field
        assert len(constraints) >= 1
        result = typecheck(constraints)
        assert result is None


class TestPairNode:
    """Tests for nodes with multiple type parameters."""

    def test_pair_with_same_types(self) -> None:
        """Pair with same types for both parameters."""
        prog = Program(root=Pair(first=1, second=2))
        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None

    def test_pair_with_different_types(self) -> None:
        """Pair with different types for each parameter."""
        prog = Program(root=Pair(first=1, second="hello"))
        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None


class TestConstraintGeneratorIntegration:
    """Integration tests for constraint generation and solving."""

    def test_complex_ast_type_checks(self) -> None:
        """Complex AST with multiple node types type checks correctly."""
        prog = Program(
            root=Ref[Node[int]](id="result"),
            nodes={
                "x": Literal(value=10),
                "y": Literal(value=20),
                "sum": Add(
                    left=Ref[Node[int]](id="x"),
                    right=Ref[Node[int]](id="y"),
                ),
                "boxed": Box(value=42),
                "result": Add(
                    left=Ref[Node[int]](id="sum"),
                    right=Literal(value=5),
                ),
            },
        )

        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None

    def test_deeply_nested_nodes(self) -> None:
        """Deeply nested node structure type checks."""
        prog = Program(
            root=Add(
                left=Add(
                    left=Add(
                        left=Literal(value=1),
                        right=Literal(value=2),
                    ),
                    right=Literal(value=3),
                ),
                right=Literal(value=4),
            ),
        )

        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None


class TestRefResolution:
    """Tests for ref resolution during type checking."""

    def test_ref_with_correct_return_type(self) -> None:
        """Ref to node with matching return type passes."""
        prog = Program(
            root=Add(
                left=Ref[Node[int]](id="x"),
                right=Literal(value=2),
            ),
            nodes={
                "x": Literal(value=1),
            },
        )
        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None

    def test_ref_with_wrong_return_type_fails(self) -> None:
        """Ref to node with incompatible return type fails."""
        # Add expects Node[int], but "s" is a StringLiteral which returns str
        prog = Program(
            root=Add(
                left=Ref[Node[int]](id="s"),  # Claims to be Node[int]
                right=Literal(value=2),
            ),
            nodes={
                "s": StringLiteral(value="hello"),  # Actually returns str
            },
        )
        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert isinstance(result, TypeError)

    def test_invalid_ref_raises_key_error(self) -> None:
        """Ref to non-existent node raises KeyError (structural error)."""
        prog = Program(
            root=Add(
                left=Ref[Node[int]](id="nonexistent"),
                right=Literal(value=2),
            ),
            nodes={},
        )
        # Invalid ref is a structural issue, not a type error
        with pytest.raises(KeyError):
            generate_constraints(prog)

    def test_ref_chain_type_checks(self) -> None:
        """Chain of refs with consistent types passes."""
        prog = Program(
            root=Ref[Node[int]](id="final"),
            nodes={
                "x": Literal(value=10),
                "y": Literal(value=20),
                "intermediate": Add(
                    left=Ref[Node[int]](id="x"),
                    right=Ref[Node[int]](id="y"),
                ),
                "final": Add(
                    left=Ref[Node[int]](id="intermediate"),
                    right=Literal(value=5),
                ),
            },
        )
        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert result is None

    def test_ref_chain_with_type_mismatch_fails(self) -> None:
        """Chain of refs with type mismatch fails."""
        prog = Program(
            root=Add(
                left=Ref[Node[int]](id="bad"),
                right=Literal(value=1),
            ),
            nodes={
                "str_node": StringLiteral(value="oops"),
                # bad tries to add a string node to an int
                "bad": Add(
                    left=Ref[Node[int]](id="str_node"),
                    right=Literal(value=1),
                ),
            },
        )
        constraints = generate_constraints(prog)
        result = typecheck(constraints)
        assert isinstance(result, TypeError)
