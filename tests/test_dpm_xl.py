"""Tests for the DPM-XL Lazy Polars interpreter example."""

from __future__ import annotations

import pytest

from typedsl import Program, Ref, from_dict, to_dict

# Skip entire module if polars is not available
polars = pytest.importorskip("polars")


# Import after polars is confirmed available
from examples.dpm_xl_lazy_polars import (  # noqa: E402
    Abs,
    Add,
    And,
    Avg,
    Count,
    Div,
    DPMContext,
    DPMInterpreter,
    Eq,
    EvalResult,
    Filter,
    Ge,
    Gt,
    IfThenElse,
    IsNull,
    Le,
    Literal_,
    Lt,
    Max,
    MaxAggr,
    Min,
    MinAggr,
    Mul,
    Not,
    Nvl,
    Or,
    Selection,
    Sub,
    Sum,
    validate,
)


@pytest.fixture
def sample_data() -> polars.LazyFrame:
    """Sample data for testing."""
    return polars.LazyFrame({
        "table": ["T1", "T1", "T1", "T1", "T2", "T2", "T2", "T2"],
        "r": ["r1", "r1", "r2", "r2", "r1", "r1", "r2", "r2"],
        "c": ["c1", "c2", "c1", "c2", "c1", "c2", "c1", "c2"],
        "value": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
    })


class TestEvalResult:
    """Tests for EvalResult class."""

    def test_scalar_detection(self) -> None:
        """Scalar results have no keys."""
        lf = polars.LazyFrame({"f": [42.0]})
        result = EvalResult(lf=lf, keys=frozenset())
        assert result.is_scalar

    def test_recordset_detection(self) -> None:
        """Recordsets have keys."""
        lf = polars.LazyFrame({"r": ["r1", "r2"], "f": [1.0, 2.0]})
        result = EvalResult(lf=lf, keys=frozenset({"r"}))
        assert not result.is_scalar

    def test_collect_materializes(self) -> None:
        """Collect materializes the lazy frame."""
        lf = polars.LazyFrame({"f": [1.0, 2.0, 3.0]})
        result = EvalResult(lf=lf, keys=frozenset())
        df = result.collect()
        assert isinstance(df, polars.DataFrame)
        assert df.height == 3


class TestLiteralNode:
    """Tests for Literal node."""

    def test_float_literal(self, sample_data: polars.LazyFrame) -> None:
        """Float literal evaluates correctly."""
        ast = Literal_(value=42.5)
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 42.5

    def test_int_literal(self, sample_data: polars.LazyFrame) -> None:
        """Int literal evaluates correctly."""
        ast = Literal_(value=10)
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 10

    def test_bool_literal(self, sample_data: polars.LazyFrame) -> None:
        """Bool literal evaluates correctly."""
        ast = Literal_(value=True)
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_none_literal(self, sample_data: polars.LazyFrame) -> None:
        """None literal evaluates correctly."""
        ast = Literal_(value=None)
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is None


class TestSelectionNode:
    """Tests for Selection node."""

    def test_selection_single_cell(self, sample_data: polars.LazyFrame) -> None:
        """Selection of single cell returns scalar."""
        ast = Selection(table="T1", rows=["r1"], cols=["c1"])
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.is_scalar
        assert result.collect().item(0, "f") == 10.0

    def test_selection_row(self, sample_data: polars.LazyFrame) -> None:
        """Selection of row returns recordset with column key."""
        ast = Selection(table="T1", rows=["r1"])
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert "c" in result.keys
        df = result.collect()
        assert df.height == 2  # c1 and c2

    def test_selection_column(self, sample_data: polars.LazyFrame) -> None:
        """Selection of column returns recordset with row key."""
        ast = Selection(table="T1", cols=["c1"])
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert "r" in result.keys
        df = result.collect()
        assert df.height == 2  # r1 and r2

    def test_selection_with_default(self) -> None:
        """Selection with default handles null values."""
        # Create data with null
        data = polars.LazyFrame({
            "table": ["T1", "T1"],
            "r": ["r1", "r2"],
            "c": ["c1", "c1"],
            "value": [10.0, None],
        })
        ast = Selection(table="T1", cols=["c1"], default=0.0)
        ctx = DPMContext(data)
        result = DPMInterpreter(ast).run(ctx)
        df = result.collect()
        # Second row should have default value
        r2_value = df.filter(polars.col("r") == "r2").item(0, "f")
        assert r2_value == 0.0


class TestArithmeticNodes:
    """Tests for arithmetic operations."""

    def test_add(self, sample_data: polars.LazyFrame) -> None:
        """Addition of multiple operands."""
        ast = Add(operands=(
            Literal_(value=10.0),
            Literal_(value=20.0),
            Literal_(value=5.0),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 35.0

    def test_sub(self, sample_data: polars.LazyFrame) -> None:
        """Subtraction of multiple operands."""
        ast = Sub(operands=(
            Literal_(value=100.0),
            Literal_(value=30.0),
            Literal_(value=10.0),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 60.0

    def test_mul(self, sample_data: polars.LazyFrame) -> None:
        """Multiplication of multiple operands."""
        ast = Mul(operands=(
            Literal_(value=2.0),
            Literal_(value=3.0),
            Literal_(value=4.0),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 24.0

    def test_div(self, sample_data: polars.LazyFrame) -> None:
        """Division of multiple operands."""
        ast = Div(operands=(
            Literal_(value=120.0),
            Literal_(value=4.0),
            Literal_(value=3.0),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 10.0

    def test_arithmetic_with_selections(self, sample_data: polars.LazyFrame) -> None:
        """Arithmetic with data selections."""
        # T1[r1,c1] + T1[r1,c2] = 10 + 20 = 30
        ast = Add(operands=(
            Selection(table="T1", rows=["r1"], cols=["c1"]),
            Selection(table="T1", rows=["r1"], cols=["c2"]),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 30.0


class TestComparisonNodes:
    """Tests for comparison operations."""

    def test_eq_equal(self, sample_data: polars.LazyFrame) -> None:
        """Equality comparison with equal values."""
        ast = Eq(operands=(Literal_(value=5.0), Literal_(value=5.0)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_eq_not_equal(self, sample_data: polars.LazyFrame) -> None:
        """Equality comparison with unequal values."""
        ast = Eq(operands=(Literal_(value=5.0), Literal_(value=10.0)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is False

    def test_eq_chained(self, sample_data: polars.LazyFrame) -> None:
        """Chained equality comparison."""
        # 5 == 5 == 5 -> True
        ast = Eq(operands=(
            Literal_(value=5.0),
            Literal_(value=5.0),
            Literal_(value=5.0),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_gt(self, sample_data: polars.LazyFrame) -> None:
        """Greater than comparison."""
        ast = Gt(left=Literal_(value=10.0), right=Literal_(value=5.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_lt(self, sample_data: polars.LazyFrame) -> None:
        """Less than comparison."""
        ast = Lt(left=Literal_(value=5.0), right=Literal_(value=10.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_ge(self, sample_data: polars.LazyFrame) -> None:
        """Greater than or equal comparison."""
        ast = Ge(left=Literal_(value=10.0), right=Literal_(value=10.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_le(self, sample_data: polars.LazyFrame) -> None:
        """Less than or equal comparison."""
        ast = Le(left=Literal_(value=5.0), right=Literal_(value=10.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True


class TestLogicalNodes:
    """Tests for logical operations."""

    def test_and_true(self, sample_data: polars.LazyFrame) -> None:
        """AND with all true values."""
        ast = And(operands=(Literal_(value=True), Literal_(value=True)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_and_false(self, sample_data: polars.LazyFrame) -> None:
        """AND with one false value."""
        ast = And(operands=(Literal_(value=True), Literal_(value=False)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is False

    def test_or_true(self, sample_data: polars.LazyFrame) -> None:
        """OR with one true value."""
        ast = Or(operands=(Literal_(value=False), Literal_(value=True)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_or_false(self, sample_data: polars.LazyFrame) -> None:
        """OR with all false values."""
        ast = Or(operands=(Literal_(value=False), Literal_(value=False)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is False

    def test_not_true(self, sample_data: polars.LazyFrame) -> None:
        """NOT of true value."""
        ast = Not(operand=Literal_(value=True))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is False

    def test_not_false(self, sample_data: polars.LazyFrame) -> None:
        """NOT of false value."""
        ast = Not(operand=Literal_(value=False))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True


class TestUnaryNodes:
    """Tests for unary operations."""

    def test_abs_positive(self, sample_data: polars.LazyFrame) -> None:
        """Absolute value of positive number."""
        ast = Abs(operand=Literal_(value=42.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 42.0

    def test_abs_negative(self, sample_data: polars.LazyFrame) -> None:
        """Absolute value of negative number."""
        ast = Abs(operand=Literal_(value=-42.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 42.0

    def test_is_null_true(self, sample_data: polars.LazyFrame) -> None:
        """IsNull for null value."""
        ast = IsNull(operand=Literal_(value=None))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is True

    def test_is_null_false(self, sample_data: polars.LazyFrame) -> None:
        """IsNull for non-null value."""
        ast = IsNull(operand=Literal_(value=42.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") is False

    def test_nvl_non_null(self, sample_data: polars.LazyFrame) -> None:
        """Nvl returns original value when not null."""
        ast = Nvl(operand=Literal_(value=42.0), default=Literal_(value=0.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 42.0

    def test_nvl_null(self, sample_data: polars.LazyFrame) -> None:
        """Nvl returns default when null."""
        ast = Nvl(operand=Literal_(value=None), default=Literal_(value=99.0))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 99.0


class TestMinMaxNodes:
    """Tests for element-wise min/max operations."""

    def test_max(self, sample_data: polars.LazyFrame) -> None:
        """Element-wise maximum."""
        ast = Max(operands=(
            Literal_(value=5.0),
            Literal_(value=10.0),
            Literal_(value=3.0),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 10.0

    def test_min(self, sample_data: polars.LazyFrame) -> None:
        """Element-wise minimum."""
        ast = Min(operands=(
            Literal_(value=5.0),
            Literal_(value=10.0),
            Literal_(value=3.0),
        ))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 3.0


class TestConditionalNode:
    """Tests for conditional expressions."""

    def test_if_then_else_true(self, sample_data: polars.LazyFrame) -> None:
        """IfThenElse with true condition."""
        ast = IfThenElse(
            condition=Literal_(value=True),
            then_=Literal_(value=1.0),
            else_=Literal_(value=2.0),
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 1.0

    def test_if_then_else_false(self, sample_data: polars.LazyFrame) -> None:
        """IfThenElse with false condition."""
        ast = IfThenElse(
            condition=Literal_(value=False),
            then_=Literal_(value=1.0),
            else_=Literal_(value=2.0),
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 2.0


class TestAggregateNodes:
    """Tests for aggregate operations."""

    def test_sum_all(self, sample_data: polars.LazyFrame) -> None:
        """Sum without group by."""
        ast = Sum(
            operand=Selection(table="T1", cols=["c1"]),
            group_by=None,
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        # T1 c1: 10 + 30 = 40
        assert result.collect().item(0, "f") == 40.0

    def test_sum_grouped(self, sample_data: polars.LazyFrame) -> None:
        """Sum with group by."""
        ast = Sum(
            operand=Selection(table="T1"),
            group_by=("r",),
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        df = result.collect().sort("r")
        # r1: 10 + 20 = 30
        # r2: 30 + 40 = 70
        assert df.filter(polars.col("r") == "r1").item(0, "f") == 30.0
        assert df.filter(polars.col("r") == "r2").item(0, "f") == 70.0

    def test_count(self, sample_data: polars.LazyFrame) -> None:
        """Count aggregation."""
        ast = Count(
            operand=Selection(table="T1"),
            group_by=None,
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 4

    def test_avg(self, sample_data: polars.LazyFrame) -> None:
        """Average aggregation."""
        ast = Avg(
            operand=Selection(table="T1", cols=["c1"]),
            group_by=None,
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        # (10 + 30) / 2 = 20
        assert result.collect().item(0, "f") == 20.0

    def test_max_aggr(self, sample_data: polars.LazyFrame) -> None:
        """Max aggregation."""
        ast = MaxAggr(
            operand=Selection(table="T1"),
            group_by=None,
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 40.0

    def test_min_aggr(self, sample_data: polars.LazyFrame) -> None:
        """Min aggregation."""
        ast = MinAggr(
            operand=Selection(table="T1"),
            group_by=None,
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert result.collect().item(0, "f") == 10.0


class TestFilterNode:
    """Tests for filter operation."""

    def test_filter(self, sample_data: polars.LazyFrame) -> None:
        """Filter data by condition."""
        # Filter T1 where value > 15
        ast = Filter(
            data=Selection(table="T1"),
            condition=Gt(
                left=Selection(table="T1"),
                right=Literal_(value=15.0),
            ),
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        df = result.collect()
        # Only values > 15: 20, 30, 40
        assert df.height == 3
        assert all(v > 15 for v in df["f"].to_list())


class TestProgramWithReferences:
    """Tests for programs using references."""

    def test_basic_refs(self, sample_data: polars.LazyFrame) -> None:
        """Program with basic references."""
        program = Program(
            root=Ref(id="result"),
            nodes={
                "a": Literal_(value=10.0),
                "b": Literal_(value=20.0),
                "result": Add(operands=(Ref(id="a"), Ref(id="b"))),
            },
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(program).run(ctx)
        assert result.collect().item(0, "f") == 30.0

    def test_nested_refs(self, sample_data: polars.LazyFrame) -> None:
        """Program with nested references."""
        program = Program(
            root=Ref(id="result"),
            nodes={
                "x": Literal_(value=2.0),
                "y": Literal_(value=3.0),
                "sum": Add(operands=(Ref(id="x"), Ref(id="y"))),
                "result": Mul(operands=(Ref(id="sum"), Ref(id="sum"))),
            },
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(program).run(ctx)
        # (2 + 3) * (2 + 3) = 25
        assert result.collect().item(0, "f") == 25.0

    def test_complex_validation_rule(self, sample_data: polars.LazyFrame) -> None:
        """Complex validation rule with references."""
        # Validate: Sum(T1[*,c1]) by row == Sum(T1[*,c2]) by row
        program = Program(
            root=Ref(id="validation"),
            nodes={
                "t1_c1": Selection(table="T1", cols=["c1"]),
                "t1_c2": Selection(table="T1", cols=["c2"]),
                "sum_c1": Sum(operand=Ref(id="t1_c1"), group_by=("r",)),
                "sum_c2": Sum(operand=Ref(id="t1_c2"), group_by=("r",)),
                "validation": Eq(operands=(Ref(id="sum_c1"), Ref(id="sum_c2"))),
            },
        )
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(program).run(ctx)
        df = result.collect()
        # r1: sum_c1=10, sum_c2=20 -> False
        # r2: sum_c1=30, sum_c2=40 -> False
        assert all(v is False for v in df["f"].to_list())


class TestSerialization:
    """Tests for AST serialization."""

    def test_literal_serialization(self) -> None:
        """Literal node serializes and deserializes."""
        node = Literal_(value=42.5)
        data = to_dict(node)
        restored = from_dict(data)
        assert isinstance(restored, Literal_)
        assert restored.value == 42.5

    def test_selection_serialization(self) -> None:
        """Selection node serializes and deserializes."""
        node = Selection(table="T1", rows=["r1", "r2"], cols=["c1"], default=0.0)
        data = to_dict(node)
        restored = from_dict(data)
        assert isinstance(restored, Selection)
        assert restored.table == "T1"
        assert restored.rows == ["r1", "r2"]
        assert restored.cols == ["c1"]
        assert restored.default == 0.0

    def test_complex_ast_serialization(self) -> None:
        """Complex AST serializes and deserializes."""
        ast = Eq(operands=(
            Add(operands=(
                Selection(table="T1", rows=["r1"], cols=["c1"]),
                Literal_(value=10.0),
            )),
            Literal_(value=20.0),
        ))
        data = to_dict(ast)
        restored = from_dict(data)
        assert isinstance(restored, Eq)

    def test_program_serialization(self, sample_data: polars.LazyFrame) -> None:
        """Program serializes, deserializes, and runs correctly."""
        program = Program(
            root=Ref(id="result"),
            nodes={
                "a": Literal_(value=5.0),
                "b": Literal_(value=3.0),
                "result": Mul(operands=(Ref(id="a"), Ref(id="b"))),
            },
        )

        # Serialize to JSON
        json_str = program.to_json()

        # Deserialize
        restored = Program.from_json(json_str)

        # Run original and restored, compare results
        ctx1 = DPMContext(sample_data)
        result1 = DPMInterpreter(program).run(ctx1)

        ctx2 = DPMContext(sample_data)
        result2 = DPMInterpreter(restored).run(ctx2)

        assert result1.collect().item(0, "f") == result2.collect().item(0, "f") == 15.0


class TestValidationHelper:
    """Tests for the validate helper function."""

    def test_validate_pass(self, sample_data: polars.LazyFrame) -> None:
        """Validation passes when all checks succeed."""
        # 10 + 20 = 30 (T1[r1,c1] + T1[r1,c2] = 30)
        rule = Eq(operands=(
            Add(operands=(
                Selection(table="T1", rows=["r1"], cols=["c1"]),
                Selection(table="T1", rows=["r1"], cols=["c2"]),
            )),
            Literal_(value=30.0),
        ))
        result = validate(rule, None, sample_data)
        assert result == "PASS"

    def test_validate_fail(self, sample_data: polars.LazyFrame) -> None:
        """Validation fails when checks fail."""
        # T1[r1,c1] = 99 (but it's 10)
        rule = Eq(operands=(
            Selection(table="T1", rows=["r1"], cols=["c1"]),
            Literal_(value=99.0),
        ))
        result = validate(rule, None, sample_data)
        assert isinstance(result, tuple)
        assert result[0] == "FAIL"

    def test_validate_with_precondition_pass(
        self, sample_data: polars.LazyFrame,
    ) -> None:
        """Validation runs when precondition passes."""
        rule = Eq(operands=(
            Selection(table="T1", rows=["r1"], cols=["c1"]),
            Literal_(value=10.0),
        ))
        precondition = Literal_(value=True)
        result = validate(rule, precondition, sample_data)
        assert result == "PASS"

    def test_validate_with_precondition_skip(
        self, sample_data: polars.LazyFrame,
    ) -> None:
        """Validation skips when precondition fails."""
        rule = Eq(operands=(Literal_(value=1.0), Literal_(value=2.0)))  # Would fail
        precondition = Literal_(value=False)
        result = validate(rule, precondition, sample_data)
        assert result == "SKIPPED"


class TestLazyExecution:
    """Tests verifying lazy execution behavior."""

    def test_result_is_lazy(self, sample_data: polars.LazyFrame) -> None:
        """EvalResult contains LazyFrame, not DataFrame."""
        ast = Add(operands=(Literal_(value=1.0), Literal_(value=2.0)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)
        assert isinstance(result.lf, polars.LazyFrame)

    def test_multiple_collects_work(self, sample_data: polars.LazyFrame) -> None:
        """Can collect result multiple times."""
        ast = Mul(operands=(Literal_(value=2.0), Literal_(value=3.0)))
        ctx = DPMContext(sample_data)
        result = DPMInterpreter(ast).run(ctx)

        # Collect multiple times
        df1 = result.collect()
        df2 = result.collect()

        assert df1.item(0, "f") == df2.item(0, "f") == 6.0
