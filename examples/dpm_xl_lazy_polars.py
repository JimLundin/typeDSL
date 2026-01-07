"""
DPM-XL Lazy Polars Interpreter
==============================

A financial validation expression evaluator demonstrating:
- Complex AST node definitions with various arities
- Custom interpreter pattern that produces Polars expressions
- Lazy evaluation model with deferred execution
- Context-based operand registration and alignment

This example shows how to implement a domain-specific language for
financial data validation rules using Polars lazy execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import polars as pl

from typedsl import Node, Program, Ref

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


# ============================================================================
# AST Node Definitions
# ============================================================================


class Literal_(Node[float], tag="dpm_literal"):
    """A constant numeric value."""

    value: float | int | bool | None


class Selection(Node[float], tag="dpm_selection"):
    """A data selection from a table.

    Note: rows and cols use list instead of tuple for JSON serialization
    compatibility (JSON arrays deserialize as lists).
    """

    table: str
    rows: list[str] | None = None
    cols: list[str] | None = None
    default: float | None = None


# --- Arithmetic (n-ary) ---


class Add(Node[float], tag="dpm_add"):
    """Addition of multiple operands."""

    operands: tuple[Node[float] | Ref[Node[float]], ...]


class Sub(Node[float], tag="dpm_sub"):
    """Subtraction: first operand minus rest."""

    operands: tuple[Node[float] | Ref[Node[float]], ...]


class Mul(Node[float], tag="dpm_mul"):
    """Multiplication of multiple operands."""

    operands: tuple[Node[float] | Ref[Node[float]], ...]


class Div(Node[float], tag="dpm_div"):
    """Division: first operand divided by rest."""

    operands: tuple[Node[float] | Ref[Node[float]], ...]


# --- Comparison ---


class Eq(Node[bool], tag="dpm_eq"):
    """Equality comparison (chained for n operands)."""

    operands: tuple[Node[float] | Ref[Node[float]], ...]


class Gt(Node[bool], tag="dpm_gt"):
    """Greater than comparison."""

    left: Node[float] | Ref[Node[float]]
    right: Node[float] | Ref[Node[float]]


class Lt(Node[bool], tag="dpm_lt"):
    """Less than comparison."""

    left: Node[float] | Ref[Node[float]]
    right: Node[float] | Ref[Node[float]]


class Ge(Node[bool], tag="dpm_ge"):
    """Greater than or equal comparison."""

    left: Node[float] | Ref[Node[float]]
    right: Node[float] | Ref[Node[float]]


class Le(Node[bool], tag="dpm_le"):
    """Less than or equal comparison."""

    left: Node[float] | Ref[Node[float]]
    right: Node[float] | Ref[Node[float]]


# --- Logical ---


class And(Node[bool], tag="dpm_and"):
    """Logical AND of multiple operands."""

    operands: tuple[Node[bool] | Ref[Node[bool]], ...]


class Or(Node[bool], tag="dpm_or"):
    """Logical OR of multiple operands."""

    operands: tuple[Node[bool] | Ref[Node[bool]], ...]


class Not(Node[bool], tag="dpm_not"):
    """Logical NOT."""

    operand: Node[bool] | Ref[Node[bool]]


# --- Unary ---


class Abs(Node[float], tag="dpm_abs"):
    """Absolute value."""

    operand: Node[float] | Ref[Node[float]]


class IsNull(Node[bool], tag="dpm_is_null"):
    """Check if value is null."""

    operand: Node[float] | Ref[Node[float]]


class Nvl(Node[float], tag="dpm_nvl"):
    """Null value replacement (coalesce)."""

    operand: Node[float] | Ref[Node[float]]
    default: Node[float] | Ref[Node[float]]


# --- Element-wise min/max ---


class Max(Node[float], tag="dpm_max"):
    """Element-wise maximum across operands."""

    operands: tuple[Node[float] | Ref[Node[float]], ...]


class Min(Node[float], tag="dpm_min"):
    """Element-wise minimum across operands."""

    operands: tuple[Node[float] | Ref[Node[float]], ...]


# --- Conditional ---


class IfThenElse(Node[float], tag="dpm_if_then_else"):
    """Conditional expression."""

    condition: Node[bool] | Ref[Node[bool]]
    then_: Node[float] | Ref[Node[float]]
    else_: Node[float] | Ref[Node[float]]


# --- Aggregates ---


class Sum(Node[float], tag="dpm_sum"):
    """Sum aggregation with optional group-by."""

    operand: Node[float] | Ref[Node[float]]
    group_by: tuple[str, ...] | None = None


class Count(Node[float], tag="dpm_count"):
    """Count aggregation with optional group-by."""

    operand: Node[float] | Ref[Node[float]]
    group_by: tuple[str, ...] | None = None


class Avg(Node[float], tag="dpm_avg"):
    """Average aggregation with optional group-by."""

    operand: Node[float] | Ref[Node[float]]
    group_by: tuple[str, ...] | None = None


class MaxAggr(Node[float], tag="dpm_max_aggr"):
    """Maximum aggregation with optional group-by."""

    operand: Node[float] | Ref[Node[float]]
    group_by: tuple[str, ...] | None = None


class MinAggr(Node[float], tag="dpm_min_aggr"):
    """Minimum aggregation with optional group-by."""

    operand: Node[float] | Ref[Node[float]]
    group_by: tuple[str, ...] | None = None


# --- Filter ---


class Filter(Node[float], tag="dpm_filter"):
    """Filter data by condition."""

    data: Node[float] | Ref[Node[float]]
    condition: Node[bool] | Ref[Node[bool]]


# ============================================================================
# Evaluation Infrastructure
# ============================================================================


@dataclass(frozen=True)
class EvalResult:
    """Lazy evaluation result - call .collect() to materialize."""

    lf: pl.LazyFrame
    keys: frozenset[str]

    @property
    def is_scalar(self) -> bool:
        """Check if result is a scalar (no key columns)."""
        return len(self.keys) == 0

    def collect(self) -> pl.DataFrame:
        """Materialize the lazy computation."""
        return self.lf.collect()


@dataclass
class Operand:
    """A registered lazy operand."""

    lf: pl.LazyFrame
    keys: frozenset[str]
    col_name: str


class Context(ABC):
    """Base evaluation context using lazy computation."""

    def __init__(self) -> None:
        """Initialize context with empty operand list."""
        self._operands: list[Operand] = []
        self._counter: int = 0

    @abstractmethod
    def fetch_selection(self, sel: Selection) -> tuple[pl.LazyFrame, frozenset[str]]:
        """Fetch data as LazyFrame. Returns (lazy_frame, keys)."""

    @abstractmethod
    def child(self) -> Context:
        """Create child context for sub-expressions."""

    def _next_col(self) -> str:
        """Generate unique column name for operand."""
        name = f"f_{self._counter}"
        self._counter += 1
        return name

    def register(self, lf: pl.LazyFrame, keys: frozenset[str]) -> pl.Expr:
        """Register a LazyFrame operand, return expression for its fact column."""
        col_name = self._next_col()
        self._operands.append(
            Operand(
                lf=lf.rename({"f": col_name}),
                keys=keys,
                col_name=col_name,
            )
        )
        return pl.col(col_name)

    def register_scalar(self, value: float | int | bool | None) -> pl.Expr:
        """Register a scalar literal."""
        lf = pl.LazyFrame({"f": [value]})
        return self.register(lf, frozenset())

    def register_selection(self, selection: Selection) -> pl.Expr:
        """Fetch and register a selection."""
        lf, keys = self.fetch_selection(selection)
        return self.register(lf, keys)

    def align(self) -> tuple[pl.LazyFrame, frozenset[str]]:
        """Build lazy join plan for all registered operands."""
        if not self._operands:
            msg = "No operands registered"
            raise ValueError(msg)

        if len(self._operands) == 1:
            op = self._operands[0]
            return op.lf, op.keys

        scalars = [op for op in self._operands if len(op.keys) == 0]
        recordsets = [op for op in self._operands if len(op.keys) > 0]

        if not recordsets:
            # All scalars: horizontal concat (lazy)
            combined = pl.concat([op.lf for op in scalars], how="horizontal")
            return combined, frozenset()

        # Build lazy join chain
        result = recordsets[0].lf
        result_keys = set(recordsets[0].keys)

        for op in recordsets[1:]:
            common = list(result_keys & op.keys)
            result_keys = result_keys | op.keys

            if common:
                result = result.join(op.lf, on=common, how="inner")
            else:
                result = result.join(op.lf, how="cross")

        # Add scalar columns as literals (lazy)
        for op in scalars:
            # Use first() to get scalar value lazily
            scalar_col = op.lf.select(pl.col(op.col_name).first()).collect().item()
            result = result.with_columns(pl.lit(scalar_col).alias(op.col_name))

        return result, frozenset(result_keys)

    def finalize(self, expr: pl.Expr) -> EvalResult:
        """Build final lazy computation plan."""
        aligned_lf, keys = self.align()
        result_lf = aligned_lf.with_columns(expr.alias("f")).select([*list(keys), "f"])
        return EvalResult(lf=result_lf, keys=keys)


# ============================================================================
# Interpreter
# ============================================================================


class DPMInterpreter:
    """Interpreter for DPM-XL expressions.

    Note: This interpreter doesn't use the typeDSL Interpreter base class
    because it follows a different pattern where:
    - eval() returns pl.Expr (not the final result)
    - run() calls eval() then ctx.finalize() to produce EvalResult
    - Context manages operand registration and alignment

    This is a deliberate design choice to support the lazy Polars execution model.
    """

    def __init__(self, program: Node[Any] | Program) -> None:
        """Initialize the interpreter with a program."""
        self.program = (
            program if isinstance(program, Program) else Program(root=program)
        )

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to its target node."""
        return self.program.resolve(ref)

    def _get_node[T](self, child: Node[T] | Ref[Node[T]]) -> Node[T]:
        """Get actual node from child (which may be ref or direct node)."""
        if isinstance(child, Ref):
            return self.resolve(child)  # type: ignore[return-value]
        return child

    def run(self, ctx: Context) -> EvalResult:
        """Evaluate AST, return lazy result."""
        root = self.program.get_root_node()
        expr = self.eval(root, ctx)
        return ctx.finalize(expr)

    def eval(self, node: Node[Any], ctx: Context) -> pl.Expr:
        """Evaluate node to Polars expression."""
        match node:
            # --- Leaves ---
            case Literal_(value=v):
                return ctx.register_scalar(v)

            case Selection() as sel:
                return ctx.register_selection(sel)

            # --- Arithmetic (n-ary) ---
            case Add(operands=ops):
                return reduce(lambda a, b: a + b, self._eval_many(ops, ctx))

            case Sub(operands=ops):
                return reduce(lambda a, b: a - b, self._eval_many(ops, ctx))

            case Mul(operands=ops):
                return reduce(lambda a, b: a * b, self._eval_many(ops, ctx))

            case Div(operands=ops):
                return reduce(lambda a, b: a / b, self._eval_many(ops, ctx))

            # --- Comparison ---
            case Eq(operands=ops):
                exprs = list(self._eval_many(ops, ctx))
                pairs = [exprs[i] == exprs[i + 1] for i in range(len(exprs) - 1)]
                return reduce(lambda a, b: a & b, pairs) if len(pairs) > 1 else pairs[0]

            case Gt(left=l, right=r):
                return self.eval(self._get_node(l), ctx) > self.eval(
                    self._get_node(r), ctx
                )

            case Lt(left=l, right=r):
                return self.eval(self._get_node(l), ctx) < self.eval(
                    self._get_node(r), ctx
                )

            case Ge(left=l, right=r):
                return self.eval(self._get_node(l), ctx) >= self.eval(
                    self._get_node(r), ctx
                )

            case Le(left=l, right=r):
                return self.eval(self._get_node(l), ctx) <= self.eval(
                    self._get_node(r), ctx
                )

            # --- Logical ---
            case And(operands=ops):
                return reduce(lambda a, b: a & b, self._eval_many(ops, ctx))

            case Or(operands=ops):
                return reduce(lambda a, b: a | b, self._eval_many(ops, ctx))

            case Not(operand=op):
                return ~self.eval(self._get_node(op), ctx)

            # --- Unary ---
            case Abs(operand=op):
                return self.eval(self._get_node(op), ctx).abs()

            case IsNull(operand=op):
                return self.eval(self._get_node(op), ctx).is_null()

            case Nvl(operand=op, default=d):
                return self.eval(self._get_node(op), ctx).fill_null(
                    self.eval(self._get_node(d), ctx)
                )

            # --- Element-wise min/max ---
            case Max(operands=ops):
                return pl.max_horizontal(*self._eval_many(ops, ctx))

            case Min(operands=ops):
                return pl.min_horizontal(*self._eval_many(ops, ctx))

            # --- Conditional ---
            case IfThenElse(condition=c, then_=t, else_=e):
                return (
                    pl.when(self.eval(self._get_node(c), ctx))
                    .then(self.eval(self._get_node(t), ctx))
                    .otherwise(self.eval(self._get_node(e), ctx))
                )

            # --- Aggregates ---
            case Sum(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.sum(), ctx)

            case Count(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.count(), ctx)

            case Avg(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.mean(), ctx)

            case MaxAggr(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.max(), ctx)

            case MinAggr(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.min(), ctx)

            # --- Filter ---
            case Filter(data=d, condition=c):
                return self._eval_filter(d, c, ctx)

            case _:
                msg = f"Unknown node: {type(node)}"
                raise ValueError(msg)

    def _eval_many(
        self, nodes: tuple[Node[Any] | Ref[Node[Any]], ...], ctx: Context
    ) -> Iterator[pl.Expr]:
        """Evaluate multiple nodes."""
        for node in nodes:
            yield self.eval(self._get_node(node), ctx)

    def _eval_aggregate(
        self,
        operand: Node[float] | Ref[Node[float]],
        group_by: tuple[str, ...] | None,
        agg_fn: Callable[[pl.Expr], pl.Expr],
        ctx: Context,
    ) -> pl.Expr:
        """Aggregate stays lazy until final collect."""
        child = ctx.child()
        inner_expr = self.eval(self._get_node(operand), child)
        inner_lf, _ = child.align()

        # Apply inner expression and aggregate (all lazy)
        inner_lf = inner_lf.with_columns(inner_expr.alias("f"))

        if group_by:
            agg_lf = inner_lf.group_by(list(group_by)).agg(
                agg_fn(pl.col("f")).alias("f")
            )
            result_keys = frozenset(group_by)
        else:
            agg_lf = inner_lf.select(agg_fn(pl.col("f")).alias("f"))
            result_keys = frozenset()

        return ctx.register(agg_lf, result_keys)

    def _eval_filter(
        self,
        data: Node[float] | Ref[Node[float]],
        condition: Node[bool] | Ref[Node[bool]],
        ctx: Context,
    ) -> pl.Expr:
        """Filter stays lazy until final collect."""
        # Evaluate data
        data_ctx = ctx.child()
        data_expr = self.eval(self._get_node(data), data_ctx)
        data_lf, data_keys = data_ctx.align()
        # Keep only keys + computed fact column
        data_lf = data_lf.with_columns(data_expr.alias("f")).select(
            [*list(data_keys), "f"]
        )

        # Evaluate condition
        cond_ctx = ctx.child()
        cond_expr = self.eval(self._get_node(condition), cond_ctx)
        cond_lf, cond_keys = cond_ctx.align()
        # Keep only keys + condition column
        cond_lf = cond_lf.with_columns(cond_expr.alias("condition")).select(
            [*list(cond_keys), "condition"]
        )

        # Semi-join on common keys where condition is true (lazy)
        common_keys = list(data_keys & cond_keys)
        true_keys = cond_lf.filter(pl.col("condition")).select(common_keys)
        filtered_lf = data_lf.join(true_keys, on=common_keys, how="semi")

        return ctx.register(filtered_lf, data_keys)


# ============================================================================
# Sample User Implementation
# ============================================================================


class DPMContext(Context):
    """User-defined context with data access."""

    def __init__(self, data: pl.LazyFrame) -> None:
        """Initialize context with data source."""
        super().__init__()
        self.data = data

    def child(self) -> DPMContext:
        """Create child context for sub-expressions."""
        return DPMContext(self.data)

    def fetch_selection(self, sel: Selection) -> tuple[pl.LazyFrame, frozenset[str]]:
        """Fetch data for a selection."""
        # Build lazy filter
        lf = self.data.filter(pl.col("table") == sel.table)

        if sel.rows is not None:
            lf = lf.filter(pl.col("r").is_in(list(sel.rows)))
        if sel.cols is not None:
            lf = lf.filter(pl.col("c").is_in(list(sel.cols)))

        # Determine keys from selection pattern
        keys: list[str] = []
        if sel.rows is None or len(sel.rows) > 1:
            keys.append("r")
        if sel.cols is None or len(sel.cols) > 1:
            keys.append("c")

        # Select and rename
        lf = lf.select([*keys, "value"]).rename({"value": "f"})

        # Apply default (lazy)
        if sel.default is not None:
            lf = lf.with_columns(pl.col("f").fill_null(sel.default))

        return lf, frozenset(keys)


# ============================================================================
# Validation Helper
# ============================================================================


def validate(
    rule: Node[Any], precondition: Node[Any] | None, data: pl.LazyFrame
) -> str | tuple[str, pl.DataFrame]:
    """Run validation rule on data.

    Args:
        rule: The validation rule (should evaluate to boolean)
        precondition: Optional precondition to check first
        data: The data to validate

    Returns:
        "PASS" if all records pass, "SKIPPED" if precondition fails,
        or ("FAIL", failures_df) with failing records

    """
    ctx = DPMContext(data)
    interp = DPMInterpreter(rule)

    # Check precondition (may need to collect for boolean check)
    if precondition:
        pre_interp = DPMInterpreter(precondition)
        pre_result = pre_interp.run(ctx.child())
        if not pre_result.collect().item(0, "f"):
            return "SKIPPED"

    # Run validation (stays lazy)
    result = interp.run(ctx)

    # Collect and check failures
    df = result.collect()
    failures = df.filter(pl.col("f") == False)  # noqa: E712

    if failures.height == 0:
        return "PASS"
    return ("FAIL", failures)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create sample data
    sample_data = pl.LazyFrame(
        {
            "table": ["T1", "T1", "T1", "T1", "T2", "T2"],
            "r": ["r1", "r1", "r2", "r2", "r1", "r2"],
            "c": ["c1", "c2", "c1", "c2", "c1", "c1"],
            "value": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0],
        }
    )

    # Build simple AST: T1[r1, c1] + T1[r1, c2] = 30
    ast = Eq(
        operands=(
            Add(
                operands=(
                    Selection(table="T1", rows=["r1"], cols=["c1"]),
                    Selection(table="T1", rows=["r1"], cols=["c2"]),
                )
            ),
            Literal_(value=30.0),
        )
    )

    # Create context and interpreter
    ctx = DPMContext(sample_data)
    interp = DPMInterpreter(ast)

    # Run (lazy until collect)
    result = interp.run(ctx)
    print("AST result (lazy):", result)
    print("Collected result:")
    print(result.collect())

    # Example with aggregation: Sum(T1[*, c1])
    sum_ast = Sum(
        operand=Selection(table="T1", cols=["c1"]),
        group_by=None,  # Sum all rows
    )

    ctx2 = DPMContext(sample_data)
    interp2 = DPMInterpreter(sum_ast)
    sum_result = interp2.run(ctx2)
    print("\nSum result:")
    print(sum_result.collect())

    # Example with Program and References
    program = Program(
        root=Ref(id="result"),
        nodes={
            "t1_c1": Selection(table="T1", cols=["c1"]),
            "t1_c2": Selection(table="T1", cols=["c2"]),
            "sum_c1": Sum(operand=Ref(id="t1_c1"), group_by=("r",)),
            "sum_c2": Sum(operand=Ref(id="t1_c2"), group_by=("r",)),
            "result": Eq(operands=(Ref(id="sum_c1"), Ref(id="sum_c2"))),
        },
    )

    ctx3 = DPMContext(sample_data)
    interp3 = DPMInterpreter(program)
    prog_result = interp3.run(ctx3)
    print("\nProgram result (Sum T1[*,c1] by r == Sum T1[*,c2] by r):")
    print(prog_result.collect())
