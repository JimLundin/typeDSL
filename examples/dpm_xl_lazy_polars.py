"""
DPM-XL Lazy Polars Interpreter
==============================

A financial validation expression evaluator demonstrating:
- Complex AST node definitions with various arities
- Interpreter extending the typeDSL base class
- Lazy evaluation model where eval() returns complete EvalResult
- Sub-interpreters for aggregate operations

This example shows how to implement a domain-specific language for
financial data validation rules using Polars lazy execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import polars as pl

from typedsl import Interpreter, Node, Program, Ref

if TYPE_CHECKING:
    from collections.abc import Sequence


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


class Context(ABC):
    """Base evaluation context - provides data access."""

    @abstractmethod
    def fetch_selection(self, sel: Selection) -> EvalResult:
        """Fetch data for a selection as EvalResult."""

    @abstractmethod
    def child(self) -> Context:
        """Create child context for sub-interpreters."""


def align_results(results: Sequence[EvalResult]) -> tuple[pl.LazyFrame, frozenset[str]]:
    """Align multiple EvalResults into a single LazyFrame.

    Returns the aligned LazyFrame with columns f_0, f_1, ... and the combined keys.
    """
    if not results:
        msg = "No results to align"
        raise ValueError(msg)

    if len(results) == 1:
        r = results[0]
        return r.lf.rename({"f": "f_0"}), r.keys

    # Map result id to its index (avoids LazyFrame comparison issues)
    result_to_idx = {id(r): i for i, r in enumerate(results)}

    scalars = [r for r in results if r.is_scalar]
    recordsets = [r for r in results if not r.is_scalar]

    if not recordsets:
        # All scalars: horizontal concat
        frames = [r.lf.rename({"f": f"f_{result_to_idx[id(r)]}"}) for r in results]
        combined = pl.concat(frames, how="horizontal")
        return combined, frozenset()

    # Start with first recordset
    first = recordsets[0]
    result_lf = first.lf.rename({"f": f"f_{result_to_idx[id(first)]}"})
    result_keys = set(first.keys)

    # Join remaining recordsets
    for r in recordsets[1:]:
        idx = result_to_idx[id(r)]
        r_lf = r.lf.rename({"f": f"f_{idx}"})
        common = list(result_keys & r.keys)
        result_keys = result_keys | r.keys

        if common:
            result_lf = result_lf.join(r_lf, on=common, how="inner")
        else:
            result_lf = result_lf.join(r_lf, how="cross")

    # Add scalar columns as literals
    for r in scalars:
        idx = result_to_idx[id(r)]
        col_name = f"f_{idx}"
        scalar_val = r.lf.select(pl.col("f").first()).collect().item()
        result_lf = result_lf.with_columns(pl.lit(scalar_val).alias(col_name))

    return result_lf, frozenset(result_keys)


def make_result(lf: pl.LazyFrame, keys: frozenset[str], expr: pl.Expr) -> EvalResult:
    """Create EvalResult from aligned LazyFrame and expression."""
    result_lf = lf.with_columns(expr.alias("f")).select([*list(keys), "f"])
    return EvalResult(lf=result_lf, keys=keys)


# ============================================================================
# Interpreter
# ============================================================================


class DPMInterpreter(Interpreter[Context, EvalResult]):
    """Interpreter for DPM-XL expressions.

    Extends the typeDSL Interpreter base class. Each eval() call returns
    a complete EvalResult. Sub-interpreters are used for aggregates.
    """

    def _get_node[T](self, child: Node[T] | Ref[Node[T]]) -> Node[T]:
        """Get actual node from child (which may be ref or direct node)."""
        if isinstance(child, Ref):
            return self.resolve(child)  # type: ignore[return-value]
        return child

    def _eval_binary(
        self,
        left: Node[Any] | Ref[Node[Any]],
        right: Node[Any] | Ref[Node[Any]],
        op: str,
    ) -> EvalResult:
        """Evaluate binary operation."""
        left_result = self.eval(self._get_node(left))
        right_result = self.eval(self._get_node(right))
        aligned_lf, keys = align_results([left_result, right_result])

        ops = {
            ">": pl.col("f_0") > pl.col("f_1"),
            "<": pl.col("f_0") < pl.col("f_1"),
            ">=": pl.col("f_0") >= pl.col("f_1"),
            "<=": pl.col("f_0") <= pl.col("f_1"),
        }
        return make_result(aligned_lf, keys, ops[op])

    def _eval_nary(
        self,
        operands: tuple[Node[Any] | Ref[Node[Any]], ...],
        combine: Any,
    ) -> EvalResult:
        """Evaluate n-ary operation."""
        results = [self.eval(self._get_node(op)) for op in operands]
        aligned_lf, keys = align_results(results)
        cols = [pl.col(f"f_{i}") for i in range(len(results))]
        expr = reduce(combine, cols)
        return make_result(aligned_lf, keys, expr)

    def _eval_aggregate(
        self,
        operand: Node[float] | Ref[Node[float]],
        group_by: tuple[str, ...] | None,
        agg_fn: Any,
    ) -> EvalResult:
        """Evaluate aggregate using sub-interpreter."""
        # Create sub-interpreter for operand with shared nodes for reference resolution
        operand_node = self._get_node(operand)
        sub_program = Program(root=operand_node, nodes=self.program.nodes)
        sub_interp = DPMInterpreter(sub_program)
        child_result = sub_interp.run(self.ctx.child())

        # Apply aggregation
        if group_by:
            agg_lf = child_result.lf.group_by(list(group_by)).agg(
                agg_fn(pl.col("f")).alias("f")
            )
            return EvalResult(lf=agg_lf, keys=frozenset(group_by))
        else:
            agg_lf = child_result.lf.select(agg_fn(pl.col("f")).alias("f"))
            return EvalResult(lf=agg_lf, keys=frozenset())

    def eval(self, node: Node[Any]) -> EvalResult:
        """Evaluate node to EvalResult."""
        match node:
            # --- Leaves ---
            case Literal_(value=v):
                lf = pl.LazyFrame({"f": [v]})
                return EvalResult(lf=lf, keys=frozenset())

            case Selection() as sel:
                return self.ctx.fetch_selection(sel)

            # --- Arithmetic (n-ary) ---
            case Add(operands=ops):
                return self._eval_nary(ops, lambda a, b: a + b)

            case Sub(operands=ops):
                return self._eval_nary(ops, lambda a, b: a - b)

            case Mul(operands=ops):
                return self._eval_nary(ops, lambda a, b: a * b)

            case Div(operands=ops):
                return self._eval_nary(ops, lambda a, b: a / b)

            # --- Comparison ---
            case Eq(operands=ops):
                results = [self.eval(self._get_node(op)) for op in ops]
                aligned_lf, keys = align_results(results)
                cols = [pl.col(f"f_{i}") for i in range(len(results))]
                pairs = [cols[i] == cols[i + 1] for i in range(len(cols) - 1)]
                expr = reduce(lambda a, b: a & b, pairs) if len(pairs) > 1 else pairs[0]
                return make_result(aligned_lf, keys, expr)

            case Gt(left=l, right=r):
                return self._eval_binary(l, r, ">")

            case Lt(left=l, right=r):
                return self._eval_binary(l, r, "<")

            case Ge(left=l, right=r):
                return self._eval_binary(l, r, ">=")

            case Le(left=l, right=r):
                return self._eval_binary(l, r, "<=")

            # --- Logical ---
            case And(operands=ops):
                return self._eval_nary(ops, lambda a, b: a & b)

            case Or(operands=ops):
                return self._eval_nary(ops, lambda a, b: a | b)

            case Not(operand=op):
                result = self.eval(self._get_node(op))
                return EvalResult(
                    lf=result.lf.with_columns((~pl.col("f")).alias("f")),
                    keys=result.keys,
                )

            # --- Unary ---
            case Abs(operand=op):
                result = self.eval(self._get_node(op))
                return EvalResult(
                    lf=result.lf.with_columns(pl.col("f").abs().alias("f")),
                    keys=result.keys,
                )

            case IsNull(operand=op):
                result = self.eval(self._get_node(op))
                return EvalResult(
                    lf=result.lf.with_columns(pl.col("f").is_null().alias("f")),
                    keys=result.keys,
                )

            case Nvl(operand=op, default=d):
                op_result = self.eval(self._get_node(op))
                default_result = self.eval(self._get_node(d))
                aligned_lf, keys = align_results([op_result, default_result])
                expr = pl.col("f_0").fill_null(pl.col("f_1"))
                return make_result(aligned_lf, keys, expr)

            # --- Element-wise min/max ---
            case Max(operands=ops):
                results = [self.eval(self._get_node(op)) for op in ops]
                aligned_lf, keys = align_results(results)
                cols = [pl.col(f"f_{i}") for i in range(len(results))]
                expr = pl.max_horizontal(*cols)
                return make_result(aligned_lf, keys, expr)

            case Min(operands=ops):
                results = [self.eval(self._get_node(op)) for op in ops]
                aligned_lf, keys = align_results(results)
                cols = [pl.col(f"f_{i}") for i in range(len(results))]
                expr = pl.min_horizontal(*cols)
                return make_result(aligned_lf, keys, expr)

            # --- Conditional ---
            case IfThenElse(condition=c, then_=t, else_=e):
                cond_result = self.eval(self._get_node(c))
                then_result = self.eval(self._get_node(t))
                else_result = self.eval(self._get_node(e))
                aligned_lf, keys = align_results([cond_result, then_result, else_result])
                expr = (
                    pl.when(pl.col("f_0"))
                    .then(pl.col("f_1"))
                    .otherwise(pl.col("f_2"))
                )
                return make_result(aligned_lf, keys, expr)

            # --- Aggregates (use sub-interpreter) ---
            case Sum(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.sum())

            case Count(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.count())

            case Avg(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.mean())

            case MaxAggr(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.max())

            case MinAggr(operand=op, group_by=gb):
                return self._eval_aggregate(op, gb, lambda c: c.min())

            # --- Filter (use sub-interpreter) ---
            case Filter(data=d, condition=c):
                # Evaluate data with sub-interpreter
                data_node = self._get_node(d)
                data_program = Program(root=data_node, nodes=self.program.nodes)
                data_interp = DPMInterpreter(data_program)
                data_result = data_interp.run(self.ctx.child())

                # Evaluate condition with sub-interpreter
                cond_node = self._get_node(c)
                cond_program = Program(root=cond_node, nodes=self.program.nodes)
                cond_interp = DPMInterpreter(cond_program)
                cond_result = cond_interp.run(self.ctx.child())

                # Semi-join on common keys where condition is true
                common_keys = list(data_result.keys & cond_result.keys)
                true_keys = (
                    cond_result.lf
                    .filter(pl.col("f"))
                    .select(common_keys)
                )
                filtered_lf = data_result.lf.join(true_keys, on=common_keys, how="semi")
                return EvalResult(lf=filtered_lf, keys=data_result.keys)

            case _:
                msg = f"Unknown node: {type(node)}"
                raise ValueError(msg)


# ============================================================================
# Sample User Implementation
# ============================================================================


class DPMContext(Context):
    """User-defined context with data access."""

    def __init__(self, data: pl.LazyFrame) -> None:
        """Initialize context with data source."""
        self.data = data

    def child(self) -> DPMContext:
        """Create child context for sub-interpreters."""
        return DPMContext(self.data)

    def fetch_selection(self, sel: Selection) -> EvalResult:
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

        return EvalResult(lf=lf, keys=frozenset(keys))


# ============================================================================
# Validation Helper
# ============================================================================


def validate(
    rule: Node[Any],
    precondition: Node[Any] | None,
    data: pl.LazyFrame,
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

    # Check precondition
    if precondition:
        pre_interp = DPMInterpreter(precondition)
        pre_result = pre_interp.run(ctx.child())
        if not pre_result.collect().item(0, "f"):
            return "SKIPPED"

    # Run validation
    interp = DPMInterpreter(rule)
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
        group_by=None,
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
