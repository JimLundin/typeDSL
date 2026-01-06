"""
Calculator DSL Example
======================

A simple mathematical expression evaluator demonstrating:
- Node definition
- The Interpreter base class
- Both simple nested trees and graph-based programs with shared subexpressions
"""

from typing import Literal
from typedsl import Node, Ref, Program, Interpreter


# ============================================================================
# Define Nodes
# ============================================================================
# Note: You can use simple tags like tag="calc_const", or multi-part signatures
# like ns="calc", name="const", version="1.0" for namespacing/versioning.

class Const(Node[float], tag="calc_const"):
    """A constant numeric value."""
    value: float


class Var(Node[float], tag="calc_var"):
    """A variable reference."""
    name: str


class BinOp(Node[float], tag="calc_binop"):
    """Binary operation: +, -, *, /"""
    op: Literal["+", "-", "*", "/"]
    left: Ref[Node[float]]
    right: Ref[Node[float]]


# ============================================================================
# Implement Interpreter
# ============================================================================

class Calculator(Interpreter[dict[str, float], float]):
    """Evaluates calculator expressions.

    The interpreter is reusable - create once with a program,
    then run multiple times with different variable contexts.
    """

    def eval(self, node: Node[float]) -> float:
        """Evaluate a node using pattern matching."""
        match node:
            case Const(value=v):
                return v

            case Var(name=n):
                if n not in self.ctx:
                    raise ValueError(f"Undefined variable: {n}")
                return self.ctx[n]

            case BinOp(op="+", left=l, right=r):
                return self.eval(self.resolve(l)) + self.eval(self.resolve(r))

            case BinOp(op="-", left=l, right=r):
                return self.eval(self.resolve(l)) - self.eval(self.resolve(r))

            case BinOp(op="*", left=l, right=r):
                return self.eval(self.resolve(l)) * self.eval(self.resolve(r))

            case BinOp(op="/", left=l, right=r):
                right_val = self.eval(self.resolve(r))
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return self.eval(self.resolve(l)) / right_val

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")


# ============================================================================
# Examples
# ============================================================================

def example_simple_nested():
    """Example 1: Simple nested tree (no Program needed)."""
    print("=" * 70)
    print("Example 1: Simple nested tree")
    print("=" * 70)

    # Build a simple nested expression: (3 + 4) * 2
    expr = BinOp(
        op="*",
        left=BinOp(op="+", left=Const(value=3.0), right=Const(value=4.0)),
        right=Const(value=2.0),
    )

    # Pass the node directly to the interpreter
    calculator = Calculator(expr)

    # Run with empty context (no variables)
    result = calculator.run({})

    print(f"Expression: (3 + 4) * 2")
    print(f"Result: {result}")
    print(f"Expected: 14.0")
    print()


def example_graph_with_shared_nodes():
    """Example 2: Graph-based program with shared subexpressions."""
    print("=" * 70)
    print("Example 2: Graph with shared nodes")
    print("=" * 70)

    # Build program with explicit node IDs: (x + y) * (x + y)
    # The subexpression (x + y) is shared via references
    program = Program(
        root=Ref(id="result"),
        nodes={
            "x": Const(value=3.0),
            "y": Const(value=4.0),
            "sum": BinOp(op="+", left=Ref(id="x"), right=Ref(id="y")),
            "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="sum")),
        },
    )

    # Serialize to JSON
    print("Program as JSON:")
    print(program.to_json())
    print()

    # Create interpreter once
    calculator = Calculator(program)

    # Run with empty context
    result = calculator.run({})

    print(f"Expression: (x + y) * (x + y) where x=3, y=4")
    print(f"Result: {result}")
    print(f"Expected: (3 + 4) * (3 + 4) = 49")
    print()


def example_reusable_interpreter():
    """Example 3: Reusing interpreter with different contexts."""
    print("=" * 70)
    print("Example 3: Reusable interpreter")
    print("=" * 70)

    # Build program with variables: a * 2
    program = Program(
        root=Ref(id="expr"),
        nodes={
            "a": Var(name="a"),
            "two": Const(value=2.0),
            "expr": BinOp(op="*", left=Ref(id="a"), right=Ref(id="two")),
        },
    )

    # Create interpreter once
    calculator = Calculator(program)

    # Run multiple times with different contexts
    result1 = calculator.run({"a": 5.0})
    result2 = calculator.run({"a": 10.0})
    result3 = calculator.run({"a": 100.0})

    print(f"Expression: a * 2")
    print(f"  a=5.0  → {result1}")
    print(f"  a=10.0 → {result2}")
    print(f"  a=100.0 → {result3}")
    print()


def main():
    """Run all examples."""
    example_simple_nested()
    example_graph_with_shared_nodes()
    example_reusable_interpreter()


if __name__ == "__main__":
    main()
