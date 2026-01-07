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
# Example 1: Simple nested tree
# ============================================================================

# Build a simple nested expression: (3 + 4) * 2
# No Program wrapper needed - pass node directly to interpreter
simple_expr = BinOp(
    op="*",
    left=BinOp(op="+", left=Const(value=3.0), right=Const(value=4.0)),
    right=Const(value=2.0),
)

calculator = Calculator(simple_expr)
result = calculator.run({})  # result = 14.0


# ============================================================================
# Example 2: Graph with shared subexpressions
# ============================================================================

# Build program with explicit node IDs: (x + y) * (x + y)
# The subexpression (x + y) is shared - referenced twice
shared_program = Program(
    root=Ref(id="result"),
    nodes={
        "x": Const(value=3.0),
        "y": Const(value=4.0),
        "sum": BinOp(op="+", left=Ref(id="x"), right=Ref(id="y")),
        "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="sum")),
    },
)

calculator = Calculator(shared_program)
result = calculator.run({})  # result = 49.0

# Program can be serialized/deserialized
json_str = shared_program.to_json()
restored = Program.from_json(json_str)


# ============================================================================
# Example 3: Reusable interpreter with variables
# ============================================================================

# Build program with variables: a * 2
var_program = Program(
    root=Ref(id="expr"),
    nodes={
        "a": Var(name="a"),
        "two": Const(value=2.0),
        "expr": BinOp(op="*", left=Ref(id="a"), right=Ref(id="two")),
    },
)

# Create interpreter once, run multiple times with different contexts
calculator = Calculator(var_program)
result1 = calculator.run({"a": 5.0})    # result1 = 10.0
result2 = calculator.run({"a": 10.0})   # result2 = 20.0
result3 = calculator.run({"a": 100.0})  # result3 = 200.0
