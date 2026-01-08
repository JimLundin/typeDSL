"""Calculator DSL Example.

A simple mathematical expression evaluator demonstrating node definition,
the Interpreter base class, and both nested trees and graph-based programs.
"""

from typing import Literal

from typedsl import Child, Interpreter, Node, Program, Ref


# Define Nodes
class Const(Node[float], tag="calc_const"):
    """A constant numeric value."""

    value: float


class Var(Node[float], tag="calc_var"):
    """A variable reference."""

    name: str


class BinOp(Node[float], tag="calc_binop"):
    """Binary operation with Child[float] fields for flexible composition."""

    op: Literal["+", "-", "*", "/"]
    left: Child[float]
    right: Child[float]


# Implement Interpreter
class Calculator(Interpreter[dict[str, float], float]):
    """Evaluates calculator expressions."""

    def eval(self, node: Node[float]) -> float:
        match node:
            case Const(value=v):
                return v
            case Var(name=n):
                return self.ctx[n]
            case BinOp(op=op, left=l, right=r):
                left = self.eval(self.resolve(l))
                right = self.eval(self.resolve(r))
                if op == "+":
                    return left + right
                if op == "-":
                    return left - right
                if op == "*":
                    return left * right
                if op == "/":
                    return left / right
                msg = f"Unknown operator: {op}"
                raise ValueError(msg)
            case _:
                raise NotImplementedError(type(node))


# Example 1: Simple nested tree - (3 + 4) * 2
simple_expr = BinOp(
    op="*",
    left=BinOp(op="+", left=Const(value=3.0), right=Const(value=4.0)),
    right=Const(value=2.0),
)
calculator = Calculator(simple_expr)
result = calculator.run({})  # 14.0


# Example 2: Graph with shared subexpressions - (x + y) * (x + y)
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
result = calculator.run({})  # 49.0


# Example 3: Reusable interpreter with variables - a * 2
var_program = Program(
    root=Ref(id="expr"),
    nodes={
        "a": Var(name="a"),
        "two": Const(value=2.0),
        "expr": BinOp(op="*", left=Ref(id="a"), right=Ref(id="two")),
    },
)
calculator = Calculator(var_program)
result1 = calculator.run({"a": 5.0})  # 10.0
result2 = calculator.run({"a": 10.0})  # 20.0
