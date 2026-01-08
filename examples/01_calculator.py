"""Calculator DSL - node definition, interpreter, and Program usage."""

from typing import Literal

from typedsl import Child, Interpreter, Node, Program, Ref


class Const(Node[float], tag="calc_const"):
    value: float


class Var(Node[float], tag="calc_var"):
    name: str


class BinOp(Node[float], tag="calc_binop"):
    op: Literal["+", "-", "*", "/"]
    left: Child[float]
    right: Child[float]


class Calculator(Interpreter[dict[str, float], float, float]):
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


# Inline tree: (3 + 4) * 2
expr = BinOp(
    op="*",
    left=BinOp(op="+", left=Const(value=3.0), right=Const(value=4.0)),
    right=Const(value=2.0),
)
assert Calculator(expr).run({}) == 14.0

# Program with refs: reusable (a * 2) with different contexts
prog = Program(
    root=Ref(id="expr"),
    nodes={
        "a": Var(name="a"),
        "expr": BinOp(op="*", left=Ref(id="a"), right=Const(value=2.0)),
    },
)
calc = Calculator(prog)
assert calc.run({"a": 5.0}) == 10.0
assert calc.run({"a": 10.0}) == 20.0
