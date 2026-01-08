"""Converting external ASTs to typeDSL (Python ast module as example)."""

import ast as python_ast
from typing import Any, Literal

from typedsl import Interpreter, Node, Program, Ref


class Constant(Node[Any], tag="py_const"):
    value: Any


class Name(Node[Any], tag="py_name"):
    id: str


class BinOp(Node[Any], tag="py_binop"):
    left: Ref[Node[Any]]
    op: Literal["Add", "Sub", "Mult", "Div"]
    right: Ref[Node[Any]]


class PythonASTConverter:
    """Converts Python AST to typeDSL Program."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node[Any]] = {}
        self.counter = 0

    def convert(self, py_node: python_ast.AST) -> Program:
        root_id = self._visit(py_node)
        return Program(root=Ref(id=root_id), nodes=self.nodes)

    def _visit(self, py_node: python_ast.AST) -> str:
        node_id = f"n{self.counter}"
        self.counter += 1

        match py_node:
            case python_ast.Constant(value=v):
                self.nodes[node_id] = Constant(value=v)
            case python_ast.Name(id=name):
                self.nodes[node_id] = Name(id=name)
            case python_ast.BinOp(left=left, op=op, right=right):
                self.nodes[node_id] = BinOp(
                    left=Ref(id=self._visit(left)),
                    op=type(op).__name__,  # type: ignore[arg-type]
                    right=Ref(id=self._visit(right)),
                )
            case _:
                msg = f"Not implemented: {type(py_node).__name__}"
                raise NotImplementedError(msg)
        return node_id


class PythonEval(Interpreter[dict[str, Any], Any, Any]):
    def eval(self, node: Node[Any]) -> Any:
        match node:
            case Constant(value=v):
                return v
            case Name(id=name):
                return self.ctx[name]
            case BinOp(left=l, op=op, right=r):
                left = self.eval(self.resolve(l))
                right = self.eval(self.resolve(r))
                if op == "Add":
                    return left + right
                if op == "Sub":
                    return left - right
                if op == "Mult":
                    return left * right
                if op == "Div":
                    return left / right
                msg = f"Unknown op: {op}"
                raise ValueError(msg)
            case _:
                raise NotImplementedError(type(node))


# Convert Python expression to typeDSL
py_ast = python_ast.parse("(a + b) * 2", mode="eval")
program = PythonASTConverter().convert(py_ast.body)

# Serialize/deserialize round-trip
restored = Program.from_json(program.to_json())

# Evaluate with different bindings
interp = PythonEval(restored)
assert interp.run({"a": 3, "b": 7}) == 20
assert interp.run({"a": 10, "b": 5}) == 30
