"""
Adapting an Existing AST
=========================

Shows how to convert an external AST (Python's ast module) to typeDSL.
This pattern applies to ANY existing AST: tree-sitter, ANTLR, proprietary parsers, etc.
"""

import ast as python_ast
from typing import Literal, Any
from typedsl import Node, Ref, Program, Interpreter


# ============================================================================
# Define nanoDSL Nodes Matching Python AST
# ============================================================================

class Constant(Node[Any], tag="py_const"):
    """Maps to python_ast.Constant"""
    value: Any  # int, float, str, bool, None


class Name(Node[Any], tag="py_name"):
    """Maps to python_ast.Name"""
    id: str


class BinOp(Node[Any], tag="py_binop"):
    """Maps to python_ast.BinOp"""
    left: Ref[Node[Any]]
    op: Literal["Add", "Sub", "Mult", "Div"]
    right: Ref[Node[Any]]


# ============================================================================
# Converter: Python AST → typeDSL
# ============================================================================

class PythonASTConverter:
    """Converts Python AST to typeDSL Program."""

    def __init__(self):
        self.nodes: dict[str, Node[Any]] = {}
        self.counter = 0

    def convert(self, py_node: python_ast.AST) -> Program:
        """Convert Python AST to typeDSL Program."""
        root_id = self._convert_node(py_node)
        return Program(root=Ref(id=root_id), nodes=self.nodes)

    def _convert_node(self, py_node: python_ast.AST) -> str:
        """Convert a single Python AST node."""
        match py_node:
            case python_ast.Constant(value=v):
                node_id = f"const_{self.counter}"
                self.counter += 1
                self.nodes[node_id] = Constant(value=v)
                return node_id

            case python_ast.Name(id=name):
                node_id = f"name_{self.counter}"
                self.counter += 1
                self.nodes[node_id] = Name(id=name)
                return node_id

            case python_ast.BinOp(left=left, op=op, right=right):
                left_id = self._convert_node(left)
                right_id = self._convert_node(right)
                op_name = type(op).__name__  # "Add", "Sub", etc.

                node_id = f"binop_{self.counter}"
                self.counter += 1
                self.nodes[node_id] = BinOp(
                    left=Ref(id=left_id),
                    op=op_name,
                    right=Ref(id=right_id)
                )
                return node_id

            case _:
                raise NotImplementedError(
                    f"Not implemented: {type(py_node).__name__}"
                )


# ============================================================================
# Interpreter for typeDSL Python AST
# ============================================================================

class PythonASTInterpreter(Interpreter[dict[str, Any], Any]):
    """Evaluates typeDSL Python AST nodes."""

    def eval(self, node: Node[Any]) -> Any:
        match node:
            case Constant(value=v):
                return v

            case Name(id=name):
                if name not in self.ctx:
                    raise NameError(f"Name '{name}' is not defined")
                return self.ctx[name]

            case BinOp(left=l, op=op, right=r):
                left_val = self.eval(self.resolve(l))
                right_val = self.eval(self.resolve(r))

                match op:
                    case "Add":
                        return left_val + right_val
                    case "Sub":
                        return left_val - right_val
                    case "Mult":
                        return left_val * right_val
                    case "Div":
                        return left_val / right_val

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")


# ============================================================================
# Example
# ============================================================================

def main():
    """Convert and evaluate: (a + b) * 2"""

    python_code = "(a + b) * 2"
    print(f"Python code: {python_code}\n")

    # Parse with Python's ast module
    py_ast = python_ast.parse(python_code, mode="eval")

    # Convert to typeDSL
    converter = PythonASTConverter()
    program = converter.convert(py_ast.body)

    print(f"Created {len(program.nodes)} typeDSL nodes\n")

    # Serialize to JSON
    json_str = program.to_json()
    print("typeDSL Program as JSON:")
    print(json_str)
    print()

    # Deserialize and evaluate
    restored_program = Program.from_json(json_str)

    # Create interpreter once, can reuse with different environments
    interpreter = PythonASTInterpreter(restored_program)

    # Run with environment
    env = {"a": 3, "b": 7}
    result = interpreter.run(env)

    print(f"Environment: {env}")
    print(f"Result: {result}")
    print(f"Expected: {eval(python_code, env)}")

    # Can reuse interpreter with different environment
    env2 = {"a": 10, "b": 5}
    result2 = interpreter.run(env2)
    print(f"\nEnvironment: {env2}")
    print(f"Result: {result2}")
    print(f"Expected: {eval(python_code, env2)}")


if __name__ == "__main__":
    main()
