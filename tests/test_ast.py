"""Tests for typedsl.ast module."""

import json
from typing import Any

import pytest

from typedsl.ast import Interpreter, Program
from typedsl.nodes import Node, Ref


class TestProgramBasics:
    """Test basic Program functionality."""

    def test_program_creation_with_ref_root(self) -> None:
        """Test creating a Program with reference root."""

        class Literal(Node[int], tag="literal_prog"):
            value: int

        nodes = {"node1": Literal(value=42), "node2": Literal(value=100)}
        prog = Program(root=Ref(id="node1"), nodes=nodes)

        assert isinstance(prog.root, Ref)
        assert prog.root.id == "node1"
        assert len(prog.nodes) == 2
        assert "node1" in prog.nodes
        assert "node2" in prog.nodes

    def test_program_creation_with_node_root(self) -> None:
        """Test creating a Program with direct node as root."""

        class Literal(Node[int], tag="literal_prog_node"):
            value: int

        root_node = Literal(value=42)
        prog = Program(root=root_node)

        assert isinstance(prog.root, Node)
        assert prog.root.value == 42
        assert len(prog.nodes) == 0

    def test_program_with_empty_nodes(self) -> None:
        """Test creating Program with no nodes dict."""

        class Literal(Node[int], tag="literal_prog_empty"):
            value: int

        prog = Program(root=Literal(value=42))

        assert isinstance(prog.root, Node)
        assert len(prog.nodes) == 0


class TestProgramResolve:
    """Test Program.resolve() reference resolution."""

    def test_resolve_simple_ref(self) -> None:
        """Test resolving a simple reference."""

        class Number(Node[int], tag="number_resolve"):
            value: int

        node = Number(value=42)
        prog = Program(root=Ref(id="num"), nodes={"num": node})

        ref = Ref[Node[int]](id="num")
        resolved = prog.resolve(ref)

        assert resolved is node
        assert resolved.value == 42

    def test_resolve_multiple_refs(self) -> None:
        """Test resolving multiple different references."""

        class Data(Node[str], tag="data_resolve"):
            text: str

        nodes = {
            "first": Data(text="hello"),
            "second": Data(text="world"),
            "third": Data(text="test"),
        }
        prog = Program(root=Ref(id="first"), nodes=nodes)

        ref1 = Ref[Node[str]](id="first")
        ref2 = Ref[Node[str]](id="second")
        ref3 = Ref[Node[str]](id="third")

        assert prog.resolve(ref1).text == "hello"
        assert prog.resolve(ref2).text == "world"
        assert prog.resolve(ref3).text == "test"

    def test_resolve_with_shared_nodes(self) -> None:
        """Test resolving refs in a graph with shared nodes."""

        class Const(Node[float], tag="const_shared"):
            value: float

        class Add(Node[float], tag="add_shared"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        # Create AST where "x" is shared
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Const(value=5.0),
                "y": Const(value=3.0),
                "sum": Add(left=Ref(id="x"), right=Ref(id="y")),
                "result": Add(left=Ref(id="sum"), right=Ref(id="x")),
            },
        )

        # Resolve x from two different contexts
        x_from_sum = prog.resolve(prog.nodes["sum"].left)
        x_from_result = prog.resolve(prog.nodes["result"].right)

        # Should be the same object
        assert x_from_sum is x_from_result
        assert x_from_sum.value == 5.0

    def test_resolve_nonexistent_ref_raises_error(self) -> None:
        """Test that resolving nonexistent ref raises KeyError with helpful message."""

        class Item(Node[int], tag="item_resolve_err"):
            value: int

        prog = Program(root=Ref(id="root"), nodes={"root": Item(value=1)})

        ref = Ref[Node[int]](id="nonexistent")

        with pytest.raises(KeyError, match="Node 'nonexistent' not found in program"):
            prog.resolve(ref)

    def test_resolve_error_lists_available_nodes(self) -> None:
        """Test that error message lists available node IDs."""

        class Item(Node[int], tag="item_resolve_list"):
            value: int

        prog = Program(
            root=Ref(id="a"),
            nodes={"a": Item(value=1), "b": Item(value=2), "c": Item(value=3)},
        )

        ref = Ref[Node[int]](id="missing")

        with pytest.raises(KeyError) as exc_info:
            prog.resolve(ref)

        error_msg = str(exc_info.value)
        # Error should mention available IDs
        assert "Available node IDs" in error_msg
        assert "'a'" in error_msg or "a" in error_msg
        assert "'b'" in error_msg or "b" in error_msg
        assert "'c'" in error_msg or "c" in error_msg

    def test_resolve_type_casting(self) -> None:
        """Test that resolve returns correctly typed nodes."""

        class StringNode(Node[str], tag="string_resolve_type"):
            text: str

        node = StringNode(text="hello")
        prog = Program(root=Ref(id="str"), nodes={"str": node})

        ref = Ref[Node[str]](id="str")
        resolved = prog.resolve(ref)

        # Should have the correct type at runtime
        assert isinstance(resolved, StringNode)
        assert resolved.text == "hello"


class TestProgramSerialization:
    """Test Program serialization methods."""

    def test_to_dict_simple(self) -> None:
        """Test serializing simple Program to dict."""

        class Num(Node[int], tag="num_ast_dict"):
            value: int

        prog = Program(
            root=Ref(id="n1"),
            nodes={"n1": Num(value=42), "n2": Num(value=100)},
        )

        result = prog.to_dict()

        assert result["root"]["tag"] == "ref"
        assert result["root"]["id"] == "n1"
        assert "nodes" in result
        assert "n1" in result["nodes"]
        assert "n2" in result["nodes"]
        assert result["nodes"]["n1"]["tag"] == "num_ast_dict"
        assert result["nodes"]["n1"]["value"] == 42

    def test_to_dict_with_refs(self) -> None:
        """Test serializing Program with references."""

        class Val(Node[int], tag="val_ast_dict_ref"):
            value: int

        class Op(Node[int], tag="op_ast_dict_ref"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        prog = Program(
            root=Ref(id="result"),
            nodes={
                "a": Val(value=1),
                "b": Val(value=2),
                "result": Op(left=Ref(id="a"), right=Ref(id="b")),
            },
        )

        result = prog.to_dict()

        assert result["root"]["tag"] == "ref"
        assert result["root"]["id"] == "result"
        assert result["nodes"]["result"]["left"]["tag"] == "ref"
        assert result["nodes"]["result"]["left"]["id"] == "a"

    def test_to_json_produces_valid_json(self) -> None:
        """Test that to_json() produces valid JSON."""

        class Simple(Node[str], tag="simple_ast_json"):
            text: str

        prog = Program(root=Ref(id="s"), nodes={"s": Simple(text="test")})

        json_str = prog.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["root"]["tag"] == "ref"
        assert parsed["root"]["id"] == "s"
        assert "nodes" in parsed

    def test_to_json_is_formatted(self) -> None:
        """Test that to_json() output is formatted."""

        class Item(Node[int], tag="item_ast_json_fmt"):
            value: int

        prog = Program(root=Ref(id="i"), nodes={"i": Item(value=1)})

        json_str = prog.to_json()

        # Should contain newlines (formatted)
        assert "\n" in json_str


class TestProgramDeserialization:
    """Test Program deserialization methods."""

    def test_from_dict_missing_root_key(self) -> None:
        """Test that from_dict raises error when 'root' key is missing."""
        data = {"nodes": {}}  # Missing 'root' key

        with pytest.raises(KeyError, match="Missing required key 'root'"):
            Program.from_dict(data)

    def test_from_dict_missing_nodes_key(self) -> None:
        """Test that from_dict raises error when 'nodes' key is missing."""
        data = {"root": "test"}  # Missing 'nodes' key

        with pytest.raises(KeyError, match="Missing required key 'nodes'"):
            Program.from_dict(data)

    def test_from_dict_simple(self) -> None:
        """Test deserializing simple Program from dict."""

        class Number(Node[int], tag="number_ast_from_dict"):
            value: int

        data = {
            "root": {"tag": "ref", "id": "n1"},
            "nodes": {
                "n1": {"tag": "number_ast_from_dict", "value": 42},
                "n2": {"tag": "number_ast_from_dict", "value": 100},
            },
        }

        prog = Program.from_dict(data)

        assert isinstance(prog.root, Ref)
        assert prog.root.id == "n1"
        assert len(prog.nodes) == 2
        assert isinstance(prog.nodes["n1"], Number)
        assert prog.nodes["n1"].value == 42

    def test_from_dict_with_refs(self) -> None:
        """Test deserializing Program with references."""

        class Leaf(Node[str], tag="leaf_ast_from_dict_ref"):
            text: str

        class Branch(Node[str], tag="branch_ast_from_dict_ref"):
            left: Ref[Node[str]]
            right: Ref[Node[str]]

        data = {
            "root": {"tag": "ref", "id": "root"},
            "nodes": {
                "a": {"tag": "leaf_ast_from_dict_ref", "text": "hello"},
                "b": {"tag": "leaf_ast_from_dict_ref", "text": "world"},
                "root": {
                    "tag": "branch_ast_from_dict_ref",
                    "left": {"tag": "ref", "id": "a"},
                    "right": {"tag": "ref", "id": "b"},
                },
            },
        }

        prog = Program.from_dict(data)

        assert isinstance(prog.root, Ref)
        assert prog.root.id == "root"
        root_node = prog.nodes["root"]
        assert isinstance(root_node, Branch)
        assert root_node.left.id == "a"
        assert root_node.right.id == "b"

    def test_from_json_simple(self) -> None:
        """Test deserializing Program from JSON string."""

        class Value(Node[int], tag="value_ast_from_json"):
            num: int

        json_str = """{
            "root": {"tag": "ref", "id": "v"},
            "nodes": {
                "v": {"tag": "value_ast_from_json", "num": 42}
            }
        }"""

        prog = Program.from_json(json_str)

        assert isinstance(prog.root, Ref)
        assert prog.root.id == "v"
        assert isinstance(prog.nodes["v"], Value)
        assert prog.nodes["v"].num == 42

    def test_from_json_with_complex_structure(self) -> None:
        """Test deserializing complex Program from JSON."""

        class Const(Node[float], tag="const_ast_from_json_complex"):
            value: float

        class Expr(Node[float], tag="expr_ast_from_json_complex"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        json_str = """{
            "root": {"tag": "ref", "id": "result"},
            "nodes": {
                "a": {"tag": "const_ast_from_json_complex", "value": 1.5},
                "b": {"tag": "const_ast_from_json_complex", "value": 2.5},
                "result": {
                    "tag": "expr_ast_from_json_complex",
                    "left": {"tag": "ref", "id": "a"},
                    "right": {"tag": "ref", "id": "b"}
                }
            }
        }"""

        prog = Program.from_json(json_str)

        assert isinstance(prog.root, Ref)
        assert prog.root.id == "result"
        assert len(prog.nodes) == 3
        result_node = prog.nodes["result"]
        assert isinstance(result_node, Expr)


class TestProgramRoundTrip:
    """Test round-trip serialization of Program."""

    def test_simple_ast_round_trip(self) -> None:
        """Test simple Program round-trip through dict."""

        class Data(Node[str], tag="data_ast_rt"):
            text: str
            count: int

        original = Program(
            root=Ref(id="d1"),
            nodes={
                "d1": Data(text="hello", count=1),
                "d2": Data(text="world", count=2),
            },
        )

        serialized = original.to_dict()
        deserialized = Program.from_dict(serialized)

        assert deserialized.root == original.root
        assert len(deserialized.nodes) == len(original.nodes)
        assert deserialized.nodes["d1"] == original.nodes["d1"]
        assert deserialized.nodes["d2"] == original.nodes["d2"]

    def test_ast_with_refs_round_trip(self) -> None:
        """Test Program with references round-trip."""

        class Point(Node[tuple[int, int]], tag="point_ast_rt"):
            x: int
            y: int

        class Line(Node[str], tag="line_ast_rt"):
            start: Ref[Node[tuple[int, int]]]
            end: Ref[Node[tuple[int, int]]]

        original = Program(
            root=Ref(id="line"),
            nodes={
                "p1": Point(x=0, y=0),
                "p2": Point(x=10, y=10),
                "line": Line(start=Ref(id="p1"), end=Ref(id="p2")),
            },
        )

        serialized = original.to_dict()
        deserialized = Program.from_dict(serialized)

        assert deserialized.root == original.root
        assert len(deserialized.nodes) == 3
        assert deserialized.nodes["p1"] == original.nodes["p1"]
        assert deserialized.nodes["line"] == original.nodes["line"]

    def test_ast_json_round_trip(self) -> None:
        """Test AST round-trip through JSON."""

        class Num(Node[int], tag="num_ast_json_rt"):
            value: int

        class Sum(Node[int], tag="sum_ast_json_rt"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        original = Program(
            root=Ref(id="result"),
            nodes={
                "a": Num(value=5),
                "b": Num(value=10),
                "result": Sum(left=Ref(id="a"), right=Ref(id="b")),
            },
        )

        json_str = original.to_json()
        deserialized = Program.from_json(json_str)

        assert deserialized.root == original.root
        assert len(deserialized.nodes) == len(original.nodes)
        assert deserialized.nodes["a"] == original.nodes["a"]
        assert deserialized.nodes["b"] == original.nodes["b"]
        assert deserialized.nodes["result"] == original.nodes["result"]


class TestProgramWithSharedNodes:
    """Test Program with shared node patterns."""

    def test_diamond_pattern(self) -> None:
        """Test Program with diamond pattern (node reused by multiple parents)."""

        class Val(Node[int], tag="val_diamond"):
            value: int

        class Op(Node[int], tag="op_diamond"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        # Diamond: top -> (left, right) -> (both reference bottom)
        prog = Program(
            root=Ref(id="top"),
            nodes={
                "bottom": Val(value=1),
                "left": Op(left=Ref(id="bottom"), right=Ref(id="bottom")),
                "right": Op(left=Ref(id="bottom"), right=Ref(id="bottom")),
                "top": Op(left=Ref(id="left"), right=Ref(id="right")),
            },
        )

        # All references to "bottom" should resolve to the same node
        bottom = prog.nodes["bottom"]
        left_node = prog.nodes["left"]
        right_node = prog.nodes["right"]

        assert prog.resolve(left_node.left) is bottom
        assert prog.resolve(left_node.right) is bottom
        assert prog.resolve(right_node.left) is bottom
        assert prog.resolve(right_node.right) is bottom

    def test_shared_subexpression(self) -> None:
        """Test Program with shared subexpression."""

        class Const(Node[float], tag="const_shared_sub"):
            value: float

        class Add(Node[float], tag="add_shared_sub"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Mul(Node[float], tag="mul_shared_sub"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        # Expression: (x + y) * (x + y) where (x + y) is shared
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Const(value=2.0),
                "y": Const(value=3.0),
                "sum": Add(left=Ref(id="x"), right=Ref(id="y")),
                "result": Mul(left=Ref(id="sum"), right=Ref(id="sum")),
            },
        )

        result = prog.nodes["result"]
        # Both sides of multiply reference the same sum node
        assert result.left.id == "sum"
        assert result.right.id == "sum"
        assert result.left == result.right


class TestProgramEdgeCases:
    """Test edge cases and special scenarios."""

    def test_ast_with_single_node(self) -> None:
        """Test Program with only one node."""

        class Single(Node[int], tag="single_ast"):
            value: int

        prog = Program(root=Ref(id="only"), nodes={"only": Single(value=42)})

        assert isinstance(prog.root, Ref)
        assert prog.root.id == "only"
        assert len(prog.nodes) == 1
        assert prog.nodes["only"].value == 42

    def test_ast_root_can_be_any_node(self) -> None:
        """Test that root can point to any node in the program."""

        class Node1(Node[int], tag="node1_ast_root"):
            value: int

        class Node2(Node[int], tag="node2_ast_root"):
            value: int

        nodes = {"a": Node1(value=1), "b": Node2(value=2), "c": Node1(value=3)}

        # Root can be any of them
        prog1 = Program(root=Ref(id="a"), nodes=nodes)
        assert isinstance(prog1.root, Ref)
        assert prog1.root.id == "a"

        prog2 = Program(root=Ref(id="b"), nodes=nodes)
        assert isinstance(prog2.root, Ref)
        assert prog2.root.id == "b"

        prog3 = Program(root=Ref(id="c"), nodes=nodes)
        assert isinstance(prog3.root, Ref)
        assert prog3.root.id == "c"

    def test_ast_with_node_containing_none(self) -> None:
        """Test Program with node that has None field."""

        class Optional(Node[str], tag="optional_ast_edge"):
            required: str
            optional: int | None

        prog = Program(
            root=Ref(id="opt"),
            nodes={"opt": Optional(required="value", optional=None)},
        )

        result = prog.to_dict()
        assert result["nodes"]["opt"]["optional"] is None

        # Round trip
        restored = Program.from_dict(result)
        assert restored.nodes["opt"].optional is None

    def test_empty_program_serialization(self) -> None:
        """Test serializing Program with empty node as root."""

        class Empty(Node[None], tag="empty_prog"):
            pass

        prog = Program(root=Empty())

        result = prog.to_dict()
        assert result["root"]["tag"] == "empty_prog"
        assert len(result["nodes"]) == 0

        # Round trip
        restored = Program.from_dict(result)
        assert isinstance(restored.root, Empty)
        assert len(restored.nodes) == 0


class TestProgramIntegrationExamples:
    """Test complete real-world-like examples using Program."""

    def test_expression_tree_example(self) -> None:
        """Test complete expression tree: (a + b) * c."""

        class Var(Node[float], tag="var_example"):
            name: str

        class BinOp(Node[float], tag="binop_example"):
            op: str
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        prog = Program(
            root=Ref(id="result"),
            nodes={
                "a": Var(name="a"),
                "b": Var(name="b"),
                "c": Var(name="c"),
                "sum": BinOp(op="+", left=Ref(id="a"), right=Ref(id="b")),
                "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="c")),
            },
        )

        # Verify structure
        assert isinstance(prog.root, Ref)
        assert prog.root.id == "result"
        result_node = prog.nodes["result"]
        assert result_node.op == "*"
        assert result_node.left.id == "sum"
        assert result_node.right.id == "c"

        # Verify serialization
        json_str = prog.to_json()
        restored = Program.from_json(json_str)
        assert restored == prog

    def test_dataflow_graph_example(self) -> None:
        """Test dataflow graph with shared inputs."""

        class Input(Node[int], tag="input_dataflow"):
            source: str

        class Transform(Node[int], tag="transform_dataflow"):
            func: str
            input: Ref[Node[int]]

        class Merge(Node[int], tag="merge_dataflow"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        # Graph: input -> (transform1, transform2) -> merge
        prog = Program(
            root=Ref(id="output"),
            nodes={
                "input": Input(source="data.csv"),
                "t1": Transform(func="normalize", input=Ref(id="input")),
                "t2": Transform(func="scale", input=Ref(id="input")),
                "output": Merge(left=Ref(id="t1"), right=Ref(id="t2")),
            },
        )

        # Both transforms reference the same input
        t1 = prog.nodes["t1"]
        t2 = prog.nodes["t2"]
        assert prog.resolve(t1.input) is prog.nodes["input"]
        assert prog.resolve(t2.input) is prog.nodes["input"]

        # Verify round-trip
        serialized = prog.to_dict()
        restored = Program.from_dict(serialized)
        assert restored.root == prog.root
        assert len(restored.nodes) == len(prog.nodes)


class TestInterpreterBasics:
    """Test basic Interpreter functionality."""

    def test_interpreter_is_abstract(self) -> None:
        """Test that Interpreter cannot be instantiated directly."""

        class Num(Node[int], tag="num_interp_abstract"):
            value: int

        prog = Program(root=Ref(id="n"), nodes={"n": Num(value=1)})

        with pytest.raises(TypeError, match="abstract"):
            Interpreter(prog)  # type: ignore[abstract]

    def test_simple_interpreter(self) -> None:
        """Test a simple interpreter that evaluates constants."""

        class Const(Node[float], tag="const_interp_simple"):
            value: float

        class Calculator(Interpreter[None, float, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case _:
                        msg = f"Unknown node: {type(node)}"
                        raise NotImplementedError(msg)

        prog = Program(root=Ref(id="c"), nodes={"c": Const(value=42.0)})
        result = Calculator(prog).run(None)

        assert result == 42.0

    def test_interpreter_with_context(self) -> None:
        """Test interpreter that uses context for variable lookup."""

        class Var(Node[float], tag="var_interp_ctx"):
            name: str

        class Calculator(Interpreter[dict[str, float], float, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Var(name=n):
                        return self.ctx[n]
                    case _:
                        msg = f"Unknown node: {type(node)}"
                        raise NotImplementedError(msg)

        prog = Program(root=Ref(id="x"), nodes={"x": Var(name="x")})
        result = Calculator(prog).run({"x": 10.0, "y": 20.0})

        assert result == 10.0

    def test_interpreter_has_ast_access(self) -> None:
        """Test that interpreter has access to the program."""

        class Num(Node[int], tag="num_interp_ast_access"):
            value: int

        class Inspector(Interpreter[None, int, int]):
            def eval(self, _node: Node[Any]) -> int:
                # Access ast from within eval
                return len(self.program.nodes)

        prog = Program(
            root=Ref(id="a"),
            nodes={"a": Num(value=1), "b": Num(value=2), "c": Num(value=3)},
        )
        result = Inspector(prog).run(None)

        assert result == 3


class TestInterpreterResolve:
    """Test Interpreter.resolve() functionality."""

    def test_resolve_returns_node(self) -> None:
        """Test that resolve returns the referenced node."""

        class Val(Node[int], tag="val_interp_resolve"):
            value: int

        class Wrapper(Node[int], tag="wrapper_interp_resolve"):
            inner: Ref[Node[int]]

        class Evaluator(Interpreter[None, int, int]):
            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Val(value=v):
                        return v
                    case Wrapper(inner=ref):
                        resolved = self.resolve(ref)
                        return self.eval(resolved)
                    case _:
                        raise NotImplementedError

        prog = Program(
            root=Ref(id="w"),
            nodes={
                "v": Val(value=42),
                "w": Wrapper(inner=Ref(id="v")),
            },
        )
        result = Evaluator(prog).run(None)

        assert result == 42

    def test_resolve_with_refs_in_expression(self) -> None:
        """Test resolving refs in a binary expression."""

        class Const(Node[float], tag="const_interp_expr"):
            value: float

        class Add(Node[float], tag="add_interp_expr"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Calculator(Interpreter[None, float, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        prog = Program(
            root=Ref(id="sum"),
            nodes={
                "a": Const(value=10.0),
                "b": Const(value=32.0),
                "sum": Add(left=Ref(id="a"), right=Ref(id="b")),
            },
        )
        result = Calculator(prog).run(None)

        assert result == 42.0


class TestInterpreterWithSharedNodes:
    """Test Interpreter with DAG patterns (shared nodes)."""

    def test_shared_node_evaluated_multiple_times(self) -> None:
        """Test that shared nodes are evaluated each time (no built-in memoization)."""

        class Counter(Node[int], tag="counter_interp_shared"):
            value: int

        class Add(Node[int], tag="add_interp_shared"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        class CountingEvaluator(Interpreter[None, int, int]):
            def __init__(self, program: Node[Any] | Program) -> None:
                super().__init__(program)
                self.eval_count = 0

            def eval(self, node: Node[Any]) -> int:
                self.eval_count += 1
                match node:
                    case Counter(value=v):
                        return v
                    case Add(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        # x is shared: result = x + x
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Counter(value=5),
                "result": Add(left=Ref(id="x"), right=Ref(id="x")),
            },
        )

        evaluator = CountingEvaluator(prog)
        result = evaluator.run(None)

        assert result == 10
        # x is evaluated twice (once for left, once for right), plus result itself
        assert evaluator.eval_count == 3

    def test_diamond_pattern_evaluation(self) -> None:
        """Test evaluation of diamond pattern."""

        class Const(Node[float], tag="const_interp_diamond"):
            value: float

        class Mul(Node[float], tag="mul_interp_diamond"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Add(Node[float], tag="add_interp_diamond"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Calculator(Interpreter[None, float, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case Mul(left=l, right=r):
                        return self.eval(self.resolve(l)) * self.eval(self.resolve(r))
                    case Add(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        # Expression: (x * y) + (x * y) where (x * y) is shared
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Const(value=3.0),
                "y": Const(value=4.0),
                "product": Mul(left=Ref(id="x"), right=Ref(id="y")),
                "result": Add(left=Ref(id="product"), right=Ref(id="product")),
            },
        )

        result = Calculator(prog).run(None)
        assert result == 24.0  # (3 * 4) + (3 * 4) = 12 + 12 = 24


class TestInterpreterUserMemoization:
    """Test that users can implement their own memoization."""

    def test_user_implemented_memoization(self) -> None:
        """Test user-implemented memoization pattern."""

        class Const(Node[int], tag="const_interp_memo"):
            value: int

        class Add(Node[int], tag="add_interp_memo"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        class MemoizingCalculator(Interpreter[None, int, int]):
            def __init__(self, program: Node[Any] | Program) -> None:
                super().__init__(program)
                self._cache: dict[str, int] = {}
                self.eval_count = 0

            def eval_ref(self, ref: Ref[Node[int]]) -> int:
                """Evaluate a ref with memoization."""
                if ref.id not in self._cache:
                    self._cache[ref.id] = self.eval(self.resolve(ref))
                return self._cache[ref.id]

            def eval(self, node: Node[Any]) -> int:
                self.eval_count += 1
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        return self.eval_ref(l) + self.eval_ref(r)
                    case _:
                        raise NotImplementedError

        # x is shared: result = x + x
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Const(value=5),
                "result": Add(left=Ref(id="x"), right=Ref(id="x")),
            },
        )

        evaluator = MemoizingCalculator(prog)
        result = evaluator.run(None)

        assert result == 10
        # With memoization, x is only evaluated once
        assert evaluator.eval_count == 2  # result + x (x cached for second use)


class TestInterpreterComplexExamples:
    """Test complete real-world-like interpreter examples."""

    def test_arithmetic_expression_evaluator(self) -> None:
        """Test complete arithmetic expression evaluator."""

        class Const(Node[float], tag="const_interp_arith"):
            value: float

        class Var(Node[float], tag="var_interp_arith"):
            name: str

        class BinOp(Node[float], tag="binop_interp_arith"):
            op: str
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class ArithmeticEvaluator(Interpreter[dict[str, float], float, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case Var(name=n):
                        return self.ctx[n]
                    case BinOp(op="+", left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case BinOp(op="-", left=l, right=r):
                        return self.eval(self.resolve(l)) - self.eval(self.resolve(r))
                    case BinOp(op="*", left=l, right=r):
                        return self.eval(self.resolve(l)) * self.eval(self.resolve(r))
                    case BinOp(op="/", left=l, right=r):
                        return self.eval(self.resolve(l)) / self.eval(self.resolve(r))
                    case _:
                        msg = f"Unknown node: {type(node)}"
                        raise NotImplementedError(msg)

        # Expression: (x + 2) * (y - 1)
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "x": Var(name="x"),
                "y": Var(name="y"),
                "two": Const(value=2.0),
                "one": Const(value=1.0),
                "sum": BinOp(op="+", left=Ref(id="x"), right=Ref(id="two")),
                "diff": BinOp(op="-", left=Ref(id="y"), right=Ref(id="one")),
                "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="diff")),
            },
        )

        result = ArithmeticEvaluator(prog).run({"x": 3.0, "y": 5.0})
        # (3 + 2) * (5 - 1) = 5 * 4 = 20
        assert result == 20.0

    def test_string_concatenation_interpreter(self) -> None:
        """Test interpreter for string operations."""

        class StrLiteral(Node[str], tag="strlit_interp"):
            value: str

        class Concat(Node[str], tag="concat_interp"):
            left: Ref[Node[str]]
            right: Ref[Node[str]]

        class StringInterpreter(Interpreter[None, str, str]):
            def eval(self, node: Node[Any]) -> str:
                match node:
                    case StrLiteral(value=v):
                        return v
                    case Concat(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        prog = Program(
            root=Ref(id="result"),
            nodes={
                "hello": StrLiteral(value="Hello"),
                "space": StrLiteral(value=" "),
                "world": StrLiteral(value="World"),
                "hello_space": Concat(left=Ref(id="hello"), right=Ref(id="space")),
                "result": Concat(left=Ref(id="hello_space"), right=Ref(id="world")),
            },
        )

        result = StringInterpreter(prog).run(None)
        assert result == "Hello World"

    def test_interpreter_with_inline_and_ref_nodes(self) -> None:
        """Test interpreter that handles both inline nodes and refs."""

        class Const(Node[int], tag="const_interp_mixed"):
            value: int

        class Add(Node[int], tag="add_interp_mixed"):
            left: Node[int] | Ref[Node[int]]
            right: Node[int] | Ref[Node[int]]

        class MixedEvaluator(Interpreter[None, int, int]):
            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        # Handle both inline nodes and refs
                        left_val = (
                            self.eval(self.resolve(l))
                            if isinstance(l, Ref)
                            else self.eval(l)
                        )
                        right_val = (
                            self.eval(self.resolve(r))
                            if isinstance(r, Ref)
                            else self.eval(r)
                        )
                        return left_val + right_val
                    case _:
                        raise NotImplementedError

        # Mix of inline and ref nodes
        prog = Program(
            root=Ref(id="result"),
            nodes={
                "shared": Const(value=10),
                # result has inline left and ref right
                "result": Add(left=Const(value=5), right=Ref(id="shared")),
            },
        )

        result = MixedEvaluator(prog).run(None)
        assert result == 15


class TestInterpreterOnResultHook:
    """Tests for the on_result hook."""

    def test_default_on_result_returns_unchanged(self) -> None:
        """Default on_result returns the result unchanged."""

        class Const(Node[int], tag="const_hook_default"):
            value: int

        class SimpleEvaluator(Interpreter[None, int, int]):
            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Const(value=v):
                        return v
                    case _:
                        raise NotImplementedError

        result = SimpleEvaluator(Const(value=42)).run(None)
        assert result == 42

    def test_on_result_transforms_result(self) -> None:
        """on_result can transform the evaluation result."""

        class Const(Node[float], tag="const_hook_transform"):
            value: float

        class RoundingCalculator(Interpreter[None, float, float]):
            def on_result(self, result: float) -> float:
                return round(result, 2)

            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case _:
                        raise NotImplementedError

        result = RoundingCalculator(Const(value=3.14159)).run(None)
        assert result == 3.14

    def test_on_result_with_validation(self) -> None:
        """on_result can be used for validation."""

        class Const(Node[int], tag="const_hook_validate"):
            value: int

        class ValidatingEvaluator(Interpreter[None, int, int]):
            def on_result(self, result: int) -> int:
                if result < 0:
                    raise ValueError("Result must be non-negative")
                return result

            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Const(value=v):
                        return v
                    case _:
                        raise NotImplementedError

        # Positive value passes
        result = ValidatingEvaluator(Const(value=10)).run(None)
        assert result == 10

        # Negative value raises
        with pytest.raises(ValueError, match="Result must be non-negative"):
            ValidatingEvaluator(Const(value=-5)).run(None)

    def test_on_result_has_access_to_context(self) -> None:
        """on_result can access self.ctx."""

        class Const(Node[float], tag="const_hook_ctx"):
            value: float

        class ContextAwareEvaluator(Interpreter[dict[str, float], float, float]):
            def on_result(self, result: float) -> float:
                multiplier = self.ctx.get("multiplier", 1.0)
                return result * multiplier

            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case _:
                        raise NotImplementedError

        evaluator = ContextAwareEvaluator(Const(value=10.0))

        # Without multiplier
        assert evaluator.run({}) == 10.0

        # With multiplier
        assert evaluator.run({"multiplier": 2.5}) == 25.0

    def test_on_result_called_once_per_run(self) -> None:
        """on_result is called exactly once per run() invocation."""

        class Const(Node[int], tag="const_hook_count"):
            value: int

        class CountingEvaluator(Interpreter[None, int, int]):
            def __init__(self, program: Node[Any] | Program) -> None:
                super().__init__(program)
                self.hook_call_count = 0

            def on_result(self, result: int) -> int:
                self.hook_call_count += 1
                return result

            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Const(value=v):
                        return v
                    case _:
                        raise NotImplementedError

        evaluator = CountingEvaluator(Const(value=42))

        evaluator.run(None)
        assert evaluator.hook_call_count == 1

        evaluator.run(None)
        assert evaluator.hook_call_count == 2

    def test_on_result_with_complex_program(self) -> None:
        """on_result works with programs containing references."""

        class Const(Node[int], tag="const_hook_complex"):
            value: int

        class Add(Node[int], tag="add_hook_complex"):
            left: Node[int] | Ref[Node[int]]
            right: Node[int] | Ref[Node[int]]

        class SummingEvaluator(Interpreter[None, int, int]):
            def on_result(self, result: int) -> int:
                return result * 10  # Scale up the final result

            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        left_val = self.eval(self.resolve(l))
                        right_val = self.eval(self.resolve(r))
                        return left_val + right_val
                    case _:
                        raise NotImplementedError

        prog = Program(
            root=Ref(id="result"),
            nodes={
                "five": Const(value=5),
                "result": Add(left=Ref(id="five"), right=Const(value=3)),
            },
        )

        result = SummingEvaluator(prog).run(None)
        assert result == 80  # (5 + 3) * 10

    def test_on_result_type_transformation(self) -> None:
        """on_result can transform from eval type E to a different run type R."""

        class Const(Node[float], tag="const_hook_transform_type"):
            value: float

        class BinOp(Node[float], tag="binop_hook_transform_type"):
            op: str
            left: Node[float] | Ref[Node[float]]
            right: Node[float] | Ref[Node[float]]

        # E=float, R=str - eval returns float, run returns str
        class StringifyingCalculator(Interpreter[None, float, str]):
            def on_result(self, result: float) -> str:
                return f"Result: {result:.2f}"

            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case BinOp(op=op, left=l, right=r):
                        left_val = self.eval(self.resolve(l))
                        right_val = self.eval(self.resolve(r))
                        if op == "+":
                            return left_val + right_val
                        if op == "*":
                            return left_val * right_val
                        raise ValueError(f"Unknown op: {op}")
                    case _:
                        raise NotImplementedError

        expr = BinOp(op="*", left=Const(value=3.0), right=Const(value=4.5))
        result = StringifyingCalculator(expr).run(None)

        assert result == "Result: 13.50"
        assert isinstance(result, str)

    def test_on_result_wrapping_type(self) -> None:
        """on_result can wrap the result in a custom container type."""
        from dataclasses import dataclass as dc

        @dc
        class EvalResult:
            value: int
            node_count: int

        class Const(Node[int], tag="const_hook_wrap"):
            value: int

        class Add(Node[int], tag="add_hook_wrap"):
            left: Node[int] | Ref[Node[int]]
            right: Node[int] | Ref[Node[int]]

        # E=int, R=EvalResult
        class WrappingEvaluator(Interpreter[None, int, EvalResult]):
            def __init__(self, program: Node[Any] | Program) -> None:
                super().__init__(program)
                self.node_count = 0

            def on_result(self, result: int) -> EvalResult:
                return EvalResult(value=result, node_count=self.node_count)

            def eval(self, node: Node[Any]) -> int:
                self.node_count += 1
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        expr = Add(left=Const(value=1), right=Add(left=Const(value=2), right=Const(value=3)))
        result = WrappingEvaluator(expr).run(None)

        assert isinstance(result, EvalResult)
        assert result.value == 6  # 1 + (2 + 3)
        assert result.node_count == 5  # 3 Const + 2 Add nodes
