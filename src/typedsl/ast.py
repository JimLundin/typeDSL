"""Program representation with support for nested trees and graphs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from typedsl.nodes import Node, Ref
from typedsl.serialization import from_dict, to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class Program:
    """A program that can be interpreted.

    Supports both nested trees and graph-based representations:
    - Pure nested: Program(root=BinOp(...))
    - Pure graph: Program(root=Ref(id="expr"), nodes={...})
    - Mixed: Program(root=BinOp(..., child=Ref(id="shared")), nodes={"shared": ...})
    """

    root: Node[Any] | Ref[Node[Any]]
    nodes: Mapping[str, Node[Any]] = field(default_factory=dict)

    def get_root_node(self) -> Node[Any]:
        """Get the root node of the program.

        Returns the actual root node, resolving the reference if necessary.
        This is the entry point for program evaluation.

        Returns:
            The root node to begin evaluation from

        Raises:
            KeyError: If root is a Ref and the ID is not found in nodes

        """
        if isinstance(self.root, Ref):
            return self.resolve(self.root)
        return self.root

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to its node.

        Args:
            ref: Reference to resolve

        Returns:
            The node referenced by the given ref

        Raises:
            KeyError: If the referenced node ID is not found in the program

        """
        if ref.id not in self.nodes:
            available = list(self.nodes.keys())
            msg = (
                f"Node '{ref.id}' not found in program. Available node IDs: {available}"
            )
            raise KeyError(msg)
        return cast("X", self.nodes[ref.id])

    def to_dict(self) -> dict[str, Any]:
        """Serialize program to dictionary."""
        return {
            "root": to_dict(self.root),
            "nodes": {k: to_dict(v) for k, v in self.nodes.items()},
        }

    def to_json(self) -> str:
        """Serialize program to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Program:
        """Deserialize program from dictionary.

        Args:
            data: Dictionary containing 'root' and 'nodes' keys

        Returns:
            Deserialized Program instance

        Raises:
            KeyError: If required keys ('root' or 'nodes') are missing
            ValueError: If node deserialization fails

        """
        if "root" not in data:
            msg = "Missing required key 'root' in program data"
            raise KeyError(msg)
        if "nodes" not in data:
            msg = "Missing required key 'nodes' in program data"
            raise KeyError(msg)

        root = cast("Node[Any] | Ref[Node[Any]]", from_dict(data["root"]))
        nodes = {k: cast("Node[Any]", from_dict(v)) for k, v in data["nodes"].items()}
        return cls(root, nodes)

    @classmethod
    def from_json(cls, s: str) -> Program:
        """Deserialize program from JSON string."""
        return cls.from_dict(json.loads(s))


class Interpreter[Ctx, R](ABC):
    """Base class for program interpreters.

    Provides program access, context management, and reference resolution.
    Subclass and implement `eval` with pattern matching on node types.

    Type Parameters:
        Ctx: Type of evaluation context (use None if no context needed)
        R: Return type of run()

    The interpreter can accept either a simple nested node tree or a full Program:
        - Simple: Interpreter(BinOp(...))
        - Complex: Interpreter(Program(root=Ref(id="expr"), nodes={...}))

    Interpreters are reusable across multiple runs with different contexts.
    """

    def __init__(self, program: Node[Any] | Program) -> None:
        """Initialize the interpreter with a program.

        Args:
            program: Either a root node (for simple nested trees) or a Program
                    (for graphs with shared nodes and references)

        """
        self.program = (
            program if isinstance(program, Program) else Program(root=program)
        )

    def run(self, ctx: Ctx) -> R:
        """Run the program with the given context.

        Args:
            ctx: The evaluation context (variables, environment, etc.)

        Returns:
            The result of evaluating the program

        """
        self.ctx = ctx
        return self.eval(self.program.get_root_node())

    def resolve[X](self, child: Node[X] | Ref[Node[X]]) -> Node[X]:
        """Resolve a child to its node, handling both inline nodes and references.

        This method accepts Child[T] fields (union of Node[T] | Ref[Node[T]])
        and uniformly returns the node - either directly if inline, or by
        resolving the reference.

        Args:
            child: Either an inline Node (returned as-is) or a Ref to resolve

        Returns:
            The node, either directly or resolved from the reference

        Raises:
            KeyError: If child is a Ref and the ID is not found in the program

        Example:
            class BinOp(Node[float]):
                left: Child[float]  # Can be inline or ref
                right: Child[float]

            class Calculator(Interpreter[None, float]):
                def eval(self, node):
                    match node:
                        case BinOp(left=l, right=r):
                            return self.eval(self.resolve(l)) + self.eval(self.resolve(r))

        """
        if isinstance(child, Node):
            return child
        return self.program.resolve(child)

    @abstractmethod
    def eval(self, node: Node[Any]) -> R:
        """Evaluate a node. Implement with pattern matching on node types.

        Type Notes:
            The signature uses Node[Any] because Python's type system cannot
            express the relationship between a node's type parameter T and
            the interpreter's return type R. At runtime, you should only
            evaluate nodes whose type parameter matches R.

            For better type safety in user code, you can narrow the signature:

                def eval(self, node: Node[float]) -> float:  # More specific
                    match node:
                        case Const(value=v): return v
                        ...

        """
        ...
