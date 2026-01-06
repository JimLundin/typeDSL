"""Program representation with support for nested trees and graphs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from typedsl.nodes import Ref
from typedsl.serialization import from_dict, to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typedsl.nodes import Node


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

        # Resolve root if it's a reference, otherwise use it directly
        if isinstance(self.program.root, Ref):
            root = self.program.nodes[self.program.root.id]
        else:
            root = self.program.root

        return self.eval(root)

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to its target node.

        Args:
            ref: The reference to resolve

        Returns:
            The node referenced by the given ref

        Raises:
            KeyError: If the referenced node ID is not found

        """
        return self.program.resolve(ref)

    @abstractmethod
    def eval(self, node: Node[Any]) -> R:
        """Evaluate a node. Implement with pattern matching on node types."""
        ...
