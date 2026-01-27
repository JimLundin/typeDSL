"""Constraint generation for type checking Programs and Nodes."""

from __future__ import annotations

import itertools
import types
from dataclasses import fields
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from typing import (
    TypeVar as TypingTypeVar,
)

from typedsl.typechecker.core import (
    Bottom,
    Constraint,
    EqConstraint,
    SourceLocation,
    SubConstraint,
    Top,
    Type,
    TypeCon,
    TypeVar,
)

if TYPE_CHECKING:
    from typedsl.ast import Program
    from typedsl.nodes import Node

# Import at runtime to avoid circular imports during type checking
# These are used for isinstance checks
_Node: type | None = None
_Ref: type | None = None


def _get_node_ref_types() -> tuple[type, type]:
    """Get Node and Ref types lazily to avoid circular imports."""
    global _Node, _Ref  # noqa: PLW0603
    if _Node is None:
        from typedsl.nodes import Node, Ref

        _Node = Node
        _Ref = Ref
    return _Node, _Ref


class ConstraintGenerator:
    """Generates type constraints from Programs and Nodes.

    Handles fresh type variable generation to avoid name collisions
    between different generic node instances.
    """

    def __init__(self) -> None:
        self._counter = itertools.count()
        self._constraints: list[Constraint] = []

    def fresh_var(self, base_name: str) -> TypeVar:
        """Create a fresh type variable with a unique name.

        Args:
            base_name: The base name for the variable (e.g., "T").

        Returns:
            A TypeVar with a unique name like "T$0", "T$1", etc.

        """
        unique_name = f"{base_name}${next(self._counter)}"
        return TypeVar(unique_name)

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the list."""
        self._constraints.append(constraint)

    def get_constraints(self) -> list[Constraint]:
        """Get all generated constraints."""
        return self._constraints

    def generate_program(self, program: Program) -> list[Constraint]:
        """Generate constraints for an entire Program.

        Args:
            program: The program to type check.

        Returns:
            List of constraints that must be satisfied.

        """
        _node_type, ref_type = _get_node_ref_types()

        # Process all named nodes
        for node_id, node in program.nodes.items():
            self._generate_node(node, f"nodes[{node_id!r}]")

        # Process root
        if isinstance(program.root, ref_type):
            # Root is a reference - ensure it points to a valid node
            ref_id = program.root.id
            if ref_id not in program.nodes:
                # Invalid ref - this is a structural error, not a type error
                # But we could add a constraint that will fail
                loc = SourceLocation("root ref target")
                self.add(EqConstraint(Bottom(), Top(), loc))
        else:
            self._generate_node(program.root, "root")

        return self.get_constraints()

    def _generate_node(self, node: Node[Any], path: str) -> Type:
        """Generate constraints for a single node and return its inferred type.

        Args:
            node: The node to process.
            path: Path for error reporting (e.g., "root.left").

        Returns:
            The inferred type of this node.

        """
        node_cls = type(node)
        hints = get_type_hints(node_cls)

        # Create fresh type variables for this node's type parameters
        type_param_map: dict[str, TypeVar] = {}
        if hasattr(node_cls, "__type_params__"):
            for param in node_cls.__type_params__:
                fresh = self.fresh_var(param.__name__)
                type_param_map[param.__name__] = fresh

        # Process each field
        for f in fields(node_cls):
            if f.name.startswith("_"):
                continue

            field_path = f"{path}.{f.name}"
            field_value = getattr(node, f.name)
            declared_type = hints.get(f.name)

            if declared_type is None:
                continue

            # Convert declared type to our Type representation
            expected_type = self._python_type_to_type(declared_type, type_param_map)

            # Infer the actual type from the value
            actual_type = self._infer_value_type(
                field_value,
                expected_type,
                field_path,
                type_param_map,
            )

            # Generate constraint: actual <: expected
            loc = SourceLocation(field_path)
            self.add(SubConstraint(actual_type, expected_type, loc))

        # Return the node's return type (for use by parent nodes)
        return self._get_return_type(node_cls, type_param_map)

    def _python_type_to_type(
        self,
        py_type: Any,
        type_param_map: dict[str, TypeVar],
    ) -> Type:
        """Convert a Python type annotation to our Type representation.

        Args:
            py_type: The Python type to convert.
            type_param_map: Mapping from type parameter names to fresh TypeVars.

        Returns:
            The corresponding Type.

        """
        node_type, ref_type = _get_node_ref_types()

        # Handle None
        if py_type is None or py_type is type(None):
            return TypeCon(type(None), ())

        # Handle TypeVar (type parameters)
        if isinstance(py_type, TypingTypeVar):
            name = py_type.__name__
            if name in type_param_map:
                return type_param_map[name]
            # Unknown type var - create fresh one
            fresh = self.fresh_var(name)
            type_param_map[name] = fresh
            return fresh

        origin = get_origin(py_type)
        args = get_args(py_type)

        # Handle Union types (including X | None)
        if origin is Union or isinstance(py_type, types.UnionType):
            # For now, treat unions as Top (any of the options)
            # A more sophisticated approach would create union constraints
            return Top()

        # Handle Ref[X]
        if origin is ref_type:
            # The type of a Ref is the Ref itself, parameterized by target type
            if args:
                target_type = self._python_type_to_type(args[0], type_param_map)
                return TypeCon(ref_type, (target_type,))
            return TypeCon(ref_type, ())

        # Handle Node[T] and node subclasses
        is_node_origin = (
            origin is not None
            and isinstance(origin, type)
            and issubclass(origin, node_type)
        )
        if is_node_origin:
            # This is a type annotation like Node[int] or SomeNode[T]
            if args:
                converted_args = tuple(
                    self._python_type_to_type(a, type_param_map) for a in args
                )
                return TypeCon(origin, converted_args)
            return TypeCon(origin, ())

        # Handle generic containers (list, dict, set, etc.)
        if origin is not None:
            converted_args = tuple(
                self._python_type_to_type(a, type_param_map) for a in args
            )
            return TypeCon(origin, converted_args)

        # Handle simple types
        if isinstance(py_type, type):
            return TypeCon(py_type, ())

        # Fallback
        return Top()

    def _infer_value_type(
        self,
        value: Any,
        expected: Type,
        path: str,
        type_param_map: dict[str, TypeVar],
    ) -> Type:
        """Infer the type of a runtime value.

        Args:
            value: The value to infer the type of.
            expected: The expected type (for context).
            path: Path for error reporting.
            type_param_map: Mapping from type parameter names to fresh TypeVars.

        Returns:
            The inferred Type.

        """
        node_type, ref_type = _get_node_ref_types()

        if value is None:
            return TypeCon(type(None), ())

        # Handle Node values
        if isinstance(value, node_type):
            # Recursively generate constraints for nested nodes
            return_type = self._generate_node(value, path)
            # Return Node[return_type] since the value IS a node
            return TypeCon(node_type, (return_type,))

        # Handle Ref values
        if isinstance(value, ref_type):
            # A Ref's type depends on what it references
            # In the DSL, Ref[Node[X]] is interchangeable with Node[X]
            # So if expected is Node[X], return Node[X]
            if isinstance(expected, TypeCon) and expected.constructor is node_type:
                # Ref[Node[X]] is used where Node[X] is expected - compatible
                return expected
            if isinstance(expected, TypeCon) and expected.constructor is ref_type:
                # Ref[X] expected, use the expected type
                return expected
            # Otherwise create a Ref type with fresh variable
            fresh_target = self.fresh_var("RefTarget")
            return TypeCon(ref_type, (fresh_target,))

        # Handle containers
        if isinstance(value, list):
            if not value:
                # Empty list - use expected element type or fresh var
                if isinstance(expected, TypeCon) and expected.args:
                    return expected
                return TypeCon(list, (self.fresh_var("ListElem"),))
            # Infer from first element (simplification)
            elem_type = self._infer_value_type(
                value[0],
                Top(),
                f"{path}[0]",
                type_param_map,
            )
            return TypeCon(list, (elem_type,))

        if isinstance(value, dict):
            if not value:
                if isinstance(expected, TypeCon) and len(expected.args) >= 2:  # noqa: PLR2004
                    return expected
                return TypeCon(
                    dict,
                    (self.fresh_var("DictKey"), self.fresh_var("DictVal")),
                )
            # Infer from first key-value pair
            k, v = next(iter(value.items()))
            key_type = self._infer_value_type(k, Top(), f"{path}.key", type_param_map)
            val_type = self._infer_value_type(v, Top(), f"{path}.val", type_param_map)
            return TypeCon(dict, (key_type, val_type))

        if isinstance(value, set):
            if not value:
                if isinstance(expected, TypeCon) and expected.args:
                    return expected
                return TypeCon(set, (self.fresh_var("SetElem"),))
            elem = next(iter(value))
            elem_type = self._infer_value_type(
                elem,
                Top(),
                f"{path}.elem",
                type_param_map,
            )
            return TypeCon(set, (elem_type,))

        if isinstance(value, tuple):
            elem_types = tuple(
                self._infer_value_type(v, Top(), f"{path}[{i}]", type_param_map)
                for i, v in enumerate(value)
            )
            return TypeCon(tuple, elem_types)

        # Handle primitives
        return TypeCon(type(value), ())

    def _get_return_type(
        self,
        node_cls: type[Node[Any]],
        type_param_map: dict[str, TypeVar],
    ) -> Type:
        """Get the return type of a node class.

        Args:
            node_cls: The node class.
            type_param_map: Mapping from type parameter names to fresh TypeVars.

        Returns:
            The return type as a Type.

        """
        node_type, _ = _get_node_ref_types()

        # Look for Node[T] in __orig_bases__
        for base in getattr(node_cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is None:
                continue
            if isinstance(origin, type) and issubclass(origin, node_type):
                args = get_args(base)
                if args:
                    return_type = args[0]
                    # If it's a TypeVar, use our fresh one
                    if isinstance(return_type, TypingTypeVar):
                        name = return_type.__name__
                        if name in type_param_map:
                            return type_param_map[name]
                    return self._python_type_to_type(return_type, type_param_map)

        # Default to Top (unknown)
        return Top()


def generate_constraints(program: Program) -> list[Constraint]:
    """Generate type constraints for a Program.

    Args:
        program: The program to type check.

    Returns:
        List of constraints that must be satisfied for the program to type check.

    """
    generator = ConstraintGenerator()
    return generator.generate_program(program)
