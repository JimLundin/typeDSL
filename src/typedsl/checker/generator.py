"""Constraint generation from Programs.

This module traverses a Program and emits type equality constraints
by comparing declared field types against actual values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

from typedsl.checker.constraints import Constraint, Location
from typedsl.checker.types import TCon, TExpr, TVarFactory, from_hint
from typedsl.nodes import Node, Ref

if TYPE_CHECKING:
    from typedsl.ast import Program


def get_node_return_type(node_class: type[Node[Any]]) -> Any | None:
    """Extract the return type T from a Node[T] subclass.

    Inspects the class's __orig_bases__ to find the Node base class
    and extract its type argument.

    Args:
        node_class: A subclass of Node.

    Returns:
        The type argument T, or None if it cannot be determined.

    """
    for base in getattr(node_class, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is Node:
            args = get_args(base)
            if args:
                return args[0]
    return None


def _get_elem_hint(declared_hint: Any, container_type: type) -> Any | None:
    """Extract element hint from a container type hint."""
    origin = get_origin(declared_hint)
    if origin is container_type:
        declared_args = get_args(declared_hint)
        return declared_args[0] if declared_args else None
    return None


class ConstraintGenerator:
    """Generates type constraints from a Program.

    Traverses the program's nodes and emits constraints equating
    declared field types with actual value types.
    """

    def __init__(self, program: Program) -> None:
        self.program = program
        self.var_factory = TVarFactory()
        self.constraints: list[Constraint] = []
        self._node_return_types: dict[str, TExpr] = {}
        self._visited: set[int] = set()

    def generate(self) -> list[Constraint]:
        """Generate all constraints for the program.

        Returns:
            A list of type constraints.

        """
        # First pass: collect return types for all named nodes
        self._collect_node_return_types()

        # Second pass: traverse from root and emit constraints
        root = self.program.root
        if isinstance(root, Ref):
            self._visit_ref(root, path=("root",))
        else:
            self._visit_node(root, node_id=None, path=("root",))

        return self.constraints

    def _collect_node_return_types(self) -> None:
        """Collect return types for all named nodes in the program."""
        for node_id, node in self.program.nodes.items():
            return_type = get_node_return_type(type(node))
            if return_type is not None:
                self._node_return_types[node_id] = from_hint(return_type)
            else:
                # Unknown return type - use a fresh type variable
                self._node_return_types[node_id] = self.var_factory.fresh()

    def _get_node_return_texpr(self, node: Node[Any], node_id: str | None) -> TExpr:
        """Get the return type TExpr for a node."""
        if node_id is not None and node_id in self._node_return_types:
            return self._node_return_types[node_id]

        return_type = get_node_return_type(type(node))
        if return_type is not None:
            return from_hint(return_type)

        return self.var_factory.fresh()

    def _visit_ref(self, ref: Ref[Any], path: tuple[str, ...]) -> TExpr:
        """Visit a reference and return its type."""
        if ref.id in self._node_return_types:
            return_type = self._node_return_types[ref.id]
        else:
            return_type = self.var_factory.fresh()

        # Visit the referenced node if we haven't already
        if ref.id in self.program.nodes:
            node = self.program.nodes[ref.id]
            node_obj_id = id(node)
            if node_obj_id not in self._visited:
                self._visited.add(node_obj_id)
                self._visit_node(node, node_id=ref.id, path=(*path, ref.id))

        return TCon(Ref, (TCon(Node, (return_type,)),))

    def _visit_node(
        self,
        node: Node[Any],
        node_id: str | None,
        path: tuple[str, ...],
    ) -> TExpr:
        """Visit a node and emit constraints for its fields."""
        node_class = type(node)
        node_tag = node_class.tag

        try:
            hints = get_type_hints(node_class)
        except Exception:  # noqa: BLE001
            hints = {}

        for field_name, declared_type in hints.items():
            if field_name in ("tag", "signature", "registry"):
                continue
            if not hasattr(node, field_name):
                continue

            value = getattr(node, field_name)
            location = Location(
                node_tag=node_tag,
                node_id=node_id,
                field_name=field_name,
                path=path,
            )

            declared_texpr = from_hint(declared_type)
            actual_texpr = self._infer_type(value, declared_type, path, field_name)

            self.constraints.append(
                Constraint(left=declared_texpr, right=actual_texpr, location=location),
            )

        return_type = self._get_node_return_texpr(node, node_id)
        return TCon(Node, (return_type,))

    def _infer_list_type(
        self,
        value: list[Any],
        declared_hint: Any,
        path: tuple[str, ...],
        field_name: str,
    ) -> TExpr:
        """Infer type for a list value."""
        elem_hint = _get_elem_hint(declared_hint, list)

        if not value:
            if elem_hint is not None:
                elem_texpr = from_hint(elem_hint)
            else:
                elem_texpr = self.var_factory.fresh()
        else:
            elem_texpr = self._infer_type(
                value[0],
                elem_hint,
                path,
                f"{field_name}[0]",
            )

        return TCon(list, (elem_texpr,))

    def _infer_dict_type(
        self,
        value: dict[Any, Any],
        declared_hint: Any,
        path: tuple[str, ...],
        field_name: str,
    ) -> TExpr:
        """Infer type for a dict value."""
        origin = get_origin(declared_hint)
        if origin is dict:
            declared_args = get_args(declared_hint)
            key_hint = declared_args[0] if len(declared_args) > 0 else None
            val_hint = declared_args[1] if len(declared_args) > 1 else None
        else:
            key_hint = None
            val_hint = None

        if not value:
            key_texpr = from_hint(key_hint) if key_hint else self.var_factory.fresh()
            val_texpr = from_hint(val_hint) if val_hint else self.var_factory.fresh()
        else:
            first_key, first_val = next(iter(value.items()))
            key_texpr = self._infer_type(first_key, key_hint, path, f"{field_name}.key")
            val_texpr = self._infer_type(first_val, val_hint, path, f"{field_name}.val")

        return TCon(dict, (key_texpr, val_texpr))

    def _infer_tuple_type(
        self,
        value: tuple[Any, ...],
        declared_hint: Any,
        path: tuple[str, ...],
        field_name: str,
    ) -> TExpr:
        """Infer type for a tuple value."""
        origin = get_origin(declared_hint)
        declared_args = get_args(declared_hint) if origin is tuple else ()

        # Variable-length tuple (T, ...)
        varlen_tuple_arity = 2
        if len(declared_args) == varlen_tuple_arity and declared_args[1] is Ellipsis:
            elem_hint = declared_args[0]
            if value:
                elem_texpr = self._infer_type(
                    value[0],
                    elem_hint,
                    path,
                    f"{field_name}[0]",
                )
            else:
                elem_texpr = from_hint(elem_hint)
            return TCon(tuple, (elem_texpr, TCon(type(...))))

        # Fixed-length tuple
        elem_texprs: list[TExpr] = []
        for i, elem in enumerate(value):
            elem_hint = declared_args[i] if i < len(declared_args) else None
            elem_texpr = self._infer_type(elem, elem_hint, path, f"{field_name}[{i}]")
            elem_texprs.append(elem_texpr)
        return TCon(tuple, tuple(elem_texprs))

    def _infer_set_type(
        self,
        value: set[Any] | frozenset[Any],
        declared_hint: Any,
        path: tuple[str, ...],
        field_name: str,
        container_type: type,
    ) -> TExpr:
        """Infer type for a set or frozenset value."""
        elem_hint = _get_elem_hint(declared_hint, container_type)

        if not value:
            if elem_hint is not None:
                elem_texpr = from_hint(elem_hint)
            else:
                elem_texpr = self.var_factory.fresh()
        else:
            first_elem = next(iter(value))
            elem_texpr = self._infer_type(
                first_elem,
                elem_hint,
                path,
                f"{field_name}.elem",
            )

        return TCon(container_type, (elem_texpr,))

    def _infer_type(
        self,
        value: Any,
        declared_hint: Any,
        path: tuple[str, ...],
        field_name: str,
    ) -> TExpr:
        """Infer the type of a value."""
        if value is None:
            return TCon(type(None))

        if isinstance(value, Ref):
            return self._visit_ref(value, path=(*path, field_name))

        if isinstance(value, Node):
            return self._visit_node(value, node_id=None, path=(*path, field_name))

        if isinstance(value, list):
            return self._infer_list_type(value, declared_hint, path, field_name)

        if isinstance(value, dict):
            return self._infer_dict_type(value, declared_hint, path, field_name)

        if isinstance(value, tuple):
            return self._infer_tuple_type(value, declared_hint, path, field_name)

        if isinstance(value, frozenset):
            return self._infer_set_type(
                value,
                declared_hint,
                path,
                field_name,
                frozenset,
            )

        if isinstance(value, set):
            return self._infer_set_type(value, declared_hint, path, field_name, set)

        return TCon(type(value))


def generate_constraints(program: Program) -> list[Constraint]:
    """Generate type constraints for a program.

    Args:
        program: The program to generate constraints for.

    Returns:
        A list of type constraints.

    """
    generator = ConstraintGenerator(program)
    return generator.generate()
