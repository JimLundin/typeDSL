"""Constraint generation from Programs.

This module traverses a Program and emits type equality constraints
by comparing declared field types against actual values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin, get_type_hints

from typedsl.checker.constraints import (
    Constraint,
    EqualityConstraint,
    Location,
    SubtypeConstraint,
)
from typedsl.checker.types import TCon, TExpr, TVarFactory, from_hint
from typedsl.checker.types import TVar as CheckerTVar
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
        self.constraints: list[EqualityConstraint] = []
        self._node_return_types: dict[str, TExpr] = {}
        self._visited: set[int] = set()

    def generate(self) -> list[EqualityConstraint]:
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
        """Pre-allocate TVars for named nodes with TypeVar return types.

        For nodes with concrete return types, we compute them on first visit.
        For nodes with TypeVar return types, we allocate a TVar placeholder
        that will be unified when the node is visited.
        """
        for node_id, node in self.program.nodes.items():
            return_type_hint = get_node_return_type(type(node))
            # Check if the return type involves a TypeVar
            is_concrete = return_type_hint is not None and not isinstance(
                return_type_hint,
                TypeVar,
            )
            if is_concrete:
                self._node_return_types[node_id] = from_hint(return_type_hint)
            else:
                # TypeVar or unknown - allocate a fresh TVar
                self._node_return_types[node_id] = self.var_factory.fresh()

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

        # Create a fresh TypeVar -> TVar mapping for this node
        typevar_map: dict[TypeVar, CheckerTVar] = {}

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

            declared_texpr = from_hint(declared_type, typevar_map, self.var_factory)
            actual_texpr = self._infer_type(value, declared_type, path, field_name)

            self.constraints.append(
                EqualityConstraint(
                    left=declared_texpr,
                    right=actual_texpr,
                    location=location,
                ),
            )

        # Get return type using the same TypeVar mapping
        return_type_hint = get_node_return_type(node_class)
        if return_type_hint is not None:
            return_type = from_hint(return_type_hint, typevar_map, self.var_factory)
        else:
            return_type = self.var_factory.fresh()

        # If this is a named node, emit constraint to unify with pre-allocated type
        if node_id is not None and node_id in self._node_return_types:
            pre_allocated = self._node_return_types[node_id]
            location = Location(
                node_tag=node_tag,
                node_id=node_id,
                field_name=None,
                path=path,
            )
            self.constraints.append(
                EqualityConstraint(
                    left=pre_allocated,
                    right=return_type,
                    location=location,
                ),
            )

        return TCon(Node, (return_type,))

    def _infer_single_elem_container_type(
        self,
        value: Any,
        declared_hint: Any,
        path: tuple[str, ...],
        field_name: str,
        container_type: type,
    ) -> TExpr:
        """Infer type for single-element containers (list, set, frozenset)."""
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
                f"{field_name}[0]",
            )

        return TCon(container_type, (elem_texpr,))

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

        # Single-element containers: list, set, frozenset
        if isinstance(value, (list, set, frozenset)):
            return self._infer_single_elem_container_type(
                value,
                declared_hint,
                path,
                field_name,
                type(value),
            )

        if isinstance(value, dict):
            return self._infer_dict_type(value, declared_hint, path, field_name)

        if isinstance(value, tuple):
            return self._infer_tuple_type(value, declared_hint, path, field_name)

        return TCon(type(value))


def generate_constraints(program: Program) -> list[Constraint]:
    """Generate type constraints for a program.

    Args:
        program: The program to generate constraints for.

    Returns:
        A list of constraints (both equality and subtype constraints).

    """
    generator = ConstraintGenerator(program)
    constraints: list[Constraint] = list(generator.generate())

    # Convert bounds to SubtypeConstraints
    for var_id, bound_types in generator.var_factory.bounds.items():
        # Use a generic location for bound constraints
        location = Location(
            node_tag="<bound>",
            node_id=None,
            field_name=None,
            path=(),
        )
        constraints.append(
            SubtypeConstraint(
                type_var=CheckerTVar(var_id),
                allowed_types=bound_types,
                location=location,
            ),
        )

    return constraints
