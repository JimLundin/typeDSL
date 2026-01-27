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

from typedsl.nodes import Node, Ref
from typedsl.typechecker.core import (
    Bottom,
    Constraint,
    EqConstraint,
    Location,
    SubConstraint,
    Top,
    Type,
    TypeCon,
    TypeVar,
)

if TYPE_CHECKING:
    from typedsl.ast import Program

# Dict types have 2 type parameters: key and value
_DICT_TYPE_ARITY = 2


class ConstraintGenerator:
    """Generates type constraints from Programs and Nodes.

    Handles fresh type variable generation to avoid name collisions
    between different generic node instances.
    """

    def __init__(self, program: Program) -> None:
        self._program = program
        self._counter = itertools.count()
        self._constraints: list[Constraint] = []
        self._node_return_types: dict[str, Type] = {}
        self._type_param_map: dict[str, TypeVar] = {}

    def fresh_var(self, base_name: str) -> TypeVar:
        """Create a fresh type variable with a unique name."""
        unique_name = f"{base_name}${next(self._counter)}"
        return TypeVar(unique_name)

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the list."""
        self._constraints.append(constraint)

    def generate(self) -> list[Constraint]:
        """Generate constraints for the program."""
        nodes_loc = Location("nodes")

        for node_id, node in self._program.nodes.items():
            node_loc = nodes_loc.index(node_id)
            return_type = self._generate_node(node, node_loc)
            self._node_return_types[node_id] = return_type

        root_loc = Location("root")
        if isinstance(self._program.root, Ref):
            if self._program.root.id not in self._program.nodes:
                self.add(EqConstraint(Bottom(), Top(), root_loc))
        else:
            self._generate_node(self._program.root, root_loc)

        return self._constraints

    def _generate_node(self, node: Node[Any], loc: Location) -> Type:
        """Generate constraints for a single node and return its inferred type."""
        # Save parent's type param map and create fresh one for this node
        parent_map = self._type_param_map
        self._type_param_map = {}

        node_cls = type(node)
        hints = get_type_hints(node_cls)

        # Create fresh type variables for this node's type parameters
        if hasattr(node_cls, "__type_params__"):
            for param in node_cls.__type_params__:
                fresh = self.fresh_var(param.__name__)
                self._type_param_map[param.__name__] = fresh

        # Process each field
        for f in fields(node_cls):
            if f.name.startswith("_"):
                continue

            field_loc = loc.child(f.name)
            field_value = getattr(node, f.name)
            declared_type = hints.get(f.name)

            if declared_type is None:
                continue

            self._generate_field_constraint(field_value, declared_type, field_loc)

        result = self._get_return_type(node_cls)

        # Restore parent's type param map
        self._type_param_map = parent_map
        return result

    def _generate_field_constraint(
        self,
        value: Any,
        declared_type: Any,
        loc: Location,
    ) -> None:
        """Generate constraints for a field value against its declared type."""
        origin = get_origin(declared_type)
        args = get_args(declared_type)

        is_node_field = (
            origin is not None and isinstance(origin, type) and issubclass(origin, Node)
        )

        if is_node_field and isinstance(value, Node):
            self._generate_node_field_constraint(value, origin, args, loc)
        elif is_node_field and isinstance(value, Ref):
            self._generate_ref_field_constraint(value, origin, args, loc)
        else:
            expected_type = self._python_type_to_type(declared_type)
            actual_type = self._infer_value_type(value, expected_type, loc)
            self.add(SubConstraint(actual_type, expected_type, loc))

    def _generate_node_field_constraint(
        self,
        value: Node[Any],
        expected_origin: type,
        args: tuple[Any, ...],
        loc: Location,
    ) -> None:
        """Generate constraints for a Node value in a node field."""
        actual_return_type = self._generate_node(value, loc)

        if args:
            expected_return_type = self._python_type_to_type(args[0])
            self.add(SubConstraint(actual_return_type, expected_return_type, loc))

        if expected_origin is not Node:
            actual_cls = type(value)
            if not issubclass(actual_cls, expected_origin):
                self.add(
                    SubConstraint(
                        TypeCon(actual_cls, ()),
                        TypeCon(expected_origin, ()),
                        loc,
                    ),
                )

    def _generate_ref_field_constraint(
        self,
        ref: Ref[Any],
        expected_origin: type,
        args: tuple[Any, ...],
        loc: Location,
    ) -> None:
        """Generate constraints for a Ref value in a node field."""
        ref_id = ref.id

        if ref_id in self._node_return_types:
            actual_return_type = self._node_return_types[ref_id]
            resolved_node = self._program.nodes.get(ref_id)
        elif ref_id in self._program.nodes:
            resolved_node = self._program.nodes[ref_id]
            ref_loc = Location("nodes").index(ref_id)
            actual_return_type = self._generate_node(resolved_node, ref_loc)
            self._node_return_types[ref_id] = actual_return_type
        else:
            self.add(EqConstraint(Bottom(), Top(), loc))
            return

        if args:
            expected_return_type = self._python_type_to_type(args[0])
            self.add(SubConstraint(actual_return_type, expected_return_type, loc))

        if expected_origin is not Node and resolved_node is not None:
            actual_cls = type(resolved_node)
            if not issubclass(actual_cls, expected_origin):
                self.add(
                    SubConstraint(
                        TypeCon(actual_cls, ()),
                        TypeCon(expected_origin, ()),
                        loc,
                    ),
                )

    def _python_type_to_type(self, py_type: Any) -> Type:
        """Convert a Python type annotation to our Type representation."""
        if py_type is None or py_type is type(None):
            return TypeCon(type(None), ())

        if isinstance(py_type, TypingTypeVar):
            name = py_type.__name__
            if name in self._type_param_map:
                return self._type_param_map[name]
            fresh = self.fresh_var(name)
            self._type_param_map[name] = fresh
            return fresh

        origin = get_origin(py_type)
        args = get_args(py_type)

        if origin is Union or isinstance(py_type, types.UnionType):
            return Top()

        if origin is Ref:
            if args:
                target_type = self._python_type_to_type(args[0])
                return TypeCon(Ref, (target_type,))
            return TypeCon(Ref, ())

        if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
            if args:
                converted_args = tuple(self._python_type_to_type(a) for a in args)
                return TypeCon(origin, converted_args)
            return TypeCon(origin, ())

        if origin is not None:
            converted_args = tuple(self._python_type_to_type(a) for a in args)
            return TypeCon(origin, converted_args)

        if isinstance(py_type, type):
            return TypeCon(py_type, ())

        return Top()

    def _infer_value_type(self, value: Any, expected: Type, loc: Location) -> Type:
        """Infer the type of a runtime value."""
        if value is None:
            return TypeCon(type(None), ())

        if isinstance(value, Node):
            return_type = self._generate_node(value, loc)
            return TypeCon(Node, (return_type,))

        if isinstance(value, Ref):
            if isinstance(expected, TypeCon) and expected.constructor is Node:
                return expected
            if isinstance(expected, TypeCon) and expected.constructor is Ref:
                return expected
            fresh_target = self.fresh_var("RefTarget")
            return TypeCon(Ref, (fresh_target,))

        if isinstance(value, list):
            if not value:
                if isinstance(expected, TypeCon) and expected.args:
                    return expected
                return TypeCon(list, (self.fresh_var("ListElem"),))
            elem_type = self._infer_value_type(value[0], Top(), loc.index(0))
            return TypeCon(list, (elem_type,))

        if isinstance(value, dict):
            if not value:
                has_key_value_types = (
                    isinstance(expected, TypeCon)
                    and len(expected.args) >= _DICT_TYPE_ARITY
                )
                if has_key_value_types:
                    return expected
                return TypeCon(
                    dict,
                    (self.fresh_var("DictKey"), self.fresh_var("DictVal")),
                )
            k, v = next(iter(value.items()))
            key_type = self._infer_value_type(k, Top(), loc.child("key"))
            val_type = self._infer_value_type(v, Top(), loc.child("val"))
            return TypeCon(dict, (key_type, val_type))

        if isinstance(value, set):
            if not value:
                if isinstance(expected, TypeCon) and expected.args:
                    return expected
                return TypeCon(set, (self.fresh_var("SetElem"),))
            elem = next(iter(value))
            elem_type = self._infer_value_type(elem, Top(), loc.child("elem"))
            return TypeCon(set, (elem_type,))

        if isinstance(value, tuple):
            elem_types = tuple(
                self._infer_value_type(v, Top(), loc.index(i))
                for i, v in enumerate(value)
            )
            return TypeCon(tuple, elem_types)

        return TypeCon(type(value), ())

    def _get_return_type(self, node_cls: type[Node[Any]]) -> Type:
        """Get the return type of a node class."""
        for base in getattr(node_cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is None:
                continue
            if isinstance(origin, type) and issubclass(origin, Node):
                args = get_args(base)
                if args:
                    return_type = args[0]
                    if isinstance(return_type, TypingTypeVar):
                        name = return_type.__name__
                        if name in self._type_param_map:
                            return self._type_param_map[name]
                    return self._python_type_to_type(return_type)

        return Top()


def generate_constraints(program: Program) -> list[Constraint]:
    """Generate type constraints for a Program."""
    generator = ConstraintGenerator(program)
    return generator.generate()
