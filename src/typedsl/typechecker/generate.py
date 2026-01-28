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
    Constraint,
    EqConstraint,
    Location,
    SubConstraint,
    TBot,
    TCon,
    TExp,
    TTop,
    TVar,
)

if TYPE_CHECKING:
    from typedsl.ast import Program

_DICT_TYPE_ARITY = 2


def _resolve_type_alias(py_type: Any) -> Any:
    """Resolve TypeAliasType to its underlying type.

    Python 3.12+ introduces TypeAliasType for `type` keyword aliases.
    This function resolves parameterized type aliases like `Child[int]`
    to their underlying union type.
    """
    origin = get_origin(py_type)

    # Check if origin is a TypeAliasType (has __value__ attribute)
    if origin is not None and hasattr(origin, "__value__"):
        # Get the underlying type and substitute type args
        underlying = origin.__value__
        type_args = get_args(py_type)
        type_params = getattr(origin, "__type_params__", ())

        if type_args and type_params:
            # Build substitution map and apply to underlying type
            subs = dict(zip(type_params, type_args, strict=False))
            return _substitute_type_params(underlying, subs)
        return underlying

    return py_type


def _substitute_type_params(py_type: Any, subs: dict[Any, Any]) -> Any:
    """Substitute type parameters in a type expression."""
    # Check if this is a type parameter we should substitute
    if py_type in subs:
        return subs[py_type]

    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is None:
        return py_type

    # Recursively substitute in type arguments
    if args:
        new_args = tuple(_substitute_type_params(a, subs) for a in args)

        # Handle typing.Union (from typing module)
        if origin is Union:
            # Reconstruct using Union[args] - noqa needed as we're building dynamically
            return Union[new_args]  # noqa: UP007

        # Handle types.UnionType (from | syntax in Python 3.10+)
        if isinstance(py_type, types.UnionType):
            result = new_args[0]
            for arg in new_args[1:]:
                result = result | arg
            return result

        # Reconstruct other parameterized types
        if hasattr(origin, "__class_getitem__") and new_args:
            # pyright doesn't understand new_args is non-empty from the guard
            return origin[new_args] if len(new_args) > 1 else origin[new_args[0]]  # pyright: ignore[reportGeneralTypeIssues]

    return py_type


class ConstraintGenerator:
    """Generates type constraints from Programs and Nodes."""

    def __init__(self, program: Program) -> None:
        self._program = program
        self._counter = itertools.count()
        self._constraints: list[Constraint] = []
        self._node_return_types: dict[str, TExp] = {}
        self._type_param_map: dict[str, TVar] = {}

    def fresh_var(self, base_name: str) -> TVar:
        """Create a fresh type variable with a unique name."""
        unique_name = f"{base_name}${next(self._counter)}"
        return TVar(unique_name)

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
                self.add(EqConstraint(TBot(), TTop(), root_loc))
        else:
            self._generate_node(self._program.root, root_loc)

        return self._constraints

    def _generate_node(self, node: Node[Any], loc: Location) -> TExp:
        """Generate constraints for a single node and return its inferred type."""
        parent_map = self._type_param_map
        self._type_param_map = {}

        node_cls = type(node)
        hints = get_type_hints(node_cls)

        if hasattr(node_cls, "__type_params__"):
            for param in node_cls.__type_params__:
                fresh = self.fresh_var(param.__name__)
                self._type_param_map[param.__name__] = fresh

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

        self._type_param_map = parent_map
        return result

    def _generate_field_constraint(
        self,
        value: Any,
        declared_type: Any,
        loc: Location,
    ) -> None:
        """Generate constraints for a field value against its declared type."""
        # Resolve type aliases like Child[T] to their underlying types
        resolved_type = _resolve_type_alias(declared_type)

        origin = get_origin(resolved_type)
        args = get_args(resolved_type)

        is_node_field = (
            origin is not None and isinstance(origin, type) and issubclass(origin, Node)
        )

        # Handle union types (like Node[T] | Ref[Node[T]])
        if isinstance(resolved_type, types.UnionType) or origin is Union:
            self._generate_union_field_constraint(value, resolved_type, loc)
        elif is_node_field and isinstance(value, Node):
            self._generate_node_field_constraint(value, origin, args, loc)
        elif is_node_field and isinstance(value, Ref):
            self._generate_ref_field_constraint(value, origin, args, loc)
        else:
            expected_type = self._python_type_to_type(declared_type)
            actual_type = self._infer_value_type(value, expected_type, loc)
            self.add(SubConstraint(actual_type, expected_type, loc))

    def _generate_union_field_constraint(
        self,
        value: Any,
        union_type: Any,
        loc: Location,
    ) -> None:
        """Generate constraints for a value against a union type.

        For unions like Node[T] | Ref[Node[T]], we find the appropriate member
        type based on the value's actual type and apply that constraint.
        """
        union_args = get_args(union_type)

        # Find matching union member for the value
        for member in union_args:
            member_origin = get_origin(member)
            member_args = get_args(member)

            if isinstance(value, Node):
                is_node_member = (
                    member_origin is not None
                    and isinstance(member_origin, type)
                    and issubclass(member_origin, Node)
                )
                if is_node_member:
                    self._generate_node_field_constraint(
                        value,
                        member_origin,
                        member_args,
                        loc,
                    )
                    return
                # Also check for bare Node type
                if isinstance(member, type) and issubclass(member, Node):
                    self._generate_node_field_constraint(value, member, (), loc)
                    return

            if isinstance(value, Ref):
                is_ref_member = member_origin is Ref
                if is_ref_member:
                    # For Ref[Node[T]], extract the Node type parameter
                    if member_args:
                        inner = member_args[0]
                        inner_origin = get_origin(inner)
                        inner_args = get_args(inner)
                        if (
                            inner_origin is not None
                            and isinstance(
                                inner_origin,
                                type,
                            )
                            and issubclass(inner_origin, Node)
                        ):
                            self._generate_ref_field_constraint(
                                value,
                                inner_origin,
                                inner_args,
                                loc,
                            )
                            return
                        if isinstance(inner, type) and issubclass(inner, Node):
                            self._generate_ref_field_constraint(value, inner, (), loc)
                            return
                    # Bare Ref without Node type
                    self._generate_ref_field_constraint(value, Node, (), loc)
                    return

        # Fall back to generic constraint if no matching member found
        expected_type = self._python_type_to_type(union_type)
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
                    SubConstraint(TCon(actual_cls, ()), TCon(expected_origin, ()), loc),
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
            self.add(EqConstraint(TBot(), TTop(), loc))
            return

        if args:
            expected_return_type = self._python_type_to_type(args[0])
            self.add(SubConstraint(actual_return_type, expected_return_type, loc))

        if expected_origin is not Node and resolved_node is not None:
            actual_cls = type(resolved_node)
            if not issubclass(actual_cls, expected_origin):
                self.add(
                    SubConstraint(TCon(actual_cls, ()), TCon(expected_origin, ()), loc),
                )

    def _python_type_to_type(self, py_type: Any) -> TExp:
        """Convert a Python type annotation to our TExp representation."""
        # Resolve TypeAliasType (from `type` keyword in Python 3.12+)
        py_type = _resolve_type_alias(py_type)

        if py_type is None or py_type is type(None):
            return TCon(type(None), ())

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
            return TTop()

        if origin is Ref:
            if args:
                target_type = self._python_type_to_type(args[0])
                return TCon(Ref, (target_type,))
            return TCon(Ref, ())

        if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
            if args:
                converted_args = tuple(self._python_type_to_type(a) for a in args)
                return TCon(origin, converted_args)
            return TCon(origin, ())

        if origin is not None:
            converted_args = tuple(self._python_type_to_type(a) for a in args)
            return TCon(origin, converted_args)

        if isinstance(py_type, type):
            return TCon(py_type, ())

        return TTop()

    def _infer_value_type(self, value: Any, expected: TExp, loc: Location) -> TExp:
        """Infer the type of a runtime value."""
        if value is None:
            return TCon(type(None), ())

        if isinstance(value, Node):
            return_type = self._generate_node(value, loc)
            return TCon(Node, (return_type,))

        if isinstance(value, Ref):
            if isinstance(expected, TCon) and expected.con is Node:
                return expected
            if isinstance(expected, TCon) and expected.con is Ref:
                return expected
            fresh_target = self.fresh_var("RefTarget")
            return TCon(Ref, (fresh_target,))

        if isinstance(value, list):
            if not value:
                if isinstance(expected, TCon) and expected.args:
                    return expected
                return TCon(list, (self.fresh_var("ListElem"),))
            elem_type = self._infer_value_type(value[0], TTop(), loc.index(0))
            return TCon(list, (elem_type,))

        if isinstance(value, dict):
            if not value:
                has_key_value_types = (
                    isinstance(expected, TCon)
                    and len(expected.args) >= _DICT_TYPE_ARITY
                )
                if has_key_value_types:
                    return expected
                return TCon(
                    dict,
                    (self.fresh_var("DictKey"), self.fresh_var("DictVal")),
                )
            k, v = next(iter(value.items()))
            key_type = self._infer_value_type(k, TTop(), loc.child("key"))
            val_type = self._infer_value_type(v, TTop(), loc.child("val"))
            return TCon(dict, (key_type, val_type))

        if isinstance(value, set):
            if not value:
                if isinstance(expected, TCon) and expected.args:
                    return expected
                return TCon(set, (self.fresh_var("SetElem"),))
            elem = next(iter(value))
            elem_type = self._infer_value_type(elem, TTop(), loc.child("elem"))
            return TCon(set, (elem_type,))

        if isinstance(value, tuple):
            elem_types = tuple(
                self._infer_value_type(v, TTop(), loc.index(i))
                for i, v in enumerate(value)
            )
            return TCon(tuple, elem_types)

        return TCon(type(value), ())

    def _get_return_type(self, node_cls: type[Node[Any]]) -> TExp:
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

        return TTop()


def generate_constraints(program: Program) -> list[Constraint]:
    """Generate type constraints for a Program."""
    generator = ConstraintGenerator(program)
    return generator.generate()
