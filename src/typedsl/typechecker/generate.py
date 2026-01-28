"""Constraint generation for type checking Programs and Nodes.

This module implements a two-phase approach to constraint generation:

1. Schema extraction (from CLASS): Extract type parameters and their
   structural relationships from the node class definition.

2. Instance binding (from INSTANCE): Generate constraints by matching
   actual values against the schema's expected types.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, fields
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
    Location,
    SubConstraint,
    TCon,
    TExp,
    TTop,
    TVar,
)

if TYPE_CHECKING:
    from typedsl.ast import Program

_DICT_TYPE_ARITY = 2


@dataclass(frozen=True)
class FieldSchema:
    """Schema for a single field."""

    name: str
    expected_type: TExp
    python_type: Any  # Original Python type annotation


@dataclass(frozen=True)
class NodeSchema:
    """Type schema extracted from a node class.

    This represents the type structure of a node class, independent of
    any particular instance. It captures:
    - Type variables declared on the class
    - Structural constraints between type variables (e.g., T == list[V])
    - Expected types for each field
    - The return type of the node
    """

    type_vars: dict[str, TVar]
    structural_constraints: tuple[Constraint, ...]
    fields: tuple[FieldSchema, ...]
    return_type: TExp


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


class _SchemaExtractor:
    """Helper class for extracting node schemas."""

    def __init__(self, loc: Location) -> None:
        self._loc = loc
        self.type_vars: dict[str, TVar] = {}

    def fresh_var(self, name: str) -> TVar:
        """Create a location-unique type variable."""
        return TVar(f"{name}@{self._loc.path}")

    def python_type_to_texp(self, py_type: Any) -> TExp:
        """Convert Python type to TExp, using existing type vars."""
        py_type = _resolve_type_alias(py_type)

        if py_type is None or py_type is type(None):
            return TCon(type(None), ())

        if isinstance(py_type, TypingTypeVar):
            name = py_type.__name__
            if name in self.type_vars:
                return self.type_vars[name]
            fresh = self.fresh_var(name)
            self.type_vars[name] = fresh
            return fresh

        origin = get_origin(py_type)
        args = get_args(py_type)

        if origin is Union or isinstance(py_type, types.UnionType):
            return TTop()

        if origin is Ref:
            if args:
                target_type = self.python_type_to_texp(args[0])
                return TCon(Ref, (target_type,))
            return TCon(Ref, ())

        if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
            if args:
                converted_args = tuple(self.python_type_to_texp(a) for a in args)
                return TCon(origin, converted_args)
            return TCon(origin, ())

        if origin is not None:
            converted_args = tuple(self.python_type_to_texp(a) for a in args)
            return TCon(origin, converted_args)

        if isinstance(py_type, type):
            return TCon(py_type, ())

        return TTop()


def _extract_type_params(
    node_cls: type[Node[Any]],
    extractor: _SchemaExtractor,
    loc: Location,
) -> list[Constraint]:
    """Extract type parameters and their bound constraints."""
    constraints: list[Constraint] = []

    if hasattr(node_cls, "__type_params__"):
        for param in node_cls.__type_params__:
            param_name = param.__name__
            fresh = extractor.fresh_var(param_name)
            extractor.type_vars[param_name] = fresh

            # Extract bound constraints (e.g., T: list[V] produces T <: list[V])
            bound = getattr(param, "__bound__", None)
            if bound is not None:
                bound_type = extractor.python_type_to_texp(bound)
                constraints.append(SubConstraint(fresh, bound_type, loc))

    return constraints


def _extract_field_schemas(
    node_cls: type[Node[Any]],
    extractor: _SchemaExtractor,
) -> list[FieldSchema]:
    """Extract field schemas from node class."""
    hints = get_type_hints(node_cls)
    field_schemas: list[FieldSchema] = []

    for f in fields(node_cls):
        if f.name.startswith("_"):
            continue
        declared_type = hints.get(f.name)
        if declared_type is None:
            continue
        expected = extractor.python_type_to_texp(declared_type)
        field_schemas.append(FieldSchema(f.name, expected, declared_type))

    return field_schemas


def _extract_return_type(
    node_cls: type[Node[Any]],
    extractor: _SchemaExtractor,
) -> TExp:
    """Extract return type from node class."""
    for base in getattr(node_cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is None:
            continue
        if isinstance(origin, type) and issubclass(origin, Node):
            args = get_args(base)
            if args:
                ret = args[0]
                if isinstance(ret, TypingTypeVar):
                    name = ret.__name__
                    if name in extractor.type_vars:
                        return extractor.type_vars[name]
                else:
                    return extractor.python_type_to_texp(ret)
            break

    return TTop()


def extract_schema(node_cls: type[Node[Any]], loc: Location) -> NodeSchema:
    """Extract type schema from a node class (Phase 1).

    This function analyzes the class definition to extract:
    - Fresh type variables for each type parameter
    - Structural constraints from type parameter bounds
    - Expected types for each field
    - The return type

    This is independent of any instance - it only looks at the class.

    Args:
        node_cls: The node class to extract schema from
        loc: Location for generating unique type variable names

    Returns:
        NodeSchema containing type structure information

    """
    extractor = _SchemaExtractor(loc)

    # Phase 1a: Extract type parameters and their constraints
    structural_constraints = _extract_type_params(node_cls, extractor, loc)

    # Phase 1b: Extract field schemas
    field_schemas = _extract_field_schemas(node_cls, extractor)

    # Phase 1c: Extract return type
    return_type = _extract_return_type(node_cls, extractor)

    return NodeSchema(
        type_vars=extractor.type_vars,
        structural_constraints=tuple(structural_constraints),
        fields=tuple(field_schemas),
        return_type=return_type,
    )


def infer_value_type(value: Any, expected: TExp, loc: Location) -> TExp:
    """Infer the TExp type from a runtime value (Phase 2 helper).

    This is a pure function that infers type from value structure.
    For simple values, it returns the concrete type.
    For containers, it infers element types recursively.

    Note: For Node and Ref values, this returns a placeholder - the actual
    constraint generation for these requires program context and is handled
    separately by NodeConstraintGenerator.

    Args:
        value: The runtime value to infer type from
        expected: The expected type (used for empty containers)
        loc: Location for generating unique type variable names

    Returns:
        The inferred TExp type

    """
    if value is None:
        return TCon(type(None), ())

    # For Node/Ref, return placeholder - needs special handling
    if isinstance(value, (Node, Ref)):
        return TTop()  # Placeholder - handled by NodeConstraintGenerator

    if isinstance(value, list):
        if not value:
            if isinstance(expected, TCon) and expected.args:
                return expected
            return TCon(list, (TVar(f"ListElem@{loc.path}"),))
        elem_type = infer_value_type(value[0], TTop(), loc.index(0))
        return TCon(list, (elem_type,))

    if isinstance(value, dict):
        if not value:
            has_key_value_types = (
                isinstance(expected, TCon) and len(expected.args) >= _DICT_TYPE_ARITY
            )
            if has_key_value_types:
                return expected
            return TCon(
                dict,
                (TVar(f"DictKey@{loc.path}"), TVar(f"DictVal@{loc.path}")),
            )
        k, v = next(iter(value.items()))
        key_type = infer_value_type(k, TTop(), loc.child("key"))
        val_type = infer_value_type(v, TTop(), loc.child("val"))
        return TCon(dict, (key_type, val_type))

    if isinstance(value, set):
        if not value:
            if isinstance(expected, TCon) and expected.args:
                return expected
            return TCon(set, (TVar(f"SetElem@{loc.path}"),))
        elem = next(iter(value))
        elem_type = infer_value_type(elem, TTop(), loc.child("elem"))
        return TCon(set, (elem_type,))

    if isinstance(value, tuple):
        elem_types = tuple(
            infer_value_type(v, TTop(), loc.index(i)) for i, v in enumerate(value)
        )
        return TCon(tuple, elem_types)

    return TCon(type(value), ())


def bind_instance(
    schema: NodeSchema,
    node: Node[Any],
    loc: Location,
) -> list[Constraint]:
    """Generate binding constraints from instance values (Phase 2).

    This function compares actual values against the schema's expected types
    and generates constraints that will bind type variables to concrete types.

    For example, if schema expects field `values: T` and the instance has
    `values=[1,2,3]`, this generates `T <: list[int]` which binds T.

    Note: This only handles primitive fields. Node/Ref fields require
    program context and are handled separately.

    Args:
        schema: The NodeSchema extracted from the class
        node: The node instance
        loc: Location for constraint generation

    Returns:
        List of binding constraints

    """
    constraints: list[Constraint] = []

    for field_schema in schema.fields:
        field_value = getattr(node, field_schema.name)
        field_loc = loc.child(field_schema.name)

        # Skip Node/Ref - handled with program context
        if isinstance(field_value, (Node, Ref)):
            continue

        # For primitive/container fields, infer actual type and constrain
        expected = field_schema.expected_type
        actual_type = infer_value_type(field_value, expected, field_loc)

        # Skip if actual type is placeholder (Node/Ref in container)
        if isinstance(actual_type, TTop):
            continue

        constraints.append(
            SubConstraint(actual_type, field_schema.expected_type, field_loc),
        )

    return constraints


class NodeConstraintGenerator:
    """Generates type constraints for a single Node using two-phase approach.

    Phase 1: Extract schema from the node CLASS (type vars, structural constraints)
    Phase 2: Bind instance VALUES to generate binding constraints

    This class handles Node/Ref fields specially since they require program context.
    Primitive/container fields use the pure bind_instance function.
    """

    def __init__(
        self,
        loc: Location,
        program: Program,
        return_type_cache: dict[str, TExp],
    ) -> None:
        """Initialize the generator for a specific node location.

        Args:
            loc: The location of the node in the program
            program: The program (for resolving refs)
            return_type_cache: Shared cache of node return types

        """
        self._loc = loc
        self._program = program
        self._return_type_cache = return_type_cache
        self._constraints: list[Constraint] = []
        self._schema: NodeSchema | None = None

    def fresh_var(self, base_name: str) -> TVar:
        """Create a type variable unique to this node location."""
        unique_name = f"{base_name}@{self._loc.path}"
        return TVar(unique_name)

    def _add(self, constraint: Constraint) -> None:
        """Add a constraint to the list."""
        self._constraints.append(constraint)

    def generate(self, node: Node[Any]) -> tuple[list[Constraint], TExp]:
        """Generate constraints for the node using two-phase approach.

        Phase 1: Extract schema from CLASS (type vars, structural constraints)
        Phase 2: Generate binding constraints from INSTANCE values

        Returns:
            A tuple of (constraints, return_type)

        """
        node_cls = type(node)

        # Phase 1: Extract schema from the class
        self._schema = extract_schema(node_cls, self._loc)

        # Add structural constraints from schema
        self._constraints.extend(self._schema.structural_constraints)

        # Phase 2: Generate binding constraints from instance
        # First, handle primitive/container fields with bind_instance
        binding_constraints = bind_instance(self._schema, node, self._loc)
        self._constraints.extend(binding_constraints)

        # Then, handle Node/Ref fields (requires program context)
        for field_schema in self._schema.fields:
            field_value = getattr(node, field_schema.name)
            if isinstance(field_value, (Node, Ref)):
                self._generate_child_constraint(
                    field_value,
                    field_schema.python_type,
                    self._loc.child(field_schema.name),
                )

        return self._constraints, self._schema.return_type

    def _generate_child_constraint(
        self,
        value: Node[Any] | Ref[Any],
        declared_type: Any,
        loc: Location,
    ) -> None:
        """Generate constraints for Node/Ref field values."""
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
        elif isinstance(value, Node):
            self._generate_node_field_constraint(value, Node, (), loc)
        elif isinstance(value, Ref):
            self._generate_ref_field_constraint(value, Node, (), loc)

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
                            and isinstance(inner_origin, type)
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
        self._add(SubConstraint(actual_type, expected_type, loc))

    def _generate_node_field_constraint(
        self,
        value: Node[Any],
        expected_origin: type,
        args: tuple[Any, ...],
        loc: Location,
    ) -> None:
        """Generate constraints for a Node value in a node field."""
        # Generate constraints for the nested node
        child_gen = NodeConstraintGenerator(loc, self._program, self._return_type_cache)
        child_constraints, actual_return_type = child_gen.generate(value)
        self._constraints.extend(child_constraints)

        if args:
            expected_return_type = self._python_type_to_type(args[0])
            self._add(SubConstraint(actual_return_type, expected_return_type, loc))

        if expected_origin is not Node:
            actual_cls = type(value)
            if not issubclass(actual_cls, expected_origin):
                self._add(
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

        # Check cache first, otherwise resolve and generate constraints
        if ref_id in self._return_type_cache:
            actual_return_type = self._return_type_cache[ref_id]
        else:
            ref_loc = Location("nodes").index(ref_id)
            resolved_node = self._program.resolve(ref)
            child_gen = NodeConstraintGenerator(
                ref_loc,
                self._program,
                self._return_type_cache,
            )
            child_constraints, actual_return_type = child_gen.generate(resolved_node)
            self._constraints.extend(child_constraints)
            self._return_type_cache[ref_id] = actual_return_type

        if args:
            expected_return_type = self._python_type_to_type(args[0])
            self._add(SubConstraint(actual_return_type, expected_return_type, loc))

        # Check if resolved node class matches expected origin
        if expected_origin is not Node:
            resolved_node = self._program.resolve(ref)
            actual_cls = type(resolved_node)
            if not issubclass(actual_cls, expected_origin):
                self._add(
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
            # Use type vars from schema if available
            if self._schema and name in self._schema.type_vars:
                return self._schema.type_vars[name]
            # Fallback to fresh var
            return self.fresh_var(name)

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
            child_gen = NodeConstraintGenerator(
                loc,
                self._program,
                self._return_type_cache,
            )
            child_constraints, return_type = child_gen.generate(value)
            self._constraints.extend(child_constraints)
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


class ConstraintGenerator:
    """Generates type constraints from Programs.

    This class handles program-level concerns:
    - Iterating through program nodes
    - Managing the return type cache
    - Coordinating NodeConstraintGenerator instances
    """

    def __init__(self, program: Program) -> None:
        self._program = program
        self._return_type_cache: dict[str, TExp] = {}

    def generate(self) -> list[Constraint]:
        """Generate constraints for the program."""
        constraints: list[Constraint] = []
        nodes_loc = Location("nodes")

        # Generate constraints for all named nodes
        for node_id, node in self._program.nodes.items():
            node_loc = nodes_loc.index(node_id)
            gen = NodeConstraintGenerator(
                node_loc,
                self._program,
                self._return_type_cache,
            )
            node_constraints, return_type = gen.generate(node)
            constraints.extend(node_constraints)
            self._return_type_cache[node_id] = return_type

        # Generate constraints for root
        root_loc = Location("root")
        root_node = self._program.get_root_node()
        gen = NodeConstraintGenerator(
            root_loc,
            self._program,
            self._return_type_cache,
        )
        root_constraints, _ = gen.generate(root_node)
        constraints.extend(root_constraints)

        return constraints


def generate_constraints(program: Program) -> list[Constraint]:
    """Generate type constraints for a Program."""
    generator = ConstraintGenerator(program)
    return generator.generate()
