"""Runtime type checker for typeDSL programs.

This module provides type checking that validates user programs against
Python's type system semantics, catching type errors that would be caught
by static type checkers like mypy or pyright.

The type checker uses Python's native type introspection:
- get_type_hints() for field annotations
- get_origin() / get_args() for generic type decomposition
- __orig_bases__ for extracting return types from Node[T]
- __type_params__ for PEP 695 generic parameters
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

from typedsl.ast import Program
from typedsl.nodes import Node, Ref
from typedsl.types import substitute_type_params


def _resolve_type_alias(annotation: Any) -> Any:
    """Resolve PEP 695 type aliases to their underlying type.

    For example, Child[float] -> Node[float] | Ref[Node[float]]
    """
    # Check if it's a subscripted type alias (e.g., Child[float])
    origin = get_origin(annotation)
    if origin is not None and hasattr(origin, "__value__"):
        # It's a subscripted PEP 695 type alias
        base_value = origin.__value__
        type_params = getattr(origin, "__type_params__", ())
        args = get_args(annotation)

        if type_params and args:
            # Build substitution map
            substitutions = dict(zip(type_params, args, strict=False))
            return substitute_type_params(base_value, substitutions)
        return base_value

    # Check if it's a bare type alias (e.g., Child without subscript)
    if hasattr(annotation, "__value__"):
        return annotation.__value__

    return annotation


@dataclass
class TypeCheckError:
    """A type error found during validation."""

    node_type: str
    field_name: str
    expected: str
    actual: str
    context: str = ""

    def __str__(self) -> str:
        msg = f"{self.node_type}.{self.field_name}: expected {self.expected}, got {self.actual}"
        if self.context:
            msg += f" ({self.context})"
        return msg


@dataclass
class TypeCheckResult:
    """Result of type checking a node or program."""

    errors: list[TypeCheckError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(
        self,
        node_type: str,
        field_name: str,
        expected: str,
        actual: str,
        context: str = "",
    ) -> None:
        self.errors.append(
            TypeCheckError(
                node_type=node_type,
                field_name=field_name,
                expected=expected,
                actual=actual,
                context=context,
            ),
        )


def get_node_return_type(node_cls: type) -> type | None:
    """Extract the return type T from a Node[T] subclass.

    Traverses __orig_bases__ to find the Node[T] base class and extracts T.
    Returns None if no return type can be determined.
    """
    # Check __orig_bases__ for parameterized Node base
    for base in getattr(node_cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is None:
            # Non-generic base - check if it's a Node subclass with its own return type
            if isinstance(base, type) and issubclass(base, Node) and base is not Node:
                # Recursively get return type from parent
                parent_return = get_node_return_type(base)
                if parent_return is not None:
                    return parent_return
            continue

        # Check if this is Node or a Node subclass
        if isinstance(origin, type) and issubclass(origin, Node):
            args = get_args(base)
            if args:
                return args[0]

    # Check parent classes
    for base in node_cls.__bases__:
        if base is Node or base is object:
            continue
        if isinstance(base, type) and issubclass(base, Node):
            result = get_node_return_type(base)
            if result is not None:
                return result

    return None


def _format_type(t: type | Any) -> str:
    """Format a type for error messages."""
    if t is type(None):
        return "None"
    if isinstance(t, type):
        return t.__name__
    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        if args:
            args_str = ", ".join(_format_type(a) for a in args)
            origin_name = getattr(origin, "__name__", str(origin))
            return f"{origin_name}[{args_str}]"
        return getattr(origin, "__name__", str(origin))
    return str(t)


def _is_node_subclass(cls: type) -> bool:
    """Check if cls is a Node subclass (but not Node itself)."""
    return isinstance(cls, type) and issubclass(cls, Node) and cls is not Node


def _is_ref_type(t: type | Any) -> bool:
    """Check if t is Ref or Ref[X]."""
    origin = get_origin(t)
    if origin is not None:
        return origin is Ref
    return t is Ref


def _get_ref_target_type(t: type | Any) -> type | Any | None:
    """Extract X from Ref[X]. Returns None if not a Ref type."""
    origin = get_origin(t)
    if origin is Ref:
        args = get_args(t)
        if args:
            return args[0]
    return None


def _is_type_compatible(
    expected: type | Any,
    actual: type | Any,
    type_params: dict[Any, type] | None = None,
) -> bool:
    """Check if actual type is compatible with expected type.

    Args:
        expected: The expected type annotation
        actual: The actual type to check
        type_params: Mapping of type parameters to their inferred types

    Returns:
        True if actual is compatible with expected

    """
    if type_params is None:
        type_params = {}

    # Handle Any - always compatible
    if expected is Any or actual is Any:
        return True

    # Handle type parameters
    if _is_type_param(expected):
        if expected in type_params:
            mapped = type_params[expected]
            # Prevent infinite recursion if mapped value is still a type param
            if mapped is expected or _is_type_param(mapped):
                return True
            return _is_type_compatible(mapped, actual, type_params)
        # Unbound type param - will be inferred
        return True

    # Handle None
    if expected is type(None):
        return actual is type(None) or actual is None

    # Get origins for generic types
    expected_origin = get_origin(expected)
    actual_origin = get_origin(actual)

    # Handle Union types (including X | Y syntax)
    if expected_origin is Union or isinstance(expected, types.UnionType):
        expected_args = get_args(expected)
        return any(
            _is_type_compatible(arg, actual, type_params) for arg in expected_args
        )

    # Handle Literal types
    if expected_origin is Literal:
        # For Literal, we're checking if a value is in the literal set
        # This is handled separately in _check_literal_value
        return True

    # Handle Node types
    if expected_origin is not None and _is_node_subclass(expected_origin):
        # Expected is Node[T] or SpecificNode[T]
        if actual_origin is not None and _is_node_subclass(actual_origin):
            # Check if actual class is compatible with expected class
            if not issubclass(actual_origin, expected_origin):
                return False
            # Check type arguments
            expected_args = get_args(expected)
            actual_args = get_args(actual)
            if expected_args and actual_args:
                return _is_type_compatible(
                    expected_args[0],
                    actual_args[0],
                    type_params,
                )
            return True
        return False

    if _is_node_subclass(expected):
        # Expected is a specific node class (non-generic like Const)
        if isinstance(actual, type) and _is_node_subclass(actual):
            return issubclass(actual, expected)
        return False

    # Handle generic Node[T]
    if expected_origin is Node or (
        isinstance(expected_origin, type) and issubclass(expected_origin, Node)
    ):
        expected_args = get_args(expected)
        if not expected_args:
            return True  # Unparameterized Node accepts any Node

        expected_return = expected_args[0]

        # actual should be a node class or generic node
        if isinstance(actual, type) and _is_node_subclass(actual):
            actual_return = get_node_return_type(actual)
            if actual_return is None:
                return True  # Can't determine, assume compatible
            return _is_type_compatible(expected_return, actual_return, type_params)

        return False

    # Handle container types (list, dict, set, etc.)
    if expected_origin is not None:
        if actual_origin != expected_origin:
            # Check for subclass relationship (e.g., list is a Sequence)
            if actual_origin is None or not (
                isinstance(expected_origin, type)
                and isinstance(actual_origin, type)
                and issubclass(actual_origin, expected_origin)
            ):
                return False
        expected_args = get_args(expected)
        actual_args = get_args(actual)
        if expected_args and actual_args:
            if len(expected_args) != len(actual_args):
                return False
            return all(
                _is_type_compatible(e, a, type_params)
                for e, a in zip(expected_args, actual_args, strict=False)
            )
        return True

    # Simple type comparison
    if isinstance(expected, type) and isinstance(actual, type):
        return issubclass(actual, expected)

    return expected == actual


def _is_type_param(t: Any) -> bool:
    """Check if t is a TypeVar or type parameter."""
    # PEP 695 TypeVar
    return type(t).__name__ == "TypeVar"


def _infer_concrete_return_type(node: Node[Any]) -> type | None:
    """Infer the concrete return type of a generic node instance.

    For generic nodes like Wrapper[T], this tries to determine the actual
    return type by looking at the values of fields that use T.
    """
    node_cls = type(node)
    base_return = get_node_return_type(node_cls)

    # If it's already a concrete type, return it
    if base_return is None or not _is_type_param(base_return):
        return base_return

    # It's a TypeVar - try to infer from fields
    type_param = base_return

    try:
        hints = get_type_hints(node_cls)
    except Exception:
        return None

    for field_name, annotation in hints.items():
        if field_name.startswith("_") or field_name in ("tag", "signature", "registry"):
            continue

        if not hasattr(node, field_name):
            continue

        value = getattr(node, field_name)

        # Check if this annotation is the type parameter directly (e.g., value: T)
        if annotation is type_param:
            return type(value)

        # Check if this annotation uses our type parameter in a Node[T]
        origin = get_origin(annotation)
        if origin is Node or (isinstance(origin, type) and issubclass(origin, Node)):
            args = get_args(annotation)
            if args and args[0] is type_param:
                # This field determines T
                if isinstance(value, Node):
                    return _infer_concrete_return_type(value) or get_node_return_type(
                        type(value),
                    )

    return None


def _get_type_param_bound(param: Any) -> type | Any | None:
    """Get the bound of a type parameter, if any."""
    # PEP 695 style bound
    if hasattr(param, "__bound__"):
        return param.__bound__
    return None


def _check_bound_satisfied(param: Any, inferred_type: type | Any) -> bool:
    """Check if inferred_type satisfies the bound of param."""
    bound = _get_type_param_bound(param)
    if bound is None:
        return True  # No bound, anything goes

    # Bound can be a union type
    bound_origin = get_origin(bound)
    if bound_origin is Union or isinstance(bound, types.UnionType):
        bound_args = get_args(bound)
        return any(_is_type_compatible(arg, inferred_type) for arg in bound_args)

    return _is_type_compatible(bound, inferred_type)


def _check_literal_value(
    literal_type: type | Any,
    value: Any,
) -> bool:
    """Check if value is one of the allowed literal values."""
    origin = get_origin(literal_type)
    if origin is not Literal:
        return True  # Not a literal type

    allowed_values = get_args(literal_type)
    return value in allowed_values


def _infer_type_params_from_value(
    annotation: type | Any,
    value: Any,
    type_params: dict[Any, type],
    node_types: Mapping[str, type] | None = None,
) -> None:
    """Infer type parameter values from a concrete value.

    Updates type_params dict with inferred mappings.
    """
    # Resolve type aliases
    annotation = _resolve_type_alias(annotation)

    if _is_type_param(annotation):
        # Direct type param - infer from value
        if isinstance(value, Node):
            inferred = get_node_return_type(type(value))
            if inferred is not None:
                if annotation in type_params:
                    # Already inferred - check consistency
                    pass  # Will be caught by validation
                else:
                    type_params[annotation] = inferred
        elif isinstance(value, Ref) and node_types is not None:
            # Infer from ref target
            target_id = value.id
            if target_id in node_types:
                target_return = get_node_return_type(node_types[target_id])
                if target_return is not None:
                    type_params[annotation] = target_return
        return

    origin = get_origin(annotation)

    # Handle Node[T] where T is a type param
    if origin is not None and _is_node_subclass(origin):
        args = get_args(annotation)
        if args and _is_type_param(args[0]) and isinstance(value, Node):
            param = args[0]
            inferred = get_node_return_type(type(value))
            # If inferred is a TypeVar, try to get concrete type
            if inferred is not None and _is_type_param(inferred):
                inferred = _infer_concrete_return_type(value)
            if inferred is not None and param not in type_params:
                type_params[param] = inferred
        return

    # Handle generic Node[T]
    if origin is Node or (isinstance(origin, type) and issubclass(origin, Node)):
        args = get_args(annotation)
        if args and _is_type_param(args[0]):
            param = args[0]
            if isinstance(value, Node):
                inferred = get_node_return_type(type(value))
                # If inferred is a TypeVar, try to get concrete type
                if inferred is not None and _is_type_param(inferred):
                    inferred = _infer_concrete_return_type(value)
                if inferred is not None and param not in type_params:
                    type_params[param] = inferred
            elif isinstance(value, Ref) and node_types is not None:
                target_id = value.id
                if target_id in node_types:
                    target_return = get_node_return_type(node_types[target_id])
                    if target_return is not None and param not in type_params:
                        type_params[param] = target_return
        return

    # Handle Ref[Node[T]]
    if origin is Ref:
        args = get_args(annotation)
        if args:
            ref_target = args[0]
            ref_origin = get_origin(ref_target)
            if ref_origin is Node or (
                isinstance(ref_origin, type) and issubclass(ref_origin, Node)
            ):
                ref_args = get_args(ref_target)
                if ref_args and _is_type_param(ref_args[0]):
                    if isinstance(value, Ref) and node_types is not None:
                        target_id = value.id
                        if target_id in node_types:
                            target_return = get_node_return_type(node_types[target_id])
                            if target_return is not None:
                                type_params[ref_args[0]] = target_return
        return

    # Handle containers
    if origin in (list, set, frozenset):
        args = get_args(annotation)
        if args and isinstance(value, (list, set, frozenset)):
            for item in value:
                _infer_type_params_from_value(args[0], item, type_params, node_types)
        return

    if origin is dict:
        args = get_args(annotation)
        if args and len(args) >= 2 and isinstance(value, dict):
            for v in value.values():
                _infer_type_params_from_value(args[1], v, type_params, node_types)
        return

    # Handle Union
    if origin is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        # Try to find matching variant and infer from it
        for arg in args:
            if arg is type(None) and value is None:
                return
            if isinstance(value, Node):
                arg_origin = get_origin(arg)
                if arg_origin is Node or (
                    isinstance(arg_origin, type) and issubclass(arg_origin, Node)
                ):
                    _infer_type_params_from_value(arg, value, type_params, node_types)
                    return
                if _is_node_subclass(arg) and isinstance(value, arg):
                    return
            if isinstance(value, Ref) and _is_ref_type(arg):
                _infer_type_params_from_value(arg, value, type_params, node_types)
                return


def _check_field_value(
    node_type_name: str,
    field_name: str,
    annotation: type | Any,
    value: Any,
    result: TypeCheckResult,
    type_params: dict[Any, type],
    node_types: Mapping[str, type] | None = None,
    checked_nodes: set[int] | None = None,
) -> None:
    """Check a single field value against its annotation.

    Args:
        node_type_name: Name of the node class (for error messages)
        field_name: Name of the field being checked
        annotation: The type annotation for this field
        value: The actual value
        result: TypeCheckResult to add errors to
        type_params: Dict mapping type params to inferred types
        node_types: For program validation, maps node IDs to their types
        checked_nodes: Set of already-checked node ids (to prevent cycles)

    """
    if checked_nodes is None:
        checked_nodes = set()

    # Resolve type aliases (e.g., Child[T] -> Node[T] | Ref[Node[T]])
    annotation = _resolve_type_alias(annotation)

    # Handle None values
    if value is None:
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            args = get_args(annotation)
            if type(None) in args or None in args:
                return  # None is valid
        if annotation is type(None) or annotation is None:
            return
        result.add_error(
            node_type_name,
            field_name,
            _format_type(annotation),
            "None",
        )
        return

    # Get origin for generic types
    origin = get_origin(annotation)

    # Handle Literal types - check the value itself
    if origin is Literal:
        if not _check_literal_value(annotation, value):
            allowed = get_args(annotation)
            result.add_error(
                node_type_name,
                field_name,
                f"Literal{list(allowed)}",
                repr(value),
            )
        return

    # Handle Union types
    if origin is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        # Check if any variant matches
        for arg in args:
            if arg is type(None) and value is None:
                return
            if _check_value_matches_type(arg, value, type_params, node_types):
                # Recursively check with the matching variant
                _check_field_value(
                    node_type_name,
                    field_name,
                    arg,
                    value,
                    result,
                    type_params,
                    node_types,
                    checked_nodes,
                )
                return
        # No variant matched
        result.add_error(
            node_type_name,
            field_name,
            _format_type(annotation),
            _format_actual_value(value),
        )
        return

    # Handle Node types (both generic Node[T] and specific SpecificNode)
    if isinstance(value, Node):
        if not _check_node_against_annotation(
            node_type_name,
            field_name,
            annotation,
            value,
            result,
            type_params,
            node_types,
        ):
            return

        # Recursively check the nested node
        node_id = id(value)
        if node_id not in checked_nodes:
            checked_nodes.add(node_id)
            _check_node_fields(value, result, type_params, node_types, checked_nodes)
        return

    # Handle Ref types
    if isinstance(value, Ref):
        _check_ref_against_annotation(
            node_type_name,
            field_name,
            annotation,
            value,
            result,
            type_params,
            node_types,
        )
        return

    # Handle containers with nodes
    if origin in (list, set, frozenset) and isinstance(value, (list, set, frozenset)):
        args = get_args(annotation)
        if args:
            element_type = args[0]
            for i, item in enumerate(value):
                _check_field_value(
                    node_type_name,
                    f"{field_name}[{i}]",
                    element_type,
                    item,
                    result,
                    type_params,
                    node_types,
                    checked_nodes,
                )
        return

    if origin is dict and isinstance(value, dict):
        args = get_args(annotation)
        if args and len(args) >= 2:
            value_type = args[1]
            for k, v in value.items():
                _check_field_value(
                    node_type_name,
                    f"{field_name}[{k!r}]",
                    value_type,
                    v,
                    result,
                    type_params,
                    node_types,
                    checked_nodes,
                )
        return

    # For primitive types, we trust the dataclass validation
    # (the dataclass will have already accepted or rejected the value)


def _check_value_matches_type(
    annotation: type | Any,
    value: Any,
    type_params: dict[Any, type],
    node_types: Mapping[str, type] | None = None,
) -> bool:
    """Check if a value matches a type annotation (without reporting errors)."""
    # Resolve type aliases
    annotation = _resolve_type_alias(annotation)

    if value is None:
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            args = get_args(annotation)
            return type(None) in args or None in args
        return annotation is type(None) or annotation is None

    origin = get_origin(annotation)

    # Handle Node[T] or SpecificNode
    if isinstance(value, Node):
        if origin is Node or (isinstance(origin, type) and issubclass(origin, Node)):
            # Generic Node[T] or SpecificNode[T]
            if origin is not Node and not isinstance(value, origin):
                return False
            args = get_args(annotation)
            if args:
                expected_return = args[0]
                actual_return = get_node_return_type(type(value))
                if actual_return is not None:
                    return _is_type_compatible(
                        expected_return, actual_return, type_params
                    )
            return True
        if _is_node_subclass(annotation):
            return isinstance(value, annotation)
        return False

    # Handle Ref
    if isinstance(value, Ref):
        if _is_ref_type(annotation):
            return True
        # Check if annotation is a union containing Ref
        if origin is Union or isinstance(annotation, types.UnionType):
            args = get_args(annotation)
            return any(_is_ref_type(arg) for arg in args)
        return False

    return True


def _format_actual_value(value: Any) -> str:
    """Format an actual value for error messages."""
    if isinstance(value, Node):
        return_type = get_node_return_type(type(value))
        # If return type is a TypeVar, try to infer concrete type
        if return_type is not None and _is_type_param(return_type):
            inferred = _infer_concrete_return_type(value)
            if inferred is not None:
                return_type = inferred
        if return_type:
            return f"{type(value).__name__} (Node[{_format_type(return_type)}])"
        return type(value).__name__
    if isinstance(value, Ref):
        return f"Ref(id={value.id!r})"
    return type(value).__name__


def _check_node_against_annotation(
    node_type_name: str,
    field_name: str,
    annotation: type | Any,
    value: Node[Any],
    result: TypeCheckResult,
    type_params: dict[Any, type],
    node_types: Mapping[str, type] | None = None,
) -> bool:
    """Check a Node value against its expected annotation.

    Returns True if check passed, False if error was added.
    """
    origin = get_origin(annotation)
    actual_cls = type(value)
    actual_return = get_node_return_type(actual_cls)

    # If actual_return is a TypeVar, try to infer concrete type from fields
    if actual_return is not None and _is_type_param(actual_return):
        inferred = _infer_concrete_return_type(value)
        if inferred is not None:
            actual_return = inferred

    # Handle type parameter in annotation
    if _is_type_param(annotation):
        if annotation in type_params:
            expected_return = type_params[annotation]
            if actual_return is not None and not _is_type_compatible(
                expected_return,
                actual_return,
                type_params,
            ):
                result.add_error(
                    node_type_name,
                    field_name,
                    f"Node[{_format_type(expected_return)}]",
                    _format_actual_value(value),
                    f"type parameter inferred as {_format_type(expected_return)}",
                )
                return False
        return True

    # Handle specific node subclass (non-generic)
    if _is_node_subclass(annotation) and origin is None:
        if not isinstance(value, annotation):
            result.add_error(
                node_type_name,
                field_name,
                annotation.__name__,
                actual_cls.__name__,
            )
            return False
        return True

    # Handle generic Node[T] or SpecificNode[T]
    if origin is not None:
        if _is_node_subclass(origin):
            # Specific generic node class like Wrapper[float]
            if not isinstance(value, origin):
                result.add_error(
                    node_type_name,
                    field_name,
                    _format_type(annotation),
                    _format_actual_value(value),
                )
                return False

            # Check type argument
            args = get_args(annotation)
            if args:
                expected_return = args[0]
                if _is_type_param(expected_return):
                    if expected_return in type_params:
                        expected_return = type_params[expected_return]
                    else:
                        return True  # Can't check yet

                if actual_return is not None and not _is_type_compatible(
                    expected_return,
                    actual_return,
                    type_params,
                ):
                    result.add_error(
                        node_type_name,
                        field_name,
                        _format_type(annotation),
                        _format_actual_value(value),
                    )
                    return False
            return True

        if origin is Node:
            # Generic Node[T]
            args = get_args(annotation)
            if args:
                expected_return = args[0]
                if _is_type_param(expected_return):
                    if expected_return in type_params:
                        expected_return = type_params[expected_return]
                    else:
                        return True  # Can't check yet

                if actual_return is not None and not _is_type_compatible(
                    expected_return,
                    actual_return,
                    type_params,
                ):
                    result.add_error(
                        node_type_name,
                        field_name,
                        f"Node[{_format_type(expected_return)}]",
                        _format_actual_value(value),
                    )
                    return False
            return True

    return True


def _check_ref_against_annotation(
    node_type_name: str,
    field_name: str,
    annotation: type | Any,
    value: Ref[Any],
    result: TypeCheckResult,
    type_params: dict[Any, type],
    node_types: Mapping[str, type] | None = None,
) -> None:
    """Check a Ref value against its expected annotation."""
    if node_types is None:
        # Can't validate ref without node type info
        return

    target_id = value.id
    if target_id not in node_types:
        result.add_error(
            node_type_name,
            field_name,
            _format_type(annotation),
            f"Ref(id={target_id!r})",
            f"node '{target_id}' not found in program",
        )
        return

    target_cls = node_types[target_id]
    target_return = get_node_return_type(target_cls)

    # Extract expected type from Ref[X]
    origin = get_origin(annotation)

    if origin is Ref:
        args = get_args(annotation)
        if args:
            ref_target_type = args[0]
            ref_target_origin = get_origin(ref_target_type)

            # Handle Ref[SpecificNode] - must be that specific class
            if _is_node_subclass(ref_target_type) and ref_target_origin is None:
                if not issubclass(target_cls, ref_target_type):
                    result.add_error(
                        node_type_name,
                        field_name,
                        f"Ref[{ref_target_type.__name__}]",
                        f"Ref to {target_cls.__name__}",
                        f"references node '{target_id}'",
                    )
                return

            # Handle Ref[Node[T]] or Ref[SpecificNode[T]]
            if ref_target_origin is not None:
                if _is_node_subclass(ref_target_origin):
                    # Check class match
                    if not issubclass(target_cls, ref_target_origin):
                        result.add_error(
                            node_type_name,
                            field_name,
                            _format_type(annotation),
                            f"Ref to {target_cls.__name__}",
                            f"references node '{target_id}'",
                        )
                        return

                ref_args = get_args(ref_target_type)
                if ref_args:
                    expected_return = ref_args[0]
                    if _is_type_param(expected_return):
                        if expected_return in type_params:
                            expected_return = type_params[expected_return]
                        else:
                            return  # Can't check yet

                    if target_return is not None and not _is_type_compatible(
                        expected_return,
                        target_return,
                        type_params,
                    ):
                        result.add_error(
                            node_type_name,
                            field_name,
                            _format_type(annotation),
                            f"Ref to Node[{_format_type(target_return)}]",
                            f"references node '{target_id}'",
                        )


def _check_node_fields(
    node: Node[Any],
    result: TypeCheckResult,
    type_params: dict[Any, type] | None = None,
    node_types: Mapping[str, type] | None = None,
    checked_nodes: set[int] | None = None,
) -> None:
    """Check all fields of a node."""
    if type_params is None:
        type_params = {}
    if checked_nodes is None:
        checked_nodes = set()

    node_cls = type(node)
    node_type_name = node_cls.__name__

    # Get type hints for this node class
    try:
        hints = get_type_hints(node_cls)
    except Exception:
        return  # Can't get hints, skip checking

    # Get class type parameters
    cls_type_params = getattr(node_cls, "__type_params__", ())

    # Create a local copy of type_params for this node
    local_type_params = dict(type_params)

    # First pass: infer type parameters from field values
    for field_name, annotation in hints.items():
        if field_name.startswith("_") or field_name in ("tag", "signature", "registry"):
            continue

        if not hasattr(node, field_name):
            continue

        value = getattr(node, field_name)
        _infer_type_params_from_value(annotation, value, local_type_params, node_types)

    # Check bounds on inferred type parameters
    for param in cls_type_params:
        if param in local_type_params:
            inferred = local_type_params[param]
            if not _check_bound_satisfied(param, inferred):
                bound = _get_type_param_bound(param)
                result.add_error(
                    node_type_name,
                    f"<type param {param}>",
                    f"type satisfying {_format_type(bound)}",
                    _format_type(inferred),
                    "bound violation",
                )

    # Second pass: validate field values
    for field_name, annotation in hints.items():
        if field_name.startswith("_") or field_name in ("tag", "signature", "registry"):
            continue

        if not hasattr(node, field_name):
            continue

        value = getattr(node, field_name)
        _check_field_value(
            node_type_name,
            field_name,
            annotation,
            value,
            result,
            local_type_params,
            node_types,
            checked_nodes,
        )


def type_check(
    target: Node[Any] | Program, *, strict: bool = False
) -> list[TypeCheckError]:
    """Type check a node or program.

    Args:
        target: A Node tree or Program to validate
        strict: If True, raise TypeError on first error

    Returns:
        List of type errors found (empty if valid)

    Raises:
        TypeError: If strict=True and errors are found

    """
    result = TypeCheckResult()

    if isinstance(target, Program):
        # Build node type map
        node_types: dict[str, type] = {}
        for node_id, node in target.nodes.items():
            node_types[node_id] = type(node)

        # Check root
        root = target.root
        if isinstance(root, Node):
            _check_node_fields(root, result, node_types=node_types)
        elif isinstance(root, Ref) and root.id not in node_types:
            result.add_error(
                "Program",
                "root",
                "valid reference",
                f"Ref(id={root.id!r})",
                f"node '{root.id}' not found",
            )

        # Check all nodes
        checked_nodes: set[int] = set()
        for node_id, node in target.nodes.items():
            node_obj_id = id(node)
            if node_obj_id not in checked_nodes:
                checked_nodes.add(node_obj_id)
                _check_node_fields(
                    node,
                    result,
                    node_types=node_types,
                    checked_nodes=checked_nodes,
                )
    else:
        # Single node
        _check_node_fields(target, result)

    if strict and result.errors:
        raise TypeError(str(result.errors[0]))

    return result.errors
