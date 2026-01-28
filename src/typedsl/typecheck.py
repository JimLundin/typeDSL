"""AST type checking for typeDSL nodes and programs.

This module provides type checking functionality for validating that AST nodes
conform to their declared schemas. It can check individual nodes or entire
programs with reference validation.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from typedsl.nodes import Node, Ref
from typedsl.schema import node_schema
from typedsl.types import (
    BoolType,
    BytesType,
    DateTimeType,
    DateType,
    DecimalType,
    DictType,
    DurationType,
    ExternalType,
    FloatType,
    FrozenSetType,
    IntType,
    ListType,
    LiteralType,
    MappingType,
    NodeType,
    NoneType,
    RefType,
    ReturnType,
    SequenceType,
    SetType,
    StrType,
    TimeType,
    TupleType,
    TypeDef,
    TypeParameter,
    TypeParameterRef,
    UnionType,
)

if TYPE_CHECKING:
    from typedsl.ast import Program


@dataclass(frozen=True)
class TypeCheckError:
    """A type error found during type checking."""

    path: str
    expected: str
    actual: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass(frozen=True)
class TypeCheckResult:
    """Result of type checking operation."""

    errors: tuple[TypeCheckError, ...]

    @property
    def is_valid(self) -> bool:
        """Return True if no type errors were found."""
        return len(self.errors) == 0

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        if self.is_valid:
            return "TypeCheckResult: valid"
        error_lines = "\n  ".join(str(e) for e in self.errors)
        return f"TypeCheckResult: {len(self.errors)} error(s)\n  {error_lines}"


# =============================================================================
# Check context: immutable state passed through checking functions
# =============================================================================


@dataclass
class CheckContext:
    """Context for type checking operations."""

    program: Program | None = None
    type_bindings: dict[str, TypeDef] = field(default_factory=dict)


# =============================================================================
# Type name formatting: TypeDef -> str
# =============================================================================

_TYPE_FORMATTERS: dict[type[TypeDef], Callable[[Any], str]] = {
    ListType: lambda t: f"list[{type_name(t.element)}]",
    SetType: lambda t: f"set[{type_name(t.element)}]",
    FrozenSetType: lambda t: f"frozenset[{type_name(t.element)}]",
    SequenceType: lambda t: f"Sequence[{type_name(t.element)}]",
    TupleType: lambda t: f"tuple[{', '.join(type_name(e) for e in t.elements)}]",
    DictType: lambda t: f"dict[{type_name(t.key)}, {type_name(t.value)}]",
    MappingType: lambda t: f"Mapping[{type_name(t.key)}, {type_name(t.value)}]",
    LiteralType: lambda t: f"Literal{list(t.values)}",
    UnionType: lambda t: " | ".join(type_name(o) for o in t.options),
    NodeType: lambda t: (
        f"{t.node_tag}[{', '.join(type_name(a) for a in t.type_args)}]"
        if t.type_args
        else t.node_tag
    ),
    ReturnType: lambda t: f"Node[{type_name(t.returns)}]",
    RefType: lambda t: f"Ref[{type_name(t.target)}]",
    TypeParameter: lambda t: t.name,
    TypeParameterRef: lambda t: t.name,
    ExternalType: lambda t: f"{t.module}.{t.name}",
}


def type_name(typedef: TypeDef) -> str:
    """Get a human-readable name for a TypeDef."""
    if formatter := _TYPE_FORMATTERS.get(type(typedef)):
        return formatter(typedef)
    return typedef.tag


# =============================================================================
# Validators: (value, path) -> TypeCheckError | None
# =============================================================================

_VALIDATORS: dict[type[TypeDef], Callable[[Any, str], TypeCheckError | None]] = {
    NoneType: lambda v, p: (
        TypeCheckError(p, "none", repr(v), "Expected None") if v is not None else None
    ),
    BoolType: lambda v, p: (
        None
        if isinstance(v, bool)
        else TypeCheckError(p, "bool", type(v).__name__, "Expected bool")
    ),
    IntType: lambda v, p: (
        TypeCheckError(p, "int", type(v).__name__, "Expected int")
        if isinstance(v, bool) or not isinstance(v, int)
        else None
    ),
    FloatType: lambda v, p: (
        TypeCheckError(p, "float", "bool", "Expected float")
        if isinstance(v, bool)
        else (
            TypeCheckError(p, "float", type(v).__name__, "Expected float")
            if not isinstance(v, (int, float))
            else None
        )
    ),
    StrType: lambda v, p: (
        None
        if isinstance(v, str)
        else TypeCheckError(p, "str", type(v).__name__, "Expected str")
    ),
    BytesType: lambda v, p: (
        None
        if isinstance(v, bytes)
        else TypeCheckError(p, "bytes", type(v).__name__, "Expected bytes")
    ),
    DecimalType: lambda v, p: (
        None
        if isinstance(v, Decimal)
        else TypeCheckError(p, "decimal", type(v).__name__, "Expected Decimal")
    ),
    DateType: lambda v, p: (
        TypeCheckError(p, "date", "datetime", "Expected date, got datetime")
        if isinstance(v, datetime.datetime)
        else (
            TypeCheckError(p, "date", type(v).__name__, "Expected date")
            if not isinstance(v, datetime.date)
            else None
        )
    ),
    TimeType: lambda v, p: (
        None
        if isinstance(v, datetime.time)
        else TypeCheckError(p, "time", type(v).__name__, "Expected time")
    ),
    DateTimeType: lambda v, p: (
        None
        if isinstance(v, datetime.datetime)
        else TypeCheckError(p, "datetime", type(v).__name__, "Expected datetime")
    ),
    DurationType: lambda v, p: (
        None
        if isinstance(v, datetime.timedelta)
        else TypeCheckError(p, "duration", type(v).__name__, "Expected timedelta")
    ),
    ListType: lambda v, p: (
        None
        if isinstance(v, list)
        else TypeCheckError(p, "list", type(v).__name__, "Expected list")
    ),
    SetType: lambda v, p: (
        None
        if isinstance(v, set)
        else TypeCheckError(p, "set", type(v).__name__, "Expected set")
    ),
    FrozenSetType: lambda v, p: (
        None
        if isinstance(v, frozenset)
        else TypeCheckError(p, "frozenset", type(v).__name__, "Expected frozenset")
    ),
    SequenceType: lambda v, p: (
        None
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes))
        else TypeCheckError(p, "sequence", type(v).__name__, "Expected sequence")
    ),
    TupleType: lambda v, p: (
        None
        if isinstance(v, tuple)
        else TypeCheckError(p, "tuple", type(v).__name__, "Expected tuple")
    ),
    DictType: lambda v, p: (
        None
        if isinstance(v, dict)
        else TypeCheckError(p, "dict", type(v).__name__, "Expected dict")
    ),
    MappingType: lambda v, p: (
        None
        if isinstance(v, Mapping)
        else TypeCheckError(p, "mapping", type(v).__name__, "Expected mapping")
    ),
}


# =============================================================================
# Child checkers: (ctx, value, expected, path) -> list[TypeCheckError]
# =============================================================================


def _check_list_children(
    ctx: CheckContext,
    v: Any,
    e: ListType,
    p: str,
) -> list[TypeCheckError]:
    errors: list[TypeCheckError] = []
    for i, item in enumerate(v):
        errors.extend(_check_value(ctx, item, e.element, f"{p}[{i}]"))
    return errors


def _check_set_children(
    ctx: CheckContext,
    v: Any,
    e: SetType | FrozenSetType,
    p: str,
) -> list[TypeCheckError]:
    errors: list[TypeCheckError] = []
    for i, item in enumerate(v):
        errors.extend(_check_value(ctx, item, e.element, f"{p}{{{i}}}"))
    return errors


def _check_sequence_children(
    ctx: CheckContext,
    v: Any,
    e: SequenceType,
    p: str,
) -> list[TypeCheckError]:
    errors: list[TypeCheckError] = []
    for i, item in enumerate(v):
        errors.extend(_check_value(ctx, item, e.element, f"{p}[{i}]"))
    return errors


def _check_tuple_children(
    ctx: CheckContext,
    v: Any,
    e: TupleType,
    p: str,
) -> list[TypeCheckError]:
    if len(v) != len(e.elements):
        return [
            TypeCheckError(
                p,
                f"tuple[{len(e.elements)}]",
                f"tuple[{len(v)}]",
                f"Expected {len(e.elements)} elements, got {len(v)}",
            ),
        ]
    errors: list[TypeCheckError] = []
    for i, (item, elem_type) in enumerate(zip(v, e.elements, strict=False)):
        errors.extend(_check_value(ctx, item, elem_type, f"{p}[{i}]"))
    return errors


def _check_dict_children(
    ctx: CheckContext,
    v: Any,
    e: DictType | MappingType,
    p: str,
) -> list[TypeCheckError]:
    errors: list[TypeCheckError] = []
    for k, val in v.items():
        errors.extend(_check_value(ctx, k, e.key, f"{p}[{k!r}].key"))
        errors.extend(_check_value(ctx, val, e.value, f"{p}[{k!r}]"))
    return errors


_CHILD_CHECKERS: dict[
    type[TypeDef],
    Callable[[CheckContext, Any, Any, str], list[TypeCheckError]],
] = {
    ListType: _check_list_children,
    SetType: _check_set_children,
    FrozenSetType: _check_set_children,
    SequenceType: _check_sequence_children,
    TupleType: _check_tuple_children,
    DictType: _check_dict_children,
    MappingType: _check_dict_children,
}


# =============================================================================
# Complex checkers: (ctx, value, expected, path) -> list[TypeCheckError]
# =============================================================================


def _check_literal(
    ctx: CheckContext,
    v: Any,
    e: LiteralType,
    p: str,
) -> list[TypeCheckError]:
    del ctx  # unused
    if v not in e.values:
        return [
            TypeCheckError(
                p,
                f"Literal{list(e.values)}",
                repr(v),
                f"Value {v!r} not in {list(e.values)}",
            ),
        ]
    return []


def _check_union(
    ctx: CheckContext,
    v: Any,
    e: UnionType,
    p: str,
) -> list[TypeCheckError]:
    for option in e.options:
        test_ctx = CheckContext(
            program=ctx.program,
            type_bindings=ctx.type_bindings.copy(),
        )
        if not _check_value(test_ctx, v, option, "test"):
            return []
    type_names = [type_name(opt) for opt in e.options]
    return [
        TypeCheckError(
            p,
            " | ".join(type_names),
            type(v).__name__,
            "Value matches no option in union",
        ),
    ]


def _check_node_type(
    ctx: CheckContext,
    v: Any,
    e: NodeType,
    p: str,
) -> list[TypeCheckError]:
    if not isinstance(v, Node):
        return [
            TypeCheckError(
                p,
                e.node_tag,
                type(v).__name__,
                f"Expected node {e.node_tag}",
            ),
        ]
    if v.tag != e.node_tag:
        return [
            TypeCheckError(
                p,
                e.node_tag,
                v.tag,
                f"Expected {e.node_tag}, got {v.tag}",
            ),
        ]
    return _check_node_impl(ctx, v, p)


def _check_return_type(
    ctx: CheckContext,
    v: Any,
    e: ReturnType,
    p: str,
) -> list[TypeCheckError]:
    if not isinstance(v, Node):
        return [
            TypeCheckError(
                p,
                f"Node[{type_name(e.returns)}]",
                type(v).__name__,
                f"Expected node returning {type_name(e.returns)}",
            ),
        ]
    schema = node_schema(type(v))
    # Unify to extract type parameter bindings
    bindings: dict[str, TypeDef] = {}
    if not _unify_types(schema.returns, e.returns, bindings):
        actual = type_name(schema.returns)
        expected = type_name(e.returns)
        return [
            TypeCheckError(
                p,
                f"Node[{expected}]",
                f"Node[{actual}]",
                f"Node returns {actual}, expected {expected}",
            ),
        ]
    # Create new context with narrowed type bindings
    narrowed_ctx = CheckContext(
        program=ctx.program,
        type_bindings={**ctx.type_bindings, **bindings},
    )
    return _check_node_impl(narrowed_ctx, v, p)


def _check_ref_type(
    ctx: CheckContext,
    v: Any,
    e: RefType,
    p: str,
) -> list[TypeCheckError]:
    if not isinstance(v, Ref):
        return [
            TypeCheckError(
                p,
                f"Ref[{type_name(e.target)}]",
                type(v).__name__,
                f"Expected Ref, got {type(v).__name__}",
            ),
        ]
    if ctx.program is not None:
        return _check_ref(ctx, v, p, e.target)
    return []


def _check_external_type(
    ctx: CheckContext,
    v: Any,
    e: ExternalType,
    p: str,
) -> list[TypeCheckError]:
    del ctx  # unused
    value_type = type(v)
    if value_type.__module__ == e.module and value_type.__name__ == e.name:
        return []
    return [
        TypeCheckError(
            p,
            f"{e.module}.{e.name}",
            type(v).__name__,
            f"Expected {e.name}, got {type(v).__name__}",
        ),
    ]


_COMPLEX_CHECKERS: dict[
    type[TypeDef],
    Callable[[CheckContext, Any, Any, str], list[TypeCheckError]],
] = {
    LiteralType: _check_literal,
    UnionType: _check_union,
    NodeType: _check_node_type,
    ReturnType: _check_return_type,
    RefType: _check_ref_type,
    ExternalType: _check_external_type,
}


# =============================================================================
# Type compatibility checking
# =============================================================================


def _compat_element(ctx: CheckContext, a: TypeDef, e: TypeDef) -> bool:
    return _types_compatible(ctx, a.element, e.element)  # type: ignore[attr-defined]


def _compat_key_value(ctx: CheckContext, a: TypeDef, e: TypeDef) -> bool:
    return (
        _types_compatible(ctx, a.key, e.key)  # type: ignore[attr-defined]
        and _types_compatible(ctx, a.value, e.value)  # type: ignore[attr-defined]
    )


def _compat_tuple(ctx: CheckContext, a: TupleType, e: TupleType) -> bool:
    return len(a.elements) == len(e.elements) and all(
        _types_compatible(ctx, ai, ei)
        for ai, ei in zip(a.elements, e.elements, strict=False)
    )


_COMPAT_SAME_TYPE: dict[
    type[TypeDef],
    Callable[[CheckContext, Any, Any], bool],
] = {
    ListType: _compat_element,
    SetType: _compat_element,
    FrozenSetType: _compat_element,
    SequenceType: _compat_element,
    DictType: _compat_key_value,
    MappingType: _compat_key_value,
    TupleType: _compat_tuple,
    RefType: lambda ctx, a, e: _types_compatible(ctx, a.target, e.target),
    ReturnType: lambda ctx, a, e: _types_compatible(ctx, a.returns, e.returns),
}


def _unify_types(
    actual: TypeDef,
    expected: TypeDef,
    bindings: dict[str, TypeDef],
) -> bool:
    """Unify actual type with expected type, extracting type parameter bindings.

    Returns True if types are compatible, False otherwise.
    Updates bindings dict with any type parameter -> concrete type mappings.
    """
    act_type = type(actual)
    exp_type = type(expected)

    # Type parameters in expected match anything (no binding needed)
    if exp_type in (TypeParameter, TypeParameterRef):
        return True

    # Type parameters in actual should bind to expected (only if expected is concrete)
    if act_type is TypeParameter:
        name = actual.name  # type: ignore[attr-defined]
        if name in bindings:
            # Already bound - check consistency
            return _unify_types(bindings[name], expected, bindings)
        bindings[name] = expected
        return True
    if act_type is TypeParameterRef:
        name = actual.name  # type: ignore[attr-defined]
        if name in bindings:
            return _unify_types(bindings[name], expected, bindings)
        bindings[name] = expected
        return True

    # Same types - recurse into structure
    if actual == expected:
        return True

    # Union: actual must match at least one option
    if exp_type is UnionType:
        return any(
            _unify_types(actual, opt, bindings)
            for opt in expected.options  # type: ignore[attr-defined]
        )

    # Covariance rules
    if exp_type is FloatType and act_type is IntType:
        return True
    if exp_type is SequenceType and act_type is ListType:
        return _unify_types(
            actual.element,  # type: ignore[attr-defined]
            expected.element,  # type: ignore[attr-defined]
            bindings,
        )
    if exp_type is MappingType and act_type is DictType:
        return _unify_types(
            actual.key,  # type: ignore[attr-defined]
            expected.key,  # type: ignore[attr-defined]
            bindings,
        ) and _unify_types(
            actual.value,  # type: ignore[attr-defined]
            expected.value,  # type: ignore[attr-defined]
            bindings,
        )

    # ListType element unification
    if exp_type is ListType and act_type is ListType:
        return _unify_types(
            actual.element,  # type: ignore[attr-defined]
            expected.element,  # type: ignore[attr-defined]
            bindings,
        )

    # Different types are incompatible
    return act_type is exp_type


def _types_compatible(
    _ctx: CheckContext,
    actual: TypeDef,
    expected: TypeDef,
) -> bool:
    """Check if actual type is compatible with expected type."""
    del _ctx  # unused - kept for API compatibility
    # Use unification but discard bindings for simple compatibility check
    return _unify_types(actual, expected, {})


# =============================================================================
# Core checking functions
# =============================================================================


def _check_node_impl(
    ctx: CheckContext,
    node: Node[Any],
    path: str,
) -> list[TypeCheckError]:
    """Check a node against its schema."""
    errors: list[TypeCheckError] = []
    schema = node_schema(type(node))
    for f in schema.fields:
        field_path = f"{path}.{f.name}"
        errors.extend(_check_value(ctx, getattr(node, f.name), f.type, field_path))
    return errors


def _check_ref(
    ctx: CheckContext,
    ref: Ref[Any],
    path: str,
    expected_type: TypeDef | None,
) -> list[TypeCheckError]:
    """Check that a reference is valid and points to correct type."""
    if ctx.program is None:
        return []
    if ref.id not in ctx.program.nodes:
        return [
            TypeCheckError(
                path,
                "valid ref",
                f"Ref({ref.id!r})",
                f"Reference {ref.id!r} not found",
            ),
        ]
    if expected_type is None:
        return []
    target = ctx.program.nodes[ref.id]
    exp_type = type(expected_type)
    if exp_type is ReturnType:
        schema = node_schema(type(target))
        if not _types_compatible(ctx, schema.returns, expected_type.returns):  # type: ignore[attr-defined]
            actual_ret = type_name(schema.returns)
            expected_ret = type_name(expected_type.returns)  # type: ignore[attr-defined]
            return [
                TypeCheckError(
                    path,
                    f"Ref[Node[{expected_ret}]]",
                    f"Ref[Node[{actual_ret}]]",
                    f"Ref target returns {actual_ret}, expected {expected_ret}",
                ),
            ]
    elif exp_type is NodeType and target.tag != expected_type.node_tag:  # type: ignore[attr-defined]
        return [
            TypeCheckError(
                path,
                f"Ref[{expected_type.node_tag}]",  # type: ignore[attr-defined]
                f"Ref[{target.tag}]",
                f"Ref target is {target.tag}, expected {expected_type.node_tag}",  # type: ignore[attr-defined]
            ),
        ]
    return []


def _check_value(
    ctx: CheckContext,
    value: Any,
    expected: TypeDef,
    path: str,
) -> list[TypeCheckError]:
    """Check a value against an expected type."""
    exp_type = type(expected)

    # Handle type parameters specially (they transform the check)
    if exp_type is TypeParameterRef:
        if expected.name in ctx.type_bindings:  # type: ignore[attr-defined]
            return _check_value(ctx, value, ctx.type_bindings[expected.name], path)  # type: ignore[attr-defined]
        return []
    if exp_type is TypeParameter:
        param_name = expected.name  # type: ignore[attr-defined]
        # Check type bindings first (from narrowing)
        if param_name in ctx.type_bindings:
            return _check_value(ctx, value, ctx.type_bindings[param_name], path)
        # Fall back to bound if present
        if expected.bound is not None:  # type: ignore[attr-defined]
            return _check_value(ctx, value, expected.bound, path)  # type: ignore[attr-defined]
        return []

    # Try simple validator
    if validator := _VALIDATORS.get(exp_type):
        if error := validator(value, path):
            return [error]
        if child_checker := _CHILD_CHECKERS.get(exp_type):
            return child_checker(ctx, value, expected, path)
        return []

    # Try complex checker
    if checker := _COMPLEX_CHECKERS.get(exp_type):
        return checker(ctx, value, expected, path)

    # Unknown type
    return [
        TypeCheckError(
            path,
            type(expected).__name__,
            type(value).__name__,
            f"Unknown type: {expected}",
        ),
    ]


# =============================================================================
# TypeChecker class - thin wrapper around functional implementation
# =============================================================================


class TypeChecker:
    """Type checker for AST nodes and programs."""

    def __init__(self) -> None:
        """Initialize a new type checker."""
        self._ctx = CheckContext()

    def check_node(
        self,
        node: Node[Any],
        *,
        program: Program | None = None,
    ) -> TypeCheckResult:
        """Type check a single node."""
        ctx = CheckContext(program=program)
        errors = _check_node_impl(ctx, node, path="root")
        return TypeCheckResult(errors=tuple(errors))

    def check_program(self, program: Program) -> TypeCheckResult:
        """Type check an entire program including all referenced nodes."""
        ctx = CheckContext(program=program)
        errors: list[TypeCheckError] = []

        if isinstance(program.root, Ref):
            errors.extend(_check_ref(ctx, program.root, "root", expected_type=None))
        else:
            errors.extend(_check_node_impl(ctx, program.root, path="root"))

        for node_id, node in program.nodes.items():
            errors.extend(_check_node_impl(ctx, node, path=f"nodes[{node_id!r}]"))

        return TypeCheckResult(errors=tuple(errors))


def typecheck(node: Node[Any]) -> TypeCheckResult:
    """Type check a single node."""
    return TypeChecker().check_node(node)


def typecheck_program(program: Program) -> TypeCheckResult:
    """Type check an entire program."""
    return TypeChecker().check_program(program)
