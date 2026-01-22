"""AST type checking for typeDSL nodes and programs.

This module provides type checking functionality for validating that AST nodes
conform to their declared schemas. It can check individual nodes or entire
programs with reference validation.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    formatter = _TYPE_FORMATTERS.get(type(typedef))
    if formatter is not None:
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
# Child checkers: (tc, value, expected, path) -> None
# =============================================================================


def _check_list_children(
    tc: TypeChecker,
    v: Any,
    e: ListType,
    p: str,
) -> None:
    for i, item in enumerate(v):
        tc._check_value(item, e.element, f"{p}[{i}]")


def _check_set_children(
    tc: TypeChecker,
    v: Any,
    e: SetType | FrozenSetType,
    p: str,
) -> None:
    for i, item in enumerate(v):
        tc._check_value(item, e.element, f"{p}{{{i}}}")


def _check_sequence_children(
    tc: TypeChecker,
    v: Any,
    e: SequenceType,
    p: str,
) -> None:
    for i, item in enumerate(v):
        tc._check_value(item, e.element, f"{p}[{i}]")


def _check_tuple_children(
    tc: TypeChecker,
    v: Any,
    e: TupleType,
    p: str,
) -> None:
    if len(v) != len(e.elements):
        tc._errors.append(
            TypeCheckError(
                p,
                f"tuple[{len(e.elements)}]",
                f"tuple[{len(v)}]",
                f"Expected {len(e.elements)} elements, got {len(v)}",
            ),
        )
        return
    for i, (item, elem_type) in enumerate(zip(v, e.elements, strict=False)):
        tc._check_value(item, elem_type, f"{p}[{i}]")


def _check_dict_children(
    tc: TypeChecker,
    v: Any,
    e: DictType | MappingType,
    p: str,
) -> None:
    for k, val in v.items():
        tc._check_value(k, e.key, f"{p}[{k!r}].key")
        tc._check_value(val, e.value, f"{p}[{k!r}]")


_CHILD_CHECKERS: dict[type[TypeDef], Callable[[TypeChecker, Any, Any, str], None]] = {
    ListType: _check_list_children,
    SetType: _check_set_children,
    FrozenSetType: _check_set_children,
    SequenceType: _check_sequence_children,
    TupleType: _check_tuple_children,
    DictType: _check_dict_children,
    MappingType: _check_dict_children,
}


# =============================================================================
# Complex checkers: (tc, value, expected, path) -> None
# =============================================================================


def _check_literal(tc: TypeChecker, v: Any, e: LiteralType, p: str) -> None:
    if v not in e.values:
        tc._errors.append(
            TypeCheckError(
                p,
                f"Literal{list(e.values)}",
                repr(v),
                f"Value {v!r} not in {list(e.values)}",
            ),
        )


def _check_union(tc: TypeChecker, v: Any, e: UnionType, p: str) -> None:
    for option in e.options:
        test = TypeChecker()
        test._program = tc._program
        test._type_bindings = tc._type_bindings.copy()
        test._check_value(v, option, "test")
        if not test._errors:
            return
    type_names = [type_name(opt) for opt in e.options]
    tc._errors.append(
        TypeCheckError(
            p,
            " | ".join(type_names),
            type(v).__name__,
            "Value matches no option in union",
        ),
    )


def _check_node_type(tc: TypeChecker, v: Any, e: NodeType, p: str) -> None:
    if not isinstance(v, Node):
        tc._errors.append(
            TypeCheckError(
                p,
                e.node_tag,
                type(v).__name__,
                f"Expected node {e.node_tag}",
            ),
        )
        return
    if v.tag != e.node_tag:
        tc._errors.append(
            TypeCheckError(
                p,
                e.node_tag,
                v.tag,
                f"Expected {e.node_tag}, got {v.tag}",
            ),
        )
        return
    tc._check_node_impl(v, p)


def _check_return_type(tc: TypeChecker, v: Any, e: ReturnType, p: str) -> None:
    if not isinstance(v, Node):
        tc._errors.append(
            TypeCheckError(
                p,
                f"Node[{type_name(e.returns)}]",
                type(v).__name__,
                f"Expected node returning {type_name(e.returns)}",
            ),
        )
        return
    schema = node_schema(type(v))
    if not tc._types_compatible(schema.returns, e.returns):
        actual = type_name(schema.returns)
        expected = type_name(e.returns)
        tc._errors.append(
            TypeCheckError(
                p,
                f"Node[{expected}]",
                f"Node[{actual}]",
                f"Node returns {actual}, expected {expected}",
            ),
        )
        return
    tc._check_node_impl(v, p)


def _check_ref_type(tc: TypeChecker, v: Any, e: RefType, p: str) -> None:
    if not isinstance(v, Ref):
        tc._errors.append(
            TypeCheckError(
                p,
                f"Ref[{type_name(e.target)}]",
                type(v).__name__,
                f"Expected Ref, got {type(v).__name__}",
            ),
        )
        return
    if tc._program is not None:
        tc._check_ref(v, p, e.target)


def _check_external_type(tc: TypeChecker, v: Any, e: ExternalType, p: str) -> None:
    value_type = type(v)
    if value_type.__module__ == e.module and value_type.__name__ == e.name:
        return
    tc._errors.append(
        TypeCheckError(
            p,
            f"{e.module}.{e.name}",
            type(v).__name__,
            f"Expected {e.name}, got {type(v).__name__}",
        ),
    )


_COMPLEX_CHECKERS: dict[type[TypeDef], Callable[[TypeChecker, Any, Any, str], None]] = {
    LiteralType: _check_literal,
    UnionType: _check_union,
    NodeType: _check_node_type,
    ReturnType: _check_return_type,
    RefType: _check_ref_type,
    ExternalType: _check_external_type,
}


# =============================================================================
# Type compatibility checkers
# =============================================================================


def _compat_element(tc: TypeChecker, a: TypeDef, e: TypeDef) -> bool:
    return tc._types_compatible(a.element, e.element)  # type: ignore[attr-defined]


def _compat_key_value(tc: TypeChecker, a: TypeDef, e: TypeDef) -> bool:
    return (
        tc._types_compatible(a.key, e.key)  # type: ignore[attr-defined]
        and tc._types_compatible(a.value, e.value)  # type: ignore[attr-defined]
    )


def _compat_tuple(tc: TypeChecker, a: TupleType, e: TupleType) -> bool:
    return len(a.elements) == len(e.elements) and all(
        tc._types_compatible(ai, ei)
        for ai, ei in zip(a.elements, e.elements, strict=False)
    )


_COMPAT_SAME_TYPE: dict[type[TypeDef], Callable[[TypeChecker, Any, Any], bool]] = {
    ListType: _compat_element,
    SetType: _compat_element,
    FrozenSetType: _compat_element,
    SequenceType: _compat_element,
    DictType: _compat_key_value,
    MappingType: _compat_key_value,
    TupleType: _compat_tuple,
    RefType: lambda tc, a, e: tc._types_compatible(a.target, e.target),
    ReturnType: lambda tc, a, e: tc._types_compatible(a.returns, e.returns),
}


# =============================================================================
# TypeChecker class
# =============================================================================


class TypeChecker:
    """Type checker for AST nodes and programs."""

    def __init__(self) -> None:
        """Initialize a new type checker."""
        self._errors: list[TypeCheckError] = []
        self._program: Program | None = None
        self._type_bindings: dict[str, TypeDef] = {}

    def check_node(
        self,
        node: Node[Any],
        *,
        program: Program | None = None,
    ) -> TypeCheckResult:
        """Type check a single node."""
        self._errors = []
        self._program = program
        self._type_bindings = {}
        self._check_node_impl(node, path="root")
        return TypeCheckResult(errors=tuple(self._errors))

    def check_program(self, program: Program) -> TypeCheckResult:
        """Type check an entire program including all referenced nodes."""
        self._errors = []
        self._program = program
        self._type_bindings = {}

        if isinstance(program.root, Ref):
            self._check_ref(program.root, path="root", expected_type=None)
        else:
            self._check_node_impl(program.root, path="root")

        for node_id, node in program.nodes.items():
            self._check_node_impl(node, path=f"nodes[{node_id!r}]")

        return TypeCheckResult(errors=tuple(self._errors))

    def _check_node_impl(self, node: Node[Any], path: str) -> None:
        schema = node_schema(type(node))
        for field in schema.fields:
            self._check_value(
                getattr(node, field.name),
                field.type,
                f"{path}.{field.name}",
            )

    def _check_value(self, value: Any, expected: TypeDef, path: str) -> None:
        exp_type = type(expected)

        # Handle type parameters specially (they transform the check)
        if exp_type is TypeParameterRef:
            if expected.name in self._type_bindings:  # type: ignore[attr-defined]
                self._check_value(value, self._type_bindings[expected.name], path)  # type: ignore[attr-defined]
            return
        if exp_type is TypeParameter:
            if expected.bound is not None:  # type: ignore[attr-defined]
                self._check_value(value, expected.bound, path)  # type: ignore[attr-defined]
            return

        # Try simple validator
        validator = _VALIDATORS.get(exp_type)
        if validator is not None:
            error = validator(value, path)
            if error:
                self._errors.append(error)
                return
            child_checker = _CHILD_CHECKERS.get(exp_type)
            if child_checker:
                child_checker(self, value, expected, path)
            return

        # Try complex checker
        checker = _COMPLEX_CHECKERS.get(exp_type)
        if checker is not None:
            checker(self, value, expected, path)
            return

        # Unknown type
        self._errors.append(
            TypeCheckError(
                path,
                type(expected).__name__,
                type(value).__name__,
                f"Unknown type: {expected}",
            ),
        )

    def _check_ref(
        self,
        ref: Ref[Any],
        path: str,
        expected_type: TypeDef | None,
    ) -> None:
        if self._program is None:
            return
        if ref.id not in self._program.nodes:
            self._errors.append(
                TypeCheckError(
                    path,
                    "valid ref",
                    f"Ref({ref.id!r})",
                    f"Reference {ref.id!r} not found",
                ),
            )
            return
        if expected_type is None:
            return
        target = self._program.nodes[ref.id]
        exp_type = type(expected_type)
        if exp_type is ReturnType:
            schema = node_schema(type(target))
            if not self._types_compatible(schema.returns, expected_type.returns):  # type: ignore[attr-defined]
                actual_ret = type_name(schema.returns)
                expected_ret = type_name(expected_type.returns)  # type: ignore[attr-defined]
                self._errors.append(
                    TypeCheckError(
                        path,
                        f"Ref[Node[{expected_ret}]]",
                        f"Ref[Node[{actual_ret}]]",
                        f"Ref target returns {actual_ret}, expected {expected_ret}",
                    ),
                )
        elif exp_type is NodeType and target.tag != expected_type.node_tag:  # type: ignore[attr-defined]
            self._errors.append(
                TypeCheckError(
                    path,
                    f"Ref[{expected_type.node_tag}]",  # type: ignore[attr-defined]
                    f"Ref[{target.tag}]",
                    f"Ref target is {target.tag}, expected {expected_type.node_tag}",  # type: ignore[attr-defined]
                ),
            )

    def _types_compatible(self, actual: TypeDef, expected: TypeDef) -> bool:
        if actual == expected:
            return True

        exp_type = type(expected)
        act_type = type(actual)

        # Type parameters are compatible with anything
        if exp_type in (TypeParameter, TypeParameterRef):
            return True
        if act_type in (TypeParameter, TypeParameterRef):
            return True

        # Union: actual must match at least one option
        if exp_type is UnionType:
            return any(
                self._types_compatible(actual, opt)
                for opt in expected.options  # type: ignore[attr-defined]
            )

        # Covariance rules
        if exp_type is FloatType and act_type is IntType:
            return True
        if exp_type is SequenceType and act_type is ListType:
            return self._types_compatible(actual.element, expected.element)  # type: ignore[attr-defined]
        if exp_type is MappingType and act_type is DictType:
            return (
                self._types_compatible(actual.key, expected.key)  # type: ignore[attr-defined]
                and self._types_compatible(actual.value, expected.value)  # type: ignore[attr-defined]
            )

        # Different types are incompatible
        if act_type is not exp_type:
            return False

        # Same type - check structural compatibility via dispatch
        compat_checker = _COMPAT_SAME_TYPE.get(exp_type)
        if compat_checker:
            return compat_checker(self, actual, expected)

        return False


def typecheck(node: Node[Any]) -> TypeCheckResult:
    """Type check a single node."""
    return TypeChecker().check_node(node)


def typecheck_program(program: Program) -> TypeCheckResult:
    """Type check an entire program."""
    return TypeChecker().check_program(program)
