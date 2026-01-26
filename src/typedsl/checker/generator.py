"""Constraint generation from AST traversal.

This module implements Phase 1 of the type checker: traversing
the AST (Program) and generating type constraints.

For each node in the program:
1. Create fresh type variables for its generic parameters
2. Get field types and convert to constraint type expressions
3. Generate constraints linking field values to expected types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typedsl.checker.constraints import (
    BoundConstraint,
    Constraint,
    EqualityConstraint,
    SourceLocation,
    SubtypeConstraint,
)
from typedsl.checker.convert import to_type_expr
from typedsl.checker.errors import TypeCheckError, UnresolvedReferenceError
from typedsl.checker.types import TCon, TVar, TypeExpr
from typedsl.nodes import Node, Ref

if TYPE_CHECKING:
    from typedsl.ast import Program


@dataclass
class ConstraintGenerator:
    """Generates type constraints from a Program.

    Traverses the AST and emits constraints that capture all
    type relationships. The constraints are declarative and
    can be solved in any order.
    """

    program: Program
    constraints: list[Constraint] = field(default_factory=list)
    errors: list[TypeCheckError] = field(default_factory=list)

    # Track processed nodes to handle cycles and references
    _processed: set[str | None] = field(default_factory=set)
    # Map node_id -> (var_map, return_type) for looking up return types
    _node_info: dict[str | None, tuple[dict[str, TVar], TypeExpr]] = field(
        default_factory=dict,
    )
    # Counter for inline nodes (which don't have IDs)
    _inline_counter: int = 0

    def generate(self) -> tuple[list[Constraint], list[TypeCheckError]]:
        """Generate all constraints for the program.

        Returns:
            Tuple of (constraints, structural_errors)

        """
        # Process all named nodes
        for node_id, node in self.program.nodes.items():
            self._process_node(node_id, node)

        # Process the root
        if isinstance(self.program.root, Ref):
            ref_id = self.program.root.id
            if ref_id not in self.program.nodes:
                self.errors.append(
                    UnresolvedReferenceError(
                        location=SourceLocation(None, type(self.program.root), None),
                        message=f"Root reference '{ref_id}' not found",
                        ref_id=ref_id,
                        available=tuple(self.program.nodes.keys()),
                    ),
                )
        else:
            self._process_node(None, self.program.root)

        return self.constraints, self.errors

    def _process_node(
        self,
        node_id: str | None,
        node: Node[Any],
    ) -> TypeExpr:
        """Process a node, generating constraints for its fields.

        Args:
            node_id: ID of the node (None for inline nodes)
            node: The node instance

        Returns:
            The return type expression for this node

        """
        # Check if already processed
        if node_id in self._processed:
            _, return_type = self._node_info[node_id]
            return return_type

        self._processed.add(node_id)
        node_cls = type(node)

        # Create fresh type variables for this node's type parameters
        # Use name-based lookup because get_type_hints() may return different
        # TypeVar objects than __type_params__
        var_map: dict[str, TVar] = {}
        type_params = getattr(node_cls, "__type_params__", ())

        for tv in type_params:
            tvar = TVar.fresh(tv.__name__)
            var_map[tv.__name__] = tvar

            # Add bound constraint if the type parameter is bounded
            bound = getattr(tv, "__bound__", None)
            if bound is not None:
                bound_expr = to_type_expr(bound, {})
                self.constraints.append(
                    BoundConstraint(
                        location=SourceLocation(node_id, node_cls, None),
                        reason=f"Type parameter {tv.__name__} has bound",
                        var=tvar,
                        bound=bound_expr,
                    ),
                )

        # Extract the return type from Node[T]
        return_type = self._extract_return_type(node_cls, var_map)
        self._node_info[node_id] = (var_map, return_type)

        # Process each field
        # Build localns with type parameters so get_type_hints can resolve them
        type_params = getattr(node_cls, "__type_params__", ())
        localns = {tp.__name__: tp for tp in type_params}

        try:
            hints = get_type_hints(node_cls, localns=localns)
        except (NameError, AttributeError, TypeError):
            # get_type_hints can fail for various reasons:
            # - Forward references that can't be resolved
            # - Missing modules
            # - Invalid annotations
            hints = {}

        for field_name, field_type in hints.items():
            # Skip ClassVar fields (tag, signature, registry)
            origin = get_origin(field_type)
            if origin is not None and getattr(origin, "__name__", "") == "ClassVar":
                continue

            field_value = getattr(node, field_name, None)
            expected = to_type_expr(field_type, var_map)

            self._constrain_value(
                node_id=node_id,
                node_cls=node_cls,
                field_name=field_name,
                value=field_value,
                expected=expected,
            )

        return return_type

    def _extract_return_type(
        self,
        node_cls: type,
        var_map: dict[str, TVar],
    ) -> TypeExpr:
        """Extract the return type T from a Node[T] class.

        Looks through the class's bases to find Node[T] and
        extracts the type argument.
        """
        # Check __orig_bases__ for the generic base
        orig_bases = getattr(node_cls, "__orig_bases__", ())

        for base in orig_bases:
            origin = get_origin(base)
            if origin is Node:
                args = get_args(base)
                if args:
                    return to_type_expr(args[0], var_map)

        # Fallback: couldn't determine return type
        # Return a fresh variable
        return TVar.fresh(f"{node_cls.__name__}_return")

    def _constrain_value(  # noqa: C901, PLR0911, PLR0912
        self,
        node_id: str | None,
        node_cls: type,
        field_name: str,
        value: object,
        expected: TypeExpr,
    ) -> None:
        """Generate constraints for a field value against expected type."""
        location = SourceLocation(node_id, node_cls, field_name)

        # None value
        if value is None:
            actual = TCon(type(None), ())
            self._add_compatibility_constraint(location, actual, expected)
            return

        # Inline Node
        if isinstance(value, Node):
            actual_return = self._process_inline_node(value)
            # The return type of the child must match what's expected
            self._add_return_type_constraint(location, actual_return, expected)
            return

        # Reference to another node
        if isinstance(value, Ref):
            ref_id = value.id
            if ref_id not in self.program.nodes:
                self.errors.append(
                    UnresolvedReferenceError(
                        location=location,
                        message=f"Reference '{ref_id}' not found",
                        ref_id=ref_id,
                        available=tuple(self.program.nodes.keys()),
                    ),
                )
                return

            # Ensure referenced node is processed
            ref_node = self.program.nodes[ref_id]
            ref_return = self._process_node(ref_id, ref_node)

            # Constrain the reference's return type
            self._add_return_type_constraint(location, ref_return, expected)
            return

        # List of values
        if isinstance(value, list):
            if isinstance(expected, TCon) and expected.con is list and expected.args:
                elem_expected = expected.args[0]
                for i, item in enumerate(value):
                    self._constrain_value(
                        node_id=node_id,
                        node_cls=node_cls,
                        field_name=f"{field_name}[{i}]",
                        value=item,
                        expected=elem_expected,
                    )
            return

        # Dict value
        if isinstance(value, dict):
            is_dict_type = isinstance(expected, TCon) and expected.con is dict
            if is_dict_type and len(expected.args) >= 2:  # noqa: PLR2004
                key_expected = expected.args[0]
                val_expected = expected.args[1]
                for k, v in value.items():
                    # Constrain key
                    key_actual = to_type_expr(type(k), {})
                    self.constraints.append(
                        EqualityConstraint(
                            location=location,
                            reason=f"Dict key type in {field_name}",
                            left=key_actual,
                            right=key_expected,
                        ),
                    )
                    # Constrain value
                    self._constrain_value(
                        node_id=node_id,
                        node_cls=node_cls,
                        field_name=f"{field_name}[{k!r}]",
                        value=v,
                        expected=val_expected,
                    )
            return

        # Tuple value
        if isinstance(value, tuple):
            if isinstance(expected, TCon) and expected.con is tuple:
                for i, (item, arg_expected) in enumerate(
                    zip(value, expected.args, strict=False),
                ):
                    self._constrain_value(
                        node_id=node_id,
                        node_cls=node_cls,
                        field_name=f"{field_name}[{i}]",
                        value=item,
                        expected=arg_expected,
                    )
            return

        # Set value
        if isinstance(value, set | frozenset):
            if isinstance(expected, TCon) and expected.args:
                elem_expected = expected.args[0]
                for item in value:
                    actual = to_type_expr(type(item), {})
                    self.constraints.append(
                        EqualityConstraint(
                            location=location,
                            reason=f"Set element type in {field_name}",
                            left=actual,
                            right=elem_expected,
                        ),
                    )
            return

        # Primitive value - infer type from value
        actual = to_type_expr(type(value), {})
        self._add_compatibility_constraint(location, actual, expected)

    def _process_inline_node(self, node: Node[Any]) -> TypeExpr:
        """Process an inline node (one without an ID in program.nodes)."""
        # Use a unique ID for tracking inline nodes
        self._inline_counter += 1
        inline_id = f"__inline_{self._inline_counter}"

        # Process it like a named node but with special ID
        return self._process_node(inline_id, node)

    def _add_compatibility_constraint(
        self,
        location: SourceLocation,
        actual: TypeExpr,
        expected: TypeExpr,
    ) -> None:
        """Add constraint for type compatibility.

        If expected is a union, uses subtype constraint.
        Otherwise, uses equality constraint.
        """
        if isinstance(expected, TCon) and expected.con is Union:
            self.constraints.append(
                SubtypeConstraint(
                    location=location,
                    reason="Value must match union type",
                    sub=actual,
                    super_=expected,
                ),
            )
        else:
            self.constraints.append(
                EqualityConstraint(
                    location=location,
                    reason="Field value type must match declaration",
                    left=actual,
                    right=expected,
                ),
            )

    def _add_return_type_constraint(
        self,
        location: SourceLocation,
        actual_return: TypeExpr,
        expected: TypeExpr,
    ) -> None:
        """Add constraint for a child node's return type.

        Handles the case where expected might be:
        - Node[T] (we want T)
        - Child[T] = Node[T] | Ref[Node[T]] (we want T)
        - A direct type T
        """
        # Unwrap Node[T], Ref[Node[T]], or Child[T] to get T
        target = self._unwrap_node_type(expected)

        if isinstance(target, TCon) and target.con is Union:
            self.constraints.append(
                SubtypeConstraint(
                    location=location,
                    reason="Child return type must match expected",
                    sub=actual_return,
                    super_=target,
                ),
            )
        else:
            self.constraints.append(
                EqualityConstraint(
                    location=location,
                    reason="Child return type must match expected",
                    left=actual_return,
                    right=target,
                ),
            )

    def _unwrap_node_type(self, expected: TypeExpr) -> TypeExpr:  # noqa: PLR0911
        """Unwrap Node[T] or Ref[Node[T]] to get T.

        For Child[T] = Node[T] | Ref[Node[T]], we need to extract the T.
        This handles the union case by looking for the common T.
        """
        if not isinstance(expected, TCon):
            return expected

        # Direct Node[T] case
        if expected.con is Node and expected.args:
            return expected.args[0]

        # Ref[X] case - unwrap X
        if expected.con is Ref and expected.args:
            inner = expected.args[0]
            return self._unwrap_node_type(inner)

        # Union case (Child[T] = Node[T] | Ref[Node[T]])
        if expected.con is Union:
            # Try to find Node[T] in the union and extract T
            for option in expected.args:
                if isinstance(option, TCon):
                    if option.con is Node and option.args:
                        return option.args[0]
                    if option.con is Ref and option.args:
                        inner = option.args[0]
                        if isinstance(inner, TCon) and inner.con is Node and inner.args:
                            return inner.args[0]
            # Couldn't unwrap - return as-is
            return expected

        return expected
