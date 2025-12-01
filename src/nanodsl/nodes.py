"""
Node system domain for AST node infrastructure.

This module provides the core AST node infrastructure with automatic registration
and generic type parameters. Nodes are immutable and type-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import dataclass_transform, ClassVar, Any

# =============================================================================
# Core Types
# =============================================================================


@dataclass(frozen=True)
class Ref[X]:
    """Reference to X by ID."""

    id: str


@dataclass_transform(frozen_default=True)
class Node[T]:
    """Base for AST nodes. T is return type."""

    _tag: ClassVar[str]
    _namespace: ClassVar[str]
    _registry: ClassVar[dict[str, type[Node[Any]]]] = {}

    def __init_subclass__(cls, tag: str | None = None, namespace: str | None = None):
        dataclass(frozen=True)(cls)

        # Store namespace and base tag
        cls._namespace = namespace or ""
        base_tag = tag or cls.__name__.lower().removesuffix("node")

        # Create full namespaced tag
        cls._tag = f"{namespace}.{base_tag}" if namespace else base_tag

        # Check for collisions
        if existing := Node._registry.get(cls._tag):
            if existing is not cls:
                raise ValueError(
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    f"Choose a different tag or namespace."
                )

        Node._registry[cls._tag] = cls


type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]
