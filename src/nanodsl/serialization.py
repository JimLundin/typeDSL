"""
Serialization domain for converting between Python objects and serialized formats.

This module handles conversion between Python objects (Node, Ref, TypeDef) and
their serialized representations (dict/JSON).
"""

from __future__ import annotations

from typing import Any
import json

from nanodsl.nodes import Node, Ref
from nanodsl.types import TypeDef
from nanodsl.adapters import JSONAdapter

# =============================================================================
# Serialization
# =============================================================================

# Module-level adapter instance
_adapter = JSONAdapter()


def to_dict(obj: Node[Any] | Ref[Any] | TypeDef) -> dict:
    """Serialize to dict."""
    if isinstance(obj, Ref):
        return {"tag": "ref", "id": obj.id}
    elif isinstance(obj, Node):
        return _adapter.serialize_node(obj)
    elif isinstance(obj, TypeDef):
        return _adapter.serialize_typedef(obj)
    else:
        raise ValueError(f"Cannot serialize {type(obj)}")


def from_dict(data: dict) -> Node | Ref | TypeDef:
    """Deserialize from dict."""
    tag = data["tag"]

    if tag == "ref":
        return Ref[Any](id=data["id"])

    # Try Node registry first, then TypeDef
    if tag in Node._registry:
        return _adapter.deserialize_node(data)
    elif tag in TypeDef._registry:
        return _adapter.deserialize_typedef(data)
    else:
        raise ValueError(f"Unknown tag: {tag}")


def to_json(obj: Node | Ref | TypeDef) -> str:
    return json.dumps(to_dict(obj), indent=2)


def from_json(s: str) -> Node | Ref | TypeDef:
    return from_dict(json.loads(s))
