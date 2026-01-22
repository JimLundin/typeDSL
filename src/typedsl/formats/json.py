"""JSON format adapter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from typedsl.codecs import from_builtins, to_builtins

if TYPE_CHECKING:
    from typedsl.nodes import Node, Ref
    from typedsl.types import TypeDef


def to_json(obj: Node[Any] | Ref[Any] | TypeDef, *, indent: int | None = 2) -> str:
    """Serialize a Node, Ref, or TypeDef to a JSON string.

    Args:
        obj: The object to serialize
        indent: JSON indentation level (default 2, None for compact)

    Returns:
        JSON string representation

    """
    builtins = to_builtins(obj)
    return json.dumps(builtins, indent=indent)


def from_json(s: str) -> Node[Any] | Ref[Any]:
    """Deserialize a JSON string to a Node or Ref.

    Args:
        s: JSON string to deserialize

    Returns:
        Deserialized Node or Ref object

    Raises:
        ValueError: If the JSON doesn't contain a valid tagged object
        KeyError: If required 'tag' field is missing

    """
    data = json.loads(s)
    if not isinstance(data, dict):
        msg = "Expected JSON object with 'tag' field"
        raise ValueError(msg)
    return from_builtins(data)
