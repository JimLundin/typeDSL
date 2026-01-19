"""Format adapters for serialization.

Each format module provides to_<format> and from_<format> functions
that work with the core to_builtins/from_builtins conversion.
"""

from typedsl.formats.json import from_json, to_json

__all__ = ["from_json", "to_json"]
