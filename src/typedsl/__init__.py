"""typeDSL - Type-safe AST node system for Python 3.12+."""

from typedsl.ast import (
    Interpreter,
    Program,
)
from typedsl.codecs import (
    TypeCodecs,
    from_builtins,
    to_builtins,
)
from typedsl.formats.json import (
    from_json,
    to_json,
)
from typedsl.nodes import (
    Child,
    Node,
    NodeRef,
    Ref,
)
from typedsl.schema import (
    FieldSchema,
    NodeSchema,
    all_schemas,
    extract_type,
    node_schema,
)

__all__ = [
    # Core types
    "Child",
    # Schema extraction
    "FieldSchema",
    # AST
    "Interpreter",
    "Node",
    "NodeRef",
    "NodeSchema",
    "Program",
    "Ref",
    # Serialization
    "TypeCodecs",
    "all_schemas",
    "extract_type",
    "from_builtins",
    "from_json",
    "node_schema",
    "to_builtins",
    "to_json",
]
