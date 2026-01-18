"""typeDSL - Type-safe AST node system for Python 3.12+."""

from typedsl.adapters import (
    # Format adapters
    JSONAdapter,
    JSONEncoder,
)
from typedsl.ast import (
    Interpreter,
    Program,
)
from typedsl.nodes import (
    Child,
    # Core types
    Node,
    NodeRef,
    Ref,
)
from typedsl.schema import (
    FieldSchema,
    # Schema dataclasses
    NodeSchema,
    all_schemas,
    # Schema extraction
    extract_type,
    node_schema,
)
from typedsl.serialization import (
    from_dict,
    from_json,
    # Serialization
    to_dict,
    to_json,
)
from typedsl.types import (
    BoolType,
    BytesType,
    DateTimeType,
    DateType,
    DecimalType,
    DictType,
    DurationType,
    ExternalType,
    ExternalTypeRecord,
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
    # Type definitions
    TypeDef,
    TypeParameter,
    TypeParameterRef,
    UnionType,
)

__all__ = [
    "BoolType",
    "BytesType",
    "Child",
    "DateTimeType",
    "DateType",
    "DecimalType",
    "DictType",
    "DurationType",
    "ExternalType",
    "ExternalTypeRecord",
    "FieldSchema",
    "FloatType",
    "FrozenSetType",
    "IntType",
    "Interpreter",
    "JSONAdapter",
    "JSONEncoder",
    "ListType",
    "LiteralType",
    "MappingType",
    # Core types
    "Node",
    "NodeRef",
    "NodeSchema",
    "NodeType",
    "NoneType",
    "Program",
    "Ref",
    "RefType",
    "ReturnType",
    "SequenceType",
    "SetType",
    "StrType",
    "TimeType",
    "TupleType",
    # Type definitions
    "TypeDef",
    "TypeParameter",
    "TypeParameterRef",
    "UnionType",
    "all_schemas",
    # Schema extraction
    "extract_type",
    "from_dict",
    "from_json",
    "node_schema",
    # Serialization
    "to_dict",
    "to_json",
]
