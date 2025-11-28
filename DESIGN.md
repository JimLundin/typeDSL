# Node Type System Design

**Target Python Version:** 3.12+

## Overview

This document describes the design of a type-safe node system for building abstract syntax trees (ASTs) and domain-specific languages (DSLs). The system provides automatic registration, serialization, and schema generation for node types.

## Core Pattern

Both `Node` and `TypeDef` use the same pattern:

- Inherit from base class
- Optionally specify `tag` in class definition
- Automatically becomes a frozen dataclass
- Automatically registered in a central registry

### Node Base Class

```python
@dataclass_transform(frozen_default=True)
class Node[T]:
    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[Node]]] = {}

    def __init_subclass__(cls, tag: str | None = None, ...):
        dataclass(frozen=True)(cls)
        cls._tag = tag or cls.__name__.lower()
        Node._registry[cls._tag] = cls
```

**Key Features:**
- Generic type parameter `T` represents the node's return/value type
- `_tag` identifies the node type for serialization
- `_registry` enables dynamic lookup and deserialization
- `__init_subclass__` hook automates dataclass conversion and registration
- Frozen by default ensures immutability

## Usage

### Defining Nodes

```python
class Literal(Node[float], tag="literal"):
    value: float

class Add(Node[float], tag="add"):
    left: Child[float]
    right: Child[float]
```

### Type Aliases

```python
type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]
```

**Purpose:**
- `NodeRef[T]`: Explicitly represents a reference to a node
- `Child[T]`: Convenient union type for inline nodes or references

## Type Definitions

The `TypeDef` system mirrors the `Node` pattern, enabling runtime type representation and validation.

### Built-in Type Definitions

```python
class PrimitiveType(TypeDef, tag="primitive"):
    primitive: type

class NodeType(TypeDef, tag="node"):
    returns: TypeDef

class RefType(TypeDef, tag="ref"):
    target: TypeDef

class UnionType(TypeDef, tag="union"):
    options: tuple[TypeDef, ...]
```

**Type Categories:**
- **PrimitiveType**: Wraps Python primitive types (int, float, str, bool, None)
- **NodeType**: Represents a node with a specific return type
- **RefType**: Represents a reference to another type
- **UnionType**: Represents multiple possible types (type unions)

## Serialization

Simple, consistent serialization API. Pattern is `{"tag": cls._tag, **fields}`:

### API

```python
to_dict(obj)   # -> dict
to_json(obj)   # -> str
from_dict(d)   # -> Node | Ref | TypeDef
from_json(s)   # -> Node | Ref | TypeDef
```

### Example

```python
node = Add(left=Literal(1.0), right=Literal(2.0))
data = to_dict(node)
# {"tag": "add", "left": {"tag": "literal", "value": 1.0}, "right": {"tag": "literal", "value": 2.0}}

restored = from_dict(data)
# Add(left=Literal(1.0), right=Literal(2.0))
```

## Schema Generation

Enable JSON Schema generation for validation and documentation.

### API

```python
node_schema(cls)  # -> dict for one node class
all_schemas()     # -> dict with all registered nodes
```

**Use Cases:**
- Generate documentation
- Validate serialized data
- Generate TypeScript/other language bindings
- IDE integration and autocomplete

## AST Container

Manages the complete abstract syntax tree with node storage and reference resolution.

```python
@dataclass
class AST:
    root: str
    nodes: dict[str, Node]

    def resolve[X](self, ref: Ref[X]) -> X: ...
```

**Responsibilities:**
- Store all nodes in a flat dictionary
- Provide reference resolution
- Maintain a single root entry point
- Enable serialization of cyclic graphs

## Primitives

```python
PRIMITIVES: frozenset[type] = frozenset({float, int, str, bool, type(None)})
```

**Single source of truth** for primitive types.

**Name Derivation:**
- Standard types: `type.__name__` (e.g., `float`, `int`, `str`, `bool`)
- Special case: `type(None)` → `"none"`

## Design Principles

1. **Immutability**: All nodes are frozen dataclasses
2. **Type Safety**: Leverage Python 3.12+ generics for compile-time type checking
3. **Automatic Registration**: No manual registry management
4. **Uniform Pattern**: Same approach for Node and TypeDef
5. **Simple Serialization**: Straightforward dict/JSON representation
6. **Reference Support**: First-class support for node references and graph structures

## Future Considerations

- Validation hooks for type checking
- Custom serialization hooks
- Node traversal utilities
- Type inference system
- Pretty printing and visualization
