# typeDSL

A type-safe node system for building Abstract Syntax Trees (ASTs) and Domain-Specific Languages (DSLs) in Python 3.12+.

## Features

- **Type-safe nodes** with generic type parameters (PEP 695)
- **Automatic registration** and schema generation
- **Immutable by default** using frozen dataclasses
- **Serialization** to JSON with extensible format adapters
- **Reference support** for graph structures with shared nodes

## Installation

```bash
pip install typedsl
```

Requires Python 3.12 or later.

## Quick Start

### Define Your First Node

Nodes are the building blocks of your AST. Each node is parameterized by the type it produces:

```python
from typedsl import Node

class Literal(Node[float]):
    value: float

class Add(Node[float]):
    left: Node[float]
    right: Node[float]

# Create an expression: 1.0 + 2.0
expr = Add(
    left=Literal(value=1.0),
    right=Literal(value=2.0)
)
```

That's it! Your classes are automatically:
- Converted to frozen dataclasses
- Registered in a central registry by tag
- Ready for serialization

### Serialize to JSON

```python
from typedsl import to_json, from_json

# Serialize
json_str = to_json(expr)
print(json_str)
# {
#   "tag": "add",
#   "left": {"tag": "literal", "value": 1.0},
#   "right": {"tag": "literal", "value": 2.0}
# }

# Deserialize
restored = from_json(json_str)
assert restored == expr
```

### Extract Schema Information

```python
from typedsl import node_schema, all_schemas

# Get schema for a specific node
schema = node_schema(Add)
print(schema.tag)      # "add"
print(schema.fields)   # (FieldSchema(name='left', ...), FieldSchema(name='right', ...))

# Get all registered schemas
schemas = all_schemas()
print(list(schemas.keys()))  # ['literal', 'add']
```

## Core Concepts

### Nodes vs Types

**Nodes** represent computation or structure in your AST:
```python
class Add(Node[float]):       # produces float
class Query(Node[DataFrame]): # produces DataFrame
```

**Types** describe the shape of values flowing between nodes. Built-in Python types work automatically:
- Primitives: `int`, `float`, `str`, `bool`, `None`
- Containers: `list[T]`, `dict[K, V]`, `set[T]`, `tuple[A, B, C]`
- Unions: `int | str`
- Literals: `Literal["a", "b", "c"]`

### Generic Nodes

Define reusable nodes with type parameters:

```python
class Filter[T](Node[list[T]]):
    """Filter a list based on a predicate."""
    input: Node[list[T]]
    predicate: Node[bool]

class Add[T: int | float](Node[T]):
    """Add two numbers (int or float)."""
    left: Node[T]
    right: Node[T]
```

### Custom Tags

By default, tags are derived from class names. Override with explicit tags:

```python
class MySpecialNode(Node[int], tag="special"):
    value: int

# Serializes with tag "special" instead of "myspecial"
```

## Working with References

typeDSL supports three composition patterns for node children:

### Pattern 1: Inline Nodes (`Node[T]`)

Direct nesting creates tree structures:

```python
from typedsl import Node

class Literal(Node[float]):
    value: float

class Add(Node[float]):
    left: Node[float]   # Inline node
    right: Node[float]

# Simple tree: 1.0 + 2.0
tree = Add(
    left=Literal(value=1.0),
    right=Literal(value=2.0)
)
```

### Pattern 2: References (`Ref[Node[T]]`)

For graph structures with shared nodes:

```python
from typedsl import Node, Ref, Program

class Literal(Node[float]):
    value: float

class Add(Node[float]):
    left: Ref[Node[float]]   # Reference to a node
    right: Ref[Node[float]]

# Graph where "x" is shared
prog = Program(
    root=Ref(id="result"),
    nodes={
        "x": Literal(value=5.0),
        "y": Literal(value=3.0),
        "sum": Add(left=Ref(id="x"), right=Ref(id="y")),
        "result": Add(left=Ref(id="sum"), right=Ref(id="x")),  # Reuses "x"
    }
)

# Resolve references
x_node = prog.resolve(Ref(id="x"))  # Returns Literal(value=5.0)
```

### Pattern 3: Flexible (`Child[T]`)

Use `Child[T]` to accept both inline nodes and references:

```python
from typedsl import Child, Node, Ref, Program

class Literal(Node[float]):
    value: float

class Add(Node[float]):
    left: Child[float]   # Accepts Node[float] OR Ref[Node[float]]
    right: Child[float]

# Works as a simple tree
tree = Add(left=Literal(value=1.0), right=Literal(value=2.0))

# Also works with references in a Program
prog = Program(
    root=Ref(id="result"),
    nodes={
        "x": Literal(value=5.0),
        "result": Add(left=Ref(id="x"), right=Literal(value=3.0)),  # Mixed!
    }
)
```

Use references when you need:
- Shared subexpressions (multiple nodes reference the same child)
- Cyclic graphs
- Flat storage with explicit IDs

## Type Definitions

typeDSL provides a complete type system for schema representation:

```python
from typedsl import (
    IntType, FloatType, StrType, BoolType, NoneType,
    ListType, DictType, SetType, TupleType,
    NodeType, RefType, UnionType, LiteralType,
    extract_type
)

# Extract TypeDef from Python type hints
extract_type(int)                    # IntType()
extract_type(list[int])              # ListType(element=IntType())
extract_type(dict[str, float])       # DictType(key=StrType(), value=FloatType())
extract_type(int | str)              # UnionType(options=(IntType(), StrType()))
extract_type(tuple[int, str, bool])  # TupleType(elements=(IntType(), StrType(), BoolType()))
```

## Registering External Types

Register third-party types for serialization:

```python
import pandas as pd
from typedsl import TypeDef

# Register pandas DataFrame
TypeDef.register(
    pd.DataFrame,
    encode=lambda df: {"data": df.to_dict()},
    decode=lambda d: pd.DataFrame(d["data"])
)

# Now you can use DataFrame in nodes
class DataSource(Node[pd.DataFrame]):
    path: str
```

## Security Considerations

### Deserializing Untrusted Data

**Warning**: Only deserialize JSON/dict data from trusted sources.

The `from_dict()` and `from_json()` functions use the node registry to instantiate classes. While typeDSL itself doesn't execute arbitrary code during deserialization, you should be aware of:

1. **Registered External Types**: If you register external types with custom `decode` functions, those functions will be executed during deserialization
2. **Node Construction**: Deserialized data is used to construct Node instances - ensure your Node classes don't have side effects in `__post_init__`

```python
# ❌ UNSAFE: Deserializing untrusted JSON from user input
user_input = request.get_json()
node = from_dict(user_input)  # Could instantiate unexpected node types

# ✅ SAFE: Validate tag before deserializing
allowed_tags = {"literal", "add", "multiply"}
if user_input.get("tag") not in allowed_tags:
    raise ValueError(f"Unexpected node tag: {user_input.get('tag')}")
node = from_dict(user_input)
```

### Best Practices

1. **Validate input**: Always validate deserialized data before using it in your application
2. **Limit registered types**: Only register external types that you control
3. **Sanitize decode functions**: Ensure custom `decode` functions don't have side effects
4. **Use allowlists**: When accepting user-provided ASTs, validate against an allowlist of expected tags
5. **Frozen by default**: Nodes are immutable, preventing accidental modification after creation

## Complete Example: Expression Evaluator

```python
from typing import Literal
from typedsl import Child, Interpreter, Node, Program, Ref

# Define nodes - Child[T] allows both inline nodes and references
class Const(Node[float]):
    value: float

class Var(Node[float]):
    name: str

class BinOp(Node[float]):
    op: Literal["+", "-", "*", "/"]
    left: Child[float]
    right: Child[float]

# Implement interpreter
class Evaluator(Interpreter[dict[str, float], float]):
    def eval(self, node: Node[float]) -> float:
        match node:
            case Const(value=v):
                return v
            case Var(name=n):
                return self.ctx[n]
            case BinOp(op=op, left=l, right=r):
                left = self.eval(self.resolve(l))  # resolve() handles Node or Ref
                right = self.eval(self.resolve(r))
                if op == "+":
                    return left + right
                if op == "-":
                    return left - right
                if op == "*":
                    return left * right
                if op == "/":
                    return left / right
                raise ValueError(f"Unknown operator: {op}")
            case _:
                raise NotImplementedError(type(node))

# Example 1: Simple tree - (3 + 4) * 2
expr = BinOp(
    op="*",
    left=BinOp(op="+", left=Const(value=3.0), right=Const(value=4.0)),
    right=Const(value=2.0),
)
print(Evaluator(expr).run({}))  # 14.0

# Example 2: Graph with shared nodes - (x + x) * 2
prog = Program(
    root=Ref(id="result"),
    nodes={
        "x": Var(name="x"),
        "sum": BinOp(op="+", left=Ref(id="x"), right=Ref(id="x")),
        "result": BinOp(op="*", left=Ref(id="sum"), right=Const(value=2.0)),
    },
)
print(Evaluator(prog).run({"x": 5.0}))  # 20.0
```

The `resolve()` method accepts `Child[T]` (union of `Node[T] | Ref[Node[T]]`) and uniformly returns the node - either directly if inline, or by resolving the reference.

## Learning typeDSL

### Getting Started

**[GETTING_STARTED.md](tutorial/GETTING_STARTED.md)** (15 min read)

An introduction to core concepts:
- What nodes are and how they work
- Using type parameters for polymorphism
- Building interpreters with pattern matching
- When to use references vs inline nodes
- Serialization basics

Focused on **understanding** with minimal examples (3 node types).

### Type Parameters

**[TYPE_PARAMETERS.md](tutorial/TYPE_PARAMETERS.md)** (10 min read)

Deep dive into generics:
- Why type parameters matter
- Type safety at construction time
- Type constraints (`T: int | float`)
- Multiple type parameters
- Common patterns

Shows the key differentiator of typeDSL.

### Examples

The `examples/` directory contains focused, educational examples:

| Example | Description | Lines |
|---------|-------------|-------|
| **[01_calculator.py](examples/01_calculator.py)** | Basic calculator DSL - node definition, `Interpreter` class, shared subexpressions | ~130 |
| **[02_adapting_existing_ast.py](examples/02_adapting_existing_ast.py)** | Converting external ASTs (Python's `ast` module) to typeDSL | ~160 |

See [examples/README.md](examples/README.md) for usage instructions and key patterns.

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Node[T]` | Base class for AST nodes producing type T |
| `Ref[X]` | Reference to X by ID |
| `Program` | Container for AST with reference resolution |
| `Interpreter[Ctx, R]` | Base class for interpreters with context `Ctx` and return type `R` |
| `TypeDef` | Base class for type definitions |

### Interpreter Methods

| Method | Description |
|--------|-------------|
| `run(ctx: Ctx) -> R` | Execute the program with given context |
| `eval(node: Node[Any]) -> R` | Abstract: evaluate a node (implement with pattern matching) |
| `resolve(child: Child[X]) -> Node[X]` | Resolve child to node (handles both inline and refs) |

### Type Aliases

| Alias | Definition |
|-------|------------|
| `NodeRef[T]` | `Ref[Node[T]]` |
| `Child[T]` | `Node[T] \| Ref[Node[T]]` |

### Serialization Functions

| Function | Description |
|----------|-------------|
| `to_dict(obj)` | Serialize Node/Ref/TypeDef to dict |
| `from_dict(data)` | Deserialize dict to Node/Ref/TypeDef |
| `to_json(obj)` | Serialize to JSON string |
| `from_json(s)` | Deserialize from JSON string |

### Schema Functions

| Function | Description |
|----------|-------------|
| `extract_type(py_type)` | Convert Python type hint to TypeDef |
| `node_schema(cls)` | Extract NodeSchema from Node class |
| `all_schemas()` | Get all registered node schemas |

## Design Principles

1. **Immutability**: All nodes and schemas are frozen dataclasses
2. **Type Safety**: Leverage Python 3.12+ generics with proper type parameters
3. **Minimal Registration**: Types don't need registration unless serialized as embedded values
4. **Dataclass-First**: Schemas are dataclasses; serialization is secondary
5. **Modern Python**: PEP 695 type parameter syntax only

## License

Apache-2.0
