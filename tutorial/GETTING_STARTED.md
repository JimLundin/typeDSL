# Getting Started with typeDSL

An introduction to building type-safe ASTs with typeDSL.

**Reading time**: 15 minutes

---

## What is typeDSL?

typeDSL helps you build **Abstract Syntax Trees (ASTs)** in Python with full type safety. If you've ever needed to:
- Represent structured data that can be validated and transformed
- Build a domain-specific language (DSL)
- Create an interpreter or compiler
- Work with tree or graph structures

...then typeDSL provides the foundation.

---

## Core Concepts

### 1. Nodes Are Your Building Blocks

A **Node** represents one element in your tree. Each node is typed by what it produces:

```python
from typedsl import Node

class Number(Node[int]):
    """A number produces an int."""
    value: int

class Add(Node[int]):
    """Addition produces an int."""
    left: Node[int]
    right: Node[int]
```

That's it. Three things happened automatically:
- ✓ Classes became frozen dataclasses (immutable)
- ✓ Classes registered with tags `"Number"` and `"Add"`
- ✓ Ready for serialization

Build a tree:
```python
expr = Add(
    left=Number(value=5),
    right=Number(value=3)
)
```

### 2. Type Parameters Enable Reuse

Without generics, you'd need separate classes for each type:
```python
class AddInt(Node[int]): ...
class AddFloat(Node[float]): ...
```

With generics, write once:
```python
class Add[T: int | float](Node[T]):
    left: Node[T]
    right: Node[T]

# Works for both
Add[int](left=int_node, right=int_node)
Add[float](left=float_node, right=float_node)
```

The type parameter `[T]` makes the node polymorphic while staying type-safe.

### 3. Interpreters Evaluate Your AST

An **Interpreter** walks your tree and computes results:

```python
from typedsl import Interpreter

class Calculator(Interpreter[None, int]):
    def eval(self, node: Node[int]) -> int:
        match node:
            case Number(value=v):
                return v
            case Add(left=l, right=r):
                return self.eval(l) + self.eval(r)

# Use it
calc = Calculator(expr)
result = calc.run(None)  # Returns 8
```

The pattern matching syntax makes it clean - each node type gets its own case.

### 4. References Enable Graph Structures

For shared nodes, use `Ref`:

```python
from typedsl import Ref, Program

class Add(Node[int]):
    left: Ref[Node[int]]   # Reference, not inline
    right: Ref[Node[int]]

# Build: (x + 5) * (x + 5) where x is shared
prog = Program(
    root=Ref(id="result"),
    nodes={
        "x": Number(value=3),
        "five": Number(value=5),
        "sum": Add(left=Ref(id="x"), right=Ref(id="five")),
        "result": Multiply(left=Ref(id="sum"), right=Ref(id="sum"))
    }
)
```

Use `Ref` when you need:
- Shared subexpressions
- DAG (directed acyclic graph) structures
- Explicit node IDs

Update your interpreter to resolve references:
```python
class Calculator(Interpreter[None, int]):
    def eval(self, node: Node[int]) -> int:
        match node:
            case Number(value=v):
                return v
            case Add(left=l, right=r):
                # Resolve references first
                return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
```

### 5. Serialization Preserves Everything

Save and load programs as JSON:

```python
# Save
json_str = prog.to_json()

# Load
restored = Program.from_json(json_str)

# Evaluate
result = Calculator(restored).run(None)
```

The structure, types, and references are all preserved.

---

## When to Use What

| Feature | Use When |
|---------|----------|
| `Node[T]` | Always - base building block |
| `Node[T]` fields | Simple trees, no sharing |
| `Ref[Node[T]]` fields | Shared nodes, DAGs, explicit IDs |
| `Program` container | Using references |
| `Interpreter[Ctx, R]` | Evaluating, validating, transforming |
| Type parameters `[T]` | Node works with multiple types |
| Serialization | Saving, loading, transmitting programs |

---

## Example: Complete Calculator

Here's everything together (just 3 node types):

```python
from typedsl import Node, Ref, Program, Interpreter
from typing import Literal

# 1. Define nodes
class Num(Node[int], tag="num"):
    value: int

class BinOp(Node[int], tag="binop"):
    op: Literal["+", "*"]
    left: Ref[Node[int]]
    right: Ref[Node[int]]

# 2. Create interpreter
class Calc(Interpreter[None, int]):
    def eval(self, node: Node[int]) -> int:
        match node:
            case Num(value=v):
                return v
            case BinOp(op="+", left=l, right=r):
                return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
            case BinOp(op="*", left=l, right=r):
                return self.eval(self.resolve(l)) * self.eval(self.resolve(r))

# 3. Build program: (3 + 5) * 2
prog = Program(
    root=Ref(id="result"),
    nodes={
        "three": Num(value=3),
        "five": Num(value=5),
        "sum": BinOp(op="+", left=Ref(id="three"), right=Ref(id="five")),
        "two": Num(value=2),
        "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="two"))
    }
)

# 4. Evaluate
calc = Calc(prog)
result = calc.run(None)  # Returns 16
```

That's the complete flow: define → build → interpret.

---

## Next Steps

- **[Examples](../examples/)**: See `01_calculator.py` and `02_adapting_existing_ast.py`
- **[API Reference](../README.md#api-reference)**: Complete API documentation

Start building your own DSL by:
1. Defining 2-3 node types for your domain
2. Creating a simple interpreter
3. Building small ASTs to test
4. Expanding from there

The best way to learn is to start small and grow organically.
