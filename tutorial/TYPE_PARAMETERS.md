# Understanding Type Parameters

How type parameters make your DSL polymorphic and type-safe.

**Reading time**: 10 minutes

---

## The Problem

You want a comparison node that works with different types:

```python
# Without generics - need separate classes
class CompareInt(Node[bool]):
    left: Node[int]
    right: Node[int]

class CompareStr(Node[bool]):
    left: Node[str]
    right: Node[str]

# This doesn't scale
```

## The Solution

Use a type parameter:

```python
class Compare[T](Node[bool]):
    """Compare two values of the same type T."""
    left: Node[T]
    right: Node[T]

# One class, many types
Compare[int](left=age_node, right=min_age_node)
Compare[str](left=name_node, right=filter_node)
```

The `[T]` says: "this works for any type T, as long as both sides match."

---

## Why This Matters

### 1. Type Safety at Construction

```python
# ✓ Type-safe: both sides are int
Compare[int](
    left=age_node,      # Node[int]
    right=limit_node    # Node[int]
)

# ✗ Compile error: types don't match
Compare[???](  # What type? Can't be both!
    left=age_node,      # int
    right=name_node     # str
)
```

The type system catches mismatches **before** you run the code.

### 2. Self-Documenting

```python
Map[int, str]  # Clearly transforms int → str
Filter[User]   # Clearly filters User objects
```

The type parameters tell you exactly what the node does.

### 3. IDE Support

Your IDE knows the types:
```python
result: Node[int] = BinOp[int](...)  # IDE knows result is Node[int]
```

---

## Type Constraints

Limit which types are allowed:

```python
class Add[T: int | float](Node[T]):
    """Only works for numbers."""
    left: Node[T]
    right: Node[T]

# ✓ Allowed
Add[int](...)
Add[float](...)

# ✗ Type error
Add[str](...)  # str not in (int | float)
```

The constraint `T: int | float` restricts T to numeric types.

---

## Multiple Type Parameters

Some nodes need multiple types:

```python
class Map[Input, Output](Node[list[Output]]):
    """Transform list[Input] → list[Output]."""
    items: Node[list[Input]]
    transform: Node[Output]

# Transform list of ints to list of strings
Map[int, str](
    items=numbers,
    transform=to_string
)
```

Each type parameter has a clear role.

---

## Real-World Example

Here's a type-safe filter (2 nodes):

```python
from typedsl import Node, Ref, Program, Interpreter

class Value[T](Node[T]):
    """A value of type T."""
    data: T

class Filter[T](Node[list[T]]):
    """Keep only items that pass the predicate."""
    items: list[Ref[Node[T]]]
    keep: callable[[T], bool]

# Use it
prog = Program(
    root=Ref(id="result"),
    nodes={
        "one": Value[int](data=1),
        "two": Value[int](data=2),
        "three": Value[int](data=3),
        "result": Filter[int](
            items=[Ref(id="one"), Ref(id="two"), Ref(id="three")],
            keep=lambda x: x > 1
        )
    }
)

# Interpreter
class Eval(Interpreter[None, Any]):
    def eval(self, node: Node[Any]) -> Any:
        match node:
            case Value(data=d):
                return d
            case Filter(items=items, keep=predicate):
                values = [self.eval(self.resolve(ref)) for ref in items]
                return [v for v in values if predicate(v)]

result = Eval(prog).run(None)  # Returns [2, 3]
```

The type parameter ensures you can't accidentally filter `list[int]` with a `str` predicate.

---

## Common Patterns

### Pattern 1: Same Input and Output Type

```python
class Transform[T](Node[T]):
    """Transform T → T."""
    input: Node[T]
    func: callable[[T], T]
```

### Pattern 2: Different Input and Output

```python
class Convert[From, To](Node[To]):
    """Convert From → To."""
    input: Node[From]
    converter: callable[[From], To]
```

### Pattern 3: Container Operations

```python
class Concat[T](Node[list[T]]):
    """Concatenate two lists of T."""
    left: Node[list[T]]
    right: Node[list[T]]
```

---

## Key Takeaways

✓ **Type parameters make nodes reusable** - write once, use with many types
✓ **Type checking happens at construction** - catch errors early
✓ **Constraints prevent misuse** - `T: int | float` restricts allowed types
✓ **Multiple parameters enable complex transformations** - `Map[A, B]`
✓ **Better IDE support** - autocomplete and type inference work correctly

Type parameters are what make typeDSL **type-safe**, not just type-annotated.

---

## Next Steps

See type parameters in action:
- **[Getting Started](GETTING_STARTED.md)**: Basic usage
- **[Examples](../examples/)**: Real code with generics
