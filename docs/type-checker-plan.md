# Type Checker Implementation Plan

## Goal

Implement a runtime type checker for typeDSL programs that validates type consistency the same way Python's static type checkers (mypy, pyright) would validate hand-written code. The checker should catch type errors that would be caught by static analysis, but operate at program construction time.

## Design Principle: Use Python Types Directly

Rather than using the TypeDef hierarchy from `types.py` (which is for schema export/codegen), the type checker should use Python's native type introspection:

- `get_type_hints()` - Get field annotations
- `get_origin()` / `get_args()` - Decompose generic types
- `__orig_bases__` - Extract base class type arguments
- `typing.get_type_hints()` for runtime type annotation access

This approach keeps the implementation simple and ensures compliance with Python's type semantics.

## Core Type Checking Scenarios

### 1. Node Return Type Extraction

Every `Node[T]` subclass has a return type `T`. This is the type the node "evaluates to" when interpreted.

```python
class Const(Node[float]):   # Returns float
    value: float

class StrNode(Node[str]):   # Returns str
    text: str
```

**Implementation**: Use `__orig_bases__` to find the `Node[T]` base and extract `T`:

```python
def get_node_return_type(cls: type[Node]) -> type:
    for base in cls.__orig_bases__:
        origin = get_origin(base)
        if origin is not None and issubclass(origin, Node):
            args = get_args(base)
            if args:
                return args[0]
    return type(None)
```

### 2. Direct Field Type Validation

When a field expects `Node[T]`, the value must be a node whose return type is compatible with `T`.

```python
class Consumer(Node[float]):
    input: Node[float]  # Expects any node returning float

# Valid
Consumer(input=Const(value=1.0))  # Const returns float

# Invalid
Consumer(input=StrNode(text="hi"))  # StrNode returns str, not float
```

### 3. Reference Type Validation (in Programs)

When using `Ref[Node[T]]`, the referenced node must return type `T`.

```python
class Consumer(Node[float]):
    input: Ref[Node[float]]

# Valid
Program(
    root=Ref(id="c"),
    nodes={
        "x": Const(value=1.0),
        "c": Consumer(input=Ref(id="x")),
    }
)

# Invalid - "s" returns str, not float
Program(
    root=Ref(id="c"),
    nodes={
        "s": StrNode(text="hi"),
        "c": Consumer(input=Ref(id="s")),  # Type error!
    }
)
```

### 4. Child[T] Type Validation

`Child[T] = Node[T] | Ref[Node[T]]` accepts both inline nodes and references, but both must return `T`.

```python
class BinOp(Node[float]):
    left: Child[float]
    right: Child[float]

# Valid with inline
BinOp(left=Const(value=1.0), right=Const(value=2.0))

# Valid with ref (in Program)
BinOp(left=Ref(id="x"), right=Const(value=2.0))

# Invalid
BinOp(left=StrNode(text="hi"), right=Const(value=2.0))  # left returns str
```

### 5. Union Types

When a field has union type, any variant should be acceptable.

```python
class Flexible(Node[str]):
    data: Node[int] | Node[str]  # Accepts either

# Both valid
Flexible(data=IntNode(value=1))
Flexible(data=StrNode(text="hi"))
```

### 6. Literal Type Validation

Fields with `Literal[...]` type must have values from the specified set.

```python
class BinOp(Node[float]):
    op: Literal["+", "-", "*", "/"]
    left: Child[float]
    right: Child[float]

# Valid
BinOp(op="+", left=..., right=...)

# Invalid
BinOp(op="%", left=..., right=...)  # "%" not in literal set
```

### 7. Generic Node Type Parameters

Generic nodes have type parameters that must be consistently instantiated.

```python
class Pair[T](Node[tuple[T, T]]):
    first: Node[T]
    second: Node[T]

# Valid - T=float is consistent
Pair(first=Const(value=1.0), second=Const(value=2.0))

# Invalid - T cannot be both float and str
Pair(first=Const(value=1.0), second=StrNode(text="hi"))
```

### 8. Bounded Type Parameters

Type parameters can have bounds that constrain valid types.

```python
class Numeric[T: int | float](Node[T]):
    value: T

# Valid
Numeric(value=1)      # T=int
Numeric(value=1.0)    # T=float

# Invalid (if we had runtime checks for bounds)
Numeric(value="hi")   # str not in int | float
```

### 9. Return Type Compatibility

The return type annotation `Node[T]` means "any node whose return type is T or a subtype of T".

```python
class Processor(Node[object]):
    input: Node[object]  # Accepts any node

# All valid because int, str, float are subtypes of object
Processor(input=IntNode(value=1))
Processor(input=StrNode(text="hi"))
Processor(input=Const(value=1.0))
```

### 10. Any Type (Escape Hatch)

`Node[Any]` should accept any node, similar to Python's `Any`.

```python
class Wrapper(Node[Any]):
    child: Node[Any]

# All valid
Wrapper(child=IntNode(value=1))
Wrapper(child=StrNode(text="hi"))
```

## Implementation Architecture

### Core Functions

```python
def get_node_return_type(node_cls: type[Node]) -> type:
    """Extract T from Node[T] subclass."""
    ...

def get_field_expected_type(cls: type[Node], field_name: str) -> type:
    """Get the expected type for a field."""
    ...

def is_type_compatible(expected: type, actual: type) -> bool:
    """Check if actual type is compatible with expected type.

    Handles:
    - Simple type equality
    - Any as universal acceptor
    - Union types (any variant matches)
    - Subtype relationships for object types
    """
    ...

def check_node_type(node: Node, expected_return: type | None = None) -> list[TypeError]:
    """Validate a node and all its children.

    Args:
        node: The node to validate
        expected_return: If provided, check node's return type matches

    Returns:
        List of type errors found
    """
    ...

def check_program(program: Program) -> list[TypeError]:
    """Validate a program including all reference relationships.

    Builds a map of node IDs to their return types, then validates
    that all Ref[Node[T]] references point to nodes returning T.
    """
    ...
```

### Type Error Reporting

Errors should include:
- Location (node type, field name)
- Expected type
- Actual type
- Helpful context (e.g., which reference failed)

```python
@dataclass
class TypeCheckError:
    node_type: str
    field_name: str
    expected: str
    actual: str
    context: str = ""

    def __str__(self) -> str:
        msg = f"{self.node_type}.{self.field_name}: expected {self.expected}, got {self.actual}"
        if self.context:
            msg += f" ({self.context})"
        return msg
```

## Edge Cases and Challenges

### 1. Self-Referential Types

A node that references nodes of its own type:

```python
class TreeNode(Node[int]):
    value: int
    children: list[Node[int]]  # Can contain TreeNode
```

### 2. Mutually Recursive Types

Two node types that reference each other:

```python
class Expr(Node[float]):
    ...

class Stmt(Node[None]):
    expr: Node[float]  # References Expr

class Block(Node[None]):
    stmts: list[Node[None]]  # References Stmt
```

### 3. Forward References

When type annotations use string forward references:

```python
class Container(Node[int]):
    nested: "Node[int]"  # String forward reference
```

Need to use `get_type_hints()` with proper namespace to resolve these.

### 4. Generic Type Inference

When a generic node is instantiated, we need to infer type parameters from the values:

```python
class Pair[T](Node[tuple[T, T]]):
    first: Node[T]
    second: Node[T]

# T=float inferred from first, then validated against second
Pair(first=Const(value=1.0), second=???)
```

### 5. Covariance/Invariance

Python generics are invariant by default. `Node[int]` is not a subtype of `Node[object]`. However, for user ergonomics, we may want to allow covariant usage in some contexts.

Design decision: Follow Python's invariant semantics by default.

### 6. References to Non-Existent Nodes

Already handled by Program.resolve(), but type checker should provide a more comprehensive report.

### 7. Cyclic References

Programs can have cycles (A references B, B references A). The type checker must handle this without infinite loops.

```python
Program(
    root=Ref(id="a"),
    nodes={
        "a": NodeA(ref=Ref(id="b")),
        "b": NodeB(ref=Ref(id="a")),
    }
)
```

### 8. Optional Fields

Fields with `None` as a valid option:

```python
class Optional(Node[int]):
    parent: Node[int] | None
```

### 9. List/Dict of Nodes

Container types holding nodes:

```python
class Multi(Node[float]):
    inputs: list[Node[float]]
    named: dict[str, Node[float]]
```

### 10. Nested Generics

Complex nested type structures:

```python
class Complex(Node[dict[str, list[int]]]):
    mapping: dict[str, Node[list[int]]]
```

## Validation Levels

### Level 1: Structural Validation (Current)
- Fields exist
- Types are dataclass-compatible
- This is what frozen dataclass provides

### Level 2: Node Type Validation (Proposed)
- Node return types match field expectations
- Reference targets match expected types
- Literal values are in allowed set

### Level 3: Program Validation (Proposed)
- All references resolve
- Reference types match
- No orphan nodes (optional)
- Graph structure validation (optional)

## API Design

### Simple API

```python
from typedsl import type_check, TypeCheckError

# Check a single node tree
errors = type_check(my_node)
if errors:
    for error in errors:
        print(error)

# Check a program with references
errors = type_check(my_program)
```

### Strict Mode

```python
from typedsl import type_check

# Raise on first error
type_check(my_program, strict=True)  # Raises TypeCheckError
```

### Integration with Node Construction

Consider adding optional validation at node construction time:

```python
# Global setting
typedsl.enable_runtime_checks()

# Now node construction validates
Consumer(input=StrNode(text="hi"))  # Raises TypeError immediately
```

## Testing Strategy

Tests should cover:

1. **Happy path**: Valid programs pass
2. **Type mismatches**: Direct and via references
3. **Generic consistency**: Type parameter inference
4. **Literal validation**: Value not in set
5. **Union handling**: All variants accepted
6. **Any handling**: Universal acceptance
7. **Optional fields**: None is valid
8. **Container types**: list/dict of nodes
9. **Cyclic references**: No infinite loops
10. **Edge cases**: Empty programs, single nodes, complex nesting

See `tests/test_type_checker.py` for comprehensive test cases.

## Implementation Phases

### Phase 1: Core Type Extraction
- Implement `get_node_return_type()`
- Implement `is_type_compatible()` for simple cases
- Test with basic node hierarchies

### Phase 2: Node Validation
- Implement `check_node_type()`
- Handle direct field validation
- Handle literal validation

### Phase 3: Program Validation
- Implement `check_program()`
- Handle reference type validation
- Handle cyclic reference detection

### Phase 4: Generic Support
- Type parameter inference
- Bounded type parameter validation
- Generic node instantiation checking

### Phase 5: API and Integration
- Clean public API
- Error message formatting
- Optional construction-time validation
- Documentation

## Open Questions

1. **Strictness level**: Should we be as strict as mypy, or more lenient for usability?
2. **Performance**: Should validation be opt-in to avoid overhead?
3. **Error recovery**: Continue after first error to collect all issues?
4. **Subtyping**: Allow covariant node types (Node[int] where Node[object] expected)?
5. **Runtime values**: How to handle type parameters that depend on runtime values?

## Conclusion

The type checker should use Python's type system directly via `get_type_hints()`, `get_origin()`, and `get_args()`. This keeps the implementation simple and ensures behavior matches what users expect from Python's type semantics.

The key insight is that we're performing at runtime what static type checkers do at analysis time - verifying that type annotations are respected when constructing program structures.
