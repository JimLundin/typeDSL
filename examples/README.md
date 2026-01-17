# typeDSL Examples

Simple, focused examples demonstrating how to build DSLs with typeDSL.

## Examples

### 1. Calculator DSL (`01_calculator.py`)

Basic mathematical expression evaluator demonstrating:
- Node definition
- The `Interpreter` base class
- Reference-based ASTs with shared subexpressions

**~130 lines** | Shows the fundamental patterns for building and evaluating a DSL.

### 2. Adapting Existing ASTs (`02_adapting_existing_ast.py`)

How to convert an external AST (Python's `ast` module) to typeDSL:
- Mapping external nodes to typeDSL nodes
- Building a converter
- Implementing an interpreter
- Round-tripping through JSON

**~160 lines** | Applies to ANY existing AST: tree-sitter, ANTLR, proprietary parsers, etc.

## Running Examples

Each example is standalone and runnable:

```bash
# Requires Python 3.12+ (PEP 695 generic syntax)
python examples/01_calculator.py
python examples/02_adapting_existing_ast.py
```

## Key Patterns

### Pattern 1: Define Nodes

```python
from typedsl import Node, Ref

class BinOp(Node[float], tag="binop"):
    """Binary operation node."""
    op: Literal["+", "-", "*", "/"]
    left: Ref[Node[float]]   # Reference to child node
    right: Ref[Node[float]]
```

### Pattern 2: Implement Interpreter

```python
from typedsl import Interpreter

class Calculator(Interpreter[dict[str, float], float]):
    def eval(self, node: Node[float]) -> float:
        match node:
            case BinOp(op="+", left=l, right=r):
                return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
            # ... more cases
```

### Pattern 3: Build and Evaluate Program

```python
from typedsl import Program, Ref

prog = Program(
    root=Ref(id="result"),
    nodes={
        "x": Const(value=3.0),
        "y": Const(value=4.0),
        "result": BinOp(op="+", left=Ref(id="x"), right=Ref(id="y"))
    }
)

calculator = Calculator(prog)
result = calculator.run({})  # Returns 7.0
```

## Further Reading

- [Main README](../README.md) - Package overview
- [Tests](../tests/) - Additional usage examples
