# Claude Code Guidelines for typeDSL

## Code Style

- **No inline imports**: All imports must be at module level. Never use imports inside functions or methods.
- **Type checking imports**: Use `if TYPE_CHECKING:` block only for imports needed exclusively for type annotations.

## Quality Assurance

Before committing, always run:

```bash
uv run ruff check --fix .  # Lint and auto-fix
uv run ruff format .       # Format code
uv run pytest tests/       # Run tests
```

All three must pass before pushing changes.

## Beta Status - No Backwards Compatibility

This library is in beta. Do not maintain backwards compatibility:

- Remove deprecated code entirely, don't add shims or compatibility layers
- Refactor freely when improving the design
- No need to preserve old APIs or behaviors

## Serialization Architecture

Serialization uses **schema-aware type reconstruction**:

- The schema system extracts Python type hints into `TypeDef` objects at runtime
- During deserialization, `TypeDef` information guides type reconstruction
- Types without native JSON representation (tuples, sets, frozensets) serialize as arrays
- The schema tells us what Python type to reconstruct from JSON arrays
- All deserialization goes through `_deserialize_value(value, typedef)` - no fallback to untyped deserialization

## Project Structure

- `src/typedsl/` - Core source code
  - `nodes.py` - Node base class and Ref type
  - `types.py` - TypeDef hierarchy for runtime type representation
  - `schema.py` - Type extraction and schema generation
  - `adapters.py` - Format adapters (JSONAdapter)
  - `serialization.py` - High-level serialization API
  - `ast.py` - Program and Interpreter classes
- `tests/` - Test suite (pytest)

## Type System

The schema system extracts Python type hints into TypeDef objects at runtime. Key types:

- Primitives: `int`, `float`, `str`, `bool`, `None`, `bytes`, `Decimal`
- Temporal: `datetime.date`, `datetime.time`, `datetime.datetime`, `datetime.timedelta`
- Containers: `list[T]`, `dict[K, V]`, `set[T]`, `frozenset[T]`, `tuple[T, ...]`
- Generic: `Sequence[T]`, `Mapping[K, V]`
- Special: `Node[T]`, `Ref[T]`, unions (`T | U`)
