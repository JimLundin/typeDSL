"""
Node Signatures Example
========================

Demonstrates how to use node signatures for:
- Namespacing nodes from different packages/modules
- Versioning node schemas
- Organizing nodes with structured metadata

The signature is stored as metadata in the NodeSchema and composed
into a single tag for serialization.
"""

from typedsl import Node, AST
from typedsl.schema import node_schema, all_schemas
from typedsl.adapters import JSONAdapter


# ============================================================================
# Simple Signature: Single Tag
# ============================================================================

class SimpleNode(Node[int], tag="simple"):
    """A node with a simple single-part signature."""
    value: int


# ============================================================================
# Multi-Part Signatures: Namespace + Name
# ============================================================================

class MathAdd(Node[float], ns="math", name="add"):
    """Addition node from the math namespace."""
    left: float
    right: float


class MathSubtract(Node[float], ns="math", name="subtract"):
    """Subtraction node from the math namespace."""
    left: float
    right: float


class StringConcat(Node[str], ns="string", name="concat"):
    """Concatenation node from the string namespace."""
    left: str
    right: str


# ============================================================================
# Versioned Signatures: Namespace + Name + Version
# ============================================================================

class AddV1(Node[float], ns="calculator", name="add", version="1.0"):
    """Version 1.0 of the add operation - basic addition."""
    left: float
    right: float


class AddV2(Node[float], ns="calculator", name="add", version="2.0"):
    """
    Version 2.0 of the add operation - supports optional rounding.

    This demonstrates schema evolution: V2 adds a new field while
    maintaining a different tag due to the version component.
    """
    left: float
    right: float
    precision: int = 2  # New field in V2


# ============================================================================
# Custom Signature Components
# ============================================================================

class CustomNode(
    Node[str],
    package="myapp",
    module="transforms",
    name="normalize",
    env="prod"
):
    """
    A node with custom signature components.

    You can define any metadata fields you want - the library doesn't
    enforce specific field names. All kwargs are composed into the tag
    in insertion order.
    """
    data: str


# ============================================================================
# Demonstration
# ============================================================================

def main():
    """Demonstrate signature features."""

    print("=" * 70)
    print("NODE SIGNATURES DEMONSTRATION")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. Show composed tags
    # ========================================================================

    print("1. COMPOSED TAGS")
    print("-" * 70)
    print(f"SimpleNode tag:       {SimpleNode._tag}")
    print(f"MathAdd tag:          {MathAdd._tag}")
    print(f"StringConcat tag:     {StringConcat._tag}")
    print(f"AddV1 tag:            {AddV1._tag}")
    print(f"AddV2 tag:            {AddV2._tag}")
    print(f"CustomNode tag:       {CustomNode._tag}")
    print()

    # ========================================================================
    # 2. Show signature metadata
    # ========================================================================

    print("2. SIGNATURE METADATA")
    print("-" * 70)

    math_add_schema = node_schema(MathAdd)
    print(f"MathAdd signature components:")
    print(f"  {math_add_schema.signature}")
    print()

    add_v2_schema = node_schema(AddV2)
    print(f"AddV2 signature components:")
    print(f"  {add_v2_schema.signature}")
    print()

    custom_schema = node_schema(CustomNode)
    print(f"CustomNode signature components:")
    print(f"  {custom_schema.signature}")
    print()

    # ========================================================================
    # 3. Serialized schema includes signature metadata
    # ========================================================================

    print("3. SERIALIZED SCHEMA WITH SIGNATURE")
    print("-" * 70)

    adapter = JSONAdapter()
    serialized_schema = adapter.serialize_node_schema(add_v2_schema)

    print("AddV2 serialized schema:")
    print(f"  tag:       {serialized_schema['tag']}")
    print(f"  signature: {serialized_schema['signature']}")
    print(f"  returns:   {serialized_schema['returns']}")
    print(f"  fields:    {[f['name'] for f in serialized_schema['fields']]}")
    print()

    # ========================================================================
    # 4. Node instances serialize with composed tag only
    # ========================================================================

    print("4. NODE INSTANCE SERIALIZATION")
    print("-" * 70)

    # Create an instance
    add_v2_node = AddV2(left=10.5, right=20.3, precision=3)

    # Serialize the instance
    serialized_node = adapter.serialize_node(add_v2_node)

    print("AddV2 instance serialization:")
    print(f"  tag:       {serialized_node['tag']}")
    print(f"  left:      {serialized_node['left']}")
    print(f"  right:     {serialized_node['right']}")
    print(f"  precision: {serialized_node['precision']}")
    print()
    print("Note: The signature metadata is NOT in the instance serialization.")
    print("      Only the composed tag is used for discriminating node types.")
    print()

    # ========================================================================
    # 5. Using signatures to organize nodes
    # ========================================================================

    print("5. ORGANIZING NODES BY SIGNATURE")
    print("-" * 70)

    # Get all schemas
    schemas = all_schemas()

    # Group by namespace (first part of signature, if present)
    by_namespace: dict[str, list[str]] = {}

    for tag, schema in schemas.items():
        if schema.signature:
            # Get namespace from signature if it exists
            ns = schema.signature.get("ns", "default")
        else:
            ns = "default"

        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(tag)

    print("Nodes grouped by namespace:")
    for ns, tags in sorted(by_namespace.items()):
        if ns in ["math", "string", "calculator"]:  # Filter to our example namespaces
            print(f"  {ns}:")
            for tag in sorted(tags):
                print(f"    - {tag}")
    print()

    # ========================================================================
    # 6. Preventing collisions across namespaces
    # ========================================================================

    print("6. NAMESPACE COLLISION PREVENTION")
    print("-" * 70)

    print("Without namespaces, these would collide:")
    print(f"  math.add       = {MathAdd._tag}")
    print(f"  calculator.add (v1.0) = {AddV1._tag}")
    print(f"  calculator.add (v2.0) = {AddV2._tag}")
    print()
    print("With namespaces and versions, each has a unique tag.")
    print("This allows different packages to define 'add' nodes without conflicts.")
    print()

    # ========================================================================
    # 7. Frontend usage example
    # ========================================================================

    print("7. FRONTEND USAGE")
    print("-" * 70)

    print("Frontend can display structured metadata:")
    print()
    print("Node: AddV2")
    print(f"  Namespace: {add_v2_schema.signature.get('ns')}")
    print(f"  Name:      {add_v2_schema.signature.get('name')}")
    print(f"  Version:   {add_v2_schema.signature.get('version')}")
    print()
    print("This allows frontends to:")
    print("  - Filter nodes by namespace")
    print("  - Show version history")
    print("  - Group related operations")
    print("  - Display package ownership")
    print()


if __name__ == "__main__":
    main()
