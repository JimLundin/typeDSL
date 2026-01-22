"""Tests for typedsl.types module."""

import pytest

from typedsl.types import (
    BoolType,
    BytesType,
    DateTimeType,
    DateType,
    DecimalType,
    DictType,
    DurationType,
    ExternalType,
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
    TypeDef,
    TypeParameter,
    TypeParameterRef,
    UnionType,
)

# All simple types (no constructor args) with their expected tags
SIMPLE_TYPES = [
    (IntType, "int"),
    (FloatType, "float"),
    (StrType, "str"),
    (BoolType, "bool"),
    (NoneType, "none"),
    (BytesType, "bytes"),
    (DecimalType, "decimal"),
    (DateType, "date"),
    (TimeType, "time"),
    (DateTimeType, "datetime"),
    (DurationType, "duration"),
]


class TestSimpleTypes:
    """Test simple TypeDef subclasses (no constructor arguments)."""

    @pytest.mark.parametrize(("type_cls", "expected_tag"), SIMPLE_TYPES)
    def test_creation_and_tag(self, type_cls: type[TypeDef], expected_tag: str) -> None:
        """Test that simple types can be created and have correct tags."""
        instance = type_cls()
        assert instance.tag == expected_tag

    @pytest.mark.parametrize(("type_cls", "_"), SIMPLE_TYPES)
    def test_frozen(self, type_cls: type[TypeDef], _: str) -> None:
        """Test that simple types are immutable."""
        instance = type_cls()
        with pytest.raises((AttributeError, TypeError)):
            instance.tag = "other"  # type: ignore[misc]


class TestContainerTypes:
    """Test container TypeDef subclasses."""

    def test_list_type(self) -> None:
        """Test ListType creation and structure."""
        lt = ListType(element=IntType())
        assert lt.tag == "list"
        assert isinstance(lt.element, IntType)

    def test_dict_type(self) -> None:
        """Test DictType creation and structure."""
        dt = DictType(key=StrType(), value=IntType())
        assert dt.tag == "dict"
        assert isinstance(dt.key, StrType)
        assert isinstance(dt.value, IntType)

    def test_set_type(self) -> None:
        """Test SetType creation and structure."""
        st = SetType(element=IntType())
        assert st.tag == "set"
        assert isinstance(st.element, IntType)

    def test_frozenset_type(self) -> None:
        """Test FrozenSetType creation and structure."""
        fst = FrozenSetType(element=IntType())
        assert fst.tag == "frozenset"
        assert isinstance(fst.element, IntType)

    def test_tuple_type(self) -> None:
        """Test TupleType creation and structure."""
        tt = TupleType(elements=(IntType(), StrType()))
        assert tt.tag == "tuple"
        assert len(tt.elements) == 2

    def test_sequence_type(self) -> None:
        """Test SequenceType creation and structure."""
        st = SequenceType(element=IntType())
        assert st.tag == "sequence"
        assert isinstance(st.element, IntType)

    def test_mapping_type(self) -> None:
        """Test MappingType creation and structure."""
        mt = MappingType(key=StrType(), value=IntType())
        assert mt.tag == "mapping"
        assert isinstance(mt.key, StrType)
        assert isinstance(mt.value, IntType)


class TestSpecialTypes:
    """Test special TypeDef subclasses."""

    def test_literal_type(self) -> None:
        """Test LiteralType creation."""
        lt = LiteralType(values=("a", "b", 1))
        assert lt.tag == "literal"
        assert lt.values == ("a", "b", 1)

    def test_union_type(self) -> None:
        """Test UnionType creation."""
        ut = UnionType(options=(IntType(), StrType()))
        assert ut.tag == "union"
        assert len(ut.options) == 2

    def test_return_type(self) -> None:
        """Test ReturnType creation."""
        rt = ReturnType(returns=FloatType())
        assert rt.tag == "return"
        assert isinstance(rt.returns, FloatType)

    def test_node_type(self) -> None:
        """Test NodeType creation."""
        nt = NodeType(node_tag="MyNode", type_args=(FloatType(),))
        assert nt.tag == "node"
        assert nt.node_tag == "MyNode"
        assert len(nt.type_args) == 1

    def test_ref_type(self) -> None:
        """Test RefType creation."""
        rt = RefType(target=IntType())
        assert rt.tag == "ref"
        assert isinstance(rt.target, IntType)

    def test_type_parameter(self) -> None:
        """Test TypeParameter creation."""
        tp = TypeParameter(name="T")
        assert tp.tag == "typeparam"
        assert tp.name == "T"
        assert tp.bound is None

    def test_type_parameter_with_bound(self) -> None:
        """Test TypeParameter with bound."""
        tp = TypeParameter(name="T", bound=IntType())
        assert tp.name == "T"
        assert isinstance(tp.bound, IntType)

    def test_type_parameter_ref(self) -> None:
        """Test TypeParameterRef creation."""
        tpr = TypeParameterRef(name="T")
        assert tpr.tag == "typeparamref"
        assert tpr.name == "T"

    def test_external_type(self) -> None:
        """Test ExternalType creation."""
        et = ExternalType(module="pandas.core.frame", name="DataFrame")
        assert et.tag == "external"
        assert et.module == "pandas.core.frame"
        assert et.name == "DataFrame"


class TestTypeDefRegistry:
    """Test TypeDef registry functionality."""

    def test_all_types_registered(self) -> None:
        """Test that all TypeDef subclasses are registered."""
        expected_tags = {
            "int",
            "float",
            "str",
            "bool",
            "none",
            "bytes",
            "decimal",
            "date",
            "time",
            "datetime",
            "duration",
            "list",
            "dict",
            "set",
            "frozenset",
            "tuple",
            "sequence",
            "mapping",
            "literal",
            "union",
            "return",
            "node",
            "ref",
            "typeparam",
            "typeparamref",
            "external",
        }
        assert expected_tags.issubset(set(TypeDef.registry.keys()))

    def test_registry_maps_correctly(self) -> None:
        """Test that registry maps tags to correct classes."""
        assert TypeDef.registry["int"] == IntType
        assert TypeDef.registry["list"] == ListType
        assert TypeDef.registry["union"] == UnionType
        assert TypeDef.registry["external"] == ExternalType


class TestNestedTypes:
    """Test nested type structures."""

    def test_list_of_list(self) -> None:
        """Test creating list[list[int]]."""
        outer = ListType(element=ListType(element=IntType()))
        assert isinstance(outer.element, ListType)
        assert isinstance(outer.element.element, IntType)

    def test_dict_of_lists(self) -> None:
        """Test creating dict[str, list[int]]."""
        dt = DictType(key=StrType(), value=ListType(element=IntType()))
        assert isinstance(dt.value, ListType)
        assert isinstance(dt.value.element, IntType)

    def test_union_of_containers(self) -> None:
        """Test creating list[int] | dict[str, float]."""
        ut = UnionType(
            options=(
                ListType(element=IntType()),
                DictType(key=StrType(), value=FloatType()),
            ),
        )
        assert isinstance(ut.options[0], ListType)
        assert isinstance(ut.options[1], DictType)
