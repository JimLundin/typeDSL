"""Tests for TypeCodecs - unified registry for type serialization."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import pytest

from typedsl.nodes import Node
from typedsl.serialization import from_dict, from_json, to_dict, to_json


# =============================================================================
# Mock external types for testing (simulating polars DataFrame, numpy array, etc.)
# =============================================================================


@dataclass
class MockDataFrame:
    """Mock DataFrame for testing external type registration."""

    columns: list[str]
    data: list[dict[str, Any]]

    def to_dicts(self) -> list[dict[str, Any]]:
        return self.data

    @classmethod
    def from_dicts(cls, data: list[dict[str, Any]]) -> MockDataFrame:
        if not data:
            return cls(columns=[], data=[])
        return cls(columns=list(data[0].keys()), data=data)


@dataclass
class MockNDArray:
    """Mock numpy array for testing external type registration."""

    data: list[Any]
    dtype: str = "float64"

    def tolist(self) -> list[Any]:
        return self.data


@dataclass
class Point:
    """Simple external type for testing."""

    x: float
    y: float


# =============================================================================
# Test: TypeCodecs Registration API
# =============================================================================


class TestTypeCodecsRegistration:
    """Test the TypeCodecs.register() API."""

    def test_register_external_type(self) -> None:
        """Test registering an external type with encode/decode functions."""
        from typedsl.codecs import TypeCodecs

        # Clear any existing registration
        TypeCodecs.clear()

        TypeCodecs.register(
            MockDataFrame,
            encode=lambda df: df.to_dicts(),
            decode=lambda data: MockDataFrame.from_dicts(data),
        )

        # Verify registration
        codec = TypeCodecs.get(MockDataFrame)
        assert codec is not None

        encode, decode = codec
        df = MockDataFrame(columns=["a", "b"], data=[{"a": 1, "b": 2}])
        encoded = encode(df)
        assert encoded == [{"a": 1, "b": 2}]

        decoded = decode(encoded)
        assert isinstance(decoded, MockDataFrame)
        assert decoded.data == [{"a": 1, "b": 2}]

    def test_register_with_lambdas(self) -> None:
        """Test registering with simple lambda functions."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()

        TypeCodecs.register(
            Point,
            encode=lambda p: [p.x, p.y],
            decode=lambda data: Point(x=data[0], y=data[1]),
        )

        codec = TypeCodecs.get(Point)
        assert codec is not None

        encode, decode = codec
        point = Point(x=1.5, y=2.5)
        assert encode(point) == [1.5, 2.5]
        assert decode([3.0, 4.0]) == Point(x=3.0, y=4.0)

    def test_get_unregistered_type_returns_none(self) -> None:
        """Test that getting an unregistered type returns None."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()

        class UnregisteredType:
            pass

        assert TypeCodecs.get(UnregisteredType) is None

    def test_clear_removes_all_registrations(self) -> None:
        """Test that clear() removes all registered codecs."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.register(Point, encode=lambda p: [p.x, p.y], decode=lambda d: Point(d[0], d[1]))
        assert TypeCodecs.get(Point) is not None

        TypeCodecs.clear()
        assert TypeCodecs.get(Point) is None

    def test_register_overwrites_existing(self) -> None:
        """Test that registering the same type overwrites the previous codec."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()

        # First registration
        TypeCodecs.register(
            Point,
            encode=lambda p: [p.x, p.y],
            decode=lambda d: Point(d[0], d[1]),
        )

        # Overwrite with different encoder
        TypeCodecs.register(
            Point,
            encode=lambda p: {"x": p.x, "y": p.y},
            decode=lambda d: Point(d["x"], d["y"]),
        )

        encode, _ = TypeCodecs.get(Point)  # type: ignore[misc]
        assert encode(Point(1, 2)) == {"x": 1, "y": 2}


# =============================================================================
# Test: Builtin Codecs (pre-registered)
# =============================================================================


class TestBuiltinCodecs:
    """Test pre-registered codecs for Python builtin types."""

    def test_bytes_codec_registered(self) -> None:
        """Test that bytes codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(bytes)
        assert codec is not None

        encode, decode = codec
        original = b"hello world"
        encoded = encode(original)
        assert isinstance(encoded, str)  # base64 string
        assert decode(encoded) == original

    def test_datetime_codec_registered(self) -> None:
        """Test that datetime codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(datetime)
        assert codec is not None

        encode, decode = codec
        original = datetime(2024, 1, 15, 10, 30, 0)
        encoded = encode(original)
        assert encoded == "2024-01-15T10:30:00"
        assert decode(encoded) == original

    def test_date_codec_registered(self) -> None:
        """Test that date codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(date)
        assert codec is not None

        encode, decode = codec
        original = date(2024, 6, 15)
        encoded = encode(original)
        assert encoded == "2024-06-15"
        assert decode(encoded) == original

    def test_time_codec_registered(self) -> None:
        """Test that time codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(time)
        assert codec is not None

        encode, decode = codec
        original = time(14, 30, 45)
        encoded = encode(original)
        assert encoded == "14:30:45"
        assert decode(encoded) == original

    def test_timedelta_codec_registered(self) -> None:
        """Test that timedelta codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(timedelta)
        assert codec is not None

        encode, decode = codec
        original = timedelta(hours=2, minutes=30)
        encoded = encode(original)
        assert encoded == 9000.0  # 2.5 hours in seconds
        assert decode(encoded) == original

    def test_decimal_codec_registered(self) -> None:
        """Test that Decimal codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(Decimal)
        assert codec is not None

        encode, decode = codec
        original = Decimal("123.456")
        encoded = encode(original)
        assert encoded == "123.456"
        assert decode(encoded) == original

    def test_set_codec_registered(self) -> None:
        """Test that set codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(set)
        assert codec is not None

        encode, decode = codec
        original = {1, 2, 3}
        encoded = encode(original)
        assert isinstance(encoded, list)
        assert set(encoded) == original
        assert decode(encoded) == original

    def test_frozenset_codec_registered(self) -> None:
        """Test that frozenset codec is pre-registered."""
        from typedsl.codecs import TypeCodecs

        codec = TypeCodecs.get(frozenset)
        assert codec is not None

        encode, decode = codec
        original = frozenset({1, 2, 3})
        encoded = encode(original)
        assert isinstance(encoded, list)
        assert frozenset(encoded) == original
        assert decode(encoded) == original


# =============================================================================
# Test: External Type Serialization in Nodes
# =============================================================================


class TestExternalTypeInNodes:
    """Test external types as fields in Node classes."""

    def test_serialize_node_with_external_type(self) -> None:
        """Test serializing a node containing an external type field."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            MockDataFrame,
            encode=lambda df: df.to_dicts(),
            decode=lambda data: MockDataFrame.from_dicts(data),
        )

        class DataNode(Node[None], tag="data_node_ext"):
            data: MockDataFrame

        df = MockDataFrame(columns=["x", "y"], data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        node = DataNode(data=df)

        result = to_dict(node)

        assert result == {
            "tag": "data_node_ext",
            "data": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        }

    def test_deserialize_node_with_external_type(self) -> None:
        """Test deserializing a node containing an external type field."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            MockDataFrame,
            encode=lambda df: df.to_dicts(),
            decode=lambda data: MockDataFrame.from_dicts(data),
        )

        class DataNodeDeser(Node[None], tag="data_node_deser"):
            data: MockDataFrame

        data = {
            "tag": "data_node_deser",
            "data": [{"a": 10, "b": 20}],
        }

        result = from_dict(data)

        assert isinstance(result, DataNodeDeser)
        assert isinstance(result.data, MockDataFrame)
        assert result.data.data == [{"a": 10, "b": 20}]

    def test_round_trip_node_with_external_type(self) -> None:
        """Test round-trip for node with external type field."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            MockDataFrame,
            encode=lambda df: df.to_dicts(),
            decode=lambda data: MockDataFrame.from_dicts(data),
        )

        class DataNodeRT(Node[None], tag="data_node_rt"):
            data: MockDataFrame
            name: str

        df = MockDataFrame(columns=["col"], data=[{"col": 100}])
        original = DataNodeRT(data=df, name="test")

        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert isinstance(deserialized, DataNodeRT)
        assert deserialized.name == "test"
        assert deserialized.data.data == original.data.data

    def test_json_round_trip_with_external_type(self) -> None:
        """Test JSON round-trip for node with external type field."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            Point,
            encode=lambda p: {"x": p.x, "y": p.y},
            decode=lambda d: Point(x=d["x"], y=d["y"]),
        )

        class PointNode(Node[None], tag="point_node_json"):
            location: Point

        original = PointNode(location=Point(x=1.5, y=2.5))

        json_str = to_json(original)
        deserialized = from_json(json_str)

        assert isinstance(deserialized, PointNode)
        assert deserialized.location == Point(x=1.5, y=2.5)


# =============================================================================
# Test: Builtin Type Serialization in Nodes
# =============================================================================


class TestBuiltinTypeInNodes:
    """Test builtin types (bytes, datetime, etc.) as fields in Node classes."""

    def test_serialize_node_with_datetime(self) -> None:
        """Test serializing a node with datetime field."""

        class TimestampNode(Node[None], tag="timestamp_node"):
            created_at: datetime

        node = TimestampNode(created_at=datetime(2024, 1, 15, 10, 30, 0))
        result = to_dict(node)

        assert result == {
            "tag": "timestamp_node",
            "created_at": "2024-01-15T10:30:00",
        }

    def test_deserialize_node_with_datetime(self) -> None:
        """Test deserializing a node with datetime field."""

        class TimestampNodeDeser(Node[None], tag="timestamp_node_deser"):
            created_at: datetime

        data = {
            "tag": "timestamp_node_deser",
            "created_at": "2024-06-20T14:45:30",
        }
        result = from_dict(data)

        assert isinstance(result, TimestampNodeDeser)
        assert result.created_at == datetime(2024, 6, 20, 14, 45, 30)

    def test_round_trip_node_with_datetime(self) -> None:
        """Test round-trip for node with datetime field."""

        class TimestampRT(Node[None], tag="timestamp_rt"):
            ts: datetime

        original = TimestampRT(ts=datetime(2024, 12, 25, 8, 0, 0))
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_serialize_node_with_date(self) -> None:
        """Test serializing a node with date field."""

        class DateNode(Node[None], tag="date_node"):
            birthday: date

        node = DateNode(birthday=date(1990, 5, 15))
        result = to_dict(node)

        assert result == {"tag": "date_node", "birthday": "1990-05-15"}

    def test_round_trip_node_with_date(self) -> None:
        """Test round-trip for node with date field."""

        class DateRT(Node[None], tag="date_rt"):
            day: date

        original = DateRT(day=date(2024, 7, 4))
        deserialized = from_dict(to_dict(original))

        assert deserialized == original

    def test_serialize_node_with_time(self) -> None:
        """Test serializing a node with time field."""

        class TimeNode(Node[None], tag="time_node"):
            start_time: time

        node = TimeNode(start_time=time(9, 30, 0))
        result = to_dict(node)

        assert result == {"tag": "time_node", "start_time": "09:30:00"}

    def test_round_trip_node_with_time(self) -> None:
        """Test round-trip for node with time field."""

        class TimeRT(Node[None], tag="time_rt"):
            t: time

        original = TimeRT(t=time(23, 59, 59))
        deserialized = from_dict(to_dict(original))

        assert deserialized == original

    def test_serialize_node_with_timedelta(self) -> None:
        """Test serializing a node with timedelta field."""

        class DurationNode(Node[None], tag="duration_node"):
            duration: timedelta

        node = DurationNode(duration=timedelta(hours=1, minutes=30))
        result = to_dict(node)

        assert result == {"tag": "duration_node", "duration": 5400.0}

    def test_round_trip_node_with_timedelta(self) -> None:
        """Test round-trip for node with timedelta field."""

        class DurationRT(Node[None], tag="duration_rt"):
            d: timedelta

        original = DurationRT(d=timedelta(days=1, hours=2, minutes=3))
        deserialized = from_dict(to_dict(original))

        assert deserialized == original

    def test_serialize_node_with_bytes(self) -> None:
        """Test serializing a node with bytes field."""

        class BinaryNode(Node[None], tag="binary_node"):
            data: bytes

        node = BinaryNode(data=b"hello")
        result = to_dict(node)

        assert result["tag"] == "binary_node"
        # Should be base64 encoded
        assert result["data"] == base64.b64encode(b"hello").decode("ascii")

    def test_round_trip_node_with_bytes(self) -> None:
        """Test round-trip for node with bytes field."""

        class BinaryRT(Node[None], tag="binary_rt"):
            payload: bytes

        original = BinaryRT(payload=b"\x00\x01\x02\xff")
        deserialized = from_dict(to_dict(original))

        assert deserialized == original

    def test_serialize_node_with_decimal(self) -> None:
        """Test serializing a node with Decimal field."""

        class MoneyNode(Node[None], tag="money_node"):
            amount: Decimal

        node = MoneyNode(amount=Decimal("99.99"))
        result = to_dict(node)

        assert result == {"tag": "money_node", "amount": "99.99"}

    def test_round_trip_node_with_decimal(self) -> None:
        """Test round-trip for node with Decimal field."""

        class MoneyRT(Node[None], tag="money_rt"):
            price: Decimal

        original = MoneyRT(price=Decimal("123.456"))
        deserialized = from_dict(to_dict(original))

        assert deserialized == original
        assert isinstance(deserialized.price, Decimal)


# =============================================================================
# Test: Nested External Types
# =============================================================================


class TestNestedExternalTypes:
    """Test external types nested within containers and other nodes."""

    def test_list_of_external_type(self) -> None:
        """Test node with list of external type."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            Point,
            encode=lambda p: [p.x, p.y],
            decode=lambda d: Point(x=d[0], y=d[1]),
        )

        class PolygonNode(Node[None], tag="polygon_node"):
            vertices: list[Point]

        polygon = PolygonNode(vertices=[Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)])

        serialized = to_dict(polygon)
        assert serialized == {
            "tag": "polygon_node",
            "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }

        deserialized = from_dict(serialized)
        assert isinstance(deserialized, PolygonNode)
        assert len(deserialized.vertices) == 4
        assert all(isinstance(v, Point) for v in deserialized.vertices)
        assert deserialized.vertices[0] == Point(0, 0)

    def test_dict_with_external_type_values(self) -> None:
        """Test node with dict containing external type values."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            Point,
            encode=lambda p: [p.x, p.y],
            decode=lambda d: Point(x=d[0], y=d[1]),
        )

        class NamedPointsNode(Node[None], tag="named_points_node"):
            points: dict[str, Point]

        node = NamedPointsNode(points={"origin": Point(0, 0), "target": Point(10, 20)})

        serialized = to_dict(node)
        assert serialized == {
            "tag": "named_points_node",
            "points": {"origin": [0, 0], "target": [10, 20]},
        }

        deserialized = from_dict(serialized)
        assert isinstance(deserialized, NamedPointsNode)
        assert deserialized.points["origin"] == Point(0, 0)
        assert deserialized.points["target"] == Point(10, 20)

    def test_optional_external_type(self) -> None:
        """Test node with optional external type field."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            Point,
            encode=lambda p: [p.x, p.y],
            decode=lambda d: Point(x=d[0], y=d[1]),
        )

        class OptionalPointNode(Node[None], tag="optional_point_node"):
            location: Point | None

        # With value
        node_with = OptionalPointNode(location=Point(5, 5))
        serialized_with = to_dict(node_with)
        assert serialized_with == {"tag": "optional_point_node", "location": [5, 5]}

        deserialized_with = from_dict(serialized_with)
        assert deserialized_with.location == Point(5, 5)

        # With None
        node_none = OptionalPointNode(location=None)
        serialized_none = to_dict(node_none)
        assert serialized_none == {"tag": "optional_point_node", "location": None}

        deserialized_none = from_dict(serialized_none)
        assert deserialized_none.location is None


# =============================================================================
# Test: Nested Builtin Types
# =============================================================================


class TestNestedBuiltinTypes:
    """Test builtin types nested within containers."""

    def test_list_of_datetime(self) -> None:
        """Test node with list of datetime."""

        class EventsNode(Node[None], tag="events_node"):
            timestamps: list[datetime]

        node = EventsNode(
            timestamps=[
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 6, 15, 12, 30, 0),
            ]
        )

        serialized = to_dict(node)
        assert serialized == {
            "tag": "events_node",
            "timestamps": ["2024-01-01T00:00:00", "2024-06-15T12:30:00"],
        }

        deserialized = from_dict(serialized)
        assert isinstance(deserialized, EventsNode)
        assert len(deserialized.timestamps) == 2
        assert all(isinstance(ts, datetime) for ts in deserialized.timestamps)

    def test_dict_with_decimal_values(self) -> None:
        """Test node with dict containing Decimal values."""

        class PricesNode(Node[None], tag="prices_node"):
            prices: dict[str, Decimal]

        node = PricesNode(prices={"apple": Decimal("1.99"), "banana": Decimal("0.50")})

        serialized = to_dict(node)
        assert serialized == {
            "tag": "prices_node",
            "prices": {"apple": "1.99", "banana": "0.50"},
        }

        deserialized = from_dict(serialized)
        assert isinstance(deserialized, PricesNode)
        assert deserialized.prices["apple"] == Decimal("1.99")
        assert isinstance(deserialized.prices["apple"], Decimal)


# =============================================================================
# Test: Mixed External and Builtin Types
# =============================================================================


class TestMixedTypes:
    """Test nodes with both external and builtin type fields."""

    def test_node_with_multiple_special_types(self) -> None:
        """Test node with multiple external and builtin type fields."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            Point,
            encode=lambda p: [p.x, p.y],
            decode=lambda d: Point(x=d[0], y=d[1]),
        )

        class ComplexNode(Node[None], tag="complex_node_mixed"):
            location: Point
            created_at: datetime
            amount: Decimal
            data: bytes

        original = ComplexNode(
            location=Point(1.5, 2.5),
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            amount=Decimal("99.99"),
            data=b"binary",
        )

        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert isinstance(deserialized, ComplexNode)
        assert deserialized.location == Point(1.5, 2.5)
        assert deserialized.created_at == datetime(2024, 1, 15, 10, 0, 0)
        assert deserialized.amount == Decimal("99.99")
        assert deserialized.data == b"binary"


# =============================================================================
# Test: Error Cases
# =============================================================================


class TestCodecErrors:
    """Test error handling for codec operations."""

    def test_serialize_unregistered_external_type_raises(self) -> None:
        """Test that serializing unregistered external type raises error."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()

        class UnknownType:
            pass

        class BadNode(Node[None], tag="bad_node_unregistered"):
            value: UnknownType

        node = BadNode(value=UnknownType())

        # Should raise because UnknownType has no codec and isn't JSON serializable
        with pytest.raises(TypeError):
            to_json(node)

    def test_deserialize_with_invalid_data_for_codec(self) -> None:
        """Test deserializing with data that doesn't match expected format."""

        class DateNodeBad(Node[None], tag="date_node_bad"):
            day: date

        data = {"tag": "date_node_bad", "day": "not-a-valid-date"}

        with pytest.raises(ValueError):
            from_dict(data)


# =============================================================================
# Test: Walrus Pattern Usage (implementation detail verification)
# =============================================================================


class TestWalrusPatternUsage:
    """Test that the walrus operator pattern works correctly with TypeCodecs.get()."""

    def test_walrus_pattern_with_registered_type(self) -> None:
        """Verify the walrus pattern works for codec lookup."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()
        TypeCodecs.register(
            Point,
            encode=lambda p: [p.x, p.y],
            decode=lambda d: Point(x=d[0], y=d[1]),
        )

        # This is the pattern used in serialization
        point = Point(1, 2)
        if codec := TypeCodecs.get(type(point)):
            encode, _ = codec
            result = encode(point)
            assert result == [1, 2]
        else:
            pytest.fail("Codec should have been found")

    def test_walrus_pattern_with_unregistered_type(self) -> None:
        """Verify the walrus pattern correctly handles unregistered types."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()

        class NotRegistered:
            pass

        obj = NotRegistered()
        if codec := TypeCodecs.get(type(obj)):
            pytest.fail("Should not find codec for unregistered type")
        else:
            # This is expected - no codec found
            pass


# =============================================================================
# Test: Encoder Returns Complex Objects (recursive encoding)
# =============================================================================


class TestRecursiveEncoding:
    """Test that encoder output is recursively processed."""

    def test_encoder_returns_nested_nodes(self) -> None:
        """Test external type whose encoder returns objects needing further encoding."""
        from typedsl.codecs import TypeCodecs

        TypeCodecs.clear()

        @dataclass
        class Container:
            items: list[datetime]

        TypeCodecs.register(
            Container,
            encode=lambda c: {"items": c.items},  # Returns datetimes, need encoding
            decode=lambda d: Container(items=d["items"]),
        )

        class ContainerNode(Node[None], tag="container_node_recursive"):
            data: Container

        node = ContainerNode(
            data=Container(items=[datetime(2024, 1, 1), datetime(2024, 6, 15)])
        )

        serialized = to_dict(node)

        # The datetimes inside should be encoded to ISO strings
        assert serialized == {
            "tag": "container_node_recursive",
            "data": {
                "items": ["2024-01-01T00:00:00", "2024-06-15T00:00:00"],
            },
        }
