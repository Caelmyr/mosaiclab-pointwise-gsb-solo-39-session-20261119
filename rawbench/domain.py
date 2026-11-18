"""Contracts shared by sensor import, processing, and saved inspection."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

MAX_PIXELS = 4_194_304
MAX_DIMENSION = 8192
PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
PATTERNS = {
    "RGGB": ("R", "G", "G", "B"),
    "BGGR": ("B", "G", "G", "R"),
    "GRBG": ("G", "R", "B", "G"),
    "GBRG": ("G", "B", "R", "G"),
}
CHANNELS = ("R", "G", "B")
ROLE_VALUES = ("scene", "dark", "flat", "defects")
IDENTIFIER = re.compile(r"^[a-f0-9]{16}$")


class InputError(ValueError):
    pass


class MissingError(LookupError):
    pass


class ConflictError(RuntimeError):
    pass


class CapacityError(RuntimeError):
    pass


def object_value(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise InputError(f"{name} must be an object.")
    return value


def reject_unknown(source: Mapping, fields: Sequence[str], name: str) -> None:
    unknown = sorted(set(source) - set(fields))
    if unknown:
        raise InputError(f"Unknown {name} field: {unknown[0]}.")


def finite(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise InputError(f"{name} must be a number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"{name} must be a number.") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise InputError(f"{name} must be between {low:g} and {high:g}.")
    return result


def integer(value: Any, name: str, low: int, high: int) -> int:
    number = finite(value, name, low, high)
    if not number.is_integer():
        raise InputError(f"{name} must be an integer.")
    return int(number)


def flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise InputError(f"{name} must be true or false.")
    return value


def choice(value: Any, name: str, options: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in options:
        raise InputError(f"{name} must be one of {', '.join(options)}.")
    return value


def text_label(value: Any, name: str, maximum: int = 80) -> str:
    if not isinstance(value, str):
        raise InputError(f"{name} must be text.")
    value = value.strip()
    if not value or len(value) > maximum:
        raise InputError(f"{name} must have 1–{maximum} characters.")
    if any(not letter.isprintable() for letter in value):
        raise InputError(f"{name} contains a control character.")
    return value


def optional_id(value: Any, name: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise InputError(f"{name} has an invalid identifier.")
    return value


def check_dimensions(width: int, height: int) -> None:
    if min(width, height) < 2:
        raise InputError("A sensor raster requires at least two rows and columns.")
    if max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise InputError(
            f"Maximum sensor size is {MAX_PIXELS:,} samples, "
            f"with no side longer than {MAX_DIMENSION:,}."
        )


def bounded_numbers(
    values: Any, label: str, count: int, low: float, high: float
) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise InputError(f"{label} requires {count} values.")
    return tuple(finite(value, f"{label} {i + 1}", low, high) for i, value in enumerate(values))


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def full(cls, width: int, height: int) -> Box:
        return cls(0, 0, width, height)

    @classmethod
    def read(
        cls, value: Any, width: int, height: int, *, minimum: int = 1
    ) -> Box:
        if value is None:
            return cls.full(width, height)
        obj = object_value(value, "Region")
        reject_unknown(obj, ("x", "y", "width", "height"), "region")
        x = integer(obj.get("x"), "Region x", 0, width - 1)
        y = integer(obj.get("y"), "Region y", 0, height - 1)
        w = integer(obj.get("width"), "Region width", minimum, width)
        h = integer(obj.get("height"), "Region height", minimum, height)
        if x + w > width or y + h > height:
            raise InputError("The region extends outside the sensor raster.")
        return cls(x, y, w, h)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, array: np.ndarray) -> np.ndarray:
        return array[self.y:self.bottom, self.x:self.right]

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def intersection(self, other: Box) -> Box | None:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= x or bottom <= y:
            return None
        return Box(x, y, right - x, bottom - y)

    def translated(self, dx: int, dy: int) -> Box:
        return Box(self.x + dx, self.y + dy, self.width, self.height)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class Layout:
    pattern: str
    origin_x: int
    origin_y: int

    @property
    def colors(self) -> tuple[str, ...]:
        return PATTERNS[self.pattern]

    def index(self, x: int, y: int) -> int:
        return ((y + self.origin_y) & 1) * 2 + ((x + self.origin_x) & 1)

    def local_offset(self, index: int) -> tuple[int, int]:
        y, x = PHASES[index]
        return (y - self.origin_y) & 1, (x - self.origin_x) & 1

    def phase_slice(self, index: int) -> tuple[slice, slice]:
        y, x = self.local_offset(index)
        return slice(y, None, 2), slice(x, None, 2)

    def shifted(self, x: int, y: int) -> Layout:
        return replace(self, origin_x=self.origin_x + x, origin_y=self.origin_y + y)

    def phase_name(self, index: int) -> str:
        color = self.colors[index]
        if color != "G":
            return color
        return "G1" if self.colors.index("G") == index else "G2"

    def local_pattern(self) -> str:
        return "".join(self.colors[self.index(x, y)] for y, x in PHASES)

    def local_index_grid(self) -> np.ndarray:
        return np.array([[self.index(x, y) for x in range(2)] for y in range(2)])


@dataclass(frozen=True)
class Sensor:
    layout: Layout
    black: tuple[float, ...]
    white: tuple[float, ...]
    active: Box
    exposure_ms: float | None
    analog_gain: float | None
    model: str

    @classmethod
    def from_values(
        cls, value: Any, width: int, height: int, maximum: int
    ) -> Sensor:
        obj = object_value(value if value is not None else {}, "Sensor metadata")
        reject_unknown(
            obj,
            ("pattern", "origin_x", "origin_y", "black", "white", "active",
             "exposure_ms", "analog_gain", "model"),
            "sensor metadata",
        )
        pattern = choice(obj.get("pattern", "RGGB"), "Bayer pattern", tuple(PATTERNS))
        ox = integer(obj.get("origin_x", 0), "Origin x", 0, 131072)
        oy = integer(obj.get("origin_y", 0), "Origin y", 0, 131072)
        black = cls._levels(obj.get("black", 0), "Black level", maximum)
        white = cls._levels(obj.get("white", maximum), "White level", maximum)
        if any(hi <= lo for lo, hi in zip(black, white)):
            raise InputError("Each white level must be greater than its black level.")
        exposure = obj.get("exposure_ms")
        gain = obj.get("analog_gain")
        raw_model = obj.get("model", "")
        model = "" if raw_model == "" else text_label(raw_model, "Sensor model")
        return cls(
            Layout(pattern, ox, oy),
            black,
            white,
            Box.read(obj.get("active"), width, height, minimum=2),
            None if exposure is None else finite(exposure, "Exposure (ms)", 0.001, 3_600_000),
            None if gain is None else finite(gain, "Analog gain", 0.01, 10000),
            model,
        )

    @staticmethod
    def _levels(raw: Any, name: str, maximum: int) -> tuple[float, ...]:
        if isinstance(raw, (tuple, list)):
            return bounded_numbers(raw, name, 4, 0, maximum)
        value = finite(raw, name, 0, maximum)
        return (value,) * 4

    def matches_raster(self, other: Sensor) -> bool:
        return (
            self.layout == other.layout
            and self.active == other.active
            and (not self.model or not other.model or self.model == other.model)
        )

    def as_dict(self) -> dict:
        return {
            "pattern": self.layout.pattern,
            "origin_x": self.layout.origin_x,
            "origin_y": self.layout.origin_y,
            "black": list(self.black),
            "white": list(self.white),
            "active": self.active.as_dict(),
            "exposure_ms": self.exposure_ms,
            "analog_gain": self.analog_gain,
            "model": self.model,
        }


@dataclass(frozen=True)
class Raster:
    values: np.ndarray
    maximum: int
    encoding: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise InputError("Only one sample per sensor pixel is supported.")
        if self.values.dtype != np.uint16:
            raise InputError("Sensor samples must be uint16.")
        check_dimensions(self.width, self.height)
        if not 1 <= self.maximum <= 65535:
            raise InputError("The declared code maximum must be 1–65535.")
        if int(self.values.max()) > self.maximum:
            raise InputError("A pixel exceeds the declared code maximum.")

    @property
    def width(self) -> int:
        return self.values.shape[1]

    @property
    def height(self) -> int:
        return self.values.shape[0]


@dataclass(frozen=True)
class Frame:
    id: str
    project: str
    name: str
    original_name: str
    role: str
    source_digest: str
    raster: Raster
    sensor: Sensor
    synthetic: bool = False

    def public(self) -> dict:
        return {
            "id": self.id,
            "project": self.project,
            "name": self.name,
            "original_name": self.original_name,
            "role": self.role,
            "source_digest": self.source_digest,
            "width": self.raster.width,
            "height": self.raster.height,
            "maximum": self.raster.maximum,
            "encoding": self.raster.encoding,
            "warnings": list(self.raster.warnings),
            "sensor": self.sensor.as_dict(),
            "local_pattern": self.sensor.layout.local_pattern(),
            "synthetic": self.synthetic,
        }


def sensor_compatible(left: Frame, right: Frame) -> bool:
    return (
        left.raster.values.shape == right.raster.values.shape
        and left.sensor.matches_raster(right.sensor)
    )
