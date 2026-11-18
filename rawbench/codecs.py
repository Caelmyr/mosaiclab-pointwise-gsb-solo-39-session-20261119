"""Import formats that can preserve a sensor site's original integer code."""
from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath

import numpy as np
from PIL import Image, UnidentifiedImageError

from .domain import (
    MAX_PIXELS,
    InputError,
    Raster,
    Sensor,
    check_dimensions,
    choice,
    integer,
    object_value,
    reject_unknown,
)

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 65536
MANIFEST_LIMIT = 16384
PGM_TOKEN_LIMIT = 20
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SPACE = b" \r\n\t\v\f"
DIGITS = re.compile(rb"^[0-9]+$")
PORTABLE_FORMAT = "mosaiclab.raw/1"
BINARY_FORMAT = "mosaiclab.raw-binary/1"
BUNDLE_PIXEL_NAMES = {PORTABLE_FORMAT: "mosaic.pgm", BINARY_FORMAT: "mosaic.raw"}
WORD_DTYPES = {"u8": "u1", "u16le": "<u2", "u16be": ">u2"}


@dataclass(frozen=True)
class BinaryEncoding:
    label: str
    depth: int
    samples_per_group: int = 1

    @property
    def group_bytes(self) -> int:
        return (self.depth * self.samples_per_group) // 8

    @property
    def maximum(self) -> int:
        return (1 << self.depth) - 1


RAW_MODES = {
    "u8": BinaryEncoding("8-bit unsigned", 8),
    "u16le": BinaryEncoding("16-bit little endian", 16),
    "u16be": BinaryEncoding("16-bit big endian", 16),
    "raw10p": BinaryEncoding("V4L2 packed RAW10", 10, 4),
    "raw12p": BinaryEncoding("V4L2 packed RAW12", 12, 2),
    "raw14p": BinaryEncoding("V4L2 packed RAW14", 14, 4),
}


def binary_encodings() -> list[dict]:
    return [
        {"id": key, "label": spec.label, "maximum": spec.maximum,
         "group_samples": spec.samples_per_group, "group_bytes": spec.group_bytes}
        for key, spec in RAW_MODES.items()
    ]


@dataclass(frozen=True)
class DecodedCapture:
    raster: Raster
    metadata: dict
    suggested_name: str
    digest: str
    synthetic: bool = False

    def sensor(self, override: Mapping | None = None) -> Sensor:
        effective = {**self.metadata, **(override or {})}
        return Sensor.from_values(
            effective, self.raster.width, self.raster.height, self.raster.maximum
        )


def digest_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def read_limited(stream, limit: int = MAX_FILE_BYTES) -> bytes:
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise InputError(f"Upload exceeds {limit // 1048576} MiB.")
    if not body:
        raise InputError("The uploaded file is empty.")
    return body


def filename_label(name: str) -> str:
    safe = PurePath(name.replace("\\", "/")).name
    safe = "".join(letter for letter in safe if letter.isprintable()).strip(". ")
    return safe[:128] or "mosaic"


def json_object(body: bytes | str, label: str) -> Mapping:
    try:
        result = json.loads(body, parse_constant=_bad_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(f"{label} is not valid JSON.") from exc
    return object_value(result, label)


def _bad_constant(value: str):
    raise ValueError(value)


class PGMReader:
    def __init__(self, body: bytes):
        self.body = body
        self.offset = 0

    def token(self, *, limit: int = PGM_TOKEN_LIMIT) -> bytes:
        self.skip_space_and_comments()
        start = self.offset
        while self.offset < len(self.body):
            current = self.body[self.offset]
            if current in SPACE or current == 35:
                break
            self.offset += 1
            if self.offset - start > limit:
                raise InputError("A PGM token is too long.")
        if start == self.offset:
            raise InputError("Expected another PGM token.")
        return self.body[start:self.offset]

    def skip_space_and_comments(self) -> None:
        while self.offset < len(self.body):
            value = self.body[self.offset]
            if value in SPACE:
                self.offset += 1
                continue
            if value == 35:
                lf = self.body.find(b"\n", self.offset + 1)
                cr = self.body.find(b"\r", self.offset + 1)
                ends = [end for end in (lf, cr) if end >= 0]
                self.offset = min(ends) if ends else len(self.body)
                continue
            break

    def positive_integer(self, name: str, minimum: int, maximum: int) -> int:
        token = self.token()
        if not DIGITS.fullmatch(token):
            raise InputError(f"Invalid PGM {name}.")
        value = int(token)
        if not minimum <= value <= maximum:
            raise InputError(f"PGM {name} must be between {minimum} and {maximum}.")
        return value

    def binary_pixels(self, width: int, height: int, maximum: int) -> np.ndarray:
        count = width * height
        code_bytes = 1 if maximum < 256 else 2
        if self.offset >= len(self.body) or self.body[self.offset] not in SPACE:
            raise InputError("PGM binary header needs a whitespace delimiter.")
        raster_start = self.offset + 1
        # Treat CRLF as a pair only when required by the exact payload length.
        if (
            self.body[self.offset:raster_start + 1] == b"\r\n"
            and len(self.body) - raster_start == count * code_bytes + 1
        ):
            raster_start += 1
        if len(self.body) - raster_start != count * code_bytes:
            raise InputError("PGM pixel bytes do not match the header dimensions.")
        dtype = "u1" if code_bytes == 1 else ">u2"
        data = np.frombuffer(self.body, dtype=dtype, offset=raster_start)
        return data.astype(np.uint16).reshape((height, width))

    def plain_pixels(self, width: int, height: int, maximum: int) -> np.ndarray:
        total = width * height
        if total > 1_048_576:
            raise InputError("Plain PGM is limited to 1,048,576 samples. Use P5 for larger rasters.")
        values = np.fromiter(
            (self.positive_integer("sample", 0, maximum) for _ in range(total)),
            dtype=np.uint16,
            count=total,
        )
        self.skip_space_and_comments()
        if self.offset != len(self.body):
            raise InputError("Extra image data after the PGM raster.")
        return values.reshape(height, width)


def decode_pgm(body: bytes) -> Raster:
    reader = PGMReader(body)
    kind = reader.token()
    if kind not in (b"P2", b"P5"):
        raise InputError("Only P2 and P5 PGM are supported.")
    width = reader.positive_integer("width", 2, 8192)
    height = reader.positive_integer("height", 2, 8192)
    check_dimensions(width, height)
    maximum = reader.positive_integer("maximum", 1, 65535)
    if reader.offset > MAX_HEADER_BYTES:
        raise InputError("PGM header exceeds the size limit.")
    loader = reader.binary_pixels if kind == b"P5" else reader.plain_pixels
    data = loader(width, height, maximum)
    return Raster(data, maximum, kind.decode())


def encode_pgm(values: np.ndarray, maximum: int) -> bytes:
    height, width = values.shape
    prefix = f"P5\n{width} {height}\n{maximum}\n".encode("ascii")
    dtype = "u1" if maximum < 256 else ">u2"
    return prefix + values.astype(dtype, copy=False).tobytes(order="C")


def decode_gray_png(body: bytes) -> Raster:
    if not body.startswith(PNG_SIGNATURE) or len(body) < 33:
        raise InputError("The file does not have a valid PNG header.")
    if body[8:16] != b"\x00\x00\x00\rIHDR":
        raise InputError("PNG header must begin with IHDR.")
    width, height, depth, color = struct.unpack_from(">IIBB", body, 16)
    check_dimensions(width, height)
    if color != 0 or depth not in (8, 16):
        raise InputError("Use 8-bit or 16-bit grayscale PNG with one code per sensor site.")
    try:
        with Image.open(io.BytesIO(body), formats=["PNG"]) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise InputError("Image sequences are not supported.")
            if image.info.get("transparency") is not None:
                raise InputError("Grayscale PNG with a transparency key is not supported.")
            image.load()
            values = np.array(image, dtype=np.uint16)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise InputError("Grayscale PNG could not be decoded.") from exc
    if values.shape != (height, width):
        raise InputError("PNG decoded dimensions do not match its header.")
    warnings = ("PNG gamma and color-profile tags are ignored for raw numeric samples.",)
    return Raster(values, (1 << depth) - 1, f"PNG{depth}", warnings)


def _numpy_header(body: bytes) -> tuple[io.BytesIO, tuple[int, ...], bool, np.dtype]:
    reader = io.BytesIO(body)
    try:
        version = np.lib.format.read_magic(reader)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                reader, max_header_size=MAX_HEADER_BYTES
            )
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                reader, max_header_size=MAX_HEADER_BYTES
            )
        else:
            raise InputError("Only NPY headers 1.0 and 2.0 are supported.")
    except InputError:
        raise
    except (TypeError, ValueError, EOFError, SyntaxError, RecursionError) as exc:
        raise InputError("The NPY header is invalid.") from exc
    return reader, shape, fortran, dtype


def decode_npy(body: bytes) -> Raster:
    reader, shape, fortran, dtype = _numpy_header(body)
    if len(shape) != 2 or dtype.kind not in "ui" or dtype.itemsize not in (1, 2, 4, 8):
        raise InputError("NPY requires one 2D integer array without objects.")
    height, width = shape
    check_dimensions(width, height)
    count = width * height
    if len(body) - reader.tell() != count * dtype.itemsize:
        raise InputError("NPY pixel bytes do not match the array shape.")
    data = np.frombuffer(body, dtype=dtype, offset=reader.tell(), count=count)
    if (dtype.kind == "i" and data.min() < 0) or data.max() > 65535:
        raise InputError("NPY sample codes must be between 0 and 65535.")
    data = data.reshape(shape, order="F" if fortran else "C")
    maximum = 255 if dtype.itemsize == 1 and dtype.kind == "u" else 65535
    return Raster(data.astype(np.uint16), maximum, f"NPY {dtype.name}")


@dataclass(frozen=True)
class BinaryLayout:
    width: int
    height: int
    format: str
    maximum: int
    row_bytes: int
    stride: int
    offset: int

    @classmethod
    def read(cls, options: Mapping) -> BinaryLayout:
        reject_unknown(
            options, ("width", "height", "encoding", "row_stride", "offset", "maximum"),
            "binary layout",
        )
        width = integer(options.get("width"), "Width", 2, 8192)
        height = integer(options.get("height"), "Height", 2, 8192)
        check_dimensions(width, height)
        encoding = choice(options.get("encoding", "u16le"), "Binary encoding", tuple(RAW_MODES))
        spec = RAW_MODES[encoding]
        if width % spec.samples_per_group:
            raise InputError(f"{spec.label} needs rows in groups of {spec.samples_per_group} sites.")
        maximum = integer(options.get("maximum", spec.maximum), "Maximum code", 1, spec.maximum)
        row_bytes = width // spec.samples_per_group * spec.group_bytes
        stride = integer(options.get("row_stride", row_bytes), "Row stride", row_bytes, MAX_FILE_BYTES)
        offset = integer(options.get("offset", 0), "Header byte offset", 0, MAX_FILE_BYTES)
        return cls(width, height, encoding, maximum, row_bytes, stride, offset)

    def check_length(self, size: int) -> int:
        last_data_byte = self.offset + (self.height - 1) * self.stride + self.row_bytes
        if size < last_data_byte:
            raise InputError(f"Binary file ends before the last complete row. Need {last_data_byte} bytes, received {size}.")
        if size > self.offset + self.height * self.stride:
            raise InputError("Binary file contains bytes beyond the declared row layout.")
        return size - last_data_byte

    def report(self, size: int) -> dict:
        return {
            "encoding": self.format,
            "width": self.width,
            "height": self.height,
            "maximum": self.maximum,
            "active_row_bytes": self.row_bytes,
            "row_stride": self.stride,
            "header_bytes": self.offset,
            "last_row_padding_bytes": self.check_length(size),
            "size_bytes": size,
        }

    def options(self) -> dict:
        return {
            "encoding": self.format,
            "maximum": self.maximum,
            "width": self.width,
            "height": self.height,
            "row_stride": self.stride,
            "offset": self.offset,
        }


def _packed_pixels(body: bytes, layout: BinaryLayout, format: BinaryEncoding) -> np.ndarray:
    groups = np.ndarray(
        shape=(layout.height, layout.width // format.samples_per_group, format.group_bytes),
        dtype=np.uint8, buffer=body, offset=layout.offset,
        strides=(layout.stride, format.group_bytes, 1),
    )
    extra_bits = format.depth - 8
    samples = format.samples_per_group
    low = np.zeros(groups.shape[:2], dtype=np.uint32)
    for byte_index in range(format.group_bytes - samples):
        low |= groups[..., samples + byte_index].astype(np.uint32) << (8 * byte_index)
    result = np.empty((layout.height, layout.width), dtype=np.uint16)
    low_mask = (1 << extra_bits) - 1
    for index in range(samples):
        tail = ((low >> (index * extra_bits)) & low_mask).astype(np.uint16)
        result[:, index::samples] = (groups[..., index].astype(np.uint16) << extra_bits) | tail
    return result


def decode_binary_layout(body: bytes, layout: BinaryLayout) -> Raster:
    layout.check_length(len(body))
    format = RAW_MODES[layout.format]
    if format.samples_per_group == 1:
        image = np.ndarray(
            shape=(layout.height, layout.width), dtype=WORD_DTYPES[layout.format],
            buffer=body, offset=layout.offset, strides=(layout.stride, format.group_bytes),
        ).astype(np.uint16)
        return Raster(image, layout.maximum, f"Binary {layout.format}")
    image = _packed_pixels(body, layout, format)
    return Raster(image, layout.maximum, format.label, (
        f"Decoded {layout.row_bytes} active bytes per row with {layout.stride}-byte row stride and {layout.offset}-byte offset.",
    ))


def decode_binary(body: bytes, options: Mapping) -> Raster:
    return decode_binary_layout(body, BinaryLayout.read(options))


def inspect_binary(body: bytes, options: Mapping) -> dict:
    layout = BinaryLayout.read(options)
    raster = decode_binary_layout(body, layout)
    signals = []
    for row, column in ((0, 0), (0, 1), (1, 0), (1, 1)):
        sites = raster.values[row::2, column::2]
        signals.append({
            "row_parity": row,
            "column_parity": column,
            "samples": int(sites.size),
            "minimum": int(sites.min()),
            "maximum": int(sites.max()),
            "median": float(np.median(sites)),
            "zeroes": int(np.count_nonzero(sites == 0)),
            "at_declared_maximum": int(np.count_nonzero(sites == layout.maximum)),
        })
    return {
        **layout.report(len(body)),
        "sha256": digest_bytes(body),
        "observed_minimum": int(raster.values.min()),
        "observed_maximum": int(raster.values.max()),
        "edge_sites": {
            "first_row_start": raster.values[0, :min(8, layout.width)].tolist(),
            "last_row_end": raster.values[-1, -min(8, layout.width):].tolist(),
        },
        "raster_parity": signals,
    }


def _encode_groups(values: np.ndarray, spec: BinaryEncoding) -> np.ndarray:
    height, width = values.shape
    samples = spec.samples_per_group
    shaped = values.reshape(height, width // samples, samples)
    extra_bits = spec.depth - 8
    low_mask = (1 << extra_bits) - 1
    groups = np.empty((*shaped.shape[:2], spec.group_bytes), dtype=np.uint8)
    groups[..., :samples] = shaped >> extra_bits
    low = np.zeros(shaped.shape[:2], dtype=np.uint32)
    for index in range(samples):
        code = shaped[..., index].astype(np.uint32)
        low |= (code & low_mask) << (extra_bits * index)
    for index in range(spec.group_bytes - samples):
        groups[..., samples + index] = (low >> (8 * index)) & 255
    return groups.reshape(height, -1)


def encode_binary(raster: Raster, encoding: str, alignment: int) -> tuple[bytes, BinaryLayout]:
    spec = RAW_MODES[encoding]
    if raster.maximum > spec.maximum:
        raise InputError(f"{spec.label} cannot represent this capture's declared code maximum ({raster.maximum}).")
    layout = BinaryLayout.read({
        "encoding": encoding, "maximum": raster.maximum,
        "width": raster.width, "height": raster.height,
    })
    if spec.samples_per_group == 1:
        values = np.ascontiguousarray(raster.values, dtype=WORD_DTYPES[encoding])
        rows = values.view(np.uint8).reshape(layout.height, layout.row_bytes)
    else:
        rows = _encode_groups(raster.values, spec)
    stride = -(-layout.row_bytes // alignment) * alignment
    if stride != layout.row_bytes:
        padding = np.zeros((layout.height, stride), dtype=np.uint8)
        padding[:, :layout.row_bytes] = rows
        rows = padding
        layout = BinaryLayout.read({**layout.options(), "row_stride": stride})
    return rows.tobytes(), layout


def _capture_member(bundle: zipfile.ZipFile, member: str, maximum: int) -> bytes:
    info = bundle.getinfo(member)
    if info.is_dir() or not 1 <= info.file_size <= maximum or info.flag_bits & 1:
        raise InputError("Capture bundle has an unsupported or overlarge member.")
    return bundle.read(info)


def decode_portable(body: bytes) -> DecodedCapture:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as bundle:
            names = bundle.namelist()
            if len(names) != 2 or "capture.json" not in names:
                raise InputError("Capture bundle needs a capture.json and one pixel file.")
            meta = json_object(_capture_member(bundle, "capture.json", MANIFEST_LIMIT), "Capture manifest")
            version = choice(meta.get("format"), "Capture bundle version", tuple(BUNDLE_PIXEL_NAMES))
            member = BUNDLE_PIXEL_NAMES[version]
            if member not in names:
                raise InputError(f"Capture manifest requires a {member} member.")
            limit = MAX_FILE_BYTES if version == BINARY_FORMAT else MAX_PIXELS * 2 + MAX_HEADER_BYTES
            data = _capture_member(bundle, member, limit)
    except (OSError, zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        raise InputError("The capture bundle could not be opened.") from exc
    fields = ("format", "name", "sensor", "mosaic_sha256", "synthetic")
    layout = None
    if version == BINARY_FORMAT:
        fields += ("binary",)
        layout = BinaryLayout.read(object_value(meta.get("binary"), "Binary capture layout"))
    reject_unknown(meta, fields, "capture manifest")
    if meta.get("mosaic_sha256") != digest_bytes(data):
        raise InputError("Capture bundle pixel checksum does not match.")
    synthetic = meta.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise InputError("Capture synthetic flag must be true or false.")
    name = filename_label(str(meta.get("name", "capture")))
    raster = decode_pgm(data) if layout is None else decode_binary_layout(data, layout)
    sensor = dict(object_value(meta.get("sensor", {}), "Capture sensor metadata"))
    Sensor.from_values(sensor, raster.width, raster.height, raster.maximum)
    return DecodedCapture(raster, sensor, name, digest_bytes(body), synthetic)


def decode_capture(
    body: bytes, name: str, *, binary: Mapping | None = None
) -> DecodedCapture:
    safe_name = filename_label(name)
    if binary is not None:
        raster = decode_binary(body, binary)
    elif body.startswith(b"P5") or body.startswith(b"P2"):
        raster = decode_pgm(body)
    elif body.startswith(PNG_SIGNATURE):
        raster = decode_gray_png(body)
    elif body.startswith(b"\x93NUMPY"):
        raster = decode_npy(body)
    elif body.startswith(b"PK\x03\x04"):
        return decode_portable(body)
    else:
        raise InputError("Use grayscale PNG, PGM, NPY, a MosaicLab bundle, or describe binary layout.")
    return DecodedCapture(raster, {}, PurePath(safe_name).stem, digest_bytes(body))


def encode_portable(
    raster: Raster, sensor: Sensor, name: str, synthetic: bool,
    *, encoding: str | None = None, alignment: int = 1,
) -> bytes:
    output = io.BytesIO()
    version = PORTABLE_FORMAT
    extra = {}
    if encoding is not None:
        version = BINARY_FORMAT
        image, layout = encode_binary(raster, encoding, alignment)
        extra["binary"] = layout.options()
    else:
        image = encode_pgm(raster.values, raster.maximum)
    manifest = {
        "format": version,
        "name": name,
        "sensor": sensor.as_dict(),
        "synthetic": synthetic,
        "mosaic_sha256": digest_bytes(image),
        **extra,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
        bundle.writestr("capture.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        bundle.writestr(BUNDLE_PIXEL_NAMES[version], image)
    return output.getvalue()
