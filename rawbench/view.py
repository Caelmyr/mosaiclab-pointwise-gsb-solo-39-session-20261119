"""Image encodings and layers for the foreground inspection canvas."""
from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .analysis import exposure_classes, number
from .calibration import Correction, normalized_source
from .cfa import neighborhood, phase_planes, sparse_false_color
from .development import Developed, quantize, srgb_encode
from .domain import Box, Frame, InputError, choice, finite, flag, integer, object_value, reject_unknown

LAYERS = ("developed", "mosaic", "phase-color", "phase-planes", "clipping", "calibration")
CALIBRATION_LAYERS = ("corrected", "gain", "defects", "validity")
PHASE_PALETTE = {
    "R": [0.95, 0.28, 0.25],
    "G1": [0.28, 0.88, 0.36],
    "G2": [0.39, 0.80, 0.78],
    "B": [0.38, 0.49, 1.0],
}


@dataclass(frozen=True)
class ViewSettings:
    layer: str = "developed"
    width: int = 1000
    sensor_gamma: float = 2.2
    green_lattice: bool = True
    calibration_layer: str = "corrected"

    @classmethod
    def read(cls, value) -> ViewSettings:
        obj = object_value(value if value is not None else {}, "View")
        reject_unknown(
            obj, ("layer", "width", "sensor_gamma", "green_lattice", "calibration_layer"), "view"
        )
        return cls(
            choice(obj.get("layer", "developed"), "Layer", LAYERS),
            integer(obj.get("width", 1000), "Preview bound", 128, 2048),
            finite(obj.get("sensor_gamma", 2.2), "Sensor preview gamma", 0.5, 5),
            flag(obj.get("green_lattice", True), "Separate green lattice"),
            choice(
                obj.get("calibration_layer", "corrected"), "Calibration layer", CALIBRATION_LAYERS
            ),
        )

    def as_dict(self) -> dict:
        return {
            "layer": self.layer,
            "width": self.width,
            "sensor_gamma": self.sensor_gamma,
            "green_lattice": self.green_lattice,
            "calibration_layer": self.calibration_layer,
        }


@dataclass
class Preview:
    pixels: np.ndarray
    raster_box: Box | None
    interpolation: str
    info: dict


def _sensor_gamma(values: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(values, 0, 1), np.float32(1 / gamma))


def _plane_montage(values: np.ndarray, frame: Frame, box: Box, gamma: float) -> Preview:
    planes = phase_planes(values, frame.sensor.layout, box)
    width = max(plane.samples.shape[1] for plane in planes)
    height = max(plane.samples.shape[0] for plane in planes)
    gap = 4
    shade = np.empty((height * 2 + gap, width * 2 + gap, 3), dtype=np.float32)
    shade[:] = (0.16, 0.20, 0.21)
    legend = []
    for plane in planes:
        row, col = divmod(plane.phase, 2)
        left, top = col * (width + gap), row * (height + gap)
        h, w = plane.samples.shape
        plane_rgb = _sensor_gamma(plane.samples, gamma)[..., None]
        color = np.array(PHASE_PALETTE[plane.name], dtype=np.float32)
        shade[top:top + h, left:left + w] = plane_rgb * (0.54 + 0.46 * color)
        legend.append({
            "phase": plane.name,
            "montage_box": Box(left, top, w, h).as_dict(),
            "raster_offset": {
                "x": box.x + plane.offset_x, "y": box.y + plane.offset_y,
            },
            "sample_step": 2,
        })
    return Preview(shade, None, "nearest", {"planes": legend, "source_box": box.as_dict()})


def _clip_map(signal: np.ndarray, gamma: float) -> tuple[np.ndarray, dict]:
    values = exposure_classes(signal)
    ground = _sensor_gamma(signal, gamma)
    rgb = np.repeat((ground * 0.56 + 0.04)[..., None], 3, axis=2)
    tones = {
        1: (0.21, 0.43, 0.96),
        2: (0.99, 0.67, 0.12),
        3: (0.97, 0.20, 0.22),
    }
    for index, tone in tones.items():
        rgb[values == index] = tone
    breakdown = {
        "under_black": int(np.count_nonzero(values == 1)),
        "near_clipping": int(np.count_nonzero(values == 2)),
        "clip": int(np.count_nonzero(values == 3)),
    }
    return rgb, breakdown


def _defect_preview(frame: Frame, box: Box, maximum: int) -> Preview:
    selected = box.crop(frame.raster.values) != 0
    total = int(np.count_nonzero(selected))
    scale = min(1, maximum / max(selected.shape))
    if scale < 1:
        # Keep every defect visible even if it is smaller than one preview pixel.
        height, width = selected.shape
        rows = max(1, round(height * scale))
        columns = max(1, round(width * scale))
        row_start = np.linspace(0, height, rows + 1, dtype=np.intp)[:-1]
        column_start = np.linspace(0, width, columns + 1, dtype=np.intp)[:-1]
        selected = np.logical_or.reduceat(selected, row_start, axis=0)
        selected = np.logical_or.reduceat(selected, column_start, axis=1)
    flags = np.empty((*selected.shape, 3), dtype=np.float32)
    flags[:] = (0.035, 0.055, 0.045)
    flags[selected] = (1.0, 0.84, 0.30)
    return Preview(flags, box, "nearest", {
        "marked_sites": total,
        "marked_preview_cells": int(np.count_nonzero(selected)),
        "aggregated_marks": scale < 1,
    })


def raw_preview(
    frame: Frame, settings: ViewSettings, box: Box, correction: Correction | None = None
) -> Preview:
    if frame.role == "defects" and settings.layer == "mosaic":
        return _defect_preview(frame, box, settings.width)
    surface = normalized_source(frame) if correction is None else correction.normalized_source
    mosaic = box.crop(surface)
    raster = frame.sensor.layout.shifted(box.x, box.y)
    if settings.layer == "mosaic":
        return Preview(_sensor_gamma(mosaic, settings.sensor_gamma), box, "nearest", {})
    if settings.layer == "phase-color":
        sparse = sparse_false_color(
            mosaic, raster, show_green_lattice=settings.green_lattice,
        )
        return Preview(_sensor_gamma(sparse, settings.sensor_gamma), box, "nearest", {})
    if settings.layer == "phase-planes":
        return _plane_montage(surface, frame, box, settings.sensor_gamma)
    if settings.layer == "clipping":
        pixels, stats = _clip_map(mosaic, settings.sensor_gamma)
        return Preview(pixels, box, "nearest", {"clip": stats})
    raise InputError("This layer requires a development or calibration result.")


def _gain_color(gain: np.ndarray) -> np.ndarray:
    relative = np.log2(np.maximum(gain, 0.0001))
    x = np.clip(relative / 1.5, -1, 1)
    damp = np.array([0.22, 0.54, 0.95], dtype=np.float32)
    amplify = np.array([0.91, 0.62, 0.26], dtype=np.float32)
    neutral = np.array([0.15, 0.18, 0.19], dtype=np.float32)
    target = np.where(x[..., None] < 0, damp, amplify)
    return neutral + np.abs(x[..., None]) * (target - neutral)


def calibration_preview(result: Correction, settings: ViewSettings, box: Box) -> Preview:
    mode = settings.calibration_layer
    signal = box.crop(result.linear)
    valid = box.crop(result.valid)
    info = {"calibration": result.report, "calibration_layer": mode}
    if mode == "gain":
        if result.gain_map is None:
            raise InputError("Select a flat reference before showing its gain.")
        gain = box.crop(result.gain_map)
        pixels = _gain_color(gain)
        pixels[box.crop(result.flat_invalid)] = (0.90, 0.24, 0.33)
        info["median_gain"] = number(np.median(gain))
        return Preview(pixels, box, "nearest", info)
    base = _sensor_gamma(signal, settings.sensor_gamma)
    if mode == "corrected":
        return Preview(base, box, "nearest", info)
    pixels = np.repeat((base * 0.5 + 0.04)[..., None], 3, axis=2)
    if mode == "defects":
        pixels[box.crop(result.repaired)] = (0.31, 0.91, 0.54)
        pixels[~valid] = (0.97, 0.24, 0.38)
    else:
        pixels[valid] = (0.25, 0.62, 0.52)
        pixels[~valid] = (0.86, 0.23, 0.40)
        pixels[box.crop(result.original_clipped)] = (0.99, 0.69, 0.17)
    return Preview(pixels, box, "nearest", info)


def developed_preview(result: Developed) -> Preview:
    pixels = result.encoded
    if result.settings.profile != "srgb":
        from .development import tone_rgb
        settings = result.settings
        pixels = srgb_encode(
            tone_rgb(result.working_rgb, settings.tone, settings.contrast, settings.saturation)
        )
    return Preview(pixels, result.box, "antialias", result.report)


def _limit_image(image: Image.Image, maximum: int, interpolation: str) -> Image.Image:
    if max(image.size) <= maximum:
        return image
    width, height = image.size
    scale = maximum / max(width, height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    mode = Image.Resampling.NEAREST if interpolation == "nearest" else Image.Resampling.LANCZOS
    return image.resize(target, mode)


def preview_png(preview: Preview, maximum: int) -> tuple[bytes, dict]:
    pixels = quantize(preview.pixels)
    image = _limit_image(Image.fromarray(pixels), maximum, preview.interpolation)
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=3)
    return output.getvalue(), {
        "width": image.width,
        "height": image.height,
        "raster_box": preview.raster_box.as_dict() if preview.raster_box else None,
        "sampling": preview.interpolation,
        **preview.info,
    }


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_rgb16(
    values: np.ndarray, *, linear: bool = False, icc_profile: bytes | None = None
) -> bytes:
    pixels = quantize(values, 16).astype(">u2", copy=False)
    height, width, channels = pixels.shape
    if channels != 3:
        raise InputError("RGB export requires three channels.")
    content = zlib.compressobj(level=4)
    block = []
    for row in pixels:
        # PNG filter 0 preserves exactly the chosen 16-bit code.
        block.append(content.compress(b"\x00" + row.tobytes()))
    block.append(content.flush())
    prefix = b"\x89PNG\r\n\x1a\n"
    if icc_profile is not None:
        label = _chunk(b"iCCP", b"MosaicLab Display P3\x00\x00" + zlib.compress(icc_profile))
    elif linear:
        label = _chunk(b"gAMA", struct.pack(">I", 100000))
        srgb_chromaticity = (31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
        label += _chunk(b"cHRM", struct.pack(">8I", *srgb_chromaticity))
    else:
        label = _chunk(b"sRGB", b"\x00")
    return b"".join((
        prefix,
        _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)),
        label, _chunk(b"IDAT", b"".join(block)), _chunk(b"IEND", b""),
    ))


def probe_report(
    frame: Frame,
    x: int,
    y: int,
    radius: int,
    correction: Correction | None = None,
    development: Developed | None = None,
) -> dict:
    loc = frame.sensor.layout.index(x, y)
    patch = neighborhood(frame.raster.values, frame.sensor.layout, x, y, radius)
    center = frame.raster.values[y, x].item()
    black = frame.sensor.black[loc]
    white = frame.sensor.white[loc]
    patch["site"] = {
        "x": x,
        "y": y,
        "sensor_x": frame.sensor.layout.origin_x + x,
        "sensor_y": frame.sensor.layout.origin_y + y,
        "phase": frame.sensor.layout.phase_name(loc),
        "raw_code": center,
        "black": black,
        "white": white,
        "normalized": number((center - black) / (white - black)),
        "active": frame.sensor.active.contains(x, y),
    }
    if correction is not None:
        patch["corrected"] = {
            "linear": number(correction.linear[y, x]),
            "valid": bool(correction.valid[y, x]),
            "repaired": bool(correction.repaired[y, x]),
            "flat_invalid": bool(correction.flat_invalid[y, x]),
        }
    if development is not None and development.box.contains(x, y):
        yy = y - development.box.y
        xx = x - development.box.x
        patch["developed"] = {
            "camera_rgb": np.round(development.reconstruction.rgb[yy, xx], 6).tolist(),
            "working_rgb": np.round(development.working_rgb[yy, xx], 6).tolist(),
            "display_rgb": quantize(development.encoded[yy, xx]).tolist(),
        }
    return patch
