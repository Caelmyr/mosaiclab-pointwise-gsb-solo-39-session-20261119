"""Compare two developments at shared sensor locations without resampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .analysis import describe_values, number
from .development import Developed, linear_luminance, srgb_encode
from .domain import CHANNELS, Box, InputError, choice, finite, integer, object_value, reject_unknown

COMPARISON_MODES = ("wipe", "checker", "absolute", "signed-luma", "chroma")
COMPARISON_SPACES = ("linear", "display")
TILE_EDGE = 48


@dataclass(frozen=True)
class CompareOptions:
    mode: str = "wipe"
    space: str = "linear"
    split: float = 0.5
    dx: int = 0
    dy: int = 0
    threshold: float = 0.05
    gain: float = 5.0

    @classmethod
    def read(cls, value) -> CompareOptions:
        obj = object_value(value if value is not None else {}, "Comparison options")
        reject_unknown(obj, ("mode", "space", "split", "dx", "dy", "threshold", "gain"), "comparison")
        return cls(
            choice(obj.get("mode", "wipe"), "Comparison mode", COMPARISON_MODES),
            choice(obj.get("space", "linear"), "Comparison space", COMPARISON_SPACES),
            finite(obj.get("split", 0.5), "Wipe position", 0, 1),
            integer(obj.get("dx", 0), "Reference horizontal shift", -4096, 4096),
            integer(obj.get("dy", 0), "Reference vertical shift", -4096, 4096),
            finite(obj.get("threshold", 0.05), "Difference threshold", 0, 2),
            finite(obj.get("gain", 5), "Difference gain", 0.25, 50),
        )

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "space": self.space,
            "split": self.split,
            "dx": self.dx,
            "dy": self.dy,
            "threshold": self.threshold,
            "gain": self.gain,
        }


@dataclass
class Comparison:
    view: np.ndarray
    a: np.ndarray
    b: np.ndarray
    difference: np.ndarray
    shared_sensor: Box
    local_a: Box
    local_b: Box
    report: dict


def _world(result: Developed, off_x: int = 0, off_y: int = 0) -> Box:
    origin = result.frame.sensor.layout
    return result.box.translated(origin.origin_x + off_x, origin.origin_y + off_y)


def _overlap(
    a: Developed, b: Developed, options: CompareOptions, selected: Box | None
) -> tuple[Box, Box, Box]:
    aa = _world(a)
    bb = _world(b, options.dx, options.dy)
    shared = aa.intersection(bb)
    if shared is None:
        raise InputError("The developments have no shared sensor coordinates.")
    if selected is not None:
        origin = a.frame.sensor.layout
        selected = selected.translated(origin.origin_x, origin.origin_y)
        shared = shared.intersection(selected)
        if shared is None:
            raise InputError("The selected region does not intersect the comparison.")
    return shared, shared.translated(-aa.x, -aa.y), shared.translated(-bb.x, -bb.y)


def _signed_colors(difference: np.ndarray, gain: float) -> np.ndarray:
    signed = np.clip(difference * gain, -1, 1)
    strength = np.abs(signed)
    neutral = np.array([0.075, 0.10, 0.12], np.float32)
    up = np.array([1.0, 0.48, 0.18], np.float32)
    down = np.array([0.16, 0.67, 0.94], np.float32)
    endpoint = np.where((signed >= 0)[..., None], up, down)
    return neutral + (endpoint - neutral) * strength[..., None]


def _chroma_distance(delta: np.ndarray) -> np.ndarray:
    rg = delta[..., 0] - delta[..., 1]
    bg = delta[..., 2] - delta[..., 1]
    return np.sqrt((rg * rg + bg * bg) * 0.5)


def _thermal(magnitude: np.ndarray, gain: float) -> np.ndarray:
    scale = np.clip(magnitude * gain, 0, 1)
    positions = np.array([0, 0.30, 0.65, 1])
    anchors = ((0.05, 0.08, 0.13), (0.16, 0.35, 0.61), (0.65, 0.25, 0.50), (1, 0.86, 0.29))
    mapper = [np.interp(scale, positions, values) for values in zip(*anchors)]
    return np.stack(mapper, axis=-1).astype(np.float32)


def _side_by_side(a: np.ndarray, b: np.ndarray, options: CompareOptions) -> np.ndarray:
    height, width = a.shape[:2]
    if options.mode == "wipe":
        split = round(width * options.split)
        mixed = a.copy()
        mixed[:, split:] = b[:, split:]
        return mixed
    cell = max(8, min(width, height) // 8)
    mask = (np.arange(height)[:, None] // cell + np.arange(width)[None, :] // cell) % 2 == 0
    return np.where(mask[..., None], a, b)


def _render(
    a: np.ndarray, b: np.ndarray, delta: np.ndarray, options: CompareOptions
) -> np.ndarray:
    if options.mode in ("wipe", "checker"):
        return _side_by_side(a, b, options)
    if options.mode == "signed-luma":
        return _signed_colors(linear_luminance(delta), options.gain)
    if options.mode == "chroma":
        return _thermal(_chroma_distance(delta), options.gain)
    return _thermal(np.max(np.abs(delta), axis=2), options.gain)


def _tile_errors(magnitude: np.ndarray, shared: Box, threshold: float) -> list[dict]:
    height, width = magnitude.shape
    bins = []
    for top in range(0, height, TILE_EDGE):
        for left in range(0, width, TILE_EDGE):
            values = magnitude[top:top + TILE_EDGE, left:left + TILE_EDGE]
            tile_height, tile_width = values.shape
            fraction = np.count_nonzero(values > threshold) / values.size
            bins.append({
                "sensor_box": Box(shared.x + left, shared.y + top, tile_width, tile_height).as_dict(),
                "p95": number(np.percentile(values, 95)),
                "mean": number(values.mean(dtype=np.float64)),
                "max": number(values.max()),
                "over_threshold": number(fraction),
            })
    return sorted(bins, key=lambda item: (-item["p95"], -item["mean"]))[:12]


def _edge_difference(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    for frame in (left, right):
        luminance = linear_luminance(frame)
        horizontal = ndimage.sobel(luminance, axis=1, mode="reflect") * 0.125
        vertical = ndimage.sobel(luminance, axis=0, mode="reflect") * 0.125
        features.append(np.hypot(horizontal, vertical))
    return features[0] - features[1], np.maximum(features[0], features[1])


def _compare_report(
    a: Developed, b: Developed, av: np.ndarray, bv: np.ndarray,
    delta: np.ndarray, shared: Box, local_a: Box, local_b: Box, options: CompareOptions,
) -> dict:
    magnitude = np.max(np.abs(delta), axis=2)
    signed = linear_luminance(delta)
    edge_delta, edges = _edge_difference(av, bv)
    edge_threshold = max(0.01, float(np.percentile(edges, 90)))
    is_edge = edges >= edge_threshold
    per_channel = []
    for index, label in enumerate(CHANNELS):
        channel = delta[..., index]
        per_channel.append({
            "channel": label,
            "mean_error": number(channel.mean(dtype=np.float64)),
            "mean_absolute_error": number(np.abs(channel).mean(dtype=np.float64)),
            "rmse": number(np.sqrt(np.mean(channel * channel, dtype=np.float64))),
        })
    notices = []
    if a.frame.sensor.model and b.frame.sensor.model and a.frame.sensor.model != b.frame.sensor.model:
        notices.append("Different camera models are compared by declared origins only.")
    if a.frame.sensor.layout.pattern != b.frame.sensor.layout.pattern:
        notices.append("Compared frames declare different Bayer patterns.")
    if a.frame.id != b.frame.id:
        notices.append("Movement between captures can appear as a difference.")
    head = {
        "left": a.frame.id,
        "right": b.frame.id,
        "shared_sensor": shared.as_dict(),
        "left_raster": local_a.translated(a.box.x, a.box.y).as_dict(),
        "right_raster": local_b.translated(b.box.x, b.box.y).as_dict(),
        "compared_pixels": shared.area,
        "over_threshold": int(np.count_nonzero(magnitude > options.threshold)),
        "threshold": options.threshold,
        "space": options.space,
        "channels": per_channel,
        "delta_luma": describe_values(signed),
        "absolute_max_by_pixel": describe_values(magnitude),
        "chroma_delta": describe_values(_chroma_distance(delta)),
        "strong_edges": int(np.count_nonzero(is_edge)),
        "mean_edge_magnitude_delta": number(np.abs(edge_delta[is_edge]).mean(dtype=np.float64)) if is_edge.any() else None,
        "largest_tiles": _tile_errors(magnitude, shared, options.threshold),
        "notices": notices,
    }
    return head


def compare(
    a: Developed, b: Developed, options: CompareOptions, selected: Box | None = None
) -> Comparison:
    shared, local_a, local_b = _overlap(a, b, options, selected)
    if options.space == "linear":
        av = local_a.crop(a.working_rgb)
        bv = local_b.crop(b.working_rgb)
    else:
        av = local_a.crop(a.encoded)
        bv = local_b.crop(b.encoded)
    delta = av - bv
    display_a = local_a.crop(a.encoded)
    display_b = local_b.crop(b.encoded)
    view = _render(display_a, display_b, delta, options)
    report = _compare_report(a, b, av, bv, delta, shared, local_a, local_b, options)
    return Comparison(view, display_a, display_b, delta, shared, local_a, local_b, report)


def comparison_probe(result: Comparison, sensor_x: int, sensor_y: int) -> dict:
    if not result.shared_sensor.contains(sensor_x, sensor_y):
        raise InputError("The comparison probe falls outside the common sensor area.")
    x = sensor_x - result.shared_sensor.x
    y = sensor_y - result.shared_sensor.y
    return {
        "sensor_x": sensor_x,
        "sensor_y": sensor_y,
        "left_encoded": np.round(result.a[y, x], 6).tolist(),
        "right_encoded": np.round(result.b[y, x], 6).tolist(),
        "signed_delta": np.round(result.difference[y, x], 6).tolist(),
    }
