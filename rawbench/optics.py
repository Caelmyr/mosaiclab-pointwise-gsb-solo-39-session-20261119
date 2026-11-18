"""Measurements on flat fields and on developed slanted-edge targets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .analysis import number
from .calibration import code_offsets, normalized_source
from .cfa import phase_planes
from .development import Developed, linear_luminance
from .domain import Box, Frame, InputError, choice, integer, object_value, reject_unknown

UNIFORMITY_TILES = 5
EDGE_MAX_WIDTH = 400
EDGE_ORIENTATIONS = ("automatic", "vertical", "horizontal")


@dataclass(frozen=True)
class EdgeSettings:
    orientation: str = "automatic"
    oversampling: int = 4
    half_width: int = 12

    @classmethod
    def read(cls, value) -> EdgeSettings:
        obj = object_value(value if value is not None else {}, "Edge options")
        reject_unknown(obj, ("orientation", "oversampling", "half_width"), "edge options")
        return cls(
            choice(obj.get("orientation", "automatic"), "Edge orientation", EDGE_ORIENTATIONS),
            integer(obj.get("oversampling", 4), "Sampling factor", 2, 8),
            integer(obj.get("half_width", 12), "Edge measurement half-width", 5, 20),
        )

    def as_dict(self) -> dict:
        return {
            "orientation": self.orientation,
            "oversampling": self.oversampling,
            "half_width": self.half_width,
        }


@dataclass(frozen=True)
class EdgeFit:
    slope: float
    intercept: float
    direction: str
    sign: int
    used: int
    residual: float
    contrast: float


def _phase_tile(plane: np.ndarray, accepted: np.ndarray) -> tuple[float | None, int, float | None]:
    sample = plane[accepted]
    if not sample.size:
        return None, 0, None
    center = np.median(sample)
    return number(center), int(sample.size), number(np.median(np.abs(sample - center)) * 1.4826)


def _tile_shape(width: int, parts: int) -> np.ndarray:
    if width < parts:
        return np.arange(0, width + 1)
    return np.linspace(0, width, parts + 1, dtype=int)


def _flat_grid(
    signal: np.ndarray, accepted: np.ndarray, frame: Frame, region: Box, tiles: int
) -> list[dict]:
    rows = _tile_shape(region.height, tiles)
    columns = _tile_shape(region.width, tiles)
    result = []
    for iy, (top, bottom) in enumerate(zip(rows[:-1], rows[1:])):
        for ix, (left, right) in enumerate(zip(columns[:-1], columns[1:])):
            box = Box(region.x + int(left), region.y + int(top), int(right - left), int(bottom - top))
            phase_data = phase_planes(signal, frame.sensor.layout, box)
            phase_mask = phase_planes(accepted, frame.sensor.layout, box)
            data = {}
            for values, valid in zip(phase_data, phase_mask):
                median, count, sigma = _phase_tile(values.samples, valid.samples)
                data[values.name] = {"median_code": median, "sites": count, "sigma_code": sigma}
            result.append({"row": iy, "column": ix, "box": box.as_dict(), "phases": data})
    return result


def _uniformity_summary(
    signal: np.ndarray, accepted: np.ndarray, frame: Frame, region: Box, grid: list[dict]
) -> list[dict]:
    planes = phase_planes(signal, frame.sensor.layout, region)
    masks = phase_planes(accepted, frame.sensor.layout, region)
    result = []
    for plane, mask in zip(planes, masks):
        clean = plane.samples[mask.samples]
        entry = {
            "phase": plane.name,
            "valid": int(clean.size),
            "rejected": plane.site_count - int(clean.size),
        }
        medians = [tile["phases"][plane.name]["median_code"] for tile in grid]
        medians = [value for value in medians if value is not None]
        if medians and clean.size:
            reference = float(np.median(clean))
            lo = min(medians)
            hi = max(medians)
            entry["median_code"] = number(reference)
            entry["low_tile_ratio"] = number(lo / reference)
            entry["high_tile_ratio"] = number(hi / reference)
            entry["tile_variation"] = number((hi - lo) / reference)
            good = np.percentile(clean, [2, 98])
            entry["p2_ratio"] = number(good[0] / reference)
            entry["p98_ratio"] = number(good[1] / reference)
        else:
            entry.update({
                "median_code": None,
                "low_tile_ratio": None,
                "high_tile_ratio": None,
                "tile_variation": None,
                "p2_ratio": None,
                "p98_ratio": None,
            })
        result.append(entry)
    return result


def uniformity_report(frame: Frame, region: Box, tiles: int = UNIFORMITY_TILES) -> dict:
    selected = frame.sensor.active.intersection(region)
    if selected is None:
        raise InputError("Uniformity requires a region inside the active sensor raster.")
    signal, _ = code_offsets(frame)
    fraction = normalized_source(frame)
    accepted = (fraction > 0.002) & (fraction < 0.98)
    grid = _flat_grid(signal, accepted, frame, selected, tiles)
    phases = _uniformity_summary(signal, accepted, frame, selected, grid)
    return {
        "region": selected.as_dict(),
        "grid_rows": min(tiles, selected.height),
        "grid_columns": min(tiles, selected.width),
        "per_phase": phases,
        "tiles": grid,
        "normalized_bounds": [0.002, 0.98],
        "meaning": "Relative response by physical Bayer phase. A nonuniform target biases the response.",
    }


def _smooth_columns(image: np.ndarray) -> np.ndarray:
    return ndimage.gaussian_filter1d(image, 0.7, axis=1, mode="nearest")


def _axis_sign(gradient: np.ndarray) -> float:
    weight = np.percentile(np.abs(gradient), 88)
    if weight == 0:
        return 0.0
    significant = gradient[np.abs(gradient) >= weight]
    return float(np.median(significant))


def _orient(image: np.ndarray, option: str) -> tuple[np.ndarray, str]:
    if option == "vertical":
        return image, option
    if option == "horizontal":
        return image.T.copy(), option
    across = np.diff(ndimage.gaussian_filter(image, 0.65), axis=1)
    down = np.diff(ndimage.gaussian_filter(image, 0.65), axis=0)
    vertical = np.abs(np.percentile(across, [1.5, 98.5])).max()
    horizontal = np.abs(np.percentile(down, [1.5, 98.5])).max()
    if vertical >= horizontal:
        return image, "vertical"
    return image.T.copy(), "horizontal"


def _subpixel_peak(row: np.ndarray, index: int) -> float:
    left = row[index - 1]
    center = row[index]
    right = row[index + 1]
    shape = (left - 2 * center + right)
    x = 0.0 if abs(shape) < 1e-8 else 0.5 * (left - right) / shape
    return index + float(np.clip(x, -0.5, 0.5)) + 0.5


def _fit(image: np.ndarray, direction: str) -> EdgeFit:
    height, width = image.shape
    gradient = np.diff(_smooth_columns(image), axis=1)
    edge = _axis_sign(gradient)
    if abs(edge) < 0.00001:
        raise InputError("The region has insufficient contrast for an edge measurement.")
    sign = 1 if edge >= 0 else -1
    gradient *= sign
    inset = max(3, width // 9)
    useful = gradient[:, inset:-inset]
    base_peak = np.percentile(np.max(useful, axis=1), 50)
    if base_peak <= 0.0001:
        raise InputError("Center the region over a single high-contrast edge.")
    offsets = np.argmax(useful, axis=1) + inset
    below = []
    edge_rows = []
    edge_cols = []
    for row_idx, col in enumerate(offsets):
        if gradient[row_idx, col] < base_peak * 0.35:
            continue
        left = np.median(image[row_idx, max(0, col - 7):max(1, col - 3)])
        right = np.median(image[row_idx, col + 4:col + 8])
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        strength = (right - left) * sign
        if strength < 0.015:
            continue
        edge_rows.append(row_idx)
        edge_cols.append(_subpixel_peak(gradient[row_idx], col))
        below.append(strength)
    if len(edge_rows) < max(12, height // 3):
        raise InputError("Too few rows contain the same clearly slanted edge.")
    x = np.array(edge_rows, dtype=np.float64)
    y = np.array(edge_cols, dtype=np.float64)
    for _ in range(3):
        slope, intercept = np.polyfit(x, y, 1)
        dist = np.abs(y - (slope * x + intercept))
        scale = np.median(dist)
        use = dist <= max(0.35, scale * 3.5)
        if use.all():
            break
        x = x[use]
        y = y[use]
        if len(y) < 12:
            raise InputError("Selected edge has too much curvature or multiple edges.")
    residual = np.abs(y - (x * slope + intercept))
    movement = abs(slope) * (height - 1)
    if movement < 1.8 or abs(slope) > 0.55:
        raise InputError("Use an edge that tilts by several pixels, but less than 30° from its axis.")
    if np.percentile(residual, 90) > 0.75:
        raise InputError("The selected edge is too curved or contains several edges.")
    return EdgeFit(
        float(slope), float(intercept), direction, sign, len(x),
        float(np.median(residual)), float(np.median(below)),
    )


def _edge_profile(image: np.ndarray, fit: EdgeFit, settings: EdgeSettings) -> tuple:
    height, width = image.shape
    radius = settings.half_width
    step = 1 / settings.oversampling
    ys, xs = np.mgrid[0:height, 0:width]
    distances = (xs - (fit.slope * ys + fit.intercept)) / np.hypot(1, fit.slope) * fit.sign
    selected = (distances >= -radius) & (distances < radius)
    bins = int(2 * radius / step)
    assignments = np.floor((distances[selected] + radius) / step).astype(int)
    count = np.bincount(assignments, minlength=bins)[:bins]
    total = np.bincount(assignments, weights=image[selected], minlength=bins)[:bins]
    good = count > 0
    if np.count_nonzero(good) < bins * 0.7:
        raise InputError("Move or widen the region to cover both sides of the edge.")
    centers = -radius + (np.arange(bins) + 0.5) * step
    samples = np.interp(centers, centers[good], total[good] / count[good])
    stem = max(2, settings.oversampling * 2)
    low = np.median(samples[:stem])
    high = np.median(samples[-stem:])
    if high - low < 0.015:
        raise InputError("The region's bright and dark sides are too similar.")
    esf = (samples - low) / (high - low)
    return centers, esf, count, step


def _frequency_response(esf: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.diff(esf) / step
    signal *= np.hanning(signal.size)
    width = 1 << max(9, (len(signal) - 1).bit_length() + 2)
    amplitude = np.abs(np.fft.rfft(signal, n=width))
    if amplitude[0] <= 1e-9:
        raise InputError("Unable to normalize the edge frequency response.")
    frequencies = np.fft.rfftfreq(width, step)
    mtf = amplitude / amplitude[0]
    baseband = frequencies <= 0.5
    return frequencies[baseband], mtf[baseband], signal


def _frequency_at(frequencies: np.ndarray, values: np.ndarray, target: float) -> float | None:
    indices = np.flatnonzero(values < target)
    if not len(indices):
        return None
    right = indices[0]
    if right == 0:
        return number(frequencies[right])
    left = right - 1
    weight = (values[left] - target) / (values[left] - values[right])
    return number(frequencies[left] + weight * (frequencies[right] - frequencies[left]))


def edge_report(
    result: Developed, region: Box, settings: EdgeSettings
) -> dict:
    selected = result.box.intersection(region)
    if selected is None:
        raise InputError("The measured edge must intersect the developed raster.")
    if min(selected.width, selected.height) < 20:
        raise InputError("The measured edge needs at least 20 rows and 20 columns.")
    if max(selected.width, selected.height) > EDGE_MAX_WIDTH:
        raise InputError("Choose a smaller region around one edge (at most 400 pixels per side).")
    box = selected.translated(-result.box.x, -result.box.y)
    image = linear_luminance(box.crop(result.working_rgb))
    image, direction = _orient(image, settings.orientation)
    fit = _fit(image, direction)
    positions, esf, count, step = _edge_profile(image, fit, settings)
    frequency, amplitude, lsf = _frequency_response(esf, step)
    return {
        "region": selected.as_dict(),
        "direction": direction,
        "edge_angle_deg_from_axis": number(np.degrees(np.arctan(fit.slope))),
        "fit_rows": fit.used,
        "fit_residual_px": number(fit.residual),
        "linear_edge_contrast": number(fit.contrast),
        "mtf50": _frequency_at(frequency, amplitude, 0.5),
        "mtf10": _frequency_at(frequency, amplitude, 0.1),
        "mtf_nyquist": number(amplitude[-1]),
        "frequency_unit": "cycles/pixel",
        "edge_positions": np.round(positions, 5).tolist(),
        "edge_spread": np.round(esf, 6).tolist(),
        "line_spread": np.round(lsf, 6).tolist(),
        "site_counts": count.tolist(),
        "frequency": np.round(frequency, 8).tolist(),
        "mtf": np.round(amplitude, 6).tolist(),
        "method": "Linear luminance slanted-edge estimate with finite-difference and Hann window.",
        "warning": "Scene chart, spatial reconstruction and optics all affect this estimate.",
    }
