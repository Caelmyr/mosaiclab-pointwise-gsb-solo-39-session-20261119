"""Bayer sampling, phase-preserving data access, and interpolation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .domain import CHANNELS, Box, InputError, Layout

FILTERS = ("nearest", "bilinear", "directional")
BILINEAR = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
HORIZONTAL = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
VERTICAL = HORIZONTAL.T
DIAGONAL = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float32)


@dataclass(frozen=True)
class Plane:
    phase: int
    color: str
    name: str
    offset_y: int
    offset_x: int
    samples: np.ndarray

    @property
    def site_count(self) -> int:
        return self.samples.size


@dataclass
class Reconstruction:
    rgb: np.ndarray
    missing_samples: dict[str, int]
    interpolated_samples: dict[str, int]


def phase_planes(
    values: np.ndarray, layout: Layout, box: Box | None = None
) -> tuple[Plane, ...]:
    if box is not None:
        values = box.crop(values)
        layout = layout.shifted(box.x, box.y)
    planes = []
    for phase, color in enumerate(layout.colors):
        row, col = layout.local_offset(phase)
        planes.append(
            Plane(phase, color, layout.phase_name(phase), row, col, values[row::2, col::2])
        )
    return tuple(planes)


def phase_field(
    shape: tuple[int, int], layout: Layout, phase_values: tuple[float, ...]
) -> np.ndarray:
    values = np.asarray(phase_values, dtype=np.float32)
    field = np.empty(shape, dtype=np.float32)
    for phase, value in enumerate(values):
        field[layout.phase_slice(phase)] = value
    return field


def color_mask(shape: tuple[int, int], layout: Layout, color: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for phase, site_color in enumerate(layout.colors):
        if color == site_color:
            mask[layout.phase_slice(phase)] = True
    return mask


def phase_codes(shape: tuple[int, int], layout: Layout) -> np.ndarray:
    tile = layout.local_index_grid().astype(np.uint8)
    height, width = shape
    return np.tile(tile, ((height + 1) // 2, (width + 1) // 2))[:height, :width]


def sample_scene(rgb: np.ndarray, layout: Layout) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != len(CHANNELS):
        raise InputError("Sensor sampling requires a three-channel linear scene.")
    mosaic = np.empty(rgb.shape[:2], dtype=np.float32)
    for phase, color in enumerate(layout.colors):
        section = layout.phase_slice(phase)
        mosaic[section] = rgb[:, :, CHANNELS.index(color)][section]
    return mosaic


def neighborhood(
    values: np.ndarray, layout: Layout, x: int, y: int, radius: int = 2
) -> dict:
    height, width = values.shape
    if not 0 <= x < width or not 0 <= y < height:
        raise InputError("Probe coordinate is outside the raster.")
    row_start = max(0, y - radius)
    row_stop = min(height, y + radius + 1)
    column_start = max(0, x - radius)
    column_stop = min(width, x + radius + 1)
    rows = []
    for row_index in range(row_start, row_stop):
        row = []
        for col_index in range(column_start, column_stop):
            p = layout.index(col_index, row_index)
            row.append({
                "x": col_index,
                "y": row_index,
                "sensor_x": col_index + layout.origin_x,
                "sensor_y": row_index + layout.origin_y,
                "phase": layout.phase_name(p),
                "value": values[row_index, col_index].item(),
                "selected": row_index == y and col_index == x,
            })
        rows.append(row)
    return {"rows": rows, "radius": radius, "center": {"x": x, "y": y}}


def _nearest(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.all():
        return values.copy()
    if not mask.any():
        return np.zeros_like(values, dtype=np.float32)
    indices = ndimage.distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    return values[tuple(indices)]


def _filtered(
    values: np.ndarray, support: np.ndarray, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    numerator = ndimage.convolve(np.where(support, values, 0), kernel, mode="constant", cval=0)
    denominator = ndimage.convolve(support.astype(np.float32), kernel, mode="constant", cval=0)
    valid = denominator > 0
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=valid
    ), valid


def _fill_missing(
    estimate: np.ndarray, has_support: np.ndarray, values: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    if has_support.all():
        return estimate
    return np.where(has_support, estimate, _nearest(values, samples))


def _bilinear(values: np.ndarray, samples: np.ndarray) -> np.ndarray:
    estimate, support = _filtered(values, samples, BILINEAR)
    estimate = _fill_missing(estimate, support, values, samples)
    return np.where(samples, values, estimate)


def _direction_gradient(
    mosaic: np.ndarray, two_step_axis: int
) -> np.ndarray:
    kernel = np.array([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    second = np.array([0.5, 0.0, -1.0, 0.0, 0.5], dtype=np.float32)
    first_delta = ndimage.correlate1d(mosaic, kernel, axis=two_step_axis, mode="mirror")
    curvature = ndimage.correlate1d(mosaic, second, axis=two_step_axis, mode="mirror")
    return np.abs(first_delta) + np.abs(curvature)


def _green_directional(
    values: np.ndarray, samples: np.ndarray, mosaic_for_gradient: np.ndarray
) -> np.ndarray:
    horizontal, h_valid = _filtered(values, samples, HORIZONTAL)
    vertical, v_valid = _filtered(values, samples, VERTICAL)
    rough_x = _direction_gradient(mosaic_for_gradient, 1)
    rough_y = _direction_gradient(mosaic_for_gradient, 0)
    w_x = np.divide(
        h_valid.astype(np.float32), rough_x + 0.002,
    )
    w_y = np.divide(
        v_valid.astype(np.float32), rough_y + 0.002,
    )
    total = w_x + w_y
    estimate = np.divide(
        horizontal * w_x + vertical * w_y, total,
        out=np.zeros_like(values), where=total > 0,
    )
    estimate = _fill_missing(estimate, total > 0, values, samples)
    return np.where(samples, values, estimate)


def _color_difference(
    values: np.ndarray, samples: np.ndarray, green: np.ndarray,
    other_color: np.ndarray,
) -> np.ndarray:
    delta = values - green
    near, near_exists = _filtered(delta, samples, BILINEAR)
    diagonal, diagonal_exists = _filtered(delta, samples, DIAGONAL)
    use_diagonal = other_color & diagonal_exists
    interpolated = np.where(use_diagonal, diagonal, near)
    exists = near_exists | use_diagonal
    interpolated = _fill_missing(interpolated, exists, delta, samples)
    return np.where(samples, values, green + interpolated)


def reconstruct(
    mosaic: np.ndarray, layout: Layout, method: str, *, valid: np.ndarray | None = None
) -> Reconstruction:
    if method not in FILTERS:
        raise InputError("Unknown interpolation method.")
    if mosaic.ndim != 2:
        raise InputError("Reconstruction requires a single-plane mosaic.")
    values = np.asarray(mosaic, dtype=np.float32)
    coverage = np.isfinite(values)
    if valid is not None:
        if valid.shape != values.shape:
            raise InputError("Validity mask does not match the mosaic dimensions.")
        coverage &= valid
    sites = {key: color_mask(values.shape, layout, key) for key in CHANNELS}
    usable = {key: coverage & loc for key, loc in sites.items()}
    if method == "directional" and usable["G"].any():
        surround = _fill_missing(values, coverage, values, coverage)
        green = _green_directional(values, usable["G"], surround)
        red = _color_difference(values, usable["R"], green, sites["B"])
        blue = _color_difference(values, usable["B"], green, sites["R"])
        # If a color is completely missing, avoid inventing it from green.
        if not usable["R"].any():
            red.fill(0)
        if not usable["B"].any():
            blue.fill(0)
        rgb = np.stack((red, green, blue), axis=-1)
    else:
        engine = _nearest if method == "nearest" else _bilinear
        rgb = np.stack([engine(values, usable[key]) for key in CHANNELS], axis=-1)
    missing = {
        chan: int(np.count_nonzero(sites[chan] & ~coverage))
        for chan in CHANNELS
    }
    interpolated = {
        chan: int(values.size - np.count_nonzero(usable[chan])) for chan in CHANNELS
    }
    return Reconstruction(rgb, missing, interpolated)


def sparse_false_color(
    mosaic: np.ndarray, layout: Layout, *, show_green_lattice: bool = False
) -> np.ndarray:
    values = np.clip(mosaic, 0, 1)
    output = np.zeros((*values.shape, 3), dtype=np.float32)
    for plane in phase_planes(values, layout):
        indices = layout.phase_slice(plane.phase)
        primary = CHANNELS.index(plane.color)
        output[indices + (primary,)] = plane.samples
        if show_green_lattice and plane.name == "G2":
            output[indices + (2,)] = plane.samples * 0.65
    return output


def separate_grid_preview(values: np.ndarray, layout: Layout) -> np.ndarray:
    height = values.shape[0] // 2
    width = values.shape[1] // 2
    result = np.zeros((height * 2 + 1, width * 2 + 1), dtype=np.float32)
    gap = 0.13
    result[height, :] = gap
    result[:, width] = gap
    for plane in phase_planes(values, layout):
        phase_y, phase_x = divmod(plane.phase, 2)
        top = phase_y * (height + 1)
        left = phase_x * (width + 1)
        result[top:top + height, left:left + width] = plane.samples[:height, :width]
    return result


def rotate_raw_clockwise(
    values: np.ndarray, layout: Layout, steps: int
) -> tuple[np.ndarray, Layout]:
    # The rotated top-left tile describes the new local Bayer phase.
    turns = steps % 4
    rotated = np.rot90(values, -turns)
    shape = values.shape
    index_grid = phase_codes(shape, layout)
    phases = np.rot90(index_grid, -turns)
    colors = "".join(layout.colors[idx] for idx in phases[:2, :2].flat)
    return rotated, Layout(colors, 0, 0)


def remap_raw_crop(
    values: np.ndarray, layout: Layout, box: Box
) -> tuple[np.ndarray, Layout]:
    if box.width < 2 or box.height < 2:
        raise InputError("A raw crop needs two columns and two rows for every phase.")
    return box.crop(values), layout.shifted(box.x, box.y)


def color_residuals(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise InputError("Chroma differences require three color planes.")
    return rgb[..., 0] - rgb[..., 1], rgb[..., 2] - rgb[..., 1]
