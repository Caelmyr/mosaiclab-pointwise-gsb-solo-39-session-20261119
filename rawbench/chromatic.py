"""Phase-resolved measurement of color edge displacement in native RAW samples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .analysis import number
from .cfa import Plane, phase_planes
from .domain import Box, Frame, InputError
from .optics import EDGE_MAX_WIDTH, EdgeSettings, _fit, _orient

MIN_SIDE = 36
MIN_CONTRAST = 0.035
MAX_GREEN_GAP = 0.8


@dataclass(frozen=True)
class Geometry:
    direction: str
    slope: float
    across_at_center: float
    center_along: float
    sign: int
    green_fits: tuple[dict, ...]

    def distance(self, across: np.ndarray, along: np.ndarray) -> np.ndarray:
        baseline = self.across_at_center + (along - self.center_along) * self.slope
        return self.sign * (across - baseline) / np.hypot(1, self.slope)

    def reference_point(self, box: Box, frame: Frame) -> dict:
        if self.direction == "vertical":
            x, y = self.across_at_center, self.center_along
        else:
            x, y = self.center_along, self.across_at_center
        x += box.x
        y += box.y
        return {
            "raster_x": number(x), "raster_y": number(y),
            "sensor_x": number(x + frame.sensor.layout.origin_x),
            "sensor_y": number(y + frame.sensor.layout.origin_y),
        }


@dataclass(frozen=True)
class Profile:
    name: str
    response: np.ndarray
    counts: np.ndarray
    low: float
    high: float
    excluded: int
    crossing: float
    transition_width: float
    spread: float | None

    def evidence(self, green_midpoint: float) -> dict:
        return {
            "phase": self.name,
            "offset_px": number(self.crossing - green_midpoint),
            "transition_width_px": number(self.transition_width),
            "strip_offset_range_px": number(self.spread) if self.spread is not None else None,
            "median_dark": number(self.low),
            "median_bright": number(self.high),
            "samples": int(self.counts.sum()),
            "excluded_saturated_sites": self.excluded,
            "response": np.round(self.response, 6).tolist(),
            "site_counts": self.counts.tolist(),
        }


def _signal(plane: Plane, frame: Frame) -> np.ndarray:
    black = frame.sensor.black[plane.phase]
    white = frame.sensor.white[plane.phase]
    return (plane.samples.astype(np.float32) - black) / (white - black)


def _axes(plane: Plane, vertical: bool) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(plane.samples.shape, dtype=np.float32)
    raster_y = plane.offset_y + y * 2
    raster_x = plane.offset_x + x * 2
    return (raster_x, raster_y) if vertical else (raster_y, raster_x)


def _geometry(frame: Frame, box: Box, greens: list[Plane], orientation: str) -> Geometry:
    first = _signal(greens[0], frame)
    _, direction = _orient(first, orientation)
    vertical = direction == "vertical"
    center = ((box.height if vertical else box.width) - 1) / 2
    fits = []
    for plane in greens:
        signal = _signal(plane, frame)
        fit = _fit(signal if vertical else signal.T.copy(), direction)
        across_origin = plane.offset_x if vertical else plane.offset_y
        along_origin = plane.offset_y if vertical else plane.offset_x
        across = across_origin + fit.slope * (center - along_origin) + 2 * fit.intercept
        fits.append((fit, across))
    left, right = fits
    max_lever = center * abs(left[0].slope - right[0].slope)
    if left[0].sign != right[0].sign or abs(left[1] - right[1]) + max_lever > 1.5:
        raise InputError("The two green lattices disagree about the edge. Select one neutral transition.")
    slope = (left[0].slope + right[0].slope) / 2
    across = (left[1] + right[1]) / 2
    summary = tuple(
        {"phase": plane.name, "fit_lines": fit.used, "fit_residual_px": number(2 * fit.residual),
         "center_crossing_px": number(crossing)}
        for plane, (fit, crossing) in zip(greens, fits)
    )
    return Geometry(direction, slope, across, center, left[0].sign, summary)


def _normalize(values: np.ndarray, positions: np.ndarray, distance: np.ndarray, label: str):
    radius = abs(positions[0])
    left = values[distance < -radius * 0.58]
    right = values[distance > radius * 0.58]
    if min(left.size, right.size) < 16:
        raise InputError(f"{label} lacks unclipped samples on both sides of the measured edge.")
    low, high = np.median(left), np.median(right)
    if high - low < MIN_CONTRAST:
        raise InputError(f"{label} has too little positive contrast across the selected edge.")
    return (values - low) / (high - low), float(low), float(high)


def _aggregate(distance: np.ndarray, response: np.ndarray, positions: np.ndarray):
    step = positions[1] - positions[0]
    index = np.floor((distance - positions[0]) / step + 0.5).astype(np.intp)
    inside = (index >= 0) & (index < positions.size)
    count = np.bincount(index[inside], minlength=positions.size)
    total = np.bincount(index[inside], weights=response[inside], minlength=positions.size)
    good = count >= 2
    if np.count_nonzero(good) < positions.size * 0.60:
        raise InputError("The Bayer plane does not sample enough of the edge. Use a wider slanted target region.")
    bound = positions[-1] * 0.62
    if positions[good][0] > -bound or positions[good][-1] < bound:
        raise InputError("The region cuts off a side of the edge profile.")
    interior = int(np.diff(np.flatnonzero(good)).max()) - 1
    if interior > max(2, int(1.6 / step)):
        raise InputError("The edge angle leaves wide gaps between measured samples.")
    filled = np.interp(positions, positions[good], total[good] / count[good])
    return filled, count


def _find_crossing(profile: np.ndarray, positions: np.ndarray, level: float, center: float = 0.0) -> float:
    crossings = np.flatnonzero((profile[:-1] < level) & (profile[1:] >= level))
    if not crossings.size:
        raise InputError("The phase response does not cross both sides of the edge.")
    index = crossings[np.argmin(np.abs(positions[crossings] - center))]
    left, right = profile[index:index + 2]
    fraction = (level - left) / (right - left)
    return float(positions[index] + fraction * (positions[index + 1] - positions[index]))


def _strip_spread(distance, signal, along, positions, center) -> float | None:
    if positions[1] - positions[0] < 0.4:
        positions = (positions[:-1:2] + positions[1::2]) / 2
    ends = np.quantile(along, [1 / 3, 2 / 3])
    section = np.searchsorted(ends, along, side="right")
    results = []
    for stripe in range(3):
        take = section == stripe
        if np.count_nonzero(take) < 40:
            continue
        try:
            profile, _ = _aggregate(distance[take], signal[take], positions)
            profile = ndimage.gaussian_filter1d(profile, 0.65)
            results.append(_find_crossing(profile, positions, 0.5, center))
        except InputError:
            continue
    return max(results) - min(results) if len(results) >= 2 else None


def _profile(plane: Plane, frame: Frame, geometry: Geometry, positions: np.ndarray) -> Profile:
    across, along = _axes(plane, geometry.direction == "vertical")
    distance = geometry.distance(across, along)
    values = _signal(plane, frame)
    radius = positions[-1] + (positions[1] - positions[0]) / 2
    window = (distance >= -radius) & (distance < radius)
    saturated = values >= 0.985
    accepted = window & ~saturated
    excluded = int(np.count_nonzero(window & saturated))
    if not np.any(accepted) or excluded > np.count_nonzero(window) * 0.20:
        raise InputError(f"{plane.name} clips on too many sites. Choose an exposure below saturation.")
    used = distance[accepted]
    signal, low, high = _normalize(values[accepted], positions, used, plane.name)
    profile, count = _aggregate(used, signal, positions)
    filtered = ndimage.gaussian_filter1d(profile, 0.65)
    crossing = _find_crossing(filtered, positions, 0.5)
    foot = _find_crossing(filtered, positions, 0.1, crossing)
    crest = _find_crossing(filtered, positions, 0.9, crossing)
    if not foot < crossing < crest or abs(crossing) > radius * 0.6:
        raise InputError(f"{plane.name} does not show a clear, centered rising edge.")
    spread = _strip_spread(used, signal, along[accepted], positions, crossing)
    return Profile(plane.name, filtered, count, low, high, excluded, crossing, crest - foot, spread)


def chromatic_report(frame: Frame, region: Box, settings: EdgeSettings) -> dict:
    selected = frame.sensor.active.intersection(region)
    if selected is None or min(selected.width, selected.height) < MIN_SIDE:
        raise InputError("Select at least 36 × 36 active sites around one slightly tilted neutral edge.")
    if max(selected.width, selected.height) > EDGE_MAX_WIDTH:
        raise InputError("Select only the edge patch, at most 400 sites on either side.")
    if frame.role != "scene":
        raise InputError("Chromatic edge displacement is measured on scene captures.")
    planes = phase_planes(frame.raster.values, frame.sensor.layout, selected)
    greens = [plane for plane in planes if plane.color == "G"]
    geometry = _geometry(frame, selected, greens, settings.orientation)
    sampling = min(4, settings.oversampling)
    radius = settings.half_width
    positions = (np.arange(radius * 2 * sampling) + 0.5) / sampling - radius
    profiles = [_profile(plane, frame, geometry, positions) for plane in planes]
    green_crossings = [profile.crossing for profile in profiles if profile.name.startswith("G")]
    green_distance = abs(green_crossings[0] - green_crossings[1])
    if green_distance > MAX_GREEN_GAP:
        raise InputError("Green profiles disagree by more than 0.8 pixels. Try a cleaner, more neutral edge.")
    green_midpoint = sum(green_crossings) / 2
    phase_evidence = [profile.evidence(green_midpoint) for profile in profiles]
    return {
        "region": selected.as_dict(),
        "geometry": {
            "direction": geometry.direction, "reference_point": geometry.reference_point(selected, frame),
            "angle_from_axis": number(np.degrees(np.arctan(geometry.slope))),
            "dark_to_light_sign": geometry.sign, "green_fits": list(geometry.green_fits),
        },
        "offset_unit": "native sensor pixels, perpendicular to edge, positive toward the bright side",
        "green_separation_px": number(green_distance),
        "positions_px": np.round(positions - green_midpoint, 5).tolist(),
        "phases": phase_evidence,
        "method": "Native Bayer profiles projected onto a shared green-edge normal; 50% transitions relative to green midpoint.",
        "warning": "Use an achromatic target. This local estimate includes lens, sensor and target effects; it is not a lens calibration.",
    }
