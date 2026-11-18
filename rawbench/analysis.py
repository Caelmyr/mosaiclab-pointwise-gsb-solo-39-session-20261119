"""Measurements made in sensor coordinates and original code space."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .calibration import Correction, normalized_source
from .cfa import phase_planes
from .domain import Box, Frame, InputError

PROFILE_AXES = ("rows", "columns")
HISTOGRAM_SPACES = ("linear", "stops")
STOPS_MINIMUM = -12.0
EXPOSURE_LEVELS = (0.05, 0.2, 0.6, 0.9, 0.98, 1.0)
PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
MIN_PROFILE_SITES = 6


def number(value, decimals: int = 6) -> float | None:
    if not np.isfinite(value):
        return None
    return round(float(value), decimals)


def describe_values(values: np.ndarray) -> dict:
    data = values[np.isfinite(values)]
    if data.size == 0:
        return {"count": 0}
    qs = np.percentile(data, PERCENTILES)
    median = qs[3]
    mad = np.median(np.abs(data - median))
    return {
        "count": int(data.size),
        "min": number(data.min()),
        "max": number(data.max()),
        "mean": number(data.mean(dtype=np.float64)),
        "std": number(data.std(dtype=np.float64)),
        "median": number(median),
        "robust_sigma": number(mad * 1.4826),
        "percentiles": {f"p{int(p)}": number(v) for p, v in zip(PERCENTILES, qs)},
    }


def _phase_stats(frame: Frame, normalized: np.ndarray, region: Box) -> list[dict]:
    layout = frame.sensor.layout
    raw_planes = phase_planes(frame.raster.values, layout, region)
    signal_planes = phase_planes(normalized, layout, region)
    measurements = []
    for raw, signal in zip(raw_planes, signal_planes):
        data = signal.samples
        measurements.append({
            "phase": raw.name,
            "color": raw.color,
            "black_level": frame.sensor.black[raw.phase],
            "white_level": frame.sensor.white[raw.phase],
            "raw": describe_values(raw.samples),
            "normalized": describe_values(data),
            "under_black": int(np.count_nonzero(data < 0)),
            "near_white": int(np.count_nonzero((0.98 <= data) & (data < 1))),
            "clipped": int(np.count_nonzero(data >= 1)),
        })
    return measurements


def measure_region(frame: Frame, region: Box, correction: Correction | None = None) -> dict:
    signal = normalized_source(frame) if correction is None else correction.normalized_source
    phases = _phase_stats(frame, signal, region)
    sample_counts = {phase["phase"]: phase["raw"]["count"] for phase in phases}
    active = region.intersection(frame.sensor.active)
    stats = {
        "region": region.as_dict(),
        "sensor_box": region.translated(
            frame.sensor.layout.origin_x, frame.sensor.layout.origin_y
        ).as_dict(),
        "active_intersection": active.as_dict() if active else None,
        "sites": region.area,
        "phase_counts": sample_counts,
        "phases": phases,
        "clipped": sum(item["clipped"] for item in phases),
        "under_black": sum(item["under_black"] for item in phases),
    }
    if correction is not None:
        stats["calibrated"] = _corrected_phase_stats(correction, region)
    return stats


def _corrected_phase_stats(correction: Correction, region: Box) -> dict:
    layout = correction.layout
    planes = phase_planes(correction.linear, layout, region)
    valid = phase_planes(correction.valid, layout, region)
    repaired = phase_planes(correction.repaired, layout, region)
    lost = phase_planes(correction.flat_invalid, layout, region)
    stats = []
    for plane, included, filled, weak in zip(planes, valid, repaired, lost):
        stats.append({
            "phase": plane.name,
            "usable": int(np.count_nonzero(included.samples)),
            "filled": int(np.count_nonzero(filled.samples)),
            "weak_flat": int(np.count_nonzero(weak.samples)),
            "values": describe_values(plane.samples[included.samples]),
        })
    return {"phases": stats}


def _normal_histogram(values: np.ndarray, bins: int) -> dict:
    counts, edges = np.histogram(values, bins=bins, range=(0, 1))
    return {
        "counts": counts.tolist(),
        "edges": np.round(edges, 6).tolist(),
        "below": int(np.count_nonzero(values < 0)),
        "above": int(np.count_nonzero(values > 1)),
        "clipped": int(np.count_nonzero(values >= 1)),
    }


def _stop_histogram(values: np.ndarray, bins: int) -> dict:
    positive = values[values > 0]
    stops = np.log2(positive)
    counts, edges = np.histogram(stops, bins=bins, range=(STOPS_MINIMUM, 0))
    return {
        "counts": counts.tolist(),
        "edges": np.round(edges, 6).tolist(),
        "below": int(np.count_nonzero(stops < STOPS_MINIMUM) + values.size - positive.size),
        "above": int(np.count_nonzero(stops > 0)),
        "clipped": int(np.count_nonzero(stops >= 0)),
    }


def exposure_histogram(
    frame: Frame,
    region: Box,
    *,
    space: str = "linear",
    bins: int = 64,
) -> dict:
    if space not in HISTOGRAM_SPACES:
        raise InputError("Unknown histogram space.")
    if bins not in (16, 32, 64, 128, 256):
        raise InputError("Histogram bins must be 16, 32, 64, 128 or 256.")
    signal = normalized_source(frame)
    calculator = _normal_histogram if space == "linear" else _stop_histogram
    phases = []
    for plane in phase_planes(signal, frame.sensor.layout, region):
        channel = calculator(plane.samples.ravel(), bins)
        channel["phase"] = plane.name
        channel["samples"] = plane.site_count
        phases.append(channel)
    return {"region": region.as_dict(), "space": space, "bins": bins, "phases": phases}


def exposure_summary(frame: Frame, region: Box, signal: np.ndarray | None = None) -> dict:
    signal = normalized_source(frame) if signal is None else signal
    volume = signal if region.area == signal.size else region.crop(signal)
    report = []
    for plane in phase_planes(signal, frame.sensor.layout, region):
        values = plane.samples
        if not values.size:
            report.append({"phase": plane.name, "samples": 0, "levels": []})
            continue
        fractions = []
        for bound in EXPOSURE_LEVELS:
            count = int(np.count_nonzero(values >= bound))
            fractions.append({"threshold": bound, "count": count, "fraction": number(count / values.size)})
        finite_peak = np.percentile(values, 99.5)
        peak = max(float(finite_peak), 1 / 65536)
        report.append({
            "phase": plane.name,
            "samples": plane.site_count,
            "headroom_stops_p99_5": number(-np.log2(peak)),
            "levels": fractions,
            "low_black": int(np.count_nonzero(values < 0)),
            "p99_5_normalized": number(finite_peak),
        })
    return {
        "region": region.as_dict(),
        "pixels": region.area,
        "phases": report,
        "clipped": int(np.count_nonzero(volume >= 1)),
        "near_white": int(np.count_nonzero((volume >= 0.98) & (volume < 1))),
        "below_black": int(np.count_nonzero(volume < 0)),
    }


def exposure_classes(signal: np.ndarray) -> np.ndarray:
    result = np.zeros(signal.shape, dtype=np.uint8)
    result[signal < 0] = 1
    result[(signal >= 0.98) & (signal < 1)] = 2
    result[signal >= 1] = 3
    return result


@dataclass
class Trace:
    phase: str
    positions: np.ndarray
    medians: np.ndarray
    spread: np.ndarray
    samples: np.ndarray


def profile_traces(
    frame: Frame,
    region: Box,
    axis: str,
    values: np.ndarray | None = None,
) -> list[Trace]:
    if axis not in PROFILE_AXES:
        raise InputError("Profile axis must be rows or columns.")
    if values is None:
        values = frame.raster.values.astype(np.float32)
    vertical = axis == "rows"
    origin = region.y if vertical else region.x
    layout = frame.sensor.layout
    sensor_origin = layout.origin_y if vertical else layout.origin_x
    traces = []
    for plane in phase_planes(values, layout, region):
        data = plane.samples if vertical else plane.samples.T
        start = plane.offset_y if vertical else plane.offset_x
        if data.size:
            positions = origin + start + sensor_origin + 2 * np.arange(data.shape[0])
            rows = data[np.isfinite(data).sum(axis=1) > 0]
            use = np.isfinite(data).sum(axis=1) > 0
            centers = np.nanmedian(rows, axis=1)
            dispersion = np.nanmedian(np.abs(rows - centers[:, None]), axis=1) * 1.4826
            samples = np.isfinite(rows).sum(axis=1)
            traces.append(Trace(plane.name, positions[use], centers, dispersion, samples))
        else:
            traces.append(Trace(
                plane.name, np.array([], np.int32), np.array([]), np.array([]), np.array([], np.int32)
            ))
    return traces


def _bin_trace(trace: Trace, limit: int) -> dict:
    rows = len(trace.positions)
    if rows <= limit:
        return {
            "phase": trace.phase,
            "positions": trace.positions.tolist(),
            "medians": np.round(trace.medians, 6).tolist(),
            "robust_sigma": np.round(trace.spread, 6).tolist(),
            "samples": trace.samples.tolist(),
            "binned": False,
        }
    cuts = np.linspace(0, rows, limit + 1, dtype=int)
    positions = []
    levels = []
    spread = []
    counts = []
    spans = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        counts.append(int(trace.samples[lo:hi].sum()))
        levels.append(number(np.median(trace.medians[lo:hi])))
        spread.append(number(np.median(trace.spread[lo:hi])))
        positions.append(number(np.mean(trace.positions[lo:hi]), 2))
        spans.append([int(trace.positions[lo]), int(trace.positions[hi - 1])])
    return {
        "phase": trace.phase,
        "positions": positions,
        "medians": levels,
        "robust_sigma": spread,
        "samples": counts,
        "span": spans,
        "binned": True,
    }


def profile_report(frame: Frame, region: Box, axis: str, normalized: bool, limit: int = 350) -> dict:
    values = normalized_source(frame) if normalized else None
    return {
        "region": region.as_dict(),
        "axis": axis,
        "unit": "fraction of phase range" if normalized else "code",
        "traces": [_bin_trace(trace, limit) for trace in profile_traces(frame, region, axis, values)],
    }


def _detrend_trace(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    good = trace.samples >= MIN_PROFILE_SITES
    if np.count_nonzero(good) < 12:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    x = trace.positions[good]
    y = trace.medians[good].astype(np.float64)
    centered_x = x - x.mean()
    slope = float(np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x))
    residual = y - (slope * centered_x + y.mean())
    return x, residual


def _frequency_peaks(power: np.ndarray, frequencies: np.ndarray, minimum_period: float) -> list[dict]:
    valid = (frequencies > 0) & (frequencies <= 1 / minimum_period)
    peak_idxs = np.where(valid & (power >= np.roll(power, 1)) & (power >= np.roll(power, -1)))[0]
    base = max(float(np.median(power[valid])) if valid.any() else 0, 1e-20)
    order = sorted(peak_idxs, key=lambda idx: (-power[idx], idx))[:4]
    result = []
    for index in order:
        if power[index] <= 1e-15:
            continue
        result.append({
            "cycles_per_sensor_site": number(frequencies[index], 8),
            "period_sites": number(1 / frequencies[index], 4),
            "amplitude": number(np.sqrt(power[index]), 8),
            "relative_to_floor_db": number(10 * np.log10(max(power[index] / base, 1)), 3),
        })
    return result


def _periodogram(positions: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = (positions - positions[0]) // 2
    if len(steps) != steps[-1] + 1:
        expected = np.arange(steps[-1] + 1)
        residual = np.interp(expected, steps, residual)
    window = np.hanning(len(residual))
    corrected = residual * window
    spectrum = np.fft.rfft(corrected)
    power = (2 * np.abs(spectrum) / max(float(window.sum()), 1)) ** 2
    return power, np.fft.rfftfreq(len(residual), d=2)


def banding_report(frame: Frame, region: Box, axis: str) -> dict:
    measurements = []
    for trace in profile_traces(frame, region, axis):
        position, residual = _detrend_trace(trace)
        if residual.size == 0:
            measurements.append({
                "phase": trace.phase, "usable": False, "sites_per_trace_minimum": MIN_PROFILE_SITES,
                "peak_periods": [],
            })
            continue
        signal, freq = _periodogram(position, residual)
        measurements.append({
            "phase": trace.phase,
            "usable": True,
            "trace_points": int(position.size),
            "rms_code": number(np.sqrt(np.mean(residual ** 2))),
            "peak_periods": _frequency_peaks(signal, freq, 4.0),
            "trace_p5_code": number(np.percentile(residual, 5)),
            "trace_p95_code": number(np.percentile(residual, 95)),
        })
    return {
        "region": region.as_dict(),
        "axis": axis,
        "unit": "site code",
        "phases": measurements,
        "method": "Line medians, linear detrend and Hann-windowed spectrum on each CFA phase.",
        "warning": "Repeated scene detail may appear as a sensor band. Use a uniform strip or dark frame.",
    }


def _matched_cfa(left: Frame, right: Frame, region: Box) -> tuple[Box, Box]:
    a = left.sensor.layout
    b = right.sensor.layout
    if a.pattern != b.pattern:
        raise InputError("Sensor comparisons require the same Bayer pattern.")
    if left.sensor.model and right.sensor.model and left.sensor.model != right.sensor.model:
        raise InputError("Sensor comparisons require matching camera models.")
    region_left = region.intersection(left.sensor.active)
    if region_left is None:
        raise InputError("The noise region does not overlap active sites in the first capture.")
    world_left = region_left.translated(a.origin_x, a.origin_y)
    world_right = right.sensor.active.translated(b.origin_x, b.origin_y)
    shared = world_left.intersection(world_right)
    if shared is None or shared.width < 2 or shared.height < 2:
        raise InputError("No shared active sensor region with every Bayer phase.")
    return shared.translated(-a.origin_x, -a.origin_y), shared.translated(-b.origin_x, -b.origin_y)


def _noise_from_pair(left: np.ndarray, right: np.ndarray) -> dict:
    good = (left > 0.005) & (right > 0.005) & (left < 0.98) & (right < 0.98)
    a = left[good]
    b = right[good]
    if a.size < 10:
        return {"usable": int(a.size), "usable_estimate": False}
    delta = a - b
    center = float(np.median(delta))
    common = (a + b) * 0.5
    noise = np.median(np.abs(delta - center)) * (1.4826 / np.sqrt(2))
    mean = float(np.median(common))
    standard = float(np.std(delta, dtype=np.float64) / np.sqrt(2))
    return {
        "usable": int(a.size),
        "usable_estimate": True,
        "median_signal": number(mean),
        "relative_offset": number(center),
        "robust_temporal_sigma": number(noise, 9),
        "rms_temporal_sigma": number(standard, 9),
        "snr_db": number(20 * np.log10(mean / noise), 3) if noise > 0 else None,
        "large_delta_fraction": number(np.count_nonzero(np.abs(delta - center) > 0.02) / a.size),
    }


def paired_noise_report(left: Frame, right: Frame, region: Box) -> dict:
    if left.id == right.id:
        raise InputError("Select two distinct captures of the same static scene.")
    region_left, region_right = _matched_cfa(left, right, region)
    basis = (
        ("exposure", left.sensor.exposure_ms, right.sensor.exposure_ms),
        ("analog gain", left.sensor.analog_gain, right.sensor.analog_gain),
    )
    warnings = []
    for label, a, b in basis:
        if a is None or b is None:
            warnings.append(f"Matching {label} was not verified because metadata is missing.")
        elif not np.isclose(a, b, rtol=0.02, atol=0):
            raise InputError(f"Temporal-noise frames need matching {label}.")
    a = phase_planes(normalized_source(left), left.sensor.layout, region_left)
    b = phase_planes(normalized_source(right), right.sensor.layout, region_right)
    phases = []
    for first, second in zip(a, b):
        info = _noise_from_pair(first.samples, second.samples)
        info["phase"] = first.name
        phases.append(info)
    return {
        "left_id": left.id,
        "right_id": right.id,
        "left_box": region_left.as_dict(),
        "right_box": region_right.as_dict(),
        "phases": phases,
        "warnings": [
            *warnings, "Motion, changing illumination and registration errors inflate the estimates.",
        ],
    }


def _single_phase_noise(sample: np.ndarray) -> dict:
    if min(sample.shape) < 3:
        return {"usable": 0, "usable_estimate": False}
    row_delta = (sample[1:, :] - sample[:-1, :]).ravel()
    col_delta = (sample[:, 1:] - sample[:, :-1]).ravel()
    r_sigma = np.median(np.abs(row_delta - np.median(row_delta))) * 1.4826 / np.sqrt(2)
    c_sigma = np.median(np.abs(col_delta - np.median(col_delta))) * 1.4826 / np.sqrt(2)
    center = ndimage.median_filter(sample, size=3, mode="mirror")
    error = sample - center
    sites = (sample > 0.005) & (sample < 0.98)
    if np.count_nonzero(sites) < 10:
        return {"usable": int(np.count_nonzero(sites)), "usable_estimate": False}
    residual = error[sites]
    return {
        "usable": int(np.count_nonzero(sites)),
        "usable_estimate": True,
        "median_signal": number(np.median(sample[sites])),
        "row_sigma": number(r_sigma, 9),
        "column_sigma": number(c_sigma, 9),
        "local_residual_sigma": number(np.median(np.abs(residual - np.median(residual))) * 1.4826, 9),
    }


def single_noise_report(frame: Frame, region: Box) -> dict:
    raw = normalized_source(frame)
    measurements = []
    for plane in phase_planes(raw, frame.sensor.layout, region):
        info = _single_phase_noise(plane.samples)
        info["phase"] = plane.name
        measurements.append(info)
    return {
        "region": region.as_dict(),
        "phases": measurements,
        "meaning": "Spatial residuals in apparently uniform areas, not temporal read noise.",
    }
