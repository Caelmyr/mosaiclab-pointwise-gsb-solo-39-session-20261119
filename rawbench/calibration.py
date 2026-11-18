"""Reference-frame correction in the native Bayer sampling lattice."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .cfa import phase_field, phase_planes
from .domain import (
    Box, Frame, InputError, Layout, finite, flag, object_value, optional_id,
    reject_unknown, sensor_compatible,
)

PHASE_CROSS = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
PHASE_SQUARE = np.ones((3, 3), dtype=np.float32)
PHASE_SQUARE[1, 1] = 0

MIN_ROW_OFFSET_SAMPLES = 6
ROW_OFFSET_CLIP = 4.0
ROW_OFFSET_TOLERANCE_FLOOR = 1.5
ROW_OFFSET_LIST_LIMIT = 64


@dataclass(frozen=True)
class CalibrationSettings:
    dark_id: str | None = None
    flat_id: str | None = None
    defects_id: str | None = None
    scale_dark_exposure: bool = False
    flat_floor: float = 0.06
    repair_detected: bool = False
    defect_threshold: float = 12.0
    row_offset_ob: bool = False

    @classmethod
    def read(cls, value) -> CalibrationSettings:
        obj = object_value(value if value is not None else {}, "Calibration")
        reject_unknown(
            obj,
            ("dark_id", "flat_id", "defects_id", "scale_dark_exposure", "flat_floor",
             "repair_detected", "defect_threshold", "row_offset_ob"),
            "calibration",
        )
        return cls(
            optional_id(obj.get("dark_id"), "Dark reference"),
            optional_id(obj.get("flat_id"), "Flat reference"),
            optional_id(obj.get("defects_id"), "Defect mask"),
            flag(obj.get("scale_dark_exposure", False), "Scale dark exposure"),
            finite(obj.get("flat_floor", 0.06), "Flat floor", 0.001, 0.9),
            flag(obj.get("repair_detected", False), "Repair detected outliers"),
            finite(obj.get("defect_threshold", 12), "Defect threshold", 3, 80),
            flag(obj.get("row_offset_ob", False), "Row offset from masked columns"),
        )

    def as_dict(self) -> dict:
        return {
            "dark_id": self.dark_id,
            "flat_id": self.flat_id,
            "defects_id": self.defects_id,
            "scale_dark_exposure": self.scale_dark_exposure,
            "flat_floor": self.flat_floor,
            "repair_detected": self.repair_detected,
            "defect_threshold": self.defect_threshold,
            "row_offset_ob": self.row_offset_ob,
        }

    def references(self) -> dict[str, str]:
        result = {}
        for role, identifier in (
            ("dark", self.dark_id), ("flat", self.flat_id), ("defects", self.defects_id)
        ):
            if identifier:
                result[role] = identifier
        return result


@dataclass
class Correction:
    linear: np.ndarray
    normalized_source: np.ndarray
    valid: np.ndarray
    repaired: np.ndarray
    flat_invalid: np.ndarray
    original_clipped: np.ndarray
    gain_map: np.ndarray | None
    row_offset: np.ndarray | None
    active: Box
    layout: Layout
    report: dict


def _compatible(scene: Frame, reference: Frame | None, role: str) -> None:
    if reference is None:
        return
    if reference.role != role:
        raise InputError(f"Reference {reference.name} must have the {role} role.")
    if scene.project != reference.project:
        raise InputError("References must belong to the same workbench.")
    if not sensor_compatible(scene, reference):
        raise InputError(
            f"{role.capitalize()} reference needs the same dimensions, sensor origin, "
            "Bayer pattern, active area and compatible camera model."
        )


def code_offsets(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    raw = frame.raster.values
    black = phase_field(raw.shape, frame.sensor.layout, frame.sensor.black)
    white = phase_field(raw.shape, frame.sensor.layout, frame.sensor.white)
    return raw.astype(np.float32) - black, white - black


def normalized_source(frame: Frame) -> np.ndarray:
    signal, full_scale = code_offsets(frame)
    return signal / full_scale


def _sensor_rows(layout: Layout, first_row: int, count: int) -> np.ndarray:
    return layout.origin_y + first_row + 2 * np.arange(count)


def estimate_row_offsets(frame: Frame, signal: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate one offset per sensor row and CFA phase from optically black columns.

    Only columns outside the active area contribute, so scene content never biases
    the estimate. Each phase keeps its own mean level: the correction is the
    row-to-row deviation of the masked columns, not their absolute value. Rows
    with too few clean masked samples are left unchanged and reported.
    """
    layout = frame.sensor.layout
    active = frame.sensor.active
    height, width = signal.shape
    strips = []
    if active.x > 0:
        strips.append(("left", slice(0, active.x)))
    if active.right < width:
        strips.append(("right", slice(active.right, width)))
    correction = np.zeros(signal.shape, dtype=np.float32)
    report = {
        "enabled": True,
        "strips": [
            {"side": side, "columns": part.stop - part.start} for side, part in strips
        ],
        "min_samples_per_row": MIN_ROW_OFFSET_SAMPLES,
        "clip_sigma": ROW_OFFSET_CLIP,
        "phases": [],
        "uncorrected_row_count": 0,
        "rows_not_corrected": [],
        "warnings": [],
    }
    if not strips:
        report["warnings"].append(
            "No optically black columns outside the active area; rows left unchanged."
        )
        report["uncorrected_row_count"] = height
        report["rows_not_corrected"] = (
            layout.origin_y + np.arange(height)[:ROW_OFFSET_LIST_LIMIT]
        ).tolist()
        return correction, report
    masked = np.zeros((height, width), dtype=bool)
    for _side, part in strips:
        masked[:, part] = True
    uncorrected = np.zeros(height, dtype=bool)
    for phase in range(4):
        name = layout.phase_name(phase)
        rows_at, columns_at = layout.phase_slice(phase)
        plane = signal[rows_at, columns_at]
        usable_columns = masked[rows_at, columns_at].any(axis=0)
        samples = plane[:, usable_columns]
        per_row = samples.shape[1]
        sensor_rows = _sensor_rows(layout, rows_at.start, plane.shape[0])
        entry = {
            "phase": name,
            "ob_samples_per_row": int(per_row),
            "rows_corrected": 0,
            "outlier_samples_rejected": 0,
            "median_offset_code": None,
            "rms_offset_code": None,
            "rows_not_corrected": [],
        }
        report["phases"].append(entry)
        if per_row < MIN_ROW_OFFSET_SAMPLES:
            report["warnings"].append(
                f"Only {per_row} masked {name} samples per row; {name} rows left unchanged."
            )
            entry["rows_not_corrected"] = sensor_rows[:ROW_OFFSET_LIST_LIMIT].tolist()
            uncorrected[sensor_rows - layout.origin_y] = True
            continue
        center = np.median(samples, axis=1)
        spread = np.median(np.abs(samples - center[:, None]), axis=1)
        tolerance = np.maximum(
            ROW_OFFSET_CLIP * 1.4826 * spread, ROW_OFFSET_TOLERANCE_FLOOR
        )
        inlier = np.abs(samples - center[:, None]) <= tolerance[:, None]
        count = inlier.sum(axis=1)
        good = count >= MIN_ROW_OFFSET_SAMPLES
        offset = np.zeros(plane.shape[0], dtype=np.float64)
        offset[good] = (
            np.where(inlier, samples, 0.0)[good].sum(axis=1) / count[good]
        )
        entry["outlier_samples_rejected"] = int(samples.size - inlier.sum())
        entry["rows_corrected"] = int(np.count_nonzero(good))
        if good.any():
            # Keep the phase's own pedestal so relative phase levels are preserved.
            offset[good] -= offset[good].mean()
            entry["median_offset_code"] = round(float(np.median(np.abs(offset[good]))), 4)
            entry["rms_offset_code"] = round(float(np.sqrt(np.mean(offset[good] ** 2))), 4)
        missing = sensor_rows[~good]
        entry["rows_not_corrected"] = missing[:ROW_OFFSET_LIST_LIMIT].tolist()
        uncorrected[missing - layout.origin_y] = True
        plane_correction = np.zeros(plane.shape, dtype=np.float32)
        plane_correction[good, :] = offset[good][:, None].astype(np.float32)
        correction[rows_at, columns_at] = plane_correction
    if not any(item["rows_corrected"] for item in report["phases"]):
        report["warnings"].append("No usable masked samples; the capture is unchanged.")
    missing_rows = np.flatnonzero(uncorrected)
    report["uncorrected_row_count"] = int(missing_rows.size)
    report["rows_not_corrected"] = (
        layout.origin_y + missing_rows[:ROW_OFFSET_LIST_LIMIT]
    ).tolist()
    return correction, report


def dark_exposure_ratio(
    scene: Frame, dark: Frame, scale: bool
) -> tuple[float, list[str]]:
    if not scale:
        messages = []
        a = scene.sensor.exposure_ms
        b = dark.sensor.exposure_ms
        if a is not None and b is not None and abs(a - b) > max(a, b) * 0.02:
            messages.append("Dark and scene exposures differ. Dark scaling is off.")
        return 1.0, messages
    if scene.sensor.exposure_ms is None or dark.sensor.exposure_ms is None:
        raise InputError("Exposure scaling requires exposure metadata on scene and dark.")
    a = scene.sensor.analog_gain
    b = dark.sensor.analog_gain
    if a is not None and b is not None and not np.isclose(a, b, rtol=0.01):
        raise InputError("Dark exposure scaling requires matching analog gain.")
    ratio = scene.sensor.exposure_ms / dark.sensor.exposure_ms
    if not 0.0625 <= ratio <= 16:
        raise InputError("Scene/dark exposure ratio must be between 1/16× and 16×.")
    return ratio, ["Dark current is assumed to scale linearly with exposure time."]


def _flat_reliability(
    flat: Frame, corrected_flat: np.ndarray, floor: float
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    shape = corrected_flat.shape
    layout = flat.sensor.layout
    normalized = normalized_source(flat)
    usable = (normalized < 0.98) & (corrected_flat > 0)
    inside = flat.sensor.active
    gain = np.ones(shape, dtype=np.float32)
    trusted = np.zeros(shape, dtype=bool)
    stats = []
    for active_plane in phase_planes(corrected_flat, layout, inside):
        idx = active_plane.phase
        phase = layout.phase_slice(idx)
        code = corrected_flat[phase]
        ok = phase_planes(usable, layout, inside)[idx].samples
        n = int(np.count_nonzero(ok))
        if n < max(1, active_plane.site_count // 20):
            raise InputError(f"Flat reference has too few usable {active_plane.name} sites.")
        reference = float(np.median(active_plane.samples[ok]))
        threshold = max(reference * floor, np.finfo(np.float32).eps)
        accepted = usable[phase] & (code >= threshold)
        divided = np.divide(reference, code, out=np.ones_like(code), where=accepted)
        gain[phase] = divided
        trusted[phase] = accepted
        stats.append({
            "phase": active_plane.name,
            "median_code_above_black": round(reference, 4),
            "minimum_usable_code": round(threshold, 4),
            "reference_sites": n,
            "invalid_active_sites": int(
                active_plane.site_count -
                np.count_nonzero(phase_planes(trusted, layout, inside)[idx].samples)
            ),
        })
    return gain, trusted, stats


def _local_scale(values: np.ndarray, baseline: np.ndarray, mode: str) -> np.ndarray:
    residual = np.abs(values - baseline)
    neighborhood = ndimage.median_filter(residual, size=5, mode=mode)
    return np.maximum(neighborhood * 1.4826, 1 / 4096)


def detect_defects(
    linear: np.ndarray,
    layout: Layout,
    valid: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    result = np.zeros(linear.shape, dtype=bool)
    report = []
    for plane in phase_planes(linear, layout):
        idx = layout.phase_slice(plane.phase)
        data = plane.samples
        allowed = valid[idx]
        center = ndimage.median_filter(data, size=3, mode="mirror")
        residual = data - center
        scale = _local_scale(data, center, "mirror")
        wide = ndimage.maximum_filter(data, size=3, mode="mirror")
        neighbors = ndimage.convolve(allowed.astype(np.float32), PHASE_SQUARE, mode="constant")
        isolated = (residual < 0) | ((wide - center) <= np.abs(residual) * 1.1)
        flagged = (
            allowed & isolated & (neighbors >= 3)
            & (np.abs(residual) > threshold * scale)
            & (np.abs(residual) >= 0.025)
        )
        result[idx] = flagged
        report.append({
            "phase": plane.name,
            "hot": int(np.count_nonzero(flagged & (residual > 0))),
            "cold": int(np.count_nonzero(flagged & (residual < 0))),
        })
    return result, report


def _interpolate_phase(
    values: np.ndarray, valid: np.ndarray, needed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    supported = valid.copy()
    unresolved = needed.copy()
    for weights in (PHASE_CROSS, PHASE_SQUARE, PHASE_SQUARE):
        if not unresolved.any():
            break
        total = ndimage.convolve(np.where(supported, result, 0), weights, mode="constant")
        count = ndimage.convolve(supported.astype(np.float32), weights, mode="constant")
        fill = unresolved & (count > 0)
        result[fill] = total[fill] / count[fill]
        supported |= fill
        unresolved &= ~fill
    return result, needed & ~unresolved


def repair_defects(
    values: np.ndarray, layout: Layout, bad: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    repaired = np.zeros(values.shape, dtype=bool)
    for phase in range(4):
        idx = layout.phase_slice(phase)
        target = bad[idx]
        if target.any():
            plane, replaced = _interpolate_phase(values[idx], valid[idx] & ~target, target)
            result[idx] = plane
            repaired[idx] = replaced
    return result, repaired


def calibrate(
    scene: Frame, settings: CalibrationSettings, references: dict[str, Frame] | None = None
) -> Correction:
    refs = references or {}
    dark = refs.get("dark")
    flat = refs.get("flat")
    defects = refs.get("defects")
    for role, reference in (("dark", dark), ("flat", flat), ("defects", defects)):
        _compatible(scene, reference, role)
    source_signal, full_scale = code_offsets(scene)
    source = source_signal / full_scale
    clipped = source >= 1
    messages = []
    ratio = None
    corrected = source_signal.copy()
    if dark is not None:
        ratio, dark_notices = dark_exposure_ratio(scene, dark, settings.scale_dark_exposure)
        messages.extend(dark_notices)
        dark_signal, dark_scale = code_offsets(dark)
        if dark.raster.maximum != scene.raster.maximum:
            raise InputError("Dark and scene files must use the same numeric code maximum.")
        if np.any(dark_signal >= dark_scale):
            messages.append("The dark reference contains saturated samples.")
        corrected -= dark_signal * ratio

    row_offsets = None
    row_offset_report = None
    if settings.row_offset_ob:
        row_offsets, row_offset_report = estimate_row_offsets(scene, corrected)
        corrected -= row_offsets

    gain = None
    valid = np.ones(source.shape, dtype=bool)
    flat_stats = None
    if flat is not None:
        flat_signal, _ = code_offsets(flat)
        gain, valid, flat_stats = _flat_reliability(flat, flat_signal, settings.flat_floor)
        corrected *= gain

    linear = corrected / full_scale
    flat_invalid = ~valid
    bad = np.zeros(linear.shape, dtype=bool)
    detected_report = None
    if defects is not None:
        bad |= defects.raster.values != 0
    if settings.repair_detected:
        detection, detected_report = detect_defects(
            linear, scene.sensor.layout, valid & ~clipped,
            threshold=settings.defect_threshold,
        )
        bad |= detection
    repaired = np.zeros(bad.shape, dtype=bool)
    if bad.any():
        linear, repaired = repair_defects(
            linear, scene.sensor.layout, bad, valid & ~clipped
        )
        valid = (valid & ~bad) | repaired

    area = scene.sensor.active
    counts = {
        "active_sites": area.area,
        "original_clipped": int(np.count_nonzero(area.crop(clipped))),
        "flat_invalid": int(np.count_nonzero(area.crop(flat_invalid))),
        "marked_defects": int(np.count_nonzero(area.crop(bad))),
        "repaired": int(np.count_nonzero(area.crop(repaired))),
        "unresolved": int(np.count_nonzero(area.crop(~valid))),
        "below_black_after_correction": int(np.count_nonzero(area.crop(linear) < 0)),
    }
    report = {
        "references": {key: value.id for key, value in refs.items()},
        "dark_ratio": ratio,
        "flat_phases": flat_stats,
        "detected": detected_report,
        "row_offset": row_offset_report,
        "counts": counts,
        "warnings": messages,
    }
    return Correction(
        linear, source, valid, repaired, flat_invalid, clipped, gain,
        row_offsets, area, scene.sensor.layout, report,
    )


def reference_summary(frame: Frame) -> dict:
    signal, _ = code_offsets(frame)
    normalized = normalized_source(frame)
    phases = []
    for plane in phase_planes(signal, frame.sensor.layout, frame.sensor.active):
        data = plane.samples
        p5, median, p95 = np.percentile(data, [5, 50, 95])
        deviation = np.median(np.abs(data - median))
        plane_normalized = phase_planes(normalized, frame.sensor.layout, frame.sensor.active)
        phases.append({
            "phase": plane.name,
            "sites": plane.site_count,
            "p5_code_above_black": round(float(p5), 5),
            "median_code_above_black": round(float(median), 5),
            "p95_code_above_black": round(float(p95), 5),
            "robust_sigma_code": round(float(deviation * 1.4826), 5),
            "near_saturation": int(np.count_nonzero(plane_normalized[plane.phase].samples >= 0.98)),
        })
    return {
        "role": frame.role,
        "active_area": frame.sensor.active.as_dict(),
        "phases": phases,
        "marked_sites": int(np.count_nonzero(frame.raster.values)) if frame.role == "defects" else None,
    }
