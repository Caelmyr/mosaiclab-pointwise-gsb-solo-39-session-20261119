"""Linear-camera development and explicit conversion to display encodings."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .calibration import CalibrationSettings, Correction, calibrate
from .cfa import FILTERS, Reconstruction, phase_planes, reconstruct
from .domain import (
    CHANNELS, Box, Frame, InputError, bounded_numbers, choice, finite,
    flag, object_value, reject_unknown,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
MATRICES = {
    "identity": IDENTITY,
    "studio": (1.09, -0.06, -0.03, -0.04, 1.08, -0.04, 0.0, -0.08, 1.08),
    "warm-practical": (1.04, 0.03, -0.07, -0.06, 1.12, -0.06, -0.01, -0.04, 1.05),
}
TONES = ("linear", "soft", "high-contrast")
PROFILES = ("srgb", "display-p3")
P3_MATRIX = np.array(
    [
        [0.822462, 0.177538, 0.0],
        [0.033194, 0.966806, 0.0],
        [0.017083, 0.072397, 0.910520],
    ],
    dtype=np.float32,
)
LUMINANCE = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class DevelopSettings:
    method: str = "directional"
    white_balance: tuple[float, ...] = (1.0, 1.0, 1.0)
    exposure_ev: float = 0.0
    matrix_name: str = "identity"
    custom_matrix: tuple[float, ...] = IDENTITY
    tone: str = "linear"
    contrast: float = 1.0
    saturation: float = 1.0
    profile: str = "srgb"
    active_only: bool = True
    calibration: CalibrationSettings = CalibrationSettings()

    @classmethod
    def read(cls, value) -> DevelopSettings:
        obj = object_value(value if value is not None else {}, "Develop settings")
        reject_unknown(
            obj,
            ("method", "white_balance", "exposure_ev", "matrix_name", "custom_matrix",
             "tone", "contrast", "saturation", "profile", "active_only", "calibration"),
            "develop settings",
        )
        matrix_name = choice(
            obj.get("matrix_name", "identity"), "Color matrix", (*MATRICES, "custom")
        )
        matrix = bounded_numbers(obj.get("custom_matrix", IDENTITY), "Color matrix", 9, -6, 6)
        if matrix_name == "custom":
            validate_matrix(matrix)
        return cls(
            choice(obj.get("method", "directional"), "Interpolation", FILTERS),
            bounded_numbers(obj.get("white_balance", (1, 1, 1)), "White balance", 3, 0.125, 8),
            finite(obj.get("exposure_ev", 0), "Exposure compensation", -6, 6),
            matrix_name,
            matrix,
            choice(obj.get("tone", "linear"), "Tone curve", TONES),
            finite(obj.get("contrast", 1), "Contrast", 0.25, 3),
            finite(obj.get("saturation", 1), "Saturation", 0, 2.5),
            choice(obj.get("profile", "srgb"), "Output profile", PROFILES),
            flag(obj.get("active_only", True), "Active area only"),
            CalibrationSettings.read(obj.get("calibration")),
        )

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "white_balance": list(self.white_balance),
            "exposure_ev": self.exposure_ev,
            "matrix_name": self.matrix_name,
            "custom_matrix": list(self.custom_matrix),
            "tone": self.tone,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "profile": self.profile,
            "active_only": self.active_only,
            "calibration": self.calibration.as_dict(),
        }

    def matrix(self) -> np.ndarray:
        values = self.custom_matrix if self.matrix_name == "custom" else MATRICES[self.matrix_name]
        return np.array(values, dtype=np.float32).reshape(3, 3)

    def preview_settings(self) -> DevelopSettings:
        if self.profile != "srgb":
            return replace(self, profile="srgb")
        return self


@dataclass
class Developed:
    correction: Correction
    reconstruction: Reconstruction
    frame: Frame
    box: Box
    settings: DevelopSettings
    working_rgb: np.ndarray
    display_linear: np.ndarray
    encoded: np.ndarray
    report: dict


def validate_matrix(values: tuple[float, ...]) -> None:
    matrix = np.array(values).reshape(3, 3)
    scale = np.linalg.norm(matrix, axis=1)
    if any(scale < 0.03):
        raise InputError("Each output channel needs a nonzero color matrix row.")
    if abs(float(np.linalg.det(matrix))) < 0.00001:
        raise InputError("The color matrix is nearly singular.")
    if not np.all(matrix.sum(axis=1) > 0):
        raise InputError("A neutral positive input must remain positive in all matrix outputs.")


def linear_luminance(rgb: np.ndarray) -> np.ndarray:
    return np.einsum("...c,c->...", rgb, LUMINANCE, optimize=False)


def srgb_encode(values: np.ndarray) -> np.ndarray:
    positive = np.clip(values, 0, 1)
    small = positive <= 0.0031308
    nonlinear = 1.055 * np.power(positive, np.float32(1 / 2.4)) - 0.055
    return np.where(small, positive * 12.92, nonlinear).astype(np.float32, copy=False)


def srgb_decode(values: np.ndarray) -> np.ndarray:
    positive = np.clip(values, 0, 1)
    nonlinear = np.power((positive + 0.055) / 1.055, 2.4)
    return np.where(positive <= 0.04045, positive / 12.92, nonlinear)


def _tone_luminance(luminance: np.ndarray, curve: str, contrast: float) -> np.ndarray:
    signal = np.maximum(luminance, 0)
    if contrast != 1:
        pivot = np.float32(0.18)
        signal = np.power(signal / pivot, contrast) * pivot
    if curve == "soft":
        shoulder = 0.72
        stem = signal - shoulder
        return np.where(
            signal <= shoulder, signal,
            shoulder + 0.28 * (1 - np.exp(-np.maximum(stem, 0) / 0.28)),
        )
    if curve == "high-contrast":
        ratio = 1.0 / (1.0 + np.maximum(signal, 0))
        return np.minimum(signal * signal * ratio * 1.9, 2.5)
    return signal


def tone_rgb(rgb: np.ndarray, curve: str, contrast: float, saturation: float) -> np.ndarray:
    luma = linear_luminance(rgb)
    toned = _tone_luminance(luma, curve, contrast)
    gain = np.divide(toned, luma, out=np.ones_like(luma), where=luma > 0.000001)
    if saturation != 1:
        saturated = luma[..., None] + (rgb - luma[..., None]) * saturation
    else:
        saturated = rgb
    return saturated * gain[..., None]


def camera_to_working(rgb: np.ndarray, settings: DevelopSettings) -> np.ndarray:
    white = np.array(settings.white_balance, dtype=np.float32)
    balanced = rgb * white * np.float32(2 ** settings.exposure_ev)
    return np.einsum("...c,kc->...k", balanced, settings.matrix(), optimize=False)


def display_gamut(rgb: np.ndarray, profile: str) -> np.ndarray:
    if profile == "display-p3":
        return np.einsum("...c,kc->...k", rgb, P3_MATRIX, optimize=False)
    return rgb


def gamut_statistics(rgb: np.ndarray, pre_tone: np.ndarray) -> dict:
    count = rgb.shape[0] * rgb.shape[1]
    per_channel = []
    before = np.max(pre_tone, axis=2)
    below = np.min(rgb, axis=2)
    above = np.max(rgb, axis=2)
    for i, key in enumerate(CHANNELS):
        channel = rgb[..., i]
        per_channel.append({
            "channel": key,
            "negative": int(np.count_nonzero(channel < 0)),
            "above_one": int(np.count_nonzero(channel > 1)),
            "p99": round(float(np.percentile(channel, 99)), 6),
        })
    return {
        "pixels": count,
        "pre_tone_over": int(np.count_nonzero(before > 1)),
        "display_negative": int(np.count_nonzero(below < 0)),
        "display_over": int(np.count_nonzero(above > 1)),
        "channels": per_channel,
    }


def develop(
    frame: Frame,
    settings: DevelopSettings,
    references: dict[str, Frame] | None = None,
) -> Developed:
    correction = calibrate(frame, settings.calibration, references)
    box = correction.active if settings.active_only else Box.full(frame.raster.width, frame.raster.height)
    layout = correction.layout.shifted(box.x, box.y)
    camera = reconstruct(
        box.crop(correction.linear), layout, settings.method,
        valid=box.crop(correction.valid),
    )
    working = camera_to_working(camera.rgb, settings)
    toned = tone_rgb(working, settings.tone, settings.contrast, settings.saturation)
    display = display_gamut(toned, settings.profile)
    encoded = srgb_encode(display)
    info = {
        "box": box.as_dict(),
        "local_pattern": layout.local_pattern(),
        "method": settings.method,
        "output_profile": settings.profile,
        "source": frame.id,
        "synthesis": frame.synthetic,
        "calibration": correction.report,
        "gamut": gamut_statistics(display, working),
        "missing_sensor_samples": camera.missing_samples,
        "interpolated_channel_values": camera.interpolated_samples,
    }
    return Developed(correction, camera, frame, box, settings, working, display, encoded, info)


def neutral_from_region(correction: Correction, region: Box) -> dict:
    intersection = region.intersection(correction.active)
    if intersection is None:
        raise InputError("The neutral patch must overlap the active sensor area.")
    valid = correction.valid & ~correction.original_clipped
    valid &= correction.linear >= 0.005
    filtered = phase_planes(valid, correction.layout, intersection)
    planes = phase_planes(correction.linear, correction.layout, intersection)
    groups: dict[str, list[np.ndarray]] = {key: [] for key in CHANNELS}
    counts = {}
    median_phase = {}
    for phase, coverage in zip(planes, filtered):
        data = phase.samples[coverage.samples]
        counts[phase.name] = int(data.size)
        median_phase[phase.name] = round(float(np.median(data)), 6) if data.size else None
        groups[phase.color].append(data)
    all_values = {key: np.concatenate(chunks) for key, chunks in groups.items()}
    if any(chunk.size < 5 for chunk in all_values.values()):
        raise InputError("Choose a larger neutral patch with at least five usable sites per color.")
    centers = {key: float(np.median(chunk)) for key, chunk in all_values.items()}
    target = centers["G"]
    gains = [target / centers[key] for key in CHANNELS]
    if any(not 0.125 <= value <= 8 for value in gains):
        raise InputError("Patch gains are outside 0.125×–8×. Choose a brighter neutral patch.")
    spread = {
        key: round(float(np.median(np.abs(samples - centers[key])) / centers[key]), 5)
        for key, samples in all_values.items()
    }
    notes = []
    if max(spread.values()) > 0.08:
        notes.append("The selected patch varies substantially in at least one color.")
    ratio = None
    if median_phase.get("G1") and median_phase.get("G2"):
        ratio = round(median_phase["G1"] / median_phase["G2"], 5)
        if abs(ratio - 1) > 0.08:
            notes.append("The two green lattices disagree; choose a more uniform patch.")
    return {
        "region": intersection.as_dict(),
        "gains": [round(gain, 6) for gain in gains],
        "phase_sites": counts,
        "phase_medians": median_phase,
        "relative_mad": spread,
        "green_ratio": ratio,
        "warnings": notes,
    }


def quantize(values: np.ndarray, bits: int = 8) -> np.ndarray:
    maximum = (1 << bits) - 1
    code = np.clip(np.rint(np.clip(values, 0, 1) * maximum), 0, maximum)
    return code.astype(np.uint8 if bits == 8 else np.uint16)
