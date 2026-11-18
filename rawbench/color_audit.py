"""Evaluate developed target patches in CIELAB independently of tone mapping."""
from __future__ import annotations

import numpy as np

from .analysis import number
from .development import Developed, srgb_decode, srgb_encode
from .domain import Box, InputError, bounded_numbers, object_value, reject_unknown, text_label

SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
WHITE = np.array([0.95047, 1.0, 1.08883])


def _lab_signal(xyz: np.ndarray) -> np.ndarray:
    eps = 6 / 29
    return np.where(xyz > eps ** 3, np.cbrt(xyz), xyz / (3 * eps * eps) + 4 / 29)


def working_to_lab(rgb: np.ndarray) -> np.ndarray:
    f = _lab_signal(rgb @ SRGB_TO_XYZ.T / WHITE)
    return np.stack(
        (116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])),
        axis=-1,
    )


def _hue(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    angle = np.mod(np.degrees(np.arctan2(b, a)), 360.0)
    return np.where(c == 0, 0.0, angle)


def _hue_components(h1: np.ndarray, h2: np.ndarray, c1: np.ndarray, c2: np.ndarray):
    zero = c1 * c2 == 0
    raw = h2 - h1
    wrapped = np.where(raw > 180, raw - 360, np.where(raw < -180, raw + 360, raw))
    delta_hue = np.where(zero, 0, wrapped)
    ordinary = (h1 + h2) * 0.5
    average = np.where(
        zero, h1 + h2,
        np.where(np.abs(raw) <= 180, ordinary, np.mod(ordinary + 180, 360)),
    )
    return 2 * np.sqrt(c1 * c2) * np.sin(np.radians(delta_hue / 2)), average


def ciede2000(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    l1, a1, b1 = np.moveaxis(np.asarray(first, np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(second, np.float64), -1, 0)
    mean_c = (np.hypot(a1, b1) + np.hypot(a2, b2)) * 0.5
    adjustment = 0.5 * (1 - np.sqrt(mean_c ** 7 / (mean_c ** 7 + 25.0 ** 7)))
    a1 = a1 * (1 + adjustment)
    a2 = a2 * (1 + adjustment)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    dh, mean_h = _hue_components(_hue(a1, b1, c1), _hue(a2, b2, c2), c1, c2)
    mean_l = (l1 + l2) * 0.5
    mean_c = (c1 + c2) * 0.5
    weight = (
        1 - 0.17 * np.cos(np.radians(mean_h - 30))
        + 0.24 * np.cos(np.radians(2 * mean_h))
        + 0.32 * np.cos(np.radians(3 * mean_h + 6))
        - 0.20 * np.cos(np.radians(4 * mean_h - 63))
    )
    scale_l = 1 + (0.015 * (mean_l - 50) ** 2) / np.sqrt(20 + (mean_l - 50) ** 2)
    scale_c = 1 + 0.045 * mean_c
    scale_h = 1 + 0.015 * mean_c * weight
    angle = 30 * np.exp(-((mean_h - 275) / 25) ** 2)
    rotate = -2 * np.sqrt(mean_c ** 7 / (mean_c ** 7 + 25.0 ** 7)) * np.sin(np.radians(2 * angle))
    dl, dc, dh = (l2 - l1) / scale_l, (c2 - c1) / scale_c, dh / scale_h
    return np.sqrt(np.maximum(dl * dl + dc * dc + dh * dh + rotate * dc * dh, 0))


def _patches(values, result: Developed) -> list[tuple[str, Box, np.ndarray]]:
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise InputError("Provide between 1 and 32 color patches.")
    patches = []
    used = set()
    for item in values:
        data = object_value(item, "Patch")
        reject_unknown(data, ("name", "region", "srgb"), "patch")
        name = text_label(data.get("name"), "Patch name", 56)
        if name in used:
            raise InputError("Give each patch a different name.")
        used.add(name)
        if "region" not in data:
            raise InputError("Every swatch needs a region.")
        region = Box.read(
            data["region"], result.frame.raster.width, result.frame.raster.height,
            minimum=3,
        )
        rgb = np.array(bounded_numbers(data.get("srgb"), "Reference sRGB", 3, 0, 1))
        patches.append((name, region, rgb))
    return patches


def _unusable_patch(name: str, region: Box, count: int, reason: str) -> dict:
    return {
        "name": name, "region": region.as_dict(), "usable": False,
        "samples": count, "reason": reason,
    }


def _patch_report(name: str, region: Box, rgb: np.ndarray, result: Developed) -> dict:
    area = result.box.intersection(region)
    if area is None or area != region:
        return _unusable_patch(name, region, 0, "Patch extends outside the developed area.")
    local = area.translated(-result.box.x, -result.box.y)
    raw_valid = area.crop(result.correction.valid)
    raw_clip = area.crop(result.correction.original_clipped)
    accepted = raw_valid & ~raw_clip
    if np.count_nonzero(accepted) < max(6, area.area // 2):
        return _unusable_patch(name, region, int(np.count_nonzero(accepted)), "Too many clipped or invalid sites.")
    working = local.crop(result.working_rgb)
    sample = working[accepted]
    center = np.median(sample, axis=0)
    measured = working_to_lab(center)
    expected = working_to_lab(srgb_decode(rgb))
    samples_lab = working_to_lab(sample)
    errors = ciede2000(samples_lab, expected)
    color_errors = ciede2000(samples_lab, measured)
    return {
        "name": name,
        "region": area.as_dict(),
        "usable": True,
        "samples": int(sample.shape[0]),
        "excluded_sites": int(area.area - sample.shape[0]),
        "target_srgb": np.round(rgb, 5).tolist(),
        "measured_srgb": np.round(srgb_encode(center), 5).tolist(),
        "measured_linear_rgb": np.round(center, 6).tolist(),
        "target_lab_d65": np.round(expected, 5).tolist(),
        "measured_lab_d65": np.round(measured, 5).tolist(),
        "delta_e2000": number(ciede2000(measured, expected)),
        "p90_delta_e": number(np.percentile(errors, 90)),
        "p90_within_patch_delta_e": number(np.percentile(color_errors, 90)),
        "over_display_gamut": int(np.count_nonzero(np.any((sample < 0) | (sample > 1), axis=1))),
    }


def color_audit(result: Developed, raw_patches) -> dict:
    readings = [_patch_report(*patch, result) for patch in _patches(raw_patches, result)]
    usable = [row for row in readings if row["usable"]]
    values = np.array([row["delta_e2000"] for row in usable])
    ranking = sorted(usable, key=lambda row: row["delta_e2000"], reverse=True)
    return {
        "patches": readings,
        "usable": len(usable),
        "total": len(readings),
        "median_delta_e": number(np.median(values)) if len(values) else None,
        "maximum_delta_e": number(values.max()) if len(values) else None,
        "worst_patch": ranking[0]["name"] if ranking else None,
        "neutral_reference": "D65",
        "measurement_space": "Working linear RGB before display tone or gamut conversion.",
    }
