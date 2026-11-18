"""Fit a 3x3 camera color matrix from target patches while protecting neutrals.

White balance and exposure are taken as given (the matrix is the only degree of
freedom).  The solver builds a ladder of candidates that trade chroma accuracy
against neutral fidelity, picks the best chroma candidate whose gray-scale
error stays within tolerance of the current matrix, and only accepts it when
the same gains also hold on patches and neutral regions that never
participated in the fit or the selection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .analysis import number
from .color_audit import ciede2000, working_to_lab
from .development import Developed, srgb_decode, srgb_encode, validate_matrix
from .domain import Box, InputError, bounded_numbers, object_value, reject_unknown, text_label

# Reference LAB chroma below which a swatch counts as a neutral.
NEUTRAL_CHROMA = 5.0
# Minimum number of usable patches of each kind before a fit is meaningful.
MIN_KIND = 2
MIN_NEUTRAL_REGIONS = 1
# Holdout acceptance gates (CIEDE2000 units).
CHROMA_GAIN = 0.5
NEUTRAL_TOLERANCE = 0.5
NEUTRAL_WEIGHTS = (1.0, 4.0, 16.0, 64.0)
SPLIT_SEED = 20261118
TRAIN_CHROMA_FRACTION = 0.6


@dataclass(frozen=True)
class Reading:
    name: str
    balanced: np.ndarray
    target_linear: np.ndarray
    neutral: bool
    source: str  # "patch" or "neutral-region"
    region: dict | None = None

    @property
    def kind(self) -> str:
        return "neutral" if self.neutral else "chroma"


def _reference_neutral(value, width: int, height: int):
    data = object_value(value, "Neutral reference")
    reject_unknown(data, ("name", "region", "srgb"), "neutral reference")
    if "region" not in data:
        raise InputError("Every neutral reference needs a region.")
    if "srgb" not in data:
        raise InputError("Every neutral reference needs an sRGB color.")
    name = text_label(data.get("name", "Neutral region"), "Neutral reference name", 56)
    region = Box.read(data["region"], width, height)
    srgb = np.array(bounded_numbers(data.get("srgb"), "Reference sRGB", 3, 0, 1))
    return name, region, srgb


def read_neutral_references(values, width: int = 0, height: int = 0) -> list[tuple[str, Box, np.ndarray]]:
    if values is None:
        return []
    if not isinstance(values, list) or not 0 <= len(values) <= 16:
        raise InputError("Provide between 0 and 16 neutral references.")
    return [_reference_neutral(item, width, height) for item in values]


def _balanced_median(result: Developed, region: Box) -> np.ndarray | None:
    area = result.box.intersection(region)
    if area is None or area != region:
        return None
    local = area.translated(-result.box.x, -result.box.y)
    valid = area.crop(result.correction.valid) & ~area.crop(result.correction.original_clipped)
    if np.count_nonzero(valid) < max(6, area.area // 2):
        return None
    camera = np.median(local.crop(result.reconstruction.rgb)[valid], axis=0).astype(np.float64)
    white = np.array(result.settings.white_balance, dtype=np.float64)
    gain = float(2 ** result.settings.exposure_ev)
    return camera * white * gain


def _is_neutral(target_linear: np.ndarray) -> bool:
    if float(target_linear.mean()) <= 0.005:
        return False
    lab = working_to_lab(target_linear.astype(np.float32))
    return float(np.hypot(lab[1], lab[2])) < NEUTRAL_CHROMA


def gather_readings(
    result: Developed,
    raw_patches: list,
    neutral_refs: list[tuple[str, Box, np.ndarray]],
) -> tuple[list[Reading], list[dict]]:
    """Collect balanced camera readings for every usable swatch."""
    if not isinstance(raw_patches, list) or not 3 <= len(raw_patches) <= 32:
        raise InputError("Provide between 3 and 32 color patches.")
    readings: list[Reading] = []
    skipped: list[dict] = []
    seen = set()
    for item in raw_patches:
        data = object_value(item, "Patch")
        reject_unknown(data, ("name", "region", "srgb"), "patch")
        name = text_label(data.get("name"), "Patch name", 56)
        if name in seen:
            raise InputError("Give each patch a different name.")
        seen.add(name)
        if "region" not in data:
            raise InputError("Every swatch needs a region.")
        if "srgb" not in data:
            raise InputError("Every swatch needs an sRGB reference color.")
        region = Box.read(data["region"], result.frame.raster.width, result.frame.raster.height, minimum=3)
        srgb = np.array(bounded_numbers(data.get("srgb"), "Reference sRGB", 3, 0, 1))
        balanced = _balanced_median(result, region)
        if balanced is None:
            skipped.append({"name": name, "reason": "Region clips, is invalid, or leaves the developed area."})
            continue
        target = srgb_decode(srgb).astype(np.float64)
        readings.append(Reading(name, balanced, target, _is_neutral(target), "patch", region.as_dict()))

    for name, region, srgb in neutral_refs:
        label = f"{name} (neutral)"
        if label in seen:
            continue
        balanced = _balanced_median(result, region)
        if balanced is None:
            skipped.append({"name": label, "reason": "Neutral region clips, is invalid, or leaves the developed area."})
            continue
        target = srgb_decode(srgb).astype(np.float64)
        readings.append(Reading(label, balanced, target, True, "neutral-region", region.as_dict()))
        seen.add(label)
    return readings, skipped


def _split(readings: list[Reading]) -> tuple[list[Reading], list[Reading]]:
    # Neutral regions supplied in addition to the chart patches are witnesses:
    # they never participate in the fit or candidate selection, only holdout.
    external_neutral = sorted(
        (r for r in readings if r.source == "neutral-region"), key=lambda r: r.name
    )
    chart_rows = [r for r in readings if r.source == "patch"]
    chroma = sorted((r for r in chart_rows if not r.neutral), key=lambda r: r.name)
    neutral = sorted((r for r in chart_rows if r.neutral), key=lambda r: r.name)
    if len(chroma) < 2 * MIN_KIND:
        raise InputError(
            f"Need at least {2 * MIN_KIND} usable colored patches to fit and validate a matrix "
            f"(found {len(chroma)})."
        )
    # Holdout neutrals come from withheld chart grays and any supplied gray
    # regions; the gray scale can only be validated when at least one exists.
    holdout_neutral_count = len(neutral) - max(MIN_KIND, len(neutral) // 2) + len(external_neutral)
    if len(neutral) < MIN_KIND or holdout_neutral_count < MIN_NEUTRAL_REGIONS:
        raise InputError(
            f"Need at least {2 * MIN_KIND} usable neutral chart patches (or gray regions) to fit "
            f"and validate the gray scale (found {len(neutral)} chart neutrals and "
            f"{len(external_neutral)} neutral regions)."
        )
    rng = np.random.default_rng(SPLIT_SEED)
    chroma_order = rng.permutation(len(chroma))
    neutral_order = rng.permutation(len(neutral))
    n_train_chroma = max(MIN_KIND, int(round(len(chroma) * TRAIN_CHROMA_FRACTION)))
    n_train_neutral = max(MIN_KIND, len(neutral) // 2) if neutral else 0
    train_idx_c = set(chroma_order[:n_train_chroma].tolist())
    train_idx_n = set(neutral_order[:n_train_neutral].tolist())
    train = [chroma[i] for i in sorted(train_idx_c)] + [neutral[i] for i in sorted(train_idx_n)]
    holdout = (
        [chroma[i] for i in range(len(chroma)) if i not in train_idx_c]
        + [neutral[i] for i in range(len(neutral)) if i not in train_idx_n]
        + external_neutral
    )
    return train, holdout


def _weighted_lsq(rows: list[Reading], w_neutral: float, w_chroma: float) -> np.ndarray:
    matrix_a, matrix_b, weights = [], [], []
    for row in rows:
        weights.append(w_neutral if row.neutral else w_chroma)
        matrix_a.append(row.balanced)
        matrix_b.append(row.target_linear)
    a = np.asarray(matrix_a, dtype=np.float64)
    b = np.asarray(matrix_b, dtype=np.float64)
    w = np.sqrt(np.asarray(weights, dtype=np.float64))
    aw = a * w[:, None]
    fitted = np.empty((3, 3))
    for channel in range(3):
        solution = lsq_linear(aw, b[:, channel] * w, bounds=(-6.0, 6.0))
        fitted[channel] = solution.x
    return fitted


def _hard_neutral_fit(rows: list[Reading], gray_direction: np.ndarray) -> np.ndarray:
    """Fit with the measured gray direction forced onto the achromatic axis."""
    direction = np.asarray(gray_direction, dtype=np.float64)
    neutral_rows = [row for row in rows if row.neutral]
    level = float(np.mean([row.target_linear.mean() for row in neutral_rows]))
    particular = np.tile((level / (3.0 * direction))[None, :], (3, 1))
    projector = np.eye(3) - np.outer(direction, direction) / np.dot(direction, direction)
    basis = np.linalg.svd(projector, full_matrices=True)[2][:2].T  # 3x2 null space basis

    def residuals(z):
        matrix = particular + basis @ z.reshape(2, 3)
        return np.concatenate([matrix @ row.balanced - row.target_linear for row in rows])

    solution = least_squares(residuals, np.zeros(6), bounds=(-6.0, 6.0), method="trf")
    return particular + basis @ solution.x.reshape(2, 3)


def _valid(matrix: np.ndarray) -> bool:
    try:
        validate_matrix(tuple(np.asarray(matrix, dtype=np.float64).reshape(-1)))
    except InputError:
        return False
    return bool(np.all(np.isfinite(matrix)) and np.max(np.abs(matrix)) <= 6.0)


def _delta_e(matrix: np.ndarray, row: Reading) -> float:
    predicted = np.clip(matrix @ row.balanced, 0.0, None).astype(np.float32)
    return float(ciede2000(working_to_lab(predicted), working_to_lab(row.target_linear.astype(np.float32))))


def _scores(matrix: np.ndarray, rows: list[Reading]) -> dict:
    values = [_delta_e(matrix, row) for row in rows]
    chroma = [v for v, row in zip(values, rows) if not row.neutral]
    neutral = [v for v, row in zip(values, rows) if row.neutral]
    return {
        "chroma_median": float(np.median(chroma)),
        "chroma_max": float(np.max(chroma)),
        "neutral_median": float(np.median(neutral)),
        "neutral_max": float(np.max(neutral)),
        "per_patch": list(zip((row.name for row in rows), values, (row.kind for row in rows))),
    }


def _candidate_matrix(matrix: np.ndarray) -> list[float]:
    return [number(float(value)) for value in np.asarray(matrix, dtype=np.float64).reshape(-1)]


def fit_color_matrix(
    result: Developed,
    raw_patches: list,
    neutral_refs: list[tuple[str, Box, np.ndarray]],
) -> dict:
    readings, skipped = gather_readings(result, raw_patches, neutral_refs)
    if not any(reading.source == "patch" for reading in readings):
        raise InputError("No usable patch readings: every swatch was clipped, invalid, or outside the image.")
    train, holdout = _split(readings)
    current = result.settings.matrix().astype(np.float64)
    train_neutrals = [row for row in train if row.neutral]
    if not train_neutrals:
        raise InputError(
            "No neutral chart patches available for fitting. Include gray swatches in the patch "
            "list so the gray axis can be protected."
        )
    gray_direction = np.mean([row.balanced for row in train_neutrals], axis=0)

    # The ladder starts from the most conservative gray-locked compromise and
    # then relaxes toward chroma-optimal weighted fits.
    ladder: list[tuple[str, np.ndarray, str]] = [("current", current, "Existing matrix, unchanged.")]
    hard = _hard_neutral_fit(train, gray_direction)
    ladder.append((
        "hard-neutral-axis",
        hard,
        "Least squares with the measured gray direction locked to the achromatic axis.",
    ))
    for weight in NEUTRAL_WEIGHTS:
        matrix = _weighted_lsq(train, weight, 1.0)
        ladder.append((
            f"weighted-neutral-{weight:g}",
            matrix,
            f"Weighted least squares with neutral patches weighted {weight:g}x.",
        ))

    baseline_train = _scores(current, train)
    baseline_holdout = _scores(current, holdout)
    candidate_reports = []
    selectable = []
    for label, matrix, note in ladder:
        valid = _valid(matrix)
        entry = {
            "label": label,
            "valid": valid,
            "note": note,
            "matrix": _candidate_matrix(matrix) if valid else None,
        }
        if not valid:
            entry["reason"] = "Fitted matrix failed the camera matrix validity checks."
            candidate_reports.append(entry)
            continue
        train_scores = _scores(matrix, train)
        holdout_scores = _scores(matrix, holdout)
        entry["training"] = {k: round(v, 4) for k, v in train_scores.items() if k != "per_patch"}
        entry["holdout"] = {k: round(v, 4) for k, v in holdout_scores.items() if k != "per_patch"}
        entry["neutral_distance_from_current"] = round(
            float(np.linalg.norm(matrix - current)), 5
        )
        candidate_reports.append(entry)
        if label == "current":
            continue
        # Selection gates on the training/selection set: protect gray first,
        # then require a real chroma benefit.
        neutral_ok = (
            train_scores["neutral_median"] <= baseline_train["neutral_median"] + NEUTRAL_TOLERANCE
            and train_scores["neutral_max"] <= baseline_train["neutral_max"] + NEUTRAL_TOLERANCE
        )
        chroma_ok = (
            baseline_train["chroma_median"] - train_scores["chroma_median"] >= CHROMA_GAIN
            and train_scores["chroma_max"] <= baseline_train["chroma_max"]
        )
        if neutral_ok and chroma_ok:
            selectable.append((label, matrix, train_scores, holdout_scores))

    decision = "kept-current"
    chosen_label = "current"
    chosen_matrix = current
    if selectable:
        # Every eligible candidate already passes the gray-scale gates.  Among
        # them prefer the best chroma fit, breaking ties toward the matrix
        # closest to the current one so the compromise never over-constrains
        # neutrals more than necessary.
        selectable.sort(key=lambda item: (
            item[2]["chroma_median"],
            item[2]["chroma_max"],
            float(np.linalg.norm(item[1] - current)),
        ))
        chosen_label, chosen_matrix, _, chosen_holdout = selectable[0]
        neutral_holds = (
            chosen_holdout["neutral_median"] <= baseline_holdout["neutral_median"] + NEUTRAL_TOLERANCE
            and chosen_holdout["neutral_max"] <= baseline_holdout["neutral_max"] + NEUTRAL_TOLERANCE
        )
        chroma_holds = (
            baseline_holdout["chroma_median"] - chosen_holdout["chroma_median"] >= CHROMA_GAIN
            and chosen_holdout["chroma_max"] <= baseline_holdout["chroma_max"]
        )
        if neutral_holds and chroma_holds:
            decision = "accepted"
            reason = "Holdout patches and gray regions confirm the improvement; the matrix was replaced."
        else:
            decision = "rejected-holdout"
            chosen_label = "current"
            chosen_matrix = current
            reason = (
                "The selected fit helped the fitting patches but did not improve the unseen holdout "
                "patches (or disturbed their gray scale), so the existing matrix was kept."
            )
        if decision != "accepted":
            selectable = []
    if decision == "kept-current":
        reason = (
            "No fitted candidate improved the colored patches by at least "
            f"{CHROMA_GAIN:g} delta E without disturbing the gray scale; the existing matrix was kept."
        )

    final_train = _scores(chosen_matrix, train)
    final_holdout = _scores(chosen_matrix, holdout)

    def patch_rows(scores):
        return [
            {"name": name, "kind": kind, "delta_e2000": round(value, 4)}
            for name, value, kind in scores["per_patch"]
        ]

    return {
        "decision": decision,
        "reason": reason,
        "selected_label": chosen_label,
        "custom_matrix": _candidate_matrix(chosen_matrix),
        "matrix_replaced": decision == "accepted",
        "white_balance": [number(float(v)) for v in result.settings.white_balance],
        "exposure_ev": number(float(result.settings.exposure_ev)),
        "gray_direction": [number(float(v)) for v in gray_direction],
        "split": {
            "seed": SPLIT_SEED,
            "training": [r.name for r in train],
            "holdout": [r.name for r in holdout],
        },
        "baseline": {
            "label": "current",
            "training": {k: round(v, 4) for k, v in baseline_train.items() if k != "per_patch"},
            "holdout": {k: round(v, 4) for k, v in baseline_holdout.items() if k != "per_patch"},
        },
        "candidates": candidate_reports,
        "final": {
            "label": chosen_label,
            "training": {k: round(v, 4) for k, v in final_train.items() if k != "per_patch"},
            "holdout": {k: round(v, 4) for k, v in final_holdout.items() if k != "per_patch"},
            "training_patches": patch_rows(final_train),
            "holdout_patches": patch_rows(final_holdout),
        },
        "skipped": skipped,
        "gates": {
            "neutral_chroma_limit": NEUTRAL_CHROMA,
            "chroma_gain_delta_e": CHROMA_GAIN,
            "neutral_tolerance_delta_e": NEUTRAL_TOLERANCE,
        },
    }


def color_target_neutral_references() -> list[tuple[str, Box, np.ndarray]]:
    """Built-in neutral card and gray ramp references for the synthetic target."""
    from .synthetic import RAMP_STOPS  # local import: synthetic imports development too

    refs = [("Neutral card", Box(71, 280, 118, 29), 163 / 255)]
    for index, stop in enumerate(RAMP_STOPS):
        refs.append((f"Gray ramp {index + 1}", Box(472, 98 + 30 * index, 25, 23), stop / 255))
    out = []
    for name, box, srgb in refs:
        goal = srgb_encode(srgb_decode(np.array([srgb, srgb, srgb], np.float32)) * 0.86)
        out.append((name, box, goal))
    return out
