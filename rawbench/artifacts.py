"""Download formats, measured reports and portable copies of raw selections."""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

import numpy as np
from PIL import Image

from .analysis import describe_values
from .cfa import remap_raw_crop, rotate_raw_clockwise
from .codecs import RAW_MODES, digest_bytes, encode_pgm, encode_portable
from .comparison import Comparison
from .development import Developed, quantize
from .domain import Box, Frame, InputError, Raster, Sensor, choice, integer, object_value, reject_unknown
from .icc import display_p3_icc
from .view import png_rgb16

OUTPUT_TYPES = ("png8", "png16", "linear16", "npz", "raw", "report", "review-zip")
OPTICAL_CSV_FIELDS = ("distance", "edge_spread", "line_spread", "sample_count")
MEDIA = {
    "png": "image/png",
    "zip": "application/zip",
    "npz": "application/octet-stream",
    "json": "application/json",
    "csv": "text/csv",
}


@dataclass(frozen=True)
class Artifact:
    filename: str
    data: bytes
    mimetype: str


def _filename(name: str, suffix: str, detail: str = "") -> str:
    stem = PurePath(name).stem
    stem = re.sub(r"[\s/\\.:]+", "-", stem).strip("-")
    stem = "".join(letter for letter in stem if letter.isalnum() or letter in "-_")
    return f"{(stem or 'capture')[:55]}{('-' + detail) if detail else ''}.{suffix}"


def _artifact(name: str, suffix: str, contents: bytes, detail: str = "") -> Artifact:
    return Artifact(_filename(name, suffix, detail), contents, MEDIA[suffix])


def json_bytes(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _crop_for_result(result: Developed, selected: Box | None) -> tuple[Box, Box]:
    box = result.box if selected is None else result.box.intersection(selected)
    if box is None:
        raise InputError("Exported region must overlap the developed image.")
    return box, box.translated(-result.box.x, -result.box.y)


def _encode8(values: np.ndarray, profile: str) -> bytes:
    image = Image.fromarray(quantize(values))
    output = io.BytesIO()
    options = {}
    if profile == "display-p3":
        options["icc_profile"] = display_p3_icc()
    image.save(output, "PNG", compress_level=5, **options)
    return output.getvalue()


def _encode_npz(result: Developed, selected: Box, local: Box) -> bytes:
    data = io.BytesIO()
    np.savez_compressed(
        data,
        linear_rgb=local.crop(result.working_rgb),
        camera_rgb=local.crop(result.reconstruction.rgb),
        raw_correction=selected.crop(result.correction.linear),
        sample_is_valid=selected.crop(result.correction.valid),
        sample_is_repaired=selected.crop(result.correction.repaired),
        original_saturation=selected.crop(result.correction.original_clipped),
        raster_origin=np.array([selected.x, selected.y], dtype=np.int32),
        sensor_origin=np.array([
            selected.x + result.frame.sensor.layout.origin_x,
            selected.y + result.frame.sensor.layout.origin_y,
        ], dtype=np.int32),
    )
    return data.getvalue()


def development_manifest(result: Developed, selected: Box, local: Box) -> dict:
    origin = result.frame.sensor.layout
    source = {
        "name": result.frame.name,
        "frame_id": result.frame.id,
        "original_file_sha256": result.frame.source_digest,
        "synthetic": result.frame.synthetic,
        "sensor": result.frame.sensor.as_dict(),
        "raster_box": selected.as_dict(),
        "sensor_box": selected.translated(origin.origin_x, origin.origin_y).as_dict(),
    }
    pixels = local.crop(result.working_rgb)
    bounds = {"below_zero": int(np.count_nonzero(pixels < 0)), "above_one": int(np.count_nonzero(pixels > 1))}
    return {
        "format": "mosaiclab.developed/1",
        "source": source,
        "settings": result.settings.as_dict(),
        "development": result.report,
        "exported_region": {
            "working_linear_rgb": [describe_values(pixels[..., idx]) for idx in range(3)],
            "linear_working_range": bounds,
            "raw_valid_sites": int(np.count_nonzero(selected.crop(result.correction.valid))),
            "raw_repaired_sites": int(np.count_nonzero(selected.crop(result.correction.repaired))),
        },
    }


def _bundle(members: dict[str, bytes], manifest: dict) -> bytes:
    summary = {
        **manifest,
        "members": {
            name: {"bytes": len(payload), "sha256": digest_bytes(payload)}
            for name, payload in members.items()
        },
    }
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as out:
        for name, payload in members.items():
            out.writestr(name, payload)
        out.writestr("manifest.json", json_bytes(summary))
    return content.getvalue()


def development_artifact(result: Developed, kind: str, selected: Box | None) -> Artifact:
    sensor_region, local = _crop_for_result(result, selected)
    name = result.frame.name
    profile = result.settings.profile
    rgb = local.crop(result.encoded)
    if kind == "png8":
        return _artifact(name, "png", _encode8(rgb, profile), "developed")
    if kind == "png16":
        icc = display_p3_icc() if profile == "display-p3" else None
        return _artifact(name, "png", png_rgb16(rgb, icc_profile=icc), "rgb16")
    if kind == "linear16":
        data = png_rgb16(local.crop(result.working_rgb), linear=True)
        return _artifact(name, "png", data, "linear16")
    if kind == "npz":
        return _artifact(name, "npz", _encode_npz(result, sensor_region, local), "linear")
    manifest = development_manifest(result, sensor_region, local)
    if kind == "report":
        return _artifact(name, "json", json_bytes(manifest), "report")
    if kind == "review-zip":
        items = {
            "developed.png": _encode8(rgb, profile),
            "source.pgm": encode_pgm(sensor_region.crop(result.frame.raster.values), result.frame.raster.maximum),
        }
        return _artifact(name, "zip", _bundle(items, manifest), "review")
    raise InputError("Unsupported development download.")


def _rotated_levels(sensor: Sensor, values: np.ndarray, box: Box, rotation: int) -> tuple:
    tile = sensor.layout.shifted(box.x, box.y).local_index_grid()
    mapping = np.tile(tile, (2, 2))[:2 + values.shape[0] % 2, :2 + values.shape[1] % 2]
    rotated = np.rot90(mapping, -(rotation % 4))
    order = rotated[:2, :2].ravel()
    return tuple(sensor.black[phase] for phase in order), tuple(sensor.white[phase] for phase in order)


def _rotated_box(box: Box, width: int, height: int, turns: int) -> Box:
    angle = turns % 4
    if angle == 1:
        return Box(height - box.bottom, box.x, box.height, box.width)
    if angle == 2:
        return Box(width - box.right, height - box.bottom, box.width, box.height)
    if angle == 3:
        return Box(box.y, width - box.right, box.height, box.width)
    return box


def raw_artifact(
    frame: Frame, selected: Box | None, rotation: int = 0,
    *, encoding: str | None = None, alignment: int = 1,
) -> Artifact:
    values = frame.raster.values
    capture_box = selected or Box.full(frame.raster.width, frame.raster.height)
    if capture_box.width < 2 or capture_box.height < 2:
        raise InputError("Raw export needs at least two rows and columns.")
    active = capture_box.intersection(frame.sensor.active)
    if active is None or min(active.width, active.height) < 2:
        raise InputError("Raw export must include at least a 2×2 portion of the active sensor.")
    active = active.translated(-capture_box.x, -capture_box.y)
    image, layout = remap_raw_crop(values, frame.sensor.layout, capture_box)
    black, white = frame.sensor.black, frame.sensor.white
    if rotation:
        black, white = _rotated_levels(frame.sensor, image, capture_box, rotation)
        active = _rotated_box(active, image.shape[1], image.shape[0], rotation)
        image, layout = rotate_raw_clockwise(image, layout, rotation)
    sensor = Sensor(
        layout, black, white, active, frame.sensor.exposure_ms,
        frame.sensor.analog_gain, frame.sensor.model,
    )
    raster = Raster(image, frame.raster.maximum, frame.raster.encoding)
    label = "raw" if rotation == 0 else f"raw-{rotation * 90}deg"
    if encoding is not None:
        label = f"{label}-{encoding}"
    data = encode_portable(raster, sensor, frame.name, frame.synthetic, encoding=encoding, alignment=alignment)
    return _artifact(frame.name, "zip", data, label)


def comparison_artifact(result: Comparison, name: str, *, report_only: bool = False) -> Artifact:
    if report_only:
        return _artifact(name, "json", json_bytes(result.report), "difference")
    members = {
        "comparison.png": _encode8(result.view, "srgb"),
        "left.png": _encode8(result.a, "srgb"),
        "right.png": _encode8(result.b, "srgb"),
        "difference.json": json_bytes(result.report),
    }
    return _artifact(name, "zip", _bundle(members, {"format": "mosaiclab.comparison/1"}), "comparison")


def _format_cell(value):
    if value is None:
        return ""
    return value


def _write_color_patch_rows(writer, patches: list[dict]) -> None:
    writer.writerow([
        "patch", "raster_x", "raster_y", "width", "height", "usable_samples", "status",
        "target_L", "target_a", "target_b", "measured_L", "measured_a", "measured_b",
        "delta_e_2000", "p90_delta_e", "p90_within_patch_delta_e", "over_display_gamut",
    ])
    for patch in patches:
        region = patch["region"]
        name = patch["name"]
        if name.lstrip().startswith(("=", "+", "-", "@")):
            name = "'" + name
        target = patch.get("target_lab_d65", (None, None, None))
        measured = patch.get("measured_lab_d65", (None, None, None))
        writer.writerow([
            name, region["x"], region["y"], region["width"], region["height"],
            patch["samples"], patch.get("reason", "usable"), *target, *measured,
            patch.get("delta_e2000"), patch.get("p90_delta_e"), patch.get("p90_within_patch_delta_e"),
            patch.get("over_display_gamut"),
        ])


def measurement_csv(kind: str, report: dict, name: str) -> Artifact:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    if kind == "edge":
        writer.writerow(OPTICAL_CSV_FIELDS)
        for i, x in enumerate(report["edge_positions"]):
            line = report["line_spread"][i] if i < len(report["line_spread"]) else None
            writer.writerow([x, report["edge_spread"][i], _format_cell(line), report["site_counts"][i]])
    elif kind == "chromatic":
        writer.writerow(["phase", "distance_from_green_px", "normalized_response", "samples", "phase_offset_px"])
        for phase in report["phases"]:
            for distance, response, sites in zip(report["positions_px"], phase["response"], phase["site_counts"]):
                writer.writerow([phase["phase"], distance, response, sites, phase["offset_px"]])
    elif kind == "histogram":
        writer.writerow(["phase", "left_edge", "right_edge", "samples"])
        for channel in report["phases"]:
            for i, total in enumerate(channel["counts"]):
                writer.writerow([channel["phase"], channel["edges"][i], channel["edges"][i + 1], total])
    elif kind == "profile":
        writer.writerow(["phase", "sensor_coordinate", "median", "robust_sigma", "samples"])
        for line in report["traces"]:
            for i, x in enumerate(line["positions"]):
                writer.writerow([line["phase"], x, line["medians"][i], line["robust_sigma"][i], line["samples"][i]])
    elif kind == "banding":
        writer.writerow(["phase", "period_sites", "cycles_per_site", "amplitude_code", "above_floor_db"])
        for phase in report["phases"]:
            for peak in phase["peak_periods"]:
                writer.writerow([
                    phase["phase"], peak["period_sites"], peak["cycles_per_sensor_site"],
                    peak["amplitude"], peak["relative_to_floor_db"],
                ])
    elif kind == "mask-audit":
        writer.writerow(["type", "sensor_x", "sensor_y", "width", "height", "marked", "coverage", "on_boundary"])
        for group in report["groups"]["shown"]:
            bounds = group["sensor_box"]
            writer.writerow([
                "group", bounds["x"], bounds["y"], bounds["width"], bounds["height"],
                group["sites"], group["occupancy"], group["touches_region_boundary"],
            ])
        for axis in ("rows", "columns"):
            for line in report[axis]["shown"]:
                across = line["run_begins_at"]
                along = line["sensor_line"]
                length = line["longest_run"]
                x, y, width, height = (across, along, length, 1) if axis == "rows" else (along, across, 1, length)
                writer.writerow([axis[:-1], x, y, width, height, line["marked"], line["fraction"], ""])
    elif kind == "color-audit":
        _write_color_patch_rows(writer, report["patches"])
    else:
        raise InputError("This measurement is available as JSON only.")
    return _artifact(name, "csv", stream.getvalue().encode("utf-8-sig"), kind)


@dataclass(frozen=True)
class DownloadOptions:
    kind: str
    rotation: int
    crop_region: bool
    raw_encoding: str | None
    raw_alignment: int

    @classmethod
    def read(cls, value) -> DownloadOptions:
        obj = object_value(value, "Download options")
        reject_unknown(obj, ("kind", "rotation", "crop_region", "raw_encoding", "raw_alignment"), "download")
        crop = obj.get("crop_region", False)
        if not isinstance(crop, bool):
            raise InputError("Crop region must be true or false.")
        kind = choice(obj.get("kind", "png8"), "Download format", OUTPUT_TYPES)
        rotation = integer(obj.get("rotation", 0), "Raw rotation", 0, 3)
        raw_encoding = obj.get("raw_encoding")
        if raw_encoding is not None:
            raw_encoding = choice(raw_encoding, "Binary encoding", tuple(RAW_MODES))
        alignment = integer(obj.get("raw_alignment", 1), "Row alignment", 1, 64)
        if alignment not in (1, 16, 32, 64):
            raise InputError("Row alignment must be 1, 16, 32 or 64 bytes.")
        if kind != "raw" and (rotation or raw_encoding is not None):
            raise InputError("RAW rotation and binary encoding only apply to a portable raw capture.")
        if alignment != 1 and raw_encoding is None:
            raise InputError("Row alignment only applies to a binary raw encoding.")
        return cls(kind, rotation, crop, raw_encoding, alignment)
