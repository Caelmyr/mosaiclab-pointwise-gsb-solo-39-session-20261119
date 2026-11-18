"""Versioned archives that reopen an inspection with its original raw samples."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, replace

from .artifacts import Artifact, json_bytes
from .codecs import MAX_HEADER_BYTES, DecodedCapture, decode_pgm, digest_bytes, encode_pgm, json_object
from .development import DevelopSettings
from .domain import (
    MAX_PIXELS, ROLE_VALUES, Frame, InputError, choice, flag, integer, object_value,
    optional_id, reject_unknown, text_label,
)
from .storage import Catalog, MAX_FRAMES, MAX_RECIPES, MAX_SAVES

LEGACY_ARCHIVE_VERSION = "mosaiclab.workbench/1"
ARCHIVE_VERSION = "mosaiclab.workbench/2"
MAX_ARCHIVE_BYTES = 96 * 1024 * 1024
MAX_MANIFEST = 768 * 1024
RAW_MEMBER_BOUND = 2 * MAX_PIXELS + MAX_HEADER_BYTES


@dataclass(frozen=True)
class ArchivedFrame:
    old_id: str
    capture: DecodedCapture
    role: str
    original_name: str
    name: str


@dataclass(frozen=True)
class ArchivedProject:
    name: str
    notes: str
    demo_key: str | None
    frames: list[ArchivedFrame]
    sessions: list[dict]
    recipes: list[dict]
    selection: dict


def _capture_entry(frame: Frame, raw: bytes) -> dict:
    return {
        "id": frame.id,
        "member": f"mosaics/{frame.id}.pgm",
        "name": frame.name,
        "role": frame.role,
        "original_name": frame.original_name,
        "source_digest": frame.source_digest,
        "sample_checksum": digest_bytes(raw),
        "sample_bytes": len(raw),
        "encoding": frame.raster.encoding,
        "encoding_notes": list(frame.raster.warnings),
        "sensor": frame.sensor.as_dict(),
        "synthetic": frame.synthetic,
    }


def archive_project(catalog: Catalog, project_id: str) -> Artifact:
    project = catalog.describe(project_id)
    total = sum(frame["width"] * frame["height"] * 2 + 128 for frame in project["frames"])
    if total > MAX_ARCHIVE_BYTES:
        raise InputError("Workbench is too large for a portable archive. Export selected captures.")
    stamp = {
        "format": ARCHIVE_VERSION,
        "name": project["name"],
        "notes": project["notes"],
        "demo_key": project["demo_key"],
        "recipes": [
            {"title": obj["title"], "settings": obj["settings"]}
            for obj in project["recipes"]
        ],
        "sessions": [
            {"title": obj["title"], "state": obj["state"]}
            for obj in project["sessions"]
        ],
        "selection": project["selection"],
        "frames": [],
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for record in project["frames"]:
            frame = catalog.load_frame(project_id, record["id"])
            body = encode_pgm(frame.raster.values, frame.raster.maximum)
            entry = _capture_entry(frame, body)
            stamp["frames"].append(entry)
            bundle.writestr(entry["member"], body)
        bundle.writestr("workbench.json", json_bytes(stamp))
    checksum_notice = "mosaiclab-workbench.zip"
    if stream.tell() > MAX_ARCHIVE_BYTES:
        raise InputError("Workbench is too large for a portable archive.")
    return Artifact(checksum_notice, stream.getvalue(), "application/zip")


def _limited_objects(value, count_limit: int, label: str) -> list:
    if not isinstance(value, list) or len(value) > count_limit:
        raise InputError(f"{label} must have at most {count_limit} items.")
    return value


def _required_id(raw, label: str) -> str:
    value = optional_id(raw, label)
    if value is None:
        raise InputError(f"Archive is missing {label.lower()}.")
    return value


def _checksum(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise InputError(f"{label} is not a SHA-256 digest.")
    if any(char not in "0123456789abcdef" for char in value):
        raise InputError(f"{label} is not a SHA-256 digest.")
    return value


def _read_member(bundle: zipfile.ZipFile, name: str, bound: int) -> bytes:
    try:
        entry = bundle.getinfo(name)
    except KeyError as exc:
        raise InputError(f"Archive is missing {name}.") from exc
    if entry.file_size > bound or entry.flag_bits & 1 or entry.is_dir():
        raise InputError(f"Invalid or overlarge archive entry {name}.")
    if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise InputError("Archive compression is unsupported.")
    body = bundle.read(entry)
    if len(body) != entry.file_size:
        raise InputError(f"Archive entry {name} was truncated.")
    return body


def _read_capture(bundle: zipfile.ZipFile, raw, *, keep_encoding: bool) -> ArchivedFrame:
    item = object_value(raw, "Archived capture")
    fields = ("id", "member", "name", "role", "original_name", "source_digest",
              "sample_checksum", "sample_bytes", "sensor", "synthetic")
    if keep_encoding:
        fields += ("encoding", "encoding_notes")
    reject_unknown(item, fields, "archived capture")
    old = _required_id(item.get("id"), "Capture identifier")
    member = f"mosaics/{old}.pgm"
    if item.get("member") != member:
        raise InputError("Archive capture path does not match its identifier.")
    expected_size = integer(item.get("sample_bytes"), "Sample byte length", 1, RAW_MEMBER_BOUND)
    body = _read_member(bundle, member, expected_size)
    if digest_bytes(body) != _checksum(item.get("sample_checksum"), "Sample checksum"):
        raise InputError("An archived capture failed its checksum.")
    if len(body) != expected_size:
        raise InputError("An archived capture does not match its declared size.")
    raster = decode_pgm(body)
    if keep_encoding:
        encoding = text_label(item.get("encoding"), "Original encoding", 80)
        notes = _limited_objects(item.get("encoding_notes"), 12, "Original encoding notes")
        raster = replace(raster, encoding=encoding, warnings=tuple(
            text_label(note, "Original encoding note", 220) for note in notes
        ))
    sensor = dict(object_value(item.get("sensor"), "Archived sensor"))
    name = text_label(item.get("name"), "Archived capture name")
    provenance = _checksum(item.get("source_digest"), "Source checksum")
    synthetic = flag(item.get("synthetic", False), "Synthetic capture")
    decoded = DecodedCapture(raster, sensor, name, provenance, synthetic)
    decoded.sensor()
    return ArchivedFrame(
        old,
        decoded,
        choice(item.get("role"), "Archived capture role", ROLE_VALUES),
        text_label(item.get("original_name"), "Original filename", 160),
        name,
    )


def _recipes(value) -> list[dict]:
    results = []
    for raw in _limited_objects(value, MAX_RECIPES, "Archived recipes"):
        item = object_value(raw, "Archived recipe")
        reject_unknown(item, ("title", "settings"), "archived recipe")
        settings = DevelopSettings.read(item.get("settings")).as_dict()
        settings.pop("calibration", None)
        results.append({"title": text_label(item.get("title"), "Recipe title"), "settings": settings})
    return results


def _sessions(value) -> list[dict]:
    results = []
    for raw in _limited_objects(value, MAX_SAVES, "Archived inspections"):
        item = object_value(raw, "Archived inspection")
        reject_unknown(item, ("title", "state"), "archived inspection")
        results.append({
            "title": text_label(item.get("title"), "Inspection title"),
            "state": dict(object_value(item.get("state"), "Archived inspection state")),
        })
    return results


def _decode_archive(bundle: zipfile.ZipFile) -> ArchivedProject:
    members = bundle.infolist()
    names = {record.filename for record in members}
    if len(members) != len(names) or len(names) > MAX_FRAMES + 1:
        raise InputError("Archive contains invalid or repeated member names.")
    if sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES + MAX_MANIFEST:
        raise InputError("The workbench archive exceeds the uncompressed size limit.")
    meta = json_object(_read_member(bundle, "workbench.json", MAX_MANIFEST), "Workbench manifest")
    reject_unknown(
        meta,
        ("format", "name", "notes", "demo_key", "recipes", "sessions", "selection", "frames"),
        "workbench manifest",
    )
    version = choice(meta.get("format"), "Workbench archive version", (LEGACY_ARCHIVE_VERSION, ARCHIVE_VERSION))
    raw_frames = _limited_objects(meta.get("frames"), MAX_FRAMES, "Archived captures")
    frames = [_read_capture(bundle, raw, keep_encoding=version == ARCHIVE_VERSION) for raw in raw_frames]
    if names != {"workbench.json", *(f"mosaics/{item.old_id}.pgm" for item in frames)}:
        raise InputError("Workbench archive contains unlisted image data.")
    if len(frames) != len(names) - 1:
        raise InputError("A capture identifier is used more than once.")
    notes = meta.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 500:
        raise InputError("Archive notes exceed 500 characters.")
    demo = meta.get("demo_key")
    if demo is not None and not isinstance(demo, str):
        raise InputError("Invalid synthetic scene key.")
    return ArchivedProject(
        text_label(meta.get("name"), "Archived workbench name"), notes, demo,
        frames, _sessions(meta.get("sessions")), _recipes(meta.get("recipes")),
        dict(object_value(meta.get("selection"), "Archived selection")),
    )


def decode_archive(body: bytes) -> ArchivedProject:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as source:
            return _decode_archive(source)
    except (OSError, zipfile.BadZipFile, NotImplementedError, RuntimeError) as error:
        raise InputError("Could not read the workbench archive.") from error


def _mapped_id(value, mapping: dict[str, str], label: str) -> str | None:
    checked = optional_id(value, label)
    if checked is None:
        return None
    if checked not in mapping:
        raise InputError(f"Archived {label.lower()} has no corresponding capture.")
    return mapping[checked]


def remap_saved_state(state: dict, mapping: dict[str, str]) -> dict:
    result = {**state}
    for key, label in (("frame_id", "Capture"), ("comparison_frame_id", "Comparison capture")):
        if key in result:
            result[key] = _mapped_id(result[key], mapping, label)
    if "measurement_options" in result:
        options = dict(object_value(result["measurement_options"], "Archived measurement options"))
        if "pair_id" in options:
            options["pair_id"] = _mapped_id(options["pair_id"], mapping, "Repeat capture")
        result["measurement_options"] = options
    for branch in ("settings", "comparison_settings"):
        if branch not in result:
            continue
        raw = dict(object_value(result[branch], "Archived develop settings"))
        calibration = dict(object_value(raw.get("calibration", {}), "Archived calibration"))
        for reference in ("dark_id", "flat_id", "defects_id"):
            if reference in calibration:
                calibration[reference] = _mapped_id(calibration[reference], mapping, reference)
        raw["calibration"] = calibration
        result[branch] = raw
    return result
