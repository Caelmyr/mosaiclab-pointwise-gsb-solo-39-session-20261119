"""SQLite catalog and immutable, file-backed sensor samples."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .codecs import DecodedCapture
from .domain import (
    CapacityError, ConflictError, Frame, InputError, MissingError, Raster, Sensor,
    object_value, optional_id, text_label,
)

MAX_PROJECTS = 32
MAX_FRAMES = 30
MAX_SAVES = 75
MAX_RECIPES = 40
MAX_STORAGE_BYTES = 640 * 1024 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    demo_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    selection_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS assets (
    digest TEXT PRIMARY KEY,
    file_name TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    stored_bytes INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    original_name TEXT NOT NULL,
    role TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    asset_digest TEXT NOT NULL REFERENCES assets(digest) ON DELETE RESTRICT,
    code_maximum INTEGER NOT NULL,
    encoding TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    sensor_json TEXT NOT NULL,
    synthetic INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_by_project ON frames(project_id, created_at, id);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    frame_id TEXT NOT NULL REFERENCES frames(id) ON DELETE RESTRICT,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_by_project ON sessions(project_id, created_at, id);
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS recipes_by_project ON recipes(project_id, created_at, id);
"""


def _unique() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _as_json(row: sqlite3.Row, key: str):
    return json.loads(row[key])


def _identifier(value: str, name: str) -> str:
    result = optional_id(value, name)
    if result is None:
        raise InputError(f"Select a {name.lower()}.")
    return result


class Catalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.assets = self.root / "mosaics"
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets.mkdir(exist_ok=True)
        self.database = self.root / "catalog.sqlite3"
        conn = sqlite3.connect(self.database, isolation_level=None, timeout=15)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()
        self._prune_orphans()

    @contextmanager
    def connect(self, *, write: bool = False):
        conn = sqlite3.connect(self.database, isolation_level=None, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _project(self, conn: sqlite3.Connection, project_id: str) -> sqlite3.Row:
        key = _identifier(project_id, "Workbench")
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (key,)).fetchone()
        if row is None:
            raise MissingError("Workbench not found.")
        return row

    def _frame_row(
        self, conn: sqlite3.Connection, project_id: str, frame_id: str
    ) -> sqlite3.Row:
        key = _identifier(frame_id, "Capture")
        row = conn.execute(
            """
            SELECT frames.*, assets.width, assets.height, assets.file_name
            FROM frames JOIN assets ON assets.digest = frames.asset_digest
            WHERE frames.id = ? AND frames.project_id = ?
            """,
            (key, project_id),
        ).fetchone()
        if row is None:
            raise MissingError("Capture not found in this workbench.")
        return row

    def _public_project(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "notes": row["notes"],
            "demo_key": row["demo_key"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "revision": row["revision"],
            "selection": _as_json(row, "selection_json"),
        }

    def _public_frame(self, row: sqlite3.Row) -> dict:
        sensor = self._sensor(row)
        return {
            "id": row["id"],
            "project": row["project_id"],
            "name": row["name"],
            "original_name": row["original_name"],
            "role": row["role"],
            "source_digest": row["source_digest"],
            "width": row["width"],
            "height": row["height"],
            "maximum": row["code_maximum"],
            "encoding": row["encoding"],
            "warnings": _as_json(row, "warnings_json"),
            "sensor": sensor.as_dict(),
            "local_pattern": sensor.layout.local_pattern(),
            "synthetic": bool(row["synthetic"]),
        }

    def list_projects(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT projects.*, (SELECT count(*) FROM frames
                    WHERE project_id = projects.id) AS frame_count
                FROM projects ORDER BY updated_at DESC, id
                """
            )
            result = []
            for row in rows:
                project = self._public_project(row)
                project["frame_count"] = row["frame_count"]
                result.append(project)
            return result

    def describe(self, project_id: str) -> dict:
        with self.connect() as conn:
            project = self._public_project(self._project(conn, project_id))
            rows = conn.execute(
                """
                SELECT frames.*, assets.width, assets.height FROM frames
                JOIN assets ON assets.digest = frames.asset_digest
                WHERE project_id = ? ORDER BY created_at, frames.id
                """,
                (project_id,),
            )
            project["frames"] = [self._public_frame(row) for row in rows]
            project["sessions"] = self._history(conn, project_id)
            project["recipes"] = self._recipes(conn, project_id)
            return project

    def create_project(
        self,
        name: str,
        *,
        demo_key: str | None = None,
        notes: str = "",
        captures: Iterable[tuple[str, DecodedCapture]] = (),
    ) -> dict:
        name = text_label(name, "Workbench name")
        identifier = _unique()
        with self.connect(write=True) as conn:
            count = conn.execute("SELECT count(*) FROM projects").fetchone()[0]
            if count >= MAX_PROJECTS:
                raise CapacityError(f"Keep at most {MAX_PROJECTS} workbenches.")
            stamp = _now()
            conn.execute(
                """INSERT INTO projects
                (id, name, notes, demo_key, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (identifier, name, notes, demo_key, stamp, stamp),
            )
            first = None
            for role, decoded in captures:
                frame = self._attach(
                    conn, identifier, decoded, decoded.suggested_name, None,
                    role, f"{decoded.suggested_name}.pgm",
                )
                if first is None and role == "scene":
                    first = frame
            if first:
                self._touch(conn, identifier, selection={"frame_id": first})
        return self.describe(identifier)

    def update_project(self, project_id: str, name: str, notes: str) -> dict:
        name = text_label(name, "Workbench name")
        if not isinstance(notes, str) or len(notes) > 500:
            raise InputError("Notes must be text of at most 500 characters.")
        with self.connect(write=True) as conn:
            self._project(conn, project_id)
            conn.execute("UPDATE projects SET name = ?, notes = ? WHERE id = ?", (name, notes, project_id))
            self._touch(conn, project_id)
        return self.describe(project_id)

    def set_selection(self, project_id: str, state: dict, revision: int) -> dict:
        with self.connect(write=True) as conn:
            row = self._project(conn, project_id)
            if row["revision"] != revision:
                raise ConflictError("This workbench changed. Reload it before saving the current selection.")
            if state.get("frame_id"):
                self._frame_row(conn, project_id, state["frame_id"])
            self._touch(conn, project_id, selection=state)
            return self._public_project(self._project(conn, project_id))

    def _touch(self, conn: sqlite3.Connection, project_id: str, selection: dict | None = None) -> None:
        now = _now()
        if selection is None:
            conn.execute(
                "UPDATE projects SET updated_at = ?, revision = revision + 1 WHERE id = ?",
                (now, project_id),
            )
        else:
            conn.execute(
                """UPDATE projects SET updated_at = ?, revision = revision + 1,
                   selection_json = ? WHERE id = ?""",
                (now, _json(selection), project_id),
            )

    def add_frame(
        self,
        project_id: str,
        decoded: DecodedCapture,
        name: str,
        sensor: dict | None,
        role: str,
        original: str,
    ) -> dict:
        with self.connect(write=True) as conn:
            project = self._project(conn, project_id)
            identifier = self._attach(conn, project_id, decoded, name, sensor, role, original)
            if not _as_json(project, "selection_json").get("frame_id") and role == "scene":
                self._touch(conn, project_id, selection={"frame_id": identifier})
            else:
                self._touch(conn, project_id)
            return self._public_frame(self._frame_row(conn, project_id, identifier))

    def _attach(
        self, conn, project_id: str, decoded: DecodedCapture, name: str,
        sensor: dict | None, role: str, original: str,
    ) -> str:
        total = conn.execute("SELECT count(*) FROM frames WHERE project_id = ?", (project_id,)).fetchone()[0]
        if total >= MAX_FRAMES:
            raise CapacityError(f"A workbench can contain up to {MAX_FRAMES} captures.")
        name = text_label(name, "Capture name")
        parsed_sensor = decoded.sensor(sensor)
        asset = self._store_asset(conn, decoded.raster.values)
        identifier = _unique()
        conn.execute(
            """INSERT INTO frames
            (id, project_id, name, original_name, role, source_digest, asset_digest,
             code_maximum, encoding, warnings_json, sensor_json, synthetic, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                identifier, project_id, name, original, role, decoded.digest, asset,
                decoded.raster.maximum, decoded.raster.encoding, _json(decoded.raster.warnings),
                _json(parsed_sensor.as_dict()), int(decoded.synthetic), _now(),
            ),
        )
        return identifier

    def _store_asset(self, conn: sqlite3.Connection, values: np.ndarray) -> str:
        raw = np.ascontiguousarray(values, dtype="<u2")
        height, width = raw.shape
        ingester = hashlib.sha256(f"{width}x{height}:".encode("ascii"))
        ingester.update(memoryview(raw).cast("B"))
        digest = ingester.hexdigest()
        record = conn.execute("SELECT file_name FROM assets WHERE digest = ?", (digest,)).fetchone()
        if record is not None:
            if not (self.assets / record["file_name"]).is_file():
                raise ConflictError("An indexed capture file is missing.")
            return digest
        total = conn.execute("SELECT COALESCE(sum(stored_bytes), 0) FROM assets").fetchone()[0]
        if total + raw.nbytes > MAX_STORAGE_BYTES:
            raise CapacityError("The local capture store has reached its 640 MiB limit.")
        filename = f"{digest}.u16"
        final = self.assets / filename
        temporary = self.assets / f".{digest}.{_unique()}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(memoryview(raw).cast("B"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final)
            descriptor = os.open(self.assets, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        conn.execute(
            "INSERT INTO assets VALUES (?, ?, ?, ?, ?)",
            (digest, filename, width, height, raw.nbytes),
        )
        return digest

    def _sensor(self, row: sqlite3.Row) -> Sensor:
        return Sensor.from_values(
            _as_json(row, "sensor_json"), row["width"], row["height"], row["code_maximum"]
        )

    def load_frame(self, project_id: str, frame_id: str) -> Frame:
        with self.connect() as conn:
            row = self._frame_row(conn, project_id, frame_id)
            raw_path = self.assets / row["file_name"]
            width, height = row["width"], row["height"]
            count = width * height
            try:
                if raw_path.stat().st_size != count * 2:
                    raise ConflictError("This capture's sample file has an unexpected length.")
                array = np.fromfile(raw_path, dtype="<u2", count=count)
            except OSError as exc:
                raise ConflictError("This capture's sample file is unavailable.") from exc
            if array.size != count:
                raise ConflictError("This capture's sample file is incomplete.")
            return Frame(
                row["id"], row["project_id"], row["name"], row["original_name"],
                row["role"], row["source_digest"],
                Raster(
                    array.reshape(height, width).astype(np.uint16, copy=False),
                    row["code_maximum"], row["encoding"],
                    tuple(_as_json(row, "warnings_json")),
                ),
                self._sensor(row), bool(row["synthetic"]),
            )

    def set_frame_metadata(
        self, project_id: str, frame_id: str, name: str, metadata: dict
    ) -> dict:
        name = text_label(name, "Capture name")
        with self.connect(write=True) as conn:
            row = self._frame_row(conn, project_id, frame_id)
            sensor = Sensor.from_values(
                {**_as_json(row, "sensor_json"), **metadata},
                row["width"], row["height"], row["code_maximum"],
            )
            for session in self._history(conn, project_id):
                keys = self._state_references(session["state"])
                if frame_id in keys and sensor != self._sensor(row):
                    raise ConflictError(
                        "A saved inspection depends on this sensor metadata. "
                        "Remove that inspection before changing it."
                    )
            conn.execute(
                "UPDATE frames SET name = ?, sensor_json = ? WHERE id = ?",
                (name, _json(sensor.as_dict()), frame_id),
            )
            self._touch(conn, project_id)
            return self._public_frame(self._frame_row(conn, project_id, frame_id))

    def _state_references(self, state: dict) -> set[str]:
        measurement = state.get("measurement_options", {})
        fields = {state.get("frame_id"), state.get("comparison_frame_id"), measurement.get("pair_id")}
        for view in ("settings", "comparison_settings"):
            calibration = (state.get(view) or {}).get("calibration") or {}
            fields.update(calibration.get(role) for role in ("dark_id", "flat_id", "defects_id"))
        return {field for field in fields if field}

    def remove_frame(self, project_id: str, frame_id: str) -> None:
        with self.connect(write=True) as conn:
            project = self._project(conn, project_id)
            row = self._frame_row(conn, project_id, frame_id)
            for item in self._history(conn, project_id):
                if frame_id in self._state_references(item["state"]):
                    raise ConflictError("A saved inspection still uses this capture.")
            conn.execute("DELETE FROM frames WHERE id = ?", (row["id"],))
            selection = _as_json(project, "selection_json")
            if frame_id in self._state_references(selection):
                self._touch(conn, project_id, selection={})
            else:
                self._touch(conn, project_id)
        self._prune_orphans()

    def remove_project(self, project_id: str) -> None:
        with self.connect(write=True) as conn:
            self._project(conn, project_id)
            conn.execute("DELETE FROM sessions WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self._prune_orphans()

    def _prune_orphans(self) -> None:
        with self.connect(write=True) as conn:
            obsolete = conn.execute(
                """SELECT digest FROM assets WHERE NOT EXISTS (
                   SELECT 1 FROM frames WHERE frames.asset_digest = assets.digest)"""
            ).fetchall()
            conn.executemany("DELETE FROM assets WHERE digest = ?", [(r["digest"],) for r in obsolete])
        with self.connect(write=True) as conn:
            known = {row[0] for row in conn.execute("SELECT file_name FROM assets")}
            for path in self.assets.iterdir():
                if path.name in known or not path.is_file():
                    continue
                if path.suffix == ".u16" or (path.suffix == ".tmp" and time.time() - path.stat().st_mtime > 60):
                    path.unlink(missing_ok=True)

    def save_inspection(self, project_id: str, title: str, state: dict) -> dict:
        title = text_label(title, "Inspection title")
        identifier = _unique()
        with self.connect(write=True) as conn:
            self._project(conn, project_id)
            for frame_id in self._state_references(state):
                self._frame_row(conn, project_id, frame_id)
            if len(self._history(conn, project_id)) >= MAX_SAVES:
                raise CapacityError(f"Keep at most {MAX_SAVES} saved inspections per workbench.")
            stamp = _now()
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (identifier, project_id, title, state["frame_id"], _json(state), stamp),
            )
            self._touch(conn, project_id, selection=state)
        return {"id": identifier, "title": title, "state": state, "created_at": stamp}

    def _history(self, conn: sqlite3.Connection, project_id: str) -> list[dict]:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE project_id = ? ORDER BY created_at DESC, id",
            (project_id,),
        )
        return [
            {
                "id": row["id"], "title": row["title"], "frame_id": row["frame_id"],
                "state": _as_json(row, "state_json"), "created_at": row["created_at"],
            }
            for row in rows
        ]

    def remove_inspection(self, project_id: str, identifier: str) -> None:
        identifier = _identifier(identifier, "Inspection")
        with self.connect(write=True) as conn:
            changed = conn.execute(
                "DELETE FROM sessions WHERE id = ? AND project_id = ?", (identifier, project_id)
            ).rowcount
            if not changed:
                raise MissingError("Inspection not found.")
            self._touch(conn, project_id)

    def add_recipe(self, project_id: str, title: str, settings: dict) -> dict:
        title = text_label(title, "Recipe title")
        identifier = _unique()
        obj = dict(object_value(settings, "Recipe"))
        obj.pop("calibration", None)
        stamp = _now()
        with self.connect(write=True) as conn:
            self._project(conn, project_id)
            if len(self._recipes(conn, project_id)) >= MAX_RECIPES:
                raise CapacityError(f"Keep at most {MAX_RECIPES} recipes in one workbench.")
            conn.execute(
                "INSERT INTO recipes VALUES (?, ?, ?, ?, ?)",
                (identifier, project_id, title, _json(obj), stamp),
            )
            self._touch(conn, project_id)
        return {"id": identifier, "title": title, "settings": obj, "created_at": stamp}

    def _recipes(self, conn: sqlite3.Connection, project_id: str) -> list[dict]:
        return [
            {
                "id": row["id"], "title": row["title"],
                "settings": _as_json(row, "settings_json"), "created_at": row["created_at"],
            }
            for row in conn.execute(
                "SELECT * FROM recipes WHERE project_id = ? ORDER BY created_at DESC, id", (project_id,)
            )
        ]

    def remove_recipe(self, project_id: str, identifier: str) -> None:
        identifier = _identifier(identifier, "Recipe")
        with self.connect(write=True) as conn:
            changed = conn.execute(
                "DELETE FROM recipes WHERE id = ? AND project_id = ?", (identifier, project_id)
            ).rowcount
            if not changed:
                raise MissingError("Recipe not found.")
            self._touch(conn, project_id)


    def restore_archive(self, archive, normalize_state, *, name: str | None = None) -> dict:
        # Imported identifiers belong to a new independent workspace.
        from .archives import remap_saved_state
        identifier = _unique()
        title = text_label(name if name else f"{archive.name[:67]} (imported)", "Workbench name")
        try:
            with self.connect(write=True) as conn:
                total = conn.execute("SELECT count(*) FROM projects").fetchone()[0]
                if total >= MAX_PROJECTS:
                    raise CapacityError(f"Keep at most {MAX_PROJECTS} workbenches.")
                stamp = _now()
                conn.execute(
                    """INSERT INTO projects (id, name, notes, demo_key, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (identifier, title, archive.notes, archive.demo_key, stamp, stamp),
                )
                ids = {}
                imported = {}
                for frame in archive.frames:
                    new = self._attach(
                        conn, identifier, frame.capture, frame.name, None, frame.role, frame.original_name,
                    )
                    ids[frame.old_id] = new
                    imported[new] = Frame(
                        new, identifier, frame.name, frame.original_name, frame.role,
                        frame.capture.digest, frame.capture.raster, frame.capture.sensor(),
                        frame.capture.synthetic,
                    )

                def normalized(raw):
                    if not raw:
                        return {}
                    state = remap_saved_state(raw, ids)
                    key = state.get("frame_id")
                    if key is None or key not in imported:
                        raise InputError("A saved inspection needs an archived capture.")
                    checked = normalize_state(state, imported[key])
                    refs = self._state_references(checked)
                    if refs - imported.keys():
                        raise InputError("A saved inspection references a missing archived capture.")
                    return checked

                for session in reversed(archive.sessions):
                    state = normalized(session["state"])
                    if not state:
                        raise InputError("An archived inspection cannot be empty.")
                    conn.execute(
                        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                        (_unique(), identifier, session["title"], state["frame_id"], _json(state), _now()),
                    )
                for recipe in reversed(archive.recipes):
                    conn.execute(
                        "INSERT INTO recipes VALUES (?, ?, ?, ?, ?)",
                        (_unique(), identifier, recipe["title"], _json(recipe["settings"]), _now()),
                    )
                self._touch(conn, identifier, selection=normalized(archive.selection))
        except (ValueError, LookupError, CapacityError, OSError):
            self._prune_orphans()
            raise
        return self.describe(identifier)
