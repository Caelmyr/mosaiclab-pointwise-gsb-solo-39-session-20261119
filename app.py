#!/usr/bin/env python3
"""HTTP entry point for the local MosaicLab workbench."""
from __future__ import annotations

import io
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from waitress import serve
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge

from rawbench.archives import MAX_ARCHIVE_BYTES
from rawbench.artifacts import DownloadOptions
from rawbench.codecs import (
    binary_encodings, decode_capture, filename_label, inspect_binary, json_object, read_limited,
)
from rawbench.domain import (
    MAX_DIMENSION, MAX_PIXELS, PATTERNS, ROLE_VALUES, CapacityError, ConflictError,
    InputError, MissingError, choice, flag, integer, object_value, reject_unknown,
)
from rawbench.storage import Catalog
from rawbench.synthetic import demo_catalog
from rawbench.workbench import MEASUREMENT_KINDS, MeasurementSettings, Probe, Processing
from rawbench.development import DevelopSettings

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"


def create_app(data: Path | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_ARCHIVE_BYTES + 1024 * 1024
    app.config["MAX_FORM_MEMORY_SIZE"] = 128 * 1024
    catalog = Catalog(data or Path(os.environ.get("MOSAICLAB_DATA_DIR", ROOT / "data")))
    engine = Processing(catalog)

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(InputError)
    @app.errorhandler(BadRequest)
    def invalid(error):
        message = str(error) if isinstance(error, InputError) else "Request data is malformed."
        return jsonify(error=message, type="input"), 400

    @app.errorhandler(MissingError)
    def missing(error):
        return jsonify(error=str(error), type="missing"), 404

    @app.errorhandler(ConflictError)
    def conflict(error):
        return jsonify(error=str(error), type="conflict"), 409

    @app.errorhandler(CapacityError)
    def capacity(error):
        return jsonify(error=str(error), type="capacity"), 429

    @app.errorhandler(RequestEntityTooLarge)
    def oversized(_error):
        return jsonify(error="Upload exceeds the workbench's size limit.", type="capacity"), 413

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/"):
            return jsonify(error=error.description, type="http"), error.code
        return error

    def payload(*fields):
        if not request.is_json:
            raise InputError("Send this request as application/json.")
        values = json_object(request.get_data(cache=False), "Request")
        if fields:
            reject_unknown(values, fields, "request")
        return values

    def send_artifact(artifact):
        return send_file(
            io.BytesIO(artifact.data), mimetype=artifact.mimetype,
            as_attachment=True, download_name=artifact.filename,
        )

    def inspect_payload(project: str, *fields):
        values = payload("inspection", *fields)
        frame, state = engine.parse(project, values.get("inspection"))
        return values, frame, state

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html", max_age=0)

    @app.get("/api/health")
    def health():
        with catalog.connect() as conn:
            conn.execute("SELECT id FROM projects LIMIT 1").fetchone()
        return jsonify(status="ok", app="MosaicLab", version=VERSION)

    @app.get("/api/options")
    def options():
        return jsonify(
            patterns=list(PATTERNS),
            roles=ROLE_VALUES,
            demo=demo_catalog(),
            binary_encodings=binary_encodings(),
            measurements=MEASUREMENT_KINDS,
            limits={"samples": MAX_PIXELS, "side": MAX_DIMENSION},
            defaults={
                "settings": DevelopSettings().as_dict(),
                "measurement_options": MeasurementSettings().as_dict(),
            },
        )

    @app.get("/api/projects")
    def projects():
        return jsonify(projects=catalog.list_projects())

    @app.post("/api/projects")
    def add_project():
        values = payload("name", "demo")
        sample = values.get("demo")
        name = values.get("name")
        if sample is not None:
            if not isinstance(sample, str):
                raise InputError("Choose a synthetic sample by name.")
            result = engine.execute(engine.new_demo, sample, name)
        else:
            result = catalog.create_project(name)
            result["suggested_regions"] = []
        return jsonify(result), 201

    @app.post("/api/project-archives")
    def import_workbench():
        upload = request.files.get("archive")
        if upload is None or not upload.filename:
            raise InputError("Choose a MosaicLab workbench archive.")
        data = read_limited(upload.stream, MAX_ARCHIVE_BYTES)
        result = engine.execute(engine.import_archive, data, request.form.get("name"))
        return jsonify(result), 201

    @app.get("/api/projects/<project>/archive")
    def archive_workbench(project):
        return send_artifact(engine.execute(engine.archive, project))

    @app.get("/api/projects/<project>")
    def project_detail(project):
        return jsonify(engine.add_demo_regions(catalog.describe(project)))

    @app.patch("/api/projects/<project>")
    def edit_project(project):
        values = payload("name", "notes")
        result = catalog.update_project(project, values.get("name"), values.get("notes", ""))
        return jsonify(engine.add_demo_regions(result))

    @app.post("/api/binary-check")
    def check_binary():
        upload = request.files.get("capture")
        if upload is None or not upload.filename:
            raise InputError("Choose a capture file.")
        layout = json_object(request.form.get("binary", ""), "Binary layout")
        return jsonify(engine.execute(inspect_binary, read_limited(upload.stream), layout))

    @app.delete("/api/projects/<project>")
    def delete_project(project):
        catalog.remove_project(project)
        return "", 204

    @app.post("/api/projects/<project>/frames")
    def add_capture(project):
        upload = request.files.get("capture")
        if upload is None or not upload.filename:
            raise InputError("Choose a capture file.")
        role = choice(request.form.get("role", "scene"), "Capture role", ROLE_VALUES)
        sensor_raw = request.form.get("sensor", "")
        metadata = dict(json_object(sensor_raw, "Sensor metadata")) if sensor_raw else None
        raw_spec = request.form.get("binary", "")
        binary = json_object(raw_spec, "Binary layout") if raw_spec else None
        data = read_limited(upload.stream)
        clean_name = filename_label(upload.filename)
        decoded = engine.execute(decode_capture, data, clean_name, binary=binary)
        name = request.form.get("name") or decoded.suggested_name
        result = catalog.add_frame(project, decoded, name, metadata, role, clean_name)
        return jsonify(result), 201

    @app.get("/api/projects/<project>/frames/<frame>")
    def capture(project, frame):
        source = catalog.load_frame(project, frame)
        return jsonify(source.public())

    @app.patch("/api/projects/<project>/frames/<frame>")
    def edit_capture(project, frame):
        values = payload("name", "sensor")
        source = catalog.load_frame(project, frame)
        changed = catalog.set_frame_metadata(
            project, frame, values.get("name", source.name),
            dict(object_value(values.get("sensor", {}), "Sensor metadata")),
        )
        return jsonify(changed)

    @app.delete("/api/projects/<project>/frames/<frame>")
    def delete_capture(project, frame):
        catalog.remove_frame(project, frame)
        return "", 204

    @app.get("/api/projects/<project>/frames/<frame>/references")
    def reference_choices(project, frame):
        source = catalog.load_frame(project, frame)
        return jsonify(engine.calibration_choices(source))

    @app.post("/api/projects/<project>/preview")
    def preview(project):
        _, frame, state = inspect_payload(project)
        return jsonify(engine.execute(engine.preview, frame, state))

    @app.post("/api/projects/<project>/probe")
    def probe(project):
        values, frame, state = inspect_payload(project, "radius", "include_developed")
        radius = integer(values.get("radius", 2), "Probe radius", 0, 6)
        developed = flag(values.get("include_developed", False), "Include developed value")
        return jsonify(engine.execute(engine.probe, frame, state, radius, developed))

    @app.post("/api/projects/<project>/neutral")
    def neutral(project):
        _, frame, state = inspect_payload(project)
        return jsonify(engine.execute(engine.neutral, frame, state))

    @app.post("/api/projects/<project>/color-audit")
    def color_target(project):
        values, frame, state = inspect_payload(project, "patches")
        return jsonify(engine.execute(engine.color_target, frame, state, values.get("patches")))

    @app.post("/api/projects/<project>/color-audit/download")
    def color_target_download(project):
        values, frame, state = inspect_payload(project, "patches", "format")
        extension = choice(values.get("format", "json"), "Format", ("json", "csv"))
        artifact = engine.execute(engine.color_target_download, frame, state, values.get("patches"), extension)
        return send_artifact(artifact)

    @app.post("/api/projects/<project>/compare")
    def compare(project):
        values, frame, state = inspect_payload(project, "sensor_probe")
        cursor = Probe.read(values.get("sensor_probe"))
        return jsonify(engine.execute(engine.comparison_preview, frame, state, cursor))

    @app.post("/api/projects/<project>/compare/download")
    def compare_download(project):
        values, frame, state = inspect_payload(project, "report_only")
        only = flag(values.get("report_only", False), "Report only")
        return send_artifact(engine.execute(engine.comparison_download, frame, state, only))

    @app.post("/api/projects/<project>/measure")
    def measure(project):
        values, frame, state = inspect_payload(project, "kind", "options")
        kind = choice(values.get("kind", state.measurement), "Measurement", MEASUREMENT_KINDS)
        settings = MeasurementSettings.read(values["options"]) if "options" in values else state.measurement_options
        return jsonify(engine.execute(engine.measurement, frame, state, kind, settings))

    @app.post("/api/projects/<project>/measure/download")
    def measure_download(project):
        values, frame, state = inspect_payload(project, "kind", "options", "format")
        kind = choice(values.get("kind", state.measurement), "Measurement", MEASUREMENT_KINDS)
        extension = choice(values.get("format", "json"), "Format", ("json", "csv"))
        settings = MeasurementSettings.read(values["options"]) if "options" in values else state.measurement_options
        out = engine.execute(engine.measured_download, frame, state, kind, settings, extension)
        return send_artifact(out)

    @app.post("/api/projects/<project>/download")
    def download(project):
        values, frame, state = inspect_payload(project, "options")
        settings = DownloadOptions.read(values.get("options", {}))
        return send_artifact(engine.execute(engine.download, frame, state, settings))

    @app.put("/api/projects/<project>/selection")
    def selection(project):
        values, frame, state = inspect_payload(project, "revision")
        revision = integer(values.get("revision"), "Workbench revision", 0, 10**12)
        checked = engine.validate_state(frame, state)
        return jsonify(catalog.set_selection(project, checked, revision))

    @app.post("/api/projects/<project>/sessions")
    def save_session(project):
        values, frame, state = inspect_payload(project, "title")
        checked = engine.validate_state(frame, state)
        return jsonify(catalog.save_inspection(project, values.get("title"), checked)), 201

    @app.delete("/api/projects/<project>/sessions/<session>")
    def remove_saved_session(project, session):
        catalog.remove_inspection(project, session)
        return "", 204

    @app.post("/api/projects/<project>/recipes")
    def save_recipe(project):
        values = payload("title", "settings")
        settings = DevelopSettings.read(values.get("settings")).as_dict()
        return jsonify(catalog.add_recipe(project, values.get("title"), settings)), 201

    @app.delete("/api/projects/<project>/recipes/<recipe>")
    def remove_recipe(project, recipe):
        catalog.remove_recipe(project, recipe)
        return "", 204

    return app


def main():
    app = create_app()
    port = integer(os.environ.get("PORT", "18942"), "Port", 1024, 65535)
    host = os.environ.get("MOSAICLAB_HOST", "127.0.0.1")
    print(f"MosaicLab listening on {host}:{port}", flush=True)
    serve(app, host=host, port=port, threads=4, max_request_body_size=MAX_ARCHIVE_BYTES + 1024 * 1024)


if __name__ == "__main__":
    main()
