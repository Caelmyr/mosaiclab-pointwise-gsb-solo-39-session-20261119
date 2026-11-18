"""Spatial inspection state and processing orchestration for the web interface."""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from threading import BoundedSemaphore

from .analysis import (
    HISTOGRAM_SPACES, PROFILE_AXES, banding_report, exposure_histogram,
    exposure_summary, measure_region, paired_noise_report, profile_report,
    single_noise_report,
)
from .archives import archive_project, decode_archive
from .artifacts import (
    Artifact, DownloadOptions, comparison_artifact, development_artifact,
    json_bytes, measurement_csv, raw_artifact,
)
from .color_audit import color_audit
from .calibration import CalibrationSettings, Correction, calibrate, reference_summary
from .comparison import CompareOptions, Comparison, compare, comparison_probe
from .chromatic import chromatic_report
from .development import DevelopSettings, Developed, develop, neutral_from_region
from .domain import (
    Box, CapacityError, Frame, InputError, choice, flag,
    integer, object_value, optional_id, reject_unknown, sensor_compatible,
)
from .mask_audit import defect_map_report
from .optics import EdgeSettings, edge_report, uniformity_report
from .storage import Catalog
from .synthetic import BLUEPRINTS, chart_swatches, make_demo, suggested_regions
from .view import (
    ViewSettings, calibration_preview, developed_preview, preview_png, probe_report, raw_preview,
    Preview,
)

MEASUREMENT_KINDS = (
    "region", "histogram", "exposure", "profile", "banding",
    "spatial-noise", "paired-noise", "uniformity", "edge", "chromatic", "reference", "mask-audit",
)
MAX_PARALLEL_OPERATIONS = 2
INSPECTION_TOOLS = (*MEASUREMENT_KINDS, "color-audit")
COLOR_HEX = re.compile(r"#[0-9a-fA-F]{6}\Z")


@dataclass(frozen=True)
class Probe:
    x: int
    y: int

    @classmethod
    def read(cls, value, *, max_x=139264, max_y=139264) -> Probe | None:
        if value is None:
            return None
        obj = object_value(value, "Probe")
        reject_unknown(obj, ("x", "y"), "probe")
        x = integer(obj.get("x"), "Probe x", 0, max_x)
        y = integer(obj.get("y"), "Probe y", 0, max_y)
        return cls(x, y)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class InspectionState:
    frame_id: str
    settings: DevelopSettings
    view: ViewSettings
    region: Box | None
    probe: Probe | None
    measurement: str
    measurement_options: MeasurementSettings
    comparison_frame_id: str | None
    comparison_settings: DevelopSettings
    comparison_options: CompareOptions

    @classmethod
    def read(cls, raw, frame: Frame) -> InspectionState:
        value = object_value(raw, "Inspection")
        reject_unknown(
            value,
            ("frame_id", "settings", "view", "region", "probe", "measurement", "measurement_options",
             "comparison_frame_id", "comparison_settings", "comparison_options"),
            "inspection",
        )
        if value.get("frame_id") != frame.id:
            raise InputError("Inspection capture does not match the workbench selection.")
        settings = DevelopSettings.read(value.get("settings"))
        alt_raw = value.get("comparison_settings")
        alt = DevelopSettings.read(alt_raw) if alt_raw is not None else DevelopSettings()
        width, height = frame.raster.width, frame.raster.height
        region_raw = value.get("region")
        region = Box.read(region_raw, width, height) if region_raw is not None else None
        return cls(
            frame.id, settings, ViewSettings.read(value.get("view")), region,
            Probe.read(value.get("probe"), max_x=width - 1, max_y=height - 1),
            choice(value.get("measurement", "region"), "Measurement", INSPECTION_TOOLS),
            MeasurementSettings.read(value.get("measurement_options")),
            optional_id(value.get("comparison_frame_id"), "Comparison capture"), alt,
            CompareOptions.read(value.get("comparison_options")),
        )

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "settings": self.settings.as_dict(),
            "view": self.view.as_dict(),
            "region": self.region.as_dict() if self.region else None,
            "probe": self.probe.as_dict() if self.probe else None,
            "measurement": self.measurement,
            "measurement_options": self.measurement_options.as_dict(),
            "comparison_frame_id": self.comparison_frame_id,
            "comparison_settings": self.comparison_settings.as_dict(),
            "comparison_options": self.comparison_options.as_dict(),
        }


@dataclass(frozen=True)
class MeasurementSettings:
    bins: int = 64
    space: str = "linear"
    axis: str = "rows"
    normalized_profile: bool = False
    pair_id: str | None = None
    tiles: int = 5
    edge: EdgeSettings = EdgeSettings()
    corrected: bool = False
    include_developed: bool = False
    target_color: str = "#ababab"

    @classmethod
    def read(cls, value) -> MeasurementSettings:
        obj = object_value(value if value is not None else {}, "Measurement options")
        reject_unknown(
            obj,
            ("bins", "space", "axis", "normalized_profile", "pair_id", "tiles", "edge", "corrected",
             "include_developed", "target_color"),
            "measurement",
        )
        n = integer(obj.get("bins", 64), "Histogram bins", 16, 256)
        if n not in (16, 32, 64, 128, 256):
            raise InputError("Histogram bins must be 16, 32, 64, 128 or 256.")
        color = obj.get("target_color", "#ababab")
        if not isinstance(color, str) or not COLOR_HEX.fullmatch(color):
            raise InputError("Reference color must use a six-digit sRGB hex value.")
        return cls(
            n,
            choice(obj.get("space", "linear"), "Histogram space", HISTOGRAM_SPACES),
            choice(obj.get("axis", "rows"), "Line direction", PROFILE_AXES),
            flag(obj.get("normalized_profile", False), "Normalized profiles"),
            optional_id(obj.get("pair_id"), "Repeat capture"),
            integer(obj.get("tiles", 5), "Uniformity subdivisions", 2, 12),
            EdgeSettings.read(obj.get("edge")),
            flag(obj.get("corrected", False), "Include calibrated values"),
            flag(obj.get("include_developed", False), "Include reconstructed colors"),
            color.lower(),
        )

    def as_dict(self) -> dict:
        return {
            "bins": self.bins,
            "space": self.space,
            "axis": self.axis,
            "normalized_profile": self.normalized_profile,
            "pair_id": self.pair_id,
            "tiles": self.tiles,
            "edge": self.edge.as_dict(),
            "corrected": self.corrected,
            "include_developed": self.include_developed,
            "target_color": self.target_color,
        }


class Processing:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.compute = BoundedSemaphore(MAX_PARALLEL_OPERATIONS)

    def new_demo(self, key: str, name: str | None) -> dict:
        if key not in BLUEPRINTS:
            raise InputError("Choose one of the available synthetic scenes.")
        captures = make_demo(key)
        result = self.catalog.create_project(
            name if name else BLUEPRINTS[key]["title"],
            demo_key=key,
            notes="Synthetic sensor raster. Values were generated for controlled comparison.",
            captures=captures,
        )
        return self.add_demo_regions(result)

    @staticmethod
    def add_demo_regions(project: dict) -> dict:
        key = project.get("demo_key")
        project["suggested_regions"] = suggested_regions(key) if key else []
        project["chart_swatches"] = chart_swatches() if key == "color-target" else []
        return project

    def archive(self, project: str) -> Artifact:
        return archive_project(self.catalog, project)

    def import_archive(self, body: bytes, name: str | None) -> dict:
        record = decode_archive(body)
        if record.demo_key not in BLUEPRINTS:
            from dataclasses import replace
            record = replace(record, demo_key=None)
        restored = self.catalog.restore_archive(
            record, lambda raw, frame: InspectionState.read(raw, frame).as_dict(),
            name=name,
        )
        return self.add_demo_regions(restored)

    def parse(self, project: str, data) -> tuple[Frame, InspectionState]:
        inspection = object_value(data, "Inspection")
        selected = optional_id(inspection.get("frame_id"), "Capture")
        if selected is None:
            raise InputError("Select a capture.")
        frame = self.catalog.load_frame(project, selected)
        return frame, InspectionState.read(inspection, frame)

    def references(self, frame: Frame, settings: CalibrationSettings) -> dict[str, Frame]:
        result = {}
        for key, identifier in settings.references().items():
            reference = self.catalog.load_frame(frame.project, identifier)
            if reference.role != key:
                raise InputError(f"{reference.name} is not a {key} reference.")
            if not sensor_compatible(frame, reference):
                raise InputError(
                    f"{reference.name} does not match the capture's dimensions and sensor layout."
                )
            result[key] = reference
        return result

    def _development(self, frame: Frame, settings: DevelopSettings) -> Developed:
        references = self.references(frame, settings.calibration)
        return develop(frame, settings, references)

    def _correction(self, frame: Frame, settings: DevelopSettings) -> Correction:
        return calibrate(frame, settings.calibration, self.references(frame, settings.calibration))

    def _second_frame(self, left: Frame, state: InspectionState) -> Frame:
        target = state.comparison_frame_id or left.id
        return self.catalog.load_frame(left.project, target)

    def _comparison(self, frame: Frame, state: InspectionState) -> Comparison:
        right = self._second_frame(frame, state)
        for item in (frame, right):
            if item.role not in ("scene", "flat", "dark"):
                raise InputError("A binary defect map cannot be developed for comparison.")
        main_settings = state.settings
        alt_settings = state.comparison_settings
        # The common comparison display is sRGB even when either export uses Display P3.
        first = self._development(frame, main_settings.preview_settings())
        if frame.id == right.id and main_settings == alt_settings:
            second = first
        else:
            second = self._development(right, alt_settings.preview_settings())
        return compare(first, second, state.comparison_options, state.region)

    def preview(self, frame: Frame, state: InspectionState) -> dict:
        prefs = state.view
        if prefs.layer == "developed":
            preview = developed_preview(self._development(frame, state.settings))
        else:
            box = frame.sensor.active if state.settings.active_only else Box.full(
                frame.raster.width, frame.raster.height
            )
            if prefs.layer == "calibration":
                preview = calibration_preview(self._correction(frame, state.settings), prefs, box)
            else:
                preview = raw_preview(frame, prefs, box)
        raw, meta = preview_png(preview, prefs.width)
        return {"url": png_url(raw), "layer": prefs.layer, "info": meta}

    def calibration_choices(self, frame: Frame) -> dict:
        project = self.catalog.describe(frame.project)
        result = {"dark": [], "flat": [], "defects": []}
        shape = frame.raster.width, frame.raster.height
        for item in project["frames"]:
            role = item["role"]
            if role not in result or frame.id == item["id"]:
                continue
            size = item["width"], item["height"]
            sensor = frame.sensor.from_values(item["sensor"], *size, item["maximum"])
            compatible = shape == size and frame.sensor.matches_raster(sensor)
            item = {
                "id": item["id"], "name": item["name"], "compatible": compatible,
                "exposure_ms": sensor.exposure_ms, "analog_gain": sensor.analog_gain,
            }
            result[role].append(item)
        return result

    def probe(self, frame: Frame, state: InspectionState, radius: int, developed: bool) -> dict:
        tap = state.probe
        if tap is None:
            raise InputError("Select one sensor site to inspect.")
        if developed:
            view = self._development(frame, state.settings)
            return probe_report(frame, tap.x, tap.y, radius, view.correction, view)
        correction = None
        if state.view.layer == "calibration":
            correction = self._correction(frame, state.settings)
        return probe_report(frame, tap.x, tap.y, radius, correction)

    def neutral(self, frame: Frame, state: InspectionState) -> dict:
        if state.region is None:
            raise InputError("Draw or enter a neutral sample region first.")
        return neutral_from_region(self._correction(frame, state.settings), state.region)

    def measurement(
        self, frame: Frame, state: InspectionState, kind: str, options: MeasurementSettings
    ) -> dict:
        region = state.region or frame.sensor.active
        if kind == "region":
            correction = self._correction(frame, state.settings) if options.corrected else None
            return measure_region(frame, region, correction)
        if kind == "exposure":
            return exposure_summary(frame, region)
        if kind == "histogram":
            return exposure_histogram(frame, region, space=options.space, bins=options.bins)
        if kind == "profile":
            return profile_report(frame, region, options.axis, options.normalized_profile)
        if kind == "banding":
            return banding_report(frame, region, options.axis)
        if kind == "spatial-noise":
            return single_noise_report(frame, region)
        if kind == "paired-noise":
            pair_id = options.pair_id or state.comparison_frame_id
            if pair_id is None:
                raise InputError("Choose the matching repeat exposure.")
            repeat = self.catalog.load_frame(frame.project, pair_id)
            return paired_noise_report(frame, repeat, region)
        if kind == "uniformity":
            return uniformity_report(frame, region, options.tiles)
        if kind == "edge":
            return edge_report(self._development(frame, state.settings), region, options.edge)
        if kind == "chromatic":
            return chromatic_report(frame, region, options.edge)
        if kind == "reference":
            return reference_summary(frame)
        if kind == "mask-audit":
            return defect_map_report(frame, region)
        raise InputError("Unknown measurement.")

    def measured_download(
        self, frame: Frame, state: InspectionState, kind: str,
        options: MeasurementSettings, extension: str,
    ) -> Artifact:
        report = self.measurement(frame, state, kind, options)
        if extension == "csv":
            return measurement_csv(kind, report, frame.name)
        document = {
            "format": "mosaiclab.measurement/1",
            "capture": frame.public(),
            "measurement": kind,
            "result": report,
        }
        return Artifact("sensor-measurement.json", json_bytes(document), "application/json")

    def download(
        self, frame: Frame, state: InspectionState, options: DownloadOptions
    ) -> Artifact:
        selected = state.region if options.crop_region else None
        if options.crop_region and selected is None:
            raise InputError("Set a region before requesting a cropped export.")
        if options.kind == "raw":
            return raw_artifact(
                frame, selected, options.rotation,
                encoding=options.raw_encoding, alignment=options.raw_alignment,
            )
        return development_artifact(self._development(frame, state.settings), options.kind, selected)

    def color_target(self, frame: Frame, state: InspectionState, patches) -> dict:
        return color_audit(self._development(frame, state.settings), patches)

    def color_target_download(self, frame: Frame, state: InspectionState, patches, extension: str) -> Artifact:
        report = self.color_target(frame, state, patches)
        if extension == "csv":
            return measurement_csv("color-audit", report, frame.name)
        document = {
            "format": "mosaiclab.measurement/1",
            "capture": frame.public(),
            "measurement": "color-audit",
            "development": state.settings.as_dict(),
            "reference_patches": patches,
            "result": report,
        }
        return Artifact("color-target-audit.json", json_bytes(document), "application/json")

    def comparison_preview(self, frame: Frame, state: InspectionState, cursor: Probe | None) -> dict:
        result = self._comparison(frame, state)
        preview = Preview(result.view, None, "antialias", {})
        payload, info = preview_png(preview, state.view.width)
        return {
            "url": png_url(payload),
            "preview": info,
            "report": result.report,
            "probe": comparison_probe(result, cursor.x, cursor.y) if cursor else None,
        }

    def comparison_download(
        self, frame: Frame, state: InspectionState, report_only: bool
    ) -> Artifact:
        return comparison_artifact(self._comparison(frame, state), frame.name, report_only=report_only)

    def validate_state(self, frame: Frame, state: InspectionState) -> dict:
        self.references(frame, state.settings.calibration)
        right = self._second_frame(frame, state)
        self.references(right, state.comparison_settings.calibration)
        pair = state.measurement_options.pair_id
        if pair is not None:
            repeat = self.catalog.load_frame(frame.project, pair)
            if repeat.id == frame.id:
                raise InputError("Choose a second capture for paired noise measurements.")
            if repeat.role != "scene":
                raise InputError("Paired noise measurements use a scene capture as the repeat.")
        return state.as_dict()

    def execute(self, action, *args, **kwargs):
        entered = self.compute.acquire(blocking=False)
        if not entered:
            raise CapacityError("The workbench is processing two requests. Try again in a moment.")
        try:
            return action(*args, **kwargs)
        finally:
            self.compute.release()


def png_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
