"""Spatial quality checks for binary defect-reference images."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .analysis import number
from .cfa import phase_planes
from .domain import Box, Frame, InputError

GRID_SIDE = 6
MAX_GROUPS = 18
MAX_LINES = 8
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class AuditWindow:
    mask: np.ndarray
    region: Box
    origin_x: int
    origin_y: int

    @classmethod
    def from_frame(cls, frame: Frame, region: Box) -> AuditWindow:
        if frame.role != "defects":
            raise InputError("Select a defect-mask capture in the Calibration references list.")
        cut = frame.sensor.active.intersection(region)
        if cut is None:
            raise InputError("The inspection region must overlap the active part of the mask.")
        return cls(
            cut.crop(frame.raster.values) != 0, cut,
            frame.sensor.layout.origin_x, frame.sensor.layout.origin_y,
        )

    def box(self, left: int, top: int, right: int, bottom: int) -> Box:
        return Box(self.region.x + left, self.region.y + top, right - left, bottom - top)

    def world(self, box: Box) -> dict:
        return box.translated(self.origin_x, self.origin_y).as_dict()


def _ranked(values: np.ndarray, maximum: int) -> np.ndarray:
    occupied = np.flatnonzero(values)
    if occupied.size <= maximum:
        return occupied[np.argsort(-values[occupied], kind="stable")]
    positive = values[occupied]
    limit = np.partition(positive, -maximum)[-maximum]
    larger = occupied[positive > limit]
    equal = occupied[positive == limit][:maximum - larger.size]
    chosen = np.concatenate((larger, equal))
    return chosen[np.argsort(-values[chosen], kind="stable")]


def _groups(window: AuditWindow) -> dict:
    mask = window.mask
    labels, count = ndimage.label(mask, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    frequencies = np.bincount(sizes[1:], minlength=2)
    chosen = _ranked(sizes, MAX_GROUPS)
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[chosen] = np.arange(1, chosen.size + 1)
    shortlist = lookup[labels]
    output = []
    for offset, (rows, cols) in enumerate(ndimage.find_objects(shortlist, max_label=chosen.size)):
        top, bottom = rows.start, rows.stop
        left, right = cols.start, cols.stop
        bounds = window.box(left, top, right, bottom)
        sites = int(sizes[chosen[offset]])
        focus = window.box(
            max(left - 4, 0), max(top - 4, 0),
            min(right + 4, mask.shape[1]), min(bottom + 4, mask.shape[0]),
        )
        output.append({
            "rank": offset + 1,
            "sites": sites,
            "raster_box": bounds.as_dict(),
            "sensor_box": window.world(bounds),
            "focus_region": focus.as_dict(),
            "occupancy": number(sites / bounds.area),
            "touches_region_boundary": top == 0 or left == 0 or bottom == mask.shape[0] or right == mask.shape[1],
        })
    return {
        "total": count,
        "singletons": int(frequencies[1]),
        "multiple_sites": count - int(frequencies[1]),
        "largest": int(sizes.max()),
        "shown": output,
    }


def _run(values: np.ndarray) -> tuple[int, int]:
    transitions = np.flatnonzero(np.diff(values, prepend=False, append=False))
    if not transitions.size:
        return 0, 0
    lengths = transitions[1::2] - transitions[::2]
    strongest = int(np.argmax(lengths))
    return int(transitions[2 * strongest]), int(lengths[strongest])


def _lines(window: AuditWindow, *, rows: bool) -> dict:
    lines = window.mask if rows else window.mask.T
    counts = np.count_nonzero(lines, axis=1)
    output = []
    across = window.region.x + window.origin_x if rows else window.region.y + window.origin_y
    along = window.region.y + window.origin_y if rows else window.region.x + window.origin_x
    for index in _ranked(counts, MAX_LINES):
        start, length = _run(lines[index])
        output.append({
            "sensor_line": along + int(index),
            "marked": int(counts[index]),
            "fraction": number(counts[index] / lines.shape[1]),
            "longest_run": length,
            "run_begins_at": across + start,
        })
    return {
        "affected": int(np.count_nonzero(counts)),
        "entirely_marked": int(np.count_nonzero(counts == lines.shape[1])),
        "total": lines.shape[0],
        "shown": output,
    }


def _density_grid(window: AuditWindow) -> dict:
    height, width = window.mask.shape
    rows = np.linspace(0, height, min(height, GRID_SIDE) + 1, dtype=int)
    columns = np.linspace(0, width, min(width, GRID_SIDE) + 1, dtype=int)
    grid = []
    for top, bottom in zip(rows[:-1], rows[1:]):
        row = []
        for left, right in zip(columns[:-1], columns[1:]):
            box = window.box(int(left), int(top), int(right), int(bottom))
            sites = int(np.count_nonzero(window.mask[top:bottom, left:right]))
            row.append({
                "box": box.as_dict(), "marked": sites, "total": box.area,
                "per_megapixel": number(sites * 1_000_000 / box.area, 2),
            })
        grid.append(row)
    densest = max((cell for row in grid for cell in row), key=lambda cell: cell["marked"] / cell["total"])
    return {
        "columns": int(columns.size) - 1,
        "rows": int(rows.size) - 1,
        "cells": grid,
        "densest": densest if densest["marked"] else None,
    }


def defect_map_report(frame: Frame, region: Box) -> dict:
    window = AuditWindow.from_frame(frame, region)
    area = frame.sensor.active
    active_count = int(np.count_nonzero(area.crop(frame.raster.values)))
    source_count = int(np.count_nonzero(frame.raster.values))
    local_layout = frame.sensor.layout.shifted(window.region.x, window.region.y)
    phases = []
    for plane in phase_planes(window.mask, local_layout):
        marked = int(np.count_nonzero(plane.samples))
        phases.append({
            "phase": plane.name,
            "marked": marked,
            "available": plane.site_count,
            "per_megapixel": number(marked * 1_000_000 / plane.site_count, 2) if plane.site_count else None,
        })
    marked = int(np.count_nonzero(window.mask))
    return {
        "region": window.region.as_dict(),
        "sensor_region": window.world(window.region),
        "marked": marked,
        "density": number(marked / window.region.area, 8),
        "active_marked": active_count,
        "outside_active": source_count - active_count,
        "groups": _groups(window),
        "rows": _lines(window, rows=True),
        "columns": _lines(window, rows=False),
        "phases": phases,
        "grid": _density_grid(window),
        "method": "All nonzero sites are marks. Pixel groups use eight adjacent neighbors within the selected active region.",
    }
