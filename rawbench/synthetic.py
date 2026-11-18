"""Deterministic synthetic RAW captures and matching calibration references."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .cfa import phase_field, sample_scene
from .codecs import DecodedCapture, digest_bytes, encode_pgm
from .development import srgb_decode
from .domain import Box, InputError, Layout, Raster, Sensor

SIZE = (540, 360)
SCENES = ("color-target", "resolution", "shadow-register", "exposure-pair")
BLUEPRINTS = {
    "color-target": {
        "title": "Color and neutral target",
        "summary": "Pigment patches, a gray card and fine colored edges.",
        "names": ("Target 01 · color chart",),
    },
    "resolution": {
        "title": "Resolution field",
        "summary": "Slanted edges, concentric rings and a line-frequency sweep.",
        "names": ("Target 02 · resolution",),
    },
    "shadow-register": {
        "title": "Low-light readout",
        "summary": "Dim bars, row banding and repeat captures with different shot noise.",
        "names": ("Shadow 01 · repeat A", "Shadow 01 · repeat B"),
    },
    "exposure-pair": {
        "title": "Exposure ladder pair",
        "summary": "Two gains for the same scene with bright patches and deep shadows.",
        "names": ("Exposure 01 · lower gain", "Exposure 01 · higher gain"),
    },
}
PATCHES = (
    (176, 92, 67), (225, 143, 118), (96, 133, 164), (111, 134, 80),
    (151, 128, 179), (103, 165, 155), (216, 134, 68), (93, 106, 172),
    (196, 83, 99), (245, 176, 76), (97, 149, 91), (82, 171, 195),
    (45, 49, 48), (79, 82, 81), (112, 115, 112), (148, 151, 147),
    (189, 191, 188), (232, 233, 228),
)
BASE_BLACK = (96.0, 101.0, 102.0, 111.0)
BASE_WHITE = (3970.0, 4035.0, 3998.0, 3925.0)
RAMP_STOPS = (41, 63, 94, 131, 165, 204, 229)
TEXT_LIGHT = (229, 231, 219)
TEXT_MUTED = (165, 183, 173)
INK = (18, 29, 24)


@dataclass(frozen=True)
class DemoSpec:
    key: str
    seed: int
    layout: Layout
    active: Box
    exposure: float
    gains: tuple[float, ...]
    spectral_bias: tuple[float, float, float]
    noise: float
    row_band: float


def demo_catalog() -> list[dict]:
    result = []
    for key in SCENES:
        value = BLUEPRINTS[key]
        result.append({
            "key": key,
            "title": value["title"],
            "summary": value["summary"],
            "scene_count": len(value["names"]),
            "includes_calibration_references": True,
        })
    return result


def _font(size: int):
    return ImageFont.load_default(size=size)


def _text(draw: ImageDraw.ImageDraw, at: tuple[int, int], text: str, size: int, color) -> None:
    draw.text(at, text, font=_font(size), fill=color)


def _scene_canvas(width: int, height: int, title: str, identifier: str) -> Image.Image:
    image = Image.new("RGB", (width, height), (22, 39, 31))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, 66), fill=(17, 28, 23))
    draw.line((25, 66, width - 26, 66), fill=(89, 117, 100), width=1)
    _text(draw, (25, 17), title, 24, TEXT_LIGHT)
    _text(draw, (27, 48), "SYNTHETIC REFERENCE • LINEAR SENSOR TEST", 9, TEXT_MUTED)
    _text(draw, (width - 82, 26), identifier, 14, (179, 227, 161))
    return image


def _target_rgb(width: int, height: int) -> np.ndarray:
    image = _scene_canvas(width, height, "SENSOR COLOR TARGET", "CT—01")
    draw = ImageDraw.Draw(image)
    patch_w = 60
    patch_h = 45
    gap = 11
    for index, color in enumerate(PATCHES):
        row, col = divmod(index, 6)
        x = 27 + col * (patch_w + gap)
        y = 91 + row * (patch_h + gap)
        draw.rectangle((x - 1, y - 1, x + patch_w, y + patch_h), fill=(89, 110, 98))
        draw.rectangle((x, y, x + patch_w - 1, y + patch_h - 1), fill=color)
        draw.text((x + 3, y + 4), f"{index + 1:02}", font=_font(9), fill=INK)
    gray_x = 27
    gray_top = 270
    neutral = (163, 163, 163)
    draw.rectangle((gray_x, gray_top, 212, gray_top + 47), fill=neutral)
    _text(draw, (gray_x, 324), "NEUTRAL CARD", 9, TEXT_MUTED)
    for index in range(104):
        hue = (210, 96, 104) if (index // 2) & 1 else (77, 155, 174)
        left = 238 + 2 * index
        draw.line((left, gray_top, left, 317), hue, width=2)
    _text(draw, (239, 324), "CHROMA FREQUENCY", 9, TEXT_MUTED)
    draw.rectangle((464, 90, 506, 317), fill=(15, 26, 19))
    for index, stop in enumerate(RAMP_STOPS):
        top = 98 + 30 * index
        draw.rectangle((472, top, 496, top + 23), fill=(stop, stop, stop))
    base = srgb_decode(np.asarray(image, dtype=np.float32) / 255)
    return base * 0.86


def _grid_labels(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    _text(draw, (28, 87), "RADIAL / ALIASING", 10, TEXT_MUTED)
    _text(draw, (293, 87), "SLANTED EDGE", 10, TEXT_MUTED)
    _text(draw, (289, 216), "FREQUENCY SWEEP", 10, TEXT_MUTED)
    draw.rectangle((23, 104, 265, 332), outline=(104, 126, 113), width=1)
    draw.rectangle((286, 104, 513, 199), outline=(104, 126, 113), width=1)
    draw.rectangle((286, 233, 513, 332), outline=(104, 126, 113), width=1)


def _resolution_rgb(width: int, height: int) -> np.ndarray:
    image = _scene_canvas(width, height, "OPTICAL RESOLUTION PLATE", "RS—02")
    _grid_labels(image)
    rgb = srgb_decode(np.asarray(image, dtype=np.float32) / 255).copy()
    y, x = np.mgrid[0:height, 0:width]
    dx = x - 146
    dy = y - 218
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    disc = radius < 103
    star = 0.72 * (np.cos(31 * angle + 0.07 * radius) > 0) + 0.055
    ring = 0.9 + 0.11 * np.cos(radius * 0.55)
    rgb[disc] = (star * ring)[disc, None]
    edge = (x >= 287) & (x < 513) & (y >= 105) & (y < 199)
    edge_level = np.clip((x - 402 - (y - 149) * 0.18) * 0.9, -0.5, 0.5) * 0.65 + 0.40
    rgb[..., 0][edge] = edge_level[edge]
    rgb[..., 1][edge] = edge_level[edge] * 0.98
    rgb[..., 2][edge] = edge_level[edge] * 0.92
    sweep = (x >= 287) & (x < 513) & (y >= 234) & (y < 332)
    wave_x = (x - 286) / 227
    wave = np.cos((wave_x**2 * 48 + wave_x * 1.2) * 2 * np.pi) > 0
    chroma_area = y > 283
    rgb[..., 0][sweep] = (0.13 + wave * 0.58)[sweep]
    rgb[..., 1][sweep] = (0.16 + (wave ^ chroma_area) * 0.49)[sweep]
    rgb[..., 2][sweep] = (0.17 + (wave ^ chroma_area) * 0.36)[sweep]
    return rgb


def _shadow_rgb(width: int, height: int) -> np.ndarray:
    image = _scene_canvas(width, height, "SHADOW RESPONSE FIELD", "DR—03")
    draw = ImageDraw.Draw(image)
    scales = (26, 39, 55, 74, 96, 128, 164)
    for index, grey in enumerate(scales):
        x = 26 + index * 70
        draw.rectangle((x, 105, x + 58, 289), fill=(grey, grey, grey))
        _text(draw, (x + 7, 311), f"{index + 1:02}", 12, TEXT_MUTED)
    rgb = srgb_decode(np.asarray(image, dtype=np.float32) / 255) * 0.24
    yy, xx = np.mgrid[0:height, 0:width]
    mask = (xx > 32) & (xx < width - 25) & (yy > 173) & (yy < 219)
    bands = np.where((xx // 3) % 2 == 0, 0.0, 0.004)
    rgb[mask] += bands[mask, None]
    return rgb


def _exposure_rgb(width: int, height: int) -> np.ndarray:
    image = _scene_canvas(width, height, "EXPOSURE LADDER", "EX—04")
    draw = ImageDraw.Draw(image)
    for i in range(9):
        x = 23 + i * 54
        level = 19 + i * 26
        draw.rectangle((x, 86, x + 48, 158), fill=(level, level, level))
        _text(draw, (x + 14, 171), f"{i - 4:+d}", 10, TEXT_MUTED)
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.hypot((xx - 323) / 110, (yy - 246) / 84)
    core = np.exp(-distance ** 2 * 2.3)
    mask = (yy > 195) & (yy < 322)
    luminance = 0.012 + 0.34 * core + 1.3 * np.maximum(core - 0.76, 0)
    stripes = np.where((xx // 5) % 2 == 0, 0.94, 1.06)
    rgb = srgb_decode(np.asarray(image, dtype=np.float32) / 255) * 0.9
    for i, weight in enumerate((1.17, 1.0, 0.91)):
        rgb[..., i][mask] = (luminance * stripes * weight)[mask]
    return rgb


def _spec(key: str) -> DemoSpec:
    if key not in BLUEPRINTS:
        raise InputError("Unknown synthetic scene.")
    index = SCENES.index(key)
    return DemoSpec(
        key, (722, 4096, 620, 1412)[index],
        (Layout("RGGB", 20, 14), Layout("BGGR", 11, 9),
         Layout("GRBG", 18, 9), Layout("GBRG", 7, 16))[index],
        (Box(0, 0, 540, 360), Box(7, 5, 525, 345),
         Box(10, 8, 520, 344), Box(4, 4, 532, 352))[index],
        (11.0, 6.0, 22.5, 10.0)[index],
        ((1.0,), (1.0,), (1.0, 1.0), (0.75, 2.8))[index],
        ((1.10, 0.97, 0.79), (1.0, 1.0, 0.99),
         (1.04, 1, 0.86), (1.11, 1.0, 0.89))[index],
        (0.42, 0.20, 1.0, 0.58)[index],
        (0.24, 0.12, 2.7, 0.45)[index],
    )


def _sensor_model(spec: DemoSpec, gain: float, exposure: float, *, mask=False) -> Sensor:
    if mask:
        black = (0.0,) * 4
        white = (1.0,) * 4
    else:
        black = BASE_BLACK
        white = BASE_WHITE
    return Sensor(spec.layout, black, white, spec.active, exposure, gain, "Synthetic ML-12")


def _capture(
    values: np.ndarray, name: str, sensor: Sensor, *, maximum: int = 4095
) -> DecodedCapture:
    array = values.astype(np.uint16)
    pgm = encode_pgm(array, maximum)
    raster = Raster(array, maximum, "Synthetic uint16")
    return DecodedCapture(raster, sensor.as_dict(), name, digest_bytes(pgm), True)


def _response_field(spec: DemoSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    width, height = SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    r2 = ((xx / width - 0.47) * 1.37) ** 2 + ((yy / height - 0.48) * 1.13) ** 2
    vignette = (1 - r2 * 0.48).clip(0.60, 1)
    response = vignette * (1 + 0.022 * rng.normal(size=(height, width)))
    rows = (
        np.sin(yy[:, 0] * 2 * np.pi / 19) * spec.row_band
        + np.sin(yy[:, 0] * 2 * np.pi / 58) * (spec.row_band * 0.45)
    )
    bias = 5 + 0.08 * spec.exposure + rows[:, None] + rng.normal(0, 0.62, (height, width))
    return response, bias


def _defect_field(rng: np.random.Generator) -> np.ndarray:
    width, height = SIZE
    defects = np.zeros((height, width), dtype=bool)
    for flat in rng.choice(defects.size, size=32, replace=False):
        defects.flat[flat] = True
    return defects


def _scene_codes(
    rgb: np.ndarray, spec: DemoSpec, response: np.ndarray, bias: np.ndarray,
    defects: np.ndarray, gain: float, rng: np.random.Generator,
) -> np.ndarray:
    width, height = SIZE
    linear = sample_scene(rgb * np.array(spec.spectral_bias, np.float32), spec.layout)
    accumulated = linear * response * phase_field((height, width), spec.layout, (1, 1, 0.992, 1))
    black = phase_field((height, width), spec.layout, BASE_BLACK)
    white = phase_field((height, width), spec.layout, BASE_WHITE)
    shot = rng.normal(size=linear.shape) * np.sqrt(np.maximum(accumulated, 0.001)) * spec.noise * 16
    read = rng.normal(size=linear.shape) * spec.noise * 2.1
    values = (accumulated * (white - black) + shot) * gain + black + bias + read
    values[defects] = white[defects]
    active = np.zeros(linear.shape, dtype=bool)
    spec.active.crop(active)[:] = True
    values[~active] = (black + bias + read)[~active]
    return np.minimum(np.maximum(np.rint(values), 0), white).astype(np.uint16)


def make_demo(key: str) -> list[tuple[str, DecodedCapture]]:
    spec = _spec(key)
    rng = np.random.default_rng(spec.seed)
    width, height = SIZE
    render = {
        "color-target": _target_rgb,
        "resolution": _resolution_rgb,
        "shadow-register": _shadow_rgb,
        "exposure-pair": _exposure_rgb,
    }[key]
    rgb = render(width, height)
    if key != "resolution":
        rgb = ndimage.gaussian_filter(rgb, sigma=(0.33, 0.33, 0), mode="nearest")
    response, bias = _response_field(spec, rng)
    defects = _defect_field(rng)
    items = []
    for name, gain in zip(BLUEPRINTS[key]["names"], spec.gains):
        sensor = _sensor_model(spec, gain, spec.exposure)
        codes = _scene_codes(rgb, spec, response, bias, defects, gain, rng)
        items.append(("scene", _capture(codes, name, sensor)))
    black = phase_field((height, width), spec.layout, BASE_BLACK)
    dark = np.rint(black + bias + rng.normal(0, 0.25, black.shape))
    flat = np.rint(
        black + response * phase_field(black.shape, spec.layout, (1780, 1870, 1850, 1670))
        + rng.normal(0, 1.55, black.shape)
    )
    sensor = _sensor_model(spec, 1.0, spec.exposure)
    items.append(("dark", _capture(dark, "Reference · dark", sensor)))
    items.append(("flat", _capture(flat, "Reference · flat", sensor)))
    items.append((
        "defects",
        _capture(defects, "Reference · marked sensor sites", _sensor_model(
            spec, 1.0, spec.exposure, mask=True,
        ), maximum=1),
    ))
    return items


def suggested_regions(key: str) -> list[dict]:
    if key == "color-target":
        return [
            {"label": "Neutral reference", "box": Box(71, 280, 118, 29).as_dict()},
            {"label": "Chroma bars", "box": Box(240, 272, 202, 43).as_dict()},
        ]
    if key == "resolution":
        return [
            {"label": "Slanted edge", "box": Box(353, 115, 96, 70).as_dict()},
            {"label": "Chroma sweep", "box": Box(293, 285, 211, 42).as_dict()},
        ]
    if key == "shadow-register":
        return [
            {"label": "Dim neutral", "box": Box(169, 113, 38, 48).as_dict()},
            {"label": "Uniform readout strip", "box": Box(319, 111, 38, 164).as_dict()},
        ]
    return [
        {"label": "Bright spot", "box": Box(300, 218, 49, 53).as_dict()},
        {"label": "Low exposure step", "box": Box(78, 94, 40, 54).as_dict()},
    ]


def chart_swatches() -> list[dict]:
    result = []
    for index, col in enumerate(PATCHES):
        row, column = divmod(index, 6)
        x = 27 + column * 71
        y = 91 + row * 56
        # Match the synthetic target illumination and avoid the patch labels and edges.
        srgb = np.asarray(col, dtype=np.float32) / 255
        from .development import srgb_encode
        goal = srgb_encode(srgb_decode(srgb) * 0.86)
        result.append({
            "name": f"Patch {index + 1:02}",
            "region": Box(x + 18, y + 13, 33, 25).as_dict(),
            "srgb": np.round(goal, 6).tolist(),
        })
    return result
