"""
heuristics.py
=============

Demo-mode captioning: produces a template-based natural-language
description of an image using classic image-processing techniques only
(Pillow + numpy). No machine learning / object recognition is involved --
this is explicitly a fallback for when no vision-capable LLM API key is
configured, and the captions it produces should be read as a description
of low-level visual properties (color, brightness, shape, complexity),
not of the image's *content* or *subject matter*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# Named color buckets used to translate an (R, G, B) pixel into a word.
# Order matters only for tie-breaking; matching is by nearest-distance.
# --------------------------------------------------------------------------
_NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (200, 30, 30),
    "orange": (230, 130, 30),
    "yellow": (230, 210, 40),
    "green": (40, 160, 70),
    "teal": (30, 150, 150),
    "blue": (40, 90, 200),
    "purple": (130, 60, 170),
    "pink": (230, 130, 180),
    "brown": (110, 70, 40),
}

# Which named colors count as "warm" vs "cool" for the overall tone summary.
_WARM_COLORS = {"red", "orange", "yellow", "pink", "brown"}
_COOL_COLORS = {"blue", "teal", "green", "purple"}


@dataclass
class ImageAnalysis:
    """Container for the low-level, non-semantic features we measure."""

    width: int
    height: int
    aspect_ratio: float
    orientation: str                 # "wide", "tall", or "square"
    mean_brightness: float           # 0-255
    brightness_label: str            # "dark", "moderately lit", "bright"
    dominant_colors: List[str]       # e.g. ["blue", "white"]
    color_tone: str                  # "warm", "cool", "neutral", or "mixed"
    saturation: float                # 0-1, average
    complexity_score: float          # 0-1, edge-density based
    complexity_label: str            # "simple", "moderately complex", "highly complex"
    file_path: str = ""
    file_size_kb: float = 0.0
    extra_notes: List[str] = field(default_factory=list)


def _nearest_color_name(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    best_name, best_dist = "gray", float("inf")
    for name, (cr, cg, cb) in _NAMED_COLORS.items():
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _dominant_colors(rgb_array: np.ndarray, k: int = 3) -> List[str]:
    """
    Cheap dominant-color extraction: downsample the image to a small grid,
    map each sampled pixel to the nearest named color bucket, and return
    the most frequent buckets (deduplicated, most common first).
    """
    h, w, _ = rgb_array.shape
    # Sample on a coarse grid rather than every pixel for speed.
    step_y = max(1, h // 24)
    step_x = max(1, w // 24)
    samples = rgb_array[::step_y, ::step_x].reshape(-1, 3)

    counts = {}
    for px in samples:
        name = _nearest_color_name(tuple(int(c) for c in px))
        counts[name] = counts.get(name, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:k]]


def _classify_tone(dominant_colors: List[str]) -> str:
    warm = sum(1 for c in dominant_colors if c in _WARM_COLORS)
    cool = sum(1 for c in dominant_colors if c in _COOL_COLORS)
    if warm and not cool:
        return "warm"
    if cool and not warm:
        return "cool"
    if warm and cool:
        return "mixed"
    return "neutral"


def _brightness_label(mean_brightness: float) -> str:
    if mean_brightness < 85:
        return "dark"
    if mean_brightness < 170:
        return "moderately lit"
    return "bright"


def _complexity_label(score: float) -> str:
    if score < 0.12:
        return "simple"
    if score < 0.30:
        return "moderately complex"
    return "highly complex"


def _edge_complexity(gray_array: np.ndarray) -> float:
    """
    Estimate visual complexity via simple finite-difference gradients
    (a lightweight Sobel-like operator built from plain numpy diffs --
    no scipy/opencv dependency needed).

    Returns a 0-1 score: fraction of pixels whose local gradient magnitude
    exceeds a threshold, i.e. a rough "how much edge/detail is present".
    """
    arr = gray_array.astype(np.float32)
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)

    # Crop to matching shape so gx/gy can be combined.
    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    gx = gx[:h, :w]
    gy = gy[:h, :w]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    threshold = 25.0  # out of a possible 0-360-ish range
    edge_fraction = float(np.mean(magnitude > threshold))
    return min(1.0, edge_fraction)


def analyze_image(image_path: str) -> ImageAnalysis:
    """Load an image and compute the low-level features used for demo captions."""
    path = Path(image_path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size

        # Keep analysis fast even for large images.
        analysis_img = img.copy()
        analysis_img.thumbnail((400, 400))

        rgb_array = np.asarray(analysis_img, dtype=np.uint8)
        gray_array = np.asarray(analysis_img.convert("L"), dtype=np.uint8)

        mean_brightness = float(gray_array.mean())

        # Average saturation via simple HSV conversion.
        hsv_array = np.asarray(analysis_img.convert("HSV"), dtype=np.uint8)
        saturation = float(hsv_array[:, :, 1].mean()) / 255.0

        dominant = _dominant_colors(rgb_array, k=3)
        tone = _classify_tone(dominant)
        complexity = _edge_complexity(gray_array)

    aspect_ratio = width / height if height else 1.0
    if aspect_ratio > 1.15:
        orientation = "wide"
    elif aspect_ratio < 0.87:
        orientation = "tall"
    else:
        orientation = "square"

    try:
        file_size_kb = path.stat().st_size / 1024.0
    except OSError:
        file_size_kb = 0.0

    return ImageAnalysis(
        width=width,
        height=height,
        aspect_ratio=round(aspect_ratio, 3),
        orientation=orientation,
        mean_brightness=round(mean_brightness, 1),
        brightness_label=_brightness_label(mean_brightness),
        dominant_colors=dominant,
        color_tone=tone,
        saturation=round(saturation, 3),
        complexity_score=round(complexity, 3),
        complexity_label=_complexity_label(complexity),
        file_path=str(path),
        file_size_kb=round(file_size_kb, 1),
    )


def _describe_colors(dominant_colors: List[str]) -> str:
    if not dominant_colors:
        return "an unclear color palette"
    if len(dominant_colors) == 1:
        return f"{dominant_colors[0]} tones"
    if len(dominant_colors) == 2:
        return f"{dominant_colors[0]} and {dominant_colors[1]} tones"
    return f"{dominant_colors[0]}, {dominant_colors[1]}, and {dominant_colors[2]} tones"


def generate_heuristic_caption(image_path: str, analysis: ImageAnalysis = None) -> str:
    """
    Build a template-based caption from an ImageAnalysis. If no analysis is
    supplied, the image is analyzed first.

    NOTE: this is DEMO MODE. It describes color/brightness/shape/complexity
    only -- it does not recognize objects, people, or scenes.
    """
    if analysis is None:
        analysis = analyze_image(image_path)

    orientation_phrase = {
        "wide": "wide",
        "tall": "tall",
        "square": "square-ish",
    }[analysis.orientation]

    tone_phrase = {
        "warm": "warm",
        "cool": "cool",
        "mixed": "mixed warm and cool",
        "neutral": "neutral",
    }[analysis.color_tone]

    saturation_phrase = "richly saturated" if analysis.saturation > 0.45 else (
        "muted / low-saturation" if analysis.saturation < 0.18 else "moderately saturated"
    )

    color_desc = _describe_colors(analysis.dominant_colors)

    caption = (
        f"A {orientation_phrase}, {analysis.brightness_label} image with {tone_phrase} "
        f"tones (dominant colors: {color_desc}), {saturation_phrase}, and "
        f"{analysis.complexity_label} visual detail."
    )
    return caption


def format_analysis_report(analysis: ImageAnalysis) -> str:
    """Human-readable multi-line dump of the raw measurements, for --verbose output."""
    lines = [
        f"Dimensions:      {analysis.width}x{analysis.height} px "
        f"(aspect ratio {analysis.aspect_ratio}, {analysis.orientation})",
        f"Brightness:      {analysis.mean_brightness}/255 ({analysis.brightness_label})",
        f"Saturation:      {analysis.saturation} "
        f"({'high' if analysis.saturation > 0.45 else 'low' if analysis.saturation < 0.18 else 'moderate'})",
        f"Dominant colors: {', '.join(analysis.dominant_colors)}",
        f"Color tone:      {analysis.color_tone}",
        f"Complexity:      {analysis.complexity_score} ({analysis.complexity_label})",
    ]
    if analysis.file_size_kb:
        lines.append(f"File size:       {analysis.file_size_kb} KB")
    return "\n".join(lines)
