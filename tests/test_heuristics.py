"""
Unit tests for captioner.heuristics -- the demo-mode captioner.

These tests only use Pillow/numpy to build synthetic test images in a
temp directory, so they run with no API key and no network access.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captioner.heuristics import (
    analyze_image,
    generate_heuristic_caption,
    format_analysis_report,
    _nearest_color_name,
    _brightness_label,
    _complexity_label,
    _classify_tone,
)


@pytest.fixture()
def tmp_image_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _save(img: Image.Image, directory: Path, name: str) -> str:
    path = directory / name
    img.save(path)
    return str(path)


class TestColorNaming:
    def test_pure_red_maps_to_red(self):
        assert _nearest_color_name((255, 0, 0)) == "red"

    def test_pure_white_maps_to_white(self):
        assert _nearest_color_name((255, 255, 255)) == "white"

    def test_pure_black_maps_to_black(self):
        assert _nearest_color_name((0, 0, 0)) == "black"


class TestBrightnessLabel:
    def test_low_value_is_dark(self):
        assert _brightness_label(10) == "dark"

    def test_mid_value_is_moderate(self):
        assert _brightness_label(128) == "moderately lit"

    def test_high_value_is_bright(self):
        assert _brightness_label(240) == "bright"


class TestComplexityLabel:
    def test_low_score_is_simple(self):
        assert _complexity_label(0.02) == "simple"

    def test_mid_score_is_moderate(self):
        assert _complexity_label(0.2) == "moderately complex"

    def test_high_score_is_high(self):
        assert _complexity_label(0.6) == "highly complex"


class TestColorTone:
    def test_all_warm_is_warm(self):
        assert _classify_tone(["red", "orange"]) == "warm"

    def test_all_cool_is_cool(self):
        assert _classify_tone(["blue", "teal"]) == "cool"

    def test_mixed_is_mixed(self):
        assert _classify_tone(["red", "blue"]) == "mixed"

    def test_grays_are_neutral(self):
        assert _classify_tone(["gray", "black", "white"]) == "neutral"


class TestAnalyzeImage:
    def test_solid_black_image_is_dark(self, tmp_image_dir):
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        path = _save(img, tmp_image_dir, "black.png")
        analysis = analyze_image(path)
        assert analysis.brightness_label == "dark"
        assert analysis.mean_brightness < 20

    def test_solid_white_image_is_bright(self, tmp_image_dir):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        path = _save(img, tmp_image_dir, "white.png")
        analysis = analyze_image(path)
        assert analysis.brightness_label == "bright"
        assert analysis.mean_brightness > 240

    def test_wide_image_orientation(self, tmp_image_dir):
        img = Image.new("RGB", (400, 100), color=(120, 120, 120))
        path = _save(img, tmp_image_dir, "wide.png")
        analysis = analyze_image(path)
        assert analysis.orientation == "wide"
        assert analysis.aspect_ratio > 1

    def test_tall_image_orientation(self, tmp_image_dir):
        img = Image.new("RGB", (100, 400), color=(120, 120, 120))
        path = _save(img, tmp_image_dir, "tall.png")
        analysis = analyze_image(path)
        assert analysis.orientation == "tall"
        assert analysis.aspect_ratio < 1

    def test_square_image_orientation(self, tmp_image_dir):
        img = Image.new("RGB", (200, 200), color=(120, 120, 120))
        path = _save(img, tmp_image_dir, "square.png")
        analysis = analyze_image(path)
        assert analysis.orientation == "square"
        assert analysis.aspect_ratio == pytest.approx(1.0)

    def test_solid_color_image_has_low_complexity(self, tmp_image_dir):
        img = Image.new("RGB", (200, 200), color=(50, 100, 150))
        path = _save(img, tmp_image_dir, "solid.png")
        analysis = analyze_image(path)
        assert analysis.complexity_score < 0.05
        assert analysis.complexity_label == "simple"

    def test_noisy_image_has_higher_complexity_than_solid(self, tmp_image_dir):
        rng = np.random.default_rng(42)
        noisy_array = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
        noisy_img = Image.fromarray(noisy_array, mode="RGB")
        noisy_path = _save(noisy_img, tmp_image_dir, "noisy.png")

        solid_img = Image.new("RGB", (200, 200), color=(50, 100, 150))
        solid_path = _save(solid_img, tmp_image_dir, "solid2.png")

        noisy_analysis = analyze_image(noisy_path)
        solid_analysis = analyze_image(solid_path)

        assert noisy_analysis.complexity_score > solid_analysis.complexity_score

    def test_dominant_color_of_solid_red_image(self, tmp_image_dir):
        img = Image.new("RGB", (100, 100), color=(220, 20, 20))
        path = _save(img, tmp_image_dir, "red.png")
        analysis = analyze_image(path)
        assert "red" in analysis.dominant_colors

    def test_analysis_has_file_metadata(self, tmp_image_dir):
        img = Image.new("RGB", (50, 50), color=(10, 10, 10))
        path = _save(img, tmp_image_dir, "meta.png")
        analysis = analyze_image(path)
        assert analysis.file_path == path
        assert analysis.file_size_kb >= 0


class TestGenerateHeuristicCaption:
    def test_caption_is_nonempty_string(self, tmp_image_dir):
        img = Image.new("RGB", (300, 200), color=(255, 180, 60))
        path = _save(img, tmp_image_dir, "sunny.png")
        caption = generate_heuristic_caption(path)
        assert isinstance(caption, str)
        assert len(caption) > 10

    def test_caption_mentions_orientation_and_brightness(self, tmp_image_dir):
        img = Image.new("RGB", (500, 100), color=(240, 240, 240))
        path = _save(img, tmp_image_dir, "wide_bright.png")
        caption = generate_heuristic_caption(path)
        assert "wide" in caption
        assert "bright" in caption

    def test_caption_reuses_precomputed_analysis(self, tmp_image_dir):
        img = Image.new("RGB", (100, 500), color=(10, 10, 60))
        path = _save(img, tmp_image_dir, "tall_dark.png")
        analysis = analyze_image(path)
        caption = generate_heuristic_caption(path, analysis=analysis)
        assert "tall" in caption
        assert "dark" in caption


class TestFormatAnalysisReport:
    def test_report_contains_key_fields(self, tmp_image_dir):
        img = Image.new("RGB", (150, 150), color=(30, 150, 90))
        path = _save(img, tmp_image_dir, "report.png")
        analysis = analyze_image(path)
        report = format_analysis_report(analysis)
        assert "Dimensions:" in report
        assert "Brightness:" in report
        assert "Dominant colors:" in report
        assert "Complexity:" in report
