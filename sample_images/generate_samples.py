#!/usr/bin/env python3
"""
generate_samples.py

Programmatically creates a few small, original PIL-drawn images for
demoing the captioner out of the box. No external/copyrighted images
are used -- everything here is generated with basic PIL drawing calls.

Run from anywhere:
    python sample_images/generate_samples.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).parent


def make_sunset_gradient(path: Path, size=(600, 300)):
    """A warm, wide horizontal gradient (orange -> pink -> deep purple)."""
    w, h = size
    top = np.array([255, 170, 40], dtype=np.float32)      # warm orange
    bottom = np.array([90, 30, 110], dtype=np.float32)    # deep purple
    t = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1)
    row = top + (bottom - top) * t  # (h, 3)
    pixels = np.repeat(row[:, np.newaxis, :], w, axis=1).astype(np.uint8)
    img = Image.fromarray(pixels, mode="RGB")

    draw = ImageDraw.Draw(img)
    # A simple "sun" circle for a bit of visual interest.
    draw.ellipse([w * 0.62, h * 0.30, w * 0.62 + 90, h * 0.30 + 90], fill=(255, 235, 160))
    img.save(path)


def make_cool_shapes(path: Path, size=(400, 400)):
    """A square, cool-toned image with several flat geometric shapes (higher complexity)."""
    img = Image.new("RGB", size, color=(20, 35, 60))  # dark navy background
    draw = ImageDraw.Draw(img)

    draw.rectangle([30, 30, 180, 180], fill=(40, 130, 200))       # blue square
    draw.ellipse([200, 40, 370, 210], fill=(30, 170, 150))        # teal circle
    draw.polygon([(60, 220), (200, 220), (130, 370)], fill=(90, 60, 180))  # purple triangle
    draw.rectangle([220, 250, 370, 370], fill=(60, 200, 220))     # light teal square

    # Scatter some small lines/dots to bump up edge complexity.
    for i in range(0, 400, 20):
        draw.line([(i, 0), (400 - i, 400)], fill=(255, 255, 255), width=1)

    img.save(path)


def make_soft_portrait_frame(path: Path, size=(300, 450)):
    """A tall, low-complexity, softly lit pastel image (simple shapes only)."""
    img = Image.new("RGB", size, color=(245, 235, 225))  # warm off-white
    draw = ImageDraw.Draw(img)

    w, h = size
    # A soft radial-ish "spotlight" effect using concentric ellipses.
    cx, cy = w // 2, int(h * 0.4)
    for i, radius in enumerate(range(160, 20, -20)):
        shade = 235 - i * 8
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(255, shade, shade - 20 if shade - 20 > 180 else 180),
        )

    # A simple grounding rectangle at the bottom (like a horizon/table line).
    draw.rectangle([0, h - 60, w, h], fill=(210, 190, 170))

    img.save(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_sunset_gradient(OUT_DIR / "sunset_gradient.png")
    make_cool_shapes(OUT_DIR / "cool_shapes.png")
    make_soft_portrait_frame(OUT_DIR / "soft_portrait.png")
    print(f"Sample images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
