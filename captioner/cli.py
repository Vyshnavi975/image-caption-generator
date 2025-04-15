"""
cli.py
======

Command-line interface for the image caption generator.

Usage:
    python caption.py --image path/to/image.jpg
    python caption.py --dir sample_images/
    python caption.py --image photo.png --demo        # force demo mode
    python caption.py --image photo.png --verbose      # show raw measurements
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .heuristics import analyze_image, generate_heuristic_caption, format_analysis_report
from .llm import generate_caption, get_active_backend, CaptionAPIError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def caption_one_image(image_path: str, force_demo: bool = False, verbose: bool = False) -> str:
    """
    Produce a caption for a single image, preferring a real API backend
    unless force_demo is set or no API key is configured. Returns the
    formatted output string (caption plus optional metadata/verbose report).
    """
    path = Path(image_path)
    if not path.exists():
        return f"[error] File not found: {image_path}"
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return f"[error] Unsupported file type: {image_path} (expected one of {sorted(SUPPORTED_EXTENSIONS)})"

    backend = "demo" if force_demo else get_active_backend()
    warnings = []
    lines = [f"Image:   {path.name}"]

    caption = None
    if backend != "demo":
        try:
            caption = generate_caption(str(path))
        except CaptionAPIError as exc:
            warnings.append(f"[warning] {exc}")
            warnings.append("Falling back to demo mode.")
            backend = "demo"

    analysis = None
    if backend == "demo" or verbose:
        analysis = analyze_image(str(path))

    if caption is None:
        caption = generate_heuristic_caption(str(path), analysis=analysis)
        if force_demo:
            backend_label = "demo (forced with --demo)"
        elif warnings:
            backend_label = "demo (fallback: API request failed, see warning below)"
        else:
            backend_label = "demo (heuristic, no API key set)"
    else:
        backend_label = backend

    lines.append(f"Backend: {backend_label}")
    lines.extend(warnings)
    lines.append(f"Caption: {caption}")

    if verbose and analysis is not None:
        lines.append("")
        lines.append("--- raw measurements ---")
        lines.append(format_analysis_report(analysis))

    return "\n".join(lines)


def iter_images_in_dir(dir_path: str):
    p = Path(dir_path)
    for child in sorted(p.iterdir()):
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield child


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caption.py",
        description="Generate a natural-language caption for an image, "
                     "using a vision LLM API if configured, or a local "
                     "heuristic demo mode otherwise.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", metavar="PATH", help="Path to a single image file.")
    group.add_argument("--dir", "-d", metavar="PATH", help="Path to a directory of images to caption.")
    parser.add_argument(
        "--demo", action="store_true",
        help="Force heuristic demo mode even if an API key is configured.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also print the raw image measurements used by demo mode.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.image:
        output = caption_one_image(args.image, force_demo=args.demo, verbose=args.verbose)
        print(output)
        return 1 if output.startswith("[error]") else 0

    # --dir mode
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"[error] Not a directory: {args.dir}", file=sys.stderr)
        return 1

    images = list(iter_images_in_dir(str(dir_path)))
    if not images:
        print(f"[error] No supported image files found in {args.dir}", file=sys.stderr)
        return 1

    had_error = False
    for i, image_path in enumerate(images):
        if i > 0:
            print()
        output = caption_one_image(str(image_path), force_demo=args.demo, verbose=args.verbose)
        print(output)
        had_error = had_error or output.startswith("[error]")

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
