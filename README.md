# Image Caption Generator

A small, self-contained Python tool that looks at an image and produces a
natural-language caption describing it.

It works two ways:

1. **Real captioning (recommended)** — if you set an `OPENAI_API_KEY`
   environment variable, the image is sent to GPT-4o vision and you get a
   genuine, content-aware caption ("A golden retriever puppy chasing a red
   ball across a grassy lawn.").
2. **Demo mode (no API key required)** — with no key configured, the tool
   falls back to a local heuristic captioner built on `Pillow` + `numpy`.
   It inspects low-level image properties (brightness, dominant colors,
   aspect ratio, color saturation, and edge/detail density) and fills in a
   template caption like *"A wide, bright image with warm tones and high
   visual complexity."* This is clearly labeled as demo mode in the output
   — **it does not recognize objects, people, or scenes**, only pixel-level
   statistics. It exists so the project is fully demoable with zero setup
   and zero API cost.

## Features

- Single-image or whole-folder captioning from the command line.
- Automatic backend selection: GPT-4o vision → local heuristic demo mode,
  based on whether an API key is set.
- Graceful fallback: if an API call fails (bad key, no network, missing
  SDK), the tool prints a warning and falls back to demo mode instead of
  crashing.
- `--verbose` flag to see the raw measurements (brightness, saturation,
  dominant colors, complexity score) behind a demo-mode caption.
- `--demo` flag to force heuristic mode even when an API key is set (useful
  for testing/demos without spending API credits).
- Three original, programmatically generated sample images (`sample_images/`)
  so you can try it immediately with no images of your own.
- Clean, tested `captioner/` package with no hidden state — easy to read,
  extend, or drop into another project.

## Project structure

```
image-caption-generator/
├── caption.py                  # CLI entry point (python caption.py ...)
├── captioner/
│   ├── __init__.py
│   ├── heuristics.py           # demo-mode: Pillow/numpy image analysis + captioning
│   ├── llm.py                  # real captioning via the OpenAI vision API
│   └── cli.py                  # argument parsing and orchestration
├── sample_images/
│   ├── generate_samples.py     # regenerates the sample images (PIL-drawn, original)
│   ├── sunset_gradient.png
│   ├── cool_shapes.png
│   └── soft_portrait.png
├── tests/
│   └── test_heuristics.py      # unit tests for the heuristic captioner (no API key needed)
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
cd image-caption-generator
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Core dependencies (enough for demo mode + tests):
pip install Pillow numpy pytest

# Optional, only if you want real API-based captions:
pip install openai         # for GPT-4o vision

# or just install everything at once:
pip install -r requirements.txt
```

The sample images are already committed to `sample_images/`, so you don't
need to regenerate them. If you ever want to (or want fresh/different
placeholder art), run:

```bash
python sample_images/generate_samples.py
```

### (Optional) enable real captioning

```bash
export OPENAI_API_KEY="sk-..."
```

With no key set, the tool automatically uses demo mode — nothing else to
configure.

## Usage

Caption a single image:

```bash
python caption.py --image sample_images/sunset_gradient.png
```

Caption every image in a folder:

```bash
python caption.py --dir sample_images/
```

Show the raw measurements behind a demo-mode caption:

```bash
python caption.py --image sample_images/cool_shapes.png --verbose
```

Force demo mode even if an API key is set:

```bash
python caption.py --image sample_images/cool_shapes.png --demo
```

All options:

```
python caption.py --help

usage: caption.py [-h] (--image PATH | --dir PATH) [--demo] [--verbose]

  --image PATH, -i PATH   Path to a single image file.
  --dir PATH, -d PATH     Path to a directory of images to caption.
  --demo                  Force heuristic demo mode even if an API key is configured.
  --verbose, -v           Also print the raw image measurements used by demo mode.
```

### Real sample output (demo mode, no API key set)

This is the actual output of running the tool against the bundled sample
images, unedited:

```
$ python caption.py --dir sample_images/ --verbose

Image:   cool_shapes.png
Backend: demo (heuristic, no API key set)
Caption: A square-ish, dark image with cool tones (dominant colors: black, teal, and blue tones), richly saturated, and simple visual detail.

--- raw measurements ---
Dimensions:      400x400 px (aspect ratio 1.0, square)
Brightness:      83.9/255 (dark)
Saturation:      0.68 (high)
Dominant colors: black, teal, blue
Color tone:      cool
Complexity:      0.118 (simple)
File size:       7.8 KB

Image:   soft_portrait.png
Backend: demo (heuristic, no API key set)
Caption: A tall, bright image with warm tones (dominant colors: white and pink tones), muted / low-saturation, and simple visual detail.

--- raw measurements ---
Dimensions:      300x450 px (aspect ratio 0.667, tall)
Brightness:      225.0/255 (bright)
Saturation:      0.177 (low)
Dominant colors: white, pink
Color tone:      warm
Complexity:      0.003 (simple)
File size:       5.5 KB

Image:   sunset_gradient.png
Backend: demo (heuristic, no API key set)
Caption: A wide, moderately lit image with warm tones (dominant colors: orange, brown, and yellow tones), richly saturated, and simple visual detail.

--- raw measurements ---
Dimensions:      600x300 px (aspect ratio 2.0, wide)
Brightness:      122.2/255 (moderately lit)
Saturation:      0.624 (high)
Dominant colors: orange, brown, yellow
Color tone:      warm
Complexity:      0.004 (simple)
File size:       2.3 KB
```

If you set `OPENAI_API_KEY` and re-run the same command, `Backend:` will
read `openai` and `Caption:` will be a real, content-aware description
generated by the vision model instead.

## How demo mode works

`captioner/heuristics.py` computes, for each image:

- **Orientation** — from width/height aspect ratio: `wide`, `tall`, or
  `square`.
- **Brightness** — mean pixel luminance (grayscale average), bucketed into
  `dark` / `moderately lit` / `bright`.
- **Dominant colors** — the image is downsampled to a coarse grid, each
  sampled pixel is mapped to the nearest of ~12 named colors (red, blue,
  teal, etc.) by simple Euclidean RGB distance, and the most frequent names
  are reported.
- **Saturation** — mean saturation channel from an HSV conversion.
- **Color tone** — whether the dominant colors skew warm, cool, mixed, or
  neutral.
- **Complexity** — a lightweight edge-density estimate: finite-difference
  gradients (`numpy.diff`, no OpenCV/SciPy needed) are computed over the
  grayscale image, and the fraction of pixels with a gradient magnitude
  above a threshold becomes a 0–1 "how much detail/edges are present" score.

These features are plugged into a caption template. This is intentionally
*not* object recognition — a photo of a red apple and a photo of a red
sports car could receive a very similar demo-mode caption, since both are
"warm-toned" images. Real object/scene understanding requires the LLM
vision backend (see above).

## Running the tests

```bash
python -m pytest tests/ -v
```

The tests only exercise `captioner/heuristics.py` (color naming, brightness
bucketing, complexity scoring, orientation detection, and caption
generation) against synthetic images built on the fly with Pillow/numpy —
no API key or network access is needed.

## Notes / limitations

- Demo-mode captions describe *visual statistics*, not *content* — they
  will never say "a cat" or "a mountain." Treat them as a stand-in for a
  real vision model, not a substitute for one.
- The real API backend makes network calls and consumes API credits/quota;
  it requires `pip install openai` — not required for demo mode.
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`.
- All sample images in `sample_images/` are generated programmatically by
  `sample_images/generate_samples.py` using basic PIL drawing calls
  (gradients, ellipses, rectangles) — no external or copyrighted images are
  used anywhere in this project.

## License

MIT — see [LICENSE](LICENSE).
