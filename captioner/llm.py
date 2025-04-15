"""
llm.py
======

Real captioning via a vision-capable LLM API. Two backends are supported:

  * Anthropic Claude (used if ANTHROPIC_API_KEY is set) -- tried first.
  * OpenAI GPT-4 vision (used if OPENAI_API_KEY is set) -- used if no
    Anthropic key is available.

If neither key is set, `get_active_backend()` returns "demo" and callers
should fall back to captioner.heuristics instead. Network/SDK errors are
caught and surfaced as CaptionAPIError so the CLI can report them cleanly
(and optionally fall back to demo mode).
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

DEFAULT_PROMPT = (
    "Describe this image in one or two natural, vivid sentences, as a "
    "photo caption would. Mention the main subject(s), setting, and any "
    "notable colors, mood, or action. Do not start with phrases like "
    "'This image shows' -- just describe it directly."
)

_SUPPORTED_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".webp": "image/webp", ".gif": "image/gif"}


class CaptionAPIError(RuntimeError):
    """Raised when a real (non-demo) captioning backend fails."""


def get_active_backend() -> str:
    """
    Return which real backend would be used ("anthropic", "openai"), or
    "demo" if neither API key is configured.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "demo"


def _read_image_as_base64(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    ext = path.suffix.lower()
    mime = _SUPPORTED_MIME.get(ext) or mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with open(path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("ascii")
    return data, mime


def _caption_with_anthropic(image_path: str, prompt: str) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise CaptionAPIError(
            "The 'anthropic' package is required for Claude vision captioning. "
            "Install it with: pip install anthropic"
        ) from exc

    b64_data, mime_type = _read_image_as_base64(image_path)
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK/network error uniformly
        raise CaptionAPIError(f"Anthropic API request failed: {exc}") from exc

    text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    caption = " ".join(text_blocks).strip()
    if not caption:
        raise CaptionAPIError("Anthropic API returned an empty response.")
    return caption


def _caption_with_openai(image_path: str, prompt: str) -> str:
    try:
        import openai
    except ImportError as exc:
        raise CaptionAPIError(
            "The 'openai' package is required for GPT-4 vision captioning. "
            "Install it with: pip install openai"
        ) from exc

    b64_data, mime_type = _read_image_as_base64(image_path)
    client = openai.OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                        },
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise CaptionAPIError(f"OpenAI API request failed: {exc}") from exc

    caption = (response.choices[0].message.content or "").strip()
    if not caption:
        raise CaptionAPIError("OpenAI API returned an empty response.")
    return caption


def generate_caption(image_path: str, prompt: str = DEFAULT_PROMPT) -> str:
    """
    Generate a caption using whichever real backend is active. Raises
    CaptionAPIError if no key is configured or the request fails -- callers
    should check get_active_backend() first, or catch this exception to
    fall back to demo mode.
    """
    backend = get_active_backend()
    if backend == "anthropic":
        return _caption_with_anthropic(image_path, prompt)
    if backend == "openai":
        return _caption_with_openai(image_path, prompt)
    raise CaptionAPIError(
        "No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
        "or use demo mode (captioner.heuristics)."
    )
