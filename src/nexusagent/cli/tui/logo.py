from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from rich.text import Text

from nexusagent.core.paths import find_project_root


FALLBACK_LOGO = [
    "  <      >  ",
    "   /\\__/\\   ",
    "  |  ||  |  ",
    "  |______|  ",
    "   ||  ||   ",
]


def default_logo_path() -> Path:
    return find_project_root() / "assets" / "logo-no-words.png"


def render_logo(path: Path | None = None, *, max_width: int = 36, max_rows: int = 12) -> Text:
    logo_path = path or default_logo_path()
    try:
        return _render_png_logo(str(logo_path), max_width=max_width, max_rows=max_rows)
    except Exception:
        return _render_fallback_logo()


@lru_cache(maxsize=16)
def _render_png_logo(path: str, *, max_width: int, max_rows: int) -> Text:
    from PIL import Image

    image = Image.open(path).convert("RGBA")
    image = _crop_visible(image)
    image.thumbnail((max_width, max_rows * 2), Image.Resampling.NEAREST)
    width, height = image.size
    if height % 2:
        image = _pad_bottom(image)
        width, height = image.size

    pixels = image.load()
    text = Text()
    for y in range(0, height, 2):
        for x in range(width):
            upper = pixels[x, y]
            lower = pixels[x, y + 1]
            text.append(_pixel_char(upper, lower), style=_pixel_style(upper, lower))
        if y + 2 < height:
            text.append("\n")
    return text


def _crop_visible(image: "Image.Image") -> "Image.Image":
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a > 16 and (r < 245 or g < 245 or b < 245):
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return rgba
    padding = 8
    left = max(0, min(xs) - padding)
    top = max(0, min(ys) - padding)
    right = min(width, max(xs) + padding + 1)
    bottom = min(height, max(ys) + padding + 1)
    return rgba.crop((left, top, right, bottom))


def _pad_bottom(image: "Image.Image") -> "Image.Image":
    from PIL import Image

    width, height = image.size
    padded = Image.new("RGBA", (width, height + 1), (255, 255, 255, 0))
    padded.paste(image, (0, 0))
    return padded


def _pixel_char(upper: tuple[int, int, int, int], lower: tuple[int, int, int, int]) -> str:
    upper_visible = _visible(upper)
    lower_visible = _visible(lower)
    if upper_visible and lower_visible:
        return "▀"
    if upper_visible:
        return "▀"
    if lower_visible:
        return "▄"
    return " "


def _pixel_style(upper: tuple[int, int, int, int], lower: tuple[int, int, int, int]) -> str:
    upper_visible = _visible(upper)
    lower_visible = _visible(lower)
    if upper_visible and lower_visible:
        return f"rgb({_rgb(upper)}) on rgb({_rgb(lower)})"
    if upper_visible:
        return f"rgb({_rgb(upper)})"
    if lower_visible:
        return f"rgb({_rgb(lower)})"
    return ""


def _visible(pixel: tuple[int, int, int, int]) -> bool:
    r, g, b, a = pixel
    return a > 16 and (r < 245 or g < 245 or b < 245)


def _rgb(pixel: tuple[int, int, int, int]) -> str:
    return ",".join(str(channel) for channel in pixel[:3])


def _render_fallback_logo(lines: Iterable[str] = FALLBACK_LOGO) -> Text:
    text = Text()
    for index, line in enumerate(lines):
        text.append(line, style="bold rgb(255,120,42)")
        if index < len(FALLBACK_LOGO) - 1:
            text.append("\n")
    return text
