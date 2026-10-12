"""Robust image reading shared by the training cache AND test.py.

Using one function for both guarantees train/test preprocessing parity (a classic silent bug source).
Pipeline: decode -> EXIF orientation -> channel normalisation -> resize shorter side to S (bicubic,
antialiased by PIL) -> centre crop S x S -> uint8 array (C, S, S).
"""
import warnings

import numpy as np
from PIL import Image, ImageFile, ImageOps

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".ppm", ".pgm")
LOADER_VERSION = "io-v1"   # bump if preprocessing changes -> invalidates caches

# Legit large photos (e.g. 3648x2736) must not trip PIL's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = 400_000_000


def _to_mode(im: Image.Image, channels: int) -> Image.Image:
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        # 16-bit / 32-bit greyscale: PIL's convert('L') clips at 255, so rescale explicitly.
        a = np.asarray(im, dtype=np.float64)
        mx = float(a.max()) if a.size else 0.0
        if mx > 255:
            a = a / (65535.0 if mx <= 65535 else mx) * 255.0
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        mode = "L"
    if mode == "P":
        im = im.convert("RGBA") if "transparency" in im.info else im.convert("RGB")
        mode = im.mode
    if mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        # Composite transparent pixels onto white (documented choice; applied identically at train/test).
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    if channels == 1:
        return im.convert("L")
    return im.convert("RGB")


def resize_center_crop(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


def load_image(path: str, size: int, channels: int = 3) -> np.ndarray:
    """Return uint8 array of shape (channels, size, size). Raises on unreadable files."""
    try:
        with Image.open(path) as im:
            im.load()
            im = _prep(im, size, channels)
    except OSError as e:
        # Truncated files: try once more in tolerant mode, warn, and use what decodes.
        if "truncated" not in str(e).lower():
            raise
        prev = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as im:
                im.load()
                im = _prep(im, size, channels)
            warnings.warn(f"truncated image decoded partially: {path}")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev
    a = np.asarray(im, dtype=np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def _prep(im, size, channels):
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    im = _to_mode(im, channels)
    return resize_center_crop(im, size)


def probe_image(path):
    """Cheap header read: (width, height, mode) without full decode."""
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode
