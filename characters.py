"""
Character image loading for the dialogue pipeline.

Loads stewie.png and petergriffin.png, removes the white/checkerboard
background, crops tight to the character, and caches the result as RGBA
numpy arrays sized for the 1080×1920 canvas — no API calls, pure PIL/scipy.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

_HERE = Path(__file__).parent

STEWIE_PATH = _HERE / "stewie.png"
PETER_PATH  = _HERE / "petergriffin.png"

# Canvas dimensions (must match video.py W, H)
CANVAS_W, CANVAS_H = 1080, 1920

# Display heights — Stewie is a baby, noticeably shorter than Peter
PETER_HEIGHT  = 620
STEWIE_HEIGHT = 380

# Horizontal margins from each edge (never touching the frame)
SIDE_MARGIN = 50   # px from left/right edge

_CACHE: dict = {}


def _remove_bg_and_crop(path: Path, brightness: int) -> Image.Image:
    """
    Remove the white/checkerboard background via connected-component
    flood-fill from border pixels, then crop tight to the character.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # Near-gray (catches both pure white and checkerboard gray) + bright
    grayness = (np.abs(r - g) < 15) & (np.abs(g - b) < 15) & (np.abs(r - b) < 15)
    bright   = (r + g + b) / 3 > brightness
    candidate = grayness & bright

    # Keep only background connected to the frame border
    labeled, _ = ndimage.label(candidate)
    border_labels = (
        set(labeled[0, :].tolist()) | set(labeled[-1, :].tolist()) |
        set(labeled[:, 0].tolist()) | set(labeled[:, -1].tolist())
    )
    border_labels.discard(0)
    bg_mask = np.isin(labeled, list(border_labels))

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:,:,:3] = arr.astype(np.uint8)
    rgba[:,:,3]  = np.where(bg_mask, 0, 255)

    out = Image.fromarray(rgba)

    # Tight crop to visible pixels
    ys, xs = np.where(rgba[:,:,3] > 0)
    if len(xs) == 0:
        return out
    pad = 8
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)
    return out.crop((x0, y0, x1, y1))


def _fit_height(img: Image.Image, target_h: int) -> Image.Image:
    scale = target_h / img.height
    return img.resize((int(img.width * scale), target_h), Image.LANCZOS)


def load_characters() -> dict:
    """
    Returns a dict with pre-processed character data, cached after first call.

    Structure:
        {
          "stewie": {"array": np.ndarray RGBA, "w": int, "h": int},
          "peter":  {"array": np.ndarray RGBA, "w": int, "h": int},
        }
    """
    global _CACHE
    if _CACHE:
        return _CACHE

    stewie_img = _fit_height(_remove_bg_and_crop(STEWIE_PATH, brightness=180), STEWIE_HEIGHT)
    peter_img  = _fit_height(_remove_bg_and_crop(PETER_PATH,  brightness=220), PETER_HEIGHT)

    _CACHE = {
        "stewie": {
            "array": np.array(stewie_img),
            "w": stewie_img.width,
            "h": stewie_img.height,
        },
        "peter": {
            "array": np.array(peter_img),
            "w": peter_img.width,
            "h": peter_img.height,
        },
    }
    return _CACHE


def character_position(speaker: str, char_info: dict) -> tuple[int, int]:
    """
    Fixed (x, y) for a character on the CANVAS_W × CANVAS_H frame.
    Stewie on the left, Peter on the right, both bottom-anchored,
    never touching the frame edge.
    """
    if speaker == "peter":
        x = SIDE_MARGIN
    else:
        x = CANVAS_W - char_info["w"] - SIDE_MARGIN
    y = CANVAS_H - char_info["h"]
    return x, y
