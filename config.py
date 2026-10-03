"""Central configuration for Product Recognition Checkout.

Every tunable value lives here so no other file hard-codes numbers or paths.
Edit this file, not the modules, when you want to change behaviour.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODEL_PATH = BASE_DIR / "models" / "best.pt"
PRODUCTS_JSON = BASE_DIR / "database" / "products.json"
INPUT_VIDEO = BASE_DIR / "input" / "checkout_input.mp4"
OUTPUT_VIDEO = BASE_DIR / "output" / "checkout_output.mp4"
SCREENSHOT_DIR = BASE_DIR / "output" / "screenshots"

# --------------------------------------------------------------------------- #
# Video source / output
# --------------------------------------------------------------------------- #
USE_WEBCAM = False          # True -> live camera, False -> INPUT_VIDEO
CAMERA_INDEX = 0            # 0 is usually the default webcam
VIDEO_WIDTH: Optional[int] = None    # None -> keep the source resolution
VIDEO_HEIGHT: Optional[int] = None   # (set BOTH to force a resolution)
SAVE_OUTPUT = True          # only used when processing a video file
SHOW_WINDOW = True
MAX_CONSECUTIVE_READ_FAILURES = 30   # webcam: give up after this many bad reads

# --------------------------------------------------------------------------- #
# Detection (YOLO)
# --------------------------------------------------------------------------- #
# Low-ish on purpose: this only filters pure noise. The stricter decision
# ("is this reliable enough for the cart?") uses CONFIRMATION_CONFIDENCE below.
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.50        # NMS overlap threshold
IMAGE_SIZE = 640            # inference resolution
DEVICE: Optional[str] = None   # None = auto, "cpu", or "0" for the first GPU

# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
TRACKER_CONFIG = "bytetrack.yaml"   # or "botsort.yaml"
TRACK_TIMEOUT_FRAMES = 30   # forget a track after this many frames unseen
HISTORY_SIZE = 15           # recent predictions kept per track (voting window)

# --------------------------------------------------------------------------- #
# Temporal verification / confidence handling
# --------------------------------------------------------------------------- #
MIN_STABLE_FRAMES = 8             # frames of consistent prediction needed
CONFIRMATION_CONFIDENCE = 0.75    # mean confidence needed to trust a product
MIN_CLASS_AGREEMENT = 0.70        # share of recent frames that must agree

# --------------------------------------------------------------------------- #
# Checkout zone
# --------------------------------------------------------------------------- #
# Normalised (0..1) coordinates work for any resolution. To use pixels instead
# (e.g. 500, 400, 1400, 900 on a 1920x1080 video) set the flag to False.
CHECKOUT_ZONE: Dict[str, float] = {"x1": 0.25, "y1": 0.45, "x2": 0.95, "y2": 0.95}
CHECKOUT_ZONE_NORMALIZED = True

# --------------------------------------------------------------------------- #
# Removal (first version: addition only)
# --------------------------------------------------------------------------- #
# False because a track can vanish simply because it was hidden for a moment.
# Removing items on track loss would then create wrong carts.
REMOVE_ITEM_WHEN_TRACK_LOST = False

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
WINDOW_NAME = "Product Recognition Checkout"
PANEL_WIDTH = 380
DISPLAY_CONFIDENCE = True
DISPLAY_TRACK_ID = True
DISPLAY_CENTROID = True
DEBUG_MODE = False          # extra per-track information on screen
LOG_LEVEL = "INFO"          # "DEBUG" prints status transitions as well


class ConfigError(ValueError):
    """Raised when a value in this file is invalid."""


def validate_config() -> None:
    """Check configuration values and raise ConfigError with a clear message."""
    errors = []

    for name, value in (
        ("CONFIDENCE_THRESHOLD", CONFIDENCE_THRESHOLD),
        ("IOU_THRESHOLD", IOU_THRESHOLD),
        ("CONFIRMATION_CONFIDENCE", CONFIRMATION_CONFIDENCE),
        ("MIN_CLASS_AGREEMENT", MIN_CLASS_AGREEMENT),
    ):
        if not 0.0 < value <= 1.0:
            errors.append(f"{name} must be in (0, 1], got {value}")

    if MIN_STABLE_FRAMES < 1:
        errors.append("MIN_STABLE_FRAMES must be >= 1")
    if HISTORY_SIZE < MIN_STABLE_FRAMES:
        errors.append("HISTORY_SIZE must be >= MIN_STABLE_FRAMES")
    if TRACK_TIMEOUT_FRAMES < 1:
        errors.append("TRACK_TIMEOUT_FRAMES must be >= 1")
    if TRACKER_CONFIG not in ("bytetrack.yaml", "botsort.yaml"):
        errors.append("TRACKER_CONFIG must be 'bytetrack.yaml' or 'botsort.yaml'")
    if PANEL_WIDTH < 200:
        errors.append("PANEL_WIDTH must be >= 200")
    if (VIDEO_WIDTH is None) != (VIDEO_HEIGHT is None):
        errors.append("Set both VIDEO_WIDTH and VIDEO_HEIGHT, or neither")

    missing = [k for k in ("x1", "y1", "x2", "y2") if k not in CHECKOUT_ZONE]
    if missing:
        errors.append(f"CHECKOUT_ZONE is missing keys: {missing}")
    else:
        z = CHECKOUT_ZONE
        if z["x2"] <= z["x1"] or z["y2"] <= z["y1"]:
            errors.append("CHECKOUT_ZONE needs x2 > x1 and y2 > y1")
        if CHECKOUT_ZONE_NORMALIZED and not all(0.0 <= v <= 1.0 for v in z.values()):
            errors.append("Normalised CHECKOUT_ZONE values must be within 0..1")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
