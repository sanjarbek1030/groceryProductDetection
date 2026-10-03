"""Product Recognition Checkout - application entry point.

Run from the project folder:   python main.py
This file handles input/output (camera, window, keys, video writer).
The decision logic lives in pipeline.py.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import config
from checkout.cart_manager import CartManager
from checkout.checkout_zone import CheckoutZone
from database.product_database import ProductDatabase, ProductDatabaseError
from detector.product_detector import ModelLoadError, ProductDetector
from pipeline import CheckoutPipeline
from recognition.product_recognizer import ProductRecognizer
from tracker.object_tracker import ObjectTracker
from ui.overlay import FrameStats, OverlayRenderer

logger = logging.getLogger("checkout")

KEY_QUIT, KEY_ESC = ord("q"), 27
DEFAULT_FPS = 30.0


class SourceError(RuntimeError):
    """Raised when the video file or camera cannot be used."""


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


class CheckoutApp:
    """Owns the capture loop, window, keyboard handling and video writer."""

    def __init__(self, detector: Optional[Any] = None) -> None:
        self._database = ProductDatabase(config.PRODUCTS_JSON)
        self._detector = detector or ProductDetector(
            model_path=config.MODEL_PATH,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            iou_threshold=config.IOU_THRESHOLD,
            tracker_config=config.TRACKER_CONFIG,
            image_size=config.IMAGE_SIZE,
            device=config.DEVICE,
        )
        logger.info("Model loaded")
        self._warn_about_missing_metadata()

        self._cart = CartManager()
        self._tracker = ObjectTracker(config.HISTORY_SIZE, config.TRACK_TIMEOUT_FRAMES)
        self._recognizer = ProductRecognizer(
            self._database, config.MIN_STABLE_FRAMES,
            config.CONFIRMATION_CONFIDENCE, config.MIN_CLASS_AGREEMENT,
        )
        self._renderer = OverlayRenderer(
            self._database, config.PANEL_WIDTH, config.DISPLAY_CONFIDENCE,
            config.DISPLAY_TRACK_ID, config.DISPLAY_CENTROID, config.DEBUG_MODE,
        )
        self._pipeline: Optional[CheckoutPipeline] = None  # built at first frame
        self._writer: Optional[cv2.VideoWriter] = None
        self._source_fps = DEFAULT_FPS
        self._frame_index = 0
        self._failed_reads = 0
        self._paused = False
        self._fps = 0.0
        self._last_tick: Optional[float] = None
        self._last_canvas: Optional[np.ndarray] = None
        self._last_stats = FrameStats()

    # ----------------------------------------------------------------- setup
    def _warn_about_missing_metadata(self) -> None:
        names = getattr(self._detector, "class_names", {})
        missing = [n for n in names.values() if n not in self._database]
        if missing:
            logger.warning("Model classes missing from products.json: %s", ", ".join(missing))

    def _open_capture(self) -> cv2.VideoCapture:
        if config.USE_WEBCAM:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
            if not cap.isOpened():
                raise SourceError(
                    f"Cannot open camera {config.CAMERA_INDEX}. Check that it is connected, "
                    "not used by another app, or try another CAMERA_INDEX."
                )
            if config.VIDEO_WIDTH and config.VIDEO_HEIGHT:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
            logger.info("Camera initialized (index %d)", config.CAMERA_INDEX)
        else:
            path = Path(config.INPUT_VIDEO)
            if not path.is_file():
                raise SourceError(
                    f"Input video not found: {path}\n"
                    "Put a video there, or set USE_WEBCAM = True in config.py."
                )
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise SourceError(f"Cannot open video (unsupported or corrupted?): {path}")
            logger.info("Video opened: %s", path.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps and not math.isnan(fps) and fps > 1 else DEFAULT_FPS
        return cap

    def _init_for_frame_size(self, width: int, height: int) -> None:
        """Create everything that depends on the frame size (zone, writer)."""
        zone = CheckoutZone.from_config(
            config.CHECKOUT_ZONE, config.CHECKOUT_ZONE_NORMALIZED, width, height
        )
        self._pipeline = CheckoutPipeline(
            self._database, zone, self._tracker, self._recognizer, self._cart,
            config.REMOVE_ITEM_WHEN_TRACK_LOST,
        )
        logger.info("Frame size %dx%d, checkout zone %s", width, height, zone.as_tuple)
        if config.SAVE_OUTPUT and not config.USE_WEBCAM:
            self._writer = self._create_writer(width + config.PANEL_WIDTH, height)

    def _create_writer(self, width: int, height: int) -> Optional[cv2.VideoWriter]:
        out = Path(config.OUTPUT_VIDEO)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Output folder unavailable (%s); video will not be saved", exc)
            return None
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self._source_fps, (width, height))
        if not writer.isOpened():
            logger.warning("Could not create output video %s; video will not be saved", out)
            return None
        logger.info("Saving annotated video to %s", out)
        return writer

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        cap = self._open_capture()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not config.USE_WEBCAM else 0
        try:
            self._loop(cap, total_frames)
        finally:
            cap.release()
            if self._writer is not None:
                self._writer.release()
            cv2.destroyAllWindows()
            logger.info("Finished. Final cart: %s | total %.2f",
                        self._cart.contents(), self._cart.total)

    def _loop(self, cap: cv2.VideoCapture, total_frames: int) -> None:
        while True:
            if not self._paused:
                frame, finished = self._read_frame(cap, total_frames)
                if finished:
                    break
                if frame is not None:
                    self._last_canvas = self._process_frame(frame)
                    if self._writer is not None:
                        self._writer.write(self._last_canvas)

            if config.SHOW_WINDOW and self._last_canvas is not None:
                cv2.imshow(config.WINDOW_NAME, self._last_canvas)
                key = cv2.waitKey(30 if self._paused else 1) & 0xFF
                window_closed = cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                if window_closed or not self._handle_key(key):
                    break

    def _read_frame(self, cap: cv2.VideoCapture, total_frames: int) -> Tuple[Optional[np.ndarray], bool]:
        """Return (frame, finished). A bad read yields (None, False) for webcams."""
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            self._failed_reads = 0
            return frame, False

        self._failed_reads += 1
        if not config.USE_WEBCAM:
            if total_frames > 0 and self._frame_index < total_frames * 0.98:
                logger.warning("Video ended early at frame %d of %d (file may be corrupted)",
                               self._frame_index, total_frames)
            else:
                logger.info("End of video")
            return None, True
        if self._failed_reads >= config.MAX_CONSECUTIVE_READ_FAILURES:
            logger.error("Camera stopped delivering frames; stopping")
            return None, True
        return None, False

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if config.VIDEO_WIDTH and config.VIDEO_HEIGHT:
            if (frame.shape[1], frame.shape[0]) != (config.VIDEO_WIDTH, config.VIDEO_HEIGHT):
                frame = cv2.resize(frame, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
        if self._pipeline is None:
            self._init_for_frame_size(frame.shape[1], frame.shape[0])
        assert self._pipeline is not None

        self._update_fps()
        detections = self._detector.track(frame)
        timestamp = self._frame_index / self._source_fps
        tracks = self._pipeline.process(detections, self._frame_index, timestamp)
        self._frame_index += 1

        self._last_stats = FrameStats(
            fps=self._fps, detected=len(detections),
            active_tracks=len(tracks), paused=False,
        )
        return self._renderer.render(frame, tracks, self._pipeline.zone, self._cart, self._last_stats)

    def _update_fps(self) -> None:
        """Smoothed frames-per-second of the whole loop (measured, not assumed)."""
        now = time.perf_counter()
        if self._last_tick is not None and now > self._last_tick:
            instant = 1.0 / (now - self._last_tick)
            self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant
        self._last_tick = now

    # ------------------------------------------------------------------ keys
    def _handle_key(self, key: int) -> bool:
        """Return False to quit."""
        if key in (KEY_QUIT, KEY_ESC):
            return False
        if key == ord("c"):
            self._cart.clear()
            logger.info("Cart cleared")
        elif key == ord("r"):
            self._reset_all()
        elif key == ord("p"):
            self._paused = not self._paused
            logger.info("Paused" if self._paused else "Resumed")
        elif key == ord("s"):
            self._save_screenshot()
        elif key == ord("d"):
            self._renderer.debug_mode = not self._renderer.debug_mode
            logger.info("Debug mode %s", "ON" if self._renderer.debug_mode else "OFF")
        return True

    def _reset_all(self) -> None:
        """Clear tracks, cart, counters and the YOLO tracker memory."""
        if self._pipeline is not None:
            self._pipeline.reset()
        else:
            self._cart.clear()
        self._detector.reset_tracking()
        self._last_tick, self._fps = None, 0.0
        logger.info("Full reset done")

    def _save_screenshot(self) -> None:
        if self._last_canvas is None:
            return
        try:
            Path(config.SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)
            path = Path(config.SCREENSHOT_DIR) / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
            if cv2.imwrite(str(path), self._last_canvas):
                logger.info("Screenshot saved: %s", path)
            else:
                logger.warning("Could not write screenshot to %s", path)
        except OSError as exc:
            logger.warning("Screenshot failed: %s", exc)


def main() -> int:
    configure_logging()
    try:
        config.validate_config()
        CheckoutApp().run()
        return 0
    except (config.ConfigError, ModelLoadError, ProductDatabaseError, SourceError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())

