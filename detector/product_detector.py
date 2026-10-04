"""YOLO wrapper: turns a camera frame into clean ``Detection`` objects.

This module is the ONLY place that knows about Ultralytics. The rest of the
project works with plain ``Detection`` dataclasses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the YOLO model cannot be loaded."""


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame.

    ``track_id`` is None for plain detection and filled in when YOLO tracking
    is used (and the tracker has assigned an ID).
    """

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: Optional[int] = None

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def centroid(self) -> Tuple[float, float]:
        """Centre of the box: cx = (x1 + x2) / 2, cy = (y1 + y2) / 2."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


class ProductDetector:
    """Loads a custom-trained YOLO model and runs detection or tracking."""

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float,
        iou_threshold: float,
        tracker_config: str,
        image_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(
                f"Model file not found: {path}\n"
                "Train a custom model (see README, 'Model Training') and copy "
                "best.pt into the models/ folder."
            )
        try:
            from ultralytics import YOLO  # imported lazily: keeps tests light
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        try:
            self._model = YOLO(str(path))
        except Exception as exc:  # corrupted / incompatible weights
            raise ModelLoadError(f"Could not load model '{path}': {exc}") from exc

        self._conf = confidence_threshold
        self._iou = iou_threshold
        self._tracker_config = tracker_config
        self._image_size = image_size
        self._device = device
        self._names: Dict[int, str] = dict(self._model.names)

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._names)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Plain detection (no IDs). Useful to test the model on its own."""
        results = self._model.predict(frame, **self._inference_args())
        return self._parse_results(results)

    def track(self, frame: np.ndarray) -> List[Detection]:
        """Detection + YOLO tracking. Detections carry a ``track_id``.

        ``persist=True`` tells Ultralytics to keep the tracker's memory between
        calls - that is what makes IDs stay the same from frame to frame.
        """
        results = self._model.track(
            frame, persist=True, tracker=self._tracker_config, **self._inference_args()
        )
        return self._parse_results(results)

    def reset_tracking(self) -> None:
        """Forget all tracker memory (new IDs start again from 1)."""
        # Dropping the predictor makes Ultralytics build a fresh tracker.
        self._model.predictor = None
        logger.info("Tracker state reset")

    def _inference_args(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "conf": self._conf,
            "iou": self._iou,
            "imgsz": self._image_size,
            "verbose": False,
        }
        if self._device:
            args["device"] = self._device
        return args

    def _parse_results(self, results: Any) -> List[Detection]:
        """Convert an Ultralytics result into a list of Detection objects."""
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        detections: List[Detection] = []
        for i in range(len(xyxy)):
            confidence = float(confs[i])
            if confidence < self._conf:
                continue
            class_id = int(class_ids[i])
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._names.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                )
            )
        return detections
