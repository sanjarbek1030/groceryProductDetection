"""Track state: remembers everything we know about each physical object.

YOLO's tracker (ByteTrack / BoT-SORT) gives every detection a ``track_id``.
This module keeps one ``TrackedProduct`` per ID so other modules can reason
over TIME instead of over single frames.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from detector.product_detector import Detection

logger = logging.getLogger(__name__)

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


class ProductStatus(str, Enum):
    """Where a track is in its life cycle (also shown on screen)."""

    VERIFYING = "VERIFYING"
    STABLE = "STABLE"
    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN PRODUCT"


@dataclass
class TrackedProduct:
    """Everything known about one tracked physical object."""

    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
    centroid: Point
    first_seen_frame: int
    last_seen_frame: int
    last_seen_timestamp: float
    previous_centroid: Optional[Point] = None
    avg_confidence: float = 0.0
    stable_frame_count: int = 0
    in_zone: bool = False
    entered_checkout_zone: bool = False
    confirmed: bool = False
    counted: bool = False
    status: ProductStatus = ProductStatus.VERIFYING
    # Recent (class_name, confidence) observations - the voting window.
    history: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=15))


class ObjectTracker:
    """Creates, updates and prunes ``TrackedProduct`` records."""

    def __init__(self, history_size: int, timeout_frames: int) -> None:
        self._history_size = history_size
        self._timeout_frames = timeout_frames
        self._tracks: Dict[int, TrackedProduct] = {}
        self.missing_id_count = 0  # detections skipped because they had no ID

    def update(
        self, detections: List[Detection], frame_index: int, timestamp: float
    ) -> Tuple[List[TrackedProduct], List[TrackedProduct]]:
        """Apply this frame's detections.

        Returns ``(seen, lost)``: tracks seen in this frame, and tracks that
        timed out and were removed.
        """
        seen: List[TrackedProduct] = []
        for det in detections:
            if det.track_id is None:
                self.missing_id_count += 1  # tracker had no ID yet; skip safely
                continue
            track = self._tracks.get(det.track_id)
            if track is None:
                track = self._create(det, frame_index, timestamp)
            else:
                self._refresh(track, det, frame_index, timestamp)
            track.history.append((det.class_name, det.confidence))
            seen.append(track)
        return seen, self._prune(frame_index)

    def get(self, track_id: int) -> Optional[TrackedProduct]:
        return self._tracks.get(track_id)

    def reset(self) -> None:
        self._tracks.clear()
        self.missing_id_count = 0

    def _create(self, det: Detection, frame_index: int, timestamp: float) -> TrackedProduct:
        track = TrackedProduct(
            track_id=det.track_id,  # type: ignore[arg-type]
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=det.bbox,
            centroid=det.centroid,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            last_seen_timestamp=timestamp,
            avg_confidence=det.confidence,
            history=deque(maxlen=self._history_size),
        )
        self._tracks[track.track_id] = track
        logger.info("Track %d detected as %s", track.track_id, track.class_name)
        return track

    @staticmethod
    def _refresh(track: TrackedProduct, det: Detection, frame_index: int, timestamp: float) -> None:
        track.previous_centroid = track.centroid
        track.centroid = det.centroid
        track.bbox = det.bbox
        track.confidence = det.confidence
        track.last_seen_frame = frame_index
        track.last_seen_timestamp = timestamp

    def _prune(self, frame_index: int) -> List[TrackedProduct]:
        lost = [
            t for t in self._tracks.values()
            if frame_index - t.last_seen_frame > self._timeout_frames
        ]
        for track in lost:
            del self._tracks[track.track_id]
        return lost
