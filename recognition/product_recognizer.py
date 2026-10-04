"""Temporal verification: decides what a track IS and whether to trust it.

A single frame can be wrong. We look at the last N frames of a track and let
them vote. A product becomes STABLE only when the vote is consistent, long
enough and confident enough.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from database.product_database import ProductDatabase
from tracker.object_tracker import ProductStatus, TrackedProduct

logger = logging.getLogger(__name__)


class ProductRecognizer:
    """Confidence-weighted voting over each track's recent predictions."""

    def __init__(
        self,
        database: ProductDatabase,
        min_stable_frames: int,
        confirmation_confidence: float,
        min_class_agreement: float,
    ) -> None:
        self._db = database
        self._min_stable_frames = min_stable_frames
        self._confirmation_confidence = confirmation_confidence
        self._min_agreement = min_class_agreement

    def evaluate(self, track: TrackedProduct) -> ProductStatus:
        """Update the track's class, confidence, stability and status.

        Call exactly once per frame for every track seen in that frame.
        """
        if track.confirmed:  # class is locked once confirmed
            track.status = ProductStatus.CONFIRMED
            return track.status
        history = list(track.history)
        if not history:
            return track.status

        dominant, avg_conf, agreement = self._vote(history)
        if dominant != track.class_name:
            track.stable_frame_count = 0  # the answer changed: start over
            track.class_name = dominant
        track.avg_confidence = avg_conf

        if history[-1][0] == dominant:
            track.stable_frame_count += 1
        else:
            track.stable_frame_count = max(0, track.stable_frame_count - 1)

        new_status = self._decide(track, len(history), avg_conf, agreement)
        if new_status != track.status:
            logger.debug("Track %d status %s -> %s", track.track_id,
                         track.status.value, new_status.value)
        track.status = new_status
        return new_status

    def ready_to_confirm(self, track: TrackedProduct) -> bool:
        """True when the track is STABLE and has entered the checkout zone."""
        return (
            track.status == ProductStatus.STABLE
            and track.entered_checkout_zone
            and not track.confirmed
        )

    @staticmethod
    def _vote(history: List[Tuple[str, float]]) -> Tuple[str, float, float]:
        """Return (winning class, its mean confidence, share of frames agreeing)."""
        weight: Dict[str, float] = defaultdict(float)
        count: Dict[str, int] = defaultdict(int)
        for class_name, confidence in history:
            weight[class_name] += confidence
            count[class_name] += 1
        winner = max(weight, key=lambda c: weight[c])
        return winner, weight[winner] / count[winner], count[winner] / len(history)

    def _decide(self, track: TrackedProduct, n_obs: int, avg_conf: float, agreement: float) -> ProductStatus:
        if track.class_name not in self._db:
            return ProductStatus.UNKNOWN          # no metadata -> never invent a product
        if n_obs < self._min_stable_frames:
            return ProductStatus.VERIFYING        # not enough evidence yet
        if agreement < self._min_agreement:
            return ProductStatus.UNKNOWN          # frames disagree too much
        if avg_conf < self._confirmation_confidence:
            return ProductStatus.UNCERTAIN        # consistent but not confident
        if track.stable_frame_count < self._min_stable_frames:
            return ProductStatus.VERIFYING
        return ProductStatus.STABLE
