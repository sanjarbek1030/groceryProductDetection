"""CheckoutPipeline: the decision logic, free of any camera or drawing code.

detections -> tracks -> zone -> temporal verification -> confirmation -> cart
Keeping this separate from main.py means it can be unit-tested without a
camera, a window or a YOLO model.
"""
from __future__ import annotations

import logging
from typing import List

from checkout.cart_manager import CartManager
from checkout.checkout_zone import CheckoutZone
from database.product_database import ProductDatabase
from detector.product_detector import Detection
from recognition.product_recognizer import ProductRecognizer
from tracker.object_tracker import ObjectTracker, ProductStatus, TrackedProduct

logger = logging.getLogger(__name__)


class CheckoutPipeline:
    """Connects tracker, zone, recognizer, database and cart."""

    def __init__(
        self,
        database: ProductDatabase,
        zone: CheckoutZone,
        tracker: ObjectTracker,
        recognizer: ProductRecognizer,
        cart: CartManager,
        remove_item_when_track_lost: bool = False,
    ) -> None:
        self._db = database
        self._zone = zone
        self._tracker = tracker
        self._recognizer = recognizer
        self._cart = cart
        self._remove_when_lost = remove_item_when_track_lost

    @property
    def zone(self) -> CheckoutZone:
        return self._zone

    @property
    def cart(self) -> CartManager:
        return self._cart

    def process(self, detections: List[Detection], frame_index: int, timestamp: float) -> List[TrackedProduct]:
        """Handle one frame. Returns the tracks seen in this frame."""
        seen, lost = self._tracker.update(detections, frame_index, timestamp)
        for track in seen:
            self._process_track(track)
        for track in lost:
            self._handle_lost_track(track)
        return seen

    def reset(self) -> None:
        """Forget all tracks and clear the cart."""
        self._tracker.reset()
        self._cart.clear()
        logger.info("Pipeline reset: tracks and cart cleared")

    def _process_track(self, track: TrackedProduct) -> None:
        if self._zone.update_track(track):
            logger.info("Track %d entered checkout zone", track.track_id)
        self._recognizer.evaluate(track)
        if self._recognizer.ready_to_confirm(track):
            self._confirm_and_count(track)

    def _confirm_and_count(self, track: TrackedProduct) -> None:
        track.confirmed = True
        track.status = ProductStatus.CONFIRMED
        logger.info("Track %d confirmed as %s", track.track_id, track.class_name)

        info = self._db.get(track.class_name)
        if info is None:  # should not happen (recognizer checks) but never crash
            logger.warning("No product metadata for '%s'; not added", track.class_name)
            return
        if self._cart.add_product(track.track_id, info.key, info.name, info.price, info.currency):
            track.counted = True
            logger.info("Added %s to cart", info.name)
            logger.info("%s quantity = %d", info.name, self._cart.quantity_of(info.key))

    def _handle_lost_track(self, track: TrackedProduct) -> None:
        logger.info("Track %d (%s) left the scene", track.track_id, track.class_name)
        if track.counted and self._remove_when_lost:
            if self._cart.remove_track(track.track_id):
                logger.info("Removed %s from cart (track %d lost)", track.class_name, track.track_id)
