"""Checkout zone: a rectangle and the logic for 'has this product entered it?'."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from tracker.object_tracker import Point, TrackedProduct


@dataclass(frozen=True)
class CheckoutZone:
    """Axis-aligned rectangle in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid checkout zone: {self}")

    @classmethod
    def from_config(
        cls, zone: Dict[str, float], normalized: bool, frame_width: int, frame_height: int
    ) -> "CheckoutZone":
        """Build a zone from config values, clamped to the frame."""
        if normalized:
            x1, x2 = zone["x1"] * frame_width, zone["x2"] * frame_width
            y1, y2 = zone["y1"] * frame_height, zone["y2"] * frame_height
        else:
            x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
        return cls(
            x1=max(0, int(x1)), y1=max(0, int(y1)),
            x2=min(frame_width - 1, int(x2)), y2=min(frame_height - 1, int(y2)),
        )

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def contains(self, point: Point) -> bool:
        """True if the point is inside (or on the edge of) the zone."""
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def update_track(self, track: TrackedProduct) -> bool:
        """Update zone flags on a track. Returns True the moment it first enters.

        Zone-crossing logic: the CENTROID decides. ``in_zone`` is the current
        state; ``entered_checkout_zone`` latches True forever after the first
        entry, so a product that is briefly nudged out is not forgotten.
        """
        track.in_zone = self.contains(track.centroid)
        if track.in_zone and not track.entered_checkout_zone:
            track.entered_checkout_zone = True
            return True
        return False
