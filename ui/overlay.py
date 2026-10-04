"""All OpenCV drawing: boxes, checkout zone, centroids, and the cart panel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from checkout.cart_manager import CartManager
from checkout.checkout_zone import CheckoutZone
from database.product_database import ProductDatabase
from tracker.object_tracker import ProductStatus, TrackedProduct

FONT = cv2.FONT_HERSHEY_SIMPLEX
Color = Tuple[int, int, int]  # BGR

COLOR_ZONE: Color = (255, 160, 0)
COLOR_PANEL_BG: Color = (32, 32, 32)
COLOR_TEXT: Color = (235, 235, 235)
COLOR_DIM: Color = (150, 150, 150)
COLOR_ACCENT: Color = (0, 215, 255)
STATUS_COLORS = {
    ProductStatus.VERIFYING: (0, 215, 255),
    ProductStatus.STABLE: (160, 230, 0),
    ProductStatus.CONFIRMED: (0, 200, 0),
    ProductStatus.UNCERTAIN: (0, 140, 255),
    ProductStatus.UNKNOWN: (255, 0, 255),
}


@dataclass
class FrameStats:
    """Numbers shown in the side panel."""

    fps: float = 0.0
    detected: int = 0
    active_tracks: int = 0
    paused: bool = False


def format_money(amount: float, currency: str) -> str:
    """OpenCV's fonts only draw ASCII, so non-USD uses the currency code."""
    return f"${amount:.2f}" if currency == "USD" else f"{amount:.2f} {currency}"


class OverlayRenderer:
    """Draws everything on top of a frame and appends the cart panel."""

    def __init__(
        self,
        database: ProductDatabase,
        panel_width: int,
        display_confidence: bool = True,
        display_track_id: bool = True,
        display_centroid: bool = True,
        debug_mode: bool = False,
    ) -> None:
        self._db = database
        self._panel_width = panel_width
        self.display_confidence = display_confidence
        self.display_track_id = display_track_id
        self.display_centroid = display_centroid
        self.debug_mode = debug_mode

    def render(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedProduct],
        zone: CheckoutZone,
        cart: CartManager,
        stats: FrameStats,
    ) -> np.ndarray:
        """Return frame + cart panel as one image (the frame is drawn on in place)."""
        scale = max(0.45, frame.shape[0] / 720 * 0.55)
        self._draw_zone(frame, zone, scale)
        for track in tracks:
            self._draw_track(frame, track, scale)
        panel = self._draw_panel(frame.shape[0], cart, stats)
        return np.hstack((frame, panel))

    # ------------------------------------------------------------------ frame
    def _draw_zone(self, frame: np.ndarray, zone: CheckoutZone, scale: float) -> None:
        x1, y1, x2, y2 = zone.as_tuple
        roi = frame[y1:y2, x1:x2]
        tint = np.full_like(roi, COLOR_ZONE)
        cv2.addWeighted(roi, 0.88, tint, 0.12, 0, roi)  # light translucent fill
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ZONE, 2)
        self._text_block(frame, ["CHECKOUT ZONE"], x1, y2, COLOR_ZONE, scale, above=True)

    def _draw_track(self, frame: np.ndarray, track: TrackedProduct, scale: float) -> None:
        color = STATUS_COLORS[track.status]
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if self.display_centroid:
            center = (int(track.centroid[0]), int(track.centroid[1]))
            cv2.circle(frame, center, 5, color, -1 if track.in_zone else 2)

        name = "Unknown" if track.status == ProductStatus.UNKNOWN else self._db.get_name(track.class_name)
        details = []
        if self.display_track_id:
            details.append(f"ID: {track.track_id}")
        if self.display_confidence:
            details.append(f"Conf: {round(track.avg_confidence * 100)}%")
        lines = [name]
        if details:
            lines.append("  ".join(details))
        lines.append(self._status_text(track))
        self._text_block(frame, lines, x1, y1, color, scale, above=True)

        if self.debug_mode:
            self._text_block(frame, self._debug_lines(track), x1, y2, (60, 60, 60), scale * 0.9,
                             above=False, text_color=COLOR_TEXT)

    @staticmethod
    def _status_text(track: TrackedProduct) -> str:
        if track.status == ProductStatus.CONFIRMED:
            return "CONFIRMED - IN CART" if track.counted else "CONFIRMED"
        if track.status == ProductStatus.STABLE and not track.entered_checkout_zone:
            return "STABLE - MOVE TO ZONE"
        if track.status == ProductStatus.UNCERTAIN:
            return "UNCERTAIN (LOW CONFIDENCE)"
        if track.status == ProductStatus.UNKNOWN:
            return "UNKNOWN PRODUCT - RE-PRESENT"
        return track.status.value

    @staticmethod
    def _debug_lines(track: TrackedProduct) -> List[str]:
        cx, cy = track.centroid
        return [
            f"id={track.track_id} cls={track.class_name} conf={track.confidence:.2f}",
            f"centroid=({cx:.0f},{cy:.0f}) stable={track.stable_frame_count}",
            f"zone={'IN' if track.in_zone else 'OUT'} entered={track.entered_checkout_zone}",
            f"confirmed={track.confirmed} counted={track.counted}",
        ]

    @staticmethod
    def _text_block(
        frame: np.ndarray, lines: List[str], anchor_x: int, anchor_y: int, bg: Color,
        scale: float, above: bool, text_color: Color = (0, 0, 0),
    ) -> None:
        """Draw lines of text on a filled rectangle next to an anchor point."""
        height, width = frame.shape[:2]
        sizes = [cv2.getTextSize(t, FONT, scale, 1)[0] for t in lines]
        line_h = max(h for _, h in sizes) + 6
        block_w = max(w for w, _ in sizes) + 8
        block_h = line_h * len(lines) + 4
        x = int(min(max(anchor_x, 0), max(0, width - block_w)))
        y_top = anchor_y - block_h if above else anchor_y
        if y_top < 0:
            y_top = anchor_y if above else 0
        y_top = int(min(y_top, max(0, height - block_h)))
        cv2.rectangle(frame, (x, y_top), (x + block_w, y_top + block_h), bg, -1)
        for i, text in enumerate(lines):
            y_text = y_top + 2 + (i + 1) * line_h - 6
            cv2.putText(frame, text, (x + 4, y_text), FONT, scale, text_color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ panel
    def _draw_panel(self, height: int, cart: CartManager, stats: FrameStats) -> np.ndarray:
        panel = np.full((height, self._panel_width, 3), COLOR_PANEL_BG, dtype=np.uint8)
        w = self._panel_width
        pad = 16
        y = 38

        def put(text: str, x: int, y_pos: int, scale: float = 0.55,
                color: Color = COLOR_TEXT, thick: int = 1) -> None:
            cv2.putText(panel, text, (x, y_pos), FONT, scale, color, thick, cv2.LINE_AA)

        def put_right(text: str, y_pos: int, scale: float = 0.55, color: Color = COLOR_TEXT) -> None:
            tw = cv2.getTextSize(text, FONT, scale, 1)[0][0]
            put(text, w - pad - tw, y_pos, scale, color)

        put("AI SMART CHECKOUT", pad, y, 0.8, COLOR_ACCENT, 2)
        y += 14
        cv2.line(panel, (pad, y), (w - pad, y), COLOR_DIM, 1)
        y += 30
        put("SHOPPING CART", pad, y, 0.6, COLOR_TEXT, 2)
        y += 12

        max_rows = max(1, (height - y - 230) // 28)
        items = cart.items
        if not items:
            y += 28
            put("(empty)", pad, y, 0.55, COLOR_DIM)
        for item in items[:max_rows]:
            y += 28
            put(item.name[:17], pad, y)
            put(f"x{item.quantity}", w - 150, y)
            put_right(format_money(item.subtotal, item.currency), y)
        if len(items) > max_rows:
            y += 28
            put(f"... +{len(items) - max_rows} more", pad, y, 0.5, COLOR_DIM)

        y += 18
        cv2.line(panel, (pad, y), (w - pad, y), COLOR_DIM, 1)
        y += 30
        put("TOTAL", pad, y, 0.7, COLOR_ACCENT, 2)
        put_right(format_money(cart.total, cart.currency), y, 0.7, COLOR_ACCENT)

        y = height - 190
        cv2.line(panel, (pad, y), (w - pad, y), COLOR_DIM, 1)
        for label in (
            f"FPS: {stats.fps:.1f}",
            f"Detected products: {stats.detected}",
            f"Confirmed products: {cart.total_quantity}",
            f"Active tracks: {stats.active_tracks}",
        ):
            y += 28
            put(label, pad, y)
        y += 30
        put("Q quit  C clear  R reset", pad, y, 0.45, COLOR_DIM)
        y += 20
        put("P pause  S shot  D debug", pad, y, 0.45, COLOR_DIM)
        if stats.paused:
            put("PAUSED", w - 110, 38 + 36, 0.7, (0, 0, 255), 2)
        return panel
