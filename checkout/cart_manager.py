"""Shopping cart with duplicate prevention based on tracking IDs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


@dataclass
class CartItem:
    """One line of the cart (one product type, any quantity)."""

    product_key: str
    name: str
    unit_price: float
    currency: str = "USD"
    quantity: int = 0
    track_ids: Set[int] = field(default_factory=set)  # physical objects counted

    @property
    def subtotal(self) -> float:
        return round(self.unit_price * self.quantity, 2)


class CartManager:
    """Holds cart lines and guarantees one track ID is counted at most once."""

    def __init__(self) -> None:
        self._items: Dict[str, CartItem] = {}
        self._counted_tracks: Dict[int, str] = {}  # track_id -> product_key

    def add_product(
        self, track_id: int, product_key: str, name: str,
        unit_price: float, currency: str = "USD",
    ) -> bool:
        """Add one physical product. Returns False if this track was already counted."""
        if track_id in self._counted_tracks:
            logger.debug("Track %d already counted; ignoring", track_id)
            return False
        item = self._items.get(product_key)
        if item is None:
            item = CartItem(product_key, name, round(unit_price, 2), currency)
            self._items[product_key] = item
        item.quantity += 1
        item.track_ids.add(track_id)
        self._counted_tracks[track_id] = product_key
        return True

    def remove_track(self, track_id: int) -> bool:
        """Remove the product that was counted for this track (future removal logic)."""
        product_key = self._counted_tracks.pop(track_id, None)
        if product_key is None:
            return False
        item = self._items[product_key]
        item.track_ids.discard(track_id)
        self._decrement_item(item)
        return True

    def increment(self, product_key: str) -> bool:
        """Manually add one more of an existing line."""
        item = self._items.get(product_key)
        if item is None:
            return False
        item.quantity += 1
        return True

    def decrement(self, product_key: str) -> bool:
        """Manually remove one of a line (line disappears at zero)."""
        item = self._items.get(product_key)
        if item is None:
            return False
        self._decrement_item(item)
        return True

    def _decrement_item(self, item: CartItem) -> None:
        item.quantity -= 1
        if item.quantity <= 0:
            del self._items[item.product_key]

    def clear(self) -> None:
        self._items.clear()
        self._counted_tracks.clear()

    def quantity_of(self, product_key: str) -> int:
        item = self._items.get(product_key)
        return item.quantity if item else 0

    @property
    def items(self) -> List[CartItem]:
        return list(self._items.values())

    @property
    def total_quantity(self) -> int:
        return sum(i.quantity for i in self._items.values())

    @property
    def total(self) -> float:
        return round(sum(i.subtotal for i in self._items.values()), 2)

    @property
    def currency(self) -> str:
        """Currency of the cart (assumes one currency per cart)."""
        return next(iter(self._items.values())).currency if self._items else "USD"

    def contents(self) -> Dict[str, Dict[str, float]]:
        """Plain-dict view of the cart, e.g. for logging or JSON."""
        return {
            key: {"quantity": i.quantity, "unit_price": i.unit_price, "subtotal": i.subtotal}
            for key, i in self._items.items()
        }
