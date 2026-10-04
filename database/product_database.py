"""Product metadata lookup (name, category, price). No detection logic here."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProductDatabaseError(RuntimeError):
    """Raised when products.json is missing or unreadable."""


@dataclass(frozen=True)
class ProductInfo:
    """Metadata for one sellable product."""

    key: str
    name: str
    category: str
    price: float
    currency: str = "USD"


class ProductDatabase:
    """Loads products.json and answers lookups by class name."""

    def __init__(self, path: Path) -> None:
        self._products: Dict[str, ProductInfo] = {}
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        if not path.is_file():
            raise ProductDatabaseError(f"Product database not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductDatabaseError(f"Cannot read product database '{path}': {exc}") from exc
        if not isinstance(raw, dict):
            raise ProductDatabaseError("products.json must contain a JSON object")

        for key, entry in raw.items():
            try:
                self._products[key] = ProductInfo(
                    key=key,
                    name=str(entry["name"]),
                    category=str(entry.get("category", "Uncategorised")),
                    price=round(float(entry["price"]), 2),
                    currency=str(entry.get("currency", "USD")),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping invalid product entry '%s' in %s", key, path.name)
        logger.info("Product database loaded: %d products", len(self._products))

    def get(self, class_name: str) -> Optional[ProductInfo]:
        """Return the product, or None if the class is not in the database."""
        return self._products.get(class_name)

    def __contains__(self, class_name: str) -> bool:
        return class_name in self._products

    def keys(self) -> List[str]:
        return list(self._products)

    def get_name(self, class_name: str) -> str:
        info = self.get(class_name)
        return info.name if info else class_name.replace("_", " ").title()

    def get_category(self, class_name: str) -> str:
        info = self.get(class_name)
        return info.category if info else "Unknown"

    def get_price(self, class_name: str) -> Optional[float]:
        info = self.get(class_name)
        return info.price if info else None
