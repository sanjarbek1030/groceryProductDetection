"""End-to-end logic tests using scripted detections (no camera, no YOLO)."""
import json
import tempfile
import unittest
from pathlib import Path

from checkout.cart_manager import CartManager
from checkout.checkout_zone import CheckoutZone
from database.product_database import ProductDatabase, ProductDatabaseError
from detector.product_detector import Detection
from pipeline import CheckoutPipeline
from recognition.product_recognizer import ProductRecognizer
from tracker.object_tracker import ObjectTracker, ProductStatus

PRODUCTS = {
    "coca_cola": {"name": "Coca-Cola", "category": "Soft Drink", "price": 1.50},
    "pepsi": {"name": "Pepsi", "category": "Soft Drink", "price": 1.45},
    "oreo": {"name": "Oreo", "category": "Snack", "price": 2.20},
}
INSIDE = (200, 200, 260, 260)   # centroid inside zone
OUTSIDE = (10, 10, 50, 50)      # centroid outside zone


def det(track_id, cls="coca_cola", conf=0.9, box=INSIDE):
    return Detection(0, cls, conf, *box, track_id=track_id)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "products.json"
        path.write_text(json.dumps(PRODUCTS), encoding="utf-8")
        self.db = ProductDatabase(path)
        self.cart = CartManager()
        self.pipe = CheckoutPipeline(
            self.db, CheckoutZone(100, 100, 400, 400), ObjectTracker(15, 30),
            ProductRecognizer(self.db, 8, 0.75, 0.70), self.cart)
        self.frame = 0

    def tearDown(self):
        self._tmp.cleanup()

    def run_frames(self, frames):
        """frames: list of detection lists."""
        last = []
        for dets in frames:
            last = self.pipe.process(dets, self.frame, self.frame / 30)
            self.frame += 1
        return last

    def test_case1_500_frames_counts_once(self):
        self.run_frames([[det(15)] for _ in range(500)])
        self.assertEqual(self.cart.quantity_of("coca_cola"), 1)

    def test_case2_three_identical_products(self):
        self.run_frames([[det(15), det(19), det(27)] for _ in range(20)])
        self.assertEqual(self.cart.quantity_of("coca_cola"), 3)

    def test_case3_two_frames_not_confirmed(self):
        self.run_frames([[det(7)], [det(7)]])
        self.assertEqual(self.cart.total_quantity, 0)

    def test_case4_flickering_class_still_resolves(self):
        seq = ["coca_cola"] * 6 + ["pepsi"] + ["coca_cola"] * 8
        self.run_frames([[det(5, c)] for c in seq])
        self.assertEqual(self.cart.quantity_of("coca_cola"), 1)
        self.assertEqual(self.cart.quantity_of("pepsi"), 0)

    def test_case5_constant_disagreement_is_unknown(self):
        seq = ["coca_cola", "pepsi"] * 8
        tracks = self.run_frames([[det(5, c)] for c in seq])
        self.assertEqual(self.cart.total_quantity, 0)
        self.assertEqual(tracks[0].status, ProductStatus.UNKNOWN)

    def test_low_confidence_is_uncertain_and_not_added(self):
        tracks = self.run_frames([[det(3, "pepsi", 0.54)] for _ in range(30)])
        self.assertEqual(self.cart.total_quantity, 0)
        self.assertEqual(tracks[0].status, ProductStatus.UNCERTAIN)

    def test_product_outside_zone_is_never_added(self):
        self.run_frames([[det(1, box=OUTSIDE)] for _ in range(30)])
        self.assertEqual(self.cart.total_quantity, 0)

    def test_enters_zone_later_then_confirms(self):
        self.run_frames([[det(1, box=OUTSIDE)] for _ in range(15)])
        self.assertEqual(self.cart.total_quantity, 0)
        self.run_frames([[det(1)] for _ in range(3)])
        self.assertEqual(self.cart.total_quantity, 1)

    def test_missing_metadata_is_not_added(self):
        tracks = self.run_frames([[det(9, "mystery_item")] for _ in range(30)])
        self.assertEqual(self.cart.total_quantity, 0)
        self.assertEqual(tracks[0].status, ProductStatus.UNKNOWN)

    def test_detection_without_track_id_is_ignored(self):
        self.run_frames([[Detection(0, "oreo", .9, *INSIDE)] for _ in range(30)])
        self.assertEqual(self.cart.total_quantity, 0)

    def test_case7_reentry_with_new_id_is_double_counted(self):
        """Documents the known limitation: a new ID is a new object."""
        self.run_frames([[det(15)] for _ in range(20)])
        self.run_frames([[] for _ in range(40)])             # leaves, track times out
        self.run_frames([[det(42)] for _ in range(20)])      # returns with a new ID
        self.assertEqual(self.cart.quantity_of("coca_cola"), 2)

    def test_removal_when_track_lost_is_optional(self):
        pipe = CheckoutPipeline(
            self.db, CheckoutZone(100, 100, 400, 400), ObjectTracker(15, 30),
            ProductRecognizer(self.db, 8, 0.75, 0.70), self.cart, remove_item_when_track_lost=True)
        for i in range(20):
            pipe.process([det(1)], i, i / 30)
        self.assertEqual(self.cart.total_quantity, 1)
        for i in range(20, 70):
            pipe.process([], i, i / 30)
        self.assertEqual(self.cart.total_quantity, 0)

    def test_reset_clears_cart_and_tracks(self):
        self.run_frames([[det(1)] for _ in range(20)])
        self.pipe.reset()
        self.assertEqual(self.cart.total_quantity, 0)
        self.run_frames([[det(1)] for _ in range(20)])        # same ID counts again
        self.assertEqual(self.cart.total_quantity, 1)


class DatabaseTests(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(ProductDatabaseError):
            ProductDatabase(Path("does_not_exist.json"))

    def test_missing_product_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "p.json"
            p.write_text(json.dumps(PRODUCTS), encoding="utf-8")
            db = ProductDatabase(p)
            self.assertIsNone(db.get("unknown"))
            self.assertEqual(db.get_name("unknown_thing"), "Unknown Thing")
            self.assertEqual(db.get_price("oreo"), 2.2)


if __name__ == "__main__":
    unittest.main()
