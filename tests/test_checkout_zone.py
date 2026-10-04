import unittest

from checkout.checkout_zone import CheckoutZone
from detector.product_detector import Detection
from tracker.object_tracker import ObjectTracker


class CheckoutZoneTests(unittest.TestCase):
    def setUp(self):
        self.zone = CheckoutZone(100, 100, 300, 300)

    def test_contains(self):
        self.assertTrue(self.zone.contains((200, 200)))
        self.assertTrue(self.zone.contains((100, 100)))   # edge counts as inside
        self.assertFalse(self.zone.contains((50, 200)))

    def test_invalid_zone_raises(self):
        with self.assertRaises(ValueError):
            CheckoutZone(300, 100, 100, 300)

    def test_from_config_normalised_and_pixels(self):
        z = CheckoutZone.from_config({"x1": .25, "y1": .5, "x2": .75, "y2": .9}, True, 1000, 500)
        self.assertEqual(z.as_tuple, (250, 250, 750, 450))
        z = CheckoutZone.from_config({"x1": 10, "y1": 10, "x2": 5000, "y2": 400}, False, 640, 480)
        self.assertEqual(z.as_tuple, (10, 10, 639, 400))

    def test_update_track_latches_entry(self):
        tracker = ObjectTracker(15, 30)
        outside = Detection(0, "oreo", .9, 0, 0, 40, 40, track_id=1)
        inside = Detection(0, "oreo", .9, 180, 180, 220, 220, track_id=1)
        track = tracker.update([outside], 0, 0.0)[0][0]
        self.assertFalse(self.zone.update_track(track))
        track = tracker.update([inside], 1, 0.03)[0][0]
        self.assertTrue(self.zone.update_track(track))      # first entry
        track = tracker.update([outside], 2, 0.06)[0][0]
        self.assertFalse(self.zone.update_track(track))
        self.assertFalse(track.in_zone)
        self.assertTrue(track.entered_checkout_zone)         # latched


if __name__ == "__main__":
    unittest.main()
