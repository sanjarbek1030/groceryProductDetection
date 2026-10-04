import unittest

from checkout.cart_manager import CartManager


def add(cart, track_id, key="coca_cola", price=1.50):
    return cart.add_product(track_id, key, key.title(), price)


class CartManagerTests(unittest.TestCase):
    def test_add_and_totals(self):
        cart = CartManager()
        add(cart, 1)
        add(cart, 2, "oreo", 2.20)
        self.assertEqual(cart.total_quantity, 2)
        self.assertAlmostEqual(cart.total, 3.70)

    def test_same_track_is_counted_once(self):
        cart = CartManager()
        results = [add(cart, 15) for _ in range(300)]
        self.assertEqual(results.count(True), 1)
        self.assertEqual(cart.quantity_of("coca_cola"), 1)

    def test_three_identical_products_have_three_tracks(self):
        cart = CartManager()
        for track_id in (15, 19, 27):
            add(cart, track_id)
        self.assertEqual(cart.quantity_of("coca_cola"), 3)
        self.assertEqual(cart.contents()["coca_cola"]["subtotal"], 4.50)

    def test_remove_track_and_decrement(self):
        cart = CartManager()
        add(cart, 1)
        add(cart, 2)
        self.assertTrue(cart.remove_track(1))
        self.assertFalse(cart.remove_track(1))
        self.assertEqual(cart.quantity_of("coca_cola"), 1)
        cart.decrement("coca_cola")
        self.assertEqual(cart.items, [])

    def test_increment_unknown_key_is_safe(self):
        self.assertFalse(CartManager().increment("nope"))

    def test_clear_allows_recounting(self):
        cart = CartManager()
        add(cart, 1)
        cart.clear()
        self.assertEqual(cart.total, 0)
        self.assertTrue(add(cart, 1))


if __name__ == "__main__":
    unittest.main()
