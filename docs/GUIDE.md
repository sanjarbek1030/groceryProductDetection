# Guide (in learning order)

## 1. Project concept
A camera watches a checkout area. Products are recognised, followed over time, and added to a cart once the system is *sure*. The difficulty is not drawing boxes; it is deciding when a noisy stream of per-frame guesses means "exactly one more Coca-Cola".

## 2. System architecture
See the module table in the README. Data flows one way: `Detection` (detector) -> `TrackedProduct` (tracker) -> status (recognizer) -> `CartItem` (cart) -> pixels (overlay). `pipeline.py` orchestrates; `main.py` does I/O. Modules communicate through small dataclasses, so each can be tested alone.

## 3. Why tracking is necessary
A detector has no memory. A bottle visible for 200 frames produces 200 detections. Counting detections gives 200 bottles. A tracker assigns an ID and keeps it as the bottle moves, so "ID 15" = one object. Counting *IDs* gives 1. Limitation: IDs are not permanent identities; an object that disappears and returns may get a new ID (needs re-identification).

## 4. Why temporal verification is necessary
Single frames lie: glare, blur, a hand, a half-hidden label. We keep the last `HISTORY_SIZE` predictions of each track and take a confidence-weighted vote. A track becomes `STABLE` only when: >= `MIN_STABLE_FRAMES` observations exist, >= `MIN_CLASS_AGREEMENT` of them agree, mean confidence >= `CONFIRMATION_CONFIDENCE`, and the winner has stayed the winner for `MIN_STABLE_FRAMES` frames. Example: `coca_cola, coca_cola, pepsi, coca_cola, coca_cola` -> coca_cola wins 4/5; the stray `pepsi` costs one stability point but does not flip the class. Two-frame detections never reach the threshold.

Statuses: `VERIFYING` (collecting evidence) -> `STABLE` -> (in zone) `CONFIRMED`. Side branches: `UNCERTAIN` (consistent but low confidence, e.g. Pepsi at 54%) and `UNKNOWN PRODUCT` (frames disagree, or class missing from the database). Neither is ever added.

## 5. Dataset strategy
Custom products need a custom dataset and model: COCO classes like "bottle" cannot tell Coca-Cola from Pepsi. Steps: collect varied images -> annotate (Roboflow/CVAT) -> split by scene into train/val/test -> `data.yaml` -> train -> validate -> copy `best.pt` to `models/`. Include hard cases: hands holding items, partial occlusion, multiple items, glare, rotation, and look-alike pairs. Consider an extra "unknown object" class or negative images.

## 6-8. Structure, installation, configuration
See README. All tunables live in `config.py`. Two confidence values exist on purpose: `CONFIDENCE_THRESHOLD` (0.40) removes noise so tracking has data; `CONFIRMATION_CONFIDENCE` (0.75) decides whether the cart may trust the product.

## 9. Product database
`ProductDatabase` loads `products.json` into `ProductInfo` dataclasses. Missing file -> clear error. Missing product -> `None` (never a crash), and the recognizer marks the track `UNKNOWN PRODUCT`. It knows nothing about detection.

## 10. Detector
`ProductDetector` wraps Ultralytics. `detect()` = plain inference, `track()` = inference + ByteTrack with `persist=True` (keeps the tracker's memory across calls). It filters by confidence and returns `Detection` objects. `track_id` can be `None` while the tracker has not yet assigned an ID; such detections are skipped safely.

## 11. Tracker
`ObjectTracker` creates a `TrackedProduct` for each new ID, updates centroid/previous centroid/last-seen, appends each frame's `(class, confidence)` to a bounded history, and deletes tracks unseen for `TRACK_TIMEOUT_FRAMES`. It does not decide anything about products.

## 12. Checkout-zone logic
Centroid inside the rectangle = in zone. First entry sets `entered_checkout_zone` (a latch) and logs once. Confirmation requires this latch, so items merely passing through the background are not billed. Zone coordinates are normalised by default so they work at any resolution.

## 13. Cart manager
Keeps `CartItem` lines plus a `track_id -> product_key` map. `add_product` returns `False` for a known track, so a product seen for 500 frames is added once, while IDs 15/19/27 give x3. It also supports `remove_track` (the hook for future removal), `increment`, `decrement`, `clear`, subtotal, total.

## 14. Main application
`main.py` opens the source, validates the first frame size, builds the zone/pipeline/writer, then loops: read -> `detector.track` -> `pipeline.process` -> `overlay.render` -> write/show -> keys. It measures FPS from the loop time, handles empty frames, early-ended (corrupted) videos, bad cameras and unwritable output folders.

## 15. Testing
`python -m unittest discover -s tests -t .` covers: cart add/duplicate/quantity/remove/clear, zone containment and entry latch, low confidence, flicker, disagreement, missing metadata, missing track ID, reset, re-entry limitation, optional removal, missing database file. Manual tests: missing model (clear error), missing video, wrong camera index, then real-world scenes (lighting, occlusion, clutter, look-alikes). Use `D` debug overlay.

## 16. README
See `README.md`.

## 17. Known limitations
- **Case 1** (500 frames): quantity = 1. **Case 2** (3 products): 3. **Case 3** (2 frames): not confirmed. **Case 4** (fluctuating confidence): voting + averaging stabilise it. **Case 5** (look-alikes): decided by votes and confidence; persistent disagreement -> UNKNOWN. **Case 6** (hidden): low confidence -> UNCERTAIN, not added.
- **Case 7** (leave and re-enter): the tracker may issue a new ID, producing a double count. Mitigations (not implemented): longer timeouts, BoT-SORT with re-ID, appearance embeddings matched against recently counted items, a "cart session" concept, or a physical zone where products stay until paid.
- Removal is addition-only by default; the cart, pipeline and config already contain the hooks.
- Not safe as an unsupervised commercial checkout.

## 18. Advanced improvements
1-2 Instance segmentation (tighter shapes, better occlusion handling). OCR as optional second stage for variants (detector says `coca_cola_family`, OCR reads "ZERO"; run only on STABLE crops, not every frame). Embeddings / CLIP-style features for look-alikes, unknown products and re-identification. Barcode detection + scanning as ground truth. Multiple cameras. SQL/PostgreSQL, Redis, REST API, web dashboard, receipts, Docker. GPU/edge inference. Automatic dataset collection and active learning from UNCERTAIN/UNKNOWN cases. Unknown-product detection via embedding distance.

## 19. Full execution instructions
1. `python -m venv .venv` and activate, `pip install -r requirements.txt`
2. Build the dataset, train, copy `best.pt` to `models/`
3. Place the test video in `input/` (or set `USE_WEBCAM = True`)
4. `python main.py`; use `D` for debug, `R` to reset between runs
5. Check `output/checkout_output.mp4`
