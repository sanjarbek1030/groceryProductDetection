# Product Recognition Checkout

A computer-vision portfolio project: place retail products in front of a camera and an automatic shopping cart is built, with prices and a total.

> **Disclaimer:** This is a computer-vision demonstration, **not** a payment system. No payment processing is implemented. Vision alone is not sufficient for commercial checkout without extensive validation and additional safeguards (see *Known Limitations*).

## Overview
Instead of running YOLO on each frame and counting detections, the system follows each physical object over time:

```
CAMERA -> FRAME -> DETECTION -> TRACKING -> TRACK STATE -> TEMPORAL VERIFICATION
 -> CHECKOUT ZONE -> CONFIRMATION -> DUPLICATE PREVENTION -> QUANTITY -> DATABASE -> CART -> TOTAL -> VISUALIZATION
```

## Features
- Webcam or video-file input; annotated output video for files
- Custom-trained YOLO detection + ByteTrack/BoT-SORT tracking (persistent IDs)
- Per-track voting over recent frames (temporal verification)
- Checkout zone using the box centroid
- One track ID is counted at most once (duplicate prevention); identical products are counted by distinct IDs
- Confidence handling: `VERIFYING`, `STABLE`, `CONFIRMED`, `UNCERTAIN`, `UNKNOWN PRODUCT`
- JSON product database, cart with subtotals and total
- OpenCV overlay: boxes, IDs, confidence, zone, cart panel, FPS, debug mode
- Structured logging, graceful error messages, unit tests

## Architecture
| Module | Responsibility |
|---|---|
| `config.py` | All settings + validation |
| `detector/` | Only code that touches Ultralytics; returns `Detection` objects |
| `tracker/` | `TrackedProduct` state per track ID, pruning of lost tracks |
| `recognition/` | Weighted voting, stability, status decision |
| `checkout/checkout_zone.py` | Zone rectangle + entry logic |
| `checkout/cart_manager.py` | Cart, quantities, duplicate prevention |
| `database/` | `products.json` + `ProductDatabase` |
| `pipeline.py` | Connects all of the above (no camera/GUI code, testable) |
| `ui/overlay.py` | All drawing |
| `main.py` | Capture loop, keys, video writer, error handling |

## Technologies
Python 3.9+, OpenCV, Ultralytics YOLO, NumPy, standard library (`logging`, `dataclasses`, `json`, `pathlib`, `unittest`).

## Project Structure
```
ProductRecognitionCheckout/
├── main.py  config.py  pipeline.py  requirements.txt  README.md
├── detector/      product_detector.py
├── tracker/       object_tracker.py
├── recognition/   product_recognizer.py
├── checkout/      checkout_zone.py  cart_manager.py
├── database/      products.json  product_database.py
├── ui/            overlay.py
├── tests/         test_cart_manager.py  test_checkout_zone.py  test_pipeline.py
├── models/        best.pt            (you provide)
├── input/         checkout_input.mp4 (you provide)
├── output/        checkout_output.mp4 (generated)
├── dataset/       images/ labels/ data.yaml
└── docs/GUIDE.md  (long-form explanations)
```

## Installation
```powershell
cd ProductRecognitionCheckout
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
**Environment setup (PyCharm):** open the folder, *Settings -> Project -> Python Interpreter -> Add -> Existing -> `.venv\Scripts\python.exe`*, then create a Run Configuration for `main.py` with the project folder as working directory.
**GPU (optional):** install the CUDA build of PyTorch first using the selector on pytorch.org, then `pip install ultralytics`. Speed depends on model size, GPU/CPU, input resolution, number of detections, tracker and hardware acceleration - measure it with the on-screen FPS.

## Dataset Preparation
COCO-pretrained YOLO does not know *your* products, so a custom dataset is required.
1. **Collect** 100-300+ images per class: varied lighting, angles, backgrounds, hands, partial occlusion, multiple items per image.
2. **Annotate** bounding boxes with Roboflow or CVAT (optional tools) or LabelImg; export in *YOLO format*.
3. **Split** into `train` / `val` / `test` (e.g. 70/20/10). Split by *scene/session*, not randomly per image, otherwise near-duplicate frames leak between sets and inflate scores.
4. **Describe** with `dataset/data.yaml` (sample included).

```
dataset/images/{train,val,test}/*.jpg
dataset/labels/{train,val,test}/*.txt   # one line per box: class cx cy w h (normalised)
```

## Model Training
```powershell
yolo detect train data=dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640
```
Model choice (`n/s/m/...`, or a newer family) depends on your installed Ultralytics version and hardware; no model is universally best. Results land in `runs/detect/train*/weights/best.pt` - copy it to `models/best.pt`. Validate with `yolo detect val model=models/best.pt data=dataset/data.yaml`.

**Metrics:** *precision* = of predicted boxes, how many were right; *recall* = of real products, how many were found; *mAP50* = average precision at IoU 0.5; *mAP50-95* = averaged over stricter IoUs (rewards tight boxes); the *confusion matrix* shows which products get mixed up (e.g. Coca-Cola vs Pepsi). High validation scores do **not** guarantee reliable checkout - test occlusion, reflections, poor light, motion blur, multiple items, look-alike products, hidden labels, rotated products and crowded scenes.

## Configuration
Everything is in `config.py`: `MODEL_PATH`, `INPUT_VIDEO`, `OUTPUT_VIDEO`, `USE_WEBCAM`, `CAMERA_INDEX`, `CONFIDENCE_THRESHOLD` (noise filter), `CONFIRMATION_CONFIDENCE` (trust level, 0.75), `IOU_THRESHOLD`, `MIN_STABLE_FRAMES`, `CHECKOUT_ZONE`, `TRACKER_CONFIG`, `DISPLAY_*`, `SAVE_OUTPUT`, `VIDEO_WIDTH/HEIGHT`, `DEBUG_MODE`. Invalid values stop the app with a readable message.

## Running the Application
1. Put `best.pt` in `models/`.
2. Video mode: put a video at `input/checkout_input.mp4` (default). Webcam mode: set `USE_WEBCAM = True`.
3. `python main.py`
4. Tests (no model needed): `python -m unittest discover -s tests -t .`

## Keyboard Controls
| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `C` | Clear cart (items already counted are *not* re-added while their tracks live) |
| `R` | Full reset: tracks, YOLO tracker memory, cart, counters |
| `P` | Pause / resume |
| `S` | Save screenshot to `output/screenshots/` |
| `D` | Toggle debug overlay |

## Example Output
```
AI SMART CHECKOUT
SHOPPING CART
Coca-Cola      x2    $3.00
Oreo Original  x1    $2.20
TOTAL                $5.20
FPS: 29.4   Detected products: 4   Confirmed products: 3   Active tracks: 4
```
Log excerpt:
```
[INFO] Track 15 detected as coca_cola
[INFO] Track 15 entered checkout zone
[INFO] Track 15 confirmed as coca_cola
[INFO] Added Coca-Cola to cart
[INFO] Coca-Cola quantity = 1
```

## How Product Tracking Works
Detection answers "what is in *this* frame?". Tracking links detections across frames and gives each physical object a `track_id` (ByteTrack associates boxes by overlap and motion). ID 7 in frames 1, 2 and 3 is **one** object, not three. `ObjectTracker` stores one `TrackedProduct` per ID.

## How Checkout Zone Works
The centroid `cx=(x1+x2)/2, cy=(y1+y2)/2` is tested against the zone rectangle. `entered_checkout_zone` latches `True` on first entry. A product is only eligible for confirmation after that.

## How Duplicate Counting Is Prevented
Two layers: (1) a `TrackedProduct` is `confirmed` once and `counted` once; (2) `CartManager` keeps `track_id -> product` and refuses to add a track ID twice. Uniqueness is the **track ID**, never the product name, so IDs 15, 19 and 27 of Coca-Cola give quantity 3.

## Shopping Cart Logic
Confirmed track -> database lookup -> `CartManager.add_product(track_id, ...)` -> quantity+1, subtotal and total recalculated. Unknown class names are never added.

## Known Limitations
- **Track IDs are not permanent identities.** A product that leaves and returns can come back as a new ID (e.g. 15 -> 42) and be counted twice. A normal tracker does not solve re-identification; it needs appearance embeddings or similar (a test documents this behaviour).
- Removal is not enabled by default (`REMOVE_ITEM_WHEN_TRACK_LOST=False`) because a lost track can just mean occlusion.
- Look-alike products, hidden labels, reflections and crowding reduce accuracy.
- One currency per cart; OpenCV text is ASCII only.
- Not a payment system; real retail needs extensive validation, weight/RFID/barcode cross-checks, fraud handling and audits.

## Future Improvements
Instance segmentation, OCR as a *second-stage* check for look-alike variants, barcode detection/scanning, embeddings/CLIP and re-identification, multi-camera, SQL database (PostgreSQL), Redis, REST API, web dashboard, receipts, Docker, GPU/edge deployment, automatic dataset collection, active learning, explicit unknown-product detection. See `docs/GUIDE.md`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Model file not found` | Train a model and copy it to `models/best.pt` |
| `Input video not found` | Add `input/checkout_input.mp4` or use the webcam |
| `Cannot open camera` | Close other apps using it; try `CAMERA_INDEX = 1` |
| Window does not appear | Do not install `opencv-python-headless` next to `opencv-python` |
| Nothing gets added | Press `D`; check `zone=IN`, `stable>=MIN_STABLE_FRAMES`, avg conf >= `CONFIRMATION_CONFIDENCE` |
| Product counted twice | Track ID changed (occlusion / re-entry); raise `TRACK_TIMEOUT_FRAMES`, try `botsort.yaml` |
| Model classes missing from products.json | Add them to `database/products.json` |
| Slow | Smaller model, smaller `IMAGE_SIZE`, GPU |

## License
MIT - see `LICENSE`.
