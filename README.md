# SCSG Project
- 140810230008 - Robby Azwan Saputra
- 140810230008 - Achmad Dzaki Azhari
- 140810230008 - Siti Nailah Eko
- 140810230008 - Yazid Dahren Fauzan
- 140810230008 - Athallah Azhar Aulia Hadi


## Social Distancing Detector (YOLO + Bird's-Eye View)

Real-time social distancing monitoring from a video file, stream, or webcam.
People are detected with YOLOv8, their **feet positions** (bottom-center of the box)
are projected onto a top-down plane via homography, and pairwise Euclidean distances
are computed there to remove perspective distortion.

## Project structure

```
social_distancing_detector/
├── config.py            # constants + Settings dataclass (JSON-overridable)
├── src/
│   ├── detector.py      # PersonDetector (YOLO) + Detections container
│   ├── perspective.py   # PerspectiveTransformer (homography / BEV)
│   ├── distance.py      # DistanceAnalyzer (cdist violation logic)
│   └── visualizer.py    # boxes, lines, HUD, bird's-eye inset
├── utils/helper.py      # FPSCounter, mouse ROI selection
├── main.py
└── requirements.txt
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
`yolov8n.pt` is downloaded automatically on first run.

## Run

```bash
# Video file, pick the ROI interactively
python main.py --source videos/street.mp4 --select-points

# Webcam
python main.py --source 0

# Reuse a saved config, only analyse people inside the ROI
python main.py --source videos/street.mp4 --config my_config.json --roi-only
```

### ROI selection
Click 4 points on the ground plane in this order: **top-left, top-right, bottom-right, bottom-left**.
Right-click = undo, `R` = reset, `Enter`/`Space` = confirm, `Q`/`Esc` = cancel.
If the default `SRC_POINTS` are still in use (and `--no-select` is not set), the picker opens automatically.
The chosen points are printed as JSON so you can paste them into a config file.

### Config file (optional)
```json
{
  "distance_threshold_px": 120,
  "confidence_threshold": 0.45,
  "src_points": [[420, 260], [860, 255], [1150, 690], [130, 700]]
}
```
Keys are the field names of `Settings` in `config.py`.

### CLI flags
| Flag | Description |
|------|-------------|
| `--source` | File path, URL, or webcam index (default `0`) |
| `--config` | JSON overrides |
| `--select-points` / `--no-select` | Force / skip the ROI picker |
| `--model`, `--conf`, `--threshold` | Override model, confidence, distance threshold |
| `--roi-only` | Ignore people whose feet are outside the ROI |
| `--no-birdseye` | Hide the top-down inset |

Press **`q`** to quit.

## Calibration note
`DISTANCE_THRESHOLD_PX` is measured in **bird's-eye pixels**, not meters. With the default
400x600 BEV plane, pick `DST_POINTS` so that the plane's scale matches real-world distances
(e.g. if the ROI covers 4 m x 6 m, 100 px ≈ 1 m, so use ~150-200 px for 1.5-2 m).

## Swapping the detector
Any class exposing `detect(frame) -> Detections` can replace `PersonDetector`; nothing else changes.
