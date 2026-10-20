# AI Traffic Monitoring System
Run: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cp .env.example .env && streamlit run app.py`

## Modules
`core/camera.py` webcam/RTSP + auto-reconnect + recording | `core/detector.py` YOLO + ByteTrack/BoT-SORT |
`core/analytics.py` counting, congestion, wrong-way, red-light, illegal stop, fall/crowd, helmet | `core/routing.py` OSRM + demo fallback | `core/db.py` SQLite

## Honest limitations
- Speed is NOT shown in km/h (no calibration). `px_speed` is uncalibrated pixels/s.
- Helmet detection is disabled until you set `HELMET_MODEL` to a model you trained
  (e.g. `yolo train data=helmet.yaml model=yolov8n.pt epochs=50`, classes `helmet`/`no_helmet`, from a public helmet dataset).
- Red-light detection runs only if signal is set to red/green; `unknown` = no violations reported.
- Fall / crowd detection are heuristics (bbox aspect ratio / mean pedestrian speed): review before acting on them.
- Route delay penalties (core/routing.py `PENALTY`) are labelled approximations, not live traffic. No forecasting model is included.
- Only use RTSP streams you are authorized to access.
