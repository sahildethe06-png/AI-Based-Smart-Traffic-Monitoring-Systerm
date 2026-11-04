import os, json, time, math, sqlite3
from collections import defaultdict, deque
import cv2, folium, numpy as np, pandas as pd, plotly.express as px, requests, streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium
from ultralytics import YOLO

load_dotenv()

# =====================================================================
# DB  (formerly db.py)
# =====================================================================
DB = os.getenv("DB_PATH", "data/traffic.db")

def db_conn():
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    return sqlite3.connect(DB)

def db_init():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts TEXT, type TEXT,
            track_id INTEGER, cls TEXT, conf REAL, region TEXT, snapshot TEXT, details TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS traffic(ts TEXT, region TEXT, count INTEGER,
            occupancy REAL, level TEXT, in_count INTEGER, out_count INTEGER)""")

def add_event(type_, tid, cls, conf, region, snap, details=""):
    with db_conn() as c:
        c.execute("INSERT INTO events(ts,type,track_id,cls,conf,region,snapshot,details) VALUES(?,?,?,?,?,?,?,?)",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), type_, tid, cls, conf, region, snap, details))

def add_traffic(region, count, occ, level, i, o):
    with db_conn() as c:
        c.execute("INSERT INTO traffic VALUES(?,?,?,?,?,?,?)",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), region, count, occ, level, i, o))

def db_df(table, limit=5000):
    with db_conn() as c:
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}", c)

# =====================================================================
# CAMERA  (formerly camera.py)
# =====================================================================
class Camera:
    """Webcam / RTSP wrapper that never raises on disconnects and auto-reconnects."""
    def __init__(self, source="0", retry_s=3.0):
        self.source, self.retry_s = source, retry_s
        self.cap, self.status, self.last_try = None, "disconnected", 0.0
        self.writer, self.rec_path = None, None

    def open(self):
        try:
            src = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(src)
            ok = self.cap.isOpened()
            self.status = "connected" if ok else "unavailable"
            return ok
        except Exception:
            self.cap, self.status = None, "error"
            return False

    def read(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                if time.time() - self.last_try > self.retry_s:
                    self.last_try = time.time()
                    self.status = "reconnecting"
                    self.open()
                return None
            ok, frame = self.cap.read()
            if not ok:
                self.cap.release(); self.cap = None; self.status = "reconnecting"
                return None
            return frame
        except Exception:
            self.cap, self.status = None, "error"
            return None

    def start_recording(self, frame, folder="recordings", fps=20):
        os.makedirs(folder, exist_ok=True)
        h, w = frame.shape[:2]
        self.rec_path = f"{folder}/rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.writer = cv2.VideoWriter(self.rec_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None

    def release(self):
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
        self.cap, self.status = None, "disconnected"

# =====================================================================
# DETECTOR  (formerly detector.py)
# =====================================================================
COCO = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class Detector:
    def __init__(self, model=None, tracker="bytetrack.yaml", conf=0.35, helmet_model=None):
        self.model = YOLO(model or os.getenv("YOLO_MODEL", "yolov8n.pt"))
        self.tracker, self.conf = tracker, conf
        hp = helmet_model or os.getenv("HELMET_MODEL")
        self.helmet = YOLO(hp) if hp and os.path.exists(hp) else None  # None => helmet detector disabled

    def track(self, frame):
        r = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                             classes=list(COCO), verbose=False)[0]
        out = []
        if r.boxes is None or r.boxes.id is None:
            return out
        for b, i, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(),
                              r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
            out.append(dict(id=i, cls=COCO[c], conf=float(p), box=tuple(float(v) for v in b)))
        return out

    def helmet_boxes(self, frame):
        """Returns [(box, label, conf)] from a user-trained helmet model (classes like 'helmet'/'no_helmet')."""
        if self.helmet is None:
            return []
        r = self.helmet(frame, conf=0.5, verbose=False)[0]
        names = r.names
        return [(tuple(b), names[int(c)].lower(), float(p)) for b, c, p in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]

# =====================================================================
# ANALYTICS  (formerly analytics.py)
# =====================================================================
LEVELS = ["Low", "Moderate", "Heavy", "Critical"]
COLORS = [(0, 200, 0), (0, 220, 220), (0, 140, 255), (0, 0, 230)]  # BGR green/yellow/orange/red
VEH = {"car", "motorcycle", "bus", "truck", "bicycle"}

DEFAULT_CFG = dict(
    lines=[dict(name="Count line", p1=[0.05, 0.6], p2=[0.95, 0.6])],
    stop_line=dict(p1=[0.05, 0.75], p2=[0.95, 0.75]),
    regions=[dict(name="Lane A", poly=[[0.05, 0.4], [0.95, 0.4], [0.95, 0.95], [0.05, 0.95]],
                  allowed_dir=[0, 1], restricted=False, thresholds=[3, 7, 12])],
    occ_thresholds=[0.15, 0.30, 0.50],
    wrong_way_secs=2.0, stop_secs=10.0)

def _side(p, a, b):
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return 0 if abs(v) < 1e-6 else (1 if v > 0 else -1)

def _lvl(v, th):
    return sum(v > t for t in th)

class Analyzer:
    def __init__(self, cfg=None, snap_dir="events"):
        self.cfg = cfg or DEFAULT_CFG
        self.signal = "unknown"           # red | green | unknown (violations ignored if unknown)
        self.snap_dir = snap_dir; os.makedirs(snap_dir, exist_ok=True)
        self.hist = defaultdict(lambda: deque(maxlen=90))
        self.unique = defaultdict(set)
        self.side, self.counted, self.fired = {}, set(), set()
        self.wrong_since, self.still_since, self.dwell_start = {}, {}, {}
        self.in_n = self.out_n = 0
        self.cross_times = deque()
        self.levels, self.heat = {}, None
        self.ratio = defaultdict(lambda: deque(maxlen=30))
        os.makedirs(snap_dir, exist_ok=True)

    def _emit(self, frame, type_, d, region="", details="", key=None):
        k = key or (type_, d["id"])
        if k in self.fired:
            return None
        self.fired.add(k)
        x1, y1, x2, y2 = map(int, d["box"])
        img = frame.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        path = f"{self.snap_dir}/{type_}_{d['id']}_{int(time.time())}.jpg"
        cv2.imwrite(path, img)
        add_event(type_, d["id"], d["cls"], d["conf"], region, path, details)
        return (type_, d["id"], d["cls"])

    def update(self, frame, dets, helmet=()):
        h, w = frame.shape[:2]; now = time.time(); alerts = []
        P = lambda p: (p[0] * w, p[1] * h)
        if self.heat is None:
            self.heat = np.zeros((h // 8 + 1, w // 8 + 1), np.float32)
        self.heat *= 0.9995
        regs = [(r, np.array([P(q) for q in r["poly"]], np.float32)) for r in self.cfg["regions"]]
        rc = {r["name"]: dict(count=0, area=0.0, dwell=[], speeds=[]) for r, _ in regs}
        pers = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]; tid = d["id"]
            c = ((x1 + x2) / 2, y2 if d["cls"] in VEH else (y1 + y2) / 2)
            hs = self.hist[tid]; hs.append((now, c)); self.unique[d["cls"]].add(tid)
            self.heat[min(int(c[1] / 8), self.heat.shape[0] - 1), min(int(c[0] / 8), self.heat.shape[1] - 1)] += 1
            old = next((q for q in hs if now - q[0] <= 1.0), hs[0])
            disp = (c[0] - old[1][0], c[1] - old[1][1]); dist = math.hypot(*disp)
            moving_secs = max(now - old[0], 1e-3)
            # ---- regions / congestion inputs ----
            cur = None
            for r, poly in regs:
                if cv2.pointPolygonTest(poly, c, False) >= 0:
                    cur = r; s = rc[r["name"]]; s["count"] += d["cls"] in VEH
                    s["area"] += (x2 - x1) * (y2 - y1) * (d["cls"] in VEH)
                    s["speeds"].append(dist / moving_secs)
                    st_ = self.dwell_start.setdefault((tid, r["name"]), now); s["dwell"].append(now - st_)
                    break
            # ---- line counting (once per ID per line) ----
            for i, ln in enumerate(self.cfg["lines"]):
                a, b = P(ln["p1"]), P(ln["p2"]); sd = _side(c, a, b); prev = self.side.get((tid, i))
                if prev and sd and sd != prev and (tid, i) not in self.counted and d["cls"] in VEH:
                    self.counted.add((tid, i)); self.cross_times.append(now)
                    if prev < 0 < sd: self.in_n += 1
                    else: self.out_n += 1
                if sd: self.side[(tid, i)] = sd
            # ---- red-light ----
            sl = self.cfg.get("stop_line")
            if sl and d["cls"] in VEH and self.signal in ("red", "green"):
                a, b = P(sl["p1"]), P(sl["p2"]); sd = _side(c, a, b); prev = self.side.get((tid, "stop"))
                if self.signal == "red" and prev and sd and sd != prev:
                    alerts.append(self._emit(frame, "red_light_violation", d, details="crossed stop line on red"))
                if sd: self.side[(tid, "stop")] = sd
            # ---- wrong way (needs allowed_dir per region) ----
            if cur and cur.get("allowed_dir") and d["cls"] in VEH and dist > 20:
                ad = np.array(cur["allowed_dir"], float); ad /= np.linalg.norm(ad) + 1e-9
                cos = float(np.dot(disp, ad) / (dist + 1e-9))
                if cos < -0.5:
                    t0 = self.wrong_since.setdefault(tid, now)
                    if now - t0 >= self.cfg["wrong_way_secs"]:
                        alerts.append(self._emit(frame, "wrong_way", d, cur["name"], f"cos={cos:.2f}"))
                else: self.wrong_since.pop(tid, None)
            # ---- stationary vehicles ----
            if d["cls"] in VEH:
                if dist < 5 and len(hs) > 10:
                    t0 = self.still_since.setdefault(tid, now)
                    if now - t0 >= self.cfg["stop_secs"]:
                        if cur and cur.get("restricted"):
                            alerts.append(self._emit(frame, "illegal_stopping", d, cur["name"]))
                        elif self.levels.get(cur["name"] if cur else None, 0) < 2:
                            alerts.append(self._emit(frame, "stopped_vehicle", d, cur["name"] if cur else "", "not in queue"))
                else: self.still_since.pop(tid, None)
            # ---- persons: fall heuristic ----
            if d["cls"] == "pedestrian":
                pers.append(dist / moving_secs)
                rt = (x2 - x1) / max(y2 - y1, 1); self.ratio[tid].append((now, rt))
                if any(now - t < 1.0 and r0 < 0.8 for t, r0 in self.ratio[tid]) and rt > 1.2:
                    alerts.append(self._emit(frame, "possible_fall", d, details="aspect-ratio heuristic"))
            # ---- helmet (needs trained model) ----
            if d["cls"] == "motorcycle":
                for hb, lab, cf in helmet:
                    cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                    if "no" in lab and x1 <= cx <= x2 and y1 - (y2 - y1) * .5 <= cy <= y2:
                        alerts.append(self._emit(frame, "no_helmet", dict(d, conf=cf), details=lab))
        if len(pers) >= 5 and np.mean(pers) > 0.25 * w:
            alerts.append(self._emit(frame, "sudden_crowd_movement", dict(id=int(now // 10), cls="pedestrian", conf=1.0, box=(0, 0, w, h)), key=("crowd", int(now // 10))))
        while self.cross_times and now - self.cross_times[0] > 60: self.cross_times.popleft()
        # ---- congestion classification ----
        stats = {}
        for r, poly in regs:
            s = rc[r["name"]]; occ = s["area"] / max(cv2.contourArea(poly), 1)
            lv = max(_lvl(s["count"], r["thresholds"]), _lvl(occ, self.cfg["occ_thresholds"]))
            self.levels[r["name"]] = lv
            stats[r["name"]] = dict(count=s["count"], occupancy=round(occ, 3), level=LEVELS[lv],
                                    avg_dwell_s=round(float(np.mean(s["dwell"])) if s["dwell"] else 0, 1),
                                    px_speed=round(float(np.mean(s["speeds"])) if s["speeds"] else 0, 1))
        return [a for a in alerts if a], dict(regions=stats, in_count=self.in_n, out_count=self.out_n,
                                              per_minute=len(self.cross_times),
                                              by_class={k: len(v) for k, v in self.unique.items()})

    def draw(self, frame, dets, stats, show_heat=False):
        h, w = frame.shape[:2]; P = lambda p: (int(p[0] * w), int(p[1] * h))
        if show_heat and self.heat is not None:
            hm = cv2.resize(self.heat, (w, h)); hm = cv2.applyColorMap(np.clip(hm / (hm.max() + 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.6, hm, 0.4, 0)
        ov = frame.copy()
        for r in self.cfg["regions"]:
            pts = np.array([P(q) for q in r["poly"]], np.int32)
            cv2.fillPoly(ov, [pts], COLORS[self.levels.get(r["name"], 0)])
        frame = cv2.addWeighted(ov, 0.2, frame, 0.8, 0)
        for ln in self.cfg["lines"]: cv2.line(frame, P(ln["p1"]), P(ln["p2"]), (255, 0, 0), 2)
        sl = self.cfg.get("stop_line")
        if sl: cv2.line(frame, P(sl["p1"]), P(sl["p2"]), (0, 0, 255) if self.signal == "red" else (0, 200, 0), 2)
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"#{d['id']} {d['cls']} {d['conf']:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return frame

# =====================================================================
# ROUTING  (formerly routing.py)
# =====================================================================
PENALTY = {"Low": 0.0, "Moderate": 0.15, "Heavy": 0.5, "Critical": 1.2}  # configurable ASSUMPTIONS, not measured speeds

def hav(a, b):
    R = 6371000; p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))

def osrm_routes(o, d):
    base = os.getenv("OSRM_URL", "https://router.project-osrm.org")
    r = requests.get(f"{base}/route/v1/driving/{o[1]},{o[0]};{d[1]},{d[0]}", timeout=8,
                     params=dict(alternatives="true", overview="full", geometries="geojson"))
    r.raise_for_status()
    return [dict(coords=[(y, x) for x, y in rt["geometry"]["coordinates"]], dist=rt["distance"], dur=rt["duration"],
                 src="OSRM free-flow estimate (no live traffic)") for rt in r.json()["routes"]]

def demo_routes(o, d):
    """Offline demo: synthetic paths, 30 km/h assumed. NOT real road data."""
    mid = ((o[0] + d[0]) / 2, (o[1] + d[1]) / 2); dx, dy = d[0] - o[0], d[1] - o[1]
    out = []
    for k, off in enumerate([0, 0.01, -0.01]):
        m = (mid[0] - dy * off * 10, mid[1] + dx * off * 10)
        pts = [o, m, d]; dist = sum(hav(pts[i], pts[i + 1]) for i in range(2)) * 1.25
        out.append(dict(coords=pts, dist=dist, dur=dist / (30 / 3.6), src="DEMO synthetic data"))
    return out

def get_routes(o, d):
    try:
        return osrm_routes(o, d), False
    except Exception:
        return demo_routes(o, d), True

def rank(routes, cam_pt, level, radius=150):
    for r in routes:
        hit = any(hav(c, cam_pt) < radius for c in r["coords"])
        r["affected"] = hit
        r["adj_dur"] = r["dur"] * (1 + PENALTY[level]) if hit else r["dur"]
        r["reason"] = (f"Passes the monitored road (camera-observed congestion: {level}); +{int(PENALTY[level]*100)}% "
                       "time is an approximation" if hit else "Avoids the monitored road")
    return sorted(routes, key=lambda r: r["adj_dur"])

# =====================================================================
# STREAMLIT APP  (formerly app.py)
# =====================================================================
st.set_page_config("AI Traffic Monitor", "🚦", layout="wide")
db_init(); ss = st.session_state
for k, v in dict(run=False, cam=None, det=None, an=None, alerts=[], stats=None, last_log=0, rec=False).items():
    ss.setdefault(k, v)

with st.sidebar:
    st.title("🚦 Controls")
    source = st.text_input("Camera (0 = webcam, or rtsp://…)", "0")
    model = st.text_input("YOLO model", "yolov8n.pt")
    tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
    conf = st.slider("Confidence", 0.1, 0.9, 0.35)
    ss_signal = st.selectbox("Signal state (manual input)", ["unknown", "red", "green"])
    heat = st.checkbox("Show heatmap")
    cfg_txt = st.text_area("Lines / regions / thresholds (JSON, normalized 0-1 coords)", json.dumps(DEFAULT_CFG, indent=1), height=300)
    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Start"):
        try:
            cfg = json.loads(cfg_txt)
            ss.det = Detector(model, tracker, conf); ss.an = Analyzer(cfg)
            ss.cam = Camera(source); ss.cam.open(); ss.run = True
        except Exception as e:
            st.error(f"Could not start: {e}")
    if c2.button("⏸ Pause"): ss.run = False
    if c3.button("⏹ Stop"):
        ss.run = False
        if ss.cam: ss.cam.release()
    if ss.cam and st.button("⏺ Stop recording" if ss.rec else "⏺ Record"):
        ss.rec = not ss.rec
        if not ss.rec: ss.cam.stop_recording()
    st.caption(f"Camera: **{ss.cam.status if ss.cam else 'not started'}** | Recording: {'🔴 ON' if ss.rec else 'off'}")

live, analytics, route_tab, events = st.tabs(["📹 Live", "📊 Analytics", "🗺 Routes", "🚨 Events"])
with live:
    metrics, vid, alert_box = st.container(), st.empty(), st.container()
with analytics:
    tdf = db_df("traffic")
    if tdf.empty: st.info("No traffic history yet. Start monitoring.")
    else:
        st.plotly_chart(px.line(tdf, x="ts", y="count", color="region", title="Vehicles per region (measured)"), use_container_width=True)
        st.plotly_chart(px.histogram(tdf, x="level", color="region", title="Congestion level distribution"), use_container_width=True)
        st.download_button("Download traffic CSV", tdf.to_csv(index=False), "traffic_report.csv")
        st.caption("Measured data only. No forecasting is shown: a reliable model needs weeks of history.")
with route_tab:
    a, b = st.columns(2)
    o = [float(x) for x in a.text_input("Origin lat,lon", "21.1458,79.0882").split(",")]
    d = [float(x) for x in b.text_input("Destination lat,lon", "21.1702,79.0822").split(",")]
    cam_pt = [float(x) for x in st.text_input("Monitored camera location lat,lon", "21.1580,79.0850").split(",")]
    worst = "Low"
    if ss.stats:
        lv = [v["level"] for v in ss.stats["regions"].values()]
        worst = max(lv, key=["Low", "Moderate", "Heavy", "Critical"].index)
    level = st.selectbox("Camera-observed congestion", ["Low", "Moderate", "Heavy", "Critical"], ["Low", "Moderate", "Heavy", "Critical"].index(worst))
    if st.button("Find routes"):
        rts, demo = get_routes(o, d); rts = rank(rts, cam_pt, level)
        if demo: st.warning("Routing service unavailable: showing DEMO synthetic routes, not real roads.")
        m = folium.Map(o, zoom_start=13); folium.Marker(cam_pt, tooltip=f"Camera ({level})", icon=folium.Icon(color="red")).add_to(m)
        for i, r in enumerate(rts):
            folium.PolyLine(r["coords"], color="green" if i == 0 else "gray", weight=6 if i == 0 else 3,
                            tooltip=f"{'Recommended' if i == 0 else 'Option'} {r['dist']/1000:.1f} km").add_to(m)
            st.write(f"**{'✅ Recommended' if i == 0 else 'Option ' + str(i+1)}** – {r['dist']/1000:.1f} km, "
                     f"~{r['adj_dur']/60:.0f} min | {r['reason']} | Source: {r['src']}")
        st_folium(m, height=450, width=900)
with events:
    ev = db_df("events"); q = st.text_input("Search events")
    if q and not ev.empty: ev = ev[ev.astype(str).apply(lambda r: q.lower() in " ".join(r).lower(), axis=1)]
    st.dataframe(ev, use_container_width=True)
    if not ev.empty:
        sel = st.selectbox("Review snapshot", ev["snapshot"].tolist())
        try: st.image(sel)
        except Exception: st.caption("Snapshot unavailable.")

if ss.run and ss.cam:
    ss.an.signal = ss_signal
    while ss.run:
        t0 = time.time(); frame = ss.cam.read()
        if frame is None:
            vid.warning(f"Camera {ss.cam.status}… retrying"); time.sleep(0.5); continue
        dets = ss.det.track(frame)
        al, stats = ss.an.update(frame, dets, ss.det.helmet_boxes(frame) if ss.det.helmet else [])
        ss.stats = stats; out = ss.an.draw(frame, dets, stats, heat)
        cv2.putText(out, f"{time.strftime('%Y-%m-%d %H:%M:%S')} FPS {1/max(time.time()-t0,1e-3):.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if ss.rec:
            if ss.cam.writer is None: ss.cam.start_recording(out)
            ss.cam.write(out)
        vid.image(out, channels="BGR", use_container_width=True)
        with metrics:
            k = st.columns(4); k[0].metric("Entering", stats["in_count"]); k[1].metric("Leaving", stats["out_count"])
            k[2].metric("Per minute", stats["per_minute"]); k[3].write(stats["by_class"])
            st.json(stats["regions"], expanded=False)
        for a in al: alert_box.error(f"🚨 {a[0]} | ID {a[1]} | {a[2]}")
        if time.time() - ss.last_log > 5:
            ss.last_log = time.time()
            for n, s in stats["regions"].items(): add_traffic(n, s["count"], s["occupancy"], s["level"], stats["in_count"], stats["out_count"])
