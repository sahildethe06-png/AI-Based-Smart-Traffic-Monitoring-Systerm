import json, time
import cv2, folium, pandas as pd, plotly.express as px, streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium
load_dotenv()
from core import db, routing
from core.camera import Camera
from core.detector import Detector
from core.analytics import Analyzer, DEFAULT_CFG

st.set_page_config("AI Traffic Monitor", "🚦", layout="wide")
db.init(); ss = st.session_state
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
    tdf = db.df("traffic")
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
        rts, demo = routing.get_routes(o, d); rts = routing.rank(rts, cam_pt, level)
        if demo: st.warning("Routing service unavailable: showing DEMO synthetic routes, not real roads.")
        m = folium.Map(o, zoom_start=13); folium.Marker(cam_pt, tooltip=f"Camera ({level})", icon=folium.Icon(color="red")).add_to(m)
        for i, r in enumerate(rts):
            folium.PolyLine(r["coords"], color="green" if i == 0 else "gray", weight=6 if i == 0 else 3,
                            tooltip=f"{'Recommended' if i == 0 else 'Option'} {r['dist']/1000:.1f} km").add_to(m)
            st.write(f"**{'✅ Recommended' if i == 0 else 'Option ' + str(i+1)}** – {r['dist']/1000:.1f} km, "
                     f"~{r['adj_dur']/60:.0f} min | {r['reason']} | Source: {r['src']}")
        st_folium(m, height=450, width=900)
with events:
    ev = db.df("events"); q = st.text_input("Search events")
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
            for n, s in stats["regions"].items(): db.add_traffic(n, s["count"], s["occupancy"], s["level"], stats["in_count"], stats["out_count"])
