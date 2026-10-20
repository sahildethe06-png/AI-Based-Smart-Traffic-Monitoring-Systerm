import os, time, math
from collections import defaultdict, deque
import cv2, numpy as np
from . import db

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
        db.add_event(type_, d["id"], d["cls"], d["conf"], region, path, details)
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
                    st = self.dwell_start.setdefault((tid, r["name"]), now); s["dwell"].append(now - st)
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
