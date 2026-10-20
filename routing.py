import os, math
import requests

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
