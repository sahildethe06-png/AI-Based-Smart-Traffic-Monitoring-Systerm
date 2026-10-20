import os, sqlite3, time
import pandas as pd
DB = os.getenv("DB_PATH", "data/traffic.db")

def conn():
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    return sqlite3.connect(DB)

def init():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts TEXT, type TEXT,
            track_id INTEGER, cls TEXT, conf REAL, region TEXT, snapshot TEXT, details TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS traffic(ts TEXT, region TEXT, count INTEGER,
            occupancy REAL, level TEXT, in_count INTEGER, out_count INTEGER)""")

def add_event(type_, tid, cls, conf, region, snap, details=""):
    with conn() as c:
        c.execute("INSERT INTO events(ts,type,track_id,cls,conf,region,snapshot,details) VALUES(?,?,?,?,?,?,?,?)",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), type_, tid, cls, conf, region, snap, details))

def add_traffic(region, count, occ, level, i, o):
    with conn() as c:
        c.execute("INSERT INTO traffic VALUES(?,?,?,?,?,?,?)",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), region, count, occ, level, i, o))

def df(table, limit=5000):
    with conn() as c:
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}", c)
