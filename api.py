#!/usr/bin/env python3
"""Serves the FireWatch database to the dashboard. Usage: python api.py [port]"""
import datetime as dt, json, math, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import firewatch as fw          # same config, database and harmonization as the collector

HERE = os.path.dirname(os.path.abspath(__file__))

def cell_range(b):
    return (math.floor(b[0] / fw.CELL), math.floor(b[2] / fw.CELL), math.floor(b[1] / fw.CELL), math.floor(b[3] / fw.CELL))

def series(code, bbox=None):
    """Harmonized daily series for a whole country or for any rectangle (w,s,e,n) inside it."""
    m, v = {}, {}
    for s, day in fw.db.execute("SELECT sensor,day FROM daily WHERE aoi=?", (code,)):
        (m if s == "MODIS" else v)[day] = 0          # covered days count as zero unless cells say otherwise
    if bbox:
        rows = fw.db.execute("SELECT sensor,day,SUM(n) FROM cells WHERE aoi=? AND cx BETWEEN ? AND ? AND cy BETWEEN ? AND ? "
                             "GROUP BY sensor,day", (code,) + cell_range(bbox))
    else:
        rows = fw.db.execute("SELECT sensor,day,n FROM daily WHERE aoi=?", (code,))
    for s, day, n in rows:
        (m if s == "MODIS" else v)[day] = n
    return fw.harmonize(m, v)

def calendar(code, bbox):
    h, k, r = series(code, bbox)
    val = {}
    for d, n in h.items():
        key = f"{d.year}-{min(52, (d.timetuple().tm_yday - 1) // 7)}"
        val[key] = round(val.get(key, 0) + n, 2)
    return {"years": sorted({d.year for d in h}), "val": val, "k": k, "r": r,
            "first_day": str(min(h)) if h else None, "last_day": str(max(h)) if h else None}

def cells(code, bbox, a, b):
    """Density of burning per grid cell for a period, in VIIRS-equivalent hotspots."""
    k = series(code, bbox)[1] if bbox else fw.harmonized(code)[1]
    cond, args = "aoi=? AND day BETWEEN ? AND ?", [code, a, b]
    if bbox:
        cond += " AND cx BETWEEN ? AND ? AND cy BETWEEN ? AND ?"
        args += list(cell_range(bbox))
    agg = {}
    for s, cx, cy, post, n in fw.db.execute(
            f"SELECT sensor,cx,cy,day>='2013-01-01',SUM(n) FROM cells WHERE {cond} GROUP BY sensor,cx,cy,day>='2013-01-01'", args):
        if (s == "VIIRS" and post) or (s == "MODIS" and not post):
            agg[(cx, cy)] = agg.get((cx, cy), 0) + (n if s == "VIIRS" else n * k)
    return {"cell": fw.CELL, "cells": [[cx, cy, round(n, 2)] for (cx, cy), n in agg.items()]}

def progress(code):
    """(chunks needed for the full history since 2000, chunks already downloaded) for one country."""
    first = dt.date(fw.today.year - fw.CFG.get("history_years", 26), 1, 1)
    exp = done = 0
    for s, (src, avail) in fw.SRC.items():
        d = max(first, avail)
        d = fw.ANCHOR + dt.timedelta(days=5 * ((d - fw.ANCHOR).days // 5))
        exp += (fw.today - d).days // 5
        done += fw.db.execute("SELECT COUNT(*) FROM chunks2 WHERE aoi=? AND sensor=? AND start>=?", (code, s, str(d))).fetchone()[0]
    return exp, min(done, exp)

def status():
    st = {k: (v, ts) for k, v, ts in fw.db.execute("SELECT k,v,ts FROM status")}
    out = {"collector": st.get("collector"), "countries": []}
    for c in fw.countries():
        code, row = c["code"], {"code": c["code"], "name": c["name"]}
        for s in ("MODIS", "VIIRS"):
            r = fw.db.execute("SELECT COUNT(*),MIN(day),MAX(day) FROM daily WHERE aoi=? AND sensor=?", (code, s)).fetchone()
            row[s] = {"days": r[0], "first": r[1], "last": r[2]}
        exp, done = progress(code)
        row["pct"] = round(100 * done / exp) if exp else 0
        row["left"] = exp - done
        row["forecast"] = bool(fw.db.execute("SELECT 1 FROM forecasts WHERE aoi=? LIMIT 1", (code,)).fetchone())
        row["model"] = (st.get("model:" + code) or [None])[0]
        row["error"] = st.get("error:" + code)
        out["countries"].append(row)
    left = sum(r["left"] for r in out["countries"])
    total = sum(progress(r["code"])[0] for r in out["countries"])
    rate = float(st["rate"][0]) if "rate" in st else 0
    out["overall"] = {"pct": round(100 * (total - left) / total) if total else 0,
                      "eta_h": round(left / rate / 60, 1) if rate > 0 else None}
    return out

class H(BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        name = q.get("aoi", [""])[0]
        bbox = [float(x) for x in q["bbox"][0].split(",")] if "bbox" in q else None
        if u.path == "/api/aois":
            return self.send([{"code": c["code"], "name": c.get("name", c["code"])} for c in fw.countries()])
        if u.path == "/api/status":
            return self.send(status())
        if u.path == "/api/calendar":
            return self.send(calendar(name, bbox))
        if u.path == "/api/cells":
            return self.send(cells(name, bbox, q.get("from", ["0000-01-01"])[0], q.get("to", ["9999-12-31"])[0]))
        if u.path == "/api/forecast":
            r = fw.db.execute("SELECT data FROM forecasts WHERE aoi=? ORDER BY id DESC LIMIT 1", (name,)).fetchone()
            return self.send(json.loads(r[0]) if r else {})
        if u.path == "/api/alerts":
            rows = fw.db.execute("SELECT ts,aoi,kind,level,p,evidence,delivered FROM alerts "
                                 "WHERE aoi IN (?, 'ALL') ORDER BY id DESC LIMIT 50", (name,)).fetchall()
            return self.send([dict(ts=r[0], aoi=r[1], kind=r[2], level=r[3], p=r[4],
                                   evidence=json.loads(r[5]), delivered=r[6]) for r in rows])
        if u.path in ("/", "/index.html"):
            return self.send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        self.send({"error": "not found"}, code=404)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Dashboard: http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), H).serve_forever()