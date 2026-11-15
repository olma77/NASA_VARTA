#!/usr/bin/env python3
"""FireWatch: country-level fire outlook (probability model) and alerting for analysts.
Commands: python firewatch.py once | run | report
Env vars: FIRMS_MAP_KEY (required), TELEGRAM_BOT_TOKEN (optional; without it messages are only printed)"""
import os, sys, json, time, math, sqlite3, datetime as dt, traceback
import numpy as np, requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CFG = json.load(open(os.environ.get("FIREWATCH_CONFIG", "config.json")))
KEY = os.environ.get("FIRMS_MAP_KEY", "")
BASE = "https://firms.modaps.eosdis.nasa.gov/api"
TH = CFG.get("thresholds", {"watch": 0.4, "warning": 0.7, "min_lift": 1.3})
TD = dt.timedelta
CELL = 0.1          # spatial grid (degrees) used for area-of-interest queries and density maps
today = dt.datetime.now(dt.timezone.utc).date()
NOW = lambda: dt.datetime.now(dt.timezone.utc)

db = sqlite3.connect(CFG.get("db", "firewatch.db"))
db.executescript("""
CREATE TABLE IF NOT EXISTS daily(aoi TEXT, sensor TEXT, day TEXT, n INTEGER, PRIMARY KEY(aoi,sensor,day));
CREATE TABLE IF NOT EXISTS cells(aoi TEXT, sensor TEXT, day TEXT, cx INTEGER, cy INTEGER, n INTEGER, PRIMARY KEY(aoi,sensor,day,cx,cy));
CREATE INDEX IF NOT EXISTS idx_cells ON cells(aoi,cx,cy);
CREATE TABLE IF NOT EXISTS chunks2(aoi TEXT, sensor TEXT, start TEXT, PRIMARY KEY(aoi,sensor,start));
CREATE TABLE IF NOT EXISTS chunks(aoi TEXT, sensor TEXT, start TEXT, PRIMARY KEY(aoi,sensor,start));
CREATE TABLE IF NOT EXISTS wx(aoi TEXT, day TEXT, tmax REAL, wind REAL, prec REAL, et0 REAL, PRIMARY KEY(aoi,day));
CREATE TABLE IF NOT EXISTS seen(aoi TEXT, key TEXT, PRIMARY KEY(aoi,key));
CREATE TABLE IF NOT EXISTS forecasts(id INTEGER PRIMARY KEY, ts TEXT, aoi TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, ts TEXT, aoi TEXT, kind TEXT, level TEXT,
  p REAL, evidence TEXT, delivered TEXT);
""")

# ------------------------------------------------------------ FIRMS (whole country, by ISO3 code)
SRC = {"MODIS": ("MODIS_SP", dt.date(2000, 11, 1)), "VIIRS": ("VIIRS_SNPP_SP", dt.date(2012, 1, 20))}
ANCHOR = dt.date(2000, 11, 1)

GEO = None

def geometry(code):
    """Country outline (outer rings, [lon,lat]) and its bounding box. Downloaded once, cached in countries.geojson."""
    global GEO
    if GEO is None:
        path = CFG.get("geojson", "countries.geojson")
        if not os.path.exists(path):
            r = requests.get("https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json", timeout=120)
            r.raise_for_status()
            open(path, "wb").write(r.content)
        GEO = {f["id"]: (f["geometry"], f["properties"]["name"]) for f in json.load(open(path, encoding="utf-8"))["features"]}
    if code not in GEO:
        sys.exit(f"Unknown country code {code} (use ISO3, e.g. POL)")
    g = GEO[code][0]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    rings = [np.array(p[0], dtype=float) for p in polys]
    pts = np.vstack(rings)
    bbox = ",".join(f"{v:.3f}" for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    return rings, bbox

def inside(rings, lon, lat):
    """Vectorised ray-casting point-in-polygon test (outer rings only)."""
    res = np.zeros(len(lon), bool)
    for R in rings:
        x, y = R[:, 0], R[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        for i in range(0, len(lon), 1000):
            px, py = lon[i:i + 1000, None], lat[i:i + 1000, None]
            cross = ((y > py) != (y2 > py)) & (px < (x2 - x) * (py - y) / (y2 - y + 1e-12) + x)
            res[i:i + 1000] |= cross.sum(1) % 2 == 1
    return res

def auto_points(code):
    """A few weather sample points spread inside the country."""
    rings, _ = geometry(code)
    pts = np.vstack(rings)
    lo = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 9)[1:-1]
    la = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 9)[1:-1]
    G = np.array([(a, b) for a in la for b in lo])
    G = G[inside(rings, G[:, 1], G[:, 0])]
    if len(G) == 0:
        return [[round(float(pts[:, 1].mean()), 2), round(float(pts[:, 0].mean()), 2)]]
    idx = np.linspace(0, len(G) - 1, min(4, len(G))).astype(int)
    return [[round(float(G[i][0]), 2), round(float(G[i][1]), 2)] for i in idx]

def countries():
    """config.json lists countries as plain ISO3 codes or as objects with optional name/points/telegram_chats."""
    out = []
    for e in CFG["countries"]:
        c = {"code": e} if isinstance(e, str) else dict(e)
        geometry(c["code"])
        c.setdefault("name", GEO[c["code"]][1])
        if "points" not in c:
            c["points"] = auto_points(c["code"])
        out.append(c)
    return out

def firms_csv(src, bbox, start=None, days=5):
    url = f"{BASE}/area/csv/{KEY}/{src}/{bbox}/{days}" + (f"/{start}" if start else "")
    for i in range(3):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
            lines = r.text.strip().split("\n")
            if lines[0].startswith("latitude"):
                return lines
            raise RuntimeError(r.text[:150])
        except Exception:
            if i == 2:
                raise
            time.sleep(2)

def keep(row, h):
    c = row[h.index("confidence")]
    return not (c == "l" or (c.replace(".", "").isdigit() and float(c) < 30))

def parse(lines, rings):
    """Rows that pass the confidence filter AND lie inside the country."""
    h = lines[0].split(",")
    R = [r for r in (l.split(",") for l in lines[1:]) if keep(r, h)]
    if not R:
        return h, []
    la = np.array([float(r[h.index("latitude")]) for r in R])
    lo = np.array([float(r[h.index("longitude")]) for r in R])
    m = inside(rings, lo, la)
    return h, [r for r, k in zip(R, m) if k]

def update_history(c, years=None):
    from concurrent.futures import ThreadPoolExecutor
    code = c["code"]
    rings, bbox = geometry(code)
    first = dt.date(today.year - (years or CFG.get("history_years", 20)), 1, 1)
    todo = []
    for s, (src, avail) in SRC.items():
        d = max(first, avail)
        d = ANCHOR + TD(days=5 * ((d - ANCHOR).days // 5))
        while d <= today:
            if not db.execute("SELECT 1 FROM chunks2 WHERE aoi=? AND sensor=? AND start=?", (code, s, str(d))).fetchone():
                todo.append((s, src, d))
            d += TD(days=5)
    if not todo:
        return
    todo.sort(key=lambda j: j[2], reverse=True)      # newest first: recent seasons become usable at once
    def fetch(job):
        s, src, d = job
        use = src if (today - (d + TD(days=4))).days > 90 else src.replace("_SP", "_NRT")
        return job, firms_csv(use, bbox, d)
    done = 0
    with ThreadPoolExecutor(CFG.get("workers", 6)) as ex:
        for i in range(0, len(todo), 60):
            for job, lines in ex.map(fetch, todo[i:i + 60]):
                s, src, d = job
                h, rs = parse(lines, rings)
                ia, ila, ilo = h.index("acq_date"), h.index("latitude"), h.index("longitude")
                cnt, cl = {}, {}
                for r in rs:
                    k = r[ia]
                    cnt[k] = cnt.get(k, 0) + 1
                    ck = (k, math.floor(float(r[ilo]) / CELL), math.floor(float(r[ila]) / CELL))
                    cl[ck] = cl.get(ck, 0) + 1
                db.execute("DELETE FROM cells WHERE aoi=? AND sensor=? AND day BETWEEN ? AND ?", (code, s, str(d), str(d + TD(days=4))))
                db.executemany("INSERT INTO cells VALUES(?,?,?,?,?,?)", [(code, s, k, cx, cy, n) for (k, cx, cy), n in cl.items()])
                for j in range(5):
                    day = d + TD(days=j)
                    if day <= today:
                        db.execute("INSERT OR REPLACE INTO daily VALUES(?,?,?,?)", (code, s, str(day), cnt.get(str(day), 0)))
                if (today - (d + TD(days=4))).days >= 3:
                    db.execute("INSERT OR REPLACE INTO chunks2 VALUES(?,?,?)", (code, s, str(d)))
            db.commit()
            done += len(todo[i:i + 60])
            print(f"[{code}] history: {done}/{len(todo)} chunks", flush=True)

def harmonize(m, v):
    """m, v: {day string: count} for MODIS and VIIRS. Returns ({date: VIIRS-equivalent count}, scale k, overlap correlation r).
    k = sum(VIIRS)/sum(MODIS) over the overlap years; r = correlation of weekly totals of the two sensors there."""
    ov = [d for d in v if d >= "2013-01-01" and d in m]
    sm, sv = sum(m[d] for d in ov), sum(v[d] for d in ov)
    k = sv / sm if sm > 0 and sv > 0 else 1.0
    wk = {}
    for d in ov:
        dd = dt.date.fromisoformat(d)
        a = wk.setdefault((dd.year, min(52, (dd.timetuple().tm_yday - 1) // 7)), [0, 0])
        a[0] += m[d]; a[1] += v[d]
    r = None
    if len(wk) >= 10:
        arr = np.array(list(wk.values()), float)
        if arr[:, 0].std() > 0 and arr[:, 1].std() > 0:
            r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    h = {d: n for d, n in v.items() if d >= "2013-01-01"}
    h.update({d: n * k for d, n in m.items() if d < "2013-01-01"})
    return {dt.date.fromisoformat(d): n for d, n in h.items()}, k, r

def harmonized(code):
    """Daily country counts in VIIRS-equivalent units. Original sensor rows stay untouched in `daily`/`cells`."""
    m, v = {}, {}
    for s, day, n in db.execute("SELECT sensor,day,n FROM daily WHERE aoi=?", (code,)):
        (m if s == "MODIS" else v)[day] = n
    h, k, _ = harmonize(m, v)
    return h, k

# ------------------------------------------------------------ weather (mean over several points in the country)
WX = ("temperature_2m_max", "wind_speed_10m_max", "precipitation_sum", "et0_fao_evapotranspiration")

def wx_fetch(url, c, **extra):
    pts = c["points"]
    j = requests.get(url, timeout=180, params=dict(
        latitude=",".join(str(p[0]) for p in pts), longitude=",".join(str(p[1]) for p in pts),
        daily=",".join(WX), timezone="UTC", **extra)).json()
    j = j if isinstance(j, list) else [j]
    out = {}
    for i, d in enumerate(j[0]["daily"]["time"]):
        vals = [[x["daily"][k][i] for x in j] for k in WX]
        if not any(None in v for v in vals):
            out[d] = tuple(float(np.mean(v)) for v in vals)
    return out

def update_wx(c, years=None):
    code = c["code"]
    first = dt.date(today.year - (years or CFG.get("history_years", 20)), 1, 1)
    end = today - TD(days=7)
    lo, hi = db.execute("SELECT MIN(day), MAX(day) FROM wx WHERE aoi=?", (code,)).fetchone()
    spans = []
    if lo is None:
        spans.append((first, end))
    else:
        lo, hi = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
        if first < lo:
            spans.append((first, lo - TD(days=1)))
        if hi < end:
            spans.append((hi + TD(days=1), end))
    for a, b in spans:
        if a > b:
            continue
        for d, v in wx_fetch("https://archive-api.open-meteo.com/v1/archive", c, start_date=str(a), end_date=str(b)).items():
            db.execute("INSERT OR REPLACE INTO wx VALUES(?,?,?,?,?,?)", (code, d) + v)
    db.commit()

def feat(w, d):
    p = [w.get(d - TD(days=i)) for i in range(7)]
    if any(x is None for x in p):
        return None
    a = d.timetuple().tm_yday * 2 * math.pi / 365.25
    return [p[0][0], p[0][1], sum(x[2] for x in p), float(np.mean([x[3] for x in p])), math.sin(a), math.cos(a)]

def clim_table(doys, y):
    """P(event day) per day of year: +-7 day window, Beta(1,1) prior -> (mean, events, days)."""
    t = {}
    for g in range(1, 367):
        mk = np.minimum(abs(doys - g), 366 - abs(doys - g)) <= 7
        k, n = int(y[mk].sum()), int(mk.sum())
        t[g] = ((k + 1) / (n + 2), k, n)
    return t

# ------------------------------------------------------------ probability analysis
def analyze(c):
    code, name = c["code"], c.get("name", c["code"])
    h, k = harmonized(code)
    w = {dt.date.fromisoformat(d): v for d, *v in db.execute("SELECT day,tmax,wind,prec,et0 FROM wx WHERE aoi=?", (code,))}
    days = sorted(set(w) & set(h))
    if len(days) < 1000:
        print(f"[{code}] insufficient history ({len(days)} days): no probabilities issued"); return
    # event = day when the country's harmonized count is in its own top (1 - q) share of days
    thr = max(float(np.quantile([h[d] for d in days], CFG.get("event_quantile", 0.9))), 1.0)
    X, y, D = [], [], []
    for d in days:
        f = feat(w, d)
        if f:
            X.append(f); y.append(1 if h[d] >= thr else 0); D.append(d)
    X, y = np.array(X), np.array(y)
    if len(y) < 1000 or y.sum() < 30:
        print(f"[{code}] too few events for a model"); return
    doys = np.array([d.timetuple().tm_yday for d in D])
    yrs_arr = np.array([d.year for d in D])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    # validation on the last 2 unseen years: model vs. climatology (Brier skill score) + calibration
    tr = np.array([d <= D[-1] - TD(days=730) for d in D])
    mdl = mk().fit(X[tr], y[tr])
    ct = clim_table(doys[tr], y[tr])
    pm = mdl.predict_proba(X[~tr])[:, 1]
    pc = np.array([ct[g][0] for g in doys[~tr]])
    bss = 1 - np.mean((pm - y[~tr]) ** 2) / np.mean((pc - y[~tr]) ** 2)
    validated = bool(bss >= CFG.get("min_skill", 0.05))
    q = np.quantile(pm, [0, .2, .4, .6, .8, 1]); calib = []
    for i in range(5):
        m = (pm >= q[i]) & ((pm <= q[i + 1]) if i == 4 else (pm < q[i + 1]))
        if m.sum():
            calib.append([round(float(pm[m].mean()), 3), round(float(y[~tr][m].mean()), 3), int(m.sum())])
    # final model + block bootstrap over years (uncertainty band that respects autocorrelation)
    mdl = mk().fit(X, y)
    ct = clim_table(doys, y)
    rng = np.random.default_rng(0)
    uy = sorted(set(yrs_arr.tolist())); boots = []
    for _ in range(30):
        idx = np.concatenate([np.where(yrs_arr == yv)[0] for yv in rng.choice(uy, len(uy))])
        if y[idx].sum() >= 10:
            boots.append(mk().fit(X[idx], y[idx]))
    wf = {dt.date.fromisoformat(d): v for d, v in wx_fetch("https://api.open-meteo.com/v1/forecast", c, past_days=7, forecast_days=7).items()}
    rows = []
    for d in [d for d in sorted(wf) if d >= today][:7]:
        f = feat(wf, d)
        if not f:
            continue
        p = float(mdl.predict_proba([f])[0, 1])
        bp = [float(b.predict_proba([f])[0, 1]) for b in boots]
        lo, hi = np.percentile(bp, [5, 95]) if len(bp) >= 10 else (p, p)
        pcl, kk, nn = ct[d.timetuple().tm_yday]
        cl, ch = np.percentile(rng.beta(1 + kk, 1 + nn - kk, 5000), [5, 95])
        rows.append(dict(date=str(d), p=round(p, 3), lo=round(float(lo), 3), hi=round(float(hi), 3), clim=round(pcl, 3),
                         clim_lo=round(float(cl), 3), clim_hi=round(float(ch), 3),
                         tmax=round(f[0], 1), wind=round(f[1], 1), rain7=round(f[2], 1)))
    if not rows:
        print(f"[{code}] forecast incomplete"); return
    anyp = lambda key, n: float(1 - np.prod([1 - r[key] for r in rows[:n]]))   # assumes independent days
    p3, c3, p7 = anyp("p", 3), anyp("clim", 3), anyp("p", 7)
    lift = p3 / c3
    level = "WARNING" if p3 >= TH["warning"] and lift >= TH["min_lift"] else \
            "WATCH" if p3 >= TH["watch"] and lift >= TH["min_lift"] else "NORMAL"
    def wsum(end):
        return sum(h.get(end - TD(days=i), 0) for i in range(1, 8))
    r7, norms = wsum(today), []
    for yv in range(today.year - 1, today.year - 9, -1):
        try:
            e = today.replace(year=yv)
        except ValueError:
            e = today.replace(year=yv, day=28)
        if (e - TD(days=7)) in h:
            norms.append(wsum(e))
    n7 = float(np.mean(norms)) if norms else None
    fc = dict(code=code, name=name, ts=NOW().isoformat(timespec="minutes"), threshold=round(thr, 1), days=rows,
              p3=round(p3, 3), clim3=round(c3, 3), p7=round(p7, 3), lift=round(lift, 2), level=level,
              bss=round(float(bss), 3), validated=validated, calib=calib, k=round(k, 3), n_days=len(y),
              event_rate=round(float(y.mean()), 3), season=[round(ct[g][0], 3) for g in range(1, 367)],
              doy=today.timetuple().tm_yday, obs7=round(r7, 1), obs7_norm=None if n7 is None else round(n7, 1))
    db.execute("INSERT INTO forecasts(ts,aoi,data) VALUES(?,?,?)", (fc["ts"], code, json.dumps(fc)))
    db.commit()
    print(f"[{code}] {level} P3d={p3:.2f} climatology={c3:.2f} lift={lift:.2f} BSS={bss:.2f} threshold={thr:.0f}")
    last = db.execute("SELECT level,ts FROM alerts WHERE aoi=? AND kind='outlook' ORDER BY id DESC LIMIT 1", (code,)).fetchone()
    rank = {"NORMAL": 0, "WATCH": 1, "WARNING": 2}
    recent = last and (NOW() - dt.datetime.fromisoformat(last[1])).total_seconds() < 86400
    if level == "NORMAL" or (recent and rank[level] <= rank[last[0]]):
        return
    if not validated and not CFG.get("send_unvalidated", False):
        print(f"[{code}] model not validated (BSS {bss:.2f}): alert withheld"); return
    text = (f"FIREWATCH {level}: {name}\nP(significant burning day in the next 3 days) = {p3:.0%} "
            f"(climatology {c3:.0%}, x{lift:.1f}); next 7 days {p7:.0%}\n"
            f"Model skill vs climatology on unseen years: BSS {bss:.2f}\n"
            f"Preconditions (forecast, country mean): Tmax {rows[0]['tmax']} C, wind {rows[0]['wind']} km/h, rain last 7 d {rows[0]['rain7']} mm\n"
            f"Observations: {r7:.0f} hotspots in the last 7 days" + (f" (normal for this time: {n7:.0f})" if n7 is not None else "") + "\n"
            f"'Significant' = >= {thr:.0f} VIIRS-equivalent hotspots/day (top {1 - CFG.get('event_quantile', 0.9):.0%} of days). "
            f"Statistical outlook, not an observed fire.")
    send(c, text, "outlook", level, p3, fc)

# ------------------------------------------------------------ new detections (one digest per cycle)
def check_detections(c):
    code, name = c["code"], c.get("name", c["code"])
    rings, bbox = geometry(code)
    h, rs = parse(firms_csv("VIIRS_SNPP_NRT", bbox, None, 1), rings)
    cells = {}
    for r in rs:
        la, lo = float(r[h.index("latitude")]), float(r[h.index("longitude")])
        t = r[h.index("acq_date")] + " " + r[h.index("acq_time")].zfill(4)
        key = f"{round(la / .05)}:{round(lo / .05)}:{t[:10]}"
        e = cells.setdefault(key, dict(la=la, lo=lo, n=0, f=0.0, t=t))
        e["n"] += 1; e["f"] = max(e["f"], float(r[h.index("frp")])); e["t"] = max(e["t"], t)
    new = {k: e for k, e in cells.items() if not db.execute("SELECT 1 FROM seen WHERE aoi=? AND key=?", (code, k)).fetchone()}
    if not new:
        return
    for k in new:
        db.execute("INSERT OR IGNORE INTO seen VALUES(?,?)", (code, k))
    top = sorted(new.values(), key=lambda e: -e["f"])[:5]
    text = (f"FIREWATCH DETECTIONS: {name}\n{len(new)} new candidate cluster(s) in the last 24 h. Strongest:\n" +
            "\n".join(f"- {e['la']:.3f}, {e['lo']:.3f} · {e['n']} px · FRP {e['f']:.0f} MW · observed {e['t'][:10]} {e['t'][11:13]}:{e['t'][13:]} UTC" for e in top) +
            f"\nReceived {NOW():%Y-%m-%d %H:%M} UTC. Incident status unknown (no dispatch integration); verification required. "
            f"Pixel centre is not the ignition point; agricultural/industrial heat not filtered.")
    send(c, text, "detection", f"{len(new)} new", None, dict(new=len(new), top=top))

# ------------------------------------------------------------ delivery + audit trail
def send(c, text, kind, level, p, evidence):
    chats = c.get("telegram_chats", CFG.get("telegram_chats", []))
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    res = []
    for ch in chats:
        if not tok:
            res.append(f"{ch}:dry-run"); continue
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", timeout=30,
                              json={"chat_id": ch, "text": text, "disable_web_page_preview": True})
            res.append(f"{ch}:{'ok' if r.ok else 'fail ' + str(r.status_code)}")
        except Exception as e:
            res.append(f"{ch}:error {e}")
    delivered = "; ".join(res) or "No operational delivery route configured"
    db.execute("INSERT INTO alerts(ts,aoi,kind,level,p,evidence,delivered) VALUES(?,?,?,?,?,?,?)",
               (NOW().isoformat(timespec="seconds"), c["code"], kind, level, p, json.dumps(evidence, default=str), delivered))
    db.commit()
    print("\n--- ALERT ---\n" + text + "\nDelivery: " + delivered + "\n")

def daily_digest():
    """One status message per day for all countries, even when everything is NORMAL."""
    if not CFG.get("daily_digest", True):
        return
    key = str(today)
    if db.execute("SELECT 1 FROM alerts WHERE kind='digest' AND level=?", (key,)).fetchone():
        return
    lines = []
    for c in CFG["countries"]:
        code = c if isinstance(c, str) else c["code"]
        r = db.execute("SELECT data FROM forecasts WHERE aoi=? ORDER BY id DESC LIMIT 1", (code,)).fetchone()
        if not r:
            continue
        f = json.loads(r[0]); d0 = f["days"][0]
        obs = f"{f['obs7']:.0f} hotspots/7d" + (f" (norm {f['obs7_norm']:.0f})" if f.get("obs7_norm") is not None else "")
        lines.append(f"{f['name']}: {f['level']}, P(3d) {f['p3']:.0%} (clim {f['clim3']:.0%}) | Tmax {d0['tmax']} C, wind {d0['wind']} km/h, rain7 {d0['rain7']} mm | {obs}")
    if lines:
        send({"code": "ALL"}, f"FIREWATCH DAILY STATUS {key}\n" + "\n".join(lines) +
             "\nStatistical outlook; observations = harmonized satellite hotspots.", "digest", key, None, {"lines": lines})

def cycle():
    cs = countries()
    deep = CFG.get("history_years", 20)
    for yrs in sorted(set(CFG.get("history_phases", [3, deep]))):
        print(f"=== history phase: last {yrs} years for all countries ===", flush=True)
        for c in cs:
            for name, step in (("history", lambda c: update_history(c, yrs)), ("weather", lambda c: update_wx(c, yrs)),
                               ("analysis", analyze), ("detections", check_detections)):
                try:
                    step(c)
                except Exception:
                    print(f"[{c['code']}] {name} failed"); traceback.print_exc()
    daily_digest()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "once"
    if cmd not in ("report", "test", "chatid") and not KEY:
        sys.exit("Set FIRMS_MAP_KEY environment variable")
    if cmd == "run":
        while True:
            cycle(); time.sleep(CFG.get("interval_minutes", 60) * 60)
    elif cmd == "report":
        for r in db.execute("SELECT ts,aoi,kind,level,p,delivered FROM alerts ORDER BY id DESC LIMIT 30"):
            print(r)
    elif cmd == "test":
        send({"code": "ALL"}, "FireWatch test message: delivery works.", "test", "test", None, {})
    elif cmd == "chatid":
        j = requests.get(f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/getUpdates", timeout=30).json()
        ids = {(m["message"]["chat"]["id"], m["message"]["chat"].get("first_name") or m["message"]["chat"].get("title")) for m in j.get("result", []) if "message" in m}
        print(ids or "Nothing yet: open your bot in Telegram and press Start / send any message, then run this again.")
    else:
        cycle()