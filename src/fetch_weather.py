#!/usr/bin/env python3
"""
Ghép dữ liệu thời tiết/khí hậu (Open-Meteo Archive / ERA5) cho cháy 2015-2026.
Kỹ thuật: gom theo ô lưới 0.25° (= độ phân giải gốc ERA5) -> mỗi ô gọi API 1 lần
lấy cả chuỗi ngày 2014-10..2026-06, rồi join mọi cháy trong ô theo ngày phát hiện.
Có RESUME: bỏ qua ô đã xong (bảng done_cells).

Output: data/processed/fires_weather.sqlite (bảng fires_wx: ROW_ID + wx_*)
Biến: nhiệt độ (max/min/mean), độ ẩm RH, gió max, VPD, ET0, mưa,
      + mưa tích luỹ 7/30/90 ngày & số ngày khô 30 ngày (proxy hạn hán).
"""
import sqlite3, os, sys, time, json, urllib.parse, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
IN_DB  = os.path.join(ROOT,"data","processed","fires_ml.sqlite")
OUT_DB = os.path.join(ROOT,"data","processed","fires_weather.sqlite")

START = "2014-10-01"          # đủ 90 ngày trước cháy 2015-01
END   = "2026-06-16"
START_D = np.datetime64(START)
DAILY = ["temperature_2m_max","temperature_2m_min","temperature_2m_mean",
         "precipitation_sum","windspeed_10m_max","et0_fao_evapotranspiration",
         "relative_humidity_2m_mean","vapour_pressure_deficit_max"]
WX_COLS = ["wx_tmax","wx_tmin","wx_tmean","wx_rh","wx_wind","wx_vpd","wx_et0",
           "wx_precip","wx_precip_7d","wx_precip_30d","wx_precip_90d","wx_dry_days_30d"]
BATCH = 3                     # số ô lưới / call (giới hạn "too much data" của API)
GRID  = 2                     # 1/GRID độ; 2 -> lưới 0.5° (~50km)
URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch(coords, tries=6):
    lats = ",".join(f"{a:.4f}" for a,_ in coords)
    lons = ",".join(f"{b:.4f}" for _,b in coords)
    qs = urllib.parse.urlencode({"latitude":lats,"longitude":lons,
        "start_date":START,"end_date":END,"daily":",".join(DAILY),"timezone":"UTC"})
    for i in range(tries):
        try:
            req = urllib.request.Request(URL+"?"+qs, headers={"User-Agent":"wildfire-ml/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            return d if isinstance(d,list) else [d]
        except urllib.error.HTTPError as e:
            if e.code==429: time.sleep(15*(i+1)); continue
            if i==tries-1: raise
            time.sleep(3*(i+1))
        except Exception:
            if i==tries-1: raise
            time.sleep(3*(i+1))
    return None

def main():
    con_in = sqlite3.connect(IN_DB)
    df = pd.read_sql("SELECT ROW_ID, LATITUDE, LONGITUDE, DISCOVERY_DATE "
                     "FROM fires_clean WHERE GEO_VALID=1 AND FIRE_YEAR>=2015 "
                     "AND DISCOVERY_DATE IS NOT NULL", con_in)
    con_in.close()
    df["d"] = pd.to_datetime(df["DISCOVERY_DATE"], errors="coerce")
    df = df.dropna(subset=["d"])
    df["clat"] = (np.round(df.LATITUDE*GRID)/GRID)
    df["clon"] = (np.round(df.LONGITUDE*GRID)/GRID)
    df["cell"] = df.clat.astype(str)+","+df.clon.astype(str)
    df["didx"] = ((df.d.values - START_D).astype("timedelta64[D]")).astype(int)
    cells = df.groupby("cell")
    cell_centers = df.drop_duplicates("cell").set_index("cell")[["clat","clon"]]
    all_cells = list(cell_centers.index)
    print(f">> {len(df):,} cháy | {len(all_cells):,} ô lưới 0.25° | batch {BATCH}")

    con = sqlite3.connect(OUT_DB)
    con.execute(f"CREATE TABLE IF NOT EXISTS fires_wx (ROW_ID INTEGER PRIMARY KEY, "
                + ", ".join(f"{c} REAL" for c in WX_COLS) + ")")
    con.execute("CREATE TABLE IF NOT EXISTS done_cells (cell TEXT PRIMARY KEY)")
    con.commit()
    done = set(r[0] for r in con.execute("SELECT cell FROM done_cells"))
    todo = [c for c in all_cells if c not in done]
    print(f">> còn {len(todo):,} ô chưa xử lý (đã xong {len(done):,})")

    ndays = int((np.datetime64(END)-START_D).astype("timedelta64[D]").astype(int))+1
    t0=time.time(); processed=0
    for bi in range(0, len(todo), BATCH):
        batch = todo[bi:bi+BATCH]
        coords = [(cell_centers.loc[c,"clat"], cell_centers.loc[c,"clon"]) for c in batch]
        res = fetch(coords)
        if res is None:
            print("!! batch lỗi, bỏ qua", bi); continue
        rows=[]
        for c, r in zip(batch, res):
            day = r.get("daily",{})
            if not day or "precipitation_sum" not in day:
                con.execute("INSERT OR IGNORE INTO done_cells VALUES (?)",(c,)); continue
            def arr(k): return np.array(day.get(k,[None]*ndays), dtype="float64")
            precip = np.nan_to_num(arr("precipitation_sum"), nan=0.0)
            cum = np.concatenate([[0.0], np.cumsum(precip)])   # cum[i]=sum precip[:i]
            dry = (precip < 1.0).astype(float)
            cum_dry = np.concatenate([[0.0], np.cumsum(dry)])
            series = {k:arr(k) for k in DAILY}
            n = len(precip)
            for rid, didx in cells.get_group(c)[["ROW_ID","didx"]].itertuples(index=False):
                if didx<0 or didx>=n: continue
                i=int(didx)
                def back(cumv, w): return float(cumv[i+1]-cumv[max(0,i-w+1)])
                rows.append((int(rid),
                    series["temperature_2m_max"][i], series["temperature_2m_min"][i],
                    series["temperature_2m_mean"][i], series["relative_humidity_2m_mean"][i],
                    series["windspeed_10m_max"][i], series["vapour_pressure_deficit_max"][i],
                    series["et0_fao_evapotranspiration"][i], float(precip[i]),
                    back(cum,7), back(cum,30), back(cum,90), back(cum_dry,30)))
            con.execute("INSERT OR IGNORE INTO done_cells VALUES (?)",(c,))
        if rows:
            con.executemany(f"INSERT OR REPLACE INTO fires_wx VALUES (?,{','.join('?'*len(WX_COLS))})", rows)
        con.commit()
        processed += len(batch)
        el=time.time()-t0; rate=processed/el if el else 0
        eta=(len(todo)-processed)/rate/60 if rate else 0
        sys.stdout.write(f"\r  ô {processed}/{len(todo)} | cháy ghép {con.execute('SELECT COUNT(*) FROM fires_wx').fetchone()[0]:,} | {rate:.1f} ô/s | ETA {eta:.1f} phút")
        sys.stdout.flush()
        time.sleep(1.5)
    print(f"\n>> XONG. Tổng cháy có thời tiết: {con.execute('SELECT COUNT(*) FROM fires_wx').fetchone()[0]:,}")
    con.close()

if __name__=="__main__":
    main()
