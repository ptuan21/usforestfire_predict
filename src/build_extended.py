#!/usr/bin/env python3
"""
Mở rộng dữ liệu cháy rừng Mỹ FPA FOD (1992-2015) tới 2026.
Nguồn:
  - 1992-2015: FPA_FOD_20170508.sqlite (file gốc, đã làm sạch)
  - 2016-2020: FPA FOD 6th edition  (ArcGIS REST, đã làm sạch)
  - 2021-2026: NIFC WFIGS Incident Locations (operational, chỉ wildfire WF)
Output:
  - fires_extended.sqlite  (bảng phẳng fires_all)
  - fires_extended.csv
"""
import sqlite3, urllib.parse, urllib.request, json, time, sys, datetime, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DB  = os.path.join(ROOT, "data", "raw", "FPA_FOD_20170508.sqlite")
OUT_DB  = os.path.join(ROOT, "data", "interim", "fires_extended.sqlite")
OUT_CSV = os.path.join(ROOT, "data", "interim", "fires_extended.csv")

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

# ------- unified target schema -------
COLS = [
    "DATA_SOURCE","SRC_ID","FIRE_YEAR","DISCOVERY_DATE","DISCOVERY_DOY",
    "DISCOVERY_TIME","CONT_DATE","CONT_DOY","STAT_CAUSE_DESCR","CAUSE_RAW",
    "FIRE_SIZE","FIRE_SIZE_CLASS","LATITUDE","LONGITUDE","OWNER_DESCR",
    "STATE","COUNTY","FIPS_CODE","FIPS_NAME","FIRE_NAME",
]

# 13-class chuẩn FPA (giữ nhất quán với 1992-2015)
def size_class(acres):
    if acres is None: return None
    try: a = float(acres)
    except: return None
    if a < 0.26: return "A"
    if a < 10:   return "B"
    if a < 100:  return "C"
    if a < 300:  return "D"
    if a < 1000: return "E"
    if a < 5000: return "F"
    return "G"

# 6th edition NWCG_GENERAL_CAUSE -> 13-class legacy
MAP_6TH = {
    "Natural":"Lightning",
    "Arson/incendiarism":"Arson",
    "Debris and open burning":"Debris Burning",
    "Equipment and vehicle use":"Equipment Use",
    "Smoking":"Smoking",
    "Recreation and ceremony":"Campfire",
    "Railroad operations and maintenance":"Railroad",
    "Power generation/transmission/distribution":"Powerline",
    "Fireworks":"Fireworks",
    "Firearms and explosives use":"Miscellaneous",
    "Misuse of fire by a minor":"Children",
    "Other causes":"Miscellaneous",
    "Missing data/not specified/undetermined":"Missing/Undefined",
}
# NIFC FireCauseGeneral -> 13-class (best effort)
MAP_NIFC_GEN = {
    "Natural":"Lightning",
    "Incendiary":"Arson",
    "Debris Burning":"Debris Burning",
    "Debris and Open Burning":"Debris Burning",
    "Equipment":"Equipment Use",
    "Equipment and Vehicle Use":"Equipment Use",
    "Smoking":"Smoking",
    "Campfire":"Campfire",
    "Recreation and Ceremony":"Campfire",
    "Railroad":"Railroad",
    "Railroad Operations and Maintenance":"Railroad",
    "Powerline":"Powerline",
    "Power Generation/Transmission/Distribution":"Powerline",
    "Fireworks":"Fireworks",
    "Firearms and Explosives Use":"Miscellaneous",
    "Coal Seam":"Miscellaneous",
    "Other":"Miscellaneous",
    "Other Causes":"Miscellaneous",
    "Playing with Fire":"Children",
    "Misuse of Fire by a Minor":"Children",
}
def nifc_cause(fc, fcg):
    if fcg and fcg in MAP_NIFC_GEN: return MAP_NIFC_GEN[fcg]
    if fc == "Natural": return "Lightning"
    if fc in ("Undetermined","Unknown", None, ""): return "Missing/Undefined"
    if fc == "Human": return "Miscellaneous"
    return "Missing/Undefined"

def ms_to_date(ms):
    if ms is None: return None, None
    try:
        dt = datetime.datetime.utcfromtimestamp(ms/1000.0)
        return dt.strftime("%Y-%m-%d"), dt.timetuple().tm_yday
    except: return None, None

def http_json(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == tries-1: raise
            time.sleep(2*(i+1))
    return None

def fetch_layer(base, where, oid_field, mapper, label):
    """Paginate an ArcGIS layer, yield mapped rows."""
    out = []
    offset = 0
    page = 2000
    while True:
        params = {
            "where": where, "outFields":"*", "returnGeometry":"false",
            "orderByFields": oid_field, "resultOffset": offset,
            "resultRecordCount": page, "f":"json",
        }
        url = base + "/query?" + urllib.parse.urlencode(params)
        d = http_json(url)
        feats = d.get("features", [])
        if not feats: break
        for f in feats:
            out.append(mapper(f["attributes"]))
        offset += len(feats)
        sys.stdout.write(f"\r[{label}] fetched {offset}"); sys.stdout.flush()
        if not d.get("exceededTransferLimit") and len(feats) < page:
            break
    print(f"\r[{label}] DONE: {offset} rows")
    return out

# ---------- mappers ----------
def map_6th(a):
    dd, doy = ms_to_date(a.get("discovery_date"))
    if a.get("discovery_doy"): doy = a.get("discovery_doy")
    cd, cdoy = ms_to_date(a.get("cont_date"))
    if a.get("cont_doy"): cdoy = a.get("cont_doy")
    gen = a.get("nwcg_general_cause")
    sz  = a.get("fire_size")
    return (
        "FPA_FOD_6TH_2016_2020", str(a.get("fod_id")), a.get("fire_year"),
        dd, doy, a.get("discovery_time"), cd, cdoy,
        MAP_6TH.get(gen, "Missing/Undefined"), gen,
        sz, a.get("fire_size_class") or size_class(sz),
        a.get("latitude"), a.get("longitude"), a.get("owner_descr"),
        a.get("state"), a.get("county"), a.get("fips_code"), a.get("fips_name"),
        a.get("fire_name"),
    )

def map_nifc(a):
    dd, doy = ms_to_date(a.get("FireDiscoveryDateTime"))
    cd, cdoy = ms_to_date(a.get("ControlDateTime") or a.get("ContainmentDateTime"))
    sz = a.get("FinalAcres") or a.get("IncidentSize") or a.get("DiscoveryAcres")
    fy = None
    if dd: fy = int(dd[:4])
    fc, fcg = a.get("FireCause"), a.get("FireCauseGeneral")
    return (
        "NIFC_WFIGS_2021_2026", a.get("LocalIncidentIdentifier") or a.get("IrwinID"),
        fy, dd, doy, None, cd, cdoy,
        nifc_cause(fc, fcg), (fcg or fc),
        sz, size_class(sz),
        a.get("InitialLatitude"), a.get("InitialLongitude"), None,
        a.get("POOState"), None, None, None,
        a.get("IncidentName"),
    )

def get_oid_field(base):
    d = http_json(base + "?f=json")
    for f in d.get("fields", []):
        if f.get("type") == "esriFieldTypeOID":
            return f["name"]
    return "objectid"

# ---------- build ----------
def main():
    if os.path.exists(OUT_DB): os.remove(OUT_DB)
    con = sqlite3.connect(OUT_DB)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cols_ddl = ",\n  ".join(f"{c} {'REAL' if c in ('FIRE_SIZE','LATITUDE','LONGITUDE') else ('INTEGER' if c in ('FIRE_YEAR','DISCOVERY_DOY','CONT_DOY') else 'TEXT')}" for c in COLS)
    cur.execute(f"CREATE TABLE fires_all (\n  ROW_ID INTEGER PRIMARY KEY AUTOINCREMENT,\n  {cols_ddl}\n)")
    placeholders = ",".join("?"*len(COLS))
    ins = f"INSERT INTO fires_all ({','.join(COLS)}) VALUES ({placeholders})"

    # 1) original 1992-2015
    print(">> Stage 1: migrate 1992-2015 from original DB")
    src = sqlite3.connect(SRC_DB)
    sc = src.cursor()
    sc.execute("""
        SELECT FOD_ID, FIRE_YEAR, date(DISCOVERY_DATE), DISCOVERY_DOY, DISCOVERY_TIME,
               date(CONT_DATE), CONT_DOY, STAT_CAUSE_DESCR, FIRE_SIZE, FIRE_SIZE_CLASS,
               LATITUDE, LONGITUDE, OWNER_DESCR, STATE, COUNTY, FIPS_CODE, FIPS_NAME, FIRE_NAME
        FROM Fires
    """)
    batch, n = [], 0
    while True:
        rows = sc.fetchmany(20000)
        if not rows: break
        for r in rows:
            (fod,fy,dd,doy,dt,cd,cdoy,cause,sz,szc,lat,lon,own,st,cty,fips,fipsn,fn) = r
            batch.append(("FPA_FOD_1992_2015", str(fod), fy, dd, doy, dt, cd, cdoy,
                          cause, cause, sz, szc, lat, lon, own, st, cty, fips, fipsn, fn))
        cur.executemany(ins, batch); n += len(batch); batch=[]
        sys.stdout.write(f"\r[orig] {n}"); sys.stdout.flush()
    src.close()
    print(f"\r[orig] DONE: {n} rows"); con.commit()

    # 2) 2016-2020 FPA FOD 6th edition
    print(">> Stage 2: fetch 2016-2020 (FPA FOD 6th edition)")
    base6 = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_FireOccurrence6thEdition_01/MapServer/29"
    oid6 = get_oid_field(base6)
    rows6 = fetch_layer(base6, "fire_year>=2016 AND fire_year<=2020", oid6, map_6th, "6th")
    cur.executemany(ins, rows6); con.commit()

    # 3) 2021-2026 NIFC WFIGS (wildfire only)
    print(">> Stage 3: fetch 2021-2026 (NIFC WFIGS, WF only)")
    baseN = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations/FeatureServer/0"
    oidN = get_oid_field(baseN)
    whereN = ("IncidentTypeCategory='WF' AND FireDiscoveryDateTime>=timestamp '2021-01-01 00:00:00' "
              "AND FireDiscoveryDateTime<timestamp '2027-01-01 00:00:00'")
    rowsN = fetch_layer(baseN, whereN, oidN, map_nifc, "nifc")
    cur.executemany(ins, rowsN); con.commit()

    # indexes
    print(">> creating indexes")
    cur.execute("CREATE INDEX idx_year ON fires_all(FIRE_YEAR)")
    cur.execute("CREATE INDEX idx_state ON fires_all(STATE)")
    cur.execute("CREATE INDEX idx_src ON fires_all(DATA_SOURCE)")
    con.commit()

    # summary
    print("\n===== SUMMARY by source =====")
    for r in cur.execute("SELECT DATA_SOURCE, MIN(FIRE_YEAR), MAX(FIRE_YEAR), COUNT(*) FROM fires_all GROUP BY DATA_SOURCE ORDER BY 2"):
        print(f"  {r[0]:<25} {r[1]}-{r[2]}  {r[3]:>9,} rows")
    total = cur.execute("SELECT COUNT(*) FROM fires_all").fetchone()[0]
    print(f"  {'TOTAL':<25}            {total:>9,} rows")
    con.commit()

    # export CSV
    print(">> exporting CSV ...")
    import csv
    with open(OUT_CSV, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["ROW_ID"]+COLS)
        for row in cur.execute(f"SELECT ROW_ID,{','.join(COLS)} FROM fires_all"):
            w.writerow(row)
    print(f"   wrote {OUT_CSV}")
    con.close()
    print("ALL DONE.")

if __name__ == "__main__":
    main()
