#!/usr/bin/env python3
"""
Làm sạch + feature engineering cho dữ liệu cháy rừng mở rộng (1992-2026).
Input : fires_extended.sqlite  (bảng fires_all)
Output: fires_ml.sqlite (bảng fires_clean) + fires_clean.csv + report in màn hình
"""
import sqlite3, numpy as np, pandas as pd, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IN_DB  = os.path.join(ROOT, "data", "interim", "fires_extended.sqlite")
OUT_DB = os.path.join(ROOT, "data", "processed", "fires_ml.sqlite")
OUT_CSV= os.path.join(ROOT, "data", "processed", "fires_clean.csv")

VALID_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI "
    "MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR".split())
REGION = {  # Census-style + nhóm theo địa lý cháy
 **{s:"West" for s in "WA OR CA NV ID MT WY UT CO AZ NM AK HI".split()},
 **{s:"Great Plains" for s in "ND SD NE KS OK TX".split()},
 **{s:"Midwest" for s in "MN IA MO WI IL MI IN OH".split()},
 **{s:"Southeast" for s in "AR LA MS AL GA FL SC NC TN KY VA WV".split()},
 **{s:"Northeast" for s in "PA NY NJ CT RI MA VT NH ME MD DE DC".split()},
 **{s:"Other" for s in "PR".split()},
}
SEASON = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
          6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}

print(">> đọc dữ liệu từ SQLite ...")
con = sqlite3.connect(IN_DB)
df = pd.read_sql("SELECT * FROM fires_all", con)
con.close()
n0 = len(df)
print(f"   {n0:,} bản ghi")

# ---------- 1. CHUẨN HOÁ KIỂU & NGÀY ----------
df["DISCOVERY_DATE"] = pd.to_datetime(df["DISCOVERY_DATE"], errors="coerce")
df["CONT_DATE"]      = pd.to_datetime(df["CONT_DATE"], errors="coerce")
for c in ["FIRE_SIZE","LATITUDE","LONGITUDE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["FIRE_YEAR","DISCOVERY_DOY","CONT_DOY"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# ---------- 2. LÀM SẠCH ----------
# 2a. State hợp lệ — chuẩn hoá tiền tố NIFC "US-CA" -> "CA"
df["STATE"] = (df["STATE"].str.strip().str.upper()
                 .str.replace(r"^US-", "", regex=True))
df["STATE_VALID"] = df["STATE"].isin(VALID_STATES)

# 2b. Toạ độ: đưa giá trị bất khả thi về NaN (lat phải [15,72], lon [-180,-65])
bad_geo = (~df["LATITUDE"].between(15,72)) | (~df["LONGITUDE"].between(-180,-65))
n_badgeo = int((bad_geo & df["LATITUDE"].notna()).sum())
df.loc[bad_geo, ["LATITUDE","LONGITUDE"]] = np.nan
df["GEO_VALID"] = df["LATITUDE"].notna() & df["LONGITUDE"].notna()

# 2c. Fire size: âm/0 -> NaN cho cột log; giữ FIRE_SIZE gốc
df.loc[df["FIRE_SIZE"]<=0, "FIRE_SIZE"] = np.nan

# 2d. Năm hợp lệ
df = df[df["FIRE_YEAR"].between(1992,2026)].copy()

# 2e. Bỏ trùng lặp hoàn toàn theo khoá tự nhiên
key = ["DATA_SOURCE","SRC_ID","DISCOVERY_DATE","LATITUDE","LONGITUDE","FIRE_SIZE"]
n_before = len(df)
df = df.drop_duplicates(subset=key).copy()
n_dup = n_before - len(df)

# ---------- 3. FEATURE ENGINEERING ----------
dd = df["DISCOVERY_DATE"]
# fallback DOY nếu thiếu ngày nhưng có DISCOVERY_DOY
df["DOY"] = dd.dt.dayofyear
df["DOY"] = df["DOY"].fillna(df["DISCOVERY_DOY"]).astype("Int64")
df["MONTH"]      = dd.dt.month.astype("Int64")
df["DAY_OF_WEEK"]= dd.dt.dayofweek.astype("Int64")   # 0=Mon
df["IS_WEEKEND"] = (df["DAY_OF_WEEK"]>=5).astype("Int64")
df["SEASON"]     = df["MONTH"].map(SEASON)
df["DECADE"]     = (df["FIRE_YEAR"]//10*10).astype("Int64")

# mã hoá tuần hoàn (chu kỳ mùa) — quan trọng cho mô hình
doy = df["DOY"].astype("float")
df["DOY_SIN"] = np.sin(2*np.pi*doy/365.25)
df["DOY_COS"] = np.cos(2*np.pi*doy/365.25)
mo = df["MONTH"].astype("float")
df["MONTH_SIN"] = np.sin(2*np.pi*mo/12)
df["MONTH_COS"] = np.cos(2*np.pi*mo/12)

# kích thước
df["FIRE_SIZE_LOG"] = np.log1p(df["FIRE_SIZE"])
df["IS_LARGE_FIRE"] = df["FIRE_SIZE_CLASS"].isin(["D","E","F","G"]).astype("Int64")

# thời gian cháy (ngày)
dur = (df["CONT_DATE"] - df["DISCOVERY_DATE"]).dt.days
dur = dur.where((dur>=0) & (dur<=365))   # loại giá trị vô lý
df["DURATION_DAYS"] = dur

# vùng địa lý
df["REGION"] = df["STATE"].map(REGION).fillna("Other")

# cờ tin cậy nhãn (cho người dùng lọc khi train)
df["CAUSE_KNOWN"] = (~df["STAT_CAUSE_DESCR"].isin(["Missing/Undefined"])).astype("Int64")

# ---------- 4. SẮP CỘT & LƯU ----------
cols = ["ROW_ID","DATA_SOURCE","SRC_ID","FIRE_YEAR","DECADE","DISCOVERY_DATE","DOY",
        "MONTH","SEASON","DAY_OF_WEEK","IS_WEEKEND","DOY_SIN","DOY_COS","MONTH_SIN","MONTH_COS",
        "STAT_CAUSE_DESCR","CAUSE_RAW","CAUSE_KNOWN","FIRE_SIZE","FIRE_SIZE_LOG","FIRE_SIZE_CLASS",
        "IS_LARGE_FIRE","DURATION_DAYS","LATITUDE","LONGITUDE","GEO_VALID","STATE","STATE_VALID",
        "REGION","COUNTY","FIPS_CODE","FIRE_NAME"]
df = df[cols]

print(">> lưu fires_ml.sqlite ...")
if os.path.exists(OUT_DB): os.remove(OUT_DB)
con = sqlite3.connect(OUT_DB)
# datetime -> ISO text cho sqlite
out = df.copy()
out["DISCOVERY_DATE"] = out["DISCOVERY_DATE"].dt.strftime("%Y-%m-%d")
out.to_sql("fires_clean", con, index=False, chunksize=50000)
con.execute("CREATE INDEX idx_year ON fires_clean(FIRE_YEAR)")
con.execute("CREATE INDEX idx_state ON fires_clean(STATE)")
con.execute("CREATE INDEX idx_src ON fires_clean(DATA_SOURCE)")
con.commit(); con.close()

print(">> lưu fires_clean.csv ...")
out.to_csv(OUT_CSV, index=False)

# ---------- 5. BÁO CÁO ----------
print("\n================ BÁO CÁO XỬ LÝ ================")
print(f"Bản ghi vào         : {n0:,}")
print(f"Toạ độ bất khả thi  : {n_badgeo:,} -> NaN")
print(f"Trùng lặp đã bỏ     : {n_dup:,}")
print(f"Bản ghi ra          : {len(df):,}")
print(f"\nSố cột feature      : {len(cols)}")
print("\n-- % thiếu các cột chính --")
for c in ["DISCOVERY_DATE","LATITUDE","FIRE_SIZE","DURATION_DAYS","MONTH"]:
    print(f"  {c:<16}: {df[c].isna().mean()*100:5.1f}%")
print("\n-- Phân bố theo REGION --")
print(df["REGION"].value_counts().to_string())
print("\n-- GEO_VALID / CAUSE_KNOWN theo nguồn --")
print(df.groupby("DATA_SOURCE")[["GEO_VALID","CAUSE_KNOWN"]].mean().round(3).to_string())
print("\nOK -> fires_ml.sqlite + fires_clean.csv")
