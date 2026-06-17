#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mở rộng bản đồ rủi ro cháy California sang 2021-2026 (dữ liệu thật gần hiện tại).
- Lấy gridMET 2021-2026 (tái dùng build_ca_panel.load_var).
- Ghép đặc trưng tĩnh (elev, hist_fire_log, clim_cell_month) từ ca_panel_feat (theo ô).
- Chấm điểm rủi ro bằng model fire_occurrence_lgbm; nhãn cháy thực tế từ NIFC.
- NỐI vào data/processed/dashboard/ca_risk.sqlite để dashboard chọn được 2021-2026.
"""
import os, sqlite3, numpy as np, pandas as pd, joblib, importlib.util, time
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed"); DASHD=os.path.join(PROC,"dashboard")

# nạp module build_ca_panel để tái dùng load_var + hằng số
spec=importlib.util.spec_from_file_location("bp",os.path.join(HERE,"build_ca_panel.py"))
bp=importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
bp.YEARS=range(2021,2027); bp.CANON.clear()

print(">> lấy gridMET 2021-2026 ...",flush=True)
t0=time.time(); panel=None
for short,agg in bp.VARS:
    d=bp.load_var(short,agg)
    panel=d if panel is None else panel.merge(d,on=["week","iy","ix"],how="inner")
    print(f">> sau {short}: {len(panel):,} dòng | {time.time()-t0:.0f}s",flush=True)
panel["lat"]=bp.CANON["lat"][panel["iy"].values]; panel["lon"]=bp.CANON["lon"][panel["ix"].values]

# đặc trưng thời gian
panel["week"]=pd.to_datetime(panel["week"]); panel["year"]=panel["week"].dt.year
woy=panel["week"].dt.isocalendar().week.astype(int)
panel["woy_sin"]=np.sin(2*np.pi*woy/52); panel["woy_cos"]=np.cos(2*np.pi*woy/52)
panel["month"]=panel["week"].dt.month
panel=panel[panel["year"].between(2021,2026)].copy()

# đặc trưng tĩnh từ ca_panel_feat (theo ô / ô-tháng)
con=sqlite3.connect(os.path.join(PROC,"ca_panel_feat.sqlite"))
feat=pd.read_sql("SELECT DISTINCT lat,lon,elev,hist_fire_log FROM panel",con)
cm=pd.read_sql("SELECT DISTINCT lat,lon,month,clim_cell_month FROM panel",con); con.close()
panel=panel.merge(feat,on=["lat","lon"],how="left").merge(cm,on=["lat","lon","month"],how="left")
panel["clim_cell_month"]=panel["clim_cell_month"].fillna(0.0)
panel["elev"]=panel["elev"].fillna(panel["elev"].median())
panel["hist_fire_log"]=panel["hist_fire_log"].fillna(0.0)

# nhãn cháy thực tế từ NIFC (CA 2021-2026)
con=sqlite3.connect(os.path.join(PROC,"fires_ml.sqlite"))
f=pd.read_sql("""SELECT LATITUDE,LONGITUDE,DISCOVERY_DATE FROM fires_clean
   WHERE STATE='CA' AND GEO_VALID=1 AND FIRE_YEAR BETWEEN 2021 AND 2026 AND DISCOVERY_DATE IS NOT NULL""",con); con.close()
f["iy"]=np.abs(f["LATITUDE"].values[:,None]-bp.CANON["lat"]).argmin(1)
f["ix"]=np.abs(f["LONGITUDE"].values[:,None]-bp.CANON["lon"]).argmin(1)
f["week"]=bp.week_start(f["DISCOVERY_DATE"])
pos=f.groupby(["week","iy","ix"]).size().reset_index(name="nf")
panel=panel.merge(pos,on=["week","iy","ix"],how="left")
panel["y"]=(panel["nf"].fillna(0)>0).astype(int)

# chấm điểm rủi ro
b=joblib.load(os.path.join(ROOT,"models","fire_occurrence_lgbm.joblib"))
model=b["model"]; FEATS=b["features"]
panel=panel.dropna(subset=FEATS).copy()
panel["risk"]=model.predict_proba(panel[FEATS])[:,1]
panel["week"]=panel["week"].dt.strftime("%Y-%m-%d")
print(f"\n>> 2021-2026: {len(panel):,} ô-tuần | positives {panel['y'].mean()*100:.2f}%")

# NỐI vào ca_risk.sqlite (chỉ giữ các cột dashboard cần)
keep=["week","year","month","lat","lon","risk","y","erc","fm1000","vs","tmmx","rmin","hist_fire_log","elev"]
con=sqlite3.connect(os.path.join(DASHD,"ca_risk.sqlite"))
# xoá nếu đã có 2021-2026 (cho phép chạy lại), rồi nối
con.execute("DELETE FROM risk WHERE year>=2021"); con.commit()
panel[keep].to_sql("risk",con,if_exists="append",index=False,chunksize=50000)
yrs=pd.read_sql("SELECT MIN(year),MAX(year),COUNT(*) FROM risk",con).iloc[0].tolist()
con.close()
print(f">> ca_risk.sqlite hiện có năm {yrs[0]}–{yrs[1]} | tổng {yrs[2]:,} ô-tuần")
