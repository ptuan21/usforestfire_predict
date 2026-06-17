#!/usr/bin/env python3
"""
TẦNG 1 (bước 3): thêm ĐỊA HÌNH + LỊCH SỬ CHÁY vào panel California.
- elev: độ cao (Open-Meteo elevation API) cho từng ô lưới.
- hist_fire_log: log(số cháy 1992-2014 trong ô) — "độ dễ cháy" tĩnh, KHÔNG leakage (panel 2015-2020).
- clim_cell_month: tỉ lệ cháy lịch sử của ô rơi vào tháng đó (khí hậu mùa cháy theo vị trí).
Output: data/processed/ca_panel_feat.sqlite (bảng panel)
"""
import sqlite3, os, time, json, urllib.parse, urllib.request, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed")

print(">> đọc panel ...")
con=sqlite3.connect(os.path.join(PROC,"ca_panel.sqlite"))
panel=pd.read_sql("SELECT * FROM panel",con); con.close()
cells=panel[["lat","lon"]].drop_duplicates().reset_index(drop=True)
print(f"   {len(panel):,} ô-tuần | {len(cells)} ô")

# ---- 1) độ cao qua Open-Meteo (batch 100) ----
print(">> lấy độ cao (elevation) ...")
elev=np.full(len(cells),np.nan)
for i in range(0,len(cells),100):
    b=cells.iloc[i:i+100]
    qs=urllib.parse.urlencode({"latitude":",".join(f"{x:.4f}" for x in b.lat),
                               "longitude":",".join(f"{x:.4f}" for x in b.lon)})
    for attempt in range(5):
        try:
            r=urllib.request.urlopen("https://api.open-meteo.com/v1/elevation?"+qs,timeout=60)
            elev[i:i+len(b)]=json.loads(r.read())["elevation"]; break
        except Exception as e:
            if attempt==4: print("   !! elev lỗi",i,e)
            time.sleep(20)
    time.sleep(8)   # tránh giới hạn ~600 vị trí/phút
cells["elev"]=elev
print(f"   elev: {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} m, thiếu {np.isnan(elev).sum()}")

# ---- 2) lịch sử cháy 1992-2014 (CA) snap vào ô lưới ----
print(">> lịch sử cháy 1992-2014 ...")
con=sqlite3.connect(os.path.join(PROC,"fires_ml.sqlite"))
h=pd.read_sql("""SELECT LATITUDE,LONGITUDE,MONTH FROM fires_clean
   WHERE STATE='CA' AND GEO_VALID=1 AND FIRE_YEAR<2015""",con); con.close()
lats=np.sort(cells["lat"].unique())[::-1]; lons=np.sort(cells["lon"].unique())
h["lat"]=lats[np.abs(h["LATITUDE"].values[:,None]-lats).argmin(1)]
h["lon"]=lons[np.abs(h["LONGITUDE"].values[:,None]-lons).argmin(1)]
# tổng theo ô
cnt=h.groupby(["lat","lon"]).size().reset_index(name="hist_n")
cells=cells.merge(cnt,on=["lat","lon"],how="left"); cells["hist_n"]=cells["hist_n"].fillna(0)
cells["hist_fire_log"]=np.log1p(cells["hist_n"])
# theo (ô, tháng): tỉ lệ mùa
cm=h.groupby(["lat","lon","MONTH"]).size().reset_index(name="cm_n")
tot=h.groupby(["lat","lon"]).size().reset_index(name="tot")
cm=cm.merge(tot,on=["lat","lon"]); cm["clim_cell_month"]=cm["cm_n"]/cm["tot"]
cm=cm.rename(columns={"MONTH":"month"})[["lat","lon","month","clim_cell_month"]]

# ---- gộp vào panel ----
panel=panel.merge(cells[["lat","lon","elev","hist_fire_log"]],on=["lat","lon"],how="left")
panel=panel.merge(cm,on=["lat","lon","month"],how="left")
panel["clim_cell_month"]=panel["clim_cell_month"].fillna(0.0)
panel["elev"]=panel["elev"].fillna(panel["elev"].median())

con=sqlite3.connect(os.path.join(PROC,"ca_panel_feat.sqlite"))
con.execute("DROP TABLE IF EXISTS panel"); panel.to_sql("panel",con,index=False,chunksize=50000)
con.execute("CREATE INDEX idx_w ON panel(week)"); con.commit(); con.close()
print(f">> lưu ca_panel_feat.sqlite | {len(panel):,} dòng, {panel.shape[1]} cột")
print(panel[["elev","hist_fire_log","clim_cell_month"]].describe().round(2).to_string())
