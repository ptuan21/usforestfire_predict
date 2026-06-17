#!/usr/bin/env python3
"""
TẦNG 1 — Dựng PANEL space-time cho California: dự báo KHẢ NĂNG PHÁT CHÁY.
- Lưới ~0.25° (gom 6 pixel gridMET 4km), độ phân giải thời gian: TUẦN, 2015-2020.
- Mỗi dòng = (ô lưới, tuần). Target y=1 nếu tuần đó ô có >=1 đám cháy phát sinh, else 0.
  -> các ô-tuần trống chính là mẫu "không cháy" (negative) tự nhiên.
- Feature nhân quả từ gridMET: ERC, BI, độ ẩm nhiên liệu (fm100/fm1000), gió, nhiệt độ,
  độ ẩm, mưa, VPD, hạn hán PDSI.
Output: data/processed/ca_panel.sqlite (bảng panel)
"""
import xarray as xr, numpy as np, pandas as pd, sqlite3, os, time
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed")
BBOX=dict(lat=slice(42.1,32.4),lon=slice(-124.5,-114.0)); COARSE=6; YEARS=range(2015,2021)
BASE="http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET"
# (tên biến, hàm gom không gian & thời gian)
# fm1000 + erc đã phản ánh khô nhiên liệu/hạn hán (PDSI không có trên THREDDS này)
VARS=[("erc","max"),("bi","max"),("fm100","min"),("fm1000","min"),("vs","max"),
      ("tmmx","max"),("rmin","min"),("pr","sum"),("vpd","max")]

def week_start(s): return pd.to_datetime(s).dt.to_period("W").dt.start_time

CANON={}   # iy/ix -> lat/lon center (lấy từ biến đầu tiên, dùng chung)

def load_var(short, agg):
    """Trả DataFrame [week, iy, ix, <short>] gom theo ô 0.25° × tuần.
    Dùng chỉ số ô nguyên iy/ix (bất biến giữa các biến) làm khoá, tránh lệch float."""
    frames=[]
    for y in YEARS:
        url=f"{BASE}/{short}/{short}_{y}.nc"
        try:
            ds=xr.open_dataset(url); v=list(ds.data_vars)[0]
            da=ds[v].sel(**BBOX)
            co=getattr(da.coarsen(lat=COARSE,lon=COARSE,boundary="trim"),agg)()
            if not CANON:
                CANON["lat"]=np.round(co["lat"].values,4); CANON["lon"]=np.round(co["lon"].values,4)
            # thay toạ độ bằng chỉ số nguyên 0..n-1
            co=co.assign_coords(lat=np.arange(co.sizes["lat"]),lon=np.arange(co.sizes["lon"]))
            co=co.rename({"lat":"iy","lon":"ix"})
            wk=getattr(co.resample(day="1W"),agg)()
            df=wk.to_dataframe().reset_index()[["day","iy","ix",v]].dropna()
            df=df.rename(columns={v:short})
            df["week"]=week_start(df["day"]); df=df.drop(columns="day")
            frames.append(df); ds.close()
            print(f"   {short} {y}: {len(df):,}",flush=True)
        except Exception as e:
            print(f"   !! {short} {y} lỗi: {e}",flush=True)
    if not frames: return None
    out=pd.concat(frames,ignore_index=True)
    # khử trùng tuần biên năm (xuất hiện ở 2 file liền kề) bằng cùng hàm gom
    pdagg={"max":"max","min":"min","sum":"sum","mean":"mean"}[agg]
    out=out.groupby(["week","iy","ix"],as_index=False)[short].agg(pdagg)
    return out

def main():
    t0=time.time()
    panel=None
    for short,agg in VARS:
        d=load_var(short,agg)
        if d is None: continue
        panel = d if panel is None else panel.merge(d,on=["week","iy","ix"],how="inner")
        print(f">> sau {short}: panel {len(panel):,} dòng | {time.time()-t0:.0f}s",flush=True)

    # map chỉ số ô -> toạ độ tâm ô
    panel["lat"]=CANON["lat"][panel["iy"].values]
    panel["lon"]=CANON["lon"][panel["ix"].values]

    # target: cháy CA 2015-2020 -> snap vào ô lưới gần nhất + tuần
    print(">> gắn nhãn cháy (positives) ...",flush=True)
    con=sqlite3.connect(os.path.join(PROC,"fires_ml.sqlite"))
    f=pd.read_sql("""SELECT LATITUDE,LONGITUDE,DISCOVERY_DATE FROM fires_clean
        WHERE STATE='CA' AND GEO_VALID=1 AND FIRE_YEAR BETWEEN 2015 AND 2020
        AND DISCOVERY_DATE IS NOT NULL""",con); con.close()
    f["iy"]=np.abs(f["LATITUDE"].values[:,None]-CANON["lat"]).argmin(1)
    f["ix"]=np.abs(f["LONGITUDE"].values[:,None]-CANON["lon"]).argmin(1)
    f["week"]=week_start(f["DISCOVERY_DATE"])
    pos=f.groupby(["week","iy","ix"]).size().reset_index(name="n_fires")
    panel=panel.merge(pos,on=["week","iy","ix"],how="left")
    panel["n_fires"]=panel["n_fires"].fillna(0)
    panel["y"]=(panel["n_fires"]>0).astype(int)

    # đặc trưng thời gian
    panel["week"]=pd.to_datetime(panel["week"])
    panel["year"]=panel["week"].dt.year
    woy=panel["week"].dt.isocalendar().week.astype(int)
    panel["woy_sin"]=np.sin(2*np.pi*woy/52); panel["woy_cos"]=np.cos(2*np.pi*woy/52)
    panel["month"]=panel["week"].dt.month
    panel=panel[panel["year"].between(2015,2020)].copy()
    panel["week"]=panel["week"].dt.strftime("%Y-%m-%d")

    print(f"\n>> PANEL: {len(panel):,} ô-tuần | {panel['lat'].nunique()} vĩ × {panel['lon'].nunique()} kinh")
    print(f">> positives: {panel['y'].sum():,} ({panel['y'].mean()*100:.2f}%)")
    vcols=[s for s,_ in VARS if s in panel.columns]
    print(panel[vcols].describe().round(2).to_string())

    con=sqlite3.connect(os.path.join(PROC,"ca_panel.sqlite"))
    con.execute("DROP TABLE IF EXISTS panel")
    panel.to_sql("panel",con,index=False,chunksize=50000)
    con.execute("CREATE INDEX idx_w ON panel(week)"); con.commit(); con.close()
    print(f">> đã lưu data/processed/ca_panel.sqlite | tổng {time.time()-t0:.0f}s")

if __name__=="__main__": main()
