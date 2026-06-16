#!/usr/bin/env python3
"""
Feature engineering nâng cao (chỉ dùng dữ liệu sẵn có, CHỐNG LEAKAGE).
- Thêm OWNER_DESCR (join từ interim theo ROW_ID).
- Đặc trưng LỊCH SỬ time-aware: với mỗi đám cháy năm Y, dùng CHỈ các năm < Y:
    hist_n_cell    : số cháy đã từng xảy ra trong ô lưới 0.25°
    hist_rate_cell : tỉ lệ cháy lớn trong ô (làm mượt Bayesian)
    hist_n_state   : số cháy đã từng xảy ra trong bang
    hist_rate_state: tỉ lệ cháy lớn trong bang
Output: data/processed/fires_model.sqlite (bảng model_data)
"""
import sqlite3, os, numpy as np, pandas as pd
from sklearn.neighbors import BallTree
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed"); INTER=os.path.join(ROOT,"data","interim")
ALPHA=20.0   # độ mượt cho tỉ lệ lịch sử

print(">> đọc fires_clean + OWNER_DESCR ...")
con=sqlite3.connect(os.path.join(PROC,"fires_ml.sqlite"))
df=pd.read_sql("""SELECT ROW_ID,DATA_SOURCE,FIRE_YEAR,MONTH,IS_LARGE_FIRE,LATITUDE,LONGITUDE,
   DOY_SIN,DOY_COS,MONTH_SIN,MONTH_COS,DAY_OF_WEEK,IS_WEEKEND,STATE,REGION,STAT_CAUSE_DESCR
   FROM fires_clean WHERE GEO_VALID=1""",con); con.close()
con=sqlite3.connect(os.path.join(INTER,"fires_extended.sqlite"))
own=pd.read_sql("SELECT ROW_ID, OWNER_DESCR FROM fires_all",con); con.close()
df=df.merge(own,on="ROW_ID",how="left")
df["OWNER_DESCR"]=df["OWNER_DESCR"].fillna("Unknown")
print(f"   {len(df):,} dòng (GEO_VALID)")

df["cell"]=(np.round(df.LATITUDE*4)/4).astype(str)+","+(np.round(df.LONGITUDE*4)/4).astype(str)
g=df[df.FIRE_YEAR<=2017]["IS_LARGE_FIRE"].mean()   # prior toàn cục từ train
print(f"   tỉ lệ cháy lớn prior (<=2017): {g:.4f}")

def hist_feats(df, keycol, prefix):
    # đếm theo (key, year)
    a=df.groupby([keycol,"FIRE_YEAR"]).agg(n=("IS_LARGE_FIRE","size"),
                                           k=("IS_LARGE_FIRE","sum")).reset_index()
    a=a.sort_values([keycol,"FIRE_YEAR"])
    # cumsum bao gồm năm hiện tại -> trừ năm hiện tại = strictly prior
    a["cn"]=a.groupby(keycol)["n"].cumsum()-a["n"]
    a["ck"]=a.groupby(keycol)["k"].cumsum()-a["k"]
    a[prefix+"_n"]=a["cn"]
    a[prefix+"_rate"]=(a["ck"]+ALPHA*g)/(a["cn"]+ALPHA)
    return a[[keycol,"FIRE_YEAR",prefix+"_n",prefix+"_rate"]]

print(">> tính đặc trưng lịch sử theo ô lưới ...")
hc=hist_feats(df,"cell","hist_cell")
print(">> tính đặc trưng lịch sử theo bang ...")
hs=hist_feats(df,"STATE","hist_state")
df=df.merge(hc,on=["cell","FIRE_YEAR"],how="left").merge(hs,on=["STATE","FIRE_YEAR"],how="left")
df["hist_cell_n"]=df["hist_cell_n"].fillna(0); df["hist_state_n"]=df["hist_state_n"].fillna(0)
df["hist_cell_rate"]=df["hist_cell_rate"].fillna(g); df["hist_state_rate"]=df["hist_state_rate"].fillna(g)
df["hist_cell_logn"]=np.log1p(df["hist_cell_n"]); df["hist_state_logn"]=np.log1p(df["hist_state_n"])

# --- (1a) tỉ lệ cháy lớn theo MÙA: ô lưới × tháng, time-aware ---
print(">> tính đặc trưng mùa-vụ (ô × tháng) ...")
df["cellmonth"]=df["cell"]+"|"+df["MONTH"].astype(str)
ALPHA_M=30.0
a=df.groupby(["cellmonth","FIRE_YEAR"]).agg(n=("IS_LARGE_FIRE","size"),
                                            k=("IS_LARGE_FIRE","sum")).reset_index().sort_values(["cellmonth","FIRE_YEAR"])
a["cn"]=a.groupby("cellmonth")["n"].cumsum()-a["n"]
a["ck"]=a.groupby("cellmonth")["k"].cumsum()-a["k"]
a["hist_cellmonth_rate"]=(a["ck"]+ALPHA_M*g)/(a["cn"]+ALPHA_M)
df=df.merge(a[["cellmonth","FIRE_YEAR","hist_cellmonth_rate"]],on=["cellmonth","FIRE_YEAR"],how="left")
df["hist_cellmonth_rate"]=df["hist_cellmonth_rate"].fillna(g)

# --- (1b) khoảng cách (km) tới cháy LỚN gần nhất trong QUÁ KHỨ (years < Y) ---
print(">> tính khoảng cách tới cháy lớn gần nhất (BallTree haversine, theo năm) ...")
R=6371.0
df["dist_largefire_km"]=np.nan
years=sorted(df.FIRE_YEAR.unique())
acc=[]; tree=None
for y in years:
    m=df.FIRE_YEAR==y
    if tree is not None:
        q=np.radians(df.loc[m,["LATITUDE","LONGITUDE"]].values)
        d,_=tree.query(q,k=1); df.loc[m,"dist_largefire_km"]=d[:,0]*R
    ly=df[(df.FIRE_YEAR==y)&(df.IS_LARGE_FIRE==1)][["LATITUDE","LONGITUDE"]].values
    if len(ly): acc.append(np.radians(ly))
    if acc: tree=BallTree(np.vstack(acc),metric="haversine")
# năm đầu chưa có quá khứ -> điền bằng phân vị 95 (coi như "rất xa")
fillv=df["dist_largefire_km"].quantile(0.95)
df["dist_largefire_km"]=df["dist_largefire_km"].fillna(fillv).clip(upper=1000)
df["dist_largefire_log"]=np.log1p(df["dist_largefire_km"])

keep=["ROW_ID","DATA_SOURCE","FIRE_YEAR","IS_LARGE_FIRE","LATITUDE","LONGITUDE",
      "DOY_SIN","DOY_COS","MONTH_SIN","MONTH_COS","DAY_OF_WEEK","IS_WEEKEND",
      "STATE","REGION","STAT_CAUSE_DESCR","OWNER_DESCR",
      "hist_cell_logn","hist_cell_rate","hist_state_logn","hist_state_rate",
      "hist_cellmonth_rate","dist_largefire_log"]
out=df[keep]
con=sqlite3.connect(os.path.join(PROC,"fires_model.sqlite"))
con.execute("DROP TABLE IF EXISTS model_data")
out.to_sql("model_data",con,index=False,chunksize=50000)
con.execute("CREATE INDEX idx_y ON model_data(FIRE_YEAR)"); con.commit(); con.close()
print(f">> lưu model_data: {len(out):,} dòng, {len(keep)} cột -> data/processed/fires_model.sqlite")
print(out[["hist_cell_rate","hist_cellmonth_rate","dist_largefire_log"]].describe().round(3).to_string())
