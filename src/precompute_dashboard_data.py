#!/usr/bin/env python3
"""
Chuẩn bị dữ liệu NHẸ cho dashboard (không cần LightGBM/sqlite lớn khi chạy app).
- ca_risk.parquet/sqlite: rủi ro cháy đã tính sẵn cho từng ô-tuần California.
- summary CSV: số cháy theo năm / bang / nguyên nhân / tháng (toàn quốc).
Output: data/processed/dashboard/
"""
import sqlite3, os, numpy as np, pandas as pd, joblib
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed"); OUT=os.path.join(PROC,"dashboard"); os.makedirs(OUT,exist_ok=True)

# ---- 1) rủi ro CA đã tính sẵn ----
print(">> tính rủi ro CA ...")
b=joblib.load(os.path.join(ROOT,"models","fire_occurrence_lgbm.joblib"))
model=b["model"]; FEATS=b["features"]
con=sqlite3.connect(os.path.join(PROC,"ca_panel_feat.sqlite"))
df=pd.read_sql("SELECT * FROM panel",con); con.close()
df=df.dropna(subset=FEATS).copy()
df["risk"]=model.predict_proba(df[FEATS])[:,1]
keep=["week","year","month","lat","lon","risk","y","erc","fm1000","vs","tmmx","rmin","hist_fire_log","elev"]
risk=df[keep].copy()
con=sqlite3.connect(os.path.join(OUT,"ca_risk.sqlite"))
con.execute("DROP TABLE IF EXISTS risk"); risk.to_sql("risk",con,index=False,chunksize=50000)
con.execute("CREATE INDEX idx_w ON risk(week)"); con.commit(); con.close()
print(f"   ca_risk.sqlite: {len(risk):,} ô-tuần, {risk['week'].nunique()} tuần")

# ---- 2) tổng hợp toàn quốc (từ fires_ml) ----
print(">> tổng hợp toàn quốc ...")
con=sqlite3.connect(os.path.join(PROC,"fires_ml.sqlite"))
def q(sql): return pd.read_sql(sql,con)
q("SELECT FIRE_YEAR year, DATA_SOURCE source, COUNT(*) n, SUM(FIRE_SIZE) acres "
  "FROM fires_clean GROUP BY FIRE_YEAR, DATA_SOURCE").to_csv(f"{OUT}/by_year.csv",index=False)
q("SELECT STATE, COUNT(*) n FROM fires_clean WHERE STATE_VALID=1 GROUP BY STATE").to_csv(f"{OUT}/by_state.csv",index=False)
q("SELECT STAT_CAUSE_DESCR cause, COUNT(*) n FROM fires_clean GROUP BY STAT_CAUSE_DESCR").to_csv(f"{OUT}/by_cause.csv",index=False)
q("SELECT MONTH month, COUNT(*) n FROM fires_clean WHERE MONTH IS NOT NULL GROUP BY MONTH").to_csv(f"{OUT}/by_month.csv",index=False)
q("SELECT FIRE_YEAR year, MONTH month, COUNT(*) n FROM fires_clean WHERE MONTH IS NOT NULL GROUP BY FIRE_YEAR,MONTH").to_csv(f"{OUT}/year_month.csv",index=False)
con.close()

# ---- 3) metrics model ----
m=b["metrics"]
pd.DataFrame([{"model":"Cháy lớn (LightGBM)","metric":"PR-AUC test","value":0.226},
   {"model":"Diện tích (hồi quy)","metric":"R²","value":0.247},
   {"model":"Nguyên nhân (12 lớp)","metric":"accuracy","value":0.394},
   {"model":"Phát cháy CA (temporal)","metric":"ROC-AUC","value":round(m["temporal_roc"],3)},
   {"model":"Phát cháy CA (temporal)","metric":"PR-AUC","value":round(m["temporal_pr"],3)},
   {"model":"Phát cháy CA (spatial CV)","metric":"ROC-AUC","value":round(m["spatial_roc"],3)},
  ]).to_csv(f"{OUT}/model_metrics.csv",index=False)
print(">> xong. Files trong",OUT)
for f in sorted(os.listdir(OUT)): print("   ",f)
