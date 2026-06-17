#!/usr/bin/env python3
"""
TẦNG 1 — BẢN ĐỒ RỦI RO CHÁY California từ model dự báo phát cháy.
Vẽ xác suất cháy theo ô lưới cho 1 tuần cao điểm (cháy nhiều nhất 2020) và 1 tuần mùa đông,
chấm đỏ = ô thực sự có cháy tuần đó. Minh hoạ model dùng được để ra bản đồ rủi ro.
Output: reports/figures/risk_map.png
"""
import sqlite3, os, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed"); FIG=os.path.join(ROOT,"reports","figures")

b=joblib.load(os.path.join(ROOT,"models","fire_occurrence_lgbm.joblib"))
model=b["model"]; FEATS=b["features"]
con=sqlite3.connect(os.path.join(PROC,"ca_panel_feat.sqlite"))
df=pd.read_sql("SELECT * FROM panel WHERE year=2020",con); con.close()
df=df.dropna(subset=FEATS).copy()
df["risk"]=model.predict_proba(df[FEATS])[:,1]

# chọn tuần cao điểm (nhiều cháy nhất) và tuần thấp điểm
byw=df.groupby("week")["y"].sum().sort_values()
wk_hi=byw.index[-1]; wk_lo=byw[byw>=0].index[0]
print("tuần cao điểm:",wk_hi,"| tuần thấp:",wk_lo)

fig,axes=plt.subplots(1,2,figsize=(15,7),sharex=True,sharey=True)
for ax,wk,title in [(axes[0],wk_hi,"Tuần CAO ĐIỂM"),(axes[1],wk_lo,"Tuần mùa đông")]:
    d=df[df.week==wk]
    sc=ax.scatter(d.lon,d.lat,c=d.risk,cmap="YlOrRd",vmin=0,vmax=1,s=34,marker="s")
    fires=d[d.y==1]
    ax.scatter(fires.lon,fires.lat,s=42,facecolors="none",edgecolors="blue",linewidths=1.3,
               label=f"cháy thực ({len(fires)} ô)")
    ax.set_title(f"{title} — {wk}\nrủi ro TB={d.risk.mean():.2f}")
    ax.set_xlabel("Kinh độ"); ax.legend(loc="upper right",fontsize=8)
axes[0].set_ylabel("Vĩ độ")
fig.colorbar(sc,ax=axes,label="Xác suất cháy (model)",shrink=.7)
fig.suptitle("Bản đồ RỦI RO CHÁY California — model dự báo phát cháy theo ô-tuần (gridMET + địa hình + lịch sử)",fontsize=12)
fig.savefig(f"{FIG}/risk_map.png",bbox_inches="tight"); plt.close()
print("đã lưu reports/figures/risk_map.png")

# đánh giá: precision@top-k ô rủi ro nhất tuần cao điểm
d=df[df.week==wk_hi].sort_values("risk",ascending=False)
for k in [20,50,100]:
    hit=d.head(k)["y"].mean()
    print(f"   top-{k} ô rủi ro nhất tuần cao điểm: {hit*100:.0f}% thực sự có cháy")
