#!/usr/bin/env python3
"""
Biểu đồ TƯƠNG QUAN feature + CHỈ SỐ MÔ HÌNH (dùng model đã lưu + model_data).
Xuất vào reports/figures/:
  corr_matrix.png            ma trận tương quan các feature số + target
  corr_target.png            tương quan từng feature với IS_LARGE_FIRE
  model_threshold_curve.png  precision/recall/F1 theo ngưỡng
  model_confusion.png        ma trận nhầm lẫn tại ngưỡng best-F1
  model_calibration.png      đường hiệu chuẩn xác suất
"""
import sqlite3, os, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, confusion_matrix)
from sklearn.calibration import calibration_curve

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
DB=os.path.join(ROOT,"data","processed","fires_model.sqlite")
FIG=os.path.join(ROOT,"reports","figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"figure.dpi":110})

con=sqlite3.connect(DB); df=pd.read_sql("SELECT * FROM model_data",con); con.close()
bundle=joblib.load(os.path.join(ROOT,"models","large_fire_hgb.joblib"))
clf=bundle["model"]; FEATS=bundle["features"]; NUM=bundle["num"]; CAT=bundle["cat"]
t_f1=bundle["thresholds"]["f1"]

# ---------- 1) TƯƠNG QUAN ----------
numcols=NUM+["IS_LARGE_FIRE"]
C=df[numcols].astype(float).corr()
fig,ax=plt.subplots(figsize=(11,9))
im=ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(len(numcols))); ax.set_xticklabels(numcols,rotation=90,fontsize=8)
ax.set_yticks(range(len(numcols))); ax.set_yticklabels(numcols,fontsize=8)
for i in range(len(numcols)):
    for j in range(len(numcols)):
        ax.text(j,i,f"{C.iloc[i,j]:.2f}",ha="center",va="center",fontsize=6,
                color="white" if abs(C.iloc[i,j])>0.5 else "black")
ax.set_title("Ma trận tương quan (Pearson) — feature số + target"); fig.colorbar(im,shrink=.8)
fig.tight_layout(); fig.savefig(f"{FIG}/corr_matrix.png"); plt.close()

# tương quan với target, sắp xếp
ct=C["IS_LARGE_FIRE"].drop("IS_LARGE_FIRE").sort_values()
fig,ax=plt.subplots(figsize=(8,6))
ax.barh(ct.index,ct.values,color=["#d73027" if v>0 else "#4575b4" for v in ct.values])
ax.axvline(0,c="gray",lw=.8); ax.set_title("Tương quan với IS_LARGE_FIRE (cháy lớn)")
ax.set_xlabel("Pearson r"); fig.tight_layout(); fig.savefig(f"{FIG}/corr_target.png"); plt.close()

# ---------- 2) dự đoán trên test 2018-2020 ----------
te=df[(df.DATA_SOURCE.isin(["FPA_FOD_1992_2015","FPA_FOD_6TH_2016_2020"]))&
      (df.FIRE_YEAR>=2018)&(df.FIRE_YEAR<=2020)].copy()
for c in CAT: te[c]=te[c].astype("category")
y=te["IS_LARGE_FIRE"].astype(int).values
p=clf.predict_proba(te[FEATS])[:,1]

# threshold curve
prec,rec,thr=precision_recall_curve(y,p)
f1=2*prec*rec/(prec+rec+1e-9)
fig,ax=plt.subplots(figsize=(9,5))
ax.plot(thr,prec[:-1],label="Precision",color="#1b9e77")
ax.plot(thr,rec[:-1],label="Recall",color="#d95f02")
ax.plot(thr,f1[:-1],label="F1",color="#7570b3")
ax.axvline(t_f1,ls="--",c="gray",label=f"best-F1 thr={t_f1:.2f}")
ax.set_xlabel("Ngưỡng xác suất"); ax.set_ylabel("Chỉ số"); ax.set_title("Precision / Recall / F1 theo ngưỡng (test)")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{FIG}/model_threshold_curve.png"); plt.close()

# confusion tại best-F1 (chuẩn hoá theo hàng)
cm=confusion_matrix(y,(p>=t_f1).astype(int))
cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(5.5,5))
im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j,i,f"{cm[i,j]:,}\n({cmn[i,j]*100:.1f}%)",ha="center",va="center",
                color="white" if cmn[i,j]>0.5 else "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["dự đoán nhỏ","dự đoán lớn"])
ax.set_yticks([0,1]); ax.set_yticklabels(["thực nhỏ","thực lớn"])
ax.set_title(f"Ma trận nhầm lẫn @ best-F1 ({t_f1:.2f})"); fig.colorbar(im,shrink=.8)
fig.tight_layout(); fig.savefig(f"{FIG}/model_confusion.png"); plt.close()

# calibration
frac,mean_pred=calibration_curve(y,p,n_bins=10,strategy="quantile")
fig,ax=plt.subplots(figsize=(6,6))
ax.plot([0,1],[0,1],"--",c="gray",label="hoàn hảo")
ax.plot(mean_pred,frac,"o-",color="#e6550d",label="HGB")
ax.set_xlabel("Xác suất dự đoán TB"); ax.set_ylabel("Tỉ lệ cháy lớn thực tế")
ax.set_title("Đường hiệu chuẩn (calibration, test)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/model_calibration.png"); plt.close()

print("Đã tạo:", *sorted(f for f in os.listdir(FIG) if f.startswith(("corr_","model_"))), sep="\n  ")
