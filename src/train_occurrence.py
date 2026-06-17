#!/usr/bin/env python3
"""
TẦNG 1 — Model DỰ BÁO KHẢ NĂNG PHÁT CHÁY (California, panel ô-tuần 2015-2020).
Target y = tuần đó ô có cháy không. Feature: gridMET (ERC/BI/độ ẩm nhiên liệu/gió/
nhiệt/ẩm/mưa/VPD) + địa hình (elev) + lịch sử cháy + mùa (woy sin/cos) + vị trí.

Đánh giá HAI cách (đều trung thực):
  1) Temporal holdout: train 2015-2019, test 2020 (dự báo năm chưa thấy).
  2) Spatial block CV: chia California thành các khối không gian, CV 4 fold
     (kiểm tra tổng quát hoá sang VÙNG chưa thấy — quan trọng cho bản đồ rủi ro).
Output: models/fire_occurrence_lgbm.joblib, reports/figures/occ_*.png, reports/occurrence_report.txt
"""
import sqlite3, os, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed")
FIG=os.path.join(ROOT,"reports","figures"); MODELS=os.path.join(ROOT,"models")
rep=[];
def log(*a): s=" ".join(str(x) for x in a); print(s,flush=True); rep.append(s)

FEATS=["erc","bi","fm100","fm1000","vs","tmmx","rmin","pr","vpd",
       "elev","hist_fire_log","clim_cell_month","woy_sin","woy_cos","lat","lon"]
TARGET="y"

log(">> đọc panel ...")
con=sqlite3.connect(os.path.join(PROC,"ca_panel_feat.sqlite"))
df=pd.read_sql("SELECT * FROM panel",con); con.close()
df=df.dropna(subset=FEATS).copy()
log(f"   {len(df):,} ô-tuần | positives {df[TARGET].mean()*100:.2f}%")

def fit_eval(tr,te,label):
    spw=(tr[TARGET]==0).sum()/(tr[TARGET]==1).sum()
    m=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=63,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
        scale_pos_weight=spw,random_state=42,n_jobs=-1,verbose=-1)
    m.fit(tr[FEATS],tr[TARGET])
    p=m.predict_proba(te[FEATS])[:,1]
    roc=roc_auc_score(te[TARGET],p); ap=average_precision_score(te[TARGET],p)
    log(f"   {label:<26} ROC-AUC={roc:.4f}  PR-AUC={ap:.4f}  (base {te[TARGET].mean()*100:.2f}%)")
    return m,p,roc,ap

# ---------- 1) Temporal holdout ----------
log("\n===== 1) TEMPORAL: train 2015-2019, test 2020 =====")
tr=df[df.year<=2019]; te=df[df.year==2020]
m,p,roc,ap=fit_eval(tr,te,"temporal 2020")
lift=ap/te[TARGET].mean()
log(f"   PR-AUC gấp {lift:.1f}x base rate")

# ---------- 2) Spatial block CV ----------
log("\n===== 2) SPATIAL BLOCK CV (4 khối không gian) =====")
# chia 4 dải kinh độ (cân bằng số ô) -> mỗi fold giữ 1 dải, đảm bảo có cháy
df["blk"]=pd.qcut(df.lon,4,labels=False)
sroc=[];sap=[]
for k in range(4):
    tr=df[df.blk!=k]; te=df[df.blk==k]
    if te[TARGET].sum()==0:
        log(f"   giữ khối {k}: bỏ qua (không có cháy)"); continue
    _,_,r,a=fit_eval(tr,te,f"giữ khối {k} (n={len(te):,})")
    sroc.append(r);sap.append(a)
log(f"   TB spatial: ROC-AUC={np.mean(sroc):.4f}  PR-AUC={np.mean(sap):.4f}")

# ---------- model cuối (toàn dữ liệu) + feature importance ----------
log("\n>> train model cuối trên toàn panel ...")
spw=(df[TARGET]==0).sum()/(df[TARGET]==1).sum()
final=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,
    subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,scale_pos_weight=spw,
    random_state=42,n_jobs=-1,verbose=-1).fit(df[FEATS],df[TARGET])
imp=pd.Series(final.feature_importances_,index=FEATS).sort_values()
log("   feature importance:\n   "+imp.sort_values(ascending=False).to_string().replace("\n","\n   "))

# ---------- biểu đồ ----------
fig,ax=plt.subplots(1,2,figsize=(13,5))
fpr,tpr,_=roc_curve(te[TARGET],p) if False else (None,None,None)
# dùng lại dự đoán temporal cho ROC/PR
tr=df[df.year<=2019]; te=df[df.year==2020]; p=m.predict_proba(te[FEATS])[:,1]
fpr,tpr,_=roc_curve(te[TARGET],p); ax[0].plot(fpr,tpr,label=f"AUC={roc:.3f}")
ax[0].plot([0,1],[0,1],"--",c="gray"); ax[0].set_title("ROC — dự báo cháy tuần (test 2020)")
ax[0].set_xlabel("FPR");ax[0].set_ylabel("TPR");ax[0].legend()
pr,rc,_=precision_recall_curve(te[TARGET],p); ax[1].plot(rc,pr,label=f"PR-AUC={ap:.3f}")
ax[1].axhline(te[TARGET].mean(),ls="--",c="gray",label=f"base={te[TARGET].mean():.3f}")
ax[1].set_title("Precision-Recall (test 2020)");ax[1].set_xlabel("Recall");ax[1].set_ylabel("Precision");ax[1].legend()
fig.tight_layout(); fig.savefig(f"{FIG}/occ_roc_pr.png"); plt.close()

fig,ax=plt.subplots(figsize=(8,6)); ax.barh(imp.index,imp.values,color="#d94801")
ax.set_title("Độ quan trọng feature — dự báo phát cháy"); fig.tight_layout()
fig.savefig(f"{FIG}/occ_feature_importance.png"); plt.close()

joblib.dump({"model":final,"features":FEATS,
             "metrics":{"temporal_pr":ap,"temporal_roc":roc,
                        "spatial_pr":float(np.mean(sap)),"spatial_roc":float(np.mean(sroc))}},
            f"{MODELS}/fire_occurrence_lgbm.joblib")
log("\n>> đã lưu models/fire_occurrence_lgbm.joblib")
open(os.path.join(ROOT,"reports","occurrence_report.txt"),"w").write("\n".join(rep))
print("DONE.")
