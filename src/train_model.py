#!/usr/bin/env python3
"""
Model CẢI TIẾN dự đoán cháy lớn (IS_LARGE_FIRE) — chống leakage.
Dữ liệu: data/processed/fires_model.sqlite (model_data) — có feature lịch sử + owner.
- Chia thời gian: train<=2016, val=2017 (chọn siêu tham số), test 2018-2020, OOD NIFC 2021-26.
- Tinh chỉnh HistGradientBoosting (grid nhỏ theo val PR-AUC).
- So sánh: LogisticRegression (baseline) vs HGB tuned.
- Tối ưu ngưỡng (best-F1 & recall>=0.80), feature importance, ROC/PR.
Output: models/large_fire_hgb.joblib, reports/figures/model_*.png, reports/model_report.txt
"""
import sqlite3, os, numpy as np, pandas as pd, joblib, itertools
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
DB=os.path.join(ROOT,"data","processed","fires_model.sqlite")
FIG=os.path.join(ROOT,"reports","figures"); MODELS=os.path.join(ROOT,"models")
os.makedirs(FIG,exist_ok=True); os.makedirs(MODELS,exist_ok=True)
rep=[]
def log(*a):
    s=" ".join(str(x) for x in a); print(s); rep.append(s)

NUM=["LATITUDE","LONGITUDE","DOY_SIN","DOY_COS","MONTH_SIN","MONTH_COS","DAY_OF_WEEK",
     "IS_WEEKEND","FIRE_YEAR","hist_cell_logn","hist_cell_rate","hist_state_logn","hist_state_rate",
     "hist_cellmonth_rate","dist_largefire_log"]
CAT=["STATE","REGION","STAT_CAUSE_DESCR","OWNER_DESCR"]
FEATS=NUM+CAT; TARGET="IS_LARGE_FIRE"

log(">> đọc model_data ...")
con=sqlite3.connect(DB); df=pd.read_sql("SELECT * FROM model_data",con); con.close()
for c in CAT: df[c]=df[c].astype("category")
for c in NUM: df[c]=pd.to_numeric(df[c],errors="coerce")
df[TARGET]=df[TARGET].astype(int)
fpa=df[df.DATA_SOURCE.isin(["FPA_FOD_1992_2015","FPA_FOD_6TH_2016_2020"])]
tr =fpa[fpa.FIRE_YEAR<=2016]; va=fpa[fpa.FIRE_YEAR==2017]
trv=fpa[fpa.FIRE_YEAR<=2017]; te=fpa[(fpa.FIRE_YEAR>=2018)&(fpa.FIRE_YEAR<=2020)]
ood=df[df.DATA_SOURCE=="NIFC_WFIGS_2021_2026"]
log(f"   train<=2016:{len(tr):,} val2017:{len(va):,} test18-20:{len(te):,} OOD:{len(ood):,}")
log(f"   base rate cháy lớn -> test {te[TARGET].mean()*100:.2f}% | OOD {ood[TARGET].mean()*100:.2f}%")

def hgb(**kw):
    return HistGradientBoostingClassifier(categorical_features="from_dtype",
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=42, max_iter=400, **kw)

log("\n>> tinh chỉnh HGB (grid theo val 2017 PR-AUC) ...")
grid=[{"learning_rate":lr,"max_leaf_nodes":ml,"min_samples_leaf":50,"l2_regularization":1.0}
      for lr,ml in itertools.product([0.1,0.2],[63,127])]
best=None
for p in grid:
    m=hgb(**p); m.fit(tr[FEATS],tr[TARGET])
    ap=average_precision_score(va[TARGET], m.predict_proba(va[FEATS])[:,1])
    log(f"   {p} -> val PR-AUC={ap:.4f}")
    if best is None or ap>best[0]: best=(ap,p)
log(f"   >>> tốt nhất: {best[1]} (val PR-AUC={best[0]:.4f})")
clf=hgb(**best[1]); clf.fit(trv[FEATS],trv[TARGET])

log("\n>> LogisticRegression baseline ...")
pre=ColumnTransformer([("num",StandardScaler(),NUM),
                       ("cat",OneHotEncoder(handle_unknown="ignore",min_frequency=50),CAT)])
lr=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=300,class_weight="balanced",n_jobs=-1))])
lr.fit(trv[FEATS],trv[TARGET])

def metrics(name,model,X,y):
    p=model.predict_proba(X)[:,1]
    roc=roc_auc_score(y,p); ap=average_precision_score(y,p)
    log(f"   {name:<34} ROC-AUC={roc:.4f}  PR-AUC={ap:.4f}  (base {y.mean()*100:.2f}%)")
    return p,roc,ap

log("\n===== SO SÁNH TRÊN TEST 2018-2020 =====")
metrics("LogReg",lr,te[FEATS],te[TARGET])
pte,roc,ap=metrics("HGB tuned (model chọn)",clf,te[FEATS],te[TARGET])
log("\n===== TRÊN OOD NIFC 2021-2026 (cảnh báo khác độ phủ) =====")
metrics("LogReg",lr,ood[FEATS],ood[TARGET]); metrics("HGB tuned",clf,ood[FEATS],ood[TARGET])

log("\n>> tối ưu ngưỡng (HGB, test) ...")
prec,rec,thr=precision_recall_curve(te[TARGET],pte)
f1=2*prec*rec/(prec+rec+1e-9); bi=int(np.nanargmax(f1[:-1])); t_f1=thr[bi]
ok=np.where(rec[:-1]>=0.80)[0]; t_r80=thr[ok[-1]] if len(ok) else 0.5
for name,t in [("mặc định 0.50",0.5),(f"best-F1={t_f1:.3f}",t_f1),(f"recall>=0.80 thr={t_r80:.3f}",t_r80)]:
    yhat=(pte>=t).astype(int); cm=confusion_matrix(te[TARGET],yhat)
    log(f"\n  --- ngưỡng {name} ---")
    log(f"  confusion [[TN FP][FN TP]]: {cm.tolist()}")
    log("  "+classification_report(te[TARGET],yhat,digits=3,target_names=["nhỏ","lớn"]).replace("\n","\n  "))

fig,ax=plt.subplots(1,2,figsize=(12,5))
fpr,tpr,_=roc_curve(te[TARGET],pte); ax[0].plot(fpr,tpr,label=f"HGB AUC={roc:.3f}")
ax[0].plot([0,1],[0,1],"--",c="gray"); ax[0].set_title("ROC (test 2018-2020)")
ax[0].set_xlabel("FPR");ax[0].set_ylabel("TPR");ax[0].legend()
ax[1].plot(rec,prec,label=f"HGB PR-AUC={ap:.3f}")
ax[1].axhline(te[TARGET].mean(),ls="--",c="gray",label=f"base={te[TARGET].mean():.3f}")
ax[1].set_title("Precision-Recall (test)");ax[1].set_xlabel("Recall");ax[1].set_ylabel("Precision");ax[1].legend()
fig.tight_layout(); fig.savefig(f"{FIG}/model_roc_pr.png"); plt.close()

log("\n>> permutation importance (mẫu 50k test) ...")
samp=te.sample(min(50000,len(te)),random_state=0)
pi=permutation_importance(clf,samp[FEATS],samp[TARGET],n_repeats=5,random_state=0,
                          scoring="average_precision",n_jobs=-1)
imp=pd.Series(pi.importances_mean,index=FEATS).sort_values()
fig,ax=plt.subplots(figsize=(8,6)); ax.barh(imp.index,imp.values,color="#3182bd")
ax.set_title("Độ quan trọng feature (permutation ΔPR-AUC)"); fig.tight_layout()
fig.savefig(f"{FIG}/model_feature_importance.png"); plt.close()
log("  "+imp.sort_values(ascending=False).round(4).to_string().replace("\n","\n  "))

joblib.dump({"model":clf,"features":FEATS,"num":NUM,"cat":CAT,
             "thresholds":{"f1":float(t_f1),"recall80":float(t_r80)}},
            f"{MODELS}/large_fire_hgb.joblib")
log(f"\n>> đã lưu models/large_fire_hgb.joblib")
log(f">> SO VỚI BASELINE v1 (PR-AUC test 0.170): HGB mới = {ap:.3f}")
open(os.path.join(ROOT,"reports","model_report.txt"),"w").write("\n".join(rep))
print("DONE.")
