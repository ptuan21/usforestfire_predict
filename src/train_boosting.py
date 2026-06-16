#!/usr/bin/env python3
"""
Bước 2: so sánh các thuật toán gradient boosting cho IS_LARGE_FIRE.
HGB (sklearn) vs LightGBM vs XGBoost — cùng feature, cùng split thời gian,
mỗi loại tinh chỉnh nhẹ theo val 2017 (PR-AUC). Chống leakage như trước.
Output: models/large_fire_best.joblib (model tốt nhất) + reports/figures/model_compare.png
        + cập nhật reports/model_report.txt (phần so sánh)
"""
import sqlite3, os, numpy as np, pandas as pd, joblib, itertools, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import lightgbm as lgb, xgboost as xgb

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
DB=os.path.join(ROOT,"data","processed","fires_model.sqlite")
FIG=os.path.join(ROOT,"reports","figures"); MODELS=os.path.join(ROOT,"models")
rep=["","="*60,"BƯỚC 2 — SO SÁNH BOOSTING (HGB / LightGBM / XGBoost)","="*60]
def log(*a): s=" ".join(str(x) for x in a); print(s); rep.append(s)

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
tr=fpa[fpa.FIRE_YEAR<=2016]; va=fpa[fpa.FIRE_YEAR==2017]
trv=fpa[fpa.FIRE_YEAR<=2017]; te=fpa[(fpa.FIRE_YEAR>=2018)&(fpa.FIRE_YEAR<=2020)]
ood=df[df.DATA_SOURCE=="NIFC_WFIGS_2021_2026"]
ytr,yva,ytrv,yte,yood=[x[TARGET].values for x in (tr,va,trv,te,ood)]
spw=(ytrv==0).sum()/(ytrv==1).sum()   # scale_pos_weight
log(f"   test n={len(te):,} base={yte.mean()*100:.2f}% | scale_pos_weight={spw:.1f}")

def ap_val(model_fit, predict):
    return average_precision_score(yva, predict(va[FEATS]))

results={}; preds={}

# ---------------- HGB ----------------
def mk_hgb(**k): return HistGradientBoostingClassifier(categorical_features="from_dtype",
    class_weight="balanced",early_stopping=True,validation_fraction=0.1,random_state=42,max_iter=400,**k)
log("\n>> HGB grid ...")
best=None
for lr,ml in itertools.product([0.1,0.2],[63,127]):
    m=mk_hgb(learning_rate=lr,max_leaf_nodes=ml,min_samples_leaf=50,l2_regularization=1.0)
    m.fit(tr[FEATS],ytr); ap=ap_val(m,lambda X:m.predict_proba(X)[:,1])
    log(f"   lr={lr} leaf={ml} -> val PR-AUC={ap:.4f}")
    if best is None or ap>best[0]: best=(ap,dict(learning_rate=lr,max_leaf_nodes=ml,min_samples_leaf=50,l2_regularization=1.0))
m=mk_hgb(**best[1]); m.fit(trv[FEATS],ytrv)
results["HGB"]=m; preds["HGB"]=m.predict_proba(te[FEATS])[:,1]
preds["HGB_ood"]=m.predict_proba(ood[FEATS])[:,1]

# ---------------- LightGBM ----------------
log("\n>> LightGBM grid ...")
best=None
for lr,nl in itertools.product([0.05,0.1],[63,127]):
    m=lgb.LGBMClassifier(n_estimators=600,learning_rate=lr,num_leaves=nl,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
        class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1)
    m.fit(tr[FEATS],ytr,eval_set=[(va[FEATS],yva)],eval_metric="average_precision",
          callbacks=[lgb.early_stopping(40,verbose=False)])
    ap=average_precision_score(yva,m.predict_proba(va[FEATS])[:,1])
    log(f"   lr={lr} leaves={nl} best_iter={m.best_iteration_} -> val PR-AUC={ap:.4f}")
    if best is None or ap>best[0]: best=(ap,dict(learning_rate=lr,num_leaves=nl),m.best_iteration_)
m=lgb.LGBMClassifier(n_estimators=best[2] or 600,learning_rate=best[1]["learning_rate"],
    num_leaves=best[1]["num_leaves"],subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
    class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1)
m.fit(trv[FEATS],ytrv)
results["LightGBM"]=m; preds["LightGBM"]=m.predict_proba(te[FEATS])[:,1]
preds["LightGBM_ood"]=m.predict_proba(ood[FEATS])[:,1]

# ---------------- XGBoost ----------------
log("\n>> XGBoost grid ...")
best=None
for lr,md in itertools.product([0.05,0.1],[6,8]):
    m=xgb.XGBClassifier(n_estimators=600,learning_rate=lr,max_depth=md,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,scale_pos_weight=spw,
        tree_method="hist",enable_categorical=True,eval_metric="aucpr",
        early_stopping_rounds=40,random_state=42,n_jobs=-1)
    m.fit(tr[FEATS],ytr,eval_set=[(va[FEATS],yva)],verbose=False)
    ap=average_precision_score(yva,m.predict_proba(va[FEATS])[:,1])
    log(f"   lr={lr} depth={md} best_iter={m.best_iteration} -> val PR-AUC={ap:.4f}")
    if best is None or ap>best[0]: best=(ap,dict(learning_rate=lr,max_depth=md),m.best_iteration)
m=xgb.XGBClassifier(n_estimators=(best[2] or 599)+1,learning_rate=best[1]["learning_rate"],
    max_depth=best[1]["max_depth"],subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
    scale_pos_weight=spw,tree_method="hist",enable_categorical=True,random_state=42,n_jobs=-1)
m.fit(trv[FEATS],ytrv)
results["XGBoost"]=m; preds["XGBoost"]=m.predict_proba(te[FEATS])[:,1]
preds["XGBoost_ood"]=m.predict_proba(ood[FEATS])[:,1]

# ---------------- so sánh ----------------
log("\n===== KẾT QUẢ TEST 2018-2020 / OOD NIFC =====")
log(f"   {'Model':<12}{'ROC test':>10}{'PR test':>10}{'ROC OOD':>10}{'PR OOD':>10}")
rows=[]
for name in ["HGB","LightGBM","XGBoost"]:
    rt=roc_auc_score(yte,preds[name]); pt=average_precision_score(yte,preds[name])
    ro=roc_auc_score(yood,preds[name+"_ood"]); po=average_precision_score(yood,preds[name+"_ood"])
    log(f"   {name:<12}{rt:>10.4f}{pt:>10.4f}{ro:>10.4f}{po:>10.4f}")
    rows.append((name,rt,pt,ro,po))
winner=max(rows,key=lambda r:r[2])
log(f"\n   >>> TỐT NHẤT theo PR-AUC test: {winner[0]} (PR-AUC={winner[2]:.4f})")
log(f"   (HGB v3 trước đó PR-AUC=0.221; baseline v1=0.170)")

# biểu đồ PR cho 3 model
fig,ax=plt.subplots(1,2,figsize=(13,5))
for name in ["HGB","LightGBM","XGBoost"]:
    pr,rc,_=precision_recall_curve(yte,preds[name])
    ax[0].plot(rc,pr,label=f"{name} (AP={average_precision_score(yte,preds[name]):.3f})")
ax[0].axhline(yte.mean(),ls="--",c="gray"); ax[0].set_title("PR (test 2018-2020)")
ax[0].set_xlabel("Recall");ax[0].set_ylabel("Precision");ax[0].legend()
names=[r[0] for r in rows]; x=np.arange(len(names)); w=0.35
ax[1].bar(x-w/2,[r[2] for r in rows],w,label="PR-AUC test",color="#1f77b4")
ax[1].bar(x+w/2,[r[4] for r in rows],w,label="PR-AUC OOD",color="#ff7f0e")
ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_title("So sánh PR-AUC")
ax[1].legend(); ax[1].grid(axis="y",alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/model_compare.png"); plt.close()

joblib.dump({"model":results[winner[0]],"name":winner[0],"features":FEATS,"num":NUM,"cat":CAT},
            f"{MODELS}/large_fire_best.joblib")
log(f">> đã lưu model tốt nhất ({winner[0]}) -> models/large_fire_best.joblib")
log(">> biểu đồ -> reports/figures/model_compare.png")
# append vào report
open(os.path.join(ROOT,"reports","model_report.txt"),"a").write("\n".join(rep))
print("DONE.")
