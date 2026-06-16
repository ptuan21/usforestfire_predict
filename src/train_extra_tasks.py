#!/usr/bin/env python3
"""
Bước 3 (THÊM, không thay thế): 2 bài toán mới, dùng lại feature đã có.
  A) Hồi quy diện tích cháy: target = log1p(FIRE_SIZE acres)   -> LightGBM Regressor
  B) Phân loại NGUYÊN NHÂN cháy (12 lớp, bỏ Missing/Undefined)  -> LightGBM multiclass
Cùng split thời gian: train<=2017, test 2018-2020 (FPA FOD). Chống leakage.
Output: models/fire_size_lgbm.joblib, models/fire_cause_lgbm.joblib
        reports/figures/size_*.png, cause_confusion.png ; nối reports/model_report.txt
"""
import sqlite3, os, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report)

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
PROC=os.path.join(ROOT,"data","processed")
FIG=os.path.join(ROOT,"reports","figures"); MODELS=os.path.join(ROOT,"models")
rep=["","="*60,"BƯỚC 3 — THÊM BÀI TOÁN: hồi quy diện tích & phân loại nguyên nhân","="*60]
def log(*a): s=" ".join(str(x) for x in a); print(s); rep.append(s)

CAT=["STATE","REGION","OWNER_DESCR"]   # STAT_CAUSE_DESCR xử lý riêng từng bài
HIST=["hist_cell_logn","hist_cell_rate","hist_state_logn","hist_state_rate",
      "hist_cellmonth_rate","dist_largefire_log"]
GEO_TIME=["LATITUDE","LONGITUDE","DOY_SIN","DOY_COS","MONTH_SIN","MONTH_COS",
          "DAY_OF_WEEK","IS_WEEKEND","FIRE_YEAR"]

log(">> đọc model_data + FIRE_SIZE ...")
con=sqlite3.connect(os.path.join(PROC,"fires_model.sqlite"))
df=pd.read_sql("SELECT * FROM model_data",con); con.close()
con=sqlite3.connect(os.path.join(PROC,"fires_ml.sqlite"))
sz=pd.read_sql("SELECT ROW_ID, FIRE_SIZE FROM fires_clean",con); con.close()
df=df.merge(sz,on="ROW_ID",how="left")
df=df[df.DATA_SOURCE.isin(["FPA_FOD_1992_2015","FPA_FOD_6TH_2016_2020"])].copy()
for c in CAT+["STAT_CAUSE_DESCR"]: df[c]=df[c].astype("category")
is_tr=df.FIRE_YEAR<=2017; is_te=(df.FIRE_YEAR>=2018)&(df.FIRE_YEAR<=2020)

# ========================= A) HỒI QUY DIỆN TÍCH =========================
log("\n########## A) HỒI QUY log(diện tích) ##########")
# cause là feature hợp lệ ở đây (biết khi điều tra); size là target
FA=GEO_TIME+HIST+CAT+["STAT_CAUSE_DESCR"]
d=df[df.FIRE_SIZE>0].copy(); d["y"]=np.log1p(d["FIRE_SIZE"])
tr=d[d.FIRE_YEAR<=2017]; te=d[(d.FIRE_YEAR>=2018)&(d.FIRE_YEAR<=2020)]
log(f"   train {len(tr):,} | test {len(te):,}")
reg=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.05,num_leaves=127,
    subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=42,n_jobs=-1,verbose=-1)
reg.fit(tr[FA],tr["y"],eval_set=[(te[FA],te["y"])],eval_metric="rmse",
        callbacks=[lgb.early_stopping(40,verbose=False)])
pred=reg.predict(te[FA])
rmse=mean_squared_error(te["y"],pred)**0.5; mae=mean_absolute_error(te["y"],pred); r2=r2_score(te["y"],pred)
# back-transform: sai số tuyệt đối trung vị (acres)
ape=np.abs(np.expm1(pred)-te["FIRE_SIZE"]);
log(f"   [log-space] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
log(f"   [acres] median |error| = {np.median(ape):.2f} acres (diện tích TB test={te['FIRE_SIZE'].mean():.1f})")
fi=pd.Series(reg.feature_importances_,index=FA).sort_values(ascending=False)
log("   top feature:\n   "+fi.head(8).to_string().replace("\n","\n   "))

# biểu đồ: dự đoán vs thực (log), hexbin
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].hexbin(te["y"],pred,gridsize=50,cmap="inferno",mincnt=1,bins="log")
lim=[0,te["y"].max()]; ax[0].plot(lim,lim,"--",c="cyan")
ax[0].set_xlabel("log(diện tích) thực"); ax[0].set_ylabel("dự đoán")
ax[0].set_title(f"Hồi quy diện tích (R²={r2:.3f})")
# RMSE theo lớp kích thước thực
te=te.assign(pred=pred,err=pred-te["y"])
bins=pd.cut(te["FIRE_SIZE"],[0,0.25,10,100,300,1000,5000,1e9],
            labels=["A","B","C","D","E","F","G"])
mae_by=te.groupby(bins,observed=True)["err"].apply(lambda x:np.abs(x).mean())
ax[1].bar(mae_by.index.astype(str),mae_by.values,color="#cc4c02")
ax[1].set_title("MAE (log-space) theo lớp kích thước thực"); ax[1].set_xlabel("Lớp"); ax[1].set_ylabel("MAE log")
fig.tight_layout(); fig.savefig(f"{FIG}/size_regression.png"); plt.close()
joblib.dump({"model":reg,"features":FA,"target":"log1p(FIRE_SIZE)"},f"{MODELS}/fire_size_lgbm.joblib")
log("   >> lưu models/fire_size_lgbm.joblib, biểu đồ size_regression.png")

# ========================= B) PHÂN LOẠI NGUYÊN NHÂN =========================
log("\n########## B) PHÂN LOẠI NGUYÊN NHÂN (12 lớp) ##########")
FC=GEO_TIME+HIST+CAT+["FIRE_SIZE"]   # size là feature; cause là target
d=df[df.STAT_CAUSE_DESCR!="Missing/Undefined"].copy()
d["STAT_CAUSE_DESCR"]=d["STAT_CAUSE_DESCR"].cat.remove_unused_categories()
classes=list(d["STAT_CAUSE_DESCR"].cat.categories)
ycodes=d["STAT_CAUSE_DESCR"].cat.codes
tr=d[d.FIRE_YEAR<=2017]; te=d[(d.FIRE_YEAR>=2018)&(d.FIRE_YEAR<=2020)]
ytr=tr["STAT_CAUSE_DESCR"].cat.codes; yte=te["STAT_CAUSE_DESCR"].cat.codes
log(f"   train {len(tr):,} | test {len(te):,} | {len(classes)} lớp")
clf=lgb.LGBMClassifier(objective="multiclass",num_class=len(classes),n_estimators=500,
    learning_rate=0.05,num_leaves=127,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
    class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1)
clf.fit(tr[FC],ytr)
pp=clf.predict(te[FC])
labs=list(range(len(classes)))
acc=accuracy_score(yte,pp); f1m=f1_score(yte,pp,average="macro",labels=labs)
# baseline đoán theo lớp phổ biến nhất
maj=ytr.value_counts().idxmax(); acc_base=(yte==maj).mean()
log(f"   accuracy={acc:.3f} (baseline đoán lớp phổ biến={acc_base:.3f}) | macro-F1={f1m:.3f}")
log("   "+classification_report(yte,pp,labels=labs,target_names=classes,digits=3,zero_division=0).replace("\n","\n   "))

cm=confusion_matrix(yte,pp,labels=labs,normalize="true")
fig,ax=plt.subplots(figsize=(10,8.5))
im=ax.imshow(cm,cmap="viridis",vmin=0,vmax=1)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes,rotation=90,fontsize=8)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes,fontsize=8)
for i in range(len(classes)):
    for j in range(len(classes)):
        if cm[i,j]>=0.1: ax.text(j,i,f"{cm[i,j]*100:.0f}",ha="center",va="center",
                                 fontsize=7,color="white" if cm[i,j]<0.6 else "black")
ax.set_xlabel("dự đoán"); ax.set_ylabel("thực tế")
ax.set_title("Ma trận nhầm lẫn nguyên nhân (chuẩn hoá theo hàng, %)"); fig.colorbar(im,shrink=.8)
fig.tight_layout(); fig.savefig(f"{FIG}/cause_confusion.png"); plt.close()
joblib.dump({"model":clf,"features":FC,"classes":classes},f"{MODELS}/fire_cause_lgbm.joblib")
log("   >> lưu models/fire_cause_lgbm.joblib, biểu đồ cause_confusion.png")

open(os.path.join(ROOT,"reports","model_report.txt"),"a").write("\n".join(rep))
print("DONE.")
