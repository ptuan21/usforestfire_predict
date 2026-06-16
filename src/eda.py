#!/usr/bin/env python3
"""EDA cho dữ liệu cháy rừng đã làm sạch (fires_ml.sqlite). Xuất PNG vào eda/."""
import sqlite3, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB   = os.path.join(ROOT, "data", "processed", "fires_ml.sqlite")
OUT  = os.path.join(ROOT, "reports", "figures"); os.makedirs(OUT, exist_ok=True)
con  = sqlite3.connect(DB)
plt.rcParams.update({"figure.dpi":110, "axes.grid":True, "grid.alpha":.3})

SRC_COLOR = {"FPA_FOD_1992_2015":"#2c7fb8","FPA_FOD_6TH_2016_2020":"#41ab5d","NIFC_WFIGS_2021_2026":"#e6550d"}

def q(sql): return pd.read_sql(sql, con)

# 1) Số cháy theo năm, tách nguồn
d = q("SELECT FIRE_YEAR, DATA_SOURCE, COUNT(*) n FROM fires_clean GROUP BY FIRE_YEAR, DATA_SOURCE")
piv = d.pivot(index="FIRE_YEAR", columns="DATA_SOURCE", values="n").fillna(0)
fig, ax = plt.subplots(figsize=(11,5))
bottom = np.zeros(len(piv))
for s in ["FPA_FOD_1992_2015","FPA_FOD_6TH_2016_2020","NIFC_WFIGS_2021_2026"]:
    if s in piv: ax.bar(piv.index, piv[s], bottom=bottom, label=s, color=SRC_COLOR[s]); bottom+=piv[s].values
ax.axvline(2020.5, ls="--", c="red", alpha=.6); ax.text(2020.6,ax.get_ylim()[1]*.8,"đứt gãy\nđộ phủ",color="red",fontsize=9)
ax.set_title("Số vụ cháy rừng Mỹ theo năm (1992–2026)"); ax.set_xlabel("Năm"); ax.set_ylabel("Số vụ"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/01_fires_by_year.png"); plt.close()

# 2) Tổng diện tích cháy (acres) theo năm
d = q("SELECT FIRE_YEAR, SUM(FIRE_SIZE) acres FROM fires_clean GROUP BY FIRE_YEAR")
fig, ax = plt.subplots(figsize=(11,4.5))
ax.fill_between(d.FIRE_YEAR, d.acres/1e6, color="#cc4c02", alpha=.8)
ax.set_title("Tổng diện tích cháy theo năm (triệu acres)"); ax.set_xlabel("Năm"); ax.set_ylabel("Triệu acres")
fig.tight_layout(); fig.savefig(f"{OUT}/02_acres_by_year.png"); plt.close()

# 3) Mùa cháy — số cháy theo tháng
d = q("SELECT MONTH, COUNT(*) n FROM fires_clean WHERE MONTH IS NOT NULL GROUP BY MONTH")
fig, ax = plt.subplots(figsize=(9,4.5))
ax.bar(d.MONTH, d.n, color="#fe9929")
ax.set_xticks(range(1,13)); ax.set_title("Phân bố cháy theo tháng (tính mùa cháy)")
ax.set_xlabel("Tháng"); ax.set_ylabel("Số vụ")
fig.tight_layout(); fig.savefig(f"{OUT}/03_fires_by_month.png"); plt.close()

# 4) Phân bố nguyên nhân
d = q("SELECT STAT_CAUSE_DESCR c, COUNT(*) n FROM fires_clean GROUP BY c ORDER BY n DESC")
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(d.c[::-1], d.n[::-1], color="#756bb1")
ax.set_title("Phân bố nguyên nhân cháy (13 lớp đã hợp nhất)"); ax.set_xlabel("Số vụ")
fig.tight_layout(); fig.savefig(f"{OUT}/04_causes.png"); plt.close()

# 5) Top 15 bang
d = q("SELECT STATE, COUNT(*) n FROM fires_clean WHERE STATE_VALID=1 GROUP BY STATE ORDER BY n DESC LIMIT 15")
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(d.STATE[::-1], d.n[::-1], color="#238b45")
ax.set_title("Top 15 bang nhiều cháy nhất"); ax.set_xlabel("Số vụ")
fig.tight_layout(); fig.savefig(f"{OUT}/05_top_states.png"); plt.close()

# 6) Phân bố size class
d = q("SELECT FIRE_SIZE_CLASS c, COUNT(*) n FROM fires_clean WHERE c IS NOT NULL GROUP BY c ORDER BY c")
fig, ax = plt.subplots(figsize=(8,4.5))
ax.bar(d.c, d.n, color="#d94801"); ax.set_yscale("log")
ax.set_title("Phân bố lớp kích thước cháy (A→G, thang log)"); ax.set_xlabel("Lớp"); ax.set_ylabel("Số vụ (log)")
fig.tight_layout(); fig.savefig(f"{OUT}/06_size_class.png"); plt.close()

# 7) Heatmap năm × tháng
d = q("SELECT FIRE_YEAR, MONTH, COUNT(*) n FROM fires_clean WHERE MONTH IS NOT NULL GROUP BY FIRE_YEAR,MONTH")
H = d.pivot(index="MONTH", columns="FIRE_YEAR", values="n").fillna(0)
fig, ax = plt.subplots(figsize=(13,4.5))
im = ax.imshow(H, aspect="auto", cmap="YlOrRd", origin="lower",
               extent=[H.columns.min(), H.columns.max(), .5, 12.5])
ax.set_yticks(range(1,13)); ax.set_title("Cường độ cháy: Năm × Tháng (số vụ)")
ax.set_xlabel("Năm"); ax.set_ylabel("Tháng"); fig.colorbar(im, label="số vụ")
fig.tight_layout(); fig.savefig(f"{OUT}/07_year_month_heatmap.png"); plt.close()

# 8) Bản đồ điểm cháy (mẫu 80k, màu theo lớp kích thước)
d = q("SELECT LATITUDE,LONGITUDE,FIRE_SIZE_CLASS FROM fires_clean WHERE GEO_VALID=1 "
      "AND LONGITUDE BETWEEN -170 AND -65 ORDER BY ROW_ID LIMIT 80000")
big = d.FIRE_SIZE_CLASS.isin(["E","F","G"])
fig, ax = plt.subplots(figsize=(11,6))
ax.scatter(d.LONGITUDE[~big], d.LATITUDE[~big], s=1, c="#888", alpha=.25, label="nhỏ (A–D)")
ax.scatter(d.LONGITUDE[big], d.LATITUDE[big], s=4, c="red", alpha=.6, label="lớn (E–G)")
ax.set_title("Phân bố không gian điểm cháy (mẫu 80k)"); ax.set_xlabel("Kinh độ"); ax.set_ylabel("Vĩ độ"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/08_spatial_map.png"); plt.close()

con.close()
print("Đã tạo 8 biểu đồ trong:", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
