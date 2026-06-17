#!/usr/bin/env python3
"""
Dashboard trực quan hoá dự án US Wildfire Prediction.
Chạy:  streamlit run src/dashboard.py
Dữ liệu nhẹ đã tính sẵn ở data/processed/dashboard/ (chạy precompute_dashboard_data.py trước).
"""
import os, sqlite3, pandas as pd, numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
DASH=os.path.join(ROOT,"data","processed","dashboard")
FIG=os.path.join(ROOT,"reports","figures")

st.set_page_config(page_title="US Wildfire Prediction", layout="wide", page_icon="🔥")

@st.cache_data
def load_csv(n): return pd.read_csv(os.path.join(DASH,n))
@st.cache_data
def load_risk():
    con=sqlite3.connect(os.path.join(DASH,"ca_risk.sqlite"))
    d=pd.read_sql("SELECT * FROM risk",con); con.close()
    d["week"]=pd.to_datetime(d["week"]); return d

st.sidebar.title("🔥 US Wildfire Prediction")
st.sidebar.caption("Dữ liệu FPA FOD + NIFC (1992–2026) · gridMET · ML")
page=st.sidebar.radio("Chọn trang",
    ["🗺️ Bản đồ rủi ro cháy (California)","📊 Tổng quan toàn quốc","🤖 Hiệu năng mô hình"])
st.sidebar.markdown("---")
st.sidebar.caption("Mã nguồn: github.com/ptuan21/usforestfire_predict")

# ============================ TRANG 1: BẢN ĐỒ RỦI RO ============================
if page.startswith("🗺️"):
    st.title("Bản đồ rủi ro phát cháy — California")
    st.caption("Model LightGBM dự báo xác suất một ô lưới (~0.25°) có cháy trong tuần, "
               "dùng gridMET (ERC, độ ẩm nhiên liệu, gió, nhiệt độ…) + địa hình + lịch sử cháy.")
    risk=load_risk()
    weeks=sorted(risk["week"].unique())
    wk_counts=risk.groupby("week")["y"].sum()
    default=wk_counts.idxmax()   # tuần nhiều cháy nhất

    c1,c2=st.columns([3,1])
    with c2:
        yr=st.selectbox("Năm",sorted(risk["week"].dt.year.unique()),
                        index=len(risk["week"].dt.year.unique())-1)
        wks=[w for w in weeks if pd.Timestamp(w).year==yr]
        di=wks.index(default) if default in wks else len(wks)//2
        wk=st.select_slider("Tuần",options=wks,value=wks[di],
                            format_func=lambda x:pd.Timestamp(x).strftime("%d/%m/%Y"))
        thr=st.slider("Ngưỡng cảnh báo",0.0,1.0,0.5,0.05)
    d=risk[risk["week"]==wk].copy()
    fires=d[d["y"]==1]
    with c1:
        fig=px.scatter_geo(d,lat="lat",lon="lon",color="risk",
            color_continuous_scale="YlOrRd",range_color=(0,1),scope="usa",
            hover_data={"risk":":.2f","erc":":.0f","fm1000":":.1f"})
        fig.update_traces(marker=dict(size=7,symbol="square"))
        if len(fires):
            fig.add_trace(go.Scattergeo(lat=fires["lat"],lon=fires["lon"],mode="markers",
                marker=dict(size=7,color="rgba(0,0,255,0)",line=dict(color="blue",width=1.5)),
                name="cháy thực tế"))
        fig.update_geos(fitbounds="locations",showland=True,landcolor="#eee",
                        showsubunits=True,subunitcolor="#999")
        fig.update_layout(height=560,margin=dict(l=0,r=0,t=10,b=0),
                          coloraxis_colorbar=dict(title="Rủi ro"))
        st.plotly_chart(fig,use_container_width=True)

    # chỉ số tuần
    d2=d.sort_values("risk",ascending=False)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Số ô có cháy thực tế",int(d["y"].sum()))
    m2.metric("Rủi ro trung bình",f"{d['risk'].mean():.2f}")
    flagged=(d["risk"]>=thr).sum()
    recall=d[d["risk"]>=thr]["y"].sum()/max(1,d["y"].sum())
    m3.metric(f"Ô cảnh báo (≥{thr:.2f})",int(flagged))
    m4.metric("Bắt được cháy (recall)",f"{recall*100:.0f}%")
    top=d2.head(20)["y"].mean()
    st.info(f"🎯 Trong 20 ô rủi ro cao nhất tuần này, **{top*100:.0f}%** thực sự có cháy.")

# ============================ TRANG 2: TỔNG QUAN ============================
elif page.startswith("📊"):
    st.title("Tổng quan cháy rừng Mỹ (1992–2026)")
    by_year=load_csv("by_year.csv")
    tot=by_year.groupby("year")["n"].sum().reset_index()
    st.caption(f"Tổng {tot['n'].sum():,.0f} vụ cháy · nguồn FPA FOD (1992–2020) + NIFC (2021–2026)")

    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(by_year,x="year",y="n",color="source",title="Số vụ cháy theo năm (theo nguồn)",
                   labels={"n":"Số vụ","year":"Năm","source":"Nguồn"})
        fig.add_vline(x=2020.5,line_dash="dash",line_color="red")
        st.plotly_chart(fig,use_container_width=True)
        cause=load_csv("by_cause.csv").sort_values("n")
        fig=px.bar(cause,x="n",y="cause",orientation="h",title="Nguyên nhân cháy",
                   labels={"n":"Số vụ","cause":""})
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        acres=by_year.groupby("year")["acres"].sum().reset_index()
        fig=px.area(acres,x="year",y="acres",title="Tổng diện tích cháy theo năm (acres)",
                    labels={"acres":"Acres","year":"Năm"})
        st.plotly_chart(fig,use_container_width=True)
        mon=load_csv("by_month.csv")
        fig=px.bar(mon,x="month",y="n",title="Phân bố theo tháng (mùa cháy)",
                   labels={"n":"Số vụ","month":"Tháng"})
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Bản đồ số vụ cháy theo bang")
    by_state=load_csv("by_state.csv")
    fig=px.choropleth(by_state,locations="STATE",locationmode="USA-states",color="n",
        scope="usa",color_continuous_scale="OrRd",labels={"n":"Số vụ"})
    st.plotly_chart(fig,use_container_width=True)

# ============================ TRANG 3: MÔ HÌNH ============================
else:
    st.title("Hiệu năng các mô hình")
    met=load_csv("model_metrics.csv")
    st.dataframe(met,use_container_width=True,hide_index=True)
    st.markdown("Model **dự báo phát cháy** (Tầng 1) mạnh nhất nhờ feature nhân quả từ gridMET.")
    figs=[("risk_map.png","Bản đồ rủi ro cháy California (tuần cao điểm vs mùa đông)"),
          ("occ_roc_pr.png","ROC & Precision-Recall — dự báo phát cháy"),
          ("occ_feature_importance.png","Feature quan trọng — dự báo phát cháy"),
          ("model_compare.png","So sánh thuật toán — model cháy lớn"),
          ("cause_confusion.png","Nhầm lẫn nguyên nhân (12 lớp)")]
    for f,cap in figs:
        p=os.path.join(FIG,f)
        if os.path.exists(p): st.image(p,caption=cap,use_container_width=True)
