#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard US Wildfire Prediction — Plotly Dash (giao diện sáng, chuyên nghiệp).
Chạy:  python3 src/dashboard_dash.py   ->  http://127.0.0.1:8050
Dữ liệu nhẹ đã tính sẵn ở data/processed/dashboard/ (chạy precompute_dashboard_data.py trước).
"""
import os, sqlite3, base64, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
DASH=os.path.join(ROOT,"data","processed","dashboard"); FIG=os.path.join(ROOT,"reports","figures")

# ---- bảng màu sáng, chuyên nghiệp ----
C_BG="#f4f6f9"; C_PANEL="#ffffff"; C_TEXT="#1f2933"; C_MUTE="#64748b"
C_ACC="#d64531"; C_BORDER="#e3e8ef"; C_LAND="#eef2f6"
TEMPLATE="plotly_white"
SHADOW="0 1px 3px rgba(16,24,40,.06), 0 1px 2px rgba(16,24,40,.04)"

def csv(n): return pd.read_csv(os.path.join(DASH,n))
con=sqlite3.connect(os.path.join(DASH,"ca_risk.sqlite"))
RISK=pd.read_sql("SELECT * FROM risk",con); con.close()
RISK["week"]=pd.to_datetime(RISK["week"]); RISK["wkstr"]=RISK["week"].dt.strftime("%Y-%m-%d")
YEARS=sorted(RISK["week"].dt.year.unique())

def style_fig(f,h=None):
    f.update_layout(paper_bgcolor=C_PANEL,plot_bgcolor=C_PANEL,font_color=C_TEXT,
        font_family="Inter,Segoe UI,sans-serif",title_font_size=15,
        margin=dict(l=12,r=12,t=44,b=12),colorway=["#d64531","#e08a1e","#2c7fb8","#41ab5d","#7b6cb3"])
    if h: f.update_layout(height=h)
    return f

by_year=csv("by_year.csv")
fig_year=style_fig(px.bar(by_year,x="year",y="n",color="source",template=TEMPLATE,
    title="Số vụ cháy theo năm (theo nguồn)",labels={"n":"Số vụ","year":"Năm","source":"Nguồn"}))
fig_year.add_vline(x=2020.5,line_dash="dash",line_color=C_ACC)
acres=by_year.groupby("year",as_index=False)["acres"].sum()
fig_acres=style_fig(px.area(acres,x="year",y="acres",template=TEMPLATE,
    title="Tổng diện tích cháy theo năm (acres)",labels={"acres":"Acres","year":"Năm"}))
fig_acres.update_traces(line_color=C_ACC,fillcolor="rgba(214,69,49,.15)")
cause=csv("by_cause.csv").sort_values("n")
fig_cause=style_fig(px.bar(cause,x="n",y="cause",orientation="h",template=TEMPLATE,
    title="Nguyên nhân cháy",labels={"n":"Số vụ","cause":""}))
fig_cause.update_traces(marker_color="#2c7fb8")
mon=csv("by_month.csv")
fig_mon=style_fig(px.bar(mon,x="month",y="n",template=TEMPLATE,title="Phân bố theo tháng (mùa cháy)",
    labels={"n":"Số vụ","month":"Tháng"}))
fig_mon.update_traces(marker_color="#e08a1e")
by_state=csv("by_state.csv")
fig_state=style_fig(px.choropleth(by_state,locations="STATE",locationmode="USA-states",color="n",
    scope="usa",color_continuous_scale="OrRd",template=TEMPLATE,labels={"n":"Số vụ"},
    title="Số vụ cháy theo bang"),h=460)
fig_state.update_geos(bgcolor=C_PANEL,lakecolor=C_PANEL)
metrics=csv("model_metrics.csv")

def img_b64(name):
    p=os.path.join(FIG,name)
    return "data:image/png;base64,"+base64.b64encode(open(p,"rb").read()).decode() if os.path.exists(p) else None

app=Dash(__name__,title="US Wildfire Prediction")

CARD={"backgroundColor":C_PANEL,"borderRadius":"10px","padding":"16px","border":f"1px solid {C_BORDER}","boxShadow":SHADOW}
TABSTY={"backgroundColor":"transparent","color":C_MUTE,"border":"none","padding":"12px 18px","fontWeight":"500"}
TABSEL={"backgroundColor":"transparent","color":C_ACC,"border":"none","borderBottom":f"2px solid {C_ACC}","padding":"12px 18px","fontWeight":"600"}

def metric_card(t,v):
    return html.Div([html.Div(t,style={"color":C_MUTE,"fontSize":"12px","textTransform":"uppercase","letterSpacing":".4px"}),
                     html.Div(v,style={"color":C_TEXT,"fontSize":"28px","fontWeight":"700","marginTop":"4px"})],
                    style={**CARD,"flex":"1","minWidth":"120px","textAlign":"center","padding":"14px"})

app.layout=html.Div(style={"backgroundColor":C_BG,"minHeight":"100vh","color":C_TEXT,
        "fontFamily":"Inter,Segoe UI,sans-serif"},children=[
    html.Div(style={"backgroundColor":C_PANEL,"borderBottom":f"1px solid {C_BORDER}","padding":"20px 36px"},children=[
        html.Div(style={"maxWidth":"1180px","margin":"0 auto"},children=[
            html.Div("US WILDFIRE PREDICTION",style={"fontSize":"20px","fontWeight":"700","letterSpacing":"1px"}),
            html.Div("Dự báo cháy rừng Mỹ — FPA FOD + NIFC (1992–2026), gridMET, Machine Learning",
                     style={"color":C_MUTE,"fontSize":"14px","marginTop":"2px"})])]),
    html.Div(style={"maxWidth":"1180px","margin":"0 auto","padding":"0 24px"},children=[
        dcc.Tabs(id="tabs",value="risk",children=[
            dcc.Tab(label="Bản đồ rủi ro (California)",value="risk",style=TABSTY,selected_style=TABSEL),
            dcc.Tab(label="Tổng quan toàn quốc",value="nat",style=TABSTY,selected_style=TABSEL),
            dcc.Tab(label="Hiệu năng mô hình",value="model",style=TABSTY,selected_style=TABSEL)]),
        html.Div(id="content",style={"padding":"18px 0 40px 0"})])])

@app.callback(Output("content","children"),Input("tabs","value"))
def render(tab):
    if tab=="risk":
        return html.Div(style={"display":"flex","gap":"18px","flexWrap":"wrap"},children=[
            html.Div(style={**CARD,"flex":"1","minWidth":"260px","maxWidth":"320px"},children=[
                html.Div("Lựa chọn",style={"fontWeight":"600","marginBottom":"10px"}),
                html.Label("Năm",style={"color":C_MUTE,"fontSize":"13px"}),
                dcc.Dropdown(id="year",options=[{"label":str(y),"value":int(y)} for y in YEARS],
                             value=int(YEARS[-1]),clearable=False),
                html.Div(style={"height":"12px"}),
                html.Label("Tuần",style={"color":C_MUTE,"fontSize":"13px"}),
                dcc.Dropdown(id="week",clearable=False),
                html.Div(style={"height":"14px"}),
                html.Label("Ngưỡng cảnh báo",style={"color":C_MUTE,"fontSize":"13px"}),
                dcc.Slider(id="thr",min=0,max=1,step=0.05,value=0.5,
                           marks={0:"0",0.5:"0.5",1:"1"},
                           tooltip={"placement":"bottom","always_visible":False}),
                html.Div(style={"height":"16px"}),
                html.Div(id="metrics",style={"display":"flex","flexWrap":"wrap","gap":"8px"}),
                html.Div(style={"height":"12px"}),
                html.Div(id="note",style={"backgroundColor":"#fff5f3","border":f"1px solid #f3c9bf",
                    "borderLeft":f"3px solid {C_ACC}","borderRadius":"8px","padding":"12px","fontSize":"14px"}),
                html.Div(style={"height":"10px"}),
                html.Div("Dữ liệu thời tiết gridMET tới ~15/06/2026. Cháy thực tế 2015–2020 từ FPA FOD, "
                         "2021–2026 từ NIFC (bỏ sót cháy nhỏ nên ít chấm xanh hơn thực tế).",
                         style={"color":C_MUTE,"fontSize":"12px","lineHeight":"1.5"})]),
            html.Div(style={**CARD,"flex":"3","minWidth":"480px","padding":"8px"},children=[
                dcc.Graph(id="risk_map",config={"displayModeBar":False},style={"height":"600px"})])])
    if tab=="nat":
        tot=int(by_year["n"].sum())
        cardwrap=lambda f:html.Div(dcc.Graph(figure=f,config={"displayModeBar":False}),style=CARD)
        return html.Div([
            html.Div(f"Tổng {tot:,} vụ cháy — FPA FOD (1992–2020) + NIFC (2021–2026)",
                     style={"color":C_MUTE,"marginBottom":"12px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"},children=[
                cardwrap(fig_year),cardwrap(fig_acres),cardwrap(fig_cause),cardwrap(fig_mon)]),
            html.Div(style={"height":"14px"}),cardwrap(fig_state)])
    rows=[html.Tr([html.Td(r["model"]),html.Td(r["metric"]),html.Td(f"{r['value']:.3f}")])
          for _,r in metrics.iterrows()]
    imgs=[]
    for f,cap in [("risk_map.png","Bản đồ rủi ro cháy California"),
                  ("occ_roc_pr.png","ROC & Precision-Recall — dự báo phát cháy"),
                  ("occ_feature_importance.png","Đặc trưng quan trọng — dự báo phát cháy"),
                  ("model_compare.png","So sánh thuật toán — mô hình cháy lớn"),
                  ("cause_confusion.png","Nhầm lẫn nguyên nhân (12 lớp)")]:
        b=img_b64(f)
        if b: imgs.append(html.Div([html.Div(cap,style={"color":C_MUTE,"marginBottom":"6px","fontWeight":"500"}),
            html.Img(src=b,style={"width":"100%","maxWidth":"900px","borderRadius":"6px","border":f"1px solid {C_BORDER}"})],
            style={**CARD,"marginBottom":"14px"}))
    return html.Div([
        html.Div([html.H3("Chỉ số các mô hình",style={"marginTop":"0"}),
            html.Table([html.Thead(html.Tr([html.Th("Mô hình"),html.Th("Chỉ số"),html.Th("Giá trị")])),
                        html.Tbody(rows)],className="mtable")],style={**CARD,"marginBottom":"14px"}),
        *imgs])

@app.callback(Output("week","options"),Output("week","value"),Input("year","value"))
def upd_week(y):
    sub=RISK[RISK["week"].dt.year==y]
    wks=sorted(sub["wkstr"].unique())
    peak=sub.groupby("wkstr")["y"].sum().idxmax()
    return [{"label":pd.Timestamp(w).strftime("%d/%m/%Y"),"value":w} for w in wks],peak

@app.callback(Output("risk_map","figure"),Output("metrics","children"),Output("note","children"),
              Input("week","value"),Input("thr","value"))
def upd_map(wk,thr):
    d=RISK[RISK["wkstr"]==wk].copy()
    fig=px.scatter_geo(d,lat="lat",lon="lon",color="risk",color_continuous_scale="YlOrRd",
        range_color=(0,1),scope="usa",template=TEMPLATE,
        labels={"risk":"Rủi ro","erc":"ERC","fm1000":"Ẩm nhiên liệu 1000h"},
        hover_data={"risk":":.2f","erc":":.0f","fm1000":":.1f","lat":False,"lon":False})
    fig.update_traces(marker=dict(size=7,symbol="square"))
    fires=d[d["y"]==1]
    if len(fires):
        fig.add_trace(go.Scattergeo(lat=fires["lat"],lon=fires["lon"],mode="markers",name="Cháy thực tế",
            marker=dict(size=8,color="rgba(0,0,0,0)",line=dict(color="#1d4ed8",width=1.5))))
    fig.update_geos(fitbounds="locations",showland=True,landcolor=C_LAND,
                    showsubunits=True,subunitcolor="#cbd5e1",bgcolor=C_PANEL,
                    showcoastlines=False,framecolor=C_BORDER)
    fig.update_layout(paper_bgcolor=C_PANEL,font_color=C_TEXT,margin=dict(l=0,r=0,t=4,b=0),
                      coloraxis_colorbar=dict(title="Rủi ro"),
                      legend=dict(bgcolor="rgba(255,255,255,.8)",bordercolor=C_BORDER,borderwidth=1,
                                  x=.01,y=.99))
    nf=int(d["y"].sum()); flagged=int((d["risk"]>=thr).sum())
    recall=d[d["risk"]>=thr]["y"].sum()/max(1,nf)
    top20=d.sort_values("risk",ascending=False).head(20)["y"].mean()
    cards=[metric_card("Ô có cháy",f"{nf}"),metric_card("Rủi ro TB",f"{d['risk'].mean():.2f}"),
           metric_card("Ô cảnh báo",f"{flagged}"),metric_card("Bắt được",f"{recall*100:.0f}%")]
    note=html.Span([html.B(f"{top20*100:.0f}%"),
        " trong 20 ô rủi ro cao nhất tuần này thực sự có cháy."])
    return fig,cards,note

if __name__=="__main__":
    app.run(debug=False,host="127.0.0.1",port=8050)
