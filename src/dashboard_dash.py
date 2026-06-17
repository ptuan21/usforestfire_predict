#!/usr/bin/env python3
"""
Dashboard US Wildfire Prediction — Plotly Dash.
Chạy:  python3 -m pip install dash plotly
       python3 src/dashboard_dash.py   ->  http://127.0.0.1:8050
Dữ liệu nhẹ đã tính sẵn ở data/processed/dashboard/ (chạy precompute_dashboard_data.py trước).
"""
import os, sqlite3, base64, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
DASH=os.path.join(ROOT,"data","processed","dashboard"); FIG=os.path.join(ROOT,"reports","figures")

# màu sắc giao diện (không emoji)
C_BG="#0f1419"; C_PANEL="#1a2128"; C_TEXT="#e6e6e6"; C_MUTE="#8aa0b0"; C_ACC="#e6552d"
TEMPLATE="plotly_dark"

def csv(n): return pd.read_csv(os.path.join(DASH,n))
con=sqlite3.connect(os.path.join(DASH,"ca_risk.sqlite"))
RISK=pd.read_sql("SELECT * FROM risk",con); con.close()
RISK["week"]=pd.to_datetime(RISK["week"]); RISK["wkstr"]=RISK["week"].dt.strftime("%Y-%m-%d")
YEARS=sorted(RISK["week"].dt.year.unique())

# ---- biểu đồ tổng quan (dựng sẵn) ----
by_year=csv("by_year.csv")
fig_year=px.bar(by_year,x="year",y="n",color="source",template=TEMPLATE,
    title="So vu chay theo nam (theo nguon)",labels={"n":"So vu","year":"Nam","source":"Nguon"})
fig_year.add_vline(x=2020.5,line_dash="dash",line_color=C_ACC)
acres=by_year.groupby("year",as_index=False)["acres"].sum()
fig_acres=px.area(acres,x="year",y="acres",template=TEMPLATE,
    title="Tong dien tich chay theo nam (acres)",labels={"acres":"Acres","year":"Nam"})
cause=csv("by_cause.csv").sort_values("n")
fig_cause=px.bar(cause,x="n",y="cause",orientation="h",template=TEMPLATE,
    title="Nguyen nhan chay",labels={"n":"So vu","cause":""})
mon=csv("by_month.csv")
fig_mon=px.bar(mon,x="month",y="n",template=TEMPLATE,title="Phan bo theo thang (mua chay)",
    labels={"n":"So vu","month":"Thang"})
by_state=csv("by_state.csv")
fig_state=px.choropleth(by_state,locations="STATE",locationmode="USA-states",color="n",
    scope="usa",color_continuous_scale="OrRd",template=TEMPLATE,labels={"n":"So vu"},
    title="So vu chay theo bang")
for f in (fig_year,fig_acres,fig_cause,fig_mon,fig_state):
    f.update_layout(paper_bgcolor=C_PANEL,plot_bgcolor=C_PANEL,font_color=C_TEXT,margin=dict(l=10,r=10,t=40,b=10))
metrics=csv("model_metrics.csv")

def img_b64(name):
    p=os.path.join(FIG,name)
    if not os.path.exists(p): return None
    return "data:image/png;base64,"+base64.b64encode(open(p,"rb").read()).decode()

app=Dash(__name__,title="US Wildfire Prediction")

CARD={"backgroundColor":C_PANEL,"borderRadius":"8px","padding":"14px","margin":"8px"}
def metric_card(t,v):
    return html.Div([html.Div(t,style={"color":C_MUTE,"fontSize":"13px"}),
                     html.Div(v,style={"color":C_TEXT,"fontSize":"26px","fontWeight":"600"})],
                    style={**CARD,"flex":"1","textAlign":"center"})

app.layout=html.Div(style={"backgroundColor":C_BG,"minHeight":"100vh","fontFamily":"Inter,Segoe UI,sans-serif",
                           "color":C_TEXT,"padding":"0 0 30px 0"},children=[
    html.Div(style={"backgroundColor":C_PANEL,"padding":"18px 28px","borderBottom":f"3px solid {C_ACC}"},children=[
        html.H2("US Wildfire Prediction",style={"margin":"0","color":C_TEXT}),
        html.Div("Du bao chay rung My — FPA FOD + NIFC (1992-2026), gridMET, Machine Learning",
                 style={"color":C_MUTE,"fontSize":"14px"})]),
    dcc.Tabs(id="tabs",value="risk",colors={"border":C_BG,"primary":C_ACC,"background":C_PANEL},children=[
        dcc.Tab(label="Ban do rui ro chay (California)",value="risk",style={"backgroundColor":C_PANEL,"color":C_MUTE},
                selected_style={"backgroundColor":C_BG,"color":C_TEXT,"borderTop":f"2px solid {C_ACC}"}),
        dcc.Tab(label="Tong quan toan quoc",value="nat",style={"backgroundColor":C_PANEL,"color":C_MUTE},
                selected_style={"backgroundColor":C_BG,"color":C_TEXT,"borderTop":f"2px solid {C_ACC}"}),
        dcc.Tab(label="Hieu nang mo hinh",value="model",style={"backgroundColor":C_PANEL,"color":C_MUTE},
                selected_style={"backgroundColor":C_BG,"color":C_TEXT,"borderTop":f"2px solid {C_ACC}"}),
    ]),
    html.Div(id="content",style={"padding":"10px 18px"}),
])

# ---------- nội dung theo tab ----------
@app.callback(Output("content","children"),Input("tabs","value"))
def render(tab):
    if tab=="risk":
        y0=YEARS[-1]
        return html.Div([
            html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"},children=[
                html.Div(style={"flex":"1","minWidth":"260px"},children=[
                    html.Label("Nam",style={"color":C_MUTE}),
                    dcc.Dropdown(id="year",options=[{"label":str(y),"value":int(y)} for y in YEARS],
                                 value=int(y0),clearable=False,style={"color":"#000"}),
                    html.Br(),
                    html.Label("Tuan",style={"color":C_MUTE}),
                    dcc.Dropdown(id="week",clearable=False,style={"color":"#000"}),
                    html.Br(),
                    html.Label("Nguong canh bao",style={"color":C_MUTE}),
                    dcc.Slider(id="thr",min=0,max=1,step=0.05,value=0.5,
                               marks={0:"0",0.5:"0.5",1:"1"}),
                    html.Br(),
                    html.Div(id="metrics",style={"display":"flex","flexWrap":"wrap","gap":"6px"}),
                    html.Div(id="note",style={**CARD,"borderLeft":f"3px solid {C_ACC}"}),
                ]),
                html.Div(style={"flex":"3","minWidth":"480px"},children=[
                    dcc.Graph(id="risk_map",style={"height":"600px"})]),
            ]),
        ])
    if tab=="nat":
        tot=int(by_year["n"].sum())
        return html.Div([
            html.P(f"Tong {tot:,} vu chay — FPA FOD (1992-2020) + NIFC (2021-2026)",style={"color":C_MUTE}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"6px"},children=[
                dcc.Graph(figure=fig_year),dcc.Graph(figure=fig_acres),
                dcc.Graph(figure=fig_cause),dcc.Graph(figure=fig_mon)]),
            dcc.Graph(figure=fig_state),
        ])
    # model
    rows=[html.Tr([html.Td(r["model"]),html.Td(r["metric"]),
                   html.Td(f"{r['value']:.3f}")]) for _,r in metrics.iterrows()]
    imgs=[]
    for f,cap in [("risk_map.png","Ban do rui ro chay California"),
                  ("occ_roc_pr.png","ROC & Precision-Recall — du bao phat chay"),
                  ("occ_feature_importance.png","Feature quan trong — du bao phat chay"),
                  ("model_compare.png","So sanh thuat toan — model chay lon"),
                  ("cause_confusion.png","Nham lan nguyen nhan (12 lop)")]:
        b=img_b64(f)
        if b: imgs+=[html.P(cap,style={"color":C_MUTE,"marginBottom":"4px"}),
                     html.Img(src=b,style={"width":"100%","maxWidth":"900px","borderRadius":"6px","marginBottom":"18px"})]
    return html.Div([
        html.H3("Chi so cac mo hinh"),
        html.Table([html.Thead(html.Tr([html.Th("Mo hinh"),html.Th("Chi so"),html.Th("Gia tri")])),
                    html.Tbody(rows)],style={"width":"100%","borderCollapse":"collapse"},className="mtable"),
        html.Br(),*imgs])

# ---------- cập nhật tuần theo năm ----------
@app.callback(Output("week","options"),Output("week","value"),Input("year","value"))
def upd_week(y):
    wks=sorted(RISK[RISK["week"].dt.year==y]["wkstr"].unique())
    peak=RISK[RISK["week"].dt.year==y].groupby("wkstr")["y"].sum().idxmax()
    opts=[{"label":pd.Timestamp(w).strftime("%d/%m/%Y"),"value":w} for w in wks]
    return opts,peak

# ---------- bản đồ + chỉ số ----------
@app.callback(Output("risk_map","figure"),Output("metrics","children"),Output("note","children"),
              Input("week","value"),Input("thr","value"))
def upd_map(wk,thr):
    d=RISK[RISK["wkstr"]==wk].copy()
    fig=px.scatter_geo(d,lat="lat",lon="lon",color="risk",color_continuous_scale="YlOrRd",
        range_color=(0,1),scope="usa",template=TEMPLATE,
        hover_data={"risk":":.2f","erc":":.0f","fm1000":":.1f","lat":False,"lon":False})
    fig.update_traces(marker=dict(size=7,symbol="square"))
    fires=d[d["y"]==1]
    if len(fires):
        fig.add_trace(go.Scattergeo(lat=fires["lat"],lon=fires["lon"],mode="markers",name="chay thuc te",
            marker=dict(size=7,color="rgba(0,0,0,0)",line=dict(color="#3aa0ff",width=1.6))))
    fig.update_geos(fitbounds="locations",showland=True,landcolor="#222a31",
                    showsubunits=True,subunitcolor="#445",bgcolor=C_PANEL)
    fig.update_layout(paper_bgcolor=C_PANEL,font_color=C_TEXT,margin=dict(l=0,r=0,t=6,b=0),
                      legend=dict(bgcolor=C_PANEL))
    nf=int(d["y"].sum()); flagged=int((d["risk"]>=thr).sum())
    recall=d[d["risk"]>=thr]["y"].sum()/max(1,nf)
    top20=d.sort_values("risk",ascending=False).head(20)["y"].mean()
    cards=[metric_card("O co chay thuc te",f"{nf}"),
           metric_card("Rui ro trung binh",f"{d['risk'].mean():.2f}"),
           metric_card(f"O canh bao (>={thr:.2f})",f"{flagged}"),
           metric_card("Bat duoc chay",f"{recall*100:.0f}%")]
    note=f"Trong 20 o rui ro cao nhat tuan nay, {top20*100:.0f}% thuc su co chay."
    return fig,cards,note

if __name__=="__main__":
    app.run(debug=False,host="127.0.0.1",port=8050)
