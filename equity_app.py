"""
equity_app.py — Web App Streamlit — Equity Pattern Discovery
Importa pipeline_ab.py e engine_c.py

Avvio: streamlit run equity_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io
from datetime import datetime, date

from pipeline_ab import (
    download_ohlcv, run_integrity_checks, apply_universe_filters,
    build_base_dataset, get_diagnostics, summary_stats,
    DEFAULT_DOWNLOAD_CONFIG, DEFAULT_FILTER_CONFIG, DEFAULT_FEATURE_CONFIG,
)
from engine_c import (
    run_grid_search, blind_pattern, get_blind_diagnostics,
    analyze_blinded_pattern, build_final_pattern_table,
    build_episode_dataset, get_canoni, get_default_grid,
    get_default_values, count_combinations,
    PATTERN_IDS, PATTERN_LABELS, PATTERN_GRIDS,
    rank_results,
)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Equity Pattern Discovery", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');
:root{--bg-primary:#0a0e17;--bg-secondary:#111827;--bg-card:#151d2e;--border:#1e2d47;--border-bright:#2a3f63;--text-primary:#e8edf5;--text-secondary:#8a9ab5;--text-dim:#4a5a75;--accent-blue:#3b82f6;--accent-cyan:#06b6d4;--accent-green:#10b981;--accent-red:#ef4444;--accent-amber:#f59e0b;--accent-purple:#8b5cf6;}
.stApp{background:var(--bg-primary);color:var(--text-primary);}
.main .block-container{padding:1.5rem 2rem;max-width:1600px;}
#MainMenu,footer,header{visibility:hidden;}.stDeployButton{display:none;}
[data-testid="stSidebar"]{background:var(--bg-secondary)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] .stRadio label{color:var(--text-secondary)!important;font-family:'JetBrains Mono',monospace;font-size:0.85rem;}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:var(--text-primary)!important;letter-spacing:-0.02em;}
p,div,span,label{font-family:'JetBrains Mono',monospace;}
.kpi-card{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:1.1rem 1.3rem;position:relative;overflow:hidden;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent-purple),var(--accent-cyan));}
.kpi-label{font-size:0.66rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.4rem;}
.kpi-value{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:700;line-height:1;margin-bottom:0.2rem;}
.kpi-sub{font-size:0.7rem;color:var(--text-secondary);}
.section-header{font-size:0.68rem;color:var(--accent-cyan);text-transform:uppercase;letter-spacing:0.2em;border-bottom:1px solid var(--border);padding-bottom:0.4rem;margin:1.4rem 0 0.8rem 0;}
.step-card{background:var(--bg-card);border:1px solid var(--border);border-left:3px solid var(--accent-purple);border-radius:8px;padding:1rem 1.3rem;margin-bottom:0.8rem;}
.step-card h4{font-family:'Syne',sans-serif!important;font-size:0.95rem;color:var(--accent-cyan)!important;margin-bottom:0.5rem;}
.step-card p{font-size:0.76rem;color:var(--text-secondary);line-height:1.6;margin:0;}
.top-bar{display:flex;align-items:center;justify-content:space-between;padding:0.7rem 0;border-bottom:1px solid var(--border);margin-bottom:1.3rem;}
.top-bar-title{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.03em;}
.top-bar-meta{font-size:0.7rem;color:var(--text-dim);}
.disclaimer{background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.2);border-left:3px solid #f59e0b;border-radius:4px;padding:0.7rem 1rem;font-size:0.7rem;color:var(--text-secondary);margin-top:0.8rem;}
.warn-box{background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.2);border-left:3px solid #ef4444;border-radius:4px;padding:0.7rem 1rem;font-size:0.72rem;color:var(--text-secondary);margin:0.5rem 0;}
.ok-box{background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.2);border-left:3px solid #10b981;border-radius:4px;padding:0.7rem 1rem;font-size:0.72rem;color:var(--text-secondary);margin:0.5rem 0;}
.blinded-badge{display:inline-block;padding:0.25rem 0.7rem;border-radius:4px;font-size:0.72rem;font-weight:700;background:rgba(16,185,129,0.15);color:#10b981;border:1px solid #10b981;}
</style>""", unsafe_allow_html=True)

DEFAULT_TICKERS = [
    "ENI.MI","ENEL.MI","ISP.MI","UCG.MI","TIT.MI","STLAM.MI",
    "MB.MI","PRY.MI","SRG.MI","AMP.MI","G.MI","SPM.MI",
    "BMED.MI","BPSO.MI","CPR.MI","CNHI.MI","ERG.MI","FBK.MI",
    "HER.MI","LDO.MI","MONC.MI","RACE.MI","REC.MI","SAP.MI",
    "TRN.MI","A2A.MI","IREN.MI","BZU.MI","PST.MI","INW.MI",
]

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(21,29,46,0.6)",
    font=dict(family="JetBrains Mono", color="#8a9ab5", size=11),
    xaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d47", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=40),
)

for k,v in {
    "ohlcv_raw":None,"integrity_report":None,"quality_df":None,"anomaly_df":None,
    "ohlcv_final":None,"valid_tickers":[],"filter_report":None,
    "base_dataset":None,"feature_cols":[],"fwd_cols":[],"diagnostics":None,
    "gs_results":{},"gs_top":{},"blinded_configs":{},"pattern_dfs":{},
    "blind_diag":{},"analysis":{},"step_data":False,"step_feat":False,"step_pattern":False,
    "start_date":"2018-01-01","end_date":datetime.today().strftime("%Y-%m-%d"),
}.items():
    if k not in st.session_state: st.session_state[k]=v

def kpi(label,val,sub,color):
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value' style='color:{color};'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)
def section(t): st.markdown(f"<div class='section-header'>{t}</div>",unsafe_allow_html=True)
def top_bar(t,m): st.markdown(f"<div class='top-bar'><div class='top-bar-title'>{t}</div><div class='top-bar-meta'>{m}</div></div>",unsafe_allow_html=True)
def step_card(t,b): st.markdown(f"<div class='step-card'><h4>{t}</h4><p>{b}</p></div>",unsafe_allow_html=True)
def ok_box(m): st.markdown(f"<div class='ok-box'>✅ {m}</div>",unsafe_allow_html=True)
def warn_box(m): st.markdown(f"<div class='warn-box'>⚠️ {m}</div>",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='padding:1rem 0 1.2rem 0;'><div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#e8edf5;'>📊 Equity Patterns</div><div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#4a5a75;margin-top:4px;text-transform:uppercase;letter-spacing:0.1em;'>Pattern Discovery — Azionario Italia</div></div>",unsafe_allow_html=True)
    si=lambda d:"🟢" if d else "⚪"
    nb=len(st.session_state.blinded_configs)
    st.markdown(f"<div style='font-size:0.62rem;color:#4a5a75;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;'>Avanzamento</div><div style='font-size:0.76rem;color:#8a9ab5;line-height:2.1;'>{si(st.session_state.step_data)} 1. Universo &amp; Dati<br>{si(st.session_state.step_feat)} 2. Feature Engineering<br>{si(st.session_state.step_pattern)} 3. Pattern Discovery<br>{si(nb>0)} 4. Risultati ({nb} blindati)</div>",unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1e2d47;margin:0.8rem 0;'>",unsafe_allow_html=True)
    page=st.radio("nav",["📥  Universo & Dati","⚙️  Feature Engineering","📉  Pattern Discovery","📋  Risultati","📤  Export"],label_visibility="collapsed")
    st.markdown("<hr style='border-color:#1e2d47;margin:0.8rem 0;'>",unsafe_allow_html=True)
    if st.button("🔄 Reset",use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    if st.session_state.step_data:
        df_s=st.session_state.ohlcv_final or st.session_state.ohlcv_raw or pd.DataFrame()
        if not df_s.empty:
            st.markdown(f"<div style='font-size:0.62rem;color:#4a5a75;margin-top:0.5rem;'>Ticker: <span style='color:#8a9ab5;'>{df_s['ticker'].nunique()}</span><br>Righe: <span style='color:#8a9ab5;'>{len(df_s):,}</span><br>Periodo: <span style='color:#8a9ab5;'>{st.session_state.start_date[:7]} → {st.session_state.end_date[:7]}</span></div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 1 — UNIVERSO & DATI
# ══════════════════════════════════════════════════════════
if "Universo" in page:
    top_bar("Universo & Dati","Step 1 — Download OHLCV · Controlli integrità · Filtri universo")
    step_card("📋 Flusso","A.1 Selezione ticker → A.2 Config download → A.3 Download OHLCV → A.4 Controlli integrità → A.5 Filtri → Dataset pulito")

    section("A.1 — SELEZIONE UNIVERSO")
    mode=st.radio("Modalità",["Lista predefinita FTSE Italia","Inserimento manuale","Upload CSV"],horizontal=True,label_visibility="collapsed")
    if mode=="Lista predefinita FTSE Italia":
        tickers_sel=st.multiselect("Ticker",DEFAULT_TICKERS,default=DEFAULT_TICKERS[:15],label_visibility="collapsed")
    elif mode=="Inserimento manuale":
        raw=st.text_area("Ticker separati da virgola",height=80,label_visibility="collapsed",placeholder="ENI.MI, ENEL.MI...")
        tickers_sel=[t.strip().upper() for t in raw.replace("\n",",").split(",") if t.strip()]
    else:
        up=st.file_uploader("CSV con colonna ticker",type=["csv"])
        tickers_sel=[]
        if up:
            dfu=pd.read_csv(up)
            if "ticker" in dfu.columns: tickers_sel=dfu["ticker"].tolist()
            else: st.error("Il CSV deve avere colonna 'ticker'")
    st.caption(f"{len(tickers_sel)} ticker selezionati")

    section("A.2 — CONFIGURAZIONE DOWNLOAD")
    ca,cb,cc=st.columns(3)
    with ca:
        start_d=st.date_input("Inizio",value=date(2018,1,1),min_value=date(2010,1,1),max_value=date.today(),label_visibility="collapsed")
        end_d=st.date_input("Fine",value=date.today(),min_value=date(2010,1,1),max_value=date.today(),label_visibility="collapsed")
    with cb:
        chunk_size=st.number_input("Batch size",5,50,20)
        max_retries=st.number_input("Retry su errore",1,5,3)
    with cc:
        min_rows=st.number_input("Min righe ticker",30,500,60)
        remove_outlier=st.checkbox("Rimuovi var>100%",value=True)
    dl_cfg={"chunk_size":chunk_size,"max_retries":max_retries,"min_rows":min_rows,"remove_outliers":remove_outlier}

    section("A.3 — DOWNLOAD OHLCV")
    if st.button("▶️ Scarica dati OHLCV",disabled=len(tickers_sel)==0):
        pb=st.progress(0,text="Inizializzazione...")
        sc={"ok":0,"empty":0,"error":0}
        def cb_dl(i,total,ticker,status):
            sc[status]=sc.get(status,0)+1
            pb.progress(i/total,text=f"{i}/{total} — {ticker} | ok:{sc['ok']} empty:{sc['empty']}")
        ohlcv_raw=download_ohlcv(tickers_sel,str(start_d),str(end_d),config=dl_cfg,progress_callback=cb_dl)
        pb.empty()
        if ohlcv_raw.empty: st.error("Nessun dato. Verifica ticker e connessione.")
        else:
            st.session_state.update({"ohlcv_raw":ohlcv_raw,"start_date":str(start_d),"end_date":str(end_d),
                "ohlcv_final":None,"base_dataset":None,"step_data":False,"step_feat":False})
            ok_box(f"Scaricati {ohlcv_raw['ticker'].nunique()} ticker, {len(ohlcv_raw):,} righe")
            st.rerun()

    if st.session_state.ohlcv_raw is not None:
        section("A.4 — CONTROLLI INTEGRITÀ")
        if st.button("🔍 Esegui controlli integrità"):
            with st.spinner(""):
                rpt,qdf,adf=run_integrity_checks(st.session_state.ohlcv_raw)
            st.session_state.update({"integrity_report":rpt,"quality_df":qdf,"anomaly_df":adf})
            st.rerun()
        if st.session_state.integrity_report:
            r=st.session_state.integrity_report
            k1,k2,k3,k4=st.columns(4)
            for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4],[
                ("Righe totali",f"{r['total_rows']:,}","dataset grezzo","#3b82f6"),
                ("Ticker raw",str(r['n_tickers_raw']),"scaricati","#8b5cf6"),
                ("Duplicati",str(r['n_duplicates']),"(ticker,date)","#10b981" if r['n_duplicates']==0 else "#ef4444"),
                ("Righe anomale",f"{r['n_anomaly_rows']} ({r['anomaly_pct']:.2f}%)","high<low|ret>100%","#10b981" if r['anomaly_pct']<1 else "#f59e0b"),
            ]): col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.4rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)
            if not r.get("check_passed",False): warn_box(f"Anomalie: {r['n_anomaly_rows']} righe saranno rimosse dai filtri.")
            if st.session_state.quality_df is not None and not st.session_state.quality_df.empty:
                with st.expander("📊 Qualità per ticker"):
                    st.dataframe(st.session_state.quality_df,use_container_width=True,hide_index=True,height=260)

        section("A.5 — FILTRI UNIVERSO")
        cf1,cf2,cf3=st.columns(3)
        with cf1:
            min_years=st.slider("Min anni storia",0.5,15.0,3.0,0.5)
            min_obs=st.number_input("Min osservazioni",50,5000,500,50)
        with cf2:
            max_miss=st.slider("Max missing %",0.0,50.0,5.0,0.5)
            excl_anom=st.checkbox("Escludi ticker con anomalie OHLCV",value=True)
        with cf3:
            max_anom=st.slider("Max anomalie % tollerato",0.0,10.0,1.0,0.1)
        fcfg={"min_years":min_years,"min_obs":min_obs,"max_missing_pct":max_miss,"exclude_anomalies":excl_anom,"max_anomaly_pct":max_anom}
        if st.button("▶️ Applica filtri universo"):
            with st.spinner(""):
                df_fin,vtk,frpt=apply_universe_filters(st.session_state.ohlcv_raw,fcfg)
            if df_fin.empty: st.error("Nessun ticker supera i filtri.")
            else:
                st.session_state.update({"ohlcv_final":df_fin,"valid_tickers":vtk,"filter_report":frpt,
                    "step_data":True,"base_dataset":None,"step_feat":False})
                st.rerun()
        if st.session_state.filter_report:
            fr=st.session_state.filter_report
            k1,k2,k3=st.columns(3)
            for col,(lbl,val,sub,c) in zip([k1,k2,k3],[
                ("Ticker validi",str(fr['n_tickers_valid']),"superato filtri","#10b981"),
                ("Ticker esclusi",str(fr['n_tickers_excluded']),"non superato","#f59e0b"),
                ("Righe finali",f"{len(st.session_state.ohlcv_final):,}" if st.session_state.ohlcv_final is not None else "0","dataset filtrato","#3b82f6"),
            ]): col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.4rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)
            if fr.get("excluded_detail"):
                with st.expander("🔍 Ticker esclusi"):
                    st.dataframe(pd.DataFrame([{"Ticker":tk,"Motivo":m} for tk,m in fr["excluded_detail"].items()]),use_container_width=True,hide_index=True)
            warn_box("Survivorship bias: Yahoo Finance non garantisce un universo point-in-time. Ticker delisted potrebbero mancare.")
            if st.session_state.ohlcv_final is not None:
                section("ANTEPRIMA")
                tk_opts=[t.replace(".MI","")+" ("+t+")" for t in st.session_state.ohlcv_final["ticker"].unique()[:20]]
                tk_sel=st.selectbox("Ticker",tk_opts,label_visibility="collapsed")
                tk_code=tk_sel.split("(")[1].rstrip(")")
                sub_p=st.session_state.ohlcv_final[st.session_state.ohlcv_final["ticker"]==tk_code].tail(90)
                fig_c=go.Figure()
                fig_c.add_candlestick(x=sub_p["date"],open=sub_p.get("open",sub_p["close"]),high=sub_p.get("high",sub_p["close"]),low=sub_p.get("low",sub_p["close"]),close=sub_p["close"],increasing_line_color="#10b981",decreasing_line_color="#ef4444")
                fig_c.update_layout(**PLOTLY_DARK,height=200,yaxis_title="Prezzo (€)",title=dict(text=""))
                st.plotly_chart(fig_c,use_container_width=True)
                ok_box("Dataset pronto → Step 2 Feature Engineering")

# ══════════════════════════════════════════════════════════
# PAGE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
elif "Feature" in page:
    top_bar("Feature Engineering","Step 2 — Blocco B.3 feature primitive + Blocco B.4 forward path")
    if not st.session_state.step_data: st.warning("⚠️ Prima Step 1"); st.stop()
    step_card("⚙️ Feature","B.3: ret_1d, prev_close, delta_close, delta_open_close, range_day, body, body_abs, lower/upper_shadow, close_pos_in_range, close_vs_open/high/low, body_range_ratio, shadow_ratios, gap_abs, gap_pct, true_range, row_idx, ticker_obs_idx | B.4: fwd_close_tH, fwd_ret_tH (±win%), fwd_max_ret_HH (MFE), fwd_min_ret_HH (MAE)")
    section("CONFIGURAZIONE")
    cc1,cc2,cc3,cc4=st.columns(4)
    with cc1: h_max=st.slider("H_max",5,20,15,5)
    with cc2: fwd_close=st.checkbox("fwd_close_tH",value=True)
    with cc3: fwd_max=st.checkbox("MFE (fwd_max)",value=True)
    with cc4: fwd_min=st.checkbox("MAE (fwd_min)",value=True)
    win_pct=st.slider("Winsorizzazione fwd_ret (%)",10,100,50,5)
    fcfg={**DEFAULT_FEATURE_CONFIG,"h_max":h_max,"winsorize_pct":win_pct,"fwd_close":fwd_close,"fwd_max_return":fwd_max,"fwd_min_return":fwd_min}
    st.caption(f"Input: {st.session_state.ohlcv_final['ticker'].nunique()} ticker, {len(st.session_state.ohlcv_final):,} righe")
    if st.button("▶️ Costruisci feature + forward path"):
        with st.spinner("In corso..."):
            ds,fc,fwdc=build_base_dataset(st.session_state.ohlcv_final,fcfg)
            diag=get_diagnostics(ds,fc,fwdc)
        st.session_state.update({"base_dataset":ds,"feature_cols":fc,"fwd_cols":fwdc,"diagnostics":diag,"step_feat":True,"step_pattern":False})
        st.rerun()
    if st.session_state.step_feat and st.session_state.base_dataset is not None:
        ds=st.session_state.base_dataset
        section("RIEPILOGO")
        k1,k2,k3,k4=st.columns(4)
        for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4],[
            ("Righe",f"{len(ds):,}","dataset base","#10b981"),
            ("Feature primitive",str(len(st.session_state.feature_cols)),"22 colonne","#3b82f6"),
            ("Forward ret",str(len([x for x in st.session_state.fwd_cols if x.startswith("fwd_ret_")])),f"H=1→{h_max}","#8b5cf6"),
            ("Ticker",str(ds["ticker"].nunique()),"","#06b6d4"),
        ]): col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.4rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)
        cd,cf=st.columns(2)
        with cd:
            section("DISTRIBUZIONE ret_1d")
            rets=ds["ret_1d"].dropna()*100
            fig_r=go.Figure()
            fig_r.add_histogram(x=rets,nbinsx=100,marker_color="#3b82f6",opacity=0.8,histnorm="probability density")
            fig_r.add_vline(x=0,line=dict(color="#4a5a75",width=1))
            fig_r.update_layout(**PLOTLY_DARK,height=230,xaxis_title="ret_1d (%)",yaxis_title="Densità",showlegend=False,title=dict(text=f"μ={rets.mean():.3f}%  σ={rets.std():.2f}%"))
            st.plotly_chart(fig_r,use_container_width=True)
        with cf:
            section("BASELINE DRIFT")
            fwd_h=[x for x in st.session_state.fwd_cols if x.startswith("fwd_ret_t")]
            hh=sorted([int(x[9:]) for x in fwd_h if x[9:].isdigit()])
            means=[ds[f"fwd_ret_t{H}"].mean()*100 for H in hh]
            fig_f=go.Figure()
            fig_f.add_scatter(x=hh,y=means,mode="lines+markers",line=dict(color="#8b5cf6",width=2),marker=dict(size=5))
            fig_f.add_hline(y=0,line=dict(color="#4a5a75",width=0.8))
            fig_f.update_layout(**PLOTLY_DARK,height=230,xaxis_title="H (giorni)",yaxis_title="Avg fwd ret (%)",showlegend=False,title=dict(text=""))
            st.plotly_chart(fig_f,use_container_width=True)
        ok_box("Feature pronte → Step 3 Pattern Discovery")

# ══════════════════════════════════════════════════════════
# PAGE 3 — PATTERN DISCOVERY
# ══════════════════════════════════════════════════════════
elif "Pattern Discovery" in page:
    top_bar("Pattern Discovery","Step 3 — Grid Search + Blindatura (Blocco C)")
    if not st.session_state.step_feat: st.warning("⚠️ Prima Step 2"); st.stop()
    step_card("📉 Procedura","1. Seleziona pattern | 2. Configura griglia (MultiSelect) | 3. Grid Search → top combinazioni lift/n_canoni | 4. Scegli params | 5. Blinda (parametri scelti PRIMA di guardare i forward return — elimina look-ahead bias) | 6. Ripeti per altri pattern")
    ds=st.session_state.base_dataset
    section("SELEZIONE PATTERN")
    pat_id=st.selectbox("Pattern",PATTERN_IDS,format_func=lambda x:f"{PATTERN_LABELS[x]} ({x})",label_visibility="collapsed")
    is_bl=pat_id in st.session_state.blinded_configs
    if is_bl:
        bc=st.session_state.blinded_configs[pat_id]
        st.markdown(f"<span class='blinded-badge'>✅ BLINDATO</span> <span style='font-size:0.72rem;color:#4a5a75;margin-left:0.8rem;'>Lift={bc.get('lift','—')} | N={bc.get('n_canoni','—')} | {bc.get('fwd_col','—')}</span>",unsafe_allow_html=True)
    section(f"GRIGLIA PARAMETRI — {PATTERN_LABELS[pat_id].upper()}")
    grid_def=PATTERN_GRIDS[pat_id]
    param_grid_sel={}
    cols_g=st.columns(min(len(grid_def),3))
    for i,(param,cfg) in enumerate(grid_def.items()):
        with cols_g[i%len(cols_g)]:
            sel=st.multiselect(param,options=cfg["options"],default=cfg["default"],key=f"g_{pat_id}_{param}")
            param_grid_sel[param]=sel
    cg1,cg2,cg3=st.columns(3)
    with cg1: max_comb=st.number_input("Max combinazioni",50,1000,300,50)
    with cg2: min_canoni=st.number_input("Min canoni",5,200,30,5)
    with cg3: min_gap_gs=st.number_input("Min gap (gg)",1,30,5,1)
    valid_grid={k:v for k,v in param_grid_sel.items() if len(v)>0}
    n_tot=count_combinations(valid_grid) if valid_grid else 0
    n_eff=min(n_tot,max_comb)
    cap=n_tot>max_comb
    st.markdown(f"<div style='font-size:0.72rem;color:#8a9ab5;margin:0.3rem 0;'>Combinazioni: <b style='color:{'#f59e0b' if cap else '#10b981'}'>{n_tot}</b>{'→ cap '+str(max_comb) if cap else ' ✅'} | Effettive: <b>{n_eff}</b></div>",unsafe_allow_html=True)
    if st.button(f"🚀 Avvia Grid Search — {PATTERN_LABELS[pat_id]}",disabled=n_eff==0):
        pb_gs=st.progress(0,text="Grid search...")
        def gs_cb(i,total): pb_gs.progress(i/total,text=f"{i}/{total}...")
        with st.spinner(""):
            gs_res=run_grid_search(ds,pat_id,valid_grid,min_gap=min_gap_gs,max_comb=max_comb,min_canoni=min_canoni,progress_callback=gs_cb)
        pb_gs.empty()
        st.session_state.gs_results[pat_id]=gs_res
        if gs_res:
            tl,tc=rank_results(gs_res,top_n=5)
            st.session_state.gs_top[pat_id]=(tl,tc)
            ok_box(f"{len(gs_res)} combinazioni valide su {n_eff} testate")
        else: warn_box("Nessuna combinazione con abbastanza canoni. Abbassa min_canoni o allarga la griglia.")
        st.rerun()
    if pat_id in st.session_state.gs_results and st.session_state.gs_results[pat_id]:
        tl,tc=st.session_state.gs_top.get(pat_id,(pd.DataFrame(),pd.DataFrame()))
        ctl,ctc=st.columns(2)
        with ctl:
            section("TOP 5 PER LIFT")
            if not tl.empty: st.dataframe(tl,use_container_width=True,hide_index=False,height=220)
        with ctc:
            section("TOP 5 PER N. CANONI")
            if not tc.empty: st.dataframe(tc,use_container_width=True,hide_index=False,height=220)
    section(f"BLINDATURA — {PATTERN_LABELS[pat_id].upper()}")
    st.markdown("<div class='step-card' style='border-left-color:#f59e0b;'><h4 style='color:#f59e0b!important;'>🔒 Procedura di blindatura</h4><p>Scegli i parametri guardando SOLO i ranking lift/n_canoni della grid search, NON la distribuzione dei forward return. Poi clicca Blinda — solo allora il sistema mostra i return effettivi. Questo elimina il look-ahead bias.</p></div>",unsafe_allow_html=True)
    blind_params={}
    cols_b=st.columns(min(len(grid_def),4))
    for i,(param,cfg_p) in enumerate(grid_def.items()):
        with cols_b[i%len(cols_b)]:
            opts=cfg_p["options"]; def_v=cfg_p["default"][0] if cfg_p["default"] else opts[0]
            blind_params[param]=st.selectbox(param,opts,index=opts.index(def_v) if def_v in opts else 0,key=f"bl_{pat_id}_{param}")
    min_gap_bl=st.number_input("Min gap blindatura",1,30,min_gap_gs,key=f"blg_{pat_id}")
    if st.button(f"🔒 Blinda — {PATTERN_LABELS[pat_id]}"):
        H_sel=blind_params.get("H",5); fwd_col=f"fwd_ret_t{H_sel}"
        params_nh={k:v for k,v in blind_params.items() if k!="H"}
        if fwd_col not in ds.columns: st.error(f"Colonna {fwd_col} non trovata. Aumenta H_max nel Blocco B.")
        else:
            with st.spinner("Blindatura..."):
                df_pat,blind_cfg=blind_pattern(ds,pat_id,params_nh,min_gap_bl,fwd_col)
                diag_bl=get_blind_diagnostics(df_pat,pat_id,fwd_col,min_gap_bl)
                analysis_=analyze_blinded_pattern(df_pat,pat_id,fwd_col,h_max=15)
            st.session_state.blinded_configs[pat_id]=blind_cfg
            st.session_state.pattern_dfs[pat_id]=df_pat
            st.session_state.blind_diag[pat_id]=diag_bl
            st.session_state.analysis[pat_id]=analysis_
            st.session_state.step_pattern=True
            ok_box(f"{PATTERN_LABELS[pat_id]} blindato! N={blind_cfg['n_canoni']} | Lift={blind_cfg.get('lift','—')}")
            st.rerun()
    if pat_id in st.session_state.blinded_configs:
        bd=st.session_state.blind_diag.get(pat_id,{})
        if bd and "n_canoni" in bd:
            section("PREVIEW POST-BLINDATURA")
            H_b=st.session_state.blinded_configs[pat_id].get("fwd_col","fwd_ret_t5").replace("fwd_ret_t","")
            k1,k2,k3,k4=st.columns(4)
            for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4],[
                ("N canoni",str(bd["n_canoni"]),f"{ds['ticker'].nunique()} ticker","#8b5cf6"),
                (f"Avg Ret H={H_b}",f"{bd['avg_ret']*100:.2f}%",f"baseline: {bd['avg_baseline']*100:.2f}%","#10b981" if bd["avg_ret"]>0 else "#ef4444"),
                ("Hit Rate",f"{bd['hit_rate']:.1%}","% canoni ret>0","#10b981" if bd["hit_rate"]>0.5 else "#ef4444"),
                ("Lift",f"{bd['lift_glob']:.2f}x",f"p={bd['p_val']:.3f} {'✅' if bd['sig'] else '⚠️'}","#10b981" if bd["lift_glob"]>1 else "#ef4444"),
            ]): col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.4rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — RISULTATI
# ══════════════════════════════════════════════════════════
elif "Risultati" in page:
    top_bar("Risultati — Analisi Post-Blindatura","Step 4 — Distribuzione · Lift · Stabilità · Breakdown ticker")
    if not st.session_state.blinded_configs: st.warning("⚠️ Blinda almeno un pattern"); st.stop()
    blinded_pids=list(st.session_state.blinded_configs.keys())
    pat_sel=st.selectbox("Pattern",blinded_pids,format_func=lambda x:f"{PATTERN_LABELS[x]} ✅",label_visibility="collapsed")
    bd=st.session_state.blind_diag.get(pat_sel,{})
    an=st.session_state.analysis.get(pat_sel,{})
    df_p=st.session_state.pattern_dfs.get(pat_sel)
    bc=st.session_state.blinded_configs.get(pat_sel,{})
    if not bd or not an or df_p is None: st.warning("Dati non disponibili"); st.stop()
    H_list=sorted(an.keys())
    H_sel=st.select_slider("Orizzonte",options=H_list,value=H_list[1] if len(H_list)>1 else H_list[0])
    m=an[H_sel]
    k1,k2,k3,k4,k5=st.columns(5)
    for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4,k5],[
        ("Episodi",str(m["n_canoni"]),f"{df_p['ticker'].nunique()} ticker","#8b5cf6"),
        ("Avg Return",f"{m['avg_ret']*100:.2f}%",f"baseline: {m['avg_baseline']*100:.2f}%","#10b981" if m["avg_ret"]>0 else "#ef4444"),
        ("Hit Rate",f"{m['hit_rate']:.1%}","% ret>0","#10b981" if m["hit_rate"]>0.5 else "#ef4444"),
        ("Lift",f"{m['lift']:.2f}x","vs baseline","#10b981" if m["lift"]>1 else "#ef4444"),
        ("P-value",f"{m['p_val']:.4f}","✅ sig." if m["sig"] else "⚠️ non sig.","#10b981" if m["sig"] else "#f59e0b"),
    ]): col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.4rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)

    section("DISTRIBUZIONE FORWARD RETURN — CANONI vs BASELINE")
    yc=m["y_canoni"]; ya=m["y_baseline"]
    fig_d=go.Figure()
    fig_d.add_histogram(x=ya,nbinsx=80,marker_color="#4a5a75",opacity=0.45,histnorm="probability density",name="Baseline")
    fig_d.add_histogram(x=yc,nbinsx=60,marker_color="#8b5cf6",opacity=0.85,histnorm="probability density",name=f"Canoni (n={len(yc)})")
    fig_d.add_vline(x=float(yc.mean()),line=dict(color="#a78bfa",dash="dash",width=2),annotation_text=f"Canoni μ={yc.mean():.2f}%",annotation_font_color="#a78bfa",annotation_font_size=10)
    fig_d.add_vline(x=float(ya.mean()),line=dict(color="#4a5a75",dash="dot",width=1.5),annotation_text=f"Baseline μ={ya.mean():.2f}%",annotation_font_color="#4a5a75",annotation_font_size=10)
    fig_d.add_vline(x=0,line=dict(color="#8a9ab5",width=0.8))
    fig_d.update_layout(**PLOTLY_DARK,height=270,xaxis_title=f"Forward Return H={H_sel}gg (%)",yaxis_title="Densità",barmode="overlay",title=dict(text=""))
    st.plotly_chart(fig_d,use_container_width=True)

    cl,cr=st.columns(2)
    with cl:
        section("PROFILO RENDIMENTO H=1→MAX")
        hh=sorted(an.keys()); cms=[an[H]["avg_ret"]*100 for H in hh]; bms=[an[H]["avg_baseline"]*100 for H in hh]
        fig_p=go.Figure()
        fig_p.add_scatter(x=hh,y=bms,mode="lines+markers",line=dict(color="#4a5a75",dash="dash",width=1.5),marker=dict(size=5),name="Baseline")
        fig_p.add_scatter(x=hh,y=cms,mode="lines+markers",line=dict(color="#8b5cf6",width=2.5),marker=dict(size=7),fill="tonexty",fillcolor="rgba(139,92,246,0.08)",name="Canoni")
        if H_sel in hh: idx=hh.index(H_sel); fig_p.add_scatter(x=[H_sel],y=[cms[idx]],mode="markers",marker=dict(color="#f59e0b",size=12,symbol="star"),name=f"H={H_sel}")
        fig_p.add_hline(y=0,line=dict(color="#4a5a75",width=0.8))
        fig_p.update_layout(**PLOTLY_DARK,height=240,xaxis_title="H (giorni)",yaxis_title="Avg Return (%)",title=dict(text=""))
        st.plotly_chart(fig_p,use_container_width=True)
    with cr:
        section("LIFT E HIT RATE PER ORIZZONTE")
        lv=[an[H]["lift"] for H in hh]; hv=[an[H]["hit_rate"] for H in hh]
        fig_l=make_subplots(specs=[[{"secondary_y":True}]])
        fig_l.add_bar(x=[f"H={H}" for H in hh],y=lv,marker_color=["#10b981" if v>1 else "#ef4444" for v in lv],opacity=0.8,name="Lift",secondary_y=False)
        fig_l.add_scatter(x=[f"H={H}" for H in hh],y=[v*100 for v in hv],mode="lines+markers",line=dict(color="#f59e0b",width=2),marker=dict(size=8),name="Hit Rate (%)",secondary_y=True)
        fig_l.add_hline(y=1.0,line=dict(color="#4a5a75",dash="dash",width=1),secondary_y=False)
        fig_l.add_hline(y=50,line=dict(color="#4a5a75",dash="dot",width=1),secondary_y=True)
        fig_l.update_layout(**PLOTLY_DARK,height=240,title=dict(text=""))
        fig_l.update_yaxes(title_text="Lift",gridcolor="#1e2d47",secondary_y=False)
        fig_l.update_yaxes(title_text="Hit Rate (%)",gridcolor="rgba(0,0,0,0)",secondary_y=True)
        st.plotly_chart(fig_l,use_container_width=True)

    cl2,cr2=st.columns(2)
    with cl2:
        section("STABILITÀ TEMPORALE — LIFT ANNUALE")
        ly=bd.get("lift_yr",pd.Series(dtype=float)); ny=bd.get("n_yr",pd.Series(dtype=int))
        if len(ly)>0:
            fig_y=make_subplots(specs=[[{"secondary_y":True}]])
            fig_y.add_bar(x=ly.index.astype(str),y=ly.values,marker_color=["#10b981" if v>=1 else "#ef4444" for v in ly.values],opacity=0.8,name="Lift",secondary_y=False)
            fig_y.add_scatter(x=ny.reindex(ly.index).index.astype(str),y=ny.reindex(ly.index).values,mode="lines+markers",line=dict(color="#f59e0b",width=1.5),marker=dict(size=5),name="N ep.",secondary_y=True)
            fig_y.add_hline(y=1.0,line=dict(color="#4a5a75",dash="dash",width=1))
            fig_y.update_layout(**PLOTLY_DARK,height=240,title=dict(text=""))
            fig_y.update_yaxes(title_text="Lift",gridcolor="#1e2d47",secondary_y=False)
            fig_y.update_yaxes(title_text="N episodi",gridcolor="rgba(0,0,0,0)",secondary_y=True)
            st.plotly_chart(fig_y,use_container_width=True)
    with cr2:
        section("TOP TICKER PER AVG RETURN")
        ptk=bd.get("per_ticker",pd.DataFrame())
        if not ptk.empty:
            top=ptk.head(12).copy(); top["tk"]=top["ticker"].str.replace(".MI","",regex=False)
            fig_t=go.Figure()
            fig_t.add_bar(x=top["tk"],y=top["avg_ret"]*100,marker_color=["#10b981" if v>0 else "#ef4444" for v in top["avg_ret"]],opacity=0.85,text=[f"{v:.1f}%" for v in top["avg_ret"]*100],textposition="outside",textfont=dict(size=10,color="#8a9ab5"))
            fig_t.add_hline(y=0,line=dict(color="#4a5a75",width=0.8))
            fig_t.update_layout(**PLOTLY_DARK,height=240,yaxis_title="Avg Ret (%)",showlegend=False,title=dict(text=""),xaxis=dict(tickangle=-35,**PLOTLY_DARK["xaxis"]))
            st.plotly_chart(fig_t,use_container_width=True)

    section("TABELLA METRICHE COMPLETA")
    rows=[{"H":H,"N canoni":s["n_canoni"],"Avg Ret":f"{s['avg_ret']*100:.3f}%","Baseline":f"{s['avg_baseline']*100:.3f}%","Lift":f"{s['lift']:.3f}x","Hit Rate":f"{s['hit_rate']:.1%}","T-stat":f"{s['t_stat']:.2f}","P-val":f"{s['p_val']:.4f}","Sig.":"✅" if s["sig"] else "⚠️"} for H,s in an.items()]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    section("ULTIMI EPISODI CANONE")
    canoni=get_canoni(df_p,pat_sel)
    if not canoni.empty:
        fwdcol=bc.get("fwd_col",f"fwd_ret_t{H_sel}")
        sc=["date","ticker","close",f"score_{pat_sel}",fwdcol]
        sc=[c for c in sc if c in canoni.columns]
        le=canoni[sc].copy().sort_values("date",ascending=False).head(25)
        le["date"]=pd.to_datetime(le["date"]).dt.strftime("%Y-%m-%d")
        le["ticker"]=le["ticker"].str.replace(".MI","",regex=False)
        if fwdcol in le.columns: le[fwdcol]=(le[fwdcol]*100).round(2).astype(str)+"%"
        st.dataframe(le,use_container_width=True,hide_index=True,height=340)

    if len(st.session_state.blinded_configs)>1:
        section("RIEPILOGO TUTTI I PATTERN BLINDATI")
        tbl=build_final_pattern_table(st.session_state.blinded_configs)
        if not tbl.empty: st.dataframe(tbl,use_container_width=True,hide_index=True)

    st.markdown("<div class='disclaimer'>⚠️ Parametri scelti prima di osservare i forward return (blindatura). Evidenze statistiche su dati storici. Non costituiscono raccomandazioni di investimento (MiFID II).</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — EXPORT
# ══════════════════════════════════════════════════════════
elif "Export" in page:
    top_bar("Export","Download risultati")
    if not st.session_state.blinded_configs: st.warning("⚠️ Blinda almeno un pattern"); st.stop()
    ca,cb=st.columns(2)
    with ca:
        section("REPORT EXCEL")
        out=io.BytesIO()
        with pd.ExcelWriter(out,engine="openpyxl") as w:
            tbl=build_final_pattern_table(st.session_state.blinded_configs)
            if not tbl.empty: tbl.to_excel(w,sheet_name="Pattern Summary",index=False)
            for pid,bc in st.session_state.blinded_configs.items():
                df_p=st.session_state.pattern_dfs.get(pid)
                if df_p is None: continue
                canoni=get_canoni(df_p,pid)
                if canoni.empty: continue
                fwda=[c for c in df_p.columns if c.startswith("fwd_ret_t")]
                cols=["date","ticker","close",f"score_{pid}"]+fwda
                cols=[c for c in cols if c in canoni.columns]
                ep=canoni[cols].copy(); ep["date"]=ep["date"].dt.strftime("%Y-%m-%d")
                ep.sort_values("date",ascending=False).to_excel(w,sheet_name=PATTERN_LABELS[pid][:28],index=False)
            for pid,an in st.session_state.analysis.items():
                rows=[{"Pattern":PATTERN_LABELS[pid],"H":H,"N":s["n_canoni"],"Avg Ret":s["avg_ret"],"Lift":s["lift"],"Hit Rate":s["hit_rate"],"P-val":s["p_val"],"Sig.":s["sig"]} for H,s in an.items()]
                if rows: pd.DataFrame(rows).to_excel(w,sheet_name=f"Metriche_{pid[:15]}",index=False)
            rows_s=[{"Pattern":PATTERN_LABELS[pid],"N Canoni":bc.get("n_canoni"),"Lift":bc.get("lift"),"Avg Ret":bc.get("avg_ret"),"Params":str(bc.get("params",{})),"Fwd":bc.get("fwd_col",""),"Blindato":bc.get("blinded_at","")} for pid,bc in st.session_state.blinded_configs.items()]
            pd.DataFrame(rows_s).to_excel(w,sheet_name="Sommario",index=False)
        out.seek(0)
        st.download_button("📥 Download Excel",data=out,file_name=f"PatternDiscovery_{datetime.now().strftime('%Y%m%d')}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    with cb:
        section("CSV EPISODI")
        pid_csv=st.selectbox("Pattern",list(st.session_state.blinded_configs.keys()),format_func=lambda x:PATTERN_LABELS[x],label_visibility="collapsed")
        df_p_csv=st.session_state.pattern_dfs.get(pid_csv)
        if df_p_csv is not None:
            can=get_canoni(df_p_csv,pid_csv)
            fwda=[c for c in df_p_csv.columns if c.startswith("fwd_ret_t")]
            cols=["date","ticker","close",f"score_{pid_csv}"]+fwda
            cols=[c for c in cols if c in can.columns]
            csv_out=can[cols].copy(); csv_out["date"]=csv_out["date"].dt.strftime("%Y-%m-%d")
            csv_out["ticker"]=csv_out["ticker"].str.replace(".MI","",regex=False)
            st.download_button("📥 Download CSV",data=csv_out.sort_values("date",ascending=False).to_csv(index=False),file_name=f"{pid_csv}_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv",use_container_width=True)
            section("ANTEPRIMA")
            st.dataframe(csv_out.head(15),use_container_width=True,hide_index=True,height=290)

