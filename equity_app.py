"""
╔══════════════════════════════════════════════════════════════╗
║  EQUITY PATTERN DISCOVERY — Tool Interattivo                 ║
║  Framework Analisi Quantitativa — Azionario Italia           ║
╚══════════════════════════════════════════════════════════════╝

Installazione:
    pip install streamlit plotly pandas numpy scipy yfinance openpyxl

Avvio:
    streamlit run equity_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as spstats
import yfinance as yf
import warnings
import io
from datetime import datetime, date

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Equity Pattern Discovery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg-primary:#0a0e17; --bg-secondary:#111827; --bg-card:#151d2e;
    --border:#1e2d47; --border-bright:#2a3f63;
    --text-primary:#e8edf5; --text-secondary:#8a9ab5; --text-dim:#4a5a75;
    --accent-blue:#3b82f6; --accent-cyan:#06b6d4; --accent-green:#10b981;
    --accent-red:#ef4444; --accent-amber:#f59e0b; --accent-purple:#8b5cf6;
}
.stApp { background: var(--bg-primary); color: var(--text-primary); }
.main .block-container { padding: 1.5rem 2rem; max-width: 1600px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stRadio label { color: var(--text-secondary) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; color: var(--text-primary) !important; letter-spacing: -0.02em; }
p,div,span,label { font-family: 'JetBrains Mono', monospace; }
.kpi-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem 1.4rem; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan)); }
.kpi-label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; line-height: 1; margin-bottom: 0.3rem; }
.kpi-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-secondary); }
.section-header { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--accent-cyan); text-transform: uppercase; letter-spacing: 0.2em; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }
.step-card { background: var(--bg-card); border: 1px solid var(--border); border-left: 3px solid var(--accent-purple); border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.step-card h4 { font-family: 'Syne', sans-serif !important; font-size: 1rem; color: var(--accent-cyan) !important; margin-bottom: 0.6rem; }
.step-card p { font-size: 0.78rem; color: var(--text-secondary); line-height: 1.6; margin: 0; }
.top-bar { display: flex; align-items: center; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.top-bar-title { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em; }
.top-bar-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dim); }
.disclaimer { background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.2); border-left: 3px solid #f59e0b; border-radius: 4px; padding: 0.8rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-secondary); margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Costanti ──────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "ENI.MI","ENEL.MI","ISP.MI","UCG.MI","TIT.MI","STLAM.MI",
    "MB.MI","PRY.MI","SRG.MI","AMP.MI","G.MI","SPM.MI",
    "BMED.MI","BPSO.MI","CPR.MI","CNHI.MI","ERG.MI","FBK.MI",
    "HER.MI","LDO.MI","MONC.MI","RACE.MI","REC.MI","SAP.MI",
    "TRN.MI","A2A.MI","IREN.MI","BZU.MI","PST.MI","INW.MI",
]
HORIZONS = [3, 5, 10, 15, 20]
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(21,29,46,0.6)",
    font=dict(family="JetBrains Mono", color="#8a9ab5", size=11),
    xaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d47", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=40),
)

# ── Session state ─────────────────────────────────────────────
for k, v in {
    "ohlcv":None,"features":None,"pat_df":None,"canoni":None,
    "stats":None,"pat_params":None,"n_tickers_ok":0,
    "start_date":"2018-01-01","end_date":datetime.today().strftime("%Y-%m-%d"),
    "step_data":False,"step_feat":False,"step_pattern":False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Pipeline functions ────────────────────────────────────────

def download_ohlcv(tickers, start, end):
    frames = []
    bar = st.progress(0, text="Download in corso...")
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df = df[cols].copy()
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            df = df.reset_index()
            df["ticker"] = t
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"])
            df["_r"] = df["close"].pct_change()
            df = df[df["_r"].abs() < 1.0].drop(columns=["_r"])
            if len(df) >= 60:
                frames.append(df)
        except Exception:
            pass
        bar.progress((i+1)/len(tickers), text=f"{i+1}/{len(tickers)}: {t}")
    bar.empty()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker","date"])


def build_features(ohlcv, h_max=20):
    df = ohlcv.copy().sort_values(["ticker","date"])
    g  = df.groupby("ticker")
    df["prev_close"]  = g["close"].transform(lambda x: x.shift(1))
    df["ret_1d"]      = g["close"].transform(lambda x: x.pct_change())
    df["delta_close"] = df["close"] - df["prev_close"]
    if all(c in df.columns for c in ["open","high","low","close"]):
        df["range_day"]          = df["high"] - df["low"]
        df["body"]               = df["close"] - df["open"]
        df["body_abs"]           = df["body"].abs()
        df["lower_shadow"]       = df[["open","close"]].min(axis=1) - df["low"]
        df["upper_shadow"]       = df["high"] - df[["open","close"]].max(axis=1)
        df["close_pos_in_range"] = np.where(df["range_day"]>0, (df["close"]-df["low"])/df["range_day"], 0.5)
        df["body_range_ratio"]   = np.where(df["range_day"]>0, df["body_abs"]/df["range_day"], 0.0)
        df["gap_abs"]  = df["open"] - df["prev_close"]
        df["gap_pct"]  = np.where(df["prev_close"]>0, df["gap_abs"]/df["prev_close"], 0.0)
        df["true_range"] = np.maximum(df["high"]-df["low"],
                           np.maximum((df["high"]-df["prev_close"]).abs(),
                                      (df["low"] -df["prev_close"]).abs()))
    if "volume" in df.columns:
        df["vol_ratio"] = g["volume"].transform(lambda x: x / x.rolling(20, min_periods=10).mean())
    for H in range(1, h_max+1):
        df[f"fwd_ret_t{H}"] = g["close"].transform(lambda x: (x.shift(-H)/x - 1).clip(-0.5, 0.5))
    return df.dropna(subset=["ret_1d"])


def detect_shock_down(features, params):
    W, Wz, zsog, cp, min_gap = params["W"], params["Wz"], params["zsog"], params["cp"], params["min_gap"]
    results = []
    for ticker, grp in features.groupby("ticker"):
        g = grp.copy().sort_values("date").reset_index(drop=True)
        cum_ret = g["close"].pct_change(W)
        mu  = cum_ret.rolling(Wz, min_periods=60).mean()
        sig = cum_ret.rolling(Wz, min_periods=60).std()
        z   = (cum_ret - mu) / sig.replace(0, np.nan)
        cand = (z < zsog).astype(int)
        if cp < 1.0 and "close_pos_in_range" in g.columns:
            cand = cand & (g["close_pos_in_range"] < cp)
        g["z_score_mr"] = z
        g["is_cand"]    = cand.fillna(0).astype(int)
        canone, last_ep = [], -999
        for i in range(len(g)):
            if g.loc[i,"is_cand"]==1 and (i-last_ep)>=min_gap:
                canone.append(1); last_ep = i
            else:
                canone.append(0)
        g["is_canone_shock_down"] = canone
        results.append(g)
    return pd.concat(results, ignore_index=True)


def compute_stats(pat_df, canoni, horizons=HORIZONS):
    out = {}
    for H in horizons:
        fc = f"fwd_ret_t{H}"
        if fc not in pat_df.columns:
            continue
        yc = canoni[fc].dropna()
        ya = pat_df[fc].dropna()
        if len(yc) < 5:
            continue
        avg_ret = float(yc.mean())
        avg_base = float(ya.mean())
        hr  = float((yc>0).mean())
        lift = avg_ret/avg_base if avg_base!=0 else np.nan
        t, p = spstats.ttest_1samp(yc, 0)
        per_tk = []
        for tk, grp in pat_df.groupby("ticker"):
            c = grp[grp["is_canone_shock_down"]==1][fc].dropna()
            if len(c)>=3:
                per_tk.append({"ticker":tk.replace(".MI",""),"n":len(c),"avg_ret":float(c.mean()),"hit_rate":float((c>0).mean())})
        per_tk_df = pd.DataFrame(per_tk).sort_values("avg_ret",ascending=False) if per_tk else pd.DataFrame()
        tmp = canoni.copy(); tmp["year"] = tmp["date"].dt.year
        at  = pat_df.copy();  at["year"]  = at["date"].dt.year
        yr_c = tmp.groupby("year")[fc].mean()
        yr_b = at.groupby("year")[fc].mean()
        lift_yr = (yr_c/yr_b).replace([np.inf,-np.inf],np.nan).dropna()
        out[H] = {"n_canoni":len(yc),"avg_ret":avg_ret,"avg_baseline":avg_base,
                  "hit_rate":hr,"lift":lift,"t_stat":float(t),"p_val":float(p),
                  "sig":float(p)<0.10,"per_ticker":per_tk_df,
                  "lift_yr":lift_yr,"n_yr":tmp.groupby("year").size()}
    return out

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#e8edf5;'>📊 Equity Patterns</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#4a5a75;margin-top:4px;text-transform:uppercase;letter-spacing:0.1em;'>Pattern Discovery — Italia</div>
    </div>""", unsafe_allow_html=True)

    def si(done): return "🟢" if done else "⚪"
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#4a5a75;margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.1em;'>Avanzamento</div>
    <div style='font-size:0.78rem;color:#8a9ab5;line-height:2;'>
        {si(st.session_state.step_data)}    1. Universo & Dati<br>
        {si(st.session_state.step_feat)}    2. Feature Engineering<br>
        {si(st.session_state.step_pattern)} 3. Mean Reversion<br>
        {si(st.session_state.stats is not None)} 4. Risultati
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d47;margin:1rem 0;'>", unsafe_allow_html=True)

    page = st.radio("nav", [
        "📥  Universo & Dati",
        "⚙️  Feature Engineering",
        "📉  Mean Reversion",
        "📋  Risultati",
        "📤  Export",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2d47;margin:1rem 0;'>", unsafe_allow_html=True)

    if st.button("🔄 Reset progetto", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.step_data:
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#4a5a75;margin-top:0.5rem;'>
            Ticker: <span style='color:#8a9ab5;'>{st.session_state.n_tickers_ok}</span><br>
            Periodo: <span style='color:#8a9ab5;'>{st.session_state.start_date[:7]} → {st.session_state.end_date[:7]}</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — UNIVERSO & DATI
# ═══════════════════════════════════════════════════════════════
if "Universo" in page:
    st.markdown('<div class="top-bar"><div class="top-bar-title">Universo & Dati</div><div class="top-bar-meta">Step 1 di 4 — Download e pulizia OHLCV</div></div>', unsafe_allow_html=True)

    st.markdown("""<div class='step-card'><h4>📋 Configurazione universo</h4><p>
    Seleziona i ticker da analizzare e il periodo storico. Dati OHLCV giornalieri da Yahoo Finance
    (auto-adjusted per dividendi e split). Pulizia automatica: rimozione variazioni >100% giornaliere.
    </p></div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([2,1])
    with col_a:
        st.markdown("<div class='section-header'>TICKER</div>", unsafe_allow_html=True)
        mode = st.radio("Input", ["Lista predefinita","Inserimento manuale","Upload CSV"], horizontal=True, label_visibility="collapsed")
        if mode == "Lista predefinita":
            sel = st.multiselect("Ticker", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:15], label_visibility="collapsed")
            tickers_final = sel
        elif mode == "Inserimento manuale":
            raw = st.text_area("es. ENI.MI, ENEL.MI", height=80, label_visibility="collapsed", placeholder="ENI.MI, ENEL.MI, ISP.MI...")
            tickers_final = [t.strip().upper() for t in raw.replace("\n",",").split(",") if t.strip()]
        else:
            up = st.file_uploader("CSV con colonna ticker", type=["csv"])
            if up:
                dfu = pd.read_csv(up)
                tickers_final = dfu["ticker"].tolist() if "ticker" in dfu.columns else []
            else:
                tickers_final = []
        st.caption(f"{len(tickers_final)} ticker selezionati")

    with col_b:
        st.markdown("<div class='section-header'>PERIODO</div>", unsafe_allow_html=True)
        start_d = st.date_input("Inizio", value=date(2018,1,1), min_value=date(2010,1,1), max_value=date.today(), label_visibility="collapsed")
        end_d   = st.date_input("Fine",   value=date.today(),   min_value=date(2010,1,1), max_value=date.today(), label_visibility="collapsed")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("▶️ Scarica dati OHLCV", disabled=len(tickers_final)==0):
        with st.spinner(""):
            ohlcv = download_ohlcv(tickers_final, str(start_d), str(end_d))
        if ohlcv.empty:
            st.error("Nessun dato. Verifica ticker e connessione.")
        else:
            st.session_state.update({
                "ohlcv":ohlcv,"step_data":True,"step_feat":False,
                "step_pattern":False,"stats":None,
                "n_tickers_ok":ohlcv["ticker"].nunique(),
                "start_date":str(start_d),"end_date":str(end_d),
            })
            st.rerun()

    if st.session_state.step_data and st.session_state.ohlcv is not None:
        ohlcv = st.session_state.ohlcv
        st.markdown("<div class='section-header'>RIEPILOGO</div>", unsafe_allow_html=True)
        k1,k2,k3,k4 = st.columns(4)
        for col,(lbl,val,sub,col_) in zip([k1,k2,k3,k4],[
            ("Ticker caricati",str(ohlcv["ticker"].nunique()),"con ≥60 sessioni","#10b981"),
            ("Righe totali",f"{len(ohlcv):,}","osservazioni OHLCV","#3b82f6"),
            ("Periodo effettivo",ohlcv["date"].min().strftime("%b %Y"),f"→ {ohlcv['date'].max().strftime('%b %Y')}","#8b5cf6"),
            ("Obs/ticker",f"{len(ohlcv)//ohlcv['ticker'].nunique():.0f}","media per ticker","#06b6d4"),
        ]):
            with col:
                st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{col_};'>{val}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>ANTEPRIMA CANDLESTICK</div>", unsafe_allow_html=True)
        tk_opts = [t.replace(".MI","")+" ("+t+")" for t in ohlcv["ticker"].unique()[:20]]
        tk_sel  = st.selectbox("Ticker", tk_opts, label_visibility="collapsed")
        tk_code = tk_sel.split("(")[1].rstrip(")")
        sub_p   = ohlcv[ohlcv["ticker"]==tk_code].tail(60)
        fig_c   = go.Figure()
        fig_c.add_candlestick(
            x=sub_p["date"],
            open=sub_p.get("open", sub_p["close"]),
            high=sub_p.get("high", sub_p["close"]),
            low=sub_p.get("low",  sub_p["close"]),
            close=sub_p["close"],
            increasing_line_color="#10b981", decreasing_line_color="#ef4444"
        )
        fig_c.update_layout(**PLOTLY_DARK, height=200, yaxis_title="Prezzo (€)", title=dict(text=""))
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#10b981;'>✅ Dati pronti → Step 2 Feature Engineering</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
elif "Feature" in page:
    st.markdown('<div class="top-bar"><div class="top-bar-title">Feature Engineering</div><div class="top-bar-meta">Step 2 di 4 — Feature primitive e forward returns</div></div>', unsafe_allow_html=True)

    if not st.session_state.step_data:
        st.warning("⚠️ Prima completa Step 1 — Universo & Dati"); st.stop()

    st.markdown("""<div class='step-card'><h4>⚙️ Feature costruite automaticamente</h4><p>
    <b>Prezzo:</b> ret_1d, prev_close, delta_close &nbsp;|&nbsp;
    <b>Candle:</b> range_day, body, lower/upper_shadow, close_pos_in_range, body_range_ratio &nbsp;|&nbsp;
    <b>Gap:</b> gap_abs, gap_pct, true_range &nbsp;|&nbsp;
    <b>Volume:</b> vol_ratio (vs media 20gg) &nbsp;|&nbsp;
    <b>Forward returns:</b> fwd_ret_t1 → fwd_ret_tH_max (winsorizzati ±50%)
    </p></div>""", unsafe_allow_html=True)

    col_cfg, col_btn = st.columns([2,1])
    with col_cfg:
        h_max = st.slider("H_max — Orizzonte massimo forward return", 5, 20, 15, 5)
        st.caption(f"Costruirà fwd_ret_t1 ... fwd_ret_t{h_max} | Input: {len(st.session_state.ohlcv):,} righe")
    with col_btn:
        st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
        run_feat = st.button("▶️ Costruisci feature", use_container_width=True)

    if run_feat:
        with st.spinner("Costruzione feature in corso..."):
            feats = build_features(st.session_state.ohlcv, h_max=h_max)
        st.session_state.update({"features":feats,"step_feat":True,"step_pattern":False,"stats":None})
        st.rerun()

    if st.session_state.step_feat and st.session_state.features is not None:
        feats = st.session_state.features
        fwd_cols = [c for c in feats.columns if c.startswith("fwd_ret_t")]
        st.markdown("<div class='section-header'>RIEPILOGO</div>", unsafe_allow_html=True)
        k1,k2,k3,k4 = st.columns(4)
        for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4],[
            ("Righe dataset",f"{len(feats):,}","dopo pulizia","#10b981"),
            ("Feature primitive",str(len([c for c in feats.columns if not c.startswith("fwd_")])),"prezzo+candle+gap","#3b82f6"),
            ("Forward returns",str(len(fwd_cols)),f"H=1→{len(fwd_cols)}","#8b5cf6"),
            ("Ticker",str(feats["ticker"].nunique()),"con feature complete","#06b6d4"),
        ]):
            with col:
                st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};'>{val}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

        col_d, col_f = st.columns(2)
        with col_d:
            st.markdown("<div class='section-header'>DISTRIBUZIONE ret_1d</div>", unsafe_allow_html=True)
            rets = feats["ret_1d"].dropna()*100
            fig_r = go.Figure()
            fig_r.add_histogram(x=rets, nbinsx=100, marker_color="#3b82f6", opacity=0.8, histnorm="probability density")
            fig_r.add_vline(x=0, line=dict(color="#4a5a75", width=1))
            fig_r.update_layout(**PLOTLY_DARK, height=230, xaxis_title="ret_1d (%)", yaxis_title="Densità",
                                showlegend=False, title=dict(text=f"μ={rets.mean():.3f}%  σ={rets.std():.2f}%"))
            st.plotly_chart(fig_r, use_container_width=True)

        with col_f:
            st.markdown("<div class='section-header'>BASELINE DRIFT PER ORIZZONTE</div>", unsafe_allow_html=True)
            hh = list(range(1, len(fwd_cols)+1))
            means = [feats[f"fwd_ret_t{H}"].mean()*100 for H in hh if f"fwd_ret_t{H}" in feats.columns]
            fig_f = go.Figure()
            fig_f.add_scatter(x=hh, y=means, mode="lines+markers", line=dict(color="#8b5cf6", width=2), marker=dict(size=5))
            fig_f.add_hline(y=0, line=dict(color="#4a5a75", width=0.8))
            fig_f.update_layout(**PLOTLY_DARK, height=230, xaxis_title="H (giorni)", yaxis_title="Avg fwd ret (%)", showlegend=False, title=dict(text=""))
            st.plotly_chart(fig_f, use_container_width=True)

        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#10b981;'>✅ Feature pronte → Step 3 Mean Reversion</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MEAN REVERSION
# ═══════════════════════════════════════════════════════════════
elif "Mean Reversion" in page:
    st.markdown('<div class="top-bar"><div class="top-bar-title">Mean Reversion — Shock Down</div><div class="top-bar-meta">Step 3 di 4 — Parametri e identificazione episodi canone</div></div>', unsafe_allow_html=True)

    if not st.session_state.step_feat:
        st.warning("⚠️ Prima completa Step 2 — Feature Engineering"); st.stop()

    st.markdown("""<div class='step-card'><h4>📉 Logica del pattern Shock Down MR</h4><p>
    Il titolo registra un return cumulativo su <b>W giorni</b> con z-score inferiore alla soglia
    (shock negativo anomalo rispetto alla storia). Il filtro <b>close position</b> conferma
    la debolezza intraday. La <b>blindatura</b> garantisce che i parametri vengano scelti
    prima di osservare i forward return — elimina il look-ahead bias.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>PARAMETRI</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        W    = st.slider("W — Return cumulativo (gg)", 2, 20, 5)
        Wz   = st.slider("Wz — Z-score rolling (gg)", 60, 252, 120)
    with c2:
        zsog = st.slider("Z-score soglia", -4.0, -0.5, -2.0, 0.1)
        cp   = st.slider("Close position max", 0.1, 1.0, 0.35, 0.05)
    with c3:
        min_gap = st.slider("Min gap episodi (gg)", 3, 20, 5)

    st.markdown(f"""<div style='background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.2);border-radius:6px;padding:0.7rem 1rem;margin:0.8rem 0;font-family:JetBrains Mono,monospace;font-size:0.73rem;color:#a78bfa;'>
    cum_ret = pct_change({W}) &nbsp;|&nbsp; z = (cum_ret − μ{Wz}d) / σ{Wz}d &nbsp;|&nbsp; candidato: z &lt; {zsog} AND close_pos &lt; {cp} &nbsp;|&nbsp; min_gap = {min_gap}gg
    </div>""", unsafe_allow_html=True)

    if st.button("▶️ Identifica episodi canone", use_container_width=False):
        params = {"W":W,"Wz":Wz,"zsog":zsog,"cp":cp,"min_gap":min_gap}
        with st.spinner("Identificazione episodi..."):
            pat_df = detect_shock_down(st.session_state.features, params)
            canoni = pat_df[pat_df["is_canone_shock_down"]==1].copy()
        if len(canoni) < 10:
            st.error(f"Solo {len(canoni)} episodi. Prova ad allargare le soglie.")
        else:
            fwd_av = [H for H in HORIZONS if f"fwd_ret_t{H}" in pat_df.columns]
            with st.spinner("Calcolo metriche..."):
                sr = compute_stats(pat_df, canoni, horizons=fwd_av)
            st.session_state.update({"pat_df":pat_df,"canoni":canoni,"stats":sr,"pat_params":params,"step_pattern":True})
            st.rerun()

    if st.session_state.step_pattern and st.session_state.stats is not None:
        m5 = st.session_state.stats.get(5, st.session_state.stats[list(st.session_state.stats.keys())[0]])
        n_ep = len(st.session_state.canoni)
        st.markdown("<div class='section-header'>PREVIEW RISULTATI</div>", unsafe_allow_html=True)
        k1,k2,k3,k4 = st.columns(4)
        for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4],[
            ("Episodi canone",str(n_ep),f"su {st.session_state.pat_df['ticker'].nunique()} ticker","#8b5cf6"),
            ("Avg Ret H=5gg",f"{m5.get('avg_ret',0)*100:.2f}%",f"baseline: {m5.get('avg_baseline',0)*100:.2f}%","#10b981" if m5.get("avg_ret",0)>0 else "#ef4444"),
            ("Hit Rate H=5gg",f"{m5.get('hit_rate',0):.1%}","% episodi ret>0","#10b981" if m5.get("hit_rate",0)>0.5 else "#ef4444"),
            ("Lift H=5gg",f"{m5.get('lift',0):.2f}x",f"p={m5.get('p_val',1):.3f} {'✅' if m5.get('sig') else '⚠️'}","#10b981" if m5.get("lift",0)>1 else "#ef4444"),
        ]):
            with col:
                st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.5rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#10b981;margin-top:1rem;'>✅ Episodi identificati → Step 4 Risultati per analisi completa</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — RISULTATI
# ═══════════════════════════════════════════════════════════════
elif "Risultati" in page:
    st.markdown('<div class="top-bar"><div class="top-bar-title">Risultati — Analisi Statistica</div><div class="top-bar-meta">Step 4 di 4 — Distribuzione, lift, stabilità, breakdown ticker</div></div>', unsafe_allow_html=True)

    if st.session_state.stats is None:
        st.warning("⚠️ Prima completa Step 3 — Mean Reversion"); st.stop()

    sr     = st.session_state.stats
    canoni = st.session_state.canoni
    pat_df = st.session_state.pat_df
    H_list = list(sr.keys())

    H_sel = st.select_slider("Orizzonte", options=H_list, value=H_list[1] if len(H_list)>1 else H_list[0])
    m = sr[H_sel]

    k1,k2,k3,k4,k5 = st.columns(5)
    for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4,k5],[
        ("Episodi",str(m["n_canoni"]),f"{pat_df['ticker'].nunique()} ticker","#8b5cf6"),
        ("Avg Return",f"{m['avg_ret']*100:.2f}%",f"baseline: {m['avg_baseline']*100:.2f}%","#10b981" if m["avg_ret"]>0 else "#ef4444"),
        ("Hit Rate",f"{m['hit_rate']:.1%}","% episodi ret>0","#10b981" if m["hit_rate"]>0.5 else "#ef4444"),
        ("Lift",f"{m['lift']:.2f}x","vs baseline","#10b981" if m["lift"]>1 else "#ef4444"),
        ("P-value",f"{m['p_val']:.4f}","✅ sig." if m["sig"] else "⚠️ non sig.","#10b981" if m["sig"] else "#f59e0b"),
    ]):
        with col:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.5rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

    # Distribuzione
    st.markdown("<div class='section-header'>DISTRIBUZIONE FORWARD RETURN — CANONI vs BASELINE</div>", unsafe_allow_html=True)
    fc = f"fwd_ret_t{H_sel}"
    yc = canoni[fc].dropna()*100
    ya = pat_df[fc].dropna()*100
    fig_d = go.Figure()
    fig_d.add_histogram(x=ya, nbinsx=80, marker_color="#4a5a75", opacity=0.45, histnorm="probability density", name="Baseline")
    fig_d.add_histogram(x=yc, nbinsx=60, marker_color="#8b5cf6", opacity=0.85, histnorm="probability density", name=f"Canoni (n={len(yc)})")
    fig_d.add_vline(x=float(yc.mean()), line=dict(color="#a78bfa", dash="dash", width=2),
                   annotation_text=f"Canoni μ={yc.mean():.2f}%", annotation_font_color="#a78bfa", annotation_font_size=10)
    fig_d.add_vline(x=float(ya.mean()), line=dict(color="#4a5a75", dash="dot", width=1.5),
                   annotation_text=f"Baseline μ={ya.mean():.2f}%", annotation_font_color="#4a5a75", annotation_font_size=10)
    fig_d.add_vline(x=0, line=dict(color="#8a9ab5", width=0.8))
    fig_d.update_layout(**PLOTLY_DARK, height=280, xaxis_title=f"Forward Return H={H_sel}gg (%)", yaxis_title="Densità", barmode="overlay", title=dict(text=""))
    st.plotly_chart(fig_d, use_container_width=True)

    # Profilo + Lift
    cl, cr = st.columns(2)
    with cl:
        st.markdown("<div class='section-header'>PROFILO RENDIMENTO H=1→MAX</div>", unsafe_allow_html=True)
        hh, cm, bm = [], [], []
        for H in range(1, max(H_list)+1):
            if f"fwd_ret_t{H}" in pat_df.columns:
                hh.append(H); cm.append(canoni[f"fwd_ret_t{H}"].mean()*100); bm.append(pat_df[f"fwd_ret_t{H}"].mean()*100)
        fig_p = go.Figure()
        fig_p.add_scatter(x=hh, y=bm, mode="lines+markers", line=dict(color="#4a5a75", dash="dash", width=1.5), marker=dict(size=5), name="Baseline")
        fig_p.add_scatter(x=hh, y=cm, mode="lines+markers", line=dict(color="#8b5cf6", width=2.5), marker=dict(size=7), fill="tonexty", fillcolor="rgba(139,92,246,0.08)", name="Canoni")
        if H_sel in hh:
            idx = hh.index(H_sel)
            fig_p.add_scatter(x=[H_sel], y=[cm[idx]], mode="markers", marker=dict(color="#f59e0b", size=12, symbol="star"), name=f"H={H_sel}")
        fig_p.add_hline(y=0, line=dict(color="#4a5a75", width=0.8))
        fig_p.update_layout(**PLOTLY_DARK, height=250, xaxis_title="H (giorni)", yaxis_title="Avg Return (%)", title=dict(text=""))
        st.plotly_chart(fig_p, use_container_width=True)

    with cr:
        st.markdown("<div class='section-header'>LIFT E HIT RATE PER ORIZZONTE</div>", unsafe_allow_html=True)
        lv = [sr[H]["lift"] for H in H_list]
        hv = [sr[H]["hit_rate"] for H in H_list]
        fig_l = make_subplots(specs=[[{"secondary_y":True}]])
        fig_l.add_bar(x=[f"H={H}" for H in H_list], y=lv, marker_color=["#10b981" if v>1 else "#ef4444" for v in lv], opacity=0.8, name="Lift", secondary_y=False)
        fig_l.add_scatter(x=[f"H={H}" for H in H_list], y=[v*100 for v in hv], mode="lines+markers", line=dict(color="#f59e0b", width=2), marker=dict(size=8), name="Hit Rate (%)", secondary_y=True)
        fig_l.add_hline(y=1.0, line=dict(color="#4a5a75", dash="dash", width=1), secondary_y=False)
        fig_l.add_hline(y=50, line=dict(color="#4a5a75", dash="dot", width=1), secondary_y=True)
        fig_l.update_layout(**PLOTLY_DARK, height=250, title=dict(text=""))
        fig_l.update_yaxes(title_text="Lift", gridcolor="#1e2d47", secondary_y=False)
        fig_l.update_yaxes(title_text="Hit Rate (%)", gridcolor="rgba(0,0,0,0)", secondary_y=True)
        st.plotly_chart(fig_l, use_container_width=True)

    # Stabilità temporale + Top ticker
    cl2, cr2 = st.columns(2)
    with cl2:
        st.markdown("<div class='section-header'>STABILITÀ TEMPORALE — LIFT ANNUALE</div>", unsafe_allow_html=True)
        ly = m.get("lift_yr", pd.Series(dtype=float))
        ny = m.get("n_yr",    pd.Series(dtype=int))
        if len(ly)>0:
            fig_y = make_subplots(specs=[[{"secondary_y":True}]])
            fig_y.add_bar(x=ly.index.astype(str), y=ly.values, marker_color=["#10b981" if v>=1 else "#ef4444" for v in ly.values], opacity=0.8, name="Lift", secondary_y=False)
            fig_y.add_scatter(x=ny.reindex(ly.index).index.astype(str), y=ny.reindex(ly.index).values, mode="lines+markers", line=dict(color="#f59e0b",width=1.5), marker=dict(size=5), name="N ep.", secondary_y=True)
            fig_y.add_hline(y=1.0, line=dict(color="#4a5a75", dash="dash", width=1))
            fig_y.update_layout(**PLOTLY_DARK, height=250, title=dict(text=""))
            fig_y.update_yaxes(title_text="Lift", gridcolor="#1e2d47", secondary_y=False)
            fig_y.update_yaxes(title_text="N episodi", gridcolor="rgba(0,0,0,0)", secondary_y=True)
            st.plotly_chart(fig_y, use_container_width=True)

    with cr2:
        st.markdown("<div class='section-header'>TOP TICKER PER AVG RETURN</div>", unsafe_allow_html=True)
        ptk = m.get("per_ticker", pd.DataFrame())
        if not ptk.empty:
            top = ptk.head(12)
            fig_t = go.Figure()
            fig_t.add_bar(x=top["ticker"], y=top["avg_ret"]*100, marker_color=["#10b981" if v>0 else "#ef4444" for v in top["avg_ret"]], opacity=0.85,
                          text=[f"{v:.1f}%" for v in top["avg_ret"]*100], textposition="outside", textfont=dict(size=10,color="#8a9ab5"))
            fig_t.add_hline(y=0, line=dict(color="#4a5a75", width=0.8))
            fig_t.update_layout(**PLOTLY_DARK, height=250, yaxis_title="Avg Ret (%)", showlegend=False, title=dict(text=""), xaxis=dict(tickangle=-35,**PLOTLY_DARK["xaxis"]))
            st.plotly_chart(fig_t, use_container_width=True)

    # Tabella completa
    st.markdown("<div class='section-header'>TABELLA METRICHE COMPLETA</div>", unsafe_allow_html=True)
    rows = [{"H":H,"N canoni":s["n_canoni"],"Avg Return":f"{s['avg_ret']*100:.3f}%","Baseline":f"{s['avg_baseline']*100:.3f}%",
             "Lift":f"{s['lift']:.3f}x","Hit Rate":f"{s['hit_rate']:.1%}","T-stat":f"{s['t_stat']:.2f}",
             "P-value":f"{s['p_val']:.4f}","Sig.":"✅" if s["sig"] else "⚠️"} for H,s in sr.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Ultimi episodi
    st.markdown("<div class='section-header'>ULTIMI EPISODI CANONE</div>", unsafe_allow_html=True)
    sc = ["date","ticker","close","z_score_mr",fc]
    sc = [c for c in sc if c in canoni.columns]
    le = canoni[sc].copy().sort_values("date",ascending=False).head(25)
    le["date"] = pd.to_datetime(le["date"]).dt.strftime("%Y-%m-%d")
    le["ticker"] = le["ticker"].str.replace(".MI","",regex=False)
    le["z_score_mr"] = le["z_score_mr"].round(2)
    le[fc] = (le[fc]*100).round(2).astype(str)+"%"
    le = le.rename(columns={"date":"Data","ticker":"Ticker","close":"Prezzo","z_score_mr":"Z-score",fc:f"Ret H={H_sel}gg"})
    st.dataframe(le, use_container_width=True, hide_index=True, height=380)

    st.markdown("""<div class='disclaimer'>
    ⚠️ I parametri sono stati scelti prima di osservare i forward return (blindatura anti-overfitting).
    I risultati mostrano evidenze statistiche su dati storici. Non costituiscono previsioni né
    raccomandazioni di investimento ai sensi di MiFID II. Strumento di supporto alla ricerca quantitativa.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — EXPORT
# ═══════════════════════════════════════════════════════════════
elif "Export" in page:
    st.markdown('<div class="top-bar"><div class="top-bar-title">Export</div><div class="top-bar-meta">Scarica i risultati</div></div>', unsafe_allow_html=True)

    if st.session_state.stats is None:
        st.warning("⚠️ Completa prima l'analisi (Step 1→4)"); st.stop()

    sr     = st.session_state.stats
    canoni = st.session_state.canoni
    pat_df = st.session_state.pat_df
    params = st.session_state.pat_params
    H_list = list(sr.keys())
    m5     = sr.get(5, sr[H_list[0]])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-header'>REPORT EXCEL</div>", unsafe_allow_html=True)
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            fcs = [f"fwd_ret_t{H}" for H in H_list if f"fwd_ret_t{H}" in canoni.columns]
            ep  = canoni[["date","ticker","close","z_score_mr"]+fcs].copy()
            ep["date"] = ep["date"].dt.strftime("%Y-%m-%d")
            ep.sort_values("date",ascending=False).to_excel(w, sheet_name="Episodi Canone", index=False)
            rows = [{"H":H,"N canoni":s["n_canoni"],"Avg Ret":s["avg_ret"],"Baseline":s["avg_baseline"],"Lift":s["lift"],"Hit Rate":s["hit_rate"],"T-stat":s["t_stat"],"P-value":s["p_val"],"Sig.":s["sig"]} for H,s in sr.items()]
            pd.DataFrame(rows).to_excel(w, sheet_name="Metriche", index=False)
            ptk = sr.get(5,{}).get("per_ticker",pd.DataFrame())
            if not ptk.empty:
                ptk.to_excel(w, sheet_name="Per Ticker", index=False)
            pd.DataFrame([{"Param":k,"Val":v} for k,v in params.items()]).to_excel(w, sheet_name="Parametri", index=False)
            pd.DataFrame([
                ["Pattern","Shock Down Mean Reversion"],
                ["Data",datetime.now().strftime("%Y-%m-%d %H:%M")],
                ["Periodo",f"{st.session_state.start_date} → {st.session_state.end_date}"],
                ["Ticker",pat_df["ticker"].nunique()],["Episodi",len(canoni)],
                ["Avg Ret H=5",f"{m5['avg_ret']*100:.3f}%"],["Lift H=5",f"{m5['lift']:.3f}x"],
                ["P-value H=5",f"{m5['p_val']:.4f}"],["Disclaimer","Supporto ricerca. Non consulenza MiFID II."],
            ], columns=["K","V"]).to_excel(w, sheet_name="Sommario", index=False)
        out.seek(0)
        st.download_button("📥 Download Excel", data=out,
                           file_name=f"PatternDiscovery_{datetime.now().strftime('%Y%m%d')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>CSV EPISODI</div>", unsafe_allow_html=True)
        fcs2 = [f"fwd_ret_t{H}" for H in H_list if f"fwd_ret_t{H}" in canoni.columns]
        csv  = canoni[["date","ticker","close","z_score_mr"]+fcs2].copy()
        csv["date"] = csv["date"].dt.strftime("%Y-%m-%d")
        st.download_button("📥 Download CSV", data=csv.sort_values("date",ascending=False).to_csv(index=False),
                           file_name=f"Episodi_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv", use_container_width=True)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>ANTEPRIMA</div>", unsafe_allow_html=True)
        prev = csv.head(12).copy()
        prev["ticker"] = prev["ticker"].str.replace(".MI","",regex=False)
        st.dataframe(prev, use_container_width=True, hide_index=True, height=300)
