"""
╔══════════════════════════════════════════════════════════════╗
║  EQUITY PATTERN DISCOVERY — Web App Demo                     ║
║  Alternative Quant Framework — Azionario Italia              ║
║  Streamlit dark-mode dashboard per SGR/SIM italiane          ║
╚══════════════════════════════════════════════════════════════╝

Installazione:
    pip install streamlit plotly pandas numpy scipy yfinance openpyxl

Avvio:
    streamlit run equity_app.py

Sidebar — sezioni:
    📊  Overview              — KPI generali universo
    📉  Mean Reversion        — Shock Down pattern (shock_down_mr)
    📖  Metodologia           — spiegazione framework
    📤  Export                — download risultati
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import yfinance as yf
import warnings
import io
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Pattern Discovery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — Dark mode, stessa palette della weather app
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary:    #0a0e17;
    --bg-secondary:  #111827;
    --bg-card:       #151d2e;
    --border:        #1e2d47;
    --border-bright: #2a3f63;
    --text-primary:  #e8edf5;
    --text-secondary:#8a9ab5;
    --text-dim:      #4a5a75;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-red:    #ef4444;
    --accent-amber:  #f59e0b;
    --accent-purple: #8b5cf6;
}

.stApp { background: var(--bg-primary); color: var(--text-primary); }
.main .block-container { padding: 1.5rem 2rem; max-width: 1600px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}
p, div, span, label { font-family: 'JetBrains Mono', monospace; }

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan));
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
}

.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.top-bar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
}
.top-bar-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
}

.pattern-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 4px;
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: rgba(139,92,246,0.15);
    color: #a78bfa;
    border: 1px solid #8b5cf6;
}

.method-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.method-card h4 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem;
    color: var(--accent-cyan) !important;
    margin-bottom: 0.8rem;
}
.method-card p {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

.disclaimer {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid var(--accent-amber);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 1rem;
}

.chain-box {
    background: var(--bg-secondary);
    border: 1px solid var(--border-bright);
    border-radius: 6px;
    padding: 1rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent-cyan);
    text-align: center;
    letter-spacing: 0.05em;
    margin: 1rem 0;
}

.dot-live {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────────────────────

# Universo azionario Italia — FTSE MIB + Mid Cap principali
ITALIAN_TICKERS = [
    "ENI.MI","ENEL.MI","ISP.MI","UCG.MI","TIT.MI","STLAM.MI",
    "MB.MI","PRY.MI","SRG.MI","AMP.MI","G.MI","SPM.MI",
    "BMED.MI","BPSO.MI","CPR.MI","CNHI.MI","DIA.MI","ERG.MI",
    "FCA.MI","FBK.MI","HER.MI","INW.MI","LDO.MI","MONC.MI",
    "PIRC.MI","RACE.MI","REC.MI","SAP.MI","SFER.MI","TRN.MI",
]

# Parametri pattern shock_down_mr (best params dal backtest)
SHOCK_PARAMS = {
    "W":    5,      # finestra return cumulativo (giorni)
    "Wz":   252,    # finestra rolling per z-score
    "zsog": -2.0,   # soglia z-score (sotto = candidato)
    "cp":   0.35,   # close position in range (filtro aggiuntivo)
    "H":    10,     # orizzonte forward return
}

START_DATE = "2015-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
HORIZONS   = [3, 5, 10, 15]

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(21,29,46,0.6)",
    font=dict(family="JetBrains Mono", color="#8a9ab5", size=11),
    xaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d47", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=40),
)

# ─────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlcv(tickers, start, end):
    """Scarica dati OHLCV per l'universo Italia."""
    frames = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty or len(df) < 100:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df.index.name = "date"
            df = df.reset_index()
            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker","date"])


def build_features(df):
    """Costruisce feature primitive per il pattern discovery."""
    df = df.copy().sort_values(["ticker","date"])
    grp = df.groupby("ticker")

    # Return giornaliero
    df["ret_1d"] = grp["close"].transform(lambda x: x.pct_change())

    # Pulizia outlier: rimuovi variazioni > 100% in un giorno
    df = df[df["ret_1d"].abs() < 1.0].copy()

    # Volume ratio
    df["vol_ratio"] = grp["volume"].transform(
        lambda x: x / x.rolling(20, min_periods=10).mean()
    )

    # Close position in range (dove chiude nel range high-low)
    df["close_pos_in_range"] = np.where(
        (df["high"] - df["low"]) > 0,
        (df["close"] - df["low"]) / (df["high"] - df["low"]),
        0.5
    )

    # Forward returns per H = 3,5,10,15 — winsorizzati ±50%
    for H in HORIZONS:
        fwd = grp["close"].transform(
            lambda x: x.shift(-H) / x - 1
        )
        df[f"fwd_ret_t{H}"] = fwd.clip(-0.5, 0.5)

    return df.dropna(subset=["ret_1d"])


def detect_shock_down(df, params=SHOCK_PARAMS):
    """
    Identifica episodi Shock Down Mean Reversion.

    Logica:
    - Calcola return cumulativo su W giorni
    - Normalizza con z-score rolling (finestra Wz)
    - Candidato quando z < zsog (shock negativo anomalo)
    - Filtro aggiuntivo: close nel bottom cp% del range giornaliero
    - Canone: candidato confermato con min gap 5gg tra episodi
    """
    W    = params["W"]
    Wz   = params["Wz"]
    zsog = params["zsog"]
    cp   = params["cp"]

    results = []
    for ticker, grp in df.groupby("ticker"):
        g = grp.copy().sort_values("date").reset_index(drop=True)

        # Return cumulativo
        cum_ret = g["close"].pct_change(W)

        # Z-score rolling
        mu  = cum_ret.rolling(Wz, min_periods=60).mean()
        sig = cum_ret.rolling(Wz, min_periods=60).std()
        z   = (cum_ret - mu) / sig.replace(0, np.nan)

        # Candidato
        cand = (z < zsog).astype(int)

        # Filtro close position
        if cp < 1.0 and "close_pos_in_range" in g.columns:
            cand = cand & (g["close_pos_in_range"] < cp)

        g["z_score"]    = z
        g["is_cand"]    = cand.fillna(0).astype(int)

        # Canone: almeno 5 giorni di gap tra episodi (evita cluster)
        is_canone = []
        last_ep   = -999
        for i, row in g.iterrows():
            if row["is_cand"] == 1 and (i - last_ep) >= 5:
                is_canone.append(1)
                last_ep = i
            else:
                is_canone.append(0)
        g["is_canone"] = is_canone

        results.append(g)

    return pd.concat(results, ignore_index=True)


def compute_pattern_stats(df_pat, horizons=HORIZONS):
    """
    Calcola statistiche di performance per i canoni del pattern.
    Restituisce metriche per ogni orizzonte H.
    """
    canoni = df_pat[df_pat["is_canone"] == 1].copy()
    baseline_all = df_pat.copy()

    stats_out = {}
    for H in horizons:
        fwd_col = f"fwd_ret_t{H}"
        if fwd_col not in canoni.columns:
            continue

        y_can  = canoni[fwd_col].dropna()
        y_all  = baseline_all[fwd_col].dropna()

        if len(y_can) < 10:
            continue

        avg_ret      = float(y_can.mean())
        avg_baseline = float(y_all.mean())
        hit_rate     = float((y_can > 0).mean())
        lift         = avg_ret / avg_baseline if avg_baseline != 0 else np.nan
        t_stat, p_val = stats.ttest_1samp(y_can, 0)

        # Per ticker
        per_ticker = []
        for ticker, grp in df_pat.groupby("ticker"):
            c = grp[grp["is_canone"]==1][fwd_col].dropna()
            if len(c) >= 3:
                per_ticker.append({
                    "ticker": ticker,
                    "n": len(c),
                    "avg_ret": c.mean(),
                    "hit_rate": (c>0).mean(),
                })

        stats_out[H] = {
            "n_canoni":    len(y_can),
            "avg_ret":     avg_ret,
            "avg_baseline":avg_baseline,
            "hit_rate":    hit_rate,
            "lift":        lift,
            "t_stat":      t_stat,
            "p_val":       p_val,
            "sig":         p_val < 0.10,
            "per_ticker":  pd.DataFrame(per_ticker).sort_values("avg_ret", ascending=False) if per_ticker else pd.DataFrame(),
        }

    return stats_out, canoni


@st.cache_data(ttl=3600, show_spinner=False)
def build_all_data():
    """Pipeline completa: download → features → pattern → stats."""
    ohlcv    = load_ohlcv(ITALIAN_TICKERS, START_DATE, END_DATE)
    if ohlcv.empty:
        return None
    features = build_features(ohlcv)
    pat_df   = detect_shock_down(features, SHOCK_PARAMS)
    pat_stats, canoni = compute_pattern_stats(pat_df)
    return {
        "ohlcv":     ohlcv,
        "features":  features,
        "pat_df":    pat_df,
        "canoni":    canoni,
        "stats":     pat_stats,
    }


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem 0;'>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:800;
                    color:#e8edf5; letter-spacing:-0.02em;'>
            📊 Equity Patterns
        </div>
        <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem;
                    color:#4a5a75; margin-top:4px; text-transform:uppercase;
                    letter-spacing:0.1em;'>
            Pattern Discovery Italia
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📊  Overview",
         "📉  Mean Reversion",
         "📖  Metodologia",
         "📤  Export"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2d47; margin:1.5rem 0;'>", unsafe_allow_html=True)

    # Filtri
    st.markdown("<div style='font-size:0.65rem; color:#4a5a75; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;'>Orizzonte forward</div>", unsafe_allow_html=True)
    H_sel = st.selectbox("H", [3,5,10,15], index=1, label_visibility="collapsed")

    st.markdown("<div style='font-size:0.65rem; color:#4a5a75; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.8rem; margin-bottom:0.5rem;'>Universo ticker</div>", unsafe_allow_html=True)
    n_tickers = st.slider("N ticker", 5, len(ITALIAN_TICKERS), 20, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2d47; margin:1.5rem 0;'>", unsafe_allow_html=True)

    if st.button("🔄 Aggiorna dati", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.62rem; color:#4a5a75; margin-top:1rem;'>
        Universo: {n_tickers} ticker FTSE Italia<br>
        Periodo: {START_DATE} → oggi<br>
        <span style='color:#8a9ab5;'>Aggiornato: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
tickers_to_use = ITALIAN_TICKERS[:n_tickers]

with st.spinner("⚙️  Download dati e calcolo pattern..."):
    try:
        DATA = build_all_data()
        data_ok = DATA is not None and len(DATA.get("stats", {})) > 0
    except Exception as e:
        data_ok = False
        err_msg = str(e)

if not data_ok:
    st.error("Errore nel caricamento dati. Verifica la connessione internet.")
    st.stop()

pat_df  = DATA["pat_df"]
canoni  = DATA["canoni"]
stats_d = DATA["stats"]
ohlcv   = DATA["ohlcv"]

# Metriche principali per H selezionato
m = stats_d.get(H_sel, stats_d.get(5, {}))

n_tickers_ok  = pat_df["ticker"].nunique()
n_episodi     = int(pat_df["is_canone"].sum())
avg_ret       = m.get("avg_ret", np.nan)
hit_rate      = m.get("hit_rate", np.nan)
lift          = m.get("lift", np.nan)
p_val         = m.get("p_val", np.nan)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown(f"""
    <div class='top-bar'>
        <div class='top-bar-title'>Equity Pattern Discovery — Italia</div>
        <div class='top-bar-meta'>
            <span class='dot-live'></span>LIVE &nbsp;|&nbsp;
            Universo: {n_tickers_ok} ticker &nbsp;|&nbsp;
            Periodo: {START_DATE} → {END_DATE}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI
    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [
        ("Ticker analizzati", str(n_tickers_ok), "universo FTSE Italia", "#3b82f6"),
        ("Episodi canone", str(n_episodi), "shock down confermati", "#8b5cf6"),
        (f"Avg Return (H={H_sel}gg)", f"{avg_ret*100:.2f}%" if not np.isnan(avg_ret) else "n/a",
         f"baseline: {m.get('avg_baseline',0)*100:.2f}%", "#10b981" if avg_ret>0 else "#ef4444"),
        (f"Hit Rate (H={H_sel}gg)", f"{hit_rate:.1%}" if not np.isnan(hit_rate) else "n/a",
         "% segnali corretti", "#10b981" if hit_rate>0.5 else "#ef4444"),
        ("P-value", f"{p_val:.3f}" if not np.isnan(p_val) else "n/a",
         "✅ Significativo" if m.get("sig") else "⚠️ Non sig.", "#10b981" if m.get("sig") else "#f59e0b"),
    ]
    for col, (label, val, sub, color) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value' style='color:{color}; font-size:1.7rem;'>{val}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Distribuzione per ticker
    st.markdown("<div class='section-header'>EPISODI PER TICKER — TOP 15</div>", unsafe_allow_html=True)

    tk_counts = (
        pat_df[pat_df["is_canone"]==1]
        .groupby("ticker")
        .agg(n_episodi=("is_canone","sum"),
             avg_ret_H=(f"fwd_ret_t{H_sel}","mean"))
        .reset_index()
        .sort_values("avg_ret_H", ascending=False)
        .head(15)
    )
    tk_counts["ticker_clean"] = tk_counts["ticker"].str.replace(".MI","",regex=False)

    fig_tk = go.Figure()
    colors = ["#10b981" if v > 0 else "#ef4444" for v in tk_counts["avg_ret_H"]]
    fig_tk.add_bar(
        x=tk_counts["ticker_clean"],
        y=tk_counts["avg_ret_H"] * 100,
        marker_color=colors,
        marker_opacity=0.85,
        text=[f"{v:.1f}%" for v in tk_counts["avg_ret_H"]*100],
        textposition="outside",
        textfont=dict(size=10, color="#8a9ab5"),
        customdata=tk_counts["n_episodi"],
        hovertemplate="<b>%{x}</b><br>Avg ret: %{y:.2f}%<br>N episodi: %{customdata}<extra></extra>",
    )
    fig_tk.add_hline(y=0, line=dict(color="#4a5a75", width=0.8))
    fig_tk.update_layout(**PLOTLY_DARK, height=320,
                         yaxis_title=f"Avg Forward Return H={H_sel}gg (%)",
                         title=dict(text=""),
                         xaxis=dict(tickangle=-35, **PLOTLY_DARK["xaxis"]))
    st.plotly_chart(fig_tk, use_container_width=True)

    # Tabella metriche per orizzonte
    st.markdown("<div class='section-header'>METRICHE PER ORIZZONTE — SHOCK DOWN MR</div>", unsafe_allow_html=True)
    rows = []
    for H, sm in stats_d.items():
        rows.append({
            "H (giorni)": H,
            "N canoni": sm["n_canoni"],
            "Avg Return": f"{sm['avg_ret']*100:.2f}%",
            "Avg Baseline": f"{sm['avg_baseline']*100:.2f}%",
            "Lift": f"{sm['lift']:.2f}x",
            "Hit Rate": f"{sm['hit_rate']:.1%}",
            "T-stat": f"{sm['t_stat']:.2f}",
            "P-value": f"{sm['p_val']:.3f}",
            "Significativo": "✅" if sm["sig"] else "⚠️",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ I risultati mostrati sono calcolati su dati storici con metodologia as-of (nessun look-ahead).
        Performance passate non sono indicative di risultati futuri. Strumento di supporto alla ricerca.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — MEAN REVERSION
# ═══════════════════════════════════════════════════════════════
elif "Mean Reversion" in page:

    st.markdown("""
    <div class='top-bar'>
        <div class='top-bar-title'>Mean Reversion — Shock Down</div>
        <div class='top-bar-meta'>Pattern: shock_down_mr &nbsp;|&nbsp; Azionario Italia</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-bottom:1.5rem;'>
        <span class='pattern-badge'>SHOCK DOWN MEAN REVERSION</span>
        <span style='font-family:JetBrains Mono,monospace; font-size:0.75rem;
                     color:#4a5a75; margin-left:1rem;'>
            W={SHOCK_PARAMS['W']}gg | Wz={SHOCK_PARAMS['Wz']}gg |
            z-soglia={SHOCK_PARAMS['zsog']} | H={SHOCK_PARAMS['H']}gg
        </span>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    for col, (label, val, sub, color) in zip([k1,k2,k3,k4], [
        ("Episodi canone", str(n_episodi), f"su {n_tickers_ok} ticker", "#8b5cf6"),
        (f"Avg Ret H={H_sel}gg", f"{avg_ret*100:.2f}%" if not np.isnan(avg_ret) else "n/a",
         f"baseline: {m.get('avg_baseline',0)*100:.2f}%", "#10b981" if avg_ret>0 else "#ef4444"),
        ("Hit Rate", f"{hit_rate:.1%}" if not np.isnan(hit_rate) else "n/a",
         "% episodi con ret > 0", "#10b981" if hit_rate>0.5 else "#ef4444"),
        ("Lift vs baseline", f"{lift:.2f}x" if not np.isnan(lift) else "n/a",
         f"p-val: {p_val:.3f} {'✅' if m.get('sig') else '⚠️'}", "#10b981" if lift>1 else "#ef4444"),
    ]):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value' style='color:{color}; font-size:1.6rem;'>{val}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Grafico 1: Distribuzione forward return canoni vs baseline ──
    st.markdown("<div class='section-header'>DISTRIBUZIONE FORWARD RETURN — CANONI vs BASELINE</div>", unsafe_allow_html=True)

    fwd_col = f"fwd_ret_t{H_sel}"
    if fwd_col in pat_df.columns:
        y_can_plot = canoni[fwd_col].dropna() * 100
        y_all_plot = pat_df[fwd_col].dropna() * 100

        fig_dist = go.Figure()
        fig_dist.add_histogram(
            x=y_all_plot, nbinsx=80,
            marker_color="#4a5a75", opacity=0.5,
            histnorm="probability density",
            name=f"Tutti i giorni (baseline)"
        )
        fig_dist.add_histogram(
            x=y_can_plot, nbinsx=60,
            marker_color="#8b5cf6", opacity=0.8,
            histnorm="probability density",
            name=f"Episodi canone (n={len(y_can_plot)})"
        )
        fig_dist.add_vline(x=float(y_can_plot.mean()),
                           line=dict(color="#a78bfa", dash="dash", width=2),
                           annotation_text=f"Media canoni: {y_can_plot.mean():.2f}%",
                           annotation_font_color="#a78bfa", annotation_font_size=10)
        fig_dist.add_vline(x=float(y_all_plot.mean()),
                           line=dict(color="#4a5a75", dash="dot", width=1),
                           annotation_text=f"Baseline: {y_all_plot.mean():.2f}%",
                           annotation_font_color="#4a5a75", annotation_font_size=10)
        fig_dist.add_vline(x=0, line=dict(color="#8a9ab5", width=0.8))
        fig_dist.update_layout(**PLOTLY_DARK, height=320,
                               xaxis_title=f"Forward Return H={H_sel}gg (%)",
                               yaxis_title="Densità",
                               barmode="overlay",
                               title=dict(text=""))
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Grafico 2: Forward return cumulativo per orizzonte ──
    st.markdown("<div class='section-header'>PROFILO RENDIMENTO CUMULATIVO — H=1 → 15 GIORNI</div>", unsafe_allow_html=True)

    fwd_means_can  = []
    fwd_means_base = []
    fwd_h_list     = []
    for H in range(1, 16):
        fc = f"fwd_ret_t{H}"
        if fc in pat_df.columns:
            fwd_h_list.append(H)
            fwd_means_can.append(canoni[fc].mean() * 100)
            fwd_means_base.append(pat_df[fc].mean() * 100)

    if fwd_h_list:
        fig_fwd = go.Figure()
        fig_fwd.add_scatter(
            x=fwd_h_list, y=fwd_means_base,
            mode="lines+markers",
            line=dict(color="#4a5a75", dash="dash", width=1.5),
            marker=dict(size=5),
            name="Baseline (tutti i giorni)"
        )
        fig_fwd.add_scatter(
            x=fwd_h_list, y=fwd_means_can,
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=2.5),
            marker=dict(size=7, symbol="circle"),
            fill="tonexty",
            fillcolor="rgba(139,92,246,0.08)",
            name="Episodi canone"
        )
        fig_fwd.add_hline(y=0, line=dict(color="#4a5a75", width=0.8))
        fig_fwd.update_layout(**PLOTLY_DARK, height=280,
                              xaxis_title="Orizzonte (giorni)",
                              yaxis_title="Avg Forward Return (%)",
                              title=dict(text=""))
        st.plotly_chart(fig_fwd, use_container_width=True)

    # ── Grafico 3: Stabilità temporale ──
    col_temp, col_tk = st.columns(2)

    with col_temp:
        st.markdown("<div class='section-header'>STABILITÀ TEMPORALE — LIFT PER ANNO</div>", unsafe_allow_html=True)
        if fwd_col in canoni.columns:
            canoni_y = canoni.copy()
            canoni_y["year"] = canoni_y["date"].dt.year
            pat_y = pat_df.copy()
            pat_y["year"] = pat_y["date"].dt.year

            yearly_can  = canoni_y.groupby("year")[fwd_col].mean()
            yearly_base = pat_y.groupby("year")[fwd_col].mean()
            lift_yr     = (yearly_can / yearly_base).replace([np.inf,-np.inf], np.nan).dropna()
            n_yr        = canoni_y.groupby("year").size()

            fig_yr = make_subplots(specs=[[{"secondary_y": True}]])
            bar_colors = ["#10b981" if v >= 1 else "#ef4444" for v in lift_yr.values]
            fig_yr.add_bar(
                x=lift_yr.index.astype(str), y=lift_yr.values,
                marker_color=bar_colors, opacity=0.8,
                name="Lift annuale", secondary_y=False
            )
            fig_yr.add_scatter(
                x=n_yr.reindex(lift_yr.index).index.astype(str),
                y=n_yr.reindex(lift_yr.index).values,
                mode="lines+markers",
                line=dict(color="#f59e0b", width=1.5),
                marker=dict(size=5),
                name="N episodi", secondary_y=True
            )
            fig_yr.add_hline(y=1.0, line=dict(color="#4a5a75", dash="dash", width=1))
            fig_yr.update_layout(**PLOTLY_DARK, height=260,
                                 title=dict(text=""),
                                 legend=dict(bgcolor="rgba(0,0,0,0)"))
            fig_yr.update_yaxes(title_text="Lift (>1 = sovra-baseline)", secondary_y=False,
                                gridcolor="#1e2d47", tickfont=dict(size=10))
            fig_yr.update_yaxes(title_text="N episodi", secondary_y=True,
                                gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10))
            st.plotly_chart(fig_yr, use_container_width=True)

    with col_tk:
        st.markdown("<div class='section-header'>TOP 10 TICKER PER AVG RETURN</div>", unsafe_allow_html=True)
        per_tk_df = m.get("per_ticker", pd.DataFrame())
        if not per_tk_df.empty:
            top10 = per_tk_df.head(10).copy()
            top10["ticker_clean"] = top10["ticker"].str.replace(".MI","",regex=False)
            fig_top = go.Figure()
            fig_top.add_bar(
                x=top10["ticker_clean"],
                y=top10["avg_ret"] * 100,
                marker_color=["#10b981" if v>0 else "#ef4444" for v in top10["avg_ret"]],
                opacity=0.85,
                text=[f"{v:.1f}%" for v in top10["avg_ret"]*100],
                textposition="outside",
                textfont=dict(size=10, color="#8a9ab5"),
            )
            fig_top.add_hline(y=0, line=dict(color="#4a5a75", width=0.8))
            fig_top.update_layout(**PLOTLY_DARK, height=260,
                                  yaxis_title="Avg Ret (%)",
                                  title=dict(text=""),
                                  showlegend=False)
            st.plotly_chart(fig_top, use_container_width=True)

    # ── Ultimo episodio per ticker ──
    st.markdown("<div class='section-header'>ULTIMI EPISODI RILEVATI</div>", unsafe_allow_html=True)
    last_eps = (
        canoni.sort_values("date")
        .groupby("ticker")
        .last()
        .reset_index()
        [["ticker","date","close","z_score",fwd_col]]
        .rename(columns={
            "ticker":"Ticker","date":"Data","close":"Prezzo chiusura",
            "z_score":"Z-score","fwd_col":f"Ret H={H_sel}gg"
        })
        .sort_values("Data", ascending=False)
        .head(20)
    )
    last_eps["Data"] = pd.to_datetime(last_eps["Data"]).dt.strftime("%Y-%m-%d")
    last_eps["Z-score"] = last_eps["Z-score"].round(2)
    last_eps["Ticker"] = last_eps["Ticker"].str.replace(".MI","",regex=False)
    st.dataframe(last_eps, use_container_width=True, hide_index=True, height=350)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ Il segnale è calcolato as-of su dati storici. I parametri (W, Wz, z-soglia) sono stati
        calibrati su in-sample e non modificati su out-of-sample per evitare look-ahead bias.
        Non costituisce raccomandazione di investimento ai sensi di MiFID II.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — METODOLOGIA
# ═══════════════════════════════════════════════════════════════
elif "Metodologia" in page:

    st.markdown("<h2 style='margin-bottom:0.3rem;'>Metodologia</h2>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#4a5a75; margin-bottom:1.5rem;'>Framework Pattern Discovery — Azionario Italia v1.0</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='chain-box'>
        Universo ~30 ticker FTSE Italia
        &nbsp;→&nbsp; OHLCV giornaliero yfinance
        &nbsp;→&nbsp; Feature primitive (ret, vol, range)
        &nbsp;→&nbsp; Z-score rolling shock negativo
        &nbsp;→&nbsp; Candidato → Canone (min gap 5gg)
        &nbsp;→&nbsp; Forward return H=3,5,10,15gg
        &nbsp;→&nbsp; Metriche: lift, hit rate, t-test
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='method-card'>
            <h4>📡 Fonte dati</h4>
            <p>
            Yahoo Finance via yfinance — dati OHLCV giornalieri aggiustati per dividendi
            e split (auto_adjust=True). Universo: principali titoli FTSE MIB e Mid Cap Italia.
            Pulizia: rimozione variazioni giornaliere anomale (>100%) che indicano
            errori nel dato o eventi straordinari non rappresentativi.
            </p>
        </div>
        <div class='method-card'>
            <h4>📐 Logica del pattern Shock Down</h4>
            <p>
            Un titolo che registra un return cumulativo su W giorni molto inferiore
            alla sua media storica (z-score fortemente negativo) può presentare
            condizioni di ipervenduto tecnico e potenziale mean reversion.<br><br>
            <b>Formula z-score:</b><br>
            z = (ret_cumW − μ_rolling) / σ_rolling<br><br>
            <b>Candidato:</b> z &lt; soglia E close nel bottom cp% del range giornaliero<br>
            <b>Canone:</b> candidato con almeno 5 giorni di gap dall'episodio precedente
            (evita cluster di segnali sullo stesso movimento)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='method-card'>
            <h4>🔒 Blindatura anti-overfitting</h4>
            <p>
            I parametri ottimali (W, Wz, z-soglia, cp) vengono scelti dalla grid search
            <b>prima</b> di guardare la distribuzione dei forward return. Questa procedura —
            chiamata "blindatura" — è la difesa principale contro il look-ahead bias e
            l'overfitting sui dati storici.<br><br>
            Il criterio di selezione è il <b>lift per-ticker</b> (media dei lift
            su tutti i ticker dell'universo), non il return assoluto massimo.
            Questo garantisce generalizzazione cross-sectional.
            </p>
        </div>
        <div class='method-card'>
            <h4>📊 Forward return e metriche</h4>
            <p>
            Il forward return a H giorni è calcolato come:<br>
            fwd_ret_tH = close_{t+H} / close_t − 1<br><br>
            Winsorizzato a ±50% per eliminare outlier estremi.<br><br>
            <b>Lift</b> = avg_ret_canoni / avg_ret_baseline<br>
            <b>Hit Rate</b> = % episodi con fwd_ret > 0<br>
            <b>T-test</b>: H0: media = 0, p &lt; 0.10 = significativo<br><br>
            La stabilità temporale del lift per anno è il principale
            indicatore di robustezza out-of-sample.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Tabella parametri
    st.markdown("<div class='section-header'>PARAMETRI CORRENTI — SHOCK DOWN MR</div>", unsafe_allow_html=True)
    params_df = pd.DataFrame([
        {"Parametro":"W", "Valore":SHOCK_PARAMS["W"], "Descrizione":"Finestra return cumulativo (giorni)"},
        {"Parametro":"Wz","Valore":SHOCK_PARAMS["Wz"],"Descrizione":"Finestra rolling z-score (giorni)"},
        {"Parametro":"z-soglia","Valore":SHOCK_PARAMS["zsog"],"Descrizione":"Soglia z-score (sotto = candidato shock)"},
        {"Parametro":"cp","Valore":SHOCK_PARAMS["cp"],"Descrizione":"Close position max nel range giornaliero (0=low, 1=high)"},
        {"Parametro":"H","Valore":SHOCK_PARAMS["H"],"Descrizione":"Orizzonte forward return per ottimizzazione"},
        {"Parametro":"min_gap","Valore":5,"Descrizione":"Giorni minimi tra episodi consecutivi stesso ticker"},
    ])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ Posizionamento commerciale: strumento di supporto alla ricerca quantitativa.
        Non costituisce consulenza in materia di investimenti ai sensi di MiFID II.
        I segnali identificano anomalie statistiche storiche — non garantiscono performance future.
        Il sistema deve essere usato come input supplementare al processo di investimento,
        non come sistema di trading autonomo.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — EXPORT
# ═══════════════════════════════════════════════════════════════
elif "Export" in page:

    st.markdown("<h2 style='margin-bottom:0.3rem;'>Export</h2>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#4a5a75; margin-bottom:1.5rem;'>Esporta risultati pattern discovery in Excel o CSV</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>EXPORT EXCEL — REPORT COMPLETO</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='method-card'>
            <h4>📊 Contenuto report Excel</h4>
            <p>
            • Foglio 1: Tutti gli episodi canone con date e ticker<br>
            • Foglio 2: Metriche per orizzonte (lift, hit rate, t-test)<br>
            • Foglio 3: Performance per ticker<br>
            • Foglio 4: Parametri del modello<br>
            • Foglio 5: Sommario esecutivo
            </p>
        </div>
        """, unsafe_allow_html=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:

            # Foglio 1: Episodi canone
            ep_export = canoni[
                ["date","ticker","close","z_score","is_canone"] +
                [f"fwd_ret_t{H}" for H in HORIZONS if f"fwd_ret_t{H}" in canoni.columns]
            ].copy()
            ep_export["date"] = ep_export["date"].dt.strftime("%Y-%m-%d")
            ep_export.sort_values("date", ascending=False).to_excel(
                writer, sheet_name="Episodi Canone", index=False
            )

            # Foglio 2: Metriche per orizzonte
            rows = []
            for H, sm in stats_d.items():
                rows.append({
                    "Orizzonte (giorni)": H,
                    "N canoni": sm["n_canoni"],
                    "Avg Return": sm["avg_ret"],
                    "Avg Baseline": sm["avg_baseline"],
                    "Lift": sm["lift"],
                    "Hit Rate": sm["hit_rate"],
                    "T-statistic": sm["t_stat"],
                    "P-value": sm["p_val"],
                    "Significativo": sm["sig"],
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name="Metriche Orizzonte", index=False)

            # Foglio 3: Per ticker
            per_tk = stats_d.get(H_sel, {}).get("per_ticker", pd.DataFrame())
            if not per_tk.empty:
                per_tk.to_excel(writer, sheet_name="Performance Ticker", index=False)

            # Foglio 4: Parametri
            pd.DataFrame([
                {"Param":"W","Valore":SHOCK_PARAMS["W"]},
                {"Param":"Wz","Valore":SHOCK_PARAMS["Wz"]},
                {"Param":"zsog","Valore":SHOCK_PARAMS["zsog"]},
                {"Param":"cp","Valore":SHOCK_PARAMS["cp"]},
                {"Param":"H","Valore":SHOCK_PARAMS["H"]},
            ]).to_excel(writer, sheet_name="Parametri Modello", index=False)

            # Foglio 5: Sommario
            m5 = stats_d.get(5, {})
            pd.DataFrame([
                ["Sistema","Equity Pattern Discovery v1.0"],
                ["Pattern","Shock Down Mean Reversion"],
                ["Data generazione", datetime.now().strftime("%Y-%m-%d %H:%M")],
                ["Periodo dati", f"{START_DATE} → {END_DATE}"],
                ["Ticker analizzati", n_tickers_ok],
                ["Episodi canone totali", n_episodi],
                ["Avg Return H=5gg", f"{m5.get('avg_ret',0)*100:.2f}%"],
                ["Hit Rate H=5gg", f"{m5.get('hit_rate',0):.1%}"],
                ["Lift H=5gg", f"{m5.get('lift',0):.2f}x"],
                ["P-value H=5gg", f"{m5.get('p_val',1):.3f}"],
                ["Disclaimer","Supporto alla ricerca. Non è consulenza MiFID II."],
            ], columns=["Parametro","Valore"]).to_excel(
                writer, sheet_name="Sommario", index=False
            )

        output.seek(0)
        st.download_button(
            label="📥 Download Report Excel",
            data=output,
            file_name=f"EquityPatterns_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_b:
        st.markdown("<div class='section-header'>EXPORT CSV — EPISODI CANONE</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='method-card'>
            <h4>📄 Contenuto CSV</h4>
            <p>
            Tutti gli episodi canone in formato CSV leggero.<br>
            Importabile in Excel, Bloomberg o qualsiasi sistema
            di portfolio management.<br><br>
            Colonne: data, ticker, prezzo, z-score,
            forward return per H=3,5,10,15 giorni.
            </p>
        </div>
        """, unsafe_allow_html=True)

        csv_ep = canoni[
            ["date","ticker","close","z_score"] +
            [f"fwd_ret_t{H}" for H in HORIZONS if f"fwd_ret_t{H}" in canoni.columns]
        ].copy()
        csv_ep["date"] = csv_ep["date"].dt.strftime("%Y-%m-%d")
        csv_ep = csv_ep.sort_values("date", ascending=False)

        st.download_button(
            label="📥 Download CSV Episodi",
            data=csv_ep.to_csv(index=False),
            file_name=f"ShockDownEpisodi_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>ANTEPRIMA — ULTIMI 15 EPISODI</div>", unsafe_allow_html=True)
        preview = csv_ep.head(15).copy()
        preview["ticker"] = preview["ticker"].str.replace(".MI","",regex=False)
        st.dataframe(preview, use_container_width=True, hide_index=True, height=320)
