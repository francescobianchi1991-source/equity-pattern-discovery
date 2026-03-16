"""
pipeline_ab.py
==============
Modulo Python puro — Blocco A e Blocco B del framework Pattern Discovery.
Nessuna UI. Importato dalla web app Streamlit.

Funzioni esportate:
    Blocco A:
        download_ohlcv(tickers, start, end, config)  -> DataFrame grezzo
        run_integrity_checks(df)                     -> (report, quality_df, anomaly_df)
        apply_universe_filters(df, filters)          -> (df_final, valid_tickers, report)

    Blocco B:
        build_primitive_features(df, config)         -> (df_features, feature_cols)
        build_forward_path(df, config)               -> (df_full, fwd_cols)
        build_base_dataset(df, config)               -> (df_final, feature_cols, fwd_cols)
        winsorize_forward(df, fwd_cols, pct)         -> df
        get_diagnostics(df, feature_cols, fwd_cols)  -> dict
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONI DEFAULT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DOWNLOAD_CONFIG = {
    "chunk_size":      20,      # ticker per batch
    "max_retries":     3,       # retry per ticker su errore
    "min_rows":        60,      # righe minime per ticker valido
    "remove_outliers": True,    # rimuovi variazioni >100% giornaliere
}

DEFAULT_FILTER_CONFIG = {
    "min_years":           3.0,   # storia minima in anni
    "min_obs":             500,   # osservazioni minime
    "max_missing_pct":     5.0,   # % missing massima su close
    "exclude_anomalies":   True,  # esclude ticker con anomalie OHLCV
    "max_anomaly_pct":     1.0,   # % max righe anomale tollerato
}

DEFAULT_FEATURE_CONFIG = {
    "features_price":      True,
    "features_candle":     True,
    "features_gap":        True,
    "features_excursion":  True,
    "features_tech_support": True,
    "h_max":               20,
    "fwd_close":           True,
    "fwd_return":          True,
    "fwd_max_return":      True,
    "fwd_min_return":      True,
    "winsorize_pct":       50.0,  # winsorizzazione forward return ±50%
}

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO A.3 — DOWNLOAD OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def download_ohlcv(
    tickers: list,
    start:   str,
    end:     str,
    config:  dict = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Scarica dati OHLCV giornalieri da Yahoo Finance per la lista di ticker.

    Parametri
    ---------
    tickers           : lista di ticker (es. ['ENI.MI', 'ENEL.MI'])
    start / end       : date in formato 'YYYY-MM-DD'
    config            : dizionario configurazione (vedi DEFAULT_DOWNLOAD_CONFIG)
    progress_callback : funzione opzionale chiamata ad ogni ticker
                        firma: callback(i, total, ticker, status)
                        status: 'ok' | 'empty' | 'error'

    Ritorna
    -------
    DataFrame con colonne: date, ticker, open, high, low, close, volume
    Ordinato per (ticker, date). Ticker con dati insufficienti esclusi.
    """
    cfg     = {**DEFAULT_DOWNLOAD_CONFIG, **(config or {})}
    frames  = []
    n       = len(tickers)

    for i, ticker in enumerate(tickers):
        status = "ok"
        for attempt in range(cfg["max_retries"]):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                if df.empty:
                    status = "empty"
                    break

                # Gestisci MultiIndex (yfinance >= 0.2.x)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                # Normalizza nomi colonne
                col_map = {c: c.lower() for c in df.columns}
                df = df.rename(columns=col_map)

                # Tieni solo le colonne necessarie
                needed = [c for c in ["open","high","low","close","volume"] if c in df.columns]
                if "close" not in needed:
                    status = "error"
                    break

                df = df[needed].copy()
                df.index.name = "date"
                df = df.reset_index()
                df["ticker"] = ticker
                df["date"]   = pd.to_datetime(df["date"])
                df           = df.dropna(subset=["close"])

                # Rimuovi outlier: variazioni giornaliere > 100%
                if cfg["remove_outliers"]:
                    ret_tmp = df["close"].pct_change().abs()
                    df      = df[ret_tmp < 1.0].copy()

                if len(df) < cfg["min_rows"]:
                    status = "empty"
                    break

                frames.append(df)
                break  # successo, esci dal loop retry

            except Exception:
                if attempt == cfg["max_retries"] - 1:
                    status = "error"

        if progress_callback:
            progress_callback(i + 1, n, ticker, status)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO A.4 — CONTROLLI DI INTEGRITÀ DEL DATO
# ─────────────────────────────────────────────────────────────────────────────

def run_integrity_checks(df: pd.DataFrame) -> tuple:
    """
    Esegue controlli di integrità su DataFrame OHLCV multi-ticker.

    Controlli eseguiti:
    1. Struttura: colonne obbligatorie presenti
    2. Duplicati: (ticker, date) duplicati
    3. Righe vuote
    4. Missing per colonna
    5. Anomalie OHLCV: high < low, prezzi negativi, volume negativo,
       close == 0, variazioni > 100%

    Ritorna
    -------
    (report_dict, quality_df, anomaly_df)
    - report_dict  : dict con metriche globali
    - quality_df   : DataFrame per-ticker con statistiche qualità
    - anomaly_df   : DataFrame con le righe anomale identificate
    """
    REQUIRED_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
    report = {}

    # 1. Struttura
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    report["missing_required_cols"] = missing_cols
    report["has_required_cols"]     = len(missing_cols) == 0

    # 2. Duplicati
    dups = df.duplicated(["ticker", "date"]).sum()
    report["n_duplicates"] = int(dups)

    # 3. Righe vuote
    report["n_empty_rows"] = int(df.isnull().all(axis=1).sum())

    # 4. Missing per colonna
    present_req = [c for c in REQUIRED_COLS if c in df.columns]
    report["missing_pct_by_col"] = (
        df[present_req].isnull().mean() * 100
    ).round(2).to_dict()

    # 5. Anomalie OHLCV
    anomaly_mask = pd.Series(False, index=df.index)
    anomaly_reasons = pd.Series("", index=df.index)

    if all(c in df.columns for c in ["high","low","open","close","volume"]):
        mask_hl   = df["high"] < df["low"]
        mask_negv = df["volume"] < 0
        mask_negp = (df[["open","high","low","close"]] < 0).any(axis=1)
        mask_zero = df["close"] == 0

        # Variazioni > 100%
        ret_abs = df.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().abs()
        )
        mask_jump = ret_abs > 1.0

        anomaly_mask = mask_hl | mask_negv | mask_negp | mask_zero | mask_jump

        # Etichetta motivo
        anomaly_reasons[mask_hl]   += "high<low "
        anomaly_reasons[mask_negv] += "vol<0 "
        anomaly_reasons[mask_negp] += "price<0 "
        anomaly_reasons[mask_zero] += "close=0 "
        anomaly_reasons[mask_jump] += "ret>100% "

    report["n_anomaly_rows"] = int(anomaly_mask.sum())
    report["anomaly_pct"]    = round(anomaly_mask.mean() * 100, 3)

    anomaly_df = df[anomaly_mask].copy()
    if "close" in anomaly_df.columns:
        anomaly_df["anomaly_reason"] = anomaly_reasons[anomaly_mask]

    # 6. Quality per ticker
    quality_rows = []
    for ticker, grp in df.groupby("ticker"):
        n_rows   = len(grp)
        n_miss   = grp["close"].isnull().sum() if "close" in grp.columns else 0
        n_anom   = int(anomaly_mask.loc[grp.index].sum())
        date_min = grp["date"].min() if "date" in grp.columns else pd.NaT
        date_max = grp["date"].max() if "date" in grp.columns else pd.NaT
        quality_rows.append({
            "ticker":       ticker,
            "n_rows":       n_rows,
            "n_missing":    int(n_miss),
            "missing_pct":  round(n_miss / n_rows * 100, 2) if n_rows > 0 else 0,
            "n_anomalies":  n_anom,
            "anomaly_pct":  round(n_anom / n_rows * 100, 3) if n_rows > 0 else 0,
            "date_min":     date_min,
            "date_max":     date_max,
        })

    quality_df = pd.DataFrame(quality_rows)
    report["n_tickers_raw"]     = int(df["ticker"].nunique()) if "ticker" in df.columns else 0
    report["total_rows"]        = len(df)
    report["check_passed"]      = (
        report["n_duplicates"] == 0 and
        report["n_anomaly_rows"] < len(df) * 0.02  # tolleranza 2%
    )

    return report, quality_df, anomaly_df


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO A.5 — FILTRI UNIVERSO
# ─────────────────────────────────────────────────────────────────────────────

def apply_universe_filters(
    df:      pd.DataFrame,
    filters: dict = None,
) -> tuple:
    """
    Applica filtri di qualità storica e restituisce il dataset pulito.

    Filtri applicati:
    - Minimo anni di storia
    - Minimo numero di osservazioni
    - Massimo % di missing su close
    - Esclusione ticker con troppe anomalie OHLCV

    Parametri
    ---------
    df      : DataFrame OHLCV grezzo (output di download_ohlcv)
    filters : dizionario filtri (vedi DEFAULT_FILTER_CONFIG)

    Ritorna
    -------
    (df_final, valid_tickers, report)
    - df_final       : DataFrame filtrato con solo ticker validi
    - valid_tickers  : lista ticker che hanno superato i filtri
    - report         : dict con dettaglio filtri applicati
    """
    cfg = {**DEFAULT_FILTER_CONFIG, **(filters or {})}

    if df.empty:
        return df, [], {"error": "DataFrame vuoto"}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Rimuovi anomalie prima dei filtri
    if cfg["exclude_anomalies"] and all(c in df.columns for c in ["high","low","close","volume"]):
        ret_abs = df.groupby("ticker")["close"].transform(lambda x: x.pct_change().abs())
        mask_ok = (
            (df["high"]   >= df["low"]) &
            (df["volume"] >= 0) &
            (df["close"]  > 0) &
            (ret_abs      < 1.0)
        )
        df = df[mask_ok].copy()

    # Calcola statistiche per ticker
    stats = []
    for ticker, grp in df.groupby("ticker"):
        grp  = grp.sort_values("date")
        n    = len(grp)
        if n == 0:
            continue
        d_min    = grp["date"].min()
        d_max    = grp["date"].max()
        n_years  = (d_max - d_min).days / 365.25
        n_miss   = grp["close"].isnull().sum()
        miss_pct = n_miss / n * 100 if n > 0 else 100

        # Anomalie residue
        if all(c in grp.columns for c in ["high","low","close"]):
            anom = ((grp["high"] < grp["low"]) | (grp["close"] <= 0)).sum()
            anom_pct = anom / n * 100
        else:
            anom_pct = 0

        stats.append({
            "ticker":    ticker,
            "n_obs":     n,
            "n_years":   round(n_years, 2),
            "missing_pct": round(miss_pct, 2),
            "anom_pct":  round(anom_pct, 3),
            "date_min":  d_min,
            "date_max":  d_max,
        })

    stats_df = pd.DataFrame(stats) if stats else pd.DataFrame()

    # Applica filtri
    excluded = {}
    valid_tickers = []

    for _, row in stats_df.iterrows():
        reasons = []
        if row["n_years"] < cfg["min_years"]:
            reasons.append(f"storia {row['n_years']:.1f}y < {cfg['min_years']}y")
        if row["n_obs"] < cfg["min_obs"]:
            reasons.append(f"obs {row['n_obs']} < {cfg['min_obs']}")
        if row["missing_pct"] > cfg["max_missing_pct"]:
            reasons.append(f"missing {row['missing_pct']:.1f}% > {cfg['max_missing_pct']}%")
        if cfg["exclude_anomalies"] and row["anom_pct"] > cfg.get("max_anomaly_pct", 1.0):
            reasons.append(f"anomalie {row['anom_pct']:.2f}% > {cfg.get('max_anomaly_pct',1.0)}%")

        if reasons:
            excluded[row["ticker"]] = "; ".join(reasons)
        else:
            valid_tickers.append(row["ticker"])

    df_final = df[df["ticker"].isin(valid_tickers)].copy()
    df_final = df_final.sort_values(["ticker","date"]).reset_index(drop=True)

    report = {
        "n_tickers_input":   len(stats_df),
        "n_tickers_valid":   len(valid_tickers),
        "n_tickers_excluded":len(excluded),
        "valid_tickers":     valid_tickers,
        "excluded_detail":   excluded,
        "filters_applied":   cfg,
        "stats_df":          stats_df,
    }

    return df_final, valid_tickers, report


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO B.3 — FEATURE PRIMITIVE
# ─────────────────────────────────────────────────────────────────────────────

def build_primitive_features(
    df:     pd.DataFrame,
    config: dict = None,
) -> tuple:
    """
    Costruisce le feature primitive sul dataset OHLCV.

    Feature costruite (22 totali, configurabili):

    PREZZO (features_price):
        prev_close, prev_open, ret_1d, delta_close, delta_open_close

    CANDLE (features_candle):
        range_day, body, body_abs, lower_shadow, upper_shadow,
        close_pos_in_range, close_vs_open, close_vs_high, close_vs_low,
        body_range_ratio, lower_shadow_ratio, upper_shadow_ratio

    GAP (features_gap):
        gap_abs, gap_pct

    ESCURSIONE (features_excursion):
        true_range

    SUPPORTO TECNICO (features_tech_support):
        row_idx, ticker_obs_idx

    Parametri
    ---------
    df     : DataFrame OHLCV filtrato (output di apply_universe_filters)
    config : dizionario configurazione (vedi DEFAULT_FEATURE_CONFIG)

    Ritorna
    -------
    (df_out, feature_cols)
    - df_out       : DataFrame con feature aggiunte
    - feature_cols : lista nomi colonne feature create
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(config or {})}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    feature_cols = []

    for tk, grp_idx in df.groupby("ticker", sort=False).groups.items():
        g = df.loc[grp_idx].copy()

        # ── Feature di prezzo ─────────────────────────────────────────────
        if cfg.get("features_price", True):
            df.loc[grp_idx, "prev_close"]       = g["close"].shift(1).values
            df.loc[grp_idx, "prev_open"]        = g["open"].shift(1).values if "open" in g.columns else np.nan
            df.loc[grp_idx, "ret_1d"]           = g["close"].pct_change(1).values
            df.loc[grp_idx, "delta_close"]      = g["close"].diff(1).values
            df.loc[grp_idx, "delta_open_close"] = (g["close"] - g["open"]).values if "open" in g.columns else np.nan

        # ── Feature di candela ────────────────────────────────────────────
        if cfg.get("features_candle", True) and all(c in g.columns for c in ["open","high","low","close"]):
            rng     = (g["high"] - g["low"]).values
            bdy     = (g["close"] - g["open"]).values
            bdy_abs = np.abs(bdy)
            lo_sh   = np.maximum(0, np.minimum(g["open"].values, g["close"].values) - g["low"].values)
            hi_sh   = np.maximum(0, g["high"].values - np.maximum(g["open"].values, g["close"].values))

            df.loc[grp_idx, "range_day"]             = rng
            df.loc[grp_idx, "body"]                  = bdy
            df.loc[grp_idx, "body_abs"]              = bdy_abs
            df.loc[grp_idx, "lower_shadow"]          = lo_sh
            df.loc[grp_idx, "upper_shadow"]          = hi_sh
            df.loc[grp_idx, "close_pos_in_range"]    = np.where(rng > 0, (g["close"].values - g["low"].values) / rng, 0.5)
            df.loc[grp_idx, "close_vs_open"]         = bdy
            df.loc[grp_idx, "close_vs_high"]         = (g["close"] - g["high"]).values
            df.loc[grp_idx, "close_vs_low"]          = (g["close"] - g["low"]).values
            df.loc[grp_idx, "body_range_ratio"]      = np.where(rng > 0, bdy_abs / rng, 0.0)
            df.loc[grp_idx, "lower_shadow_ratio"]    = np.where(rng > 0, lo_sh / rng, 0.0)
            df.loc[grp_idx, "upper_shadow_ratio"]    = np.where(rng > 0, hi_sh / rng, 0.0)

        # ── Feature di gap ────────────────────────────────────────────────
        if cfg.get("features_gap", True) and "open" in g.columns:
            prev_c = g["close"].shift(1).values
            df.loc[grp_idx, "gap_abs"] = g["open"].values - prev_c
            df.loc[grp_idx, "gap_pct"] = np.where(
                prev_c > 0,
                (g["open"].values - prev_c) / prev_c,
                np.nan
            )

        # ── Feature di escursione (true range) ───────────────────────────
        if cfg.get("features_excursion", True) and all(c in g.columns for c in ["high","low","close"]):
            prev_c = g["close"].shift(1).values
            hl     = g["high"].values  - g["low"].values
            hc     = np.abs(g["high"].values  - prev_c)
            lc     = np.abs(g["low"].values   - prev_c)
            df.loc[grp_idx, "true_range"] = np.maximum(hl, np.maximum(hc, lc))

        # ── Feature di supporto tecnico ───────────────────────────────────
        if cfg.get("features_tech_support", True):
            df.loc[grp_idx, "row_idx"]         = np.arange(len(grp_idx))
            df.loc[grp_idx, "ticker_obs_idx"]  = np.arange(len(grp_idx))

    # Costruisci lista feature effettivamente create
    potential = [
        "prev_close","prev_open","ret_1d","delta_close","delta_open_close",
        "range_day","body","body_abs","lower_shadow","upper_shadow",
        "close_pos_in_range","close_vs_open","close_vs_high","close_vs_low",
        "body_range_ratio","lower_shadow_ratio","upper_shadow_ratio",
        "gap_abs","gap_pct","true_range","row_idx","ticker_obs_idx",
    ]
    feature_cols = [c for c in potential if c in df.columns]

    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO B.4 — FORWARD PATH
# ─────────────────────────────────────────────────────────────────────────────

def build_forward_path(
    df:     pd.DataFrame,
    config: dict = None,
) -> tuple:
    """
    Costruisce le colonne forward path per ogni ticker.

    Colonne costruite:
    - fwd_close_tH       : prezzo di chiusura a H giorni (se fwd_close=True)
    - fwd_ret_tH         : return close-to-close a H giorni (se fwd_return=True)
    - fwd_max_ret_HH     : max return raggiungibile entro H giorni / MFE proxy (se fwd_max_return=True)
    - fwd_min_ret_HH     : min return raggiungibile entro H giorni / MAE proxy (se fwd_min_return=True)

    I forward return vengono winsorizzati a ±winsorize_pct%.

    NOTA METODOLOGICA:
    Le ultime H_max righe per ticker hanno forward path incompleto (NaN).
    Questo è corretto e atteso — il Blocco C li escluderà automaticamente.

    Parametri
    ---------
    df     : DataFrame con feature primitive (output di build_primitive_features)
    config : dizionario configurazione (vedi DEFAULT_FEATURE_CONFIG)

    Ritorna
    -------
    (df_out, fwd_cols)
    - df_out   : DataFrame completo con colonne forward
    - fwd_cols : lista nomi colonne forward create
    """
    cfg   = {**DEFAULT_FEATURE_CONFIG, **(config or {})}
    h_max = cfg.get("h_max", 20)
    w_pct = cfg.get("winsorize_pct", 50.0) / 100.0

    df = df.copy()
    fwd_cols = []

    for tk, grp_idx in df.groupby("ticker", sort=False).groups.items():
        g = df.loc[grp_idx, "close"].copy()
        n = len(grp_idx)

        # ── Forward close ─────────────────────────────────────────────────
        if cfg.get("fwd_close", True):
            for h in range(1, h_max + 1):
                col = f"fwd_close_t{h}"
                df.loc[grp_idx, col] = g.shift(-h).values
                if col not in fwd_cols:
                    fwd_cols.append(col)

        # ── Forward return close-to-close ─────────────────────────────────
        if cfg.get("fwd_return", True):
            for h in range(1, h_max + 1):
                col    = f"fwd_ret_t{h}"
                fwd_c  = g.shift(-h).values
                g_vals = g.values
                ret    = np.where(g_vals > 0, (fwd_c - g_vals) / g_vals, np.nan)
                # Winsorizzazione
                ret    = np.clip(ret, -w_pct, w_pct)
                df.loc[grp_idx, col] = ret
                if col not in fwd_cols:
                    fwd_cols.append(col)

        # ── Max forward return entro H (MFE proxy) ────────────────────────
        if cfg.get("fwd_max_return", True) and cfg.get("fwd_close", True):
            for h in range(1, h_max + 1):
                col = f"fwd_max_ret_H{h}"
                close_cols = [f"fwd_close_t{j}" for j in range(1, h+1) if f"fwd_close_t{j}" in df.columns]
                if close_cols:
                    max_h  = df.loc[grp_idx, close_cols].max(axis=1).values
                    g_vals = g.values
                    mfe    = np.where(g_vals > 0, (max_h - g_vals) / g_vals, np.nan)
                    mfe    = np.clip(mfe, -w_pct, w_pct)
                    df.loc[grp_idx, col] = mfe
                    if col not in fwd_cols:
                        fwd_cols.append(col)

        # ── Min forward return entro H (MAE proxy) ────────────────────────
        if cfg.get("fwd_min_return", True) and cfg.get("fwd_close", True):
            for h in range(1, h_max + 1):
                col = f"fwd_min_ret_H{h}"
                close_cols = [f"fwd_close_t{j}" for j in range(1, h+1) if f"fwd_close_t{j}" in df.columns]
                if close_cols:
                    min_h  = df.loc[grp_idx, close_cols].min(axis=1).values
                    g_vals = g.values
                    mae    = np.where(g_vals > 0, (min_h - g_vals) / g_vals, np.nan)
                    mae    = np.clip(mae, -w_pct, w_pct)
                    df.loc[grp_idx, col] = mae
                    if col not in fwd_cols:
                        fwd_cols.append(col)

    return df, fwd_cols


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER BLOCCO B COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def build_base_dataset(
    df_filtered: pd.DataFrame,
    config:      dict = None,
) -> tuple:
    """
    Esegue B.3 + B.4 in sequenza e restituisce il dataset base completo.

    Parametri
    ---------
    df_filtered : DataFrame OHLCV filtrato (output di apply_universe_filters)
    config      : dizionario configurazione (vedi DEFAULT_FEATURE_CONFIG)

    Ritorna
    -------
    (df_final, feature_cols, fwd_cols)
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(config or {})}

    # B.3 — Feature primitive
    df_feat, feature_cols = build_primitive_features(df_filtered, cfg)

    # B.4 — Forward path
    df_full, fwd_cols = build_forward_path(df_feat, cfg)

    # Rimuovi righe senza ret_1d (prima riga di ogni ticker)
    if "ret_1d" in df_full.columns:
        df_full = df_full.dropna(subset=["ret_1d"]).copy()

    df_full = df_full.reset_index(drop=True)
    return df_full, feature_cols, fwd_cols


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO B.5 — DIAGNOSTICA BASE DATI
# ─────────────────────────────────────────────────────────────────────────────

def get_diagnostics(
    df:           pd.DataFrame,
    feature_cols: list,
    fwd_cols:     list,
) -> dict:
    """
    Produce report diagnostico sul dataset base completato.

    Ritorna dict con:
    - n_rows, n_tickers, date_range
    - feature_stats  : statistiche descrittive delle feature primitive
    - fwd_stats      : statistiche dei forward return per H
    - missing_summary: % missing per colonna
    - column_catalog : lista colonne con tipo (feature/forward/ohlcv)
    """
    if df.empty:
        return {"error": "DataFrame vuoto"}

    diag = {}

    # Info generali
    diag["n_rows"]      = len(df)
    diag["n_tickers"]   = df["ticker"].nunique() if "ticker" in df.columns else 0
    diag["date_min"]    = df["date"].min() if "date" in df.columns else None
    diag["date_max"]    = df["date"].max() if "date" in df.columns else None
    diag["n_feature_cols"] = len(feature_cols)
    diag["n_fwd_cols"]     = len(fwd_cols)

    # Statistiche feature primitive
    feat_present = [c for c in feature_cols if c in df.columns]
    if feat_present:
        diag["feature_stats"] = df[feat_present].describe().round(4).to_dict()

    # Statistiche forward return (solo fwd_ret_tH)
    fwd_ret_cols = [c for c in fwd_cols if c.startswith("fwd_ret_t")]
    if fwd_ret_cols:
        fwd_stats = {}
        for col in fwd_ret_cols:
            s = df[col].dropna()
            fwd_stats[col] = {
                "n":    len(s),
                "mean": round(float(s.mean()), 5),
                "std":  round(float(s.std()),  5),
                "p5":   round(float(s.quantile(0.05)), 5),
                "p95":  round(float(s.quantile(0.95)), 5),
                "hit_rate_0": round(float((s > 0).mean()), 4),
            }
        diag["fwd_stats"] = fwd_stats

    # Missing summary
    diag["missing_pct"] = (df.isnull().mean() * 100).round(2).to_dict()

    # Column catalog
    ohlcv_cols   = ["date","ticker","open","high","low","close","volume"]
    catalog      = []
    for col in df.columns:
        if col in ohlcv_cols:
            ctype = "ohlcv"
        elif col in feature_cols:
            ctype = "feature"
        elif col in fwd_cols:
            ctype = "forward"
        else:
            ctype = "other"
        catalog.append({"col": col, "type": ctype, "n_missing": int(df[col].isnull().sum())})
    diag["column_catalog"] = catalog

    return diag


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_fwd_ret_columns(df: pd.DataFrame) -> list:
    """Ritorna la lista delle colonne fwd_ret_tH presenti nel DataFrame."""
    return sorted(
        [c for c in df.columns if c.startswith("fwd_ret_t") and c[9:].isdigit()],
        key=lambda x: int(x[9:])
    )


def get_feature_columns(df: pd.DataFrame) -> list:
    """Ritorna le colonne che sono feature primitive (non ohlcv, non forward)."""
    ohlcv = {"date","ticker","open","high","low","close","volume"}
    fwd   = set(c for c in df.columns if c.startswith("fwd_"))
    return [c for c in df.columns if c not in ohlcv and c not in fwd]


def summary_stats(df: pd.DataFrame) -> dict:
    """Statistiche rapide sul dataset per la sidebar dell'app."""
    if df.empty:
        return {}
    return {
        "n_tickers": df["ticker"].nunique() if "ticker" in df.columns else 0,
        "n_rows":    len(df),
        "date_min":  df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else "",
        "date_max":  df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else "",
        "n_fwd_ret": len(get_fwd_ret_columns(df)),
        "n_features":len(get_feature_columns(df)),
    }
