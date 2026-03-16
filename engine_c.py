"""
engine_c.py
===========
Modulo Python puro — Blocco C del framework Pattern Discovery.
Nessuna UI. Importato dalla web app Streamlit.

Contiene:
    ENGINE COMUNE
        sample_grid(param_grid, max_comb)
        compute_metrics(df_tk, candidate_col, fwd_col, min_gap)
        aggregate_metrics(results_per_ticker)
        rank_results(gs_results, top_n)
        build_pattern_dataset(df, candidate_fn, params, pattern_id, fwd_col, min_gap)
        build_episode_dataset(df_pattern, pattern_ids)
        blind_diagnostics(df_pat, pattern_id, fwd_col, params, min_gap)

    PATTERN — funzioni candidato e griglia di default
        p1_shock_down_mr        (W, Wz, zsog, cp, H)
        p2_gap_down_rev         (gap_sog, rec, body, H)
        p3_volume_cap_rev       (Wv, vrel, ret, ls, H)
        p4_exhaustion_bar       (Wr, rrel, cp, ls, H)
        p5_volatility_spike     (Wa, ar, ret, H)
        p6_multiday_oversold    (N, daily, Wc, cum, H)

    GRID SEARCH
        run_grid_search(df, pattern_id, param_grid, min_gap, max_comb, min_canoni)

    BLINDATURA
        blind_pattern(df, pattern_id, params, min_gap, fwd_col)
        get_blind_diagnostics(df_pat, pattern_id, fwd_col, min_gap)

    ANALISI POST-BLINDATURA
        analyze_blinded_pattern(df_pat, pattern_id, fwd_col, h_max)
        build_final_pattern_table(blinded_configs)
"""

import pandas as pd
import numpy as np
import random
import itertools
import warnings
from datetime import datetime
from scipy import stats as spstats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────────────────────────────────────

PATTERN_IDS = [
    "shock_down_mr",
    "gap_down_rev",
    "volume_cap_rev",
    "exhaustion_bar",
    "volatility_spike",
    "multiday_oversold",
]

PATTERN_LABELS = {
    "shock_down_mr":     "Shock Down Mean Reversion",
    "gap_down_rev":      "Gap Down Reversal",
    "volume_cap_rev":    "Volume Capitulation Reversal",
    "exhaustion_bar":    "Exhaustion Bar Reversal",
    "volatility_spike":  "Volatility Spike Reversal",
    "multiday_oversold": "Multi-Day Oversold Reversal",
}

# Griglie di default per ogni pattern (opzioni + valori default)
PATTERN_GRIDS = {
    "shock_down_mr": {
        "W":    {"options": [3,5,10,15,20],       "default": [5,10]},
        "Wz":   {"options": [20,40,60,120,252],   "default": [60,120]},
        "zsog": {"options": [-3.0,-2.5,-2.0,-1.5],"default": [-2.5,-2.0]},
        "cp":   {"options": [0.3,0.4,0.5,1.0],   "default": [0.4,1.0]},
        "H":    {"options": [3,5,10,15,20],       "default": [5,10]},
    },
    "gap_down_rev": {
        "gap_sog": {"options": [-0.07,-0.06,-0.05,-0.04,-0.03], "default": [-0.06,-0.05]},
        "rec":     {"options": [0.005,0.01,0.015,0.02],         "default": [0.005,0.01]},
        "body":    {"options": [0.2,0.3,0.4,0.5],               "default": [0.3,0.4]},
        "H":       {"options": [3,5,10,15,20],                  "default": [5,10]},
    },
    "volume_cap_rev": {
        "Wv":   {"options": [10,20,30,60],           "default": [20,30]},
        "vrel": {"options": [1.5,2.0,2.5,3.0],      "default": [2.0,2.5]},
        "ret":  {"options": [-0.03,-0.02,-0.01,-0.005],"default": [-0.02,-0.01]},
        "ls":   {"options": [0.0,0.1,0.2,0.3],      "default": [0.0,0.1]},
        "H":    {"options": [3,5,10,15,20],          "default": [5,10]},
    },
    "exhaustion_bar": {
        "Wr":   {"options": [10,20,30,60],      "default": [20,30]},
        "rrel": {"options": [1.5,2.0,2.5,3.0], "default": [2.0,2.5]},
        "cp":   {"options": [0.3,0.4,0.5],     "default": [0.3,0.4]},
        "ls":   {"options": [0.1,0.2,0.3,0.4], "default": [0.2,0.3]},
        "H":    {"options": [3,5,10,15,20],    "default": [5,10]},
    },
    "volatility_spike": {
        "Wa":  {"options": [10,14,20,30],            "default": [14,20]},
        "ar":  {"options": [1.5,2.0,2.5,3.0],       "default": [2.0,2.5]},
        "ret": {"options": [-0.04,-0.03,-0.02,-0.01],"default": [-0.02,-0.01]},
        "H":   {"options": [3,5,10,15,20],           "default": [5,10]},
    },
    "multiday_oversold": {
        "N":     {"options": [3,4,5,6,7],               "default": [4,5]},
        "daily": {"options": [-0.005,-0.003,-0.001],    "default": [-0.003,-0.001]},
        "Wc":    {"options": [5,10,15,20],              "default": [10,15]},
        "cum":   {"options": [-0.05,-0.04,-0.03,-0.02], "default": [-0.04,-0.03]},
        "H":     {"options": [3,5,10,15,20],            "default": [5,10]},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE COMUNE
# ─────────────────────────────────────────────────────────────────────────────

def sample_grid(param_grid: dict, max_comb: int) -> list:
    """
    Genera tutte le combinazioni dalla griglia di parametri.
    Se il numero supera max_comb, applica un random cap.

    Parametri
    ---------
    param_grid : dict {param_name: [val1, val2, ...]}
    max_comb   : numero massimo di combinazioni (random cap)

    Ritorna
    -------
    Lista di dict, ognuno è una combinazione di parametri.
    """
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    if len(combos) > max_comb:
        combos = random.sample(combos, max_comb)
    return [dict(zip(keys, c)) for c in combos]


def compute_metrics(
    df_tk:         pd.DataFrame,
    candidate_col: str,
    fwd_col:       str,
    min_gap:       int = 1,
) -> dict:
    """
    Calcola metriche di performance di un pattern su un singolo ticker.

    Applica il filtro min_gap: tra un giorno canone e il successivo
    devono passare almeno min_gap giorni di trading.

    Ritorna dict con: n_cand, n_canoni, freq, lift, avg_ret, ratio
    """
    cands  = df_tk[df_tk[candidate_col] == 1].copy()
    n_cand = len(cands)

    if n_cand == 0:
        return {"n_cand":0, "n_canoni":0, "freq":0.0,
                "lift":0.0, "avg_ret":np.nan, "ratio":np.nan}

    # Selezione giorni canone con min_gap
    canoni_idx = []
    last_idx   = -999
    for row_i in cands.index:
        if (row_i - last_idx) >= min_gap:
            canoni_idx.append(row_i)
            last_idx = row_i

    n_canoni = len(canoni_idx)
    n_total  = len(df_tk)

    y_canon = (df_tk.loc[canoni_idx, fwd_col].dropna()
               if fwd_col in df_tk.columns else pd.Series(dtype=float))
    y_all   = (df_tk[fwd_col].dropna()
               if fwd_col in df_tk.columns else pd.Series(dtype=float))

    avg_ret_canon = float(y_canon.mean()) if len(y_canon) > 0 else np.nan
    avg_ret_base  = float(y_all.mean())   if len(y_all)   > 0 else np.nan

    lift = (avg_ret_canon / avg_ret_base
            if (avg_ret_base and not np.isnan(avg_ret_base) and avg_ret_base != 0)
            else np.nan)
    freq  = n_canoni / n_total if n_total > 0 else 0.0
    ratio = n_cand   / n_canoni if n_canoni > 0 else np.nan

    return {
        "n_cand":   n_cand,
        "n_canoni": n_canoni,
        "freq":     round(freq,  5),
        "lift":     round(lift,  4) if not np.isnan(lift)    else np.nan,
        "avg_ret":  round(avg_ret_canon, 5) if not np.isnan(avg_ret_canon) else np.nan,
        "ratio":    round(ratio, 2) if not np.isnan(ratio)   else np.nan,
    }


def aggregate_metrics(results_per_ticker: list) -> dict:
    """
    Aggrega le metriche di più ticker in un unico dict.
    Usa la media dei lift per-ticker (non il lift globale).
    """
    if not results_per_ticker:
        return {"n_cand":0, "n_canoni":0, "freq":0.0,
                "lift":np.nan, "avg_ret":np.nan, "ratio":np.nan}

    df_r = pd.DataFrame(results_per_ticker)
    return {
        "n_cand":   int(df_r["n_cand"].sum()),
        "n_canoni": int(df_r["n_canoni"].sum()),
        "freq":     round(float(df_r["freq"].mean()), 5),
        "lift":     round(float(df_r["lift"].dropna().mean()), 4)
                    if df_r["lift"].notna().any() else np.nan,
        "avg_ret":  round(float(df_r["avg_ret"].dropna().mean()), 5)
                    if df_r["avg_ret"].notna().any() else np.nan,
        "ratio":    round(float(df_r["ratio"].dropna().mean()), 2)
                    if df_r["ratio"].notna().any() else np.nan,
    }


def rank_results(gs_results: list, top_n: int = 5) -> tuple:
    """
    Produce top-N risultati per lift e top-N per n_canoni.

    Ritorna (top_lift_df, top_canoni_df)
    """
    if not gs_results:
        return pd.DataFrame(), pd.DataFrame()

    df_gs = pd.DataFrame(gs_results).dropna(subset=["lift"])
    if df_gs.empty:
        return pd.DataFrame(), pd.DataFrame()

    top_lift   = df_gs.nlargest(top_n, "lift").reset_index(drop=True)
    top_canoni = df_gs.nlargest(top_n, "n_canoni").reset_index(drop=True)
    top_lift.index   = top_lift.index   + 1
    top_canoni.index = top_canoni.index + 1
    return top_lift, top_canoni


def build_pattern_dataset(
    df:           pd.DataFrame,
    candidate_fn,
    params:       dict,
    pattern_id:   str,
    fwd_col:      str,
    min_gap:      int = 1,
) -> pd.DataFrame:
    """
    Applica candidate_fn su ogni ticker e costruisce le colonne:
        candidate_{pattern_id}  : 0/1 candidati grezzi
        score_{pattern_id}      : punteggio/z-score
        is_canone_{pattern_id}  : 0/1 canoni (con min_gap)

    candidate_fn(df_ticker, params) -> (candidate_series, score_series)
    """
    df = df.copy()
    cand_col   = f"candidate_{pattern_id}"
    score_col  = f"score_{pattern_id}"
    canone_col = f"is_canone_{pattern_id}"

    df[cand_col]   = 0
    df[score_col]  = np.nan
    df[canone_col] = 0

    for tk, grp_idx in df.groupby("ticker", sort=False).groups.items():
        grp = df.loc[grp_idx].copy()
        cands, scores = candidate_fn(grp, params)

        df.loc[grp_idx, cand_col]  = cands.values
        df.loc[grp_idx, score_col] = scores.values

        # Applica min_gap per selezionare i canoni
        cand_rows = [idx for idx in grp_idx if df.loc[idx, cand_col] == 1]
        last_idx  = -999
        for row_i in cand_rows:
            if (row_i - last_idx) >= min_gap:
                df.loc[row_i, canone_col] = 1
                last_idx = row_i

    return df


def build_episode_dataset(
    df_pattern:  pd.DataFrame,
    pattern_ids: list,
) -> pd.DataFrame:
    """
    Costruisce il dataset episodi multi-pattern.
    Una riga per ogni giorno canone di ogni pattern.

    Colonne: date, ticker, pattern_id, pattern_label
    """
    rows = []
    for pid in pattern_ids:
        col = f"is_canone_{pid}"
        if col not in df_pattern.columns:
            continue
        sub = df_pattern[df_pattern[col] == 1][["date","ticker"]].copy()
        sub["pattern_id"]    = pid
        sub["pattern_label"] = PATTERN_LABELS.get(pid, pid)
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    return (pd.concat(rows, ignore_index=True)
            .sort_values(["date","ticker"])
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# FUNZIONI CANDIDATO — 6 PATTERN
# ─────────────────────────────────────────────────────────────────────────────

def _p1_candidate(df_tk: pd.DataFrame, params: dict) -> tuple:
    """
    Shock Down Mean Reversion.
    Candidato: z-score del return cumulativo su W giorni < soglia
               AND close nel bottom cp% del range giornaliero.
    Score: z-score (più negativo = più estremo).
    """
    W    = params["W"]
    Wz   = params["Wz"]
    zsog = params["zsog"]
    cp   = params["cp"]

    cum_ret = df_tk["close"].pct_change(W)
    mu      = cum_ret.rolling(Wz).mean()
    sig     = cum_ret.rolling(Wz).std()
    z       = (cum_ret - mu) / sig.replace(0, np.nan)

    cand = (z < zsog).astype(int)
    if cp < 1.0 and "close_pos_in_range" in df_tk.columns:
        cand = cand & (df_tk["close_pos_in_range"] < cp)

    return cand.fillna(0).astype(int), z.fillna(0)


def _p2_candidate(df_tk: pd.DataFrame, params: dict) -> tuple:
    """
    Gap Down Reversal.
    Candidato: gap negativo < soglia AND recupero intraday > rec
               AND body_range_ratio > body (candela con corpo significativo).
    Score: gap_pct (più negativo = gap più ampio).
    """
    gap_sog = params["gap_sog"]
    rec     = params["rec"]
    body    = params["body"]

    gap_pct = df_tk.get("gap_pct", pd.Series(0, index=df_tk.index))
    cand    = gap_pct < gap_sog

    if "close_vs_open" in df_tk.columns:
        cand = cand & (df_tk["close_vs_open"] > rec)

    if "body_range_ratio" in df_tk.columns and body > 0:
        cand = cand & (df_tk["body_range_ratio"] > body)

    score = gap_pct.fillna(0)
    return cand.fillna(False).astype(int), score


def _p3_candidate(df_tk: pd.DataFrame, params: dict) -> tuple:
    """
    Volume Capitulation Reversal.
    Candidato: volume relativo > soglia (capitolazione) AND ret_1d < soglia
               AND lower_shadow_ratio > ls (coda inferiore = supporto).
    Score: volume relativo.
    """
    Wv    = params["Wv"]
    vrel  = params["vrel"]
    ret_s = params["ret"]
    ls    = params["ls"]

    vol_mean = df_tk["volume"].rolling(Wv).mean() if "volume" in df_tk.columns else pd.Series(1, index=df_tk.index)
    vol_rel  = df_tk["volume"] / vol_mean.replace(0, np.nan) if "volume" in df_tk.columns else pd.Series(0, index=df_tk.index)

    cand = (vol_rel > vrel) & (df_tk.get("ret_1d", pd.Series(0, index=df_tk.index)) < ret_s)

    if "lower_shadow_ratio" in df_tk.columns and ls > 0:
        cand = cand & (df_tk["lower_shadow_ratio"] > ls)

    return cand.fillna(False).astype(int), vol_rel.fillna(0)


def _p4_candidate(df_tk: pd.DataFrame, params: dict) -> tuple:
    """
    Exhaustion Bar Reversal.
    Candidato: range giornaliero relativo > soglia (barra ampia = esaurimento)
               AND close nel bottom cp% del range
               AND lower_shadow_ratio > ls.
    Score: range relativo.
    """
    Wr   = params["Wr"]
    rrel = params["rrel"]
    cp   = params["cp"]
    ls   = params["ls"]

    if "range_day" not in df_tk.columns:
        return pd.Series(0, index=df_tk.index), pd.Series(0, index=df_tk.index)

    range_mean = df_tk["range_day"].rolling(Wr).mean()
    range_rel  = df_tk["range_day"] / range_mean.replace(0, np.nan)

    cand = range_rel > rrel

    if "close_pos_in_range" in df_tk.columns:
        cand = cand & (df_tk["close_pos_in_range"] < cp)

    if "lower_shadow_ratio" in df_tk.columns:
        cand = cand & (df_tk["lower_shadow_ratio"] > ls)

    return cand.fillna(False).astype(int), range_rel.fillna(0)


def _p5_candidate(df_tk: pd.DataFrame, params: dict) -> tuple:
    """
    Volatility Spike Reversal.
    Candidato: ATR ratio > soglia (spike di volatilità) AND ret_1d < soglia.
    Score: ATR ratio.
    """
    Wa    = params["Wa"]
    ar    = params["ar"]
    ret_s = params["ret"]

    if "true_range" not in df_tk.columns:
        return pd.Series(0, index=df_tk.index), pd.Series(0, index=df_tk.index)

    atr       = df_tk["true_range"].rolling(Wa).mean()
    atr_ratio = df_tk["true_range"] / atr.replace(0, np.nan)

    cand = (atr_ratio > ar) & (df_tk.get("ret_1d", pd.Series(0, index=df_tk.index)) < ret_s)

    return cand.fillna(False).astype(int), atr_ratio.fillna(0)


def _p6_candidate(df_tk: pd.DataFrame, params: dict) -> tuple:
    """
    Multi-Day Oversold Reversal.
    Candidato: streak di almeno N giorni con ret_1d < daily
               AND return cumulativo su Wc giorni < cum.
    Score: lunghezza dello streak.
    """
    N     = params["N"]
    daily = params["daily"]
    Wc    = params["Wc"]
    cum_s = params["cum"]

    r      = df_tk.get("ret_1d", pd.Series(0, index=df_tk.index))
    is_neg = (r < daily).astype(int)

    # Calcola streak consecutivi
    streak = is_neg.groupby((is_neg != is_neg.shift()).cumsum()).cumsum()

    cum_ret = df_tk["close"].pct_change(Wc)
    cand    = (streak >= N) & (cum_ret < cum_s)

    return cand.fillna(False).astype(int), streak.fillna(0)


# Mappa pattern_id → funzione candidato
CANDIDATE_FNS = {
    "shock_down_mr":     _p1_candidate,
    "gap_down_rev":      _p2_candidate,
    "volume_cap_rev":    _p3_candidate,
    "exhaustion_bar":    _p4_candidate,
    "volatility_spike":  _p5_candidate,
    "multiday_oversold": _p6_candidate,
}


# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    df:         pd.DataFrame,
    pattern_id: str,
    param_grid: dict,
    min_gap:    int = 5,
    max_comb:   int = 300,
    min_canoni: int = 30,
    progress_callback=None,
) -> list:
    """
    Esegue la grid search per un pattern su tutto il dataset.

    Per ogni combinazione di parametri:
    1. Applica la funzione candidato su ogni ticker
    2. Calcola le metriche (lift, n_canoni, freq, avg_ret)
    3. Aggrega i risultati per-ticker

    Parametri
    ---------
    df          : DataFrame con feature (output build_base_dataset)
    pattern_id  : uno dei 6 pattern_ids
    param_grid  : dict {param: [val1, val2, ...]}
    min_gap     : gap minimo tra canoni
    max_comb    : cap combinazioni (random sample se superato)
    min_canoni  : scarta combinazioni con n_canoni < min_canoni
    progress_callback : fn(i, total) opzionale

    Ritorna
    -------
    Lista di dict, ognuno è una combinazione con le sue metriche.
    """
    if pattern_id not in CANDIDATE_FNS:
        raise ValueError(f"pattern_id '{pattern_id}' non riconosciuto. "
                         f"Valori validi: {list(CANDIDATE_FNS.keys())}")

    candidate_fn = CANDIDATE_FNS[pattern_id]
    combos       = sample_grid(param_grid, max_comb)
    results      = []

    for i, params in enumerate(combos):
        H       = params.get("H", 5)
        fwd_col = f"fwd_ret_t{H}"

        if fwd_col not in df.columns:
            if progress_callback:
                progress_callback(i+1, len(combos))
            continue

        # Calcola metriche per-ticker
        per_ticker = []
        for tk, grp_idx in df.groupby("ticker", sort=False).groups.items():
            grp = df.loc[grp_idx].copy()

            # Costruisci colonna candidato temporanea
            cand_col = "__cand_tmp__"
            cands, _ = candidate_fn(grp, params)
            grp[cand_col] = cands.values

            m = compute_metrics(grp, cand_col, fwd_col, min_gap)
            per_ticker.append(m)

        agg = aggregate_metrics(per_ticker)

        # Scarta se n_canoni insufficienti
        if agg["n_canoni"] < min_canoni:
            if progress_callback:
                progress_callback(i+1, len(combos))
            continue

        row = {**params, **agg, "fwd_col": fwd_col}
        results.append(row)

        if progress_callback:
            progress_callback(i+1, len(combos))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BLINDATURA
# ─────────────────────────────────────────────────────────────────────────────

def blind_pattern(
    df:         pd.DataFrame,
    pattern_id: str,
    params:     dict,
    min_gap:    int,
    fwd_col:    str,
) -> tuple:
    """
    Blinda un pattern con i parametri scelti dalla grid search.

    PROCEDURA DI BLINDATURA:
    I parametri vengono scelti PRIMA di guardare la distribuzione
    dei forward return. Una volta scelti, si costruisce il dataset
    definitivo e si misurano le metriche — questo è il momento
    in cui si "apre la busta" e si guarda il risultato.

    Parametri
    ---------
    df         : DataFrame base (output build_base_dataset)
    pattern_id : id del pattern
    params     : parametri scelti dalla grid search
    min_gap    : gap minimo tra canoni
    fwd_col    : colonna forward return target (es. 'fwd_ret_t5')

    Ritorna
    -------
    (df_pat, blind_config)
    - df_pat      : DataFrame con colonne candidate/score/is_canone aggiunte
    - blind_config: dict con metriche di blindatura
    """
    if pattern_id not in CANDIDATE_FNS:
        raise ValueError(f"pattern_id '{pattern_id}' non valido")

    if fwd_col not in df.columns:
        raise ValueError(f"Colonna {fwd_col} non trovata nel dataset. "
                         f"Verifica H_max del Blocco B.")

    candidate_fn = CANDIDATE_FNS[pattern_id]
    canone_col   = f"is_canone_{pattern_id}"

    # Costruisci dataset con le colonne del pattern
    df_pat = build_pattern_dataset(
        df, candidate_fn, params, pattern_id, fwd_col, min_gap
    )

    # Calcola metriche di blindatura
    n_cand  = int(df_pat[f"candidate_{pattern_id}"].sum())
    n_can   = int(df_pat[canone_col].sum())
    freq    = n_can / len(df_pat) if len(df_pat) > 0 else 0.0

    y_can   = df_pat.loc[df_pat[canone_col]==1, fwd_col].dropna()
    y_all   = df_pat[fwd_col].dropna()

    avg_ret = float(y_can.mean()) if len(y_can) > 0 else np.nan
    avg_base= float(y_all.mean()) if len(y_all) > 0 else np.nan
    lift    = (avg_ret / avg_base
               if (avg_base and not np.isnan(avg_base) and avg_base != 0)
               else np.nan)

    blind_config = {
        "pattern_id":  pattern_id,
        "pattern_label": PATTERN_LABELS.get(pattern_id, pattern_id),
        "params":      params,
        "min_gap":     min_gap,
        "fwd_col":     fwd_col,
        "n_cand":      n_cand,
        "n_canoni":    n_can,
        "freq":        round(freq,  5),
        "lift":        round(lift,  4) if not np.isnan(lift)    else None,
        "avg_ret":     round(avg_ret, 5) if not np.isnan(avg_ret) else None,
        "avg_baseline":round(avg_base, 5) if not np.isnan(avg_base) else None,
        "blinded_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return df_pat, blind_config


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICA POST-BLINDATURA
# ─────────────────────────────────────────────────────────────────────────────

def get_blind_diagnostics(
    df_pat:     pd.DataFrame,
    pattern_id: str,
    fwd_col:    str,
    min_gap:    int = 5,
) -> dict:
    """
    Calcola il report diagnostico completo post-blindatura.

    Include:
    - Metriche globali (avg_ret, avg_baseline, lift globale, lift per-ticker)
    - Stabilità temporale: lift per anno
    - Concentrazione: distribuzione episodi per ticker
    - T-test e p-value sulla media dei canoni
    - Hit rate

    Ritorna dict con tutte le metriche.
    """
    canone_col = f"is_canone_{pattern_id}"
    if canone_col not in df_pat.columns or fwd_col not in df_pat.columns:
        return {"error": "Colonne mancanti"}

    df_c  = df_pat[df_pat[canone_col] == 1].copy()
    y_all = df_pat[fwd_col].dropna()
    y_can = df_c[fwd_col].dropna()

    if len(y_can) == 0:
        return {"error": "Nessun episodio canone"}

    avg_ret      = float(y_can.mean())
    avg_baseline = float(y_all.mean()) if len(y_all) > 0 else np.nan
    lift_glob    = (avg_ret / avg_baseline
                    if (avg_baseline and not np.isnan(avg_baseline) and avg_baseline != 0)
                    else np.nan)

    # Lift per-ticker
    per_tk = []
    for tk, grp_idx in df_pat.groupby("ticker", sort=False).groups.items():
        grp = df_pat.loc[grp_idx].copy()
        m   = compute_metrics(grp, canone_col, fwd_col, min_gap)
        m["ticker"] = tk
        per_tk.append(m)
    agg       = aggregate_metrics(per_tk)
    lift_tk   = agg.get("lift", np.nan)
    per_tk_df = pd.DataFrame(per_tk).sort_values("avg_ret", ascending=False) if per_tk else pd.DataFrame()

    # T-test
    t_stat, p_val = spstats.ttest_1samp(y_can, 0)
    hit_rate      = float((y_can > 0).mean())

    # Stabilità temporale
    df_c_yr   = df_c.copy()
    all_yr    = df_pat.copy()
    df_c_yr["year"] = pd.to_datetime(df_c_yr["date"]).dt.year
    all_yr["year"]  = pd.to_datetime(all_yr["date"]).dt.year
    yr_can   = df_c_yr.groupby("year")[fwd_col].mean()
    yr_base  = all_yr.groupby("year")[fwd_col].mean()
    lift_yr  = (yr_can / yr_base).replace([np.inf,-np.inf], np.nan).dropna()
    n_yr     = df_c_yr.groupby("year").size()

    # Concentrazione ticker
    tk_counts = df_c["ticker"].value_counts() if "ticker" in df_c.columns else pd.Series()
    top1_pct  = (tk_counts.iloc[0] / len(df_c) * 100) if len(tk_counts) > 0 else 0

    return {
        "n_canoni":     len(y_can),
        "avg_ret":      avg_ret,
        "avg_baseline": avg_baseline,
        "lift_glob":    lift_glob,
        "lift_tk":      lift_tk,
        "hit_rate":     hit_rate,
        "t_stat":       float(t_stat),
        "p_val":        float(p_val),
        "sig":          float(p_val) < 0.10,
        "lift_yr":      lift_yr,
        "n_yr":         n_yr,
        "per_ticker":   per_tk_df,
        "top1_pct":     top1_pct,
    }


def analyze_blinded_pattern(
    df_pat:     pd.DataFrame,
    pattern_id: str,
    fwd_col:    str,
    h_max:      int = 15,
) -> dict:
    """
    Analisi post-blindatura su tutti gli orizzonti H disponibili.

    Per ogni H costruisce:
    - Distribuzione forward return canoni vs baseline
    - Profilo medio per orizzonte H=1..h_max
    - Metriche complete (lift, hit_rate, t-test)

    Ritorna dict {H: metriche_dict}
    """
    canone_col = f"is_canone_{pattern_id}"
    if canone_col not in df_pat.columns:
        return {}

    df_c = df_pat[df_pat[canone_col] == 1].copy()
    out  = {}

    for H in range(1, h_max + 1):
        fc = f"fwd_ret_t{H}"
        if fc not in df_pat.columns:
            continue

        y_can  = df_c[fc].dropna()
        y_all  = df_pat[fc].dropna()

        if len(y_can) < 5:
            continue

        avg_ret  = float(y_can.mean())
        avg_base = float(y_all.mean()) if len(y_all) > 0 else np.nan
        lift     = (avg_ret / avg_base
                    if (avg_base and not np.isnan(avg_base) and avg_base != 0)
                    else np.nan)
        hr       = float((y_can > 0).mean())
        t, p     = spstats.ttest_1samp(y_can, 0)

        out[H] = {
            "n_canoni":     len(y_can),
            "avg_ret":      avg_ret,
            "avg_baseline": avg_base,
            "lift":         lift,
            "hit_rate":     hr,
            "t_stat":       float(t),
            "p_val":        float(p),
            "sig":          float(p) < 0.10,
            "y_canoni":     y_can * 100,    # in % per i grafici
            "y_baseline":   y_all * 100,
        }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# TABELLA FINALE PATTERN
# ─────────────────────────────────────────────────────────────────────────────

def build_final_pattern_table(blinded_configs: dict) -> pd.DataFrame:
    """
    Costruisce la tabella riepilogativa di tutti i pattern blindati.

    Input: dict {pattern_id: blind_config} (output di blind_pattern)
    Output: DataFrame ordinabile con colonne:
        pattern_id, pattern_label, n_canoni, lift, avg_ret,
        avg_baseline, fwd_col, params
    """
    rows = []
    for pid, cfg in blinded_configs.items():
        rows.append({
            "pattern_id":    pid,
            "pattern_label": cfg.get("pattern_label", pid),
            "n_canoni":      cfg.get("n_canoni", 0),
            "lift":          cfg.get("lift"),
            "avg_ret":       cfg.get("avg_ret"),
            "avg_baseline":  cfg.get("avg_baseline"),
            "hit_rate":      None,  # calcolato dopo in get_blind_diagnostics
            "fwd_col":       cfg.get("fwd_col", ""),
            "params":        str(cfg.get("params", {})),
            "blinded_at":    cfg.get("blinded_at", ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Ordina per lift descrescente
    df = df.sort_values("lift", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_default_grid(pattern_id: str) -> dict:
    """Ritorna la griglia di default per un pattern."""
    grid_def = PATTERN_GRIDS.get(pattern_id, {})
    return {k: v["options"] for k, v in grid_def.items()}


def get_default_values(pattern_id: str) -> dict:
    """Ritorna i valori di default (preselezionati) per un pattern."""
    grid_def = PATTERN_GRIDS.get(pattern_id, {})
    return {k: v["default"] for k, v in grid_def.items()}


def count_combinations(param_grid: dict) -> int:
    """Conta il numero totale di combinazioni nella griglia."""
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


def get_canoni(df_pat: pd.DataFrame, pattern_id: str) -> pd.DataFrame:
    """Estrae solo le righe canone da un DataFrame pattern."""
    col = f"is_canone_{pattern_id}"
    if col not in df_pat.columns:
        return pd.DataFrame()
    return df_pat[df_pat[col] == 1].copy()
