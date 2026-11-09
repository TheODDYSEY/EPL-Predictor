import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, poisson
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EPL Analytics Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  —  dark stadium theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@400;500;600&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0e1a !important;
    color: #e8eaf0 !important;
}
.stApp { background-color: #0a0e1a !important; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 100% !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Nav bar ── */
.nav-bar {
    display: flex; align-items: center; gap: 0;
    background: #0d1220;
    border-bottom: 1px solid #1e2540;
    padding: 0 2rem;
    margin: 0 -2rem 2rem -2rem;
    position: sticky; top: 0; z-index: 999;
}
.nav-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900; font-size: 1.3rem; letter-spacing: 0.05em;
    color: #fff; padding: 1rem 2rem 1rem 0;
    border-right: 1px solid #1e2540;
    white-space: nowrap;
}
.nav-logo span { color: #7c3aed; }

/* ── Section tabs (replace sidebar) ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #1e2540 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: 0.08em !important;
    color: #6b7280 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.9rem 1.6rem !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 3px solid #7c3aed !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 0 0 0 !important; }

/* ── Hero title block ── */
.hero-block {
    padding: 2.5rem 0 1.5rem 0;
}
.hero-season-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.15em;
    color: #7c3aed; text-transform: uppercase; margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 4rem; font-weight: 900; line-height: 0.95;
    color: #fff; margin: 0 0 0.5rem 0;
}
.hero-title span { color: #a78bfa; }
.hero-subtitle {
    font-size: 0.95rem; color: #6b7280; margin: 0;
}
.live-badge {
    display: inline-block;
    background: #14532d; color: #4ade80;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 0.2rem 0.6rem; border-radius: 2rem;
    margin-top: 0.8rem;
}

/* ── Stat hero cards ── */
.stat-card {
    background: #111827;
    border: 1px solid #1e2540;
    border-radius: 8px;
    padding: 1.4rem 1.2rem;
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.stat-card.purple::before { background: #7c3aed; }
.stat-card.pink::before   { background: #db2777; }
.stat-card.blue::before   { background: #2563eb; }
.stat-card.green::before  { background: #059669; }
.stat-card.orange::before { background: #d97706; }

.stat-card .icon { font-size: 1.4rem; margin-bottom: 0.6rem; }
.stat-card .value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.6rem; font-weight: 900; color: #fff;
    line-height: 1; margin: 0;
}
.stat-card .label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    color: #6b7280; text-transform: uppercase; margin: 0.3rem 0 0 0;
}
.stat-card .sub {
    font-size: 0.8rem; color: #a78bfa; margin-top: 0.4rem;
}

/* ── Champion card ── */
.champion-card {
    background: linear-gradient(135deg, #1a0a3d 0%, #111827 60%);
    border: 1px solid #4c1d95;
    border-radius: 8px;
    padding: 1.6rem;
}
.champion-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    color: #f59e0b; text-transform: uppercase;
}
.champion-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.8rem; font-weight: 900; color: #fff;
    margin: 0.4rem 0 0.8rem 0;
}
.champion-pts {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3rem; font-weight: 900; color: #a78bfa;
}
.champion-stat-row { display: flex; gap: 1.2rem; margin-top: 0.8rem; }
.ch-stat { text-align: center; }
.ch-stat .v { font-weight: 700; font-size: 1.2rem; color: #fff; }
.ch-stat .l { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; }

/* ── Club rank cards ── */
.club-card {
    background: #111827;
    border: 1px solid #1e2540;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    display: flex; align-items: center; gap: 1rem;
}
.club-rank {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem; font-weight: 900; color: #4b5563;
    min-width: 2rem; text-align: center;
}
.club-rank.top3 { color: #a78bfa; }
.club-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem; font-weight: 700; color: #fff;
}
.club-pts { font-size: 0.85rem; color: #a78bfa; font-weight: 600; }
.form-dots { display: flex; gap: 3px; margin-top: 0.3rem; }
.form-dot {
    width: 10px; height: 10px; border-radius: 50%;
}
.form-dot.W { background: #4ade80; }
.form-dot.D { background: #facc15; }
.form-dot.L { background: #f87171; }

/* ── Match predictor card ── */
.match-card {
    background: #111827;
    border: 1px solid #1e2540;
    border-radius: 10px;
    padding: 2rem;
}
.match-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
    color: #6b7280; text-transform: uppercase; margin-bottom: 1.5rem;
}
.score-display {
    display: flex; align-items: center; justify-content: center;
    gap: 1.5rem;
}
.score-team { text-align: center; flex: 1; }
.score-team-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.2rem; font-weight: 700; color: #9ca3af;
}
.score-goals {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 4.5rem; font-weight: 900; color: #fff; line-height: 1;
}
.score-goals.home { color: #a78bfa; }
.score-goals.away { color: #38bdf8; }
.score-divider {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem; color: #374151; font-weight: 900;
}

/* ── Probability bar ── */
.prob-bar-wrap { margin: 1.5rem 0 0.5rem 0; }
.prob-labels { display: flex; justify-content: space-between; margin-bottom: 0.4rem; }
.prob-label { font-size: 0.8rem; font-weight: 600; }
.prob-bar-track {
    height: 8px; background: #1f2937; border-radius: 4px; overflow: hidden;
    display: flex;
}
.prob-seg-h { height: 100%; background: #7c3aed; transition: width 0.6s ease; }
.prob-seg-d { height: 100%; background: #374151; }
.prob-seg-a { height: 100%; background: #2563eb; transition: width 0.6s ease; }
.prob-pct-row { display: flex; justify-content: space-between; margin-top: 0.4rem; }
.prob-pct { font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; font-weight: 700; }
.prob-pct.home { color: #a78bfa; }
.prob-pct.draw { color: #6b7280; }
.prob-pct.away { color: #38bdf8; }

/* ── Model comparison grid ── */
.model-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.8rem; margin-top: 1.2rem; }
.model-cell {
    background: #0d1220;
    border: 1px solid #1e2540;
    border-radius: 6px;
    padding: 0.8rem;
    text-align: center;
}
.model-cell .mname { font-size: 0.68rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; }
.model-cell .mscore {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem; font-weight: 900; color: #fff; margin: 0.3rem 0;
}
.model-cell .mrmse { font-size: 0.7rem; color: #4b5563; }

/* ── Table styles ── */
.league-table-wrap { overflow-x: auto; }
table.league-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.88rem;
}
table.league-table th {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    color: #4b5563; text-transform: uppercase;
    padding: 0.6rem 0.8rem; border-bottom: 1px solid #1e2540;
    text-align: right; white-space: nowrap;
}
table.league-table th:first-child,
table.league-table th:nth-child(2) { text-align: left; }
table.league-table td {
    padding: 0.7rem 0.8rem; border-bottom: 1px solid #111827;
    text-align: right; white-space: nowrap;
}
table.league-table td:first-child { text-align: center; color: #6b7280; font-weight: 700; }
table.league-table td:nth-child(2) { text-align: left; font-weight: 600; color: #fff; }
table.league-table tr:hover td { background: #131929; }
.zone-cl   { border-left: 3px solid #2563eb; }
.zone-el   { border-left: 3px solid #f59e0b; }
.zone-rel  { border-left: 3px solid #dc2626; }
.pts-bold  { font-family: 'Barlow Condensed', sans-serif; font-size: 1.05rem; font-weight: 900; color: #a78bfa; }

/* ── H2H stats ── */
.h2h-bar-wrap { margin: 0.8rem 0; }
.h2h-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: #9ca3af; margin-bottom: 0.3rem; }
.h2h-bar-track { height: 6px; background: #1f2937; border-radius: 3px; overflow: hidden; display: flex; }
.h2h-seg-a { height: 100%; background: #a78bfa; }
.h2h-seg-d { height: 100%; background: #374151; }
.h2h-seg-b { height: 100%; background: #38bdf8; }

/* ── Section headings ── */
.section-heading {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem; font-weight: 900; color: #fff;
    margin: 2rem 0 1rem 0;
    display: flex; align-items: center; gap: 0.8rem;
}
.section-heading::after {
    content: ''; flex: 1;
    height: 1px; background: #1e2540;
}
.section-pill {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    color: #a78bfa; background: #1e1040;
    padding: 0.15rem 0.5rem; border-radius: 2rem; text-transform: uppercase;
}

/* ── Selectbox / input overrides ── */
.stSelectbox label, .stSlider label { color: #9ca3af !important; font-size: 0.8rem !important; }
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border-color: #1e2540 !important;
    color: #fff !important;
}
[data-testid="stSelectbox"] svg { fill: #6b7280 !important; }

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e2540;
    border-radius: 8px;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'Barlow Condensed', sans-serif !important; font-size: 2rem !important; }

/* ── Plotly chart background ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Divider ── */
hr { border-color: #1e2540 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }

/* ── Info / warning box ── */
.info-box {
    background: #0f172a; border-left: 3px solid #7c3aed;
    padding: 0.8rem 1rem; border-radius: 0 4px 4px 0;
    font-size: 0.85rem; color: #9ca3af; margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA LOADING & CACHING
# ══════════════════════════════════════════════════════════════

import os

# ── Engineered CSVs (features used by ML models) ──
SEASON_MAP = {
    '2013-14': 'engg_data/epl1314.csv',
    '2014-15': 'engg_data/epl1415.csv',
    '2015-16': 'engg_data/epl1516.csv',
    '2016-17': 'engg_data/epl1617.csv',
    '2017-18': 'engg_data/epl1718.csv',
    '2018-19': 'engg_data/epl1819.csv',
    '2019-20': 'engg_data/epl1920.csv',
    '2020-21': 'engg_data/epl2021.csv',
    '2021-22': 'engg_data/epl2122.csv',
    '2022-23': 'engg_data/epl2223.csv',
    '2023-24': 'engg_data/epl2324.csv',
    '2024-25': 'engg_data/epl2425.csv',
    '2025-26': 'engg_data/epl2526.csv',
    '2026-27': 'engg_data/epl2627.csv',
}

# ── Raw CSVs/Excel from football-data.co.uk (contain FTR, HS, AS, HST, AST) ──
RAW_MAP = {
    '2013-14': 'data/epl1314.csv',
    '2014-15': 'data/epl1415.csv',
    '2015-16': 'data/epl1516.csv',
    '2016-17': 'data/epl1617.csv',
    '2017-18': 'data/epl1718.csv',
    '2018-19': 'data/epl1819.csv',
    '2019-20': 'data/epl1920.csv',
    '2020-21': 'data/epl2021.csv',
    '2021-22': 'data/epl2122.csv',
    '2022-23': 'data/epl2223.csv',
    '2023-24': 'data/epl2324.csv',
    '2024-25': 'data/epl2425.csv',
    '2025-26': 'data/epl2526.csv',
    '2026-27': 'data/epl2627.csv',
}

RAW_COLS_NEEDED = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']


# ── Live fetch URLs from football-data.co.uk (free, no API key) ──
# Used when no local file exists for a season. Refreshes every 6 hours.
LIVE_URLS = {
    '2025-26': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
    '2026-27': 'https://www.football-data.co.uk/mmz4281/2627/E0.csv',
}

@st.cache_data(ttl=21600, show_spinner=False)   # 6-hour cache
def _fetch_live(season_key: str) -> pd.DataFrame | None:
    """Download the current season CSV from football-data.co.uk."""
    url = LIVE_URLS.get(season_key)
    if not url:
        return None
    try:
        import urllib.request
        import io
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(raw), on_bad_lines='skip')
        if {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}.issubset(df.columns):
            return df
    except Exception:
        pass
    return None


def _read_file(path: str):
    for ext in ['', '.csv', '.xlsx', '.xls']:
        full = path if ext == '' else (path.rsplit('.', 1)[0] + ext if '.' in path else path + ext)
        if os.path.exists(full):
            try:
                if full.endswith(('.xlsx', '.xls')):
                    return pd.read_excel(full)
                return pd.read_csv(full, encoding='utf-8', on_bad_lines='skip')
            except Exception:
                try:
                    return pd.read_csv(full, encoding='latin-1', on_bad_lines='skip')
                except Exception:
                    pass
    return None


def _normalise_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names from football-data.co.uk raw files."""
    df = df.copy()
    rename = {}
    if 'HG' in df.columns and 'FTHG' not in df.columns:
        rename['HG'] = 'FTHG'
    if 'AG' in df.columns and 'FTAG' not in df.columns:
        rename['AG'] = 'FTAG'
    if rename:
        df = df.rename(columns=rename)
    if 'FTR' in df.columns:
        df = df[df['FTR'].isin(['H', 'A', 'D'])].copy()
    # ── FIX: assign MW so chart blocks are not skipped ──
    if 'MW' not in df.columns:
        df = df.reset_index(drop=True)
        df['MW'] = (df.index // 10) + 1
    return df


@st.cache_data(show_spinner=False)
def load_all_data():
    engg  = {}
    raw   = {}
    merged = {}

    for key, path in SEASON_MAP.items():
        df = _read_file(path)
        if df is not None:
            engg[key] = df

    for key, path in RAW_MAP.items():
        df = _read_file(path)
        if df is None:
            stem = path.rsplit('.', 1)[0]
            for ext in ['.xlsx', '.xls']:
                df = _read_file(stem + ext)
                if df is not None:
                    break
        # If still no local file, try fetching live from football-data.co.uk
        if df is None and key in LIVE_URLS:
            df = _fetch_live(key)
        if df is not None:
            raw[key] = _normalise_raw(df)

    data_dir = 'data'
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith(('.xlsx', '.xls', '.csv')):
                stem = fname.rsplit('.', 1)[0].lower()
                matched_key = None
                for key, path in RAW_MAP.items():
                    if os.path.splitext(os.path.basename(path))[0].lower() == stem:
                        matched_key = key
                        break
                if matched_key and matched_key not in raw:
                    df = _read_file(os.path.join(data_dir, fname))
                    if df is not None:
                        raw[matched_key] = _normalise_raw(df)

    all_keys = sorted(set(engg.keys()) | set(raw.keys()))
    for key in all_keys:
        if key in engg and key in raw:
            raw_cols = [c for c in RAW_COLS_NEEDED if c in raw[key].columns]
            raw_slim = raw[key][raw_cols].copy().reset_index(drop=True)
            eng_df   = engg[key].copy().reset_index(drop=True)
            cols_to_add = [c for c in raw_slim.columns
                           if c not in eng_df.columns or c in ('FTR','HS','AS','HST','AST')]
            eng_df = eng_df.drop(columns=[c for c in cols_to_add if c in eng_df.columns
                                           and c not in ('HomeTeam','AwayTeam')], errors='ignore')
            merged_df = eng_df.merge(
                raw_slim[['HomeTeam','AwayTeam'] + [c for c in cols_to_add
                           if c not in ('HomeTeam','AwayTeam')]],
                on=['HomeTeam','AwayTeam'],
                how='left',
            )
            # Carry MW from raw into the merged frame if missing
            if 'MW' not in merged_df.columns and 'MW' in raw[key].columns:
                merged_df = merged_df.merge(
                    raw[key][['HomeTeam','AwayTeam','MW']],
                    on=['HomeTeam','AwayTeam'],
                    how='left',
                )
            merged[key] = merged_df

        elif key in engg:
            df = engg[key].copy()
            if 'FTR' not in df.columns and 'FTHG' in df.columns and 'FTAG' in df.columns:
                df['FTR'] = np.where(df['FTHG'] > df['FTAG'], 'H',
                             np.where(df['FTAG'] > df['FTHG'], 'A', 'D'))
            for col in ['HS','AS','HST','AST']:
                if col not in df.columns:
                    df[col] = 0
            if 'MW' not in df.columns:
                df = df.reset_index(drop=True)
                df['MW'] = (df.index // 10) + 1
            merged[key] = df

        else:
            df = raw[key].copy()
            if 'FTR' not in df.columns and 'FTHG' in df.columns and 'FTAG' in df.columns:
                df['FTR'] = np.where(df['FTHG'] > df['FTAG'], 'H',
                             np.where(df['FTAG'] > df['FTHG'], 'A', 'D'))
            merged[key] = df

    return merged, raw


data, raw_data = load_all_data()
available_seasons = sorted(data.keys())

if not available_seasons:
    st.error("⚠️  No data files found. Place engineered CSVs in `engg_data/` and raw files (CSV or Excel) in `data/`.")
    st.stop()

latest_season = available_seasons[-1]


# ══════════════════════════════════════════════════════════════
#  FEATURE SELECTION  (cached per season)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_best_features(df):
    corr = df.corr(numeric_only=True)
    hg = corr['FTHG'].abs().sort_values(ascending=False)[1:7].index.tolist()
    ag = corr['FTAG'].abs().sort_values(ascending=False)[1:7].index.tolist()
    return hg, ag


# ══════════════════════════════════════════════════════════════
#  CORE STATS FUNCTIONS
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(df['HomeTeam'].unique())
    rows  = []
    for t in teams:
        hm = df[df['HomeTeam'] == t]
        aw = df[df['AwayTeam'] == t]
        last_h = hm.iloc[-1] if len(hm) else None
        last_a = aw.iloc[-1] if len(aw) else None

        gf  = (hm['FTHG'].sum() if last_h is not None else 0) + \
              (aw['FTAG'].sum() if last_a is not None else 0)
        ga  = (hm['FTAG'].sum() if last_h is not None else 0) + \
              (aw['FTHG'].sum() if last_a is not None else 0)

        hw  = len(hm[hm['FTHG'] > hm['FTAG']]) if last_h is not None else 0
        hd  = len(hm[hm['FTHG'] == hm['FTAG']]) if last_h is not None else 0
        hl  = len(hm[hm['FTHG'] < hm['FTAG']]) if last_h is not None else 0
        aw_ = len(aw[aw['FTAG'] > aw['FTHG']]) if last_a is not None else 0
        ad  = len(aw[aw['FTAG'] == aw['FTHG']]) if last_a is not None else 0
        al  = len(aw[aw['FTAG'] < aw['FTHG']]) if last_a is not None else 0

        w, d, l = hw + aw_, hd + ad, hl + al
        pts = 3*w + d
        gd  = gf - ga
        mp  = w + d + l

        rows.append({
            'Team': t, 'MP': mp, 'W': w, 'D': d, 'L': l,
            'GF': gf, 'GA': ga, 'GD': gd, 'Pts': pts,
            'HW': hw, 'HD': hd, 'HL': hl,
            'AW': aw_, 'AD': ad, 'AL': al,
        })

    tbl = pd.DataFrame(rows)
    tbl = tbl.sort_values(['Pts','GD','GF'], ascending=False).reset_index(drop=True)
    tbl.index += 1
    return tbl


@st.cache_data(show_spinner=False)
def team_form(df: pd.DataFrame, team: str, last_n: int = 5) -> list:
    has_ftr = 'FTR' in df.columns
    results = []
    for _, row in df.iterrows():
        if row['HomeTeam'] == team:
            if has_ftr and pd.notna(row.get('FTR')):
                ftr = row['FTR']
                results.append('W' if ftr == 'H' else ('D' if ftr == 'D' else 'L'))
            elif 'FTHG' in df.columns and 'FTAG' in df.columns:
                hg, ag = row['FTHG'], row['FTAG']
                results.append('W' if hg > ag else ('D' if hg == ag else 'L'))
        elif row['AwayTeam'] == team:
            if has_ftr and pd.notna(row.get('FTR')):
                ftr = row['FTR']
                results.append('W' if ftr == 'A' else ('D' if ftr == 'D' else 'L'))
            elif 'FTHG' in df.columns and 'FTAG' in df.columns:
                hg, ag = row['FTHG'], row['FTAG']
                results.append('W' if ag > hg else ('D' if ag == hg else 'L'))
    return results[-last_n:]


@st.cache_data(show_spinner=False)
def h2h_stats(season_key: str, team_a: str, team_b: str) -> dict:
    records = []
    for s_key, df in data.items():
        has_ftr = 'FTR' in df.columns
        m1 = df[(df['HomeTeam'] == team_a) & (df['AwayTeam'] == team_b)]
        m2 = df[(df['HomeTeam'] == team_b) & (df['AwayTeam'] == team_a)]
        for _, row in m1.iterrows():
            hg, ag = int(row['FTHG']), int(row['FTAG'])
            ftr = row['FTR'] if has_ftr and pd.notna(row.get('FTR')) else ('H' if hg>ag else ('A' if ag>hg else 'D'))
            records.append({'season': s_key, 'home': team_a, 'away': team_b,
                            'hg': hg, 'ag': ag, 'ftr': ftr})
        for _, row in m2.iterrows():
            hg, ag = int(row['FTHG']), int(row['FTAG'])
            ftr = row['FTR'] if has_ftr and pd.notna(row.get('FTR')) else ('H' if hg>ag else ('A' if ag>hg else 'D'))
            records.append({'season': s_key, 'home': team_b, 'away': team_a,
                            'hg': hg, 'ag': ag, 'ftr': ftr})

    if not records:
        return {}

    total = len(records)
    wins_a = sum(1 for r in records if (r['home']==team_a and r['ftr']=='H') or (r['away']==team_a and r['ftr']=='A'))
    wins_b = sum(1 for r in records if (r['home']==team_b and r['ftr']=='H') or (r['away']==team_b and r['ftr']=='A'))
    draws  = sum(1 for r in records if r['ftr']=='D')
    goals_a = sum(r['hg'] if r['home']==team_a else r['ag'] for r in records)
    goals_b = sum(r['hg'] if r['home']==team_b else r['ag'] for r in records)

    return {
        'total': total, 'wins_a': wins_a, 'wins_b': wins_b, 'draws': draws,
        'goals_a': goals_a, 'goals_b': goals_b, 'records': records,
    }


# ══════════════════════════════════════════════════════════════
#  PREDICTION ENGINE  (Poisson + ML)
# ══════════════════════════════════════════════════════════════

def rmse(pred, true):
    return float(np.sqrt(((np.array(pred) - np.array(true)) ** 2).mean()))


def poisson_predict(pois_row):
    hthgs = float(pois_row['HTHGS'])
    atagc = float(pois_row['ATAGC'])
    atags = float(pois_row['ATAGS'])
    hthgc = float(pois_row['HTHGC'])
    avg_h = float(pois_row['avg_HGPG'])
    avg_a = float(pois_row['avg_AGPG'])

    lam_h = (hthgs * atagc / avg_h) if avg_h > 0 else 1.2
    lam_a = (atags * hthgc / avg_a) if avg_a > 0 else 1.0

    max_goals = 9
    score_matrix = np.outer(
        [poisson.pmf(i, lam_h) for i in range(max_goals)],
        [poisson.pmf(i, lam_a) for i in range(max_goals)],
    )
    p_home = float(np.sum(np.tril(score_matrix, -1)))
    p_away = float(np.sum(np.triu(score_matrix, 1)))
    p_draw = float(np.sum(np.diag(score_matrix)))

    return lam_h, lam_a, p_home, p_draw, p_away


@st.cache_resource
def get_models():
    return {
        'rf_hg': RandomForestRegressor(n_estimators=94, max_depth=2, n_jobs=-1, random_state=42),
        'rf_ag': RandomForestRegressor(n_estimators=280, max_depth=2, n_jobs=-1, random_state=42),
        'xgb_hg': XGBRegressor(n_estimators=7, max_depth=2, verbosity=0),
        'xgb_ag': XGBRegressor(n_estimators=13, max_depth=1, verbosity=0),
    }


def get_poisson_features(df, home_team, away_team, match_position=None):
    required = [
        'HTHGS', 'ATAGS', 'HTHGC',
        'ATAGC', 'avg_HGPG', 'avg_AGPG'
    ]

    if all(col in df.columns for col in required):
        if match_position is not None:
            existing_row = df.iloc[match_position]
            values = existing_row[required]
            if values.notna().all():
                return values.astype(float)

    history = (
        df.iloc[:match_position].copy()
        if match_position is not None
        else df.copy()
    )

    if history.empty:
        history = pd.DataFrame(columns=df.columns)

    for col in ['FTHG', 'FTAG']:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors='coerce')

    home_history = history[history['HomeTeam'] == home_team]
    away_history = history[history['AwayTeam'] == away_team]

    league_home_avg = (
        history['FTHG'].mean() if not history.empty else np.nan
    )
    league_away_avg = (
        history['FTAG'].mean() if not history.empty else np.nan
    )

    if pd.isna(league_home_avg) or league_home_avg <= 0:
        league_home_avg = 1.35
    if pd.isna(league_away_avg) or league_away_avg <= 0:
        league_away_avg = 1.10

    home_scored = (
        home_history['FTHG'].mean()
        if not home_history.empty else league_home_avg
    )
    home_conceded = (
        home_history['FTAG'].mean()
        if not home_history.empty else league_away_avg
    )
    away_scored = (
        away_history['FTAG'].mean()
        if not away_history.empty else league_away_avg
    )
    away_conceded = (
        away_history['FTHG'].mean()
        if not away_history.empty else league_home_avg
    )

    values = {
        'HTHGS': home_scored,
        'ATAGS': away_scored,
        'HTHGC': home_conceded,
        'ATAGC': away_conceded,
        'avg_HGPG': league_home_avg,
        'avg_AGPG': league_away_avg,
    }

    return pd.Series(values, dtype=float)


def run_prediction(season_key: str, home_team: str, away_team: str):
    df = data[season_key]

    hg_feats, ag_feats = get_best_features(df)
    models = get_models()

    X_hg = df[hg_feats]
    X_ag = df[ag_feats]
    y_hg = df['FTHG']
    y_ag = df['FTAG']

    match_row = df[
        (df['HomeTeam'] == home_team) &
        (df['AwayTeam'] == away_team)
    ]

    match_played = not match_row.empty

    if match_played:
        true_hg = int(match_row['FTHG'].values[0])
        true_ag = int(match_row['FTAG'].values[0])

        hg_sample = match_row[hg_feats].values
        ag_sample = match_row[ag_feats].values

        match_position = np.flatnonzero(
            (
                (df['HomeTeam'] == home_team) &
                (df['AwayTeam'] == away_team)
            ).to_numpy()
        )[0]

        pois_row = get_poisson_features(
            df, home_team, away_team, match_position=match_position
        )

    else:
        true_hg = None
        true_ag = None

        ht_rows = df[df['HomeTeam'] == home_team]
        at_rows = df[df['AwayTeam'] == away_team]

        if ht_rows.empty or at_rows.empty:
            return None

        ht_last = ht_rows.iloc[-1]
        at_last = at_rows.iloc[-1]

        hg_sample = ht_last[hg_feats].values.reshape(1, -1)
        ag_sample = at_last[ag_feats].values.reshape(1, -1)

        pois_row = get_poisson_features(df, home_team, away_team)

    lam_h, lam_a, p_home, p_draw, p_away = poisson_predict(pois_row)

    models['rf_hg'].fit(X_hg, y_hg)
    models['rf_ag'].fit(X_ag, y_ag)

    rf_hg = float(
        models['rf_hg'].predict(
            hg_sample.reshape(1, -1) if hg_sample.ndim == 1 else hg_sample
        )[0]
    )
    rf_ag = float(
        models['rf_ag'].predict(
            ag_sample.reshape(1, -1) if ag_sample.ndim == 1 else ag_sample
        )[0]
    )

    models['xgb_hg'].fit(X_hg, y_hg)
    models['xgb_ag'].fit(X_ag, y_ag)

    xgb_hg = float(
        models['xgb_hg'].predict(
            hg_sample.reshape(1, -1) if hg_sample.ndim == 1 else hg_sample
        )[0]
    )
    xgb_ag = float(
        models['xgb_ag'].predict(
            ag_sample.reshape(1, -1) if ag_sample.ndim == 1 else ag_sample
        )[0]
    )

    ens_hg = (rf_hg + xgb_hg) / 2
    ens_ag = (rf_ag + xgb_ag) / 2

    return {
        'match_played': match_played,
        'true_hg': true_hg,
        'true_ag': true_ag,
        'poisson': (lam_h, lam_a),
        'probs': (p_home, p_draw, p_away),
        'rf': (rf_hg, rf_ag),
        'xgb': (xgb_hg, xgb_ag),
        'ensemble': (ens_hg, ens_ag),
        'rmse': {
            'poisson': (
                rmse(lam_h, true_hg) if match_played else None,
                rmse(lam_a, true_ag) if match_played else None
            ),
            'rf': (
                rmse(rf_hg, true_hg) if match_played else None,
                rmse(rf_ag, true_ag) if match_played else None
            ),
            'xgb': (
                rmse(xgb_hg, true_hg) if match_played else None,
                rmse(xgb_ag, true_ag) if match_played else None
            ),
            'ensemble': (
                rmse(ens_hg, true_hg) if match_played else None,
                rmse(ens_ag, true_ag) if match_played else None
            )
        }
    }


# ══════════════════════════════════════════════════════════════
#  HTML HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════

def stat_card(icon, value, label, sub="", color="purple"):
    return f"""
    <div class="stat-card {color}">
      <div class="icon">{icon}</div>
      <div class="value">{value}</div>
      <div class="label">{label}</div>
      {"<div class='sub'>"+sub+"</div>" if sub else ""}
    </div>"""


def form_dots_html(form_list):
    dots = "".join(f'<div class="form-dot {r}"></div>' for r in form_list)
    return f'<div class="form-dots">{dots}</div>'


def section_heading(title, pill=None):
    pill_html = f'<span class="section-pill">{pill}</span>' if pill else ''
    st.markdown(f'<div class="section-heading">{pill_html} {title}</div>', unsafe_allow_html=True)


def prob_bar(p_home, p_draw, p_away, team_a, team_b):
    ph = int(p_home * 100)
    pd_ = int(p_draw * 100)
    pa = 100 - ph - pd_
    return f"""
    <div class="prob-bar-wrap">
      <div class="prob-labels">
        <span class="prob-label" style="color:#a78bfa">{team_a}</span>
        <span class="prob-label" style="color:#9ca3af">Draw</span>
        <span class="prob-label" style="color:#38bdf8">{team_b}</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-seg-h" style="width:{ph}%"></div>
        <div class="prob-seg-d" style="width:{pd_}%"></div>
        <div class="prob-seg-a" style="width:{pa}%"></div>
      </div>
      <div class="prob-pct-row">
        <span class="prob-pct home">{ph}%</span>
        <span class="prob-pct draw">{pd_}%</span>
        <span class="prob-pct away">{pa}%</span>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════
#  NAV BAR
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">⚽ EPL <span>Analytics</span></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  GLOBAL SEASON SELECTOR
# ══════════════════════════════════════════════════════════════

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    st.markdown(f"""
    <div class="hero-block">
      <div class="hero-season-label">Season Review</div>
      <div class="hero-title">PREMIER<br><span>LEAGUE</span></div>
      <div class="hero-subtitle">Expected goals · Machine learning predictions · Season statistics</div>
      <div class="live-badge">● DATA LOADED — {len(available_seasons)} SEASONS &nbsp;·&nbsp; LIVE AUTO-FETCH ON</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    selected_season = st.selectbox(
        "Season",
        options=available_seasons[::-1],
        index=0,
        key="global_season",
        label_visibility="visible",
    )

season_df = data[selected_season]
season_tbl = build_season_summary(season_df)
all_teams  = sorted(season_df['HomeTeam'].unique())


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════

tab_home, tab_predictor, tab_table, tab_team, tab_h2h = st.tabs([
    "🏠  Season Home",
    "🎯  Match Predictor",
    "📊  League Table",
    "🔍  Team Deep Dive",
    "⚔️  Head to Head",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — SEASON HOME
# ══════════════════════════════════════════════════════════════

with tab_home:

    total_goals   = int(season_df['FTHG'].sum() + season_df['FTAG'].sum())
    total_matches = len(season_df)
    avg_gpg       = total_goals / max(total_matches, 1)
    total_mw      = int(season_df['MW'].max()) if 'MW' in season_df.columns else 38

    if 'FTR' in season_df.columns:
        home_wins = int((season_df['FTR'] == 'H').sum())
        away_wins = int((season_df['FTR'] == 'A').sum())
        draws     = int((season_df['FTR'] == 'D').sum())
    else:
        home_wins = int((season_df['FTHG'] > season_df['FTAG']).sum())
        away_wins = int((season_df['FTAG'] > season_df['FTHG']).sum())
        draws     = int((season_df['FTHG'] == season_df['FTAG']).sum())

    top_team_row = season_tbl.iloc[0]
    champion_name = top_team_row['Team']
    champion_pts  = int(top_team_row['Pts'])
    champion_w    = int(top_team_row['W'])
    champion_d    = int(top_team_row['D'])
    champion_l    = int(top_team_row['L'])
    champion_gf   = int(top_team_row['GF'])

    best_attack  = season_tbl.loc[season_tbl['GF'].idxmax(), 'Team'] if not season_tbl.empty else 'N/A'
    best_defence = season_tbl.loc[season_tbl['GA'].idxmin(), 'Team'] if not season_tbl.empty else 'N/A'
    most_clean   = season_tbl.loc[(season_tbl['GF'] - season_tbl['GD']).idxmin(), 'Team'] if not season_tbl.empty else 'N/A'

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("⚽", f"{total_goals:,}", "Total Goals", f"Avg {avg_gpg:.2f} per match", "purple"), unsafe_allow_html=True)
    c2.markdown(stat_card("🏟️", f"{total_matches}", "Matches Played", f"{total_mw} Match Weeks", "blue"), unsafe_allow_html=True)
    c3.markdown(stat_card("🏆", f"{champion_gf}", "Top Attack Goals", champion_name, "pink"), unsafe_allow_html=True)
    c4.markdown(stat_card("🧤", "—", "Clean Sheets Leader", best_defence, "green"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_ch, col_top5 = st.columns([1, 2])

    with col_ch:
        section_heading("Season Champion", "🥇")
        st.markdown(f"""
        <div class="champion-card">
          <div class="champion-label">🏆 League Champion</div>
          <div class="champion-name">{champion_name.upper()}</div>
          <div>Final Points: <span class="champion-pts">{champion_pts}</span></div>
          <div class="champion-stat-row">
            <div class="ch-stat"><div class="v">{champion_w}</div><div class="l">Wins</div></div>
            <div class="ch-stat"><div class="v">{champion_d}</div><div class="l">Draws</div></div>
            <div class="ch-stat"><div class="v">{champion_l}</div><div class="l">Losses</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        section_heading("Season Snapshot", "📈")
        snap_c1, snap_c2 = st.columns(2)
        snap_c1.markdown(stat_card("🏠", f"{home_wins}", "Home Wins", "", "purple"), unsafe_allow_html=True)
        snap_c2.markdown(stat_card("✈️", f"{away_wins}", "Away Wins", "", "blue"), unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        snap_c3, snap_c4 = st.columns(2)
        snap_c3.markdown(stat_card("🤝", f"{draws}", "Draws", "", "orange"), unsafe_allow_html=True)
        snap_c4.markdown(stat_card("⚡", f"{home_wins + away_wins}", "Decisive Results", "", "pink"), unsafe_allow_html=True)

    with col_top5:
        section_heading("Top 5 Clubs", "⭐")
        for i, row in season_tbl.head(5).iterrows():
            form = team_form(season_df, row['Team'], 5)
            rank_cls = "top3" if i <= 3 else ""
            dots_html = form_dots_html(form)
            st.markdown(f"""
            <div class="club-card" style="margin-bottom:0.5rem;">
              <div class="club-rank {rank_cls}">{i}</div>
              <div style="flex:1">
                <div class="club-name">{row['Team']}</div>
                <div class="club-pts">pts: {int(row['Pts'])}  ·  {int(row['W'])}W {int(row['D'])}D {int(row['L'])}L  ·  GD {int(row['GD']):+d}</div>
                {dots_html}
              </div>
              <div style="text-align:right">
                <div style="font-family:'Barlow Condensed';font-size:1.6rem;font-weight:900;color:#fff">{int(row['GF'])} <span style="color:#4b5563;font-size:1rem">/ {int(row['GA'])}</span></div>
                <div style="font-size:0.68rem;color:#6b7280">GF / GA</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    section_heading("Goal Distribution by Matchweek", "📉")

    if 'MW' in season_df.columns:
        mw_goals = season_df.groupby('MW').agg(
            HG=('FTHG','sum'), AG=('FTAG','sum')
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mw_goals['MW'], y=mw_goals['HG'],
            name='Home Goals', marker_color='#7c3aed',
            hovertemplate='MW %{x}: %{y} home goals<extra></extra>',
        ))
        fig.add_trace(go.Bar(
            x=mw_goals['MW'], y=mw_goals['AG'],
            name='Away Goals', marker_color='#2563eb',
            hovertemplate='MW %{x}: %{y} away goals<extra></extra>',
        ))
        fig.update_layout(
            barmode='group', plot_bgcolor='#111827', paper_bgcolor='#111827',
            font=dict(family='Barlow', color='#9ca3af', size=11),
            legend=dict(bgcolor='#111827', bordercolor='#1e2540', borderwidth=1),
            margin=dict(l=0, r=0, t=10, b=0), height=260,
            xaxis=dict(gridcolor='#1e2540', title='Matchweek'),
            yaxis=dict(gridcolor='#1e2540', title='Goals'),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Matchweek data not available for this season.")

    section_heading("Result Split by Matchweek", "🔄")
    if 'MW' in season_df.columns:
        if 'FTR' in season_df.columns:
            mw_res = season_df.groupby('MW').agg(
                HW=('FTR', lambda x: (x=='H').sum()),
                D=('FTR',  lambda x: (x=='D').sum()),
                AW=('FTR', lambda x: (x=='A').sum()),
            ).reset_index()
        else:
            _df = season_df.copy()
            _df['_HW'] = (_df['FTHG'] > _df['FTAG']).astype(int)
            _df['_AW'] = (_df['FTAG'] > _df['FTHG']).astype(int)
            _df['_D']  = (_df['FTHG'] == _df['FTAG']).astype(int)
            mw_res = _df.groupby('MW').agg(HW=('_HW','sum'), D=('_D','sum'), AW=('_AW','sum')).reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=mw_res['MW'], y=mw_res['HW'].cumsum(),
            name='Cumulative Home Wins', line=dict(color='#a78bfa', width=2),
            fill='tozeroy', fillcolor='rgba(124,58,237,0.1)',
        ))
        fig2.add_trace(go.Scatter(
            x=mw_res['MW'], y=mw_res['AW'].cumsum(),
            name='Cumulative Away Wins', line=dict(color='#38bdf8', width=2),
            fill='tozeroy', fillcolor='rgba(37,99,235,0.1)',
        ))
        fig2.update_layout(
            plot_bgcolor='#111827', paper_bgcolor='#111827',
            font=dict(family='Barlow', color='#9ca3af', size=11),
            legend=dict(bgcolor='#111827', bordercolor='#1e2540', borderwidth=1),
            margin=dict(l=0, r=0, t=10, b=0), height=240,
            xaxis=dict(gridcolor='#1e2540', title='Matchweek'),
            yaxis=dict(gridcolor='#1e2540', title='Cumulative Wins'),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Matchweek data not available for this season.")


# ══════════════════════════════════════════════════════════════
#  TAB 2 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════

with tab_predictor:

    section_heading("Match Predictor", "🎯")

    col_sel, col_res = st.columns([1, 2])

    with col_sel:
        pred_home = st.selectbox("Home Team", all_teams, key="pred_home")
        pred_away = st.selectbox("Away Team", [t for t in all_teams if t != pred_home],
                                 key="pred_away")

        predict_btn = st.button("▶  Run Prediction", type="primary", use_container_width=True)

    if predict_btn or ('pred_result' in st.session_state):
        if predict_btn:
            with st.spinner("Running models…"):
                result = run_prediction(selected_season, pred_home, pred_away)
            st.session_state['pred_result'] = result
            st.session_state['pred_home_name'] = pred_home
            st.session_state['pred_away_name'] = pred_away
        else:
            result = st.session_state.get('pred_result')

        if result is None:
            st.warning("Not enough data to predict this fixture.")
        else:
            p_home, p_draw, p_away = result['probs']
            lam_h, lam_a = result['poisson']
            ens_hg, ens_ag = result['ensemble']

            with col_res:
                st.markdown(f"""
                <div class="match-card">
                  <div class="match-title">Ensemble Projected Score</div>
                  <div class="score-display">
                    <div class="score-team">
                      <div class="score-team-name">{pred_home.upper()}</div>
                      <div class="score-goals home">{ens_hg:.2f}</div>
                      <div style="font-size:0.72rem;color:#6b7280">xG</div>
                    </div>
                    <div class="score-divider">VS</div>
                    <div class="score-team">
                      <div class="score-team-name">{pred_away.upper()}</div>
                      <div class="score-goals away">{ens_ag:.2f}</div>
                      <div style="font-size:0.72rem;color:#6b7280">xG</div>
                    </div>
                  </div>
                  {prob_bar(p_home, p_draw, p_away, pred_home, pred_away)}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            section_heading("Model Comparison", "🤖")
            m1, m2, m3, m4 = st.columns(4)

            def model_card_col(col, name, hg, ag, rmse_h=None, rmse_a=None, color="#a78bfa"):
                rmse_txt = f"RMSE: {rmse_h:.2f} / {rmse_a:.2f}" if rmse_h is not None else "RMSE: —"
                col.markdown(f"""
                <div class="model-cell">
                  <div class="mname">{name}</div>
                  <div class="mscore" style="color:{color}">{hg:.2f} – {ag:.2f}</div>
                  <div class="mrmse">{rmse_txt}</div>
                </div>
                """, unsafe_allow_html=True)

            model_card_col(m1, "Poisson", lam_h, lam_a,
                           result['rmse']['poisson'][0], result['rmse']['poisson'][1], "#a78bfa")
            model_card_col(m2, "Random Forest", *result['rf'],
                           result['rmse']['rf'][0], result['rmse']['rf'][1], "#38bdf8")
            model_card_col(m3, "XGBoost", *result['xgb'],
                           result['rmse']['xgb'][0], result['rmse']['xgb'][1], "#4ade80")
            model_card_col(m4, "Ensemble", ens_hg, ens_ag,
                           result['rmse']['ensemble'][0], result['rmse']['ensemble'][1], "#f59e0b")

            if result['match_played']:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                section_heading("Actual Result", "✅")
                ra, rb = st.columns(2)
                ra.metric(f"{pred_home} Goals", result['true_hg'])
                rb.metric(f"{pred_away} Goals", result['true_ag'])

                models_names = ['Poisson','RandomForest','XGBoost','Ensemble']
                rmse_hg_vals = [
                    result['rmse']['poisson'][0], result['rmse']['rf'][0],
                    result['rmse']['xgb'][0], result['rmse']['ensemble'][0]
                ]
                rmse_ag_vals = [
                    result['rmse']['poisson'][1], result['rmse']['rf'][1],
                    result['rmse']['xgb'][1], result['rmse']['ensemble'][1]
                ]
                fig_rmse = go.Figure()
                fig_rmse.add_trace(go.Bar(x=models_names, y=rmse_hg_vals,
                    name='Home Goals RMSE', marker_color='#7c3aed'))
                fig_rmse.add_trace(go.Bar(x=models_names, y=rmse_ag_vals,
                    name='Away Goals RMSE', marker_color='#2563eb'))
                fig_rmse.update_layout(
                    barmode='group', plot_bgcolor='#111827', paper_bgcolor='#111827',
                    font=dict(family='Barlow', color='#9ca3af', size=11),
                    margin=dict(l=0, r=0, t=10, b=0), height=220,
                    yaxis=dict(gridcolor='#1e2540', title='RMSE'),
                    legend=dict(bgcolor='#111827', bordercolor='#1e2540', borderwidth=1),
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

            section_heading("Score Probability Matrix", "🔢")
            st.markdown('<div class="info-box">Probability of each exact scoreline via Poisson distribution (first 6 goals each team)</div>', unsafe_allow_html=True)

            max_g = 6
            score_matrix = np.array([
                [poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) for j in range(max_g)]
                for i in range(max_g)
            ])
            fig_heat = go.Figure(go.Heatmap(
                z=score_matrix * 100,
                x=[str(i) for i in range(max_g)],
                y=[str(i) for i in range(max_g)],
                colorscale=[[0,'#0d1220'],[0.3,'#4c1d95'],[1,'#a78bfa']],
                text=[[f"{v:.1f}%" for v in row] for row in score_matrix * 100],
                texttemplate="%{text}",
                hovertemplate="Home %{y} – Away %{x}: %{z:.2f}%<extra></extra>",
                showscale=False,
            ))
            fig_heat.update_layout(
                plot_bgcolor='#111827', paper_bgcolor='#111827',
                font=dict(family='Barlow', color='#9ca3af', size=10),
                margin=dict(l=40, r=0, t=10, b=40), height=300,
                xaxis=dict(title=f'{pred_away} Goals', side='bottom'),
                yaxis=dict(title=f'{pred_home} Goals'),
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3 — LEAGUE TABLE
# ══════════════════════════════════════════════════════════════

with tab_table:

    section_heading("League Table", "📊")

    tbl = season_tbl.copy()

    def zone_class(pos):
        if pos <= 4:  return "zone-cl"
        if pos <= 6:  return "zone-el"
        if pos >= 18: return "zone-rel"
        return ""

    rows_html = ""
    for pos, row in tbl.iterrows():
        zc = zone_class(pos)
        f  = team_form(season_df, row['Team'], 5)
        dots = "".join(
            f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            f'background:{"#4ade80" if r=="W" else "#facc15" if r=="D" else "#f87171"};'
            f'margin-right:2px"></span>' for r in f
        )
        rows_html += f"""
        <tr>
          <td class="{zc}">{pos}</td>
          <td>{row['Team']}</td>
          <td>{int(row['MP'])}</td>
          <td>{int(row['W'])}</td>
          <td>{int(row['D'])}</td>
          <td>{int(row['L'])}</td>
          <td>{int(row['GF'])}</td>
          <td>{int(row['GA'])}</td>
          <td>{int(row['GD']):+d}</td>
          <td class="pts-bold">{int(row['Pts'])}</td>
          <td>{int(row['HW'])}/{int(row['HD'])}/{int(row['HL'])}</td>
          <td>{int(row['AW'])}/{int(row['AD'])}/{int(row['AL'])}</td>
          <td>{dots}</td>
        </tr>"""

    table_html = f"""
    <div class="league-table-wrap">
    <table class="league-table">
      <thead>
        <tr>
          <th>#</th><th>Team</th><th>MP</th><th>W</th><th>D</th><th>L</th>
          <th>GF</th><th>GA</th><th>GD</th><th>Pts</th>
          <th>Home W/D/L</th><th>Away W/D/L</th><th>Form</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    <div style="display:flex;gap:1.5rem;margin-top:0.8rem;font-size:0.75rem;color:#6b7280">
      <span><span style="color:#2563eb">■</span> Champions League</span>
      <span><span style="color:#f59e0b">■</span> Europa League</span>
      <span><span style="color:#dc2626">■</span> Relegation Zone</span>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    section_heading("Points Race", "📈")

    if 'MW' in season_df.columns:
        top6_teams = season_tbl.head(6)['Team'].tolist()
        colors_6 = ['#a78bfa','#f59e0b','#38bdf8','#4ade80','#f87171','#fb923c']

        fig_pts = go.Figure()
        for i, t in enumerate(top6_teams):
            pts_by_mw = []
            cumpts = 0
            hm = season_df[season_df['HomeTeam'] == t]
            aw = season_df[season_df['AwayTeam'] == t]
            mws = sorted(season_df['MW'].unique())

            has_ftr_col = 'FTR' in season_df.columns
            for mw in mws:
                hm_mw = hm[hm['MW'] == mw]
                aw_mw = aw[aw['MW'] == mw]
                for _, row in hm_mw.iterrows():
                    if has_ftr_col and pd.notna(row.get('FTR')):
                        cumpts += 3 if row['FTR']=='H' else (1 if row['FTR']=='D' else 0)
                    else:
                        cumpts += 3 if row['FTHG']>row['FTAG'] else (1 if row['FTHG']==row['FTAG'] else 0)
                for _, row in aw_mw.iterrows():
                    if has_ftr_col and pd.notna(row.get('FTR')):
                        cumpts += 3 if row['FTR']=='A' else (1 if row['FTR']=='D' else 0)
                    else:
                        cumpts += 3 if row['FTAG']>row['FTHG'] else (1 if row['FTAG']==row['FTHG'] else 0)
                pts_by_mw.append(cumpts)

            fig_pts.add_trace(go.Scatter(
                x=mws, y=pts_by_mw, name=t,
                line=dict(color=colors_6[i], width=2),
                hovertemplate=f'{t}: %{{y}} pts (MW %{{x}})<extra></extra>',
            ))

        fig_pts.update_layout(
            plot_bgcolor='#111827', paper_bgcolor='#111827',
            font=dict(family='Barlow', color='#9ca3af', size=11),
            legend=dict(bgcolor='#111827', bordercolor='#1e2540', borderwidth=1),
            margin=dict(l=0, r=0, t=10, b=0), height=340,
            xaxis=dict(gridcolor='#1e2540', title='Matchweek'),
            yaxis=dict(gridcolor='#1e2540', title='Cumulative Points'),
        )
        st.plotly_chart(fig_pts, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 4 — TEAM DEEP DIVE
# ══════════════════════════════════════════════════════════════

with tab_team:

    section_heading("Team Deep Dive", "🔍")

    selected_team = st.selectbox("Select Team", all_teams, key="team_dive")

    t_row = season_tbl[season_tbl['Team'] == selected_team]
    if t_row.empty:
        st.warning("No data for this team in this season.")
    else:
        t_row = t_row.iloc[0]
        rank  = season_tbl[season_tbl['Team']==selected_team].index[0]
        form  = team_form(season_df, selected_team, 5)

        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2540;border-radius:10px;
                    padding:1.5rem 2rem;margin-bottom:1.2rem;">
          <div style="font-family:'Barlow Condensed';font-size:2.2rem;font-weight:900;color:#fff">
            {selected_team.upper()}
          </div>
          <div style="color:#6b7280;font-size:0.85rem;margin-top:0.3rem">
            Season {selected_season}  ·  League Position <span style="color:#a78bfa;font-weight:700">{rank}</span>
          </div>
          <div style="margin-top:0.8rem">{form_dots_html(form)}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(stat_card("🏆","Pts", f"{int(t_row['Pts'])}", "", "purple"), unsafe_allow_html=True)
        c2.markdown(stat_card("✅", f"{int(t_row['W'])}", "Wins", "", "green"), unsafe_allow_html=True)
        c3.markdown(stat_card("⚽", f"{int(t_row['GF'])}", "Goals For", "", "pink"), unsafe_allow_html=True)
        c4.markdown(stat_card("🛡️", f"{int(t_row['GA'])}", "Goals Against", "", "blue"), unsafe_allow_html=True)
        c5.markdown(stat_card("📊", f"{int(t_row['GD']):+d}", "Goal Diff", "", "orange"), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        col_hm, col_aw = st.columns(2)
        with col_hm:
            section_heading("Home Record", "🏠")
            fig_donut_h = go.Figure(go.Pie(
                values=[int(t_row['HW']), int(t_row['HD']), int(t_row['HL'])],
                labels=['Win','Draw','Loss'],
                hole=0.6,
                marker_colors=['#7c3aed','#374151','#dc2626'],
                textinfo='percent+label',
                textfont=dict(size=11, family='Barlow', color='#fff'),
            ))
            fig_donut_h.update_layout(
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font=dict(color='#9ca3af'), margin=dict(l=0,r=0,t=10,b=0), height=200,
                legend=dict(bgcolor='#111827'),
                annotations=[dict(text=f'{int(t_row["HW"])}W', x=0.5, y=0.5,
                                  font_size=20, showarrow=False, font_color='#fff',
                                  font_family='Barlow Condensed')],
            )
            st.plotly_chart(fig_donut_h, use_container_width=True)

        with col_aw:
            section_heading("Away Record", "✈️")
            fig_donut_a = go.Figure(go.Pie(
                values=[int(t_row['AW']), int(t_row['AD']), int(t_row['AL'])],
                labels=['Win','Draw','Loss'],
                hole=0.6,
                marker_colors=['#2563eb','#374151','#dc2626'],
                textinfo='percent+label',
                textfont=dict(size=11, family='Barlow', color='#fff'),
            ))
            fig_donut_a.update_layout(
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font=dict(color='#9ca3af'), margin=dict(l=0,r=0,t=10,b=0), height=200,
                legend=dict(bgcolor='#111827'),
                annotations=[dict(text=f'{int(t_row["AW"])}W', x=0.5, y=0.5,
                                  font_size=20, showarrow=False, font_color='#fff',
                                  font_family='Barlow Condensed')],
            )
            st.plotly_chart(fig_donut_a, use_container_width=True)

        if 'MW' in season_df.columns:
            section_heading("Goals Per Matchweek", "📉")
            hm = season_df[season_df['HomeTeam'] == selected_team][['MW','FTHG','FTAG']].copy()
            hm.columns = ['MW','GF','GA']
            aw = season_df[season_df['AwayTeam'] == selected_team][['MW','FTAG','FTHG']].copy()
            aw.columns = ['MW','GF','GA']
            team_mw = pd.concat([hm, aw]).sort_values('MW')

            fig_gpm = go.Figure()
            fig_gpm.add_trace(go.Bar(x=team_mw['MW'], y=team_mw['GF'],
                name='Goals Scored', marker_color='#7c3aed'))
            fig_gpm.add_trace(go.Bar(x=team_mw['MW'], y=-team_mw['GA'],
                name='Goals Conceded', marker_color='#dc2626'))
            fig_gpm.update_layout(
                barmode='relative', plot_bgcolor='#111827', paper_bgcolor='#111827',
                font=dict(family='Barlow', color='#9ca3af', size=11),
                margin=dict(l=0,r=0,t=10,b=0), height=260,
                xaxis=dict(gridcolor='#1e2540', title='Matchweek'),
                yaxis=dict(gridcolor='#1e2540', title='Goals'),
                legend=dict(bgcolor='#111827', bordercolor='#1e2540', borderwidth=1),
            )
            st.plotly_chart(fig_gpm, use_container_width=True)

        section_heading("All Results", "📋")
        _base_cols = ['MW','HomeTeam','AwayTeam','FTHG','FTAG']
        _has_mw = 'MW' in season_df.columns
        _cols = _base_cols if _has_mw else ['HomeTeam','AwayTeam','FTHG','FTAG']

        hm = season_df[season_df['HomeTeam'] == selected_team][_cols].copy()
        hm['Venue'] = 'Home'
        if not _has_mw:
            hm['MW'] = range(1, len(hm)+1)
        if 'FTR' in season_df.columns:
            hm['Result'] = season_df.loc[hm.index, 'FTR'].map({'H':'W','D':'D','A':'L'}).values
        else:
            hm['Result'] = np.where(hm['FTHG']>hm['FTAG'],'W', np.where(hm['FTHG']==hm['FTAG'],'D','L'))

        aw = season_df[season_df['AwayTeam'] == selected_team][_cols].copy()
        aw['Venue'] = 'Away'
        if not _has_mw:
            aw['MW'] = range(1, len(aw)+1)
        if 'FTR' in season_df.columns:
            aw['Result'] = season_df.loc[aw.index, 'FTR'].map({'A':'W','D':'D','H':'L'}).values
        else:
            aw['Result'] = np.where(aw['FTAG']>aw['FTHG'],'W', np.where(aw['FTAG']==aw['FTHG'],'D','L'))

        all_res = pd.concat([hm, aw]).sort_values('MW')

        res_rows = ""
        for _, r in all_res.iterrows():
            col_map = {'W': '#4ade80', 'D': '#facc15', 'L': '#f87171'}
            c = col_map.get(r['Result'], '#9ca3af')
            opponent = r['AwayTeam'] if r['Venue']=='Home' else r['HomeTeam']
            score = f"{int(r['FTHG'])}–{int(r['FTAG'])}"
            res_rows += f"""<tr>
              <td style="color:#6b7280">{int(r['MW'])}</td>
              <td style="color:#9ca3af;font-size:0.75rem">{r['Venue']}</td>
              <td style="color:#fff">{opponent}</td>
              <td style="font-family:'Barlow Condensed';font-size:1rem;color:#fff;text-align:center">{score}</td>
              <td style="color:{c};font-weight:700;text-align:center">{r['Result']}</td>
            </tr>"""

        st.markdown(f"""
        <div class="league-table-wrap">
        <table class="league-table">
          <thead><tr>
            <th>MW</th><th>Venue</th><th>Opponent</th><th>Score</th><th>Result</th>
          </tr></thead>
          <tbody>{res_rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 5 — HEAD TO HEAD
# ══════════════════════════════════════════════════════════════

with tab_h2h:

    section_heading("Head to Head", "⚔️")

    hh_c1, hh_c2 = st.columns(2)
    team_a = hh_c1.selectbox("Team A", all_teams, key="h2h_a")
    team_b = hh_c2.selectbox("Team B", [t for t in all_teams if t != team_a],
                              key="h2h_b", index=1)

    h2h = h2h_stats(selected_season, team_a, team_b)

    if not h2h:
        st.markdown('<div class="info-box">No head-to-head records found between these two clubs across any loaded season.</div>', unsafe_allow_html=True)
    else:
        total = h2h['total']
        wa, wb, dr = h2h['wins_a'], h2h['wins_b'], h2h['draws']
        ga, gb = h2h['goals_a'], h2h['goals_b']

        pct_a = wa/total if total > 0 else 0
        pct_d = dr/total if total > 0 else 0
        pct_b = wb/total if total > 0 else 0

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.markdown(stat_card("📅", total, "Meetings", "across all seasons", "purple"), unsafe_allow_html=True)
        s2.markdown(stat_card("🏆", wa, f"{team_a} Wins", f"{pct_a*100:.0f}%", "pink"), unsafe_allow_html=True)
        s3.markdown(stat_card("🤝", dr, "Draws", f"{pct_d*100:.0f}%", "orange"), unsafe_allow_html=True)
        s4.markdown(stat_card("🏆", wb, f"{team_b} Wins", f"{pct_b*100:.0f}%", "blue"), unsafe_allow_html=True)
        s5.markdown(stat_card("⚽", f"{ga}–{gb}", "Goals", f"Avg {(ga+gb)/max(total,1):.1f}/match", "green"), unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        bar_a = int(pct_a * 100)
        bar_d = int(pct_d * 100)
        bar_b = 100 - bar_a - bar_d
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2540;border-radius:8px;padding:1.2rem 1.5rem">
          <div style="font-family:'Barlow Condensed';font-size:0.7rem;letter-spacing:0.1em;
                      color:#6b7280;text-transform:uppercase;margin-bottom:0.8rem">
            Win Share
          </div>
          <div class="h2h-label">
            <span style="color:#a78bfa;font-weight:600">{team_a} ({bar_a}%)</span>
            <span style="color:#6b7280">Draw ({bar_d}%)</span>
            <span style="color:#38bdf8;font-weight:600">{team_b} ({bar_b}%)</span>
          </div>
          <div class="h2h-bar-track" style="height:10px">
            <div class="h2h-seg-a" style="width:{bar_a}%"></div>
            <div class="h2h-seg-d" style="width:{bar_d}%"></div>
            <div class="h2h-seg-b" style="width:{bar_b}%"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:0.6rem;
                      font-family:'Barlow Condensed';font-size:1.4rem;font-weight:900">
            <span style="color:#a78bfa">{ga} goals</span>
            <span style="color:#38bdf8">{gb} goals</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        section_heading("Season by Season", "📅")

        seas_rows = ""
        for r in sorted(h2h['records'], key=lambda x: x['season'], reverse=True):
            home, away = r['home'], r['away']
            hg, ag = int(r['hg']), int(r['ag'])
            is_a_home = home == team_a
            winner = team_a if r['ftr']==('H' if is_a_home else 'A') else (
                     team_b if r['ftr']==('A' if is_a_home else 'H') else 'Draw')
            win_color = '#4ade80' if winner==team_a else ('#f87171' if winner==team_b else '#facc15')
            seas_rows += f"""<tr>
              <td style="color:#6b7280">{r['season']}</td>
              <td style="color:#fff">{home}</td>
              <td style="font-family:'Barlow Condensed';font-size:1.1rem;color:#fff;text-align:center">{hg}–{ag}</td>
              <td style="color:#fff">{away}</td>
              <td style="color:{win_color};font-weight:600;text-align:center">{winner}</td>
            </tr>"""

        st.markdown(f"""
        <div class="league-table-wrap">
        <table class="league-table">
          <thead><tr>
            <th>Season</th><th>Home</th><th>Score</th><th>Away</th><th>Winner</th>
          </tr></thead>
          <tbody>{seas_rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        if h2h['records']:
            section_heading("Goals Per Meeting", "📉")
            recs = sorted(h2h['records'], key=lambda x: (x['season']))
            goals_a_per = [r['hg'] if r['home']==team_a else r['ag'] for r in recs]
            goals_b_per = [r['hg'] if r['home']==team_b else r['ag'] for r in recs]

            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Bar(x=list(range(len(recs))), y=goals_a_per,
                name=team_a, marker_color='#a78bfa'))
            fig_h2h.add_trace(go.Bar(x=list(range(len(recs))), y=goals_b_per,
                name=team_b, marker_color='#38bdf8'))
            fig_h2h.update_layout(
                barmode='group', plot_bgcolor='#111827', paper_bgcolor='#111827',
                font=dict(family='Barlow', color='#9ca3af', size=11),
                margin=dict(l=0,r=0,t=10,b=0), height=240,
                xaxis=dict(tickvals=list(range(len(recs))),
                           ticktext=[r['season'] for r in recs],
                           gridcolor='#1e2540'),
                yaxis=dict(gridcolor='#1e2540', title='Goals'),
                legend=dict(bgcolor='#111827', bordercolor='#1e2540', borderwidth=1),
            )
            st.plotly_chart(fig_h2h, use_container_width=True)