# ----------------------------- NFL PICKS TRACKER (CLEAN) -----------------------------
# One-source-of-truth: master schedule + weekly odds pulled from GitHub (with local fallback)
# Guardrails for columns, dups, week mismatches, and helpful errors in the UI.

import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
import streamlit as st

# ====== CONFIG (edit these two lines as needed) ======================================
GITHUB_USER_REPO = "freejust2bme/nfl-picks-repo"  # <— your repo
DEFAULT_WEEK = 5                                   # <— change this week-to-week (or use selector below)
# =====================================================================================

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER_REPO}/main/data"
SCHEDULE_URL = f"{RAW_BASE}/nfl_2025_master_schedule.csv"

def odds_url_for_week(week: int) -> str:
    return f"{RAW_BASE}/week_{week}_odds_template.csv"

# Local fallbacks (optional; useful if deploying with data/)
LOCAL_SCHEDULE = Path("data/nfl_2025_master_schedule.csv")
def local_odds_path(week: int) -> Path:
    return Path(f"data/week_{week}_odds_template.csv")

# ----------------------------- UTILITIES -----------------------------
REQ_SCHED_COLS = ["week", "home_team", "away_team"]
REQ_ODDS_COLS  = ["week","away_team","home_team","home_ml","away_ml","home_spread","away_spread","total","book_time"]

def load_csv_smart(primary_url: str, local_path: Path) -> pd.DataFrame:
    """Try GitHub raw first; fall back to local path; raise a clear error."""
    try:
        return pd.read_csv(primary_url)
    except Exception as e1:
        try:
            return pd.read_csv(local_path)
        except Exception as e2:
            raise RuntimeError(
                f"Unable to load CSV from URL or local.\n"
                f"URL: {primary_url}\nLocal: {local_path}\n"
                f"URL error: {repr(e1)}\nLocal error: {repr(e2)}"
            )

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def validate_has_columns(df: pd.DataFrame, required: list, label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} file is missing columns: {missing}\nFound: {list(df.columns)}")

def validate_no_duplicates(df: pd.DataFrame, keys: list, label: str):
    if df.duplicated(keys).any():
        dups = df[df.duplicated(keys, keep=False)].sort_values(keys)
        raise ValueError(f"{label} has duplicate rows on {keys}:\n{dups}")

def safe_merge(schedule: pd.DataFrame, odds: pd.DataFrame, week: int) -> pd.DataFrame:
    # only rows for selected week
    sched_w = schedule.query("week == @week").copy()
    odds_w  = odds.query("week == @week").copy()

    # Validate basic shapes
    if sched_w.empty:
        raise ValueError(f"No schedule rows found for week {week}.")
    if odds_w.empty:
        # Provide friendly hint
        raise ValueError(
            f"No odds rows found for week {week}. "
            f"Make sure your week_{week}_odds_template.csv has 'week' filled with {week}."
        )

    # Ensure 1 row per matchup in both
    validate_no_duplicates(sched_w, ["home_team","away_team"], "Schedule")
    validate_no_duplicates(odds_w,  ["home_team","away_team"], "Odds")

    merged = pd.merge(
        sched_w,
        odds_w[REQ_ODDS_COLS],  # keep ordered/clean
        on=["week","home_team","away_team"],
        how="left",
        validate="one_to_one",
    )
    return merged

def compute_confidence(row) -> int:
    """
    Super-light placeholder logic:
    - If spreads missing → 0 stars
    - Heavier favorite (|home_spread| >= 6) → 3 stars
    - Moderate (>= 3) → 2 stars
    - Slight (< 3) → 1 star
    """
    try:
        hs = float(row.get("home_spread")) if pd.notna(row.get("home_spread")) and row.get("home_spread") != "" else None
    except Exception:
        hs = None
    if hs is None:
        return 0
    a = abs(hs)
    if a >= 6: return 3
    if a >= 3: return 2
    return 1

def favorite_side(row) -> str:
    """
    If home_spread is negative → home favored; if positive → away favored.
    """
    try:
        hs = float(row.get("home_spread"))
    except Exception:
        return ""
    if pd.isna(hs): return ""
    if hs < 0: return "HOME"
    if hs > 0: return "AWAY"
    return "EVEN"

# ----------------------------- UI -----------------------------
st.set_page_config(page_title="NFL Picks Tracker", layout="wide")
st.title("🏈 NFL Picks Tracker — Clean Mode")

with st.sidebar:
    st.markdown("### Data Source")
    st.caption("Master schedule + weekly odds from GitHub (raw) with local fallback.")

# --- WEEK SELECTOR (dropdown version) ---
available_weeks = list(range(1, 19))
week = st.selectbox(
    "📅 Select Week",
    available_weeks,
    index=available_weeks.index(DEFAULT_WEEK) if DEFAULT_WEEK in available_weeks else 0,
    help="Select the week to view matchups and odds."
)

# Optional: manual refresh button
if st.button("🔄 Refresh This Week"):
    st.experimental_rerun()

# ----------------------------- LOAD DATA -----------------------------
# Schedule
try:
    schedule = load_csv_smart(SCHEDULE_URL, LOCAL_SCHEDULE)
    schedule = normalize_columns(schedule)
    validate_has_columns(schedule, REQ_SCHED_COLS, "Schedule")
except Exception as e:
    st.error(f"Failed to load schedule.\n\n{e}")
    st.stop()

# Odds for selected week
ODDS_URL = odds_url_for_week(week)
try:
    odds = load_csv_smart(ODDS_URL, local_odds_path(week))
    odds = normalize_columns(odds)
    validate_has_columns(odds, REQ_ODDS_COLS, "Odds")
except Exception as e:
    st.error(
        f"Failed to load odds for week {week}.\n\n{e}\n\n"
        f"Tip: Make sure this file exists: `data/week_{week}_odds_template.csv` in your repo."
    )
    st.stop()

# ----------------------------- MERGE + GUARDRAILS -----------------------------
try:
    merged = safe_merge(schedule, odds, week)
except Exception as e:
    st.error(f"Could not combine schedule + odds for week {week}.\n\n{e}")
    st.stop()

# Compute helpers
merged["favorite"] = merged.apply(favorite_side, axis=1)
merged["confidence_stars"] = merged.apply(compute_confidence, axis=1)

# Nice ordering if columns exist
preferred_cols = (
    ["week","away_team","home_team"]
    + [c for c in ["home_ml","away_ml","home_spread","away_spread","total"] if c in merged.columns]
    + ["favorite","confidence_stars"]
)
merged = merged[[c for c in preferred_cols if c in merged.columns] + [c for c in merged.columns if c not in preferred_cols]]

# ----------------------------- DISPLAY -----------------------------
st.subheader(f"Week {week} — Matchups + Odds")
st.dataframe(merged, use_container_width=True)

# Quick summary block
left, right = st.columns(2)
with left:
    st.metric("Games this week", len(merged))
with right:
    ready = merged["confidence_stars"].astype(int).ge(1).sum()
    st.metric("Games with odds (≥1★)", int(ready))

st.markdown("---")

# ----------------------------- EXPORTS (OPTIONAL) -----------------------------
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.download_button(
        "📥 Download Week Sheet (CSV)",
        data=merged.to_csv(index=False),
        file_name=f"week_{week}_merged.csv",
        mime="text/csv",
        use_container_width=True
    )

with exp_col2:
    template = merged.copy()
    # provide a clean odds-entry template for the same week
    for c in ["home_ml","away_ml","home_spread","away_spread","total"]:
        if c in template.columns:
            template[c] = ""
    st.download_button(
        "📄 Download Blank Odds Template for This Week",
        data=template[REQ_ODDS_COLS].to_csv(index=False),
        file_name=f"week_{week}_odds_template.csv",
        mime="text/csv",
        use_container_width=True
    )

# ----------------------------- NOTES -----------------------------
st.caption(
    "Notes: This simplified app ignores old `data/versioned_weeks/` files. "
    "Keep `nfl_2025_master_schedule.csv` and `week_X_odds_template.csv` in your repo's `/data` folder. "
    "Update the number at the top (DEFAULT_WEEK) or use the selector here."
)
# --------------------------------------------------------------------------------
