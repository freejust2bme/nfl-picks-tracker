
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="NFL Picks Tracker (Master Schedule)", layout="wide")
st.title("🏈 NFL Picks Tracker — Master Schedule Mode")

DATA_DIR = Path("data")
MASTER_FILE = DATA_DIR / "nfl_2025_master_schedule.csv"

st.sidebar.header("Controls")

upl = st.sidebar.file_uploader("Upload full-season schedule CSV (optional)", type=["csv"])
if upl is not None:
    master_df = pd.read_csv(upl)
    st.sidebar.success(f"Loaded uploaded schedule: {upl.name} ({len(master_df)} rows)")
else:
    if MASTER_FILE.exists():
        master_df = pd.read_csv(MASTER_FILE)
        st.sidebar.info(f"Using {MASTER_FILE.name} ({len(master_df)} rows)")
    else:
        st.error("Full-season schedule file not found.\n\nExpected: data/nfl_2025_master_schedule.csv\n\nUpload a CSV or add it to the repo.")
        st.stop()

master_df.columns = [c.strip().lower().replace(" ", "_") for c in master_df.columns]

def pick_col(df, options, contains=None):
    for c in df.columns:
        if c in options:
            return c
    if contains:
        for c in df.columns:
            if contains in c:
                return c
    return None

week_col = pick_col(master_df, ["week","wk"], contains="week")
away_col = pick_col(master_df, ["away_team","away","visitor"], contains="away")
home_col = pick_col(master_df, ["home_team","home"], contains="home")
date_col = pick_col(master_df, ["date","game_date"], contains="date")
time_col = pick_col(master_df, ["time","game_time"], contains="time")
iso_col  = pick_col(master_df, ["kickoff_iso","kickoff_datetime","datetime","kickoff"], contains="datetime")

if week_col is None or away_col is None or home_col is None:
    st.error("Couldn't find required columns. Need at minimum: Week, Away Team, Home Team.")
    st.stop()

weeks = sorted(pd.unique(master_df[week_col]))
week_labels = [f"Week {int(w) if str(w).isdigit() else w}" for w in weeks]
wk_idx = st.sidebar.selectbox("Select Week", list(range(len(weeks))), format_func=lambda i: week_labels[i])
selected_week = weeks[wk_idx]

wdf = master_df[master_df[week_col] == selected_week].copy()

disp = pd.DataFrame({
    "Week": wdf[week_col].astype(str),
    "Away Team": wdf[away_col].astype(str),
    "Home Team": wdf[home_col].astype(str),
})
if date_col:
    disp["Date"] = wdf[date_col].astype(str)
if time_col:
    disp["Time"] = wdf[time_col].astype(str)
if iso_col:
    dt = pd.to_datetime(wdf[iso_col], errors="coerce")
    disp["Kickoff ISO"] = dt.dt.strftime("%Y-%m-%d %H:%M").fillna(wdf[iso_col].astype(str))

st.subheader(f"📅 {week_labels[wk_idx]} Schedule")
st.dataframe(disp, use_container_width=True)

st.markdown("---")
st.subheader("📋 Original Picks (demo)")
st.dataframe(pd.DataFrame([
    ["Dolphins @ Bills", "Bills", "Bills -11.5", "31–21 BUF", "✅", "❌"]
], columns=["Matchup","Original ML","Original ATS","Final Score","ML Result","ATS Result"]), use_container_width=True)

st.subheader("🛡️ Adjusted Picks (Guardrails) (demo)")
st.dataframe(pd.DataFrame([
    ["Dolphins @ Bills", "Bills (ML Only)", "—", "⚠️ Neutral", "Bills O top-5 EPA", "Low", "Bounce-back", "Sharp fade on ATS", "TNF big fav: ML-only", "31–21 BUF", "✅", "—"]
], columns=["Matchup","Adjusted ML","Adjusted ATS","History","Metrics","Injuries","Situational","Sharps","Guardrail Note","Final Score","ML Result","ATS Result"]), use_container_width=True)

st.markdown("---")
st.subheader("📈 Season Master Tracker (demo)")
st.dataframe(pd.DataFrame([
    ["Week 1", "13–3", "16–0 ✅", "7–9", "6–0 ✅"],
    ["Week 2", "11–5", "16–0 ✅", "6–10", "5–0 ✅"],
    ["Week 3", "6–5 (pending)", "9–0 (pending) ✅", "5–6", "3–0 (pending) ✅"],
    ["Season-to-Date", "30–13 (~70%)", "41–0 (100%) ✅", "18–25 (~42%)", "14–0 (100%) ✅"],
], columns=["Week","Original ML","Adjusted ML","Original ATS","Adjusted ATS (Selective)"]), use_container_width=True)

st.caption("Upload one full-season CSV (or place it at data/nfl_2025_master_schedule.csv) and pick the week in the sidebar.")
