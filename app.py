
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="NFL Picks Tracker", layout="wide")

st.title("🏈 NFL Picks Tracker — Original vs Guardrails")

DATA_DIR = Path("data")

# Sidebar: week selector & toggles
week = st.sidebar.selectbox("Select Week", ["Week 1", "Week 2", "Week 3"])
show_original = st.sidebar.checkbox("Show Original Picks", True)
show_adjusted = st.sidebar.checkbox("Show Adjusted (Guardrails)", True)
show_changed_only = st.sidebar.checkbox("What Changed? (Only where picks differ)", False)

st.sidebar.markdown("---")
st.sidebar.header("Data Files")
uploaded = st.sidebar.file_uploader("Upload a schedule CSV (optional)", type=["csv"])
if uploaded is not None:
    tmp_df = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {uploaded.name} with {len(tmp_df)} rows.")
else:
    tmp_df = None

# Load schedule
def load_week_csv(week_name):
    fname = {
        "Week 1": "week1_schedule_with_times.csv",
        "Week 2": "week2_schedule_with_times.csv",
        "Week 3": "week3_schedule_with_times.csv",
    }.get(week_name, "week3_schedule_with_times.csv")
    path = DATA_DIR / fname
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

schedule_df = tmp_df if tmp_df is not None else load_week_csv(week)

st.subheader(f"📅 {week} Schedule")
if schedule_df.empty:
    st.info("No schedule file found for this week yet. Upload a CSV in the sidebar or add a file under `data/`.")
else:
    st.dataframe(schedule_df, use_container_width=True)

# Demo in-memory pick boards (you can wire these to your model/logic storage):
# In a real app, these would be loaded from your picks engine.
original_cols = ["Matchup","Original ML","Original ATS","Final Score","ML Result","ATS Result"]
adjusted_cols = ["Matchup","Adjusted ML","Adjusted ATS","History","Metrics","Injuries","Situational","Sharps","Guardrail Note","Final Score","ML Result","ATS Result"]

# Minimal demo rows so the app renders; replace with real data from your tracker
demo_rows = [
    ["Dolphins @ Bills", "Bills", "Bills -11.5", "31–21 BUF", "✅", "❌"],
    ["Panthers @ Falcons", "Falcons", "Falcons -5.5", "30–10 CAR", "❌", "❌"],
    ["Packers @ Browns", "Packers", "Packers -7.5", "13–10 CLE", "❌", "❌"],
]
original_df = pd.DataFrame(demo_rows, columns=original_cols)

demo_adj = [
    ["Dolphins @ Bills", "Bills (ML Only)", "—", "⚠️ Neutral", "Bills O top-5 EPA", "Low", "✅ Bounce-back", "❌ Sharp fade on ATS", "TNF big fav: ML-only", "31–21 BUF", "✅", "—"],
    ["Panthers @ Falcons", "— (No Play)", "—", "⚠️ Neutral", "ATL run O strong", "Low", "Trap", "❌ Sharps on CAR", "Road fav trap: skip", "30–10 CAR", "—", "—"],
    ["Packers @ Browns", "Browns", "Browns +7.5", "⚠️ Neutral", "CLE D > GB O (form)", "Low", "Home dog live", "✅ Sharps on CLE", "Sharp fade: flip side", "13–10 CLE", "✅", "✅"],
]
adjusted_df = pd.DataFrame(demo_adj, columns=adjusted_cols)

# Filter to "What Changed?"
if show_changed_only:
    # crude heuristic: where Original ML != Adjusted ML (not counting ML-only/No Play placeholders)
    comp = original_df.merge(adjusted_df, on="Matchup", how="inner", suffixes=("_orig","_adj"))
    mask = (comp["Original ML"] != comp["Adjusted ML"]) & (comp["Adjusted ML"].notna())
    changed_df = comp[mask]
    st.subheader("🔎 What Changed? (Guardrail Flips/Saves)")
    if changed_df.empty:
        st.info("No differences detected in this demo set.")
    else:
        st.dataframe(changed_df[["Matchup","Original ML","Adjusted ML","Original ATS","Adjusted ATS","Guardrail Note"]], use_container_width=True)

# Render tables
cols = st.columns(2)
with cols[0]:
    if show_original:
        st.subheader("📋 Original Picks")
        st.dataframe(original_df, use_container_width=True)
with cols[1]:
    if show_adjusted:
        st.subheader("🛡️ Adjusted Picks (Guardrails)")
        st.dataframe(adjusted_df, use_container_width=True)

st.markdown("---")
st.subheader("📈 Season Master Tracker (Running Totals)")
season_rows = [
    ["Week 1", "13–3", "16–0 ✅", "7–9", "6–0 ✅"],
    ["Week 2", "11–5", "16–0 ✅", "6–10", "5–0 ✅"],
    ["Week 3", "6–5 (pending)", "9–0 (pending) ✅", "5–6", "3–0 (pending) ✅"],
    ["Season-to-Date", "30–13 (~70%)", "41–0 (100%) ✅", "18–25 (~42%)", "14–0 (100%) ✅"],
]
season_df = pd.DataFrame(season_rows, columns=["Week","Original ML","Adjusted ML","Original ATS","Adjusted ATS (Selective)"])
st.dataframe(season_df, use_container_width=True)

st.caption("Tip: Replace the demo dataframes with your live tracker data sources. The UI will stay the same.")
