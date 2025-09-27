
# app.py — NFL Picks Tracker (Six-File Weekly Bundle)
import io
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Picks Tracker", layout="wide")
st.title("🏈 NFL Picks Tracker — Weekly Bundle")

DATA_DIR = Path("data")
VERS_DIR = DATA_DIR / "versioned_weeks"
ROOT_MASTER = DATA_DIR / "season_master_tracker.csv"

# ---------- Helpers ----------
def exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def load_csv(p: Path, cols=None):
    if not exists(p): return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if cols:
            keep = [c for c in cols if c in df.columns]
            if keep: df = df[keep]
        return df
    except Exception as e:
        st.error(f"Error reading {p}: {e}")
        return pd.DataFrame()

def norm_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize Adjusted file headers
    rename_map = {
        "Adjusted Pick": "Adjusted ML",
        "Match-up": "Matchup",
        "Confidence%": "Confidence %",
        "confidence": "Confidence %",
        "confidence %": "Confidence %",
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    cols_order = [c for c in ["Matchup","Adjusted ML","Confidence %","Guardrail Note"] if c in df.columns]
    if not cols_order:
        return df
    out = df[cols_order].copy()
    if "Confidence %" in out.columns:
        out["Confidence %"] = pd.to_numeric(out["Confidence %"], errors="coerce").round(0).astype("Int64")
    return out

def group_rows(d: pd.DataFrame, sizes=(6,6,4)):
    out, start = [], 0
    for size in sizes:
        if start >= len(d): break
        out.append(d.iloc[start:start+size].reset_index(drop=True))
        start += size
    if start < len(d):
        out.append(d.iloc[start:].reset_index(drop=True))
    return out

def band(v):
    try: v = float(v)
    except: return "—"
    if v >= 75: return "🟩 strong"
    if v >= 60: return "🟨 solid"
    return "🟥 lean"

# ---------- Sidebar ----------
st.sidebar.header("Controls")
week = st.sidebar.number_input("Week", min_value=1, max_value=18, value=4, step=1)
group_pattern = st.sidebar.text_input("Group pattern (comma separated)", value="6,6,4")
try:
    sizes = tuple(int(x.strip()) for x in group_pattern.split(",") if x.strip())
except:
    sizes = (6,6,4)

# ---------- Expected Weekly Files ----------
draft_p     = VERS_DIR / f"Week{week}_Draft.csv"
adjusted_p  = VERS_DIR / f"Week{week}_Adjusted.csv"
final_p     = VERS_DIR / f"Week{week}_Final.csv"
odds_p      = VERS_DIR / f"Week{week}_Odds.csv"
inj_p       = VERS_DIR / f"Week{week}_Injuries.csv"
guard_p     = VERS_DIR / f"Week{week}_GuardrailReport.csv"

st.subheader(f"📦 Week {week} Bundle")
colA, colB, colC = st.columns(3)
with colA:
    st.write(("✅" if exists(draft_p) else "❌") + f" Draft  — `{draft_p}`")
    st.write(("✅" if exists(adjusted_p) else "❌") + f" Adjusted — `{adjusted_p}`")
with colB:
    st.write(("✅" if exists(final_p) else "❌") + f" Final  — `{final_p}`")
    st.write(("✅" if exists(odds_p) else "❌") + f" Odds   — `{odds_p}`")
with colC:
    st.write(("✅" if exists(inj_p) else "❌") + f" Injuries — `{inj_p}`")
    st.write(("✅" if exists(guard_p) else "❌") + f" Guardrail Report — `{guard_p}`")

st.divider()

# ---------- Tabs for the six artifacts ----------
tab_adj, tab_draft, tab_final, tab_odds, tab_inj, tab_guard = st.tabs(
    ["Adjusted (Report View)", "Draft", "Final", "Odds", "Injuries", "Guardrails"]
)

# Adjusted (primary)
with tab_adj:
    # Fallbacks if versioned file not found (kept for convenience)
    adjusted_df = load_csv(adjusted_p)
    if adjusted_df.empty:
        fallback_candidates = [
            DATA_DIR / f"week{week}_adjusted_picks_CORRECTED.csv",
            DATA_DIR / f"week{week}_adjusted_picks.csv",
        ]
        for p in fallback_candidates:
            adjusted_df = load_csv(p)
            if not adjusted_df.empty:
                st.caption(f"Loaded fallback: `{p}`")
                break

    if adjusted_df.empty:
        st.warning("Adjusted file not found. Provide any of: "
                   f"`{adjusted_p.name}`, `week{week}_adjusted_picks_CORRECTED.csv`, or `week{week}_adjusted_picks.csv` in `data/`.")
    else:
        report_df = norm_adjusted(adjusted_df).copy()
        if "Confidence %" in report_df.columns:
            report_df["Band"] = report_df["Confidence %"].apply(band)
            show_cols = ["Matchup","Adjusted ML","Confidence %","Band"]
            if "Guardrail Note" in report_df.columns:
                show_cols.append("Guardrail Note")
        else:
            show_cols = list(report_df.columns)
        st.dataframe(report_df[show_cols], use_container_width=True)

        # Ticket groups
        st.markdown("### 🎟️ Tickets")
        groups = group_rows(report_df, sizes=sizes)
        for i, g in enumerate(groups, start=1):
            gshow = g.copy()
            if "Confidence %" in gshow.columns and "Band" not in gshow.columns:
                gshow["Band"] = gshow["Confidence %"].apply(band)
            st.markdown(f"**Group {i}**")
            st.dataframe(gshow[show_cols], use_container_width=True)

        # Downloads
        c1,c2,c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ Download ALL (CSV)",
                               report_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"Week{week}_All.csv", mime="text/csv")
        with c2:
            buf = io.StringIO()
            for i, g in enumerate(groups, start=1):
                buf.write(f"# Group {i}\n")
                g.to_csv(buf, index=False)
                buf.write("\n")
            st.download_button("⬇️ Download Grouped (CSV)",
                               buf.getvalue().encode("utf-8"),
                               file_name=f"Week{week}_Grouped.csv", mime="text/csv")
        with c3:
            try:
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib import colors
                def build_pdf(df_groups, title="NFL Weekly Picks Report"):
                    styles = getSampleStyleSheet()
                    out = io.BytesIO()
                    doc = SimpleDocTemplate(out, pagesize=letter)
                    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
                    for i, g in enumerate(df_groups, start=1):
                        story.append(Paragraph(f"Group {i}", styles["Heading2"]))
                        data = [list(g.columns)] + g.astype(str).values.tolist()
                        t = Table(data, repeatRows=1)
                        t.setStyle(TableStyle([
                            ("BACKGROUND",(0,0),(-1,0),colors.grey),
                            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
                            ("ALIGN",(0,0),(-1,-1),"CENTER"),
                            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                            ("BOTTOMPADDING",(0,0),(-1,0),6),
                            ("BACKGROUND",(0,1),(-1,-1),colors.beige),
                            ("GRID",(0,0),(-1,-1),0.5,colors.black),
                        ]))
                        story.append(t); story.append(Spacer(1, 12))
                    doc.build(story)
                    return out.getvalue()
                pdf_bytes = build_pdf(groups, title=f"NFL Week {week} Picks Report")
                st.download_button("🧾 Download PDF", pdf_bytes,
                                   file_name=f"Week{week}_Picks_Report.pdf",
                                   mime="application/pdf")
            except Exception:
                st.caption("Install `reportlab` to enable one-click PDF export: `pip install reportlab`")

# Draft
with tab_draft:
    df = load_csv(draft_p, cols=["Matchup","Away","Home","Spread","Over/Under"])
    if df.empty: st.info("Upload WeekX_Draft.csv to view raw schedule/odds.")
    else: st.dataframe(df, use_container_width=True)

# Final
with tab_final:
    df = load_csv(final_p)
    if df.empty: st.info("Upload WeekX_Final.csv after games to lock & display results.")
    else: st.dataframe(df, use_container_width=True)

# Odds
with tab_odds:
    df = load_csv(odds_p)
    if df.empty: st.info("Upload WeekX_Odds.csv for the week’s market totals.")
    else: st.dataframe(df, use_container_width=True)

# Injuries
with tab_inj:
    df = load_csv(inj_p)
    if df.empty: st.info("Upload WeekX_Injuries.csv to see the week’s injury sheet.")
    else: st.dataframe(df, use_container_width=True)

# Guardrails
with tab_guard:
    df = load_csv(guard_p)
    if df.empty:
        st.info("Upload WeekX_GuardrailReport.csv to view triggered guardrails and effects.")
        st.markdown("- **Active Global Guardrail:** MNF Home-Dog Fade — if sharps heavily back a road dog on MNF, auto-downgrade or flip to home ML unless contradicted by stronger guardrails.")
    else:
        st.dataframe(df, use_container_width=True)

# ---------- Season Master (read-only) ----------
st.divider()
st.header("📈 Season Master Tracker (read-only)")
if ROOT_MASTER.exists():
    try:
        m = pd.read_csv(ROOT_MASTER)
        st.dataframe(m, use_container_width=True)
        if "ML Result" in m.columns:
            ml = m["ML Result"].astype(str).str.strip()
            total = (ml == "✅").sum() + (ml == "❌").sum()
            wins = (ml == "✅").sum()
            if total > 0:
                st.caption(f"Season ML Record: **{wins}-{total - wins}**  ({wins/total:.1%})")
    except Exception as e:
        st.error(f"Could not read season master: {e}")
else:
    st.info("Add `data/season_master_tracker.csv` to show season-to-date results.")import streamlit as st
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
