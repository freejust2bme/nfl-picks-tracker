
# app.py — NFL Picks Tracker (Compat Build: no modern type-hints)
# Works on Python 3.7+ / Streamlit 1.x

import io
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Picks Tracker", layout="wide")
st.title("🏈 NFL Picks Tracker")

DATA_DIR = Path("data")
VERS_DIR = DATA_DIR / "versioned_weeks"
ROOT_MASTER = DATA_DIR / "season_master_tracker.csv"

# -------------------- Helpers --------------------
def exists(p):
    try:
        return p.exists()
    except Exception:
        return False

def load_csv(p, cols=None):
    """Safely load a CSV; optionally filter to specific columns if present."""
    if not exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if cols:
            keep = [c for c in cols if c in df.columns]
            if keep:
                df = df[keep]
        return df
    except Exception as e:
        st.error("Error reading {}: {}".format(p, e))
        return pd.DataFrame()

def normalize_adjusted(df):
    """Normalize Adjusted picks to columns: Matchup | Adjusted ML | Confidence % | Guardrail Note."""
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "Adjusted Pick": "Adjusted ML",
        "Match-up": "Matchup",
        "Confidence%": "Confidence %",
        "confidence": "Confidence %",
        "confidence %": "Confidence %",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    cols_order = [c for c in ["Matchup", "Adjusted ML", "Confidence %", "Guardrail Note"] if c in df.columns]
    if not cols_order:
        return df.copy()

    out = df[cols_order].copy()
    if "Confidence %" in out.columns:
        out["Confidence %"] = pd.to_numeric(out["Confidence %"], errors="coerce").round(0)
    return out

def group_rows(d, sizes=(6, 6, 4)):
    """Split rows into groups (tickets). Leftover rows go into a final group."""
    out = []
    start = 0
    for size in sizes:
        if start >= len(d):
            break
        out.append(d.iloc[start:start + size].reset_index(drop=True))
        start += size
    if start < len(d):
        out.append(d.iloc[start:].reset_index(drop=True))
    return out

def band(v):
    """Confidence band for quick scan."""
    try:
        v = float(v)
    except Exception:
        return "—"
    if v >= 75:
        return "🟩 strong"
    if v >= 60:
        return "🟨 solid"
    return "🟥 lean"

def build_grouped_csv(groups):
    buf = io.StringIO()
    for i, g in enumerate(groups, start=1):
        buf.write("# Group {}\n".format(i))
        g.to_csv(buf, index=False)
        buf.write("\n")
    return buf.getvalue().encode("utf-8")

def build_pdf_from_groups(groups, title="NFL Weekly Picks Report"):
    """Create a simple PDF report (if reportlab installed)."""
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors

        styles = getSampleStyleSheet()
        doc_buf = io.BytesIO()
        doc = SimpleDocTemplate(doc_buf, pagesize=letter)
        story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
        for i, g in enumerate(groups, start=1):
            story.append(Paragraph("Group {}".format(i), styles["Heading2"]))
            data = [list(g.columns)] + g.astype(str).values.tolist()
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))
        doc.build(story)
        return doc_buf.getvalue()
    except Exception:
        return None

# ---------- Lock Week helpers ----------
def load_master(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def normalize_final(df, week_num):
    """
    Normalize Final CSV for merge:
      Week | Matchup | Adjusted ML | Final Score | ML Result | ATS Result | Notes
    Common variants are mapped.
    """
    if df.empty:
        return df.copy()

    rename = {
        "Pick": "Adjusted ML",
        "Adjusted Pick": "Adjusted ML",
        "Result": "ML Result"
    }
    for k, v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    for col in ["Week", "Matchup", "Adjusted ML", "Final Score", "ML Result", "ATS Result", "Notes"]:
        if col not in df.columns:
            df[col] = ""

    # If Week missing or blank, set to current selected week
    if (df["Week"] == "").all() or df["Week"].isna().all():
        df["Week"] = week_num

    keep = ["Week", "Matchup", "Adjusted ML", "Final Score", "ML Result", "ATS Result", "Notes"]
    return df[keep]

def merge_into_master(master_df, week_final_df):
    """Replace existing rows for this week/matchups, then append the new finals."""
    if master_df.empty:
        return week_final_df.copy()
    keys_new = set(zip(week_final_df["Week"].astype(str), week_final_df["Matchup"].astype(str)))
    mask_old = master_df.apply(lambda r: (str(r.get("Week", "")), str(r.get("Matchup", ""))) in keys_new, axis=1)
    master_df = master_df.loc[~mask_old].copy()
    return pd.concat([master_df, week_final_df], ignore_index=True)

# -------------------- Sidebar --------------------
st.sidebar.header("Controls")
week = st.sidebar.number_input("Week", min_value=1, max_value=18, value=4, step=1)
group_pattern = st.sidebar.text_input("Group pattern (comma separated)", value="6,6,4")
try:
    sizes = tuple(int(x.strip()) for x in group_pattern.split(",") if x.strip())
except Exception:
    sizes = (6, 6, 4)

# -------------------- Expected Weekly Files --------------------
draft_p    = VERS_DIR / "Week{}_Draft.csv".format(week)
adjusted_p = VERS_DIR / "Week{}_Adjusted.csv".format(week)
final_p    = VERS_DIR / "Week{}_Final.csv".format(week)
odds_p     = VERS_DIR / "Week{}_Odds.csv".format(week)
inj_p      = VERS_DIR / "Week{}_Injuries.csv".format(week)
guard_p    = VERS_DIR / "Week{}_GuardrailReport.csv".format(week)

st.subheader("📦 Week {} Bundle — File Check".format(week))
colA, colB, colC = st.columns(3)
with colA:
    st.write(("✅" if exists(draft_p) else "❌") + " Draft — `{}`".format(draft_p))
    st.write(("✅" if exists(adjusted_p) else "❌") + " Adjusted — `{}`".format(adjusted_p))
with colB:
    st.write(("✅" if exists(final_p) else "❌") + " Final — `{}`".format(final_p))
    st.write(("✅" if exists(odds_p) else "❌") + " Odds — `{}`".format(odds_p))
with colC:
    st.write(("✅" if exists(inj_p) else "❌") + " Injuries — `{}`".format(inj_p))
    st.write(("✅" if exists(guard_p) else "❌") + " Guardrail Report — `{}`".format(guard_p))

st.divider()

# -------------------- Tabs --------------------
tab_adj, tab_draft, tab_final, tab_odds, tab_inj, tab_guard = st.tabs(
    ["Adjusted (Report View)", "Draft", "Final", "Odds", "Injuries", "Guardrails"]
)

# Adjusted
with tab_adj:
    adjusted_df = load_csv(adjusted_p)
    if adjusted_df.empty:
        # fallbacks (for convenience)
        fallbacks = [
            DATA_DIR / "week{}_adjusted_picks_CORRECTED.csv".format(week),
            DATA_DIR / "week{}_adjusted_picks.csv".format(week),
        ]
        for p in fallbacks:
            adjusted_df = load_csv(p)
            if not adjusted_df.empty:
                st.caption("Loaded fallback: `{}`".format(p))
                break

    if adjusted_df.empty:
        st.warning(
            "Adjusted file not found. Provide one of: `{}`, `week{}_adjusted_picks_CORRECTED.csv`, or `week{}_adjusted_picks.csv` in `data/`."
            .format(adjusted_p.name, week, week)
        )
    else:
        report_df = normalize_adjusted(adjusted_df).copy()
        if "Confidence %" in report_df.columns:
            report_df["Band"] = report_df["Confidence %"].apply(band)
            show_cols = ["Matchup", "Adjusted ML", "Confidence %", "Band"]
            if "Guardrail Note" in report_df.columns:
                show_cols.append("Guardrail Note")
        else:
            show_cols = list(report_df.columns)

        st.caption("Source: `{}`".format(adjusted_p if exists(adjusted_p) else "fallback file"))
        st.dataframe(report_df[show_cols], use_container_width=True)

        # Tickets
        st.markdown("### 🎟️ Ticket Groups")
        groups = group_rows(report_df, sizes=sizes)
        for i, g in enumerate(groups, start=1):
            gshow = g.copy()
            if "Confidence %" in gshow.columns and "Band" not in gshow.columns:
                gshow["Band"] = gshow["Confidence %"].apply(band)
            st.markdown("**Group {}**".format(i))
            st.dataframe(gshow[show_cols], use_container_width=True)

        # Downloads
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ Download ALL (CSV)",
                report_df.to_csv(index=False).encode("utf-8"),
                file_name="Week{}_All.csv".format(week),
                mime="text/csv"
            )
        with c2:
            st.download_button(
                "⬇️ Download Grouped (CSV)",
                build_grouped_csv(groups),
                file_name="Week{}_Grouped.csv".format(week),
                mime="text/csv"
            )
        with c3:
            pdf_bytes = build_pdf_from_groups(groups, title="NFL Week {} Picks Report".format(week))
            if pdf_bytes:
                st.download_button(
                    "🧾 Download PDF",
                    pdf_bytes,
                    file_name="Week{}_Picks_Report.pdf".format(week),
                    mime="application/pdf"
                )
            else:
                st.caption("Install `reportlab` for one-click PDF export: `pip install reportlab`")

# Draft
with tab_draft:
    df = load_csv(draft_p, cols=["Matchup", "Away", "Home", "Spread", "Over/Under"])
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

# -------------------- Lock Week --------------------
st.divider()
st.subheader("🔒 Lock Week & Update Season Master")

final_df_raw = load_csv(final_p)
if final_df_raw.empty:
    st.info("To lock a week, provide `WeekX_Final.csv` in `data/versioned_weeks/`.")
else:
    week_final_df = normalize_final(final_df_raw, week_num=week)
    st.write("Preview — rows to merge into Season Master:")
    st.dataframe(week_final_df, use_container_width=True)

    problems = []
    if "Matchup" not in week_final_df.columns:
        problems.append("Missing `Matchup` column in Final file.")
    if ROOT_MASTER.exists() and not ROOT_MASTER.is_file():
        problems.append("Season master path exists but is not a file.")
    if not exists(adjusted_p):
        problems.append("Adjusted file missing — best practice is Draft → Adjusted → Final before lock.")

    if problems:
        st.error("Cannot lock this week due to:\n- " + "\n- ".join(problems))
    else:
        if st.button("Lock Week {} and Update Season Master".format(week)):
            master_df = load_master(ROOT_MASTER)
            merged = merge_into_master(master_df, week_final_df)
            try:
                merged.to_csv(ROOT_MASTER, index=False)
                if "ML Result" in merged.columns:
                    ml = merged["ML Result"].astype(str).str.strip()
                    total = (ml == "✅").sum() + (ml == "❌").sum()
                    wins = (ml == "✅").sum()
                    pct = (wins / total) if total else 0.0
                    st.success("Locked Week {} and updated Season Master. Season ML record: **{}-{} ({:.1%})**".format(
                        week, wins, total - wins, pct
                    ))
                else:
                    st.success("Locked Week {} and updated Season Master.".format(week))
            except Exception as e:
                st.error("Failed to write season master: {}".format(e))

# -------------------- Season Master (read-only) --------------------
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
                st.caption("Season ML Record: **{}-{}**  ({:.1%})".format(wins, total - wins, wins/total))
    except Exception as e:
        st.error("Could not read season master: {}".format(e))
else:
    st.info("Add `data/season_master_tracker.csv` to show season-to-date results.")
