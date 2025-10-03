# app.py
# Streamlit NFL Picks (ML-only) — with guardrail badges + one-click weekly refresh
# -------------------------------------------------------------------------------
# Files expected (CSV):
#   data/versioned_weeks/nfl_2025_week{W}_schedule.csv
#     -> GameID,Week,HomeTeam,AwayTeam,Site(optional),StartET(optional)
#   data/versioned_weeks/nfl_2025_week{W}_odds.csv
#     -> GameID,Market,Team,Price,Line,Book(optional)   (use Market="ML" rows)
#   data/versioned_weeks/Week{W}_Injuries.csv
#     -> Team,Player,Position,Injury,GameStatus,Notes,LastUpdated
#
# Outputs (written on refresh):
#   Week{W}_Tickets_DETAILED.csv  (legs with "PickTeam over Opponent — ML (-xxx)" + badges)
#   data/versioned_weeks/Week{W}_Tickets_VIEW.csv  (compact per-ticket card)
#   Week{W}_Tickets.csv           (simple summary; optional for other tabs)
#
# If something broke after last update, this file is a safe reset:
#   - ML-only (no spreads/totals)
#   - Clear "Display" showing the team you picked
#   - Guardrail badges: ✅ Anchor/Safe, ⚠️ Moderate, ❌ Pass (Pass legs are filtered out)

import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
import streamlit as st
# --- Load NFL schedule + Week 5 odds from GitHub repo ---

# GitHub raw file URLs
schedule_url = "https://raw.githubusercontent.com/freejust2bme/nfl-picks-repo/main/data/nfl_2025_master_schedule.csv"
week5_odds_url = "https://raw.githubusercontent.com/freejust2bme/nfl-picks-repo/main/data/week_5_odds_template.csv"

# Load full season schedule
schedule = pd.read_csv(schedule_url)

# Load Week 5 odds
week5_odds = pd.read_csv(week5_odds_url)

# --- Example checks ---
print("Schedule shape:", schedule.shape)
print("Week 5 odds shape:", week5_odds.shape)

# ------------------------ Basic setup ------------------------
st.set_page_config(page_title="NFL Picks — ML Guardrails", layout="wide")

APP_DATA_DIR = Path("data/versioned_weeks")
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------ Helpers ------------------------
def ml_to_decimal(ml):
    ml = float(ml)
    return 1 + (ml/100.0) if ml > 0 else 1 + (100.0/abs(ml))

def implied_prob_from_ml(ml):
    ml = float(ml)
    return (100/(ml+100)) if ml > 0 else (abs(ml)/(abs(ml)+100))

def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def badge_for_class(cls):
    return {"Anchor": "✅", "Safe": "✅", "Moderate": "⚠️", "Pass": "❌"}.get(cls, "")

# ------------------------ Guardrails ------------------------
def guardrail_rating(row, injuries_df):
    """Return Anchor | Safe | Moderate | Pass based on injuries + spot (ML only)."""
    pick   = row["PickTeam"]
    ml     = float(row["PriceML"])
    site   = str(row.get("Site","")).lower()

    # Injury red flags for PickTeam: any Out / Doubtful
    team_inj = injuries_df[injuries_df["Team"].str.lower()==pick.lower()] if not injuries_df.empty else pd.DataFrame()
    has_out = False
    if not team_inj.empty:
        gs = team_inj["GameStatus"].astype(str).str.lower()
        has_out = gs.str.contains("out").any() or gs.str.contains("doubtful").any()

    # Situational: neutral/intl sites -> more variance
    neutral = site in {"neutral", "london", "mexico", "brazil"}

    # Road favorite band can be noisy: -170 to -260 and away
    is_road = (pick == row["AwayTeam"])
    road_fav_band = (ml <= -170 and ml >= -260 and is_road)

    if has_out:
        return "Pass"
    if neutral:
        return "Moderate"
    if road_fav_band:
        return "Moderate"
    if ml <= -500:
        return "Anchor"
    if ml <= -250:
        return "Safe"
    return "Moderate"

# ------------------------ Loads ------------------------
def load_week_inputs(week:int):
    sched_path = APP_DATA_DIR / f"nfl_2025_week{week}_schedule.csv"
    odds_path  = APP_DATA_DIR / f"nfl_2025_week{week}_odds.csv"
    inj_path   = APP_DATA_DIR / f"Week{week}_Injuries.csv"

    missing = []
    if not sched_path.exists(): missing.append(sched_path)
    if not odds_path.exists():  missing.append(odds_path)
    if not inj_path.exists():   # injuries optional; we’ll create empty if missing
        inj = pd.DataFrame(columns=["Team","Player","Position","Injury","GameStatus","Notes","LastUpdated"])
    else:
        inj = pd.read_csv(inj_path)

    if missing:
        return None, None, inj, missing

    sched = pd.read_csv(sched_path)
    odds  = pd.read_csv(odds_path)
    return sched, odds, inj, []

# ------------------------ Build Board (ML only) ------------------------
def safest_board(sched, odds, inj):
    # Keep ML only
    ml = odds[odds["Market"].astype(str).str.upper().eq("ML")].copy()
    if ml.empty:
        return pd.DataFrame()

    # Most negative = favorite; take one per game
    ml["Price"] = ml["Price"].astype(float)
    favs = (ml.sort_values(["GameID","Price"])  # most negative first
              .groupby("GameID", as_index=False)
              .first())

    # join schedule
    use_cols = ["GameID","Week","HomeTeam","AwayTeam","Site"] if "Site" in sched.columns else ["GameID","Week","HomeTeam","AwayTeam"]
    favs = favs.merge(sched[use_cols], on="GameID", how="left")

    # rows
    rows = []
    for _, r in favs.iterrows():
        pick_team = r["Team"]
        opponent  = r["HomeTeam"] if pick_team == r["AwayTeam"] else r["AwayTeam"]
        rows.append({
            "Week": int(r["Week"]),
            "GameID": r["GameID"],
            "HomeTeam": r["HomeTeam"],
            "AwayTeam": r["AwayTeam"],
            "PickTeam": pick_team,
            "Opponent": opponent,
            "Market": "ML",
            "Line": "",
            "PriceML": float(r["Price"]),
            "Site": r.get("Site","") if "Site" in r else ""
        })
    board = pd.DataFrame(rows)
    if board.empty: 
        return board

    # guardrails
    board["GuardrailClass"] = board.apply(lambda x: guardrail_rating(x, inj), axis=1)
    board["Badge"] = board["GuardrailClass"].map(badge_for_class)
    # rank anchors/safe first, most negative price first
    class_rank = {"Anchor":0,"Safe":1,"Moderate":2,"Pass":9}
    board["ClassRank"] = board["GuardrailClass"].map(class_rank)
    board.sort_values(["ClassRank","PriceML"], inplace=True)
    return board.reset_index(drop=True)

# ------------------------ Tickets ------------------------
def build_tickets_from_board(board, week:int):
    """Create three safest tickets: 2-leg, 3-leg, 4-leg (Anchor/Safe only; exclude Pass)."""
    if board.empty:
        return pd.DataFrame(), pd.DataFrame()

    pool = board.query("GuardrailClass in ['Anchor','Safe']").copy()
    if pool.empty:
        # if nothing is Anchor/Safe, allow Moderate so we still show *something*
        pool = board.query("GuardrailClass != 'Pass'").copy()

    # choose top legs
    legs_2 = pool.head(2)
    legs_3 = pool.head(3)
    legs_4 = pool.head(4)

    # helper: display text with badge
    def make_disp(r):
        opp = r["Opponent"]
        price = int(r["PriceML"])
        return f"{r['Badge']} {r['PickTeam']} over {opp} — ML ({price})"

    def add_ticket(name, dflegs):
        if dflegs.empty: 
            return []
        out = []
        for _, r in dflegs.iterrows():
            out.append({
                "TicketName": name,
                "Week": week,
                "GameID": r["GameID"],
                "HomeTeam": r["HomeTeam"],
                "AwayTeam": r["AwayTeam"],
                "PickTeam": r["PickTeam"],
                "Market": "ML",
                "Line": "",
                "PriceML": int(r["PriceML"]),
                "GuardrailClass": r["GuardrailClass"],
                "Display": make_disp(r)
            })
        return out

    tix = []
    tix += add_ticket("Ticket 1 — Anchor 2-Leg", legs_2)
    tix += add_ticket("Ticket 2 — Safe 3-Leg",   legs_3)
    tix += add_ticket("Ticket 3 — Safe 4-Leg",   legs_4)

    tix_df = pd.DataFrame(tix)

    # summaries
    def summarize(g):
        decs = [ml_to_decimal(x) for x in g["PriceML"]]
        dec_prod = float(np.prod(decs)) if decs else 1.0
        payout_20 = round(20*dec_prod,2)
        return pd.Series({
            "Legs": len(g),
            "DecimalOdds": round(dec_prod,3),
            "Payout_$20": payout_20,
            "Updated": now_str()
        })
    view = tix_df.groupby("TicketName", as_index=False).apply(summarize) if not tix_df.empty else pd.DataFrame()
    return tix_df, view

# REPLACE the existing write_week_outputs() with this version
def write_week_outputs(week:int, tix_df, view_df):
    """Write detailed legs + compact ticket cards. Ensures folder exists."""
    out_dir = Path("data/versioned_weeks")
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = Path(f"Week{week}_Tickets_DETAILED.csv")
    tix_df.to_csv(detailed_path, index=False)

    view_path = out_dir / f"Week{week}_Tickets_VIEW.csv"
    # ---- fix: make sure we use the function arg 'view_df' ----
    view_df.to_csv(view_path, index=False)

    simple_path = Path(f"Week{week}_Tickets.csv")
    simple = view_df.rename(columns={"TicketName":"Ticket","Payout_$20":"PotentialReturn"})
    simple.to_csv(simple_path, index=False)

    return detailed_path, view_path, simple_path

def one_click_refresh(week:int):
    sched, odds, inj, missing = load_week_inputs(week)
    if missing:
        return None, None, None, missing
    board = safest_board(sched, odds, inj)
    if board.empty:
        return pd.DataFrame(), pd.DataFrame(), [], []
    tix, view = build_tickets_from_board(board, week)
    if tix.empty:
        return tix, view, [], []
    d, v, s = write_week_outputs(week, tix, view)
    return (tix, view, [d, v, s], [])

# ------------------------ UI ------------------------
st.title("NFL Picks — Moneyline Guardrails (ML-Only)")
st.caption("Vision: 100% accuracy • Mission: create tickets that fulfill the vision")

week = st.number_input("Select Week", min_value=1, max_value=18, value=5, step=1)

colA, colB = st.columns([1,1])
with colA:
    if st.button("🔄 Refresh this Week (Schedule + Odds + Injuries → Tickets)"):
        legs_df, cards_df, paths, missing = one_click_refresh(int(week))
        if missing:
            st.error("Missing required files:")
            for p in missing:
                st.write("•", str(p))
        elif legs_df is None:
            st.error("Could not load inputs. Check your CSVs and try again.")
        elif legs_df.empty:
            st.warning("No tickets produced (no Anchor/Safe MLs found). Check odds file.")
        else:
            st.success(f"Week {int(week)} refreshed!")
            st.markdown("**Ticket Legs (with chosen side + badge):**")
            st.dataframe(legs_df[["TicketName","Display","PriceML","GuardrailClass"]], use_container_width=True)
            st.markdown("**Ticket Cards (summary):**")
            st.dataframe(cards_df, use_container_width=True)
            st.caption("Files written:")
            for p in paths:
                st.caption(f"• {p}")
with colB:
    st.info("Tip: ML-only mode is locked in. Spreads/totals are disabled by design to protect accuracy.")

st.divider()

# Quick viewer for current week's detailed file if it exists
detailed_path = Path(f"Week{int(week)}_Tickets_DETAILED.csv")
if detailed_path.exists():
    st.subheader(f"Week {int(week)} — Current Tickets (Detailed)")
    st.dataframe(pd.read_csv(detailed_path), use_container_width=True)
else:
    st.caption("No detailed ticket file found yet. Click refresh after adding schedule/odds/injuries CSVs.")




        
