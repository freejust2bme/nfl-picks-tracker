# NFL PICKS TRACKER (CLEAN)

import json
from pathlib import Path
import streamlit as st

st.set_page_config(layout="wide", page_title="NFL Ticket App")

# --- helpers ---
@st.cache_data
def load_tickets(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return {}

def render_ticket_card(title: str, legs):
    st.markdown(f"### 🎟️ {title}")
    if not legs:
        st.info("No legs found for this ticket yet.")
        return
    box = st.container()  # (no border=True to avoid version issues)
    with box:
        st.caption("PARLAY — all legs must win")
        for i, leg in enumerate(legs, 1):
            if isinstance(leg, dict):
                label = leg.get("label") or f"{leg.get('PickTeam','?')} {leg.get('Spread','')}".strip()
                odds  = leg.get("Odds") or leg.get("odds") or ""
            else:
                label, odds = str(leg), ""
            st.write(f"**{i}.** {label}  {odds}")

# --- UI ---
tickets = load_tickets("out/week5_tickets.json")  # adjust path/week if needed

colA, colB, colC = st.columns(3)
with colA:
    render_ticket_card("Ticket A — Anchor",      tickets.get("A"))
with colB:
    render_ticket_card("Ticket B — Multiplier",  tickets.get("B"))
with colC:
    render_ticket_card("Ticket C — Upset",       tickets.get("C"))

# manual safe reload (no experimental attributes)
if st.button("🔄 Reload tickets"):
    load_tickets.clear()
    if hasattr(st, "rerun"):
        st.rerun()                 # modern API
    else:
        st.experimental_rerun()    # fallback for older versions

{
  "A": [
    {"label": "Bills -9 vs Patriots", "odds": "-110"},
    {"label": "Eagles -3.5 vs Broncos", "odds": "-110"}
  ],
  "B": [
    {"label": "Texans -2.5 @ Ravens", "odds": "-110"},
    {"label": "Cowboys -2.5 @ Jets", "odds": "-110"}
  ],
  "C": [
    {"label": "Giants +3 vs Saints", "odds": "-105"},
    {"label": "Vikings -3.5 @ Browns", "odds": "-110"}
  ]
}
