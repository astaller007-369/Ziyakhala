"""
sisonke_app.py
================

The Streamlit UI for the Sisonke Football Predictive Terminal. All the
actual math lives in sisonke_engine.py (tested independently - see
test_engine.py, integration_dry_run.py, and bias_stress_test.py) - this
file is purely the dashboard wiring: tabs, sidebar, inputs, charts.

RUN:
    streamlit run sisonke_app.py

HONESTY NOTES:
- The Gold Mine + League Playstyle panels are general football-reputation
  starting points, NOT the output of a rigorous statistical backtest of
  each specific league - see sisonke_engine.py's own notes on these.
- Tactical multiplier percentages have been reviewed against general
  football-analytics literature (see apply_tactical_multipliers'
  docstring for specifics on what was adjusted and why), but several -
  especially the counter-press style, pitch/weather effects, and referee
  strictness beyond the one spec'd value - remain reasoned estimates,
  not values fitted to your own data. They're editable in the sidebar for
  exactly that reason.
- This model is built for standard home/away league play (round-robin
  points tables) - NOT cup/knockout/tournament competitions, which follow
  different incentive and squad-rotation patterns entirely. Divisions
  that look like cups (by name) are filtered out of the workspace
  selector automatically.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sisonke_engine as E

st.set_page_config(page_title="⚽ Sisonke Football Predictive Terminal", page_icon="⚽", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_FILE = DATA_DIR / "master_sisonke_database.csv"


# ---------------------------------------------------------------------------
# Local storage (Section 1 upload port + requested download/clear controls)
# ---------------------------------------------------------------------------
def _atomic_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_database_from_disk():
    if DB_FILE.exists():
        try:
            return E.standardise_columns(pd.read_csv(DB_FILE, dtype=str))
        except Exception:
            return None
    return None


def save_database_to_disk(df: pd.DataFrame):
    _atomic_write_csv(df, DB_FILE)


def prepare_raw_upload(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardises columns AND normalises team-name casing so 'Chelsea'
    and 'chelsea' resolve to one team, right at load time - before
    anything else touches the data."""
    df = E.standardise_columns(raw_df)
    df = E.normalize_name_casing(df, ["home_team", "away_team"])
    return df


def coerce_working_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_raw_upload(raw_df)
    numeric_cols = [
        "home_goals", "away_goals",
        "home_shots_on_target", "away_shots_on_target",
        "home_big_chances", "away_big_chances",
        "home_box_touches", "away_box_touches",
    ]
    df = E.coerce_numeric(df, numeric_cols)
    df = E.parse_dates(df, "date")
    return df


if "raw_db" not in st.session_state:
    st.session_state.raw_db = load_database_from_disk()
if "bookmaker_odds" not in st.session_state:
    st.session_state.bookmaker_odds = {m: 2.00 for m in E.MARKET_LIST}
if "title_odds" not in st.session_state:
    st.session_state.title_odds = {}


def find_division_series(df: pd.DataFrame):
    col = E.find_division_column(df)
    if col is None:
        return None, None, None
    all_divisions = sorted(df[col].dropna().unique().tolist())
    standard, excluded = E.filter_to_standard_leagues(all_divisions)
    return col, standard, excluded


# ---------------------------------------------------------------------------
# Sidebar (Section 1 + local storage controls)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚽ Sisonke Control Deck")
    active_tab = st.radio(
        "Active workspace",
        ["📁 Research & Sentiment Tracker", "📊 Active Projections Matrix"],
        key="active_workspace",
    )

    st.divider()
    st.subheader("📤 Historical Matchday Upload")
    uploaded = st.file_uploader("Upload your database (.csv)", type=["csv"], key="db_uploader")
    if uploaded is not None:
        try:
            new_raw = pd.read_csv(uploaded, dtype=str)
            new_raw = prepare_raw_upload(new_raw)
            st.session_state.raw_db = new_raw
            save_database_to_disk(new_raw)
            st.success(f"✅ Saved as master_sisonke_database.csv ({len(new_raw)} rows).")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read that file: {exc}")

    if st.session_state.raw_db is not None:
        st.caption(f"💾 Database loaded: {len(st.session_state.raw_db)} row(s).")

        st.download_button(
            "⬇️ Download current database (.csv)",
            data=st.session_state.raw_db.to_csv(index=False).encode("utf-8"),
            file_name="master_sisonke_database.csv",
            mime="text/csv",
            key="db_download_btn",
        )

        with st.expander("🧹 Clear data"):
            _dcol, _divs, _ = find_division_series(st.session_state.raw_db)
            if _dcol:
                clear_division = st.selectbox("League to clear", ["(pick one)"] + _divs, key="clear_division_pick")
                if st.button("Clear this league only", key="clear_one_league"):
                    if clear_division != "(pick one)":
                        remaining = st.session_state.raw_db[st.session_state.raw_db[_dcol] != clear_division]
                        st.session_state.raw_db = remaining
                        save_database_to_disk(remaining)
                        st.success(f"Cleared '{clear_division}'.")
                        st.rerun()
            confirm_wipe = st.checkbox("I understand this deletes the ENTIRE database", key="confirm_wipe_db")
            if st.button("🗑️ Clear ALL data", disabled=not confirm_wipe, key="clear_all_db"):
                st.session_state.raw_db = None
                if DB_FILE.exists():
                    DB_FILE.unlink()
                st.success("Database cleared.")
                st.rerun()
    else:
        st.caption("No database loaded yet.")

    st.divider()
    with st.expander("📱 Telegram Notifications"):
        st.caption("Sends the current prediction on demand - never runs automatically, and is the only part of this app that needs internet access.")
        tg_token = st.text_input("Bot Token", type="password", key="tg_token")
        tg_chat_id = st.text_input("Chat ID", key="tg_chat_id")
        st.session_state["tg_token"], st.session_state["tg_chat_id"] = tg_token, tg_chat_id


if st.session_state.raw_db is None:
    st.title("⚽ SISONKE FOOTBALL HUB")
    st.info("👋 Upload your historical matchday database in the sidebar to get started.")
    st.stop()

working_df = coerce_working_frame(st.session_state.raw_db)
division_col, divisions, excluded_divisions = find_division_series(working_df)

if division_col is None:
    st.error(
        "⚠️ Couldn't find a division column - your CSV needs one named "
        "`league_country`, `league`, or `competition`."
    )
    st.stop()

if not divisions:
    st.error("⚠️ Every division in this file looks like a cup/tournament competition - this model is built strictly for standard league play.")
    st.stop()

if excluded_divisions:
    st.sidebar.caption(
        f"🚫 {len(excluded_divisions)} cup/tournament competition(s) hidden "
        f"(this model is for standard league play only): {', '.join(excluded_divisions[:3])}"
        + ("..." if len(excluded_divisions) > 3 else "")
    )


def league_profile_banner(division: str):
    hint = E.gold_mine_hint(division)
    style = E.league_playstyle_profile(division)
    st.info(f"💡 **{hint}**")
    st.caption(f"🎨 League profile: {style}")


# ---------------------------------------------------------------------------
# Section 2: Research & Sentiment Tracker (offline tab)
# ---------------------------------------------------------------------------
def render_sentiment_tracker():
    st.title("📁 Research & Sentiment Tracker")
    st.caption("🔒 An isolated, offline screening workspace - nothing here blocks the main analytics hub downstream.")

    division = st.selectbox("🏆 League workspace", divisions, key="sentiment_division")
    league_profile_banner(division)
    division_df = working_df[working_df[division_col] == division]
    settled_div, upcoming_div = E.split_played_unplayed(division_df)

    if upcoming_div.empty:
        st.warning("No unplayed fixtures detected for this division (no blank/comma goal cells found).")
        return

    fixture_labels = [
        f"{r.home_team} vs {r.away_team}" + (f" ({r.date.date()})" if pd.notna(r.date) else "")
        for r in upcoming_div.itertuples()
    ]
    fixture_choice = st.selectbox("🎯 Select Target Upcoming Fixture", fixture_labels, key="sentiment_fixture")

    st.subheader("📋 The 7-Day Diary Checklist")
    c1, c2, c3, c4 = st.columns(4)
    d7 = c1.checkbox("📅 7 Days Out - initial team news scanned", key="diary_7d")
    d72 = c2.checkbox("🕐 72 Hours Out - press conference checked", key="diary_72h")
    d24 = c3.checkbox("⏰ 24 Hours Out - confirmed absentees noted", key="diary_24h")
    d60 = c4.checkbox("⏱️ 60 Mins Out - final lineup confirmed", key="diary_60m")
    ticked = sum([d7, d72, d24, d60])

    st.subheader("🎭 Sentiment")
    sentiment = st.selectbox(
        "Current season motivation", ["🏖️ Beach Mode", "📉 Relegation Battle", "📈 Promotion Race", "🔥 Derby"],
        key="sentiment_choice",
    )

    confidence = round((ticked / 4) * 10)
    st.subheader("🎯 Confidence Rating")
    st.metric("Confidence Score (out of 10)", confidence)
    if confidence <= 3:
        st.error("🔴 PASS / NO BET - insufficient information gathered for this fixture.")
    else:
        st.success(f"🟢 Sufficient research depth logged ({ticked}/4 checklist items).")
    st.caption("ℹ️ This rating is advisory only - it never locks or breaks the Active Projections Matrix downstream, even at a PASS rating.")


# ---------------------------------------------------------------------------
# Active Projections Matrix tab (the main calculator)
# ---------------------------------------------------------------------------
def render_projections_matrix():
    division = st.selectbox("🏆 League workspace", divisions, key="matrix_division")
    league_profile_banner(division)
    division_df = working_df[working_df[division_col] == division]
    settled_div, upcoming_div = E.split_played_unplayed(division_df)

    if upcoming_div.empty:
        st.warning("⚠️ No unplayed fixtures detected for this division.")
        return
    if len(settled_div) < E.MIN_SAMPLE_ROWS:
        st.warning(
            f"⚠️ Only {len(settled_div)} settled match(es) in this division - below the "
            f"{E.MIN_SAMPLE_ROWS}-match safety rail. Calculations will fall back to "
            "neutral baselines wherever a team's own sample is too small."
        )

    fixture_labels = [f"{r.home_team} vs {r.away_team}" for r in upcoming_div.itertuples()]
    fixture_choice = st.selectbox("🎯 Select fixture to project", fixture_labels, key="matrix_fixture")
    home_team, away_team = fixture_choice.split(" vs ")

    # --- Core Parameter A: time decay ---
    st.subheader("⏳ Time-Decay Half-Life")
    freeze_decay = st.checkbox("🧊 Freeze Decay (static 45-day window)", key="freeze_decay")
    if freeze_decay:
        half_life = E.FROZEN_HALF_LIFE_DAYS
        st.caption(f"Frozen at {half_life} days.")
    else:
        with st.spinner("Backtesting half-life candidates against real results..."):
            half_life, hl_info = E.optimize_half_life(settled_div)
        st.caption(f"⚙️ Optimal half-life selected via Brier-score backtest: **{half_life} days**.")

    reference_date = settled_div["date"].max() if settled_div["date"].notna().any() else pd.Timestamp.now()

    # --- Core Parameter B: volatility ---
    st.subheader("🎛️ Volatility Auto-Calibrator")
    vol_profile = E.compute_volatility_profile(settled_div)
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("📊 Dispersion Ratio", f"{vol_profile.dispersion_ratio:.3f}")
    vc2.metric("🔄 Squad Turnover Index", f"{vol_profile.squad_turnover_index:.3f}")
    vc3.metric("🌡️ Volatility Dampener", f"{vol_profile.vol_dampener:.3f}" + (" (adjusted)" if vol_profile.adjusted else ""))

    # --- Territory vectors + baseline ---
    baseline = E.compute_league_baseline(settled_div)
    home_profile = E.team_territory_profile(settled_div, home_team, "home", half_life, reference_date)
    away_profile = E.team_territory_profile(settled_div, away_team, "away", half_life, reference_date)

    home_attack = E.attack_strength(home_profile, baseline, "home")
    home_defense = E.defense_strength(home_profile, baseline, "home")
    away_attack = E.attack_strength(away_profile, baseline, "away")
    away_defense = E.defense_strength(away_profile, baseline, "away")
    home_attack_raw, away_attack_raw = home_attack, away_attack

    # --- Momentum banner ---
    home_mom_mult, home_mom_desc = E.team_streak_multiplier(settled_div, home_team)
    away_mom_mult, away_mom_desc = E.team_streak_multiplier(settled_div, away_team)
    st.subheader("🔥 Momentum & Streak Banner")
    mb1, mb2 = st.columns(2)
    mb1.info(f"🏠 **{home_team}**: {home_mom_desc}")
    mb2.info(f"✈️ **{away_team}**: {away_mom_desc}")
    home_attack *= home_mom_mult
    away_attack *= away_mom_mult

    # --- Section 6: tactical multipliers ---
    st.subheader("🎚️ Tactical & Environmental Multipliers")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown(f"**🏠 {home_team} (Host)**")
        home_newly_relegated = st.checkbox("🔽 Newly relegated", key="home_relegated")
        home_relegation_threat = st.checkbox("📉 Live relegation threat", key="home_threat")
        home_striker_injury = st.checkbox("🏥⚽ Key striker/attacker out", key="home_striker_inj")
        home_defender_injury = st.checkbox("🏥🛡️ Key defender out", key="home_defender_inj")
        home_bogey = st.checkbox("🔮 Historical bogey hex (home venue)", key="home_bogey")
        home_new_manager = st.checkbox("🧠 New manager bounce", key="home_manager")
        home_boardroom_crisis = st.checkbox("⚠️ Boardroom crisis", key="home_crisis")
        home_dead_rubber = st.checkbox("🥱 Dead rubber / beach mode", key="home_dead")
        home_cup_distraction = st.checkbox("🏆 Look-ahead cup penalty", key="home_cup")
        host_travel_units = st.selectbox("🚌 Host's own mid-week travel fatigue", [0, 1, 2, 3], key="host_travel")
        home_tactical_setup = st.selectbox(
            "📐 Host tactical setup",
            ["Standard Open Play", "Deep Ultra-Defensive Low-Block", "High-Intensity Counter-Pressing Style"],
            key="home_tactic_setup",
        )
    with tc2:
        st.markdown(f"**✈️ {away_team} (Visitor)**")
        away_newly_relegated = st.checkbox("🔽 Newly relegated", key="away_relegated")
        away_relegation_threat = st.checkbox("📉 Live relegation threat", key="away_threat")
        away_striker_injury = st.checkbox("🏥⚽ Key striker/attacker out", key="away_striker_inj")
        away_defender_injury = st.checkbox("🏥🛡️ Key defender out", key="away_defender_inj")
        away_bogey = st.checkbox("🔮 Historical bogey hex (away venue)", key="away_bogey")
        away_new_manager = st.checkbox("🧠 New manager bounce", key="away_manager")
        away_boardroom_crisis = st.checkbox("⚠️ Boardroom crisis", key="away_crisis")
        away_dead_rubber = st.checkbox("🥱 Dead rubber / beach mode", key="away_dead")
        away_cup_distraction = st.checkbox("🏆 Look-ahead cup penalty", key="away_cup")
        away_travel_units = st.selectbox("🚌 Visitor's mid-week travel fatigue (arriving here)", [0, 1, 2, 3], key="away_travel")
        away_tactical_setup = st.selectbox(
            "📐 Visitor tactical setup",
            ["Standard Open Play", "Deep Ultra-Defensive Low-Block", "High-Intensity Counter-Pressing Style"],
            key="away_tactic_setup",
        )

    st.markdown("**🌍 Universal Match Conditions**")
    uc1, uc2, uc3, uc4 = st.columns(4)
    coastal_shock = uc1.checkbox("🌦️ High-humidity coastal shock (visitor)", key="coastal")
    pre_season = uc2.checkbox("🌱 Pre-season fixture", key="pre_season")
    pitch_surface = uc3.selectbox(
        "🌱 Pitch surface", ["Standard Optimized Turf", "Waterlogged Mud", "Dry Uneven Grass, short and narrow"],
        key="pitch_surface",
    )
    weather = uc4.selectbox(
        "☁️ Weather outlook", ["Clear Sky / Ideal Climate", "Torrential Rain Storm", "Gale-Force Wind Interference"],
        key="weather",
    )
    referee_strictness = st.radio(
        "🟨 Referee Strictness Profile",
        ["Lenient (Flow Enforcer)", "Standard Average", "Hyper-Strict (Card Trigger)"],
        horizontal=True, key="referee_strictness",
    )

    tactics = E.TacticalInputs(
        home_newly_relegated=home_newly_relegated, away_newly_relegated=away_newly_relegated,
        home_relegation_threat=home_relegation_threat, away_relegation_threat=away_relegation_threat,
        home_striker_injury=home_striker_injury, away_striker_injury=away_striker_injury,
        home_defender_injury=home_defender_injury, away_defender_injury=away_defender_injury,
        home_bogey=home_bogey, away_bogey=away_bogey,
        home_new_manager=home_new_manager, away_new_manager=away_new_manager,
        home_boardroom_crisis=home_boardroom_crisis, away_boardroom_crisis=away_boardroom_crisis,
        home_dead_rubber=home_dead_rubber, away_dead_rubber=away_dead_rubber,
        home_travel_fatigue_units=away_travel_units, host_travel_fatigue_units=host_travel_units,
        coastal_shock=coastal_shock,
        home_cup_distraction=home_cup_distraction, away_cup_distraction=away_cup_distraction,
        home_tactical_setup=home_tactical_setup, away_tactical_setup=away_tactical_setup,
        pitch_surface=pitch_surface, weather=weather, referee_strictness=referee_strictness,
        pre_season_fixture=pre_season,
    )
    home_adj, away_adj, vol_adjusted, tactic_log = E.apply_tactical_multipliers(
        home_attack, home_defense, away_attack, away_defense, vol_profile.vol_dampener, tactics,
    )
    if tactic_log:
        with st.expander("📜 Applied multiplier log"):
            for line in tactic_log:
                st.text(line)

    # --- Run both engines ---
    lam_home, lam_away = E.expected_goals(home_adj.attack, away_adj.defense, away_adj.attack, home_adj.defense, baseline)
    st.caption(f"⚽ Model expected goals — {home_team}: **{lam_home:.2f}**, {away_team}: **{lam_away:.2f}**")

    rho = E.fit_rho(settled_div)
    st.caption(f"📐 Calculated Dixon-Coles ρ (fitted from this division's own low-score history): **{rho:.3f}**")
    matrix = E.build_score_matrix(lam_home, lam_away, rho)
    dc_probs = E.market_probs_from_matrix(matrix)

    hg_sim, ag_sim = E.monte_carlo_simulate(lam_home, lam_away, volatility_dampener=vol_adjusted, iterations=E.MC_ITERATIONS)
    mc_probs = E.market_probs_from_simulation(hg_sim, ag_sim)

    # --- Dynamic prediction explanation ---
    st.subheader("🧾 Why This Prediction Was Made")
    explanation = E.generate_prediction_explanation(
        home_team, away_team, half_life, freeze_decay,
        home_attack_raw, away_attack_raw,
        home_mom_mult, home_mom_desc, away_mom_mult, away_mom_desc,
        tactic_log, rho, lam_home, lam_away, dc_probs, mc_probs, vol_adjusted,
    )
    st.markdown(explanation)

    # --- Charts: exact scoreline comparison + goal totals for both engines ---
    st.subheader("📈 Engine Comparison Charts")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Dixon-Coles: most likely exact scorelines")
        size = matrix.shape[0]
        flat = [(f"{h}-{a}", matrix[h, a]) for h in range(min(size, 5)) for a in range(min(size, 5))]
        flat_sorted = sorted(flat, key=lambda x: -x[1])[:8]
        score_df = pd.DataFrame(flat_sorted, columns=["Scoreline", "Probability"]).set_index("Scoreline")
        st.bar_chart(score_df)
    with ch2:
        st.caption("Monte Carlo: simulated total-goals distribution")
        total_goals_sim = hg_sim + ag_sim
        totals_counts = pd.Series(total_goals_sim).value_counts().sort_index()
        totals_counts.index = totals_counts.index.astype(str)
        st.bar_chart(totals_counts)

    engine_compare_df = pd.DataFrame({
        "Dixon-Coles %": [dc_probs["Home Win"] * 100, dc_probs["Draw"] * 100, dc_probs["Away Win"] * 100],
        "Monte Carlo %": [mc_probs["Home Win"] * 100, mc_probs["Draw"] * 100, mc_probs["Away Win"] * 100],
    }, index=["Home Win", "Draw", "Away Win"])
    st.caption("Home / Draw / Away probability - both engines side by side")
    st.bar_chart(engine_compare_df)

    # --- Section 8: valuation sheet ---
    st.subheader("📋 22-Market Options Valuation Sheet")
    st.caption("Edit the Bookmaker Odds for any market you want to check - everything else recalculates live.")

    edited_odds = {}
    odds_cols = st.columns(2)
    for i, market in enumerate(E.MARKET_LIST):
        col = odds_cols[i % 2]
        edited_odds[market] = col.number_input(
            f"{market} odds", min_value=1.01, max_value=100.0,
            value=float(st.session_state.bookmaker_odds.get(market, 2.00)),
            step=0.01, key=f"odds_{market}",
        )
    st.session_state.bookmaker_odds = edited_odds

    sheet_rows = E.build_valuation_sheet(dc_probs, mc_probs, edited_odds, vol_adjusted)
    sheet_df = pd.DataFrame([
        {
            "Market": r.market,
            "Bookmaker Odds": r.bookmaker_odds,
            "Dixon-Coles %": round(r.dc_prob * 100, 2),
            "Monte Carlo %": round(r.mc_prob * 100, 2),
            "Convergence %": round(r.convergence * 100, 1),
            "Fair Odds": round(r.fair_odds, 2) if math.isfinite(r.fair_odds) else None,
            "EV Edge %": round(r.ev * 100, 2),
            "Volatility Tier": r.volatility_tier,
            "Verdict": r.verdict,
            "Recommended Action": r.recommended_action,
        }
        for r in sheet_rows
    ])
    st.dataframe(sheet_df, use_container_width=True, hide_index=True, height=560)

    # --- Section 9: parlay & Kelly builder ---
    render_parlay_builder(sheet_rows, home_team, away_team, explanation)


def render_parlay_builder(sheet_rows, home_team, away_team, explanation_text):
    st.subheader("🎟️ Sisonke Multi-Leg Parlay & Kelly Slip Builder")
    row_by_market = {r.market: r for r in sheet_rows}
    chosen_markets = st.multiselect(
        "Pick 2 or more qualifying value lines", list(row_by_market.keys()), key="parlay_legs"
    )
    if len(chosen_markets) < 2:
        st.caption("Select at least 2 legs to build a parlay slip.")
        return

    legs = [row_by_market[m] for m in chosen_markets]
    combined_odds, combined_prob = E.combine_parlay_legs(legs)
    combined_ev = E.expected_value(combined_prob, combined_odds)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("💰 Combined Odds", f"{combined_odds:.2f}")
    pc2.metric("🎯 Joint Model Probability", f"{combined_prob * 100:.2f}%")
    pc3.metric("📈 Combined EV", f"{combined_ev * 100:.2f}%")

    kelly_mult = st.slider("🎚️ Fractional Kelly", 0.05, 1.00, 0.25, step=0.05, key="kelly_slider")
    bankroll = st.number_input("💵 Matchday bankroll (R)", min_value=0.0, value=1000.0, step=50.0, key="bankroll")

    if combined_ev <= 0:
        st.error("🚫 Combined expected value is negative - staking locked out for safety.")
        stake = 0.0
    else:
        kelly_frac = E.kelly_stake_fraction(combined_prob, combined_odds, kelly_mult)
        raw_stake = kelly_frac * bankroll
        stake = E.round_to_nearest(raw_stake, 10)
        st.success(f"✅ Suggested stake: **R{stake:.0f}** (Kelly fraction: {kelly_frac * 100:.2f}%, rounded to nearest R10)")

    ticket_lines = [
        "SISONKE MULTI-LEG PARLAY SLIP", "=" * 40,
        f"Fixture context: {home_team} vs {away_team}", "", "LEGS:",
    ]
    for r in legs:
        ticket_lines.append(f"  - {r.market} @ {r.bookmaker_odds:.2f} (model {max(r.dc_prob, r.mc_prob) * 100:.1f}%, EV {r.ev * 100:.1f}%)")
    ticket_lines += [
        "", f"Combined Odds: {combined_odds:.2f}", f"Joint Model Probability: {combined_prob * 100:.2f}%",
        f"Combined EV: {combined_ev * 100:.2f}%", f"Kelly Fraction Used: {kelly_mult:.2f}", f"Suggested Stake: R{stake:.0f}",
    ]
    ticket_text = "\n".join(ticket_lines)
    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇️ Download Ticket (.txt)", data=ticket_text, file_name="sisonke_bet_slip.txt", mime="text/plain")

    if dl2.button("📱 Send to Telegram", key="send_telegram_btn"):
        token = st.session_state.get("tg_token", "")
        chat_id = st.session_state.get("tg_chat_id", "")
        message = f"{explanation_text}\n\n{ticket_text}"
        ok, msg = E.send_telegram_message(token, chat_id, message)
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")


# ---------------------------------------------------------------------------
# Live Standings Ledger tab
# ---------------------------------------------------------------------------
def render_standings_ledger():
    division = st.selectbox("🏆 League workspace", divisions, key="standings_division")
    division_df = working_df[working_df[division_col] == division]
    settled_div, upcoming_div = E.split_played_unplayed(division_df)

    st.subheader("📊 Deserved Points Table (xPts)")
    st.caption("Sorted by real (actual) points, highest first.")
    if settled_div.empty:
        st.caption("No settled matches yet for this division.")
    else:
        xpts_table = E.compute_xpts_table(settled_div)
        display_cols = ["team", "played", "wins", "draws", "losses", "goal_difference",
                         "actual_points", "expected_points", "points_difference"]
        st.dataframe(xpts_table[display_cols], use_container_width=True, hide_index=True)

    st.subheader("🔮 10,000-Run Season Forecast")
    if upcoming_div.empty:
        st.caption("No remaining fixtures to simulate - season looks complete in this dataset.")
        return

    all_teams_here = sorted(pd.unique(division_df[["home_team", "away_team"]].values.ravel("K")))
    all_teams_here = [t for t in all_teams_here if isinstance(t, str)]
    with st.expander("💰 Title odds input (optional - adds an Edge column)"):
        title_odds_input = {}
        odd_cols = st.columns(3)
        for i, t in enumerate(all_teams_here):
            val = odd_cols[i % 3].number_input(f"{t} title odds", min_value=0.0, value=0.0, step=1.0, key=f"title_odds_{t}")
            if val > 0:
                title_odds_input[t] = val
        st.session_state.title_odds = title_odds_input

    if st.button("▶️ Run 10,000-iteration season simulation", key="run_season_sim"):
        with st.spinner("Simulating 10,000 seasons..."):
            forecast = E.simulate_season(
                settled_div, upcoming_div, iterations=E.MC_ITERATIONS, title_odds=st.session_state.title_odds
            )
        st.session_state["season_forecast"] = forecast

    if "season_forecast" in st.session_state:
        forecast = st.session_state["season_forecast"]
        core_cols = ["team", "current_points", "title_win_pct", "relegation_risk_pct", "relegation_flag"]
        if "title_odds" in forecast.columns:
            core_cols += ["title_odds", "title_edge_pct"]
        st.caption("Sorted by current real points, highest first.")
        st.dataframe(forecast[core_cols], use_container_width=True, hide_index=True)

        with st.expander("📍 Detailed finishing-position distribution (% chance of each exact position)"):
            pos_cols = [c for c in forecast.columns if c.startswith("finish_pos_")]
            pos_display = forecast[["team"] + pos_cols].copy()
            pos_display.columns = ["team"] + [f"P{c.split('_')[-1]}" for c in pos_cols]
            st.dataframe(pos_display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Performance Backtester tab
# ---------------------------------------------------------------------------
def render_backtester():
    division = st.selectbox("🏆 League workspace", divisions, key="backtest_division")
    division_df = working_df[working_df[division_col] == division]
    settled_div, _ = E.split_played_unplayed(division_df)
    if settled_div.empty:
        st.caption("No settled matches for this division.")
        return

    st.subheader("🎛️ Backtest Controls")
    bc1, bc2 = st.columns(2)
    manual_override_pct = bc1.slider(
        "🎚️ Manual Override (shift Home Win probability, percentage points)",
        -20.0, 20.0, 0.0, step=1.0, key="manual_override_slider",
        help="Nudges every backtest prediction's Home Win probability by this many points (rebalancing Draw/Away proportionally) - a sensitivity check, not a permanent model change.",
    )
    accuracy_floor = bc2.slider(
        "📏 Accuracy Floor (%) - only count high-confidence picks", 0, 100, 0, step=5, key="accuracy_floor_slider",
        help="Filters the accuracy metric to only predictions where the model's top pick exceeded this probability - shows how good the model is when it's genuinely confident.",
    )

    half_life_choice = st.radio(
        "Half-life used for this backtest", ["Auto-optimised", "Frozen 45-day", "Effectively no decay (raw/unweighted)"],
        horizontal=True, key="backtest_hl_choice",
    )
    if half_life_choice == "Frozen 45-day":
        hl_for_backtest = E.FROZEN_HALF_LIFE_DAYS
    elif half_life_choice == "Effectively no decay (raw/unweighted)":
        hl_for_backtest = 100_000.0
    else:
        hl_for_backtest, _ = E.optimize_half_life(settled_div)

    with st.spinner(f"Running a walk-forward backtest across all {len(settled_div)} settled matches..."):
        backtest_df = E.walk_forward_backtest(settled_div, half_life_days=hl_for_backtest)

    if backtest_df.empty:
        st.warning("Not enough history yet to backtest (need several settled matches before the first prediction can be made).")
        return

    if manual_override_pct != 0:
        adjusted = backtest_df.copy()
        for i, row in adjusted.iterrows():
            ph, pd_, pa = E.apply_manual_override(row["p_home"], row["p_draw"], row["p_away"], manual_override_pct)
            adjusted.at[i, "p_home"], adjusted.at[i, "p_draw"], adjusted.at[i, "p_away"] = ph, pd_, pa
            adjusted.at[i, "predicted_pick"] = max([("H", ph), ("D", pd_), ("A", pa)], key=lambda kv: kv[1])[0]
            adjusted.at[i, "correct"] = adjusted.at[i, "predicted_pick"] == row["actual"]
        backtest_df = adjusted

    bss = E.brier_skill_score(backtest_df)
    accuracy = E.backtest_accuracy_pct(backtest_df)
    confident_mask = backtest_df[["p_home", "p_draw", "p_away"]].max(axis=1) * 100 >= accuracy_floor
    filtered_df = backtest_df[confident_mask]
    filtered_accuracy = E.backtest_accuracy_pct(filtered_df) if not filtered_df.empty else float("nan")

    st.subheader("📐 Model Skill Metrics (whole dataset)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Brier Skill Score", f"{bss:.3f}" if not math.isnan(bss) else "n/a", help=">0 beats naive guessing, <0 is worse than it")
    m2.metric("✅ Accuracy (all picks)", f"{accuracy:.1f}%" if not math.isnan(accuracy) else "n/a")
    m3.metric(f"🔎 Accuracy (≥{accuracy_floor}% confidence)", f"{filtered_accuracy:.1f}%" if not math.isnan(filtered_accuracy) else "n/a", help=f"{len(filtered_df)}/{len(backtest_df)} predictions met the floor")
    m4.metric("📊 Matches backtested", len(backtest_df))

    st.subheader("📈 Weighted vs Raw Comparison")
    raw_backtest = E.walk_forward_backtest(settled_div, half_life_days=100_000.0)
    weighted_backtest = E.walk_forward_backtest(settled_div, half_life_days=E.FROZEN_HALF_LIFE_DAYS)
    compare_df = pd.DataFrame({
        "Accuracy %": [E.backtest_accuracy_pct(raw_backtest), E.backtest_accuracy_pct(weighted_backtest)],
        "Brier Skill Score": [E.brier_skill_score(raw_backtest), E.brier_skill_score(weighted_backtest)],
    }, index=["Raw (no decay)", "Weighted (45-day half-life)"])
    st.bar_chart(compare_df[["Accuracy %"]])
    st.dataframe(compare_df, use_container_width=True)

    st.subheader("📅 Full Backtest Log (whole dataset)")
    st.dataframe(
        backtest_df[["date", "home_team", "away_team", "home_goals", "away_goals", "goal_difference",
                     "p_home", "p_draw", "p_away", "actual", "predicted_pick", "correct"]],
        use_container_width=True, hide_index=True, height=400,
    )


# ---------------------------------------------------------------------------
# Full Database View tab
# ---------------------------------------------------------------------------
def render_full_database():
    st.subheader("📑 Full Database View")
    st.caption(f"{len(st.session_state.raw_db)} row(s), as originally uploaded (team-name casing already normalised).")

    display_df = working_df.copy()
    if "home_goals" in display_df.columns and "away_goals" in display_df.columns:
        display_df["goal_difference"] = display_df["home_goals"] - display_df["away_goals"]

    if all(c in display_df.columns for c in ["home_big_chances", "away_big_chances", "home_shots_on_target", "away_shots_on_target", "home_box_touches", "away_box_touches"]):
        display_df["home_implied_xg"] = (
            0.55 * display_df["home_big_chances"].fillna(0) * 0.36
            + 0.35 * display_df["home_shots_on_target"].fillna(0) * 0.11
            + 0.10 * display_df["home_box_touches"].fillna(0) * 0.015
        ).round(2)
        display_df["away_implied_xg"] = (
            0.55 * display_df["away_big_chances"].fillna(0) * 0.36
            + 0.35 * display_df["away_shots_on_target"].fillna(0) * 0.11
            + 0.10 * display_df["away_box_touches"].fillna(0) * 0.015
        ).round(2)
        st.caption("ℹ️ `*_implied_xg` is a lightweight per-match estimate from THAT match's own registered stats (not the full team-strength model) - for a quick eyeball check, not a substitute for the Projections Matrix.")

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)


# ---------------------------------------------------------------------------
# Router (Section 1)
# ---------------------------------------------------------------------------
if active_tab == "📁 Research & Sentiment Tracker":
    render_sentiment_tracker()
else:
    st.title("⚽ SISONKE FOOTBALL HUB")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Active Projections Matrix", "📊 Live Standings Ledger",
        "📅 Performance Backtester", "📑 Full Database View",
    ])
    with tab1:
        render_projections_matrix()
    with tab2:
        render_standings_ledger()
    with tab3:
        render_backtester()
    with tab4:
        render_full_database()
