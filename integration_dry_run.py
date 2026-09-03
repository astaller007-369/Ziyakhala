"""
integration_dry_run.py
=========================

Not part of the shipped app - a one-off script to exercise the FULL
pipeline end to end (data load -> division filter -> fixture pick ->
half-life -> volatility -> territory vectors -> momentum -> tactical
multipliers -> both engines -> valuation sheet -> parlay -> xPts ->
season simulation -> gold mine hint) against a synthetic but realistic
dataset, to catch integration issues the per-function unit tests can't.
"""
import numpy as np
import pandas as pd

import sisonke_engine as E

rng = np.random.default_rng(7)

TEAMS = [
    "Sisonke City", "Umlazi United", "Durban Rangers", "Cape Coast FC",
    "Joburg Athletic", "Bushveld Bulls", "Highveld Hawks", "Karoo Kings",
]

rows = []
start = pd.Timestamp("2025-08-01")
match_day = 0
# Round-robin double leg -> plenty of settled matches per team (well above
# the 5-match safety rail), each with realistic-looking stat columns.
for leg in range(2):
    for i, home in enumerate(TEAMS):
        for j, away in enumerate(TEAMS):
            if home == away:
                continue
            if leg == 1 and i > j:
                continue  # avoid literally doubling every fixture twice in leg 1 pass, keep it a real double round-robin (home/away swap)
            match_day += 1
            date = start + pd.Timedelta(days=match_day * 3)
            home_goals = rng.poisson(1.4)
            away_goals = rng.poisson(1.1)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "league_country": "Premier Division (South Africa)",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_shots_on_target": rng.poisson(5) + 1,
                "away_shots_on_target": rng.poisson(4) + 1,
                "home_big_chances": rng.poisson(2),
                "away_big_chances": rng.poisson(1.5),
                "home_box_touches": rng.poisson(22) + 10,
                "away_box_touches": rng.poisson(18) + 10,
            })

# Add a handful of genuinely unplayed fixtures (blank goals + a comma-junk one)
rows.append({
    "date": "2026-09-01", "league_country": "Premier Division (South Africa)",
    "home_team": "Sisonke City", "away_team": "Karoo Kings",
    "home_goals": "", "away_goals": "",
    "home_shots_on_target": None, "away_shots_on_target": None,
    "home_big_chances": None, "away_big_chances": None,
    "home_box_touches": None, "away_box_touches": None,
})
rows.append({
    "date": "2026-09-02", "league_country": "Premier Division (South Africa)",
    "home_team": "Umlazi United", "away_team": "Bushveld Bulls",
    "home_goals": "2,5", "away_goals": "1,8",  # odds-style comma junk some spreadsheets paste in
    "home_shots_on_target": None, "away_shots_on_target": None,
    "home_big_chances": None, "away_big_chances": None,
    "home_box_touches": None, "away_box_touches": None,
})
# A brand-new team with ZERO matches, to specifically exercise the
# crash-proof safety shield in simulate_season and the 5-match safety
# rail elsewhere.
rows.append({
    "date": "2026-09-03", "league_country": "Premier Division (South Africa)",
    "home_team": "Brand New FC", "away_team": "Sisonke City",
    "home_goals": "", "away_goals": "",
    "home_shots_on_target": None, "away_shots_on_target": None,
    "home_big_chances": None, "away_big_chances": None,
    "home_box_touches": None, "away_box_touches": None,
})

raw_df = pd.DataFrame(rows)
print(f"Synthetic dataset: {len(raw_df)} total rows")

# ---- exactly mirror sisonke_app.py's loading/coercion path ----
df = E.standardise_columns(raw_df)
numeric_cols = [
    "home_goals", "away_goals", "home_shots_on_target", "away_shots_on_target",
    "home_big_chances", "away_big_chances", "home_box_touches", "away_box_touches",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = E.parse_dates(df, "date")

division_col = E.find_division_column(df)
assert division_col == "league_country", f"expected league_country, got {division_col}"

divisions = sorted(df[division_col].dropna().unique().tolist())
assert divisions == ["Premier Division (South Africa)"]
division = divisions[0]
division_df = df[df[division_col] == division]

settled, upcoming = E.split_played_unplayed(division_df)
print(f"Settled: {len(settled)}, Upcoming (incl. comma-junk + brand-new team): {len(upcoming)}")
assert len(upcoming) == 3, f"expected exactly 3 unplayed rows, got {len(upcoming)}"
assert len(settled) == len(division_df) - 3

fixture_labels = [f"{r.home_team} vs {r.away_team}" for r in upcoming.itertuples()]
assert "Sisonke City vs Karoo Kings" in fixture_labels
assert "Brand New FC vs Sisonke City" in fixture_labels
print("Fixtures available to pick:", fixture_labels)

home_team, away_team = "Sisonke City", "Karoo Kings"

# Core Parameter A: half-life
half_life, hl_info = E.optimize_half_life(settled)
assert 15 <= half_life <= 180
print(f"Optimised half-life: {half_life} days")

reference_date = settled["date"].max()

# Core Parameter B: volatility
vol_profile = E.compute_volatility_profile(settled)
assert vol_profile.dispersion_ratio > 0
print(f"Dispersion ratio: {vol_profile.dispersion_ratio:.3f}, vol dampener: {vol_profile.vol_dampener:.3f}")

baseline = E.compute_league_baseline(settled)
home_profile = E.team_territory_profile(settled, home_team, "home", half_life, reference_date)
away_profile = E.team_territory_profile(settled, away_team, "away", half_life, reference_date)
assert home_profile is not None and away_profile is not None

home_attack = E.attack_strength(home_profile, baseline, "home")
home_defense = E.defense_strength(home_profile, baseline, "home")
away_attack = E.attack_strength(away_profile, baseline, "away")
away_defense = E.defense_strength(away_profile, baseline, "away")
for v in (home_attack, home_defense, away_attack, away_defense):
    assert v > 0 and not np.isnan(v)
print(f"Home attack/def: {home_attack:.3f}/{home_defense:.3f} | Away attack/def: {away_attack:.3f}/{away_defense:.3f}")

home_mom_mult, home_mom_desc = E.team_streak_multiplier(settled, home_team)
away_mom_mult, away_mom_desc = E.team_streak_multiplier(settled, away_team)
assert 0.85 <= home_mom_mult <= 1.15  # sane range around the 0.88-1.12 spec caps
print(f"Momentum: {home_team} {home_mom_desc} ({home_mom_mult:.3f}) | {away_team} {away_mom_desc} ({away_mom_mult:.3f})")
home_attack *= home_mom_mult
away_attack *= away_mom_mult

# Section 6: tactical multipliers - flip on a representative subset
tactics = E.TacticalInputs(
    home_new_manager=True,          # +10% home attack
    away_relegation_threat=True,    # +8% away defense grit
    home_travel_fatigue_units=0,
    away_tactical_setup="Deep Ultra-Defensive Low-Block",  # cuts away attack hard + squeezes volatility
)
home_adj, away_adj, vol_adjusted, tactic_log = E.apply_tactical_multipliers(
    home_attack, home_defense, away_attack, away_defense, vol_profile.vol_dampener, tactics,
)
print("Tactical log:", tactic_log)
assert home_adj.attack > home_attack, "new manager bounce should have increased home attack"
assert away_adj.attack < away_attack, "low block should have cut away attack"

lam_home, lam_away = E.expected_goals(home_adj.attack, away_adj.defense, away_adj.attack, home_adj.defense, baseline)
assert lam_home > 0 and lam_away > 0 and not np.isnan(lam_home) and not np.isnan(lam_away)
print(f"Expected goals: home {lam_home:.3f}, away {lam_away:.3f}")

rho = E.fit_rho(settled)
assert -0.25 <= rho <= 0.25
matrix = E.build_score_matrix(lam_home, lam_away, rho)
assert abs(matrix.sum() - 1.0) < 1e-6
dc_probs = E.market_probs_from_matrix(matrix)
assert len(dc_probs) == 22, f"expected 22 markets, got {len(dc_probs)}"

hg_sim, ag_sim = E.monte_carlo_simulate(lam_home, lam_away, volatility_dampener=vol_adjusted, iterations=10000)
mc_probs = E.market_probs_from_simulation(hg_sim, ag_sim)
assert len(mc_probs) == 22

print(f"DC Home Win: {dc_probs['Home Win']*100:.1f}% | MC Home Win: {mc_probs['Home Win']*100:.1f}%")
assert abs(dc_probs["Home Win"] - mc_probs["Home Win"]) < 0.15, "the two engines should roughly agree for a real, non-degenerate matchup"

edited_odds = {m: 2.20 for m in E.MARKET_LIST}
sheet_rows = E.build_valuation_sheet(dc_probs, mc_probs, edited_odds, vol_adjusted)
assert len(sheet_rows) == 22
assert all(hasattr(r, "recommended_action") and r.recommended_action for r in sheet_rows)
assert not any("home advantage" in (r.recommended_action or "").lower() for r in sheet_rows)
print("Sample valuation row:", sheet_rows[0])

row_by_market = {r.market: r for r in sheet_rows}
legs = [row_by_market[m] for m in list(row_by_market.keys())[:2]]
combined_odds, combined_prob = E.combine_parlay_legs(legs)
assert combined_odds == legs[0].bookmaker_odds * legs[1].bookmaker_odds
combined_ev = E.expected_value(combined_prob, combined_odds)
print(f"Parlay combined odds {combined_odds:.2f}, joint prob {combined_prob*100:.2f}%, EV {combined_ev*100:.2f}%")

xpts_table = E.compute_xpts_table(settled)
assert len(xpts_table) == len(TEAMS)
print(xpts_table.head(3))

forecast = E.simulate_season(settled, upcoming, iterations=2000)  # smaller N for a fast dry run
assert "Brand New FC" in forecast["team"].values or "Brand New FC" in list(forecast.iloc[:, 0])
print("Season forecast includes Brand New FC (safety shield engaged correctly):")
print(forecast[forecast.iloc[:, 0] == "Brand New FC"] if "team" not in forecast.columns else forecast[forecast["team"] == "Brand New FC"])

hint = E.gold_mine_hint(division)
assert "South Africa" in hint or "Premier Division" in hint
print("Gold mine hint:", hint)

print()
print("=" * 60)
print("FULL INTEGRATION DRY RUN PASSED - no exceptions, all assertions held.")
