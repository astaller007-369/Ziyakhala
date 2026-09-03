"""
bias_stress_test.py
=====================

Directly tests the exact failure mode reported: "home team always ~40%,
away/draw ~28% with almost no variance, even when the away team is
clearly stronger." Builds several datasets with DELIBERATELY lopsided
team strength (not random noise) and checks whether the model's output
actually moves the way real team strength should move it.
"""
import numpy as np
import pandas as pd

import sisonke_engine as E

rng = np.random.default_rng(1)


def build_league(home_strong_team, away_strong_team, n_matches_each=10):
    """Builds a small synthetic league where one named team is
    DELIBERATELY much stronger (more big chances/SOT/box touches created,
    fewer allowed) than a named weak team, regardless of which venue role
    they're in for the fixture we'll actually test."""
    teams = ["Strong FC", "Weak FC", "Filler A", "Filler B", "Filler C", "Filler D"]
    rows = []
    start = pd.Timestamp("2025-08-01")
    day = 0

    def stats_for(team, is_for_side):
        # "Strong FC" creates a lot, allows little. "Weak FC" is the
        # mirror opposite. Fillers are average, just to give the league
        # baseline something sane to compute from.
        if team == "Strong FC":
            bc, sot, box = (3.2, 7.5, 30) if is_for_side else (0.6, 2.0, 10)
        elif team == "Weak FC":
            bc, sot, box = (0.6, 2.0, 10) if is_for_side else (3.2, 7.5, 30)
        else:
            bc, sot, box = (1.4, 4.0, 18) if is_for_side else (1.4, 4.0, 18)
        return bc, sot, box

    for _ in range(n_matches_each):
        for home, away in [(t1, t2) for t1 in teams for t2 in teams if t1 != t2]:
            day += 1
            date = start + pd.Timedelta(days=day)
            hbc, hsot, hbox = stats_for(home, True)
            abc, asot, abox = stats_for(away, True)
            hbc_ag, hsot_ag, hbox_ag = stats_for(home, False)
            abc_ag, asot_ag, abox_ag = stats_for(away, False)
            # blend "for" and opponent's "against" tendency with a little
            # noise so it isn't a perfectly noiseless synthetic signal
            home_big_chances = max(0, rng.normal((hbc + abc_ag) / 2, 0.3))
            away_big_chances = max(0, rng.normal((abc + hbc_ag) / 2, 0.3))
            home_sot = max(0, rng.normal((hsot + asot_ag) / 2, 0.5))
            away_sot = max(0, rng.normal((asot + hsot_ag) / 2, 0.5))
            home_box = max(0, rng.normal((hbox + abox_ag) / 2, 1.5))
            away_box = max(0, rng.normal((abox + hbox_ag) / 2, 1.5))
            home_goals = rng.poisson(max(0.1, home_big_chances * 0.35))
            away_goals = rng.poisson(max(0.1, away_big_chances * 0.35))
            rows.append({
                "date": date.strftime("%Y-%m-%d"), "league_country": "Test League",
                "home_team": home, "away_team": away,
                "home_goals": home_goals, "away_goals": away_goals,
                "home_shots_on_target": home_sot, "away_shots_on_target": away_sot,
                "home_big_chances": home_big_chances, "away_big_chances": away_big_chances,
                "home_box_touches": home_box, "away_box_touches": away_box,
            })
    return pd.DataFrame(rows)


def project(settled_df, home_team, away_team):
    half_life, _ = E.optimize_half_life(settled_df)
    reference_date = settled_df["date"].max()
    baseline = E.compute_league_baseline(settled_df)
    home_profile = E.team_territory_profile(settled_df, home_team, "home", half_life, reference_date)
    away_profile = E.team_territory_profile(settled_df, away_team, "away", half_life, reference_date)
    ha = E.attack_strength(home_profile, baseline, "home")
    hd = E.defense_strength(home_profile, baseline, "home")
    aa = E.attack_strength(away_profile, baseline, "away")
    ad = E.defense_strength(away_profile, baseline, "away")
    lam_h, lam_a = E.expected_goals(ha, ad, aa, hd, baseline)
    rho = E.fit_rho(settled_df)
    matrix = E.build_score_matrix(lam_h, lam_a, rho)
    probs = E.market_probs_from_matrix(matrix)
    return probs["Home Win"], probs["Draw"], probs["Away Win"], lam_h, lam_a


raw = build_league("Strong FC", "Weak FC")
df = E.standardise_columns(raw)
for c in ["home_goals", "away_goals", "home_shots_on_target", "away_shots_on_target",
          "home_big_chances", "away_big_chances", "home_box_touches", "away_box_touches"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = E.parse_dates(df, "date")
settled, _ = E.split_played_unplayed(df)

print("=" * 70)
print("TEST 1: Weak home team hosts Strong away team (the exact reported bug)")
print("=" * 70)
h, d, a, lam_h, lam_a = project(settled, "Weak FC", "Strong FC")
print(f"Expected goals: home(Weak FC)={lam_h:.2f}, away(Strong FC)={lam_a:.2f}")
print(f"Home Win: {h*100:.1f}% | Draw: {d*100:.1f}% | Away Win: {a*100:.1f}%")
assert a > h, "BUG: model still favors the home team even though the away team is clearly stronger."
print("PASS: model correctly favors the away (stronger) team despite home venue.")

print()
print("=" * 70)
print("TEST 2: Strong home team hosts Weak away team (sanity check, opposite direction)")
print("=" * 70)
h2, d2, a2, lam_h2, lam_a2 = project(settled, "Strong FC", "Weak FC")
print(f"Expected goals: home(Strong FC)={lam_h2:.2f}, away(Weak FC)={lam_a2:.2f}")
print(f"Home Win: {h2*100:.1f}% | Draw: {d2*100:.1f}% | Away Win: {a2*100:.1f}%")
assert h2 > a2, "BUG: model failed to favor even a genuinely stronger home team."
print("PASS: model correctly favors the home team here, because it's ALSO the stronger team.")

print()
print("=" * 70)
print("TEST 3: Variance check - do probabilities actually spread out across")
print("several different matchups, or cluster tightly like the reported bug?")
print("=" * 70)
matchups = [
    ("Strong FC", "Weak FC"), ("Weak FC", "Strong FC"),
    ("Filler A", "Filler B"), ("Filler B", "Filler A"),
    ("Strong FC", "Filler C"), ("Filler C", "Strong FC"),
    ("Weak FC", "Filler D"), ("Filler D", "Weak FC"),
]
home_win_probs = []
for home, away in matchups:
    h_, d_, a_, lh, la = project(settled, home, away)
    home_win_probs.append(h_)
    print(f"  {home:10s} (H) vs {away:10s} (A) -> Home {h_*100:5.1f}% | Draw {d_*100:5.1f}% | Away {a_*100:5.1f}%  (λh={lh:.2f}, λa={la:.2f})")

spread = max(home_win_probs) - min(home_win_probs)
print(f"\nHome Win probability range across matchups: {min(home_win_probs)*100:.1f}% to {max(home_win_probs)*100:.1f}% (spread: {spread*100:.1f} points)")

# Hard assertions, not just printed PASS/FAIL text - this file is meant to
# run as part of the regular test suite, so a real regression actually
# fails the run (non-zero exit / raised AssertionError) instead of just
# printing something a person has to notice and read.
assert a > h, f"BUG: weaker home team ({h*100:.1f}%) still beat a clearly stronger away team ({a*100:.1f}%)"
assert h2 > a2, f"BUG: model failed to favor a genuinely stronger home team ({h2*100:.1f}% vs {a2*100:.1f}%)"
assert spread > 0.30, f"BUG: only a {spread*100:.1f}-point spread across very different matchups - probabilities are flattening toward a fixed split"

print("PASS: wide spread - the model is clearly responding to actual team strength, not flattening to a fixed number.")
print()
print("=" * 70)
print("OVERALL: NO SIGN OF THE REPORTED HOME BIAS IN THE CURRENT ENGINE.")
