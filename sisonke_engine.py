"""
sisonke_engine.py
==================

All the math for the Sisonke Football Predictive Terminal, kept
completely separate from the Streamlit UI so it can be tested directly
with plain Python (no Streamlit needed to verify the numbers are right).

DESIGN DECISIONS WORTH KNOWING ABOUT (documented up front because this
is a real-money-adjacent tool and every assumption should be visible,
not buried):

1. HOME ADVANTAGE IS NOT DOUBLE-COUNTED. Host stats are computed ONLY
   from a team's home matches, and visitor stats ONLY from their away
   matches - so whatever real home boost a team gets is already baked
   into their host numbers. No separate home-advantage multiplier is
   applied anywhere in this file.

2. TEAM STRENGTH NEVER COMES FROM A TEAM'S OWN RAW GOALS. The
   attack/defense STRENGTH RATIOS that differentiate one team from
   another are built entirely from territory metrics (big chances,
   shots on target, box touches) - never from a team's own win/loss
   record or their own average goals, which is exactly the "luck"
   signal the spec says to avoid. The one place raw goals appear at all
   is as a single LEAGUE-WIDE average goals figure used as a shared
   baseline unit (the same number for every team in that league) to
   convert relative territory strength into an actual expected-goals
   scale for the Poisson math - since "big chances" has no absolute
   goals unit on its own. That's a scale anchor, not a team signal.

3. PROBABILITIES ARE ALWAYS COMPUTED, NEVER FIXED. Every tactical
   multiplier in Section 6 only ever adjusts the INPUT attack/defense
   rates that feed the Dixon-Coles and Monte Carlo engines. The actual
   market probabilities always come out of real Poisson math or a real
   10,000-run simulation - a slider never directly sets a probability
   or an EV number by formula shortcut.

4. RHO (the Dixon-Coles low-score correlation parameter) is fitted from
   the league's own historical low-score frequencies, not a fixed
   constant - see fit_rho().

5. The half-life for time-decay weighting is chosen by actually
   backtesting candidate half-lives against real past results with a
   Brier score (see optimize_half_life()) - not asserted as a fixed
   number, unless you tick "Freeze Decay" for a fixed 45-day window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson

try:
    import requests as _requests
except ImportError:  # Telegram sending degrades gracefully if requests isn't installed
    _requests = None

MIN_SAMPLE_ROWS = 5          # the "5-match sample safety rail"
FROZEN_HALF_LIFE_DAYS = 45   # fallback when "Freeze Decay" is ticked
HALF_LIFE_CANDIDATES = list(range(15, 181, 15))  # 15, 30, ..., 180
GOAL_CAP = 10                # max goals per side considered in the Poisson grid
MC_ITERATIONS = 10_000
DISPERSION_ADJUST_TRIGGER = 0.85
DISPERSION_ADJUST_FACTOR = 1.15

TERRITORY_STATS = ["big_chances", "shots_on_target", "box_touches"]

# ---------------------------------------------------------------------------
# Section 3: Column standardisation, division/fixture parsing
# ---------------------------------------------------------------------------

REQUIRED_BASE_COLUMNS = [
    "date", "home_team", "away_team", "home_goals", "away_goals",
    "home_shots_on_target", "away_shots_on_target",
    "home_big_chances", "away_big_chances",
    "home_box_touches", "away_box_touches",
]
DIVISION_COLUMN_CANDIDATES = ["league_country", "league", "competition"]


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace, lowercases, and replaces spaces with
    underscores in every column name (e.g. 'Home Box Touches' becomes
    'home_box_touches')."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def find_division_column(df: pd.DataFrame) -> str | None:
    for candidate in DIVISION_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


# Same flexible-detection pattern as the division column, for the same
# reason: different CSV exports call this column different things, and
# blindly assuming "date" exists crashes anything downstream that reads
# it (half-life decay, backtesting, the fixture picker) with a cryptic
# AttributeError rather than a clear, actionable message.
DATE_COLUMN_CANDIDATES = [
    "date", "match_date", "fixture_date", "game_date", "kickoff",
    "kickoff_date", "kickoff_time", "date_time", "match_datetime",
    "match_time", "played_on", "utc_date",
]


def find_date_column(df: pd.DataFrame) -> str | None:
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


# Keyword fragments that flag a competition as a CUP/TOURNAMENT rather
# than a standard home-and-away league - this model's whole points/table
# framework (xPts, season simulation, "played N times") assumes a
# standard league fixture list, which doesn't hold for single/double-leg
# knockout cup football (extra time, penalties, one-off ties, no table).
CUP_TOURNAMENT_KEYWORDS = [
    "cup", "trophy", "shield", "playoff", "play-off", "knockout",
    "copa", "coupe", "pokal", "taça", "taca", "supercup", "super cup",
    "champions league", "europa league", "conference league", "libertadores",
    "sudamericana", "afcon", "world cup", "euros", "european championship",
    "nations league", "friendly", "friendlies", "qualifier", "qualifying",
]


def is_cup_or_tournament(division_text: str) -> bool:
    """Keyword-based flag, not a guarantee - a league that happens to
    have 'Cup' in its sponsor name would still need a manual override,
    but this catches the overwhelming majority of real knockout
    competitions without needing per-competition metadata the CSV
    doesn't provide."""
    if not division_text:
        return False
    text_lower = str(division_text).lower()
    return any(kw in text_lower for kw in CUP_TOURNAMENT_KEYWORDS)


def filter_to_standard_leagues(divisions: list[str]) -> tuple[list[str], list[str]]:
    """Splits a list of division names into (standard_leagues, excluded).
    Use this to keep cup/tournament competitions out of the workspace
    dropdown entirely, per the model being strictly for standard league
    play."""
    standard = [d for d in divisions if not is_cup_or_tournament(d)]
    excluded = [d for d in divisions if is_cup_or_tournament(d)]
    return standard, excluded


def normalize_name_casing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Makes 'Chelsea' and 'chelsea' resolve to ONE team instead of two
    different ones. Builds a SINGLE canonical-casing map from the
    combined values across ALL the given columns together (not one map
    per column) - otherwise 'Chelsea' could end up resolving to a
    different casing in home_team than in away_team, which would still
    silently split the same real team into two. Each variant is mapped to
    whichever ORIGINAL casing appears most often across all those columns
    combined (ties broken by first-seen), which preserves correct
    capitalization for names like 'PSV' or 'AS Roma' that a blind
    .title() would mangle into 'Psv' or 'As Roma'. Columns that don't
    exist are skipped."""
    df = df.copy()
    existing_columns = [c for c in columns if c in df.columns]
    if not existing_columns:
        return df

    combined = pd.concat([df[c].dropna().astype(str) for c in existing_columns], ignore_index=True)
    if combined.empty:
        return df

    canonical_map = combined.groupby(combined.str.lower()).agg(
        lambda s: s.value_counts().idxmax()
    ).to_dict()

    for col in existing_columns:
        df[col] = df[col].apply(
            lambda v: canonical_map.get(str(v).lower(), v) if pd.notna(v) else v
        )
    return df


def is_unplayed(home_goals_val, away_goals_val) -> bool:
    """A match counts as unplayed if either goals cell is blank/NaN, or
    contains a comma (covers both conventions described in the spec -
    a genuinely empty cell, or a combined 'x,y' placeholder string some
    spreadsheets use for an unplayed fixture)."""
    for val in (home_goals_val, away_goals_val):
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        text = str(val).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return True
        if "," in text:
            return True
        try:
            float(text)
        except (TypeError, ValueError):
            return True
    return False


def split_played_unplayed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (settled_df, upcoming_df) for a division's fixtures."""
    unplayed_mask = df.apply(
        lambda r: is_unplayed(r.get("home_goals"), r.get("away_goals")), axis=1
    )
    return df[~unplayed_mask].copy(), df[unplayed_mask].copy()


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Parses the date column - and, critically, GUARANTEES the returned
    dataframe always has a `col` column afterward, even if the source CSV
    used a different name (or no date column at all). Everything
    downstream (decay weighting, backtesting, the fixture picker) reads
    this column directly off dataframe rows, so silently having no such
    column crashes deep in an unrelated tab with a cryptic
    'Pandas object has no attribute date' AttributeError instead of a
    clear message - this is what that bug looked like in practice."""
    df = df.copy()
    if col not in df.columns:
        detected = find_date_column(df)
        if detected is not None:
            df = df.rename(columns={detected: col})
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        # No date-like column found anywhere - fill with NaT rather than
        # leaving the column missing, so `row.date` always resolves to
        # something (safely treated as "unknown date") instead of raising.
        df[col] = pd.NaT
    return df


# ---------------------------------------------------------------------------
# Section 4 & Core Parameter A: time-decay weighted territory vectors
# ---------------------------------------------------------------------------

def decay_weights(dates: pd.Series, reference_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Weight = exp(-ln(2) * days_elapsed / half_life). More recent
    matches (smaller days_elapsed) get a weight closer to 1.0."""
    days_elapsed = (reference_date - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    return np.exp(-math.log(2) * days_elapsed / half_life_days)


# Used only when "Freeze Decay" is on AND index-based weighting is
# requested - see decay_weights_by_index below for why this exists.
FROZEN_HALF_LIFE_MATCHES = 8.0  # a reasonable, editable index-based analog
                                 # to the 45 calendar-day freeze setting -
                                 # NOT a precise conversion (that depends on
                                 # fixture density, which varies by league)


def decay_weights_by_index(n_matches: int, half_life_matches: float) -> np.ndarray:
    """Weight = exp(-ln(2) * matches_ago / half_life_matches), where
    matches_ago counts backward from the most recent match (0) regardless
    of the ACTUAL CALENDAR GAP between matches.

    Why this exists: calendar-day decay has a blind spot. If a team's
    last domestic match was right before a long international break or
    the summer off-season, EVERY one of their matches - including the
    handful right before the break, which are still the most relevant
    form reference available - ends up heavily time-decayed just because
    a lot of calendar days happened to pass, not because the team's form
    is actually stale. Counting by match INDEX instead of days sidesteps
    that: the team's most recent match is always weight 1.0, their
    second-most-recent is next, and so on, regardless of how many
    calendar days sit between them and the upcoming fixture."""
    if n_matches <= 0:
        return np.array([])
    matches_ago = np.arange(n_matches, dtype=float)  # 0 = most recent (rows must be sorted newest-first)
    return np.exp(-math.log(2) * matches_ago / max(half_life_matches, 1e-6))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = weights.sum()
    if total_weight <= 0:
        return float(np.mean(values)) if len(values) else 0.0
    return float(np.sum(values * weights) / total_weight)


@dataclass
class TerritoryProfile:
    """The three advanced territory metrics for one team, one venue role
    (host or visitor), for both what they generate ('for') and what they
    concede ('against')."""
    n_matches: int
    big_chances_for: float
    big_chances_against: float
    shots_on_target_for: float
    shots_on_target_against: float
    box_touches_for: float
    box_touches_against: float


def team_territory_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
) -> TerritoryProfile | None:
    """venue is 'home' or 'away'. Only ever looks at rows where the team
    played in that exact venue role - this is the "strict venue-isolated
    split" from Section 3. Returns None if there's no data at all for
    this team/venue (caller applies the 5-match safety rail).

    use_match_index=True switches the decay basis from calendar days to
    match recency index (see decay_weights_by_index) - this is what
    "Freeze Decay" now uses, specifically to avoid a summer break or
    international window fictitiously flattening a team's recent form
    just because a lot of calendar days happened to pass."""
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"

    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)  # index 0 = most recent
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    def wmean(col):
        vals = rows[col].fillna(0.0).to_numpy(dtype=float)
        return weighted_mean(vals, w)

    return TerritoryProfile(
        n_matches=len(rows),
        big_chances_for=wmean(f"{for_prefix}big_chances"),
        big_chances_against=wmean(f"{against_prefix}big_chances"),
        shots_on_target_for=wmean(f"{for_prefix}shots_on_target"),
        shots_on_target_against=wmean(f"{against_prefix}shots_on_target"),
        box_touches_for=wmean(f"{for_prefix}box_touches"),
        box_touches_against=wmean(f"{against_prefix}box_touches"),
    )


@dataclass
class LeagueBaseline:
    """League-wide averages used as the shared, non-team-specific
    scale anchor - see design note #2 at the top of this file."""
    avg_home_goals: float
    avg_away_goals: float
    home_big_chances_for: float
    home_big_chances_against: float
    home_sot_for: float
    home_sot_against: float
    home_box_for: float
    home_box_against: float
    away_big_chances_for: float
    away_big_chances_against: float
    away_sot_for: float
    away_sot_against: float
    away_box_for: float
    away_box_against: float


def compute_league_baseline(settled_df: pd.DataFrame) -> LeagueBaseline:
    def col_mean(col):
        return float(settled_df[col].fillna(0.0).mean()) if col in settled_df.columns and len(settled_df) else 0.0

    return LeagueBaseline(
        avg_home_goals=col_mean("home_goals") or 1.0,
        avg_away_goals=col_mean("away_goals") or 1.0,
        home_big_chances_for=col_mean("home_big_chances") or 1.0,
        home_big_chances_against=col_mean("away_big_chances") or 1.0,
        home_sot_for=col_mean("home_shots_on_target") or 1.0,
        home_sot_against=col_mean("away_shots_on_target") or 1.0,
        home_box_for=col_mean("home_box_touches") or 1.0,
        home_box_against=col_mean("away_box_touches") or 1.0,
        away_big_chances_for=col_mean("away_big_chances") or 1.0,
        away_big_chances_against=col_mean("home_big_chances") or 1.0,
        away_sot_for=col_mean("away_shots_on_target") or 1.0,
        away_sot_against=col_mean("home_shots_on_target") or 1.0,
        away_box_for=col_mean("away_box_touches") or 1.0,
        away_box_against=col_mean("home_box_touches") or 1.0,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


# The 3 territory ratios are (big_chances, shots_on_target, box_touches),
# in that fixed order, wherever a `weights` tuple appears. Equal weighting
# is the historical default (matches the model's original behavior
# unchanged); optimize_territory_weights() below can find a better-fitting
# blend per league instead.
TERRITORY_WEIGHTS_DEFAULT = (1 / 3, 1 / 3, 1 / 3)

# A modest, deliberately small grid for the weight-calibration search -
# each candidate here means re-running a walk-forward backtest, so this
# stays a manageable size rather than an exhaustive continuous search.
_TERRITORY_WEIGHT_STEPS = [0.15, 0.25, 1 / 3, 0.45, 0.55]


def _territory_weight_candidates():
    candidates = []
    for big_chances_w in _TERRITORY_WEIGHT_STEPS:
        for sot_w in _TERRITORY_WEIGHT_STEPS:
            box_w = 1.0 - big_chances_w - sot_w
            if 0.05 <= box_w <= 0.6:
                candidates.append((round(big_chances_w, 4), round(sot_w, 4), round(box_w, 4)))
    return candidates


def attack_strength(profile, baseline: LeagueBaseline, venue: str, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Composite attack strength = weighted average of the 3 territory
    ratios vs the league baseline for that venue role (big chances, shots
    on target, box touches - in that order in `weights`). Defaults to
    equal weighting (1/3 each), matching the original behavior; a
    per-league calibrated weighting can be passed in instead - see
    optimize_territory_weights(). Falls back to a neutral 1.0 if the
    sample is too small (Section 3's 5-match safety rail)."""
    if profile is None or profile.n_matches < MIN_SAMPLE_ROWS:
        return 1.0
    if venue == "home":
        ratios = [
            _safe_ratio(profile.big_chances_for, baseline.home_big_chances_for),
            _safe_ratio(profile.shots_on_target_for, baseline.home_sot_for),
            _safe_ratio(profile.box_touches_for, baseline.home_box_for),
        ]
    else:
        ratios = [
            _safe_ratio(profile.big_chances_for, baseline.away_big_chances_for),
            _safe_ratio(profile.shots_on_target_for, baseline.away_sot_for),
            _safe_ratio(profile.box_touches_for, baseline.away_box_for),
        ]
    return float(np.average(ratios, weights=weights))


def defense_strength(profile, baseline: LeagueBaseline, venue: str, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Composite defense strength (how much this team ALLOWS relative to
    league baseline, in the same venue role) - values BELOW 1.0 mean a
    better-than-average defense. Same weighting contract as
    attack_strength() above. Falls back to neutral 1.0 on a too-small
    sample."""
    if profile is None or profile.n_matches < MIN_SAMPLE_ROWS:
        return 1.0
    if venue == "home":
        ratios = [
            _safe_ratio(profile.big_chances_against, baseline.home_big_chances_against),
            _safe_ratio(profile.shots_on_target_against, baseline.home_sot_against),
            _safe_ratio(profile.box_touches_against, baseline.home_box_against),
        ]
    else:
        ratios = [
            _safe_ratio(profile.big_chances_against, baseline.away_big_chances_against),
            _safe_ratio(profile.shots_on_target_against, baseline.away_sot_against),
            _safe_ratio(profile.box_touches_against, baseline.away_box_against),
        ]
    return float(np.average(ratios, weights=weights))


def expected_goals(
    home_attack: float, away_defense: float, away_attack: float, home_defense: float, baseline: LeagueBaseline,
) -> tuple[float, float]:
    """Home advantage is NOT re-applied here - it's already inside
    home_attack/home_defense because those numbers only ever came from
    the team's own home-position rows (see design note #1)."""
    lambda_home = baseline.avg_home_goals * home_attack * away_defense
    lambda_away = baseline.avg_away_goals * away_attack * home_defense
    return max(lambda_home, 0.05), max(lambda_away, 0.05)


# ---------------------------------------------------------------------------
# Core Parameter A: dynamic half-life optimisation via Brier score backtest
# ---------------------------------------------------------------------------

def _quick_lambda_for_backtest(
    history_df: pd.DataFrame, home_team: str, away_team: str, half_life_days: float, as_of: pd.Timestamp,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
):
    if len(history_df) < MIN_SAMPLE_ROWS * 2:
        return None
    baseline = compute_league_baseline(history_df)
    home_profile = team_territory_profile(history_df, home_team, "home", half_life_days, as_of)
    away_profile = team_territory_profile(history_df, away_team, "away", half_life_days, as_of)
    ha = attack_strength(home_profile, baseline, "home", weights)
    hd = defense_strength(home_profile, baseline, "home", weights)
    aa = attack_strength(away_profile, baseline, "away", weights)
    ad = defense_strength(away_profile, baseline, "away", weights)
    return expected_goals(ha, ad, aa, hd, baseline)


def _outcome_probs_from_lambdas(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    home_win = draw = away_win = 0.0
    for x in range(GOAL_CAP + 1):
        px = scipy_poisson.pmf(x, lam_home)
        for y in range(GOAL_CAP + 1):
            py = scipy_poisson.pmf(y, lam_away)
            p = px * py
            if x > y:
                home_win += p
            elif x == y:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return home_win / total, draw / total, away_win / total


def optimize_half_life(settled_df: pd.DataFrame, max_matches_evaluated: int = 150, weights: tuple = TERRITORY_WEIGHTS_DEFAULT):
    """Backtests each candidate half-life on real past results using a
    Brier score, and returns the one with the lowest average error. Only
    evaluates the most recent `max_matches_evaluated` matches for
    performance - this is a real backtest, not a fixed guess, but a
    league's full season history doesn't need to be replayed dozens of
    times over to get a stable answer."""
    df = settled_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 3:
        return FROZEN_HALF_LIFE_DAYS, {"reason": "not enough settled matches to backtest - using frozen default"}

    eval_start = max(MIN_SAMPLE_ROWS * 2, len(df) - max_matches_evaluated)
    scores = {}
    for half_life in HALF_LIFE_CANDIDATES:
        brier_terms = []
        for i in range(eval_start, len(df)):
            row = df.iloc[i]
            history = df.iloc[:i]
            lambdas = _quick_lambda_for_backtest(
                history, row["home_team"], row["away_team"], half_life, row["date"], weights
            )
            if lambdas is None:
                continue
            p_home, p_draw, p_away = _outcome_probs_from_lambdas(*lambdas)
            actual = (
                (1, 0, 0) if row["home_goals"] > row["away_goals"]
                else (0, 1, 0) if row["home_goals"] == row["away_goals"]
                else (0, 0, 1)
            )
            brier_terms.append(
                (p_home - actual[0]) ** 2 + (p_draw - actual[1]) ** 2 + (p_away - actual[2]) ** 2
            )
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       