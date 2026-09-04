# Sisonke Football Predictive Terminal

A self-contained Streamlit betting-market valuation terminal built strictly
for standard round-robin league play (not cups/tournaments). Upload your
own historical match CSV, and it calculates real, data-derived
probabilities via two independent engines (Dixon-Coles Poisson and a
10,000-run Monte Carlo simulator), cross-checks them against each other,
and surfaces expected-value edges across 22 betting markets.

## Files

- `sisonke_engine.py` - all the math, kept separate from the UI so it can
  be tested on its own (no Streamlit needed to run the tests).
- `sisonke_app.py` - the Streamlit dashboard (sidebar, tabs, inputs, charts).
- `test_engine.py` - 127 direct unit tests covering every formula in the
  engine, including edge cases (no-lookahead backtesting, Brier score
  bounds, position-distribution sums, case-insensitive name merging, the
  striker-vs-defender injury split, cup/tournament filtering). Run with
  `python3 test_engine.py`.
- `integration_dry_run.py` - an end-to-end smoke test against a synthetic
  season (including a brand-new team with zero matches, and unplayed
  fixtures marked with blanks/comma-junk) exercising the exact call
  sequence the live app makes. Run with `python3 integration_dry_run.py`.
- `bias_stress_test.py` - specifically tests for home-team bias: builds a
  deliberately lopsided league and asserts the model correctly favors the
  stronger team regardless of venue, and that Home Win probability
  actually spreads out across different matchups rather than clustering
  near a fixed number. Run with `python3 bias_stress_test.py`.
- `requirements.txt`

## Run it

```
pip install -r requirements.txt
streamlit run sisonke_app.py
```

## What your CSV needs

Column names are auto-standardised (lowercased, spaces -> underscores).
Team names are also auto-normalised for casing (`Chelsea` / `chelsea` /
`CHELSEA` all merge into one team - whichever casing appears most often
in your file is kept as the display form). Required columns:

- A division column named `league_country`, `league`, or `competition`
- `date`, `home_team`, `away_team`
- `home_goals`, `away_goals` - leave blank (or comma-junk placeholder
  text) for fixtures not yet played
- `home_shots_on_target`, `away_shots_on_target`
- `home_big_chances`, `away_big_chances`
- `home_box_touches`, `away_box_touches`

Divisions that look like cups/tournaments by name (Cup, Champions League,
Europa, Copa, Playoff, Final, etc.) are automatically filtered out of the
workspace selector - this model assumes round-robin fixture structure
(used throughout xPts and the season simulator), which doesn't hold for
knockout competitions.

## What's in the dashboard

- **Sentiment Tracker** (offline, advisory-only tab): league profile
  banner, 7-day diary checklist, sentiment logging, a confidence score
  that never blocks the main analytics.
- **Active Projections Matrix**: half-life (auto-optimised or frozen),
  volatility auto-calibration, momentum/streak banner, a full tactical
  panel (separate Home/Away columns, exactly as specced) including
  split striker-vs-defender injury checkboxes, tactical setup dropdown
  (Standard / Low-Block / Counter-Press), pitch surface, weather,
  referee strictness, pre-season flag, travel fatigue as a dropdown
  (0-3), and a manual **key player signing/departure impact slider**
  (-20% to +20%, Home and Away separately). A dynamic, auto-generated paragraph explains WHY each
  prediction came out the way it did - referencing the real strength
  numbers, momentum, and every applied multiplier, not a canned
  template. Engine-comparison charts (scoreline bars, MC goal
  distribution, DC-vs-MC bar chart) sit above the full 22-market
  valuation sheet, the parlay/Kelly builder, and a Telegram "send this
  prediction" button.
- **Live Standings Ledger**: xPts table (games played / W / D / L / goal
  difference, sorted by real points) and a 10,000-run season forecast
  with per-team title odds input, an Edge column, relegation risk, and a
  full finishing-position distribution (chance of landing in each exact
  table position).
- **Performance Backtester**: runs across the WHOLE settled dataset (not
  just the last 10 matches), walk-forward - each historical prediction
  only ever uses data from strictly before that match's date. Shows
  Brier Skill Score and accuracy % (both overall and filtered by a
  confidence floor slider), a manual-override sensitivity slider, and a
  weighted-vs-raw-data comparison chart.
- **Full Database View**: your raw data plus computed goal difference and
  a lightweight per-row implied-xG estimate.
- **Sidebar**: local CSV download, per-league or full-database clearing,
  and an optional Telegram bot connection (token + chat ID, sends on
  demand only - never automatically).

## Design notes worth knowing

- **No separate home-field-advantage multiplier anywhere.** Home
  advantage is captured structurally by splitting every team's stats into
  "as host" vs "as visitor" streams from the start - adding a second,
  independent HFA bonus on top of that would double-count the same
  effect. Verified directly in `test_engine.py`.
- **Every probability is calculated, not assumed.** Dixon-Coles' rho is
  fitted from your own data's low-scoring match frequency. The Monte
  Carlo engine draws independently from raw expected-goals rates rather
  than resampling the Dixon-Coles matrix, so the two act as genuine
  cross-checks - see the Convergence Score column.
- **Tactical multiplier values were checked against real research, not
  just assumed.** E.g. "new manager bounce" is documented against actual
  Premier League points-per-game data (a ~41% raw swing that's mostly
  regression to the mean) and deliberately damped to a conservative 10%
  rather than applying the full raw number. Striker vs defender injuries
  are now separate checkboxes because they affect different things
  (attack quality loss vs defensive variance/errors) - see
  `apply_tactical_multipliers`'s docstring in `sisonke_engine.py` for the
  full reasoning and citations on each one. A "stacking realism clamp"
  also stops many bad-luck factors ticked at once from compounding into
  an implausible swing no real match would show.
- **Recommended Action column** combines edge size AND how much the two
  engines agree - a big edge the two models disagree on is flagged for
  caution, not treated the same as one they both confirm.

## Honest limitations

- **The Gold Mine strategy hints and league playstyle tags (49 leagues)
  are general football-reputation starting points**, not the output of a
  rigorous statistical backtest of each specific league - stated in
  `sisonke_app.py`'s docstring too. Once you've loaded a season of your
  own data, use the Standings Ledger and Full Database View to check a
  league's real tendencies against your own numbers.
- **Test suite**: now 141 unit tests (see `test_engine.py`), including
  the key-player-transfer slider's math and its correct sign convention
  in both directions.
- **Several environmental multipliers are reasoned estimates, not
  numbers fitted to data**: the counter-pressing tactical style, pitch
  surface, weather, and the lenient-referee value weren't specified with
  exact numbers in the original spec, so I assigned defensible starting
  values and documented them as such. They're all editable in the
  sidebar for exactly that reason - once you have backtest results
  against your own league, adjust them. **The key player transfer
  impact is different from these**: it's deliberately a manual slider,
  not even a starting-point constant, because unlike a manager change
  there's no multi-season league-wide dataset to check a specific
  transfer's impact against - it depends too much on the individual
  player, position, and replacement quality for one fixed number to be
  honest.
- **The Telegram integration is structurally correct but I couldn't test
  an actual send** - I don't have a real bot token or chat ID in the
  environment I built this in. What's verified: `send_telegram_message`
  correctly rejects a missing token/chat ID before attempting a request.
  What's NOT verified: an actual successful delivery to a real Telegram
  chat - try it with your own bot and let me know if anything's off.
- **I could not launch the live Streamlit UI myself** (no network access
  to install Streamlit in the sandbox I built this in). What I verified
  directly: all 127 unit tests pass, a full integration dry run against
  a synthetic season completes with zero exceptions, and a dedicated
  bias stress test confirms the model correctly favors a genuinely
  stronger team regardless of home/away venue (Home Win probability
  ranged from 9.4% to 65.0% across different matchups in that test - not
  clustering near a fixed number). What I did NOT verify myself: the
  actual browser rendering and click-through behavior of the Streamlit
  widgets. Run it yourself and tell me if any UI element doesn't behave
  as expected.
