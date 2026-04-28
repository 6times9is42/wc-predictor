# 🏆 FIFA 2026 World Cup Win Probability Predictor

A full-stack ML system that predicts 2026 FIFA World Cup win probabilities for all 48 participating nations. Powered by XGBoost, expected-goals regression, and a 100,000-run Monte Carlo tournament simulation built on real Transfermarkt squad data.

**🌐 Live App → [wc-predictor-wine.vercel.app](https://wc-predictor-wine.vercel.app/)**

---

## Contents
1. [Architecture](#architecture)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Modeling](#modeling)
5. [Tournament Simulation](#tournament-simulation)
6. [Web App](#web-app)
7. [Testing](#testing)
8. [Caveats](#caveats)

---

## Architecture

```
Transfermarkt scrape + International match history
                    │
                    ▼
        Squad/player ratings + match results
                    │
                    ▼
      Team features + matchup differential features
                    │
                    ▼
   XGBoost match outcome model + xG regressors + calibration
                    │
                    ▼
         Precomputed H2H match predictions (2,256 pairs)
                    │
                    ▼
       100,000-run Monte Carlo tournament simulation
                    │
                    ▼
         Static JSON export → Vercel web app
```

**Layers:**
- `src/scraper` — Transfermarkt squad/player extraction
- `src/features` — team-level and matchup-level feature generation
- `src/model` — match outcome and expected-goals inference
- `src/simulation` — official 2026 bracket simulation and results aggregation
- `src/api` — optional local FastAPI endpoints
- `web` — Vercel static frontend
- `scripts/export_vercel_data.py` — exports model outputs to static JSON

---

## Data Pipeline

| Source | Data | Output |
|--------|------|--------|
| Transfermarkt | Squads, market values, ages, positions, clubs, league tiers | `data/raw/squads.csv`, `data/raw/player_ratings.csv` |
| International results dataset | Match outcomes, goals, dates, tournaments | `data/raw/international_matches.csv` |
| Static config | 2026 groups, host flags, tournament settings | `data/external/wc2026_groups.json` |

Since Transfermarkt doesn't provide FIFA-style ratings, the project derives a deterministic composite rating from market value, age curve, caps, position, club league strength, and availability. This keeps squad strength grounded in real scraped data rather than arbitrary generation.

---

## Feature Engineering

**Team-level features:**
- Recency-weighted win/draw/loss rates, goals scored/conceded, clean-sheet rate, goal difference
- ELO-style team strength rating
- Tournament and World Cup historical experience
- Squad average rating, starter rating, depth score, position-group ratings (GK/DEF/MID/FWD)
- Squad avg age, caps, top-league ratio, total market value

**Matchup features** (built for all 48×47 = 2,256 ordered team pairs):
- ELO differential + absolute ELO values for both sides
- Form, goals, tournament history, and all squad-quality differentials
- Smoothed head-to-head history with shrinkage for small samples
- Neutral-venue flag, host-nation flag, same-confederation indicator

All matchup predictions are precomputed into `data/processed/match_predictions.csv` so simulation runs are fast and consistent.

---

## Modeling

The primary model is an **XGBoost multiclass classifier** predicting away win / draw / home win probabilities. Training uses chronological splits to prevent leakage. A calibrated wrapper is applied on top.

Separate **XGBoost regressors** predict home and away expected goals (xG), used for scoreline simulation and group-stage tiebreakers.

**Model artifacts:**
```
models/match_outcome_model.pkl
models/match_outcome_model_xgb.pkl
models/match_outcome_model_lgbm.pkl
models/calibrator.pkl
models/xg_home_model.pkl
models/xg_away_model.pkl
```

**SHAP explanations** are pre-exported to JSON. Team-specific SHAP handles the multiclass SHAP tensor correctly — a team is explained against a median-strength opponent, and positive/negative drivers are reported without forcing balance (very strong or weak teams will naturally show skewed drivers).

---

## Tournament Simulation

Implements the official **2026 48-team format**:
- 12 groups of 4 teams
- Top 2 per group qualify automatically; best 8 third-place teams also advance
- 32-team knockout bracket: R32 → R16 → QF → SF → Final

Group-stage goals are sampled from xG-driven Poisson distributions, adjusted to match the sampled result class. Standings use points → goal difference → goals for → random draw as final tiebreaker.

The R32 bracket uses the official 2026 slot assignments for third-place teams. Knockout matches resolve through extra time (ELO-nudged) and penalties (GK rating–nudged).

At **100,000 simulations**, the standard error on a 14% win probability is ±0.11% — results are stable.

Output: `data/processed/simulation_results.csv`

---

## Web App

The production app is a **fully static Vercel deployment** — no Python runs at request time. Model outputs are pre-exported to JSON under `web/public/data/` and the browser renders:

- **Tournament Overview** — all 48 teams ranked by win probability
- **Team Deep Dive** — ELO, squad strength, recent form, tournament path, SHAP drivers
- **Head-to-Head Predictor** — probability breakdown, expected goals, key factors
- **Group Simulator** — browser-side Monte Carlo for expected standings and R32 qualification %
- **Model Explainability** — global feature importances and team-specific SHAP waterfall

To regenerate static data after retraining:
```bash
PYTHONPATH=. python scripts/export_vercel_data.py
```

An optional local FastAPI backend is available for experimentation:
```bash
uvicorn src.api.main:app --reload --port 8000
```

---

## Testing

```bash
PYTHONPATH=. pytest
```

**20 tests pass**, covering:
- API smoke tests and same-team H2H rejection
- Feature engineering utilities
- Model prediction behavior and probability validity
- Simulation primitives and aggregation
- Transfermarkt parser behavior
- Vercel static build (`npm run build`)

H2H data audit: 2,256 matchup rows — 0 self-matches, 0 duplicate pairs, 0 missing reciprocals, 0 invalid probability sums, 0 out-of-range probabilities.

---

## Caveats

- **It's probabilistic, not prescient.** Injuries, tactical surprises, and knockout volatility aren't predictable. Even the model's top-ranked team typically carries only a ~12–18% win probability — football is genuinely uncertain.
- **Transfermarkt-derived ratings** are deterministic proxies, not official ratings. Market value reflects hype, contract status, and league visibility alongside ability.
- **Group simulator R32 probabilities** account for the 2026 best-third-place rule — a team finishing third frequently across simulations can still have a meaningful R32 qualification rate.
- These probabilities are for analysis only. Not for gambling or financial decisions.