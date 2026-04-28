# FIFA 2026 World Cup Win Probability Predictor

A full-stack machine learning project that predicts 2026 FIFA World Cup win probabilities for all 48 participating nations. The system combines scraped squad/player data, international match history, an XGBoost matchup model, expected-goals models, and a 100,000-run Monte Carlo tournament simulation.

The current project no longer uses fabricated random player ratings or squads. Squad composition, player market values, club metadata, league tiers, and derived player ratings are built from Transfermarkt-scraped data, then rolled up into national-team features.

## Contents

1. [Architecture](#architecture)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Modeling](#modeling)
5. [Tournament Simulation](#tournament-simulation)
6. [Vercel Web App](#vercel-web-app)
7. [Generated Artifacts](#generated-artifacts)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Testing](#testing)
11. [Caveats](#caveats)

## Architecture

```text
Raw data + Transfermarkt scrape
          |
          v
Squads, player ratings, match history
          |
          v
Team features + matchup features
          |
          v
Match outcome model + xG models + calibration
          |
          v
Precomputed H2H matchup predictions
          |
          v
100,000-run Monte Carlo tournament simulation
          |
          v
Vercel static web app
```

The main layers are:

- `src/scraper`: Transfermarkt squad/player extraction and parsing.
- `src/features`: team-level and matchup-level feature generation.
- `src/model`: match outcome and expected-goals inference.
- `src/simulation`: official 2026 bracket simulation and results aggregation.
- `src/api`: optional local FastAPI endpoints for teams, matchups, explanations, groups, and tournament results.
- `web`: Vercel-ready static web app for probabilities, team deep dives, SHAP explanations, group simulation, and H2H matchups.
- `scripts/export_vercel_data.py`: exports model outputs into static JSON files used by the Vercel app.

## Data Pipeline

The pipeline uses three primary data sources:

| Source | Data | Output |
|--------|------|--------|
| Transfermarkt | national squads, player market values, ages, positions, clubs, club leagues | `data/raw/squads.csv`, `data/raw/player_ratings.csv` |
| International results dataset | historical match outcomes, goals, dates, tournaments | `data/raw/international_matches.csv` |
| Project config/static files | 2026 groups, host flags, tournament settings | `data/external/wc2026_groups.json`, `config.yaml` |

Transfermarkt does not provide a direct FIFA-style player rating, so the project derives a deterministic composite rating from scraped player and club information. The rating is based on market value, age curve, caps/experience when available, position, club league strength, league tier, and availability signals. This keeps squad strength tied to real scraped data rather than random generation.

The pipeline also fixes club-league fields so player and squad records carry meaningful `club_league` and `club_league_tier` values instead of falling back to `Unknown` and `0`.

## Feature Engineering

Team features combine recent national-team performance with squad quality:

- Recency-weighted win/draw/loss rates.
- Recency-weighted goals scored and conceded.
- Clean-sheet and goal-difference trends.
- ELO-style team strength.
- Tournament and World Cup experience.
- Squad average rating, starter rating, depth score, position-group ratings, average age, caps, top-league ratio, and market value.
- Host-nation and confederation metadata.

Matchup features are built directionally for every ordered team pair:

- ELO difference plus absolute team ELO values.
- Form, goals, tournament history, and squad-quality differentials.
- Goalkeeper, defender, midfielder, forward, depth, and market-value differentials.
- Neutral-venue and host flags.
- Same-confederation indicator.
- Smoothed head-to-head history with shrinkage for small samples.

All tournament matchup predictions are precomputed into `data/processed/match_predictions.csv` so the Monte Carlo simulation and group simulator can run quickly and consistently.

## Modeling

The primary match outcome model is an XGBoost multiclass classifier predicting:

- away win
- draw
- home win

The model is trained from historical international fixtures using chronological splits to reduce leakage. A calibrated model is saved when available, and expected-goals regressors predict home and away xG for scoreline simulation and group-stage tiebreakers.

Current model artifacts include:

- `models/match_outcome_model.pkl`
- `models/match_outcome_model_xgb.pkl`
- `models/match_outcome_model_lgbm.pkl`
- `models/calibrator.pkl`
- `models/xg_home_model.pkl`
- `models/xg_away_model.pkl`

SHAP explanations are available in the Vercel app from pre-exported JSON. Team-specific SHAP handles the model's multiclass SHAP tensor correctly and explains the selected team against a median-strength opponent. Positive and negative drivers are not forced to be balanced; very strong or weak teams can naturally show mostly positive or mostly negative drivers.

## Tournament Simulation

The simulation implements the 2026 48-team format:

- 12 groups of 4 teams.
- Top 2 teams in each group qualify automatically.
- The 8 best third-place teams also qualify.
- 32 teams enter the first knockout round.
- Knockout matches proceed through R32, R16, quarterfinals, semifinals, and final.

Group-stage matches are sampled from the precomputed matchup probabilities. Goals are sampled from xG-driven Poisson distributions and adjusted to match the sampled result class. Group standings use points, goal difference, goals for, and random tie-breaking as a final fallback.

The R32 bracket uses the official 2026 slot structure and assigns third-place teams only to eligible slots. Knockout draws are resolved through extra time and penalties. Extra-time resolution is nudged by ELO; penalties are nudged by goalkeeper rating.

The production simulation is run at 100,000 iterations. Results are saved to:

```text
data/processed/simulation_results.csv
```

## Vercel Web App

The production web app is a static Vercel deployment. It does not run Streamlit or the Python API at request time. Instead, the current model outputs are exported to JSON under `web/public/data`, and the browser app renders:

- Tournament probability chart sorted by descending win probability, using one country color.
- Team Deep Dive with ELO, squad strength, recent win rate, tournament path, and SHAP drivers.
- Model Explainability with global feature importances and team-specific SHAP analysis.
- Head-to-Head Matchup Predictor with probability donut, key factors, expected goals, and no duplicate-team selection.
- Group Simulator with browser-side Monte Carlo logic for expected standings, R32 qualification probability, and top-two finish probability.

Vercel deployment files:

```text
package.json
vercel.json
web/index.html
web/styles.css
web/app.js
web/public/data/*.json
```

Import the repository into Vercel from the repo root. The included `vercel.json` sets `npm run build` as the build command and `dist` as the output directory.

Local preview:

```bash
npm run dev
```

Open:

```text
http://localhost:4173
```

### Optional Local API

The FastAPI backend remains available for local experimentation and regenerating explanations:

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Generated Artifacts

Important generated files:

```text
data/raw/international_matches.csv
data/raw/squads.csv
data/raw/player_ratings.csv
data/processed/team_features.csv
data/processed/matchup_features.csv
data/processed/match_predictions.csv
data/processed/simulation_results.csv
web/public/data/simulation_results.json
web/public/data/teams.json
web/public/data/match_predictions.json
web/public/data/groups.json
web/public/data/feature_importance.json
web/public/data/explanations.json
models/*.pkl
```

`match_predictions.csv` should contain every ordered matchup between the 48 teams, excluding self-matches. With 48 teams, that is `48 * 47 = 2256` rows.

## Installation

```bash
git clone https://github.com/your-username/wc2026-predictor
cd wc2026-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If using `football-data.org`, add your API key to `.env`.

## Usage

Build the Vercel static app:

```bash
npm run build
```

Preview the Vercel app locally:

```bash
npm run dev
```

Regenerate the static JSON data after retraining or rerunning simulations:

```bash
PYTHONPATH=. python scripts/export_vercel_data.py
```

Run the full pipeline:

```bash
bash scripts/run_pipeline.sh
```

Run only the 100,000-simulation Monte Carlo step:

```bash
PYTHONPATH=. python src/simulation/monte_carlo.py
```

Run the optional local API:

```bash
uvicorn src.api.main:app --reload --port 8000
```

For local virtualenv usage in this repo, the commands are typically:

```bash
PYTHONPATH=. venv/bin/python src/simulation/monte_carlo.py
PYTHONPATH=. venv/bin/uvicorn src.api.main:app --reload --port 8000
PYTHONPATH=. venv/bin/python scripts/export_vercel_data.py
```

## Testing

Run the test suite:

```bash
PYTHONPATH=. pytest
```

or:

```bash
PYTHONPATH=. venv/bin/pytest
```

The suite covers:

- API smoke tests and same-team H2H rejection.
- Feature engineering utilities.
- Model prediction behavior.
- Simulation primitives and aggregation.
- Transfermarkt parser behavior.
- Vercel static build via `npm run build`.

The current project test suite passes with:

```text
20 passed
```

The latest H2H data audit found:

- 2256 matchup rows.
- 0 self-matches.
- 0 duplicate directed pairs.
- 0 missing reciprocal fixtures.
- 0 invalid probability sums.
- 0 out-of-range probabilities.

## Caveats

This is a probabilistic model, not an oracle. It cannot know future injuries, tactical changes, squad-selection surprises, or match-specific volatility. Knockout football is especially noisy, so even the best team should not be expected to carry an overwhelming tournament win probability.

The Transfermarkt-derived ratings are deterministic and data-driven, but they are still engineered proxies rather than official player ratings. Market value can reflect age, contract, league visibility, and hype as well as ability, so it should be interpreted as one signal among many.

The group simulator reports Round of 32 qualification for the 2026 format, not only top-two group advancement. A team can finish third often and still have a meaningful R32 probability if its third-place record is strong enough relative to the other groups.

These probabilities are for analysis and experimentation only and should not be used for gambling or financial decision-making.
