# FIFA 2026 World Cup Win Probability Predictor

A full-stack machine learning project that predicts 2026 FIFA World Cup win probabilities for all 48 participating nations. The system combines scraped squad/player data, international match history, an XGBoost matchup model, expected-goals models, and a 100,000-run Monte Carlo tournament simulation.

The current project no longer uses fabricated random player ratings or squads. Squad composition, player market values, club metadata, league tiers, and derived player ratings are built from Transfermarkt-scraped data, then rolled up into national-team features.

## Contents

1. [Architecture](#architecture)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Modeling](#modeling)
5. [Tournament Simulation](#tournament-simulation)
6. [API and Dashboard](#api-and-dashboard)
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
FastAPI backend + Streamlit dashboard
```

The main layers are:

- `src/scraper`: Transfermarkt squad/player extraction and parsing.
- `src/features`: team-level and matchup-level feature generation.
- `src/model`: match outcome and expected-goals inference.
- `src/simulation`: official 2026 bracket simulation and results aggregation.
- `src/api`: FastAPI endpoints for teams, matchups, explanations, groups, and tournament results.
- `src/dashboard`: Streamlit app for probabilities, team deep dives, SHAP explanations, group simulation, and H2H matchups.

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

SHAP explanations are available through the API and dashboard. Team-specific SHAP now handles the model's multiclass SHAP tensor correctly and explains the selected team against a median-strength opponent. Positive and negative drivers are not forced to be balanced; very strong or weak teams can naturally show mostly positive or mostly negative drivers.

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

## API and Dashboard

### FastAPI

Run the API with:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Useful endpoints:

- `GET /teams/{team_name}`: team-level details.
- `GET /predictions/tournament`: latest tournament simulation table.
- `GET /predictions/match?home=Spain&away=France`: H2H matchup prediction.
- `GET /predictions/group/F`: quick group simulation with R32 qualification probabilities.
- `GET /predictions/explain/Spain`: SHAP explanation for a team.
- `POST /simulate`: trigger a simulation job.
- `GET /simulate/status/{job_id}`: check simulation job status.

The H2H endpoint rejects identical teams, and the dashboard prevents selecting the same team in both H2H dropdowns.

### Streamlit Dashboard

Run the dashboard with:

```bash
streamlit run src/dashboard/app.py
```

Open:

```text
http://localhost:8501
```

Dashboard views:

- Tournament probability chart sorted by descending win probability, using a single color for all countries.
- Team Deep Dive with ELO, squad strength, recent win rate, and SHAP drivers.
- Model Explainability with global feature importances and team-specific SHAP analysis.
- Head-to-Head Matchup Predictor with probability donut, key factors, and expected goals.
- Group Simulator showing expected standings, average points, R32 qualification probability, and top-two finish probability.

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

Run the full pipeline:

```bash
bash scripts/run_pipeline.sh
```

Run only the 100,000-simulation Monte Carlo step:

```bash
PYTHONPATH=. python src/simulation/monte_carlo.py
```

Run API and dashboard:

```bash
uvicorn src.api.main:app --reload --port 8000
streamlit run src/dashboard/app.py
```

For local virtualenv usage in this repo, the commands are typically:

```bash
PYTHONPATH=. venv/bin/python src/simulation/monte_carlo.py
PYTHONPATH=. venv/bin/uvicorn src.api.main:app --reload --port 8000
venv/bin/streamlit run src/dashboard/app.py
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
