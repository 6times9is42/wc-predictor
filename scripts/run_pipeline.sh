#!/bin/bash
set -e

echo "=== WC2026 PREDICTOR PIPELINE ==="
export PYTHONPATH=.

echo "[1/6] Scraping match results..."
python -m src.scraper.match_results

echo "[2/6] Scraping squad data..."
python -m src.scraper.squad_data

echo "[3/6] Scraping player ratings..."
python -m src.scraper.player_ratings

echo "[4/6] Building feature vectors..."
python -m src.features.team_features
python -m src.features.matchup_features

echo "[5/6] Training and calibrating model..."
python -m src.model.train
python -m src.model.calibrate
python -m src.model.evaluate

echo "[6/6] Running Monte Carlo simulation (100,000 runs)..."
python -m src.simulation.monte_carlo

echo "=== PIPELINE COMPLETE ==="
echo "Results saved to data/processed/simulation_results.csv"
echo "Start API: uvicorn src.api.main:app --reload"
echo "Start Dashboard: streamlit run src/dashboard/app.py"
