#!/bin/bash
set -e

echo "=== WC PREDICTOR BACKTESTING ==="
export PYTHONPATH=.

python -m src.model.evaluate
