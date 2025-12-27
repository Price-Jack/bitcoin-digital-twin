# bitcoin-digital-twin

Real-time Bitcoin “digital twin” notebook: data ingestion, feature engineering, ML/LSTM forecasting, and evaluation.

## What this does
- Pulls BTC-USD price data from Yahoo Finance (yfinance)
- Builds technical + volume features (returns, moving averages, RSI, MACD, Bollinger width, etc.)
- Trains baseline models (linear regression + logistic regression)
- Trains an LSTM classifier on rolling sequences
- Tunes the probability threshold to maximize F1
- Compares a confidence-weighted strategy vs. baseline buy/hold (simple backtest)

## Key results (from the latest run)
Baseline Logistic Regression (direction):
- Accuracy: ~0.509
- Precision: ~0.499
- Recall: ~0.810
- F1: ~0.618

LSTM (direction) after threshold tuning:
- Best threshold: ~0.44
- Accuracy: ~0.492
- Precision: ~0.491
- Recall: ~0.997
- F1: ~0.658

Backtest summary:
- Strategy volatility lower than baseline in this run (results not statistically significant)

## How to run
### Option A: Google Colab (recommended)
Open the notebook in Colab and run all cells.

### Option B: Run locally
```bash
git clone https://github.com/Price-Jack/bitcoin-digital-twin.git
cd bitcoin-digital-twin
pip install -r requirements.txt
jupyter notebook
