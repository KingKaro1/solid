# Stock Strategy Agent

An agent design for reading market data, analyzing chart structure, testing strategy templates, and ranking the most suitable trading approaches for a given symbol and timeframe.

## What It Does

- Ingests OHLCV market data and optional fundamentals or news
- Computes technical indicators and market-regime features
- Evaluates a fixed library of strategy templates
- Backtests each candidate with walk-forward validation
- Ranks strategies by return, drawdown, stability, and regime fit
- Explains the result in plain language

## What It Does Not Do

- It does not claim to find a universal "best" trading strategy
- It does not rely on an LLM to invent trade signals
- It does not place live trades by default
- It does not recommend a strategy without test evidence

## Architecture

### 1. Data Layer

- Historical OHLCV bars
- Optional volume profile, corporate actions, earnings calendar, and news
- Persistent storage for raw and normalized data

### 2. Feature Layer

- Moving averages
- RSI, MACD, ATR, Bollinger Bands, VWAP
- Trend strength, volatility, support/resistance
- Market regime classification

### 3. Strategy Engine

Strategy templates are parameterized and tested rather than hand-waved.

- Trend following
- Mean reversion
- Breakout
- Momentum continuation
- Volatility expansion

Each strategy includes:

- Entry rules
- Exit rules
- Stop loss
- Take profit
- Position sizing
- Invalidation conditions

### 4. Backtesting Engine

- Runs each strategy over historical data
- Uses walk-forward validation
- Measures return, Sharpe, max drawdown, win rate, and trade frequency
- Checks sensitivity to slippage and regime changes

### 5. Ranking and Explanation

Scores each strategy using a weighted blend of:

- Risk-adjusted return
- Drawdown control
- Stability across folds
- Regime fit
- Trade quality

The output includes:

- Ranked strategies
- Confidence level
- Key assumptions
- Invalidation levels
- Main risks

## Recommended Tech Stack

- Backend: Python
- API: FastAPI
- Data: pandas, numpy
- Indicators: pandas-ta or custom implementations
- Backtesting: vectorbt, backtrader, or custom event-driven simulation
- Storage: PostgreSQL for metadata, Parquet for time series
- UI: Streamlit or React dashboard

## MVP Scope

Start small:

- US equities only
- Daily and hourly bars only
- Three strategies only
- No live execution
- One dashboard with chart, indicators, regime, ranking, and explanation

## Safety Rules

- Refuse to recommend when data quality is poor
- Block recommendations during obvious event shocks unless event logic exists
- Log all recommendations and inputs
- Keep the LLM in a summarization role, not a signal-generation role

## Example Output

```text
Current regime: trend-following with elevated volatility
Top strategy: breakout with volume confirmation
Backtest: Sharpe 1.4, max drawdown 12%, stable across folds
Main risk: false breakouts during earnings week
Invalidation: close back below prior resistance
```

## Next Steps

1. Build the data fetcher
2. Compute indicators and regime labels
3. Implement three strategy templates
4. Add backtesting and scoring
5. Add a minimal dashboard
6. Connect GitHub and publish the repo

