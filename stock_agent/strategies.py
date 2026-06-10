from __future__ import annotations

from dataclasses import dataclass

from .data import Bar
from .indicators import FeatureSet


@dataclass(frozen=True)
class StrategyResult:
    name: str
    equity_curve: list[float]
    total_return: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    trade_count: int
    score: float


def evaluate_strategies(bars: list[Bar], features: FeatureSet, regime: str) -> list[StrategyResult]:
    strategies = [
        _trend_following,
        _mean_reversion,
        _breakout,
    ]
    results = [strategy(bars, regime) for strategy in strategies]
    return sorted(results, key=lambda result: result.score, reverse=True)


def _trend_following(bars: list[Bar], regime: str) -> StrategyResult:
    return _run_strategy(
        bars,
        name="trend_following",
        entry=lambda i, state: state["sma_fast"][i - 1] <= state["sma_slow"][i - 1] and state["sma_fast"][i] > state["sma_slow"][i],
        exit=lambda i, state: state["sma_fast"][i] < state["sma_slow"][i] or state["rsi"][i] < 45,
        regime=regime,
    )


def _mean_reversion(bars: list[Bar], regime: str) -> StrategyResult:
    return _run_strategy(
        bars,
        name="mean_reversion",
        entry=lambda i, state: state["rsi"][i] < 30 and state["close"][i] < state["lower_band"][i],
        exit=lambda i, state: state["rsi"][i] > 50 or state["close"][i] > state["sma_slow"][i],
        regime=regime,
    )


def _breakout(bars: list[Bar], regime: str) -> StrategyResult:
    return _run_strategy(
        bars,
        name="breakout",
        entry=lambda i, state: state["close"][i] > state["rolling_high"][i - 1] and state["volume"][i] > state["avg_volume"][i - 1] * 1.1,
        exit=lambda i, state: state["close"][i] < state["sma_fast"][i] or state["rsi"][i] < 50,
        regime=regime,
    )


def _run_strategy(bars: list[Bar], name: str, entry, exit, regime: str) -> StrategyResult:
    state = _build_state(bars)
    equity = 1.0
    in_position = False
    entry_price = 0.0
    entry_equity = 1.0
    equity_curve = [equity]
    trades: list[float] = []

    for i in range(1, len(bars)):
        if not in_position and i >= 20 and entry(i, state):
            in_position = True
            entry_price = state["close"][i]
            entry_equity = equity
        elif in_position and exit(i, state):
            exit_price = state["close"][i]
            trade_return = (exit_price - entry_price) / entry_price
            equity = entry_equity * (1.0 + trade_return)
            trades.append(trade_return)
            in_position = False

        current_equity = equity
        if in_position:
            current_equity = entry_equity * (state["close"][i] / entry_price)
        equity_curve.append(current_equity)

    if in_position:
        exit_price = state["close"][-1]
        trade_return = (exit_price - entry_price) / entry_price
        equity = entry_equity * (1.0 + trade_return)
        trades.append(trade_return)
        equity_curve[-1] = equity

    if not equity_curve:
        equity_curve = [1.0]

    total_return = equity_curve[-1] - 1.0
    max_drawdown = _max_drawdown(equity_curve)
    sharpe = _sharpe(trades)
    win_rate = sum(1 for trade in trades if trade > 0) / len(trades) if trades else 0.0
    trade_count = len(trades)
    score = _score_strategy(name, regime, total_return, max_drawdown, sharpe, win_rate, trade_count)

    return StrategyResult(
        name=name,
        equity_curve=equity_curve,
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        win_rate=win_rate,
        trade_count=trade_count,
        score=score,
    )


def _build_state(bars: list[Bar]) -> dict[str, list[float]]:
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]

    sma_fast = [_mean(closes[max(0, i - 9) : i + 1]) for i in range(len(closes))]
    sma_slow = [_mean(closes[max(0, i - 29) : i + 1]) for i in range(len(closes))]
    rsi_values = [_rolling_rsi(closes[: i + 1]) for i in range(len(closes))]
    lower_band = [_bollinger_lower(closes[: i + 1]) for i in range(len(closes))]
    rolling_high = [max(highs[max(0, i - 19) : i + 1]) for i in range(len(highs))]
    avg_volume = [_mean(volumes[max(0, i - 19) : i + 1]) for i in range(len(volumes))]

    return {
        "close": closes,
        "volume": volumes,
        "high": highs,
        "low": lows,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "rsi": rsi_values,
        "lower_band": lower_band,
        "rolling_high": rolling_high,
        "avg_volume": avg_volume,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rolling_rsi(values: list[float]) -> float:
    if len(values) < 2:
        return 50.0
    gains = 0.0
    losses = 0.0
    window = values[-15:]
    for previous, current in zip(window[:-1], window[1:]):
        change = current - previous
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger_lower(values: list[float]) -> float:
    window = values[-20:]
    mean = _mean(window)
    variance = sum((value - mean) ** 2 for value in window) / len(window)
    deviation = variance ** 0.5
    return mean - 2.0 * deviation


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = (peak - value) / peak if peak else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd


def _sharpe(trades: list[float]) -> float:
    if len(trades) < 2:
        return 0.0
    mean = sum(trades) / len(trades)
    variance = sum((trade - mean) ** 2 for trade in trades) / (len(trades) - 1)
    if variance == 0:
        return 0.0
    return (mean / variance**0.5) * (len(trades) ** 0.5)


def _score_strategy(
    name: str,
    regime: str,
    total_return: float,
    max_drawdown: float,
    sharpe: float,
    win_rate: float,
    trade_count: int,
) -> float:
    regime_bonus = {
        "trend": {"trend_following": 0.25, "breakout": 0.18, "mean_reversion": -0.1},
        "mean_reversion": {"mean_reversion": 0.25, "trend_following": -0.08, "breakout": -0.05},
        "breakout": {"breakout": 0.28, "trend_following": 0.12, "mean_reversion": -0.12},
        "high_volatility": {"breakout": 0.1, "trend_following": 0.05, "mean_reversion": -0.05},
    }
    adjustment = regime_bonus.get(regime, {}).get(name, 0.0)
    activity_bonus = min(trade_count / 10.0, 1.0) * 0.05
    return (
        total_return * 2.0
        + sharpe * 0.35
        + win_rate * 0.25
        - max_drawdown * 1.8
        + adjustment
        + activity_bonus
    )
