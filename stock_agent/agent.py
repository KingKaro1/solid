from __future__ import annotations

from dataclasses import dataclass

from .data import Bar, load_bars
from .indicators import FeatureSet, compute_features
from .strategies import StrategyResult, evaluate_strategies


@dataclass(frozen=True)
class Recommendation:
    symbol: str
    timeframe: str
    regime: str
    strategy: str
    confidence: float
    rationale: list[str]
    risks: list[str]
    metrics: dict[str, float]


def classify_regime(features: FeatureSet) -> str:
    trend_strength = features.sma_fast - features.sma_slow
    volatility_ratio = features.atr / features.close if features.close else 0.0
    breakout_signal = features.close >= features.rolling_high * 0.995 and features.volume_ratio >= 1.05

    if volatility_ratio >= 0.035:
        return "high_volatility"
    if breakout_signal:
        return "breakout"
    if trend_strength > 0 and features.trend_slope > 0 and features.rsi >= 55:
        return "trend"
    if trend_strength < 0 and features.rsi <= 45:
        return "mean_reversion"
    if abs(features.price_vs_sma) < 0.015 and features.rsi < 55:
        return "mean_reversion"
    return "trend"


def recommend(
    symbol: str,
    timeframe: str,
    csv_path: str | None = None,
    bars: list[Bar] | None = None,
) -> tuple[Recommendation, list[StrategyResult], FeatureSet]:
    market_bars = bars if bars is not None else load_bars(symbol, timeframe, csv_path=csv_path)
    features = compute_features(market_bars)
    regime = classify_regime(features)
    results = evaluate_strategies(market_bars, features, regime)
    best = results[0]

    confidence = _confidence(best, results)
    rationale = _build_rationale(best, regime, features)
    risks = _build_risks(best, features)

    recommendation = Recommendation(
        symbol=symbol.upper(),
        timeframe=timeframe,
        regime=regime,
        strategy=best.name,
        confidence=confidence,
        rationale=rationale,
        risks=risks,
        metrics={
            "return": best.total_return,
            "max_drawdown": best.max_drawdown,
            "sharpe": best.sharpe,
            "win_rate": best.win_rate,
            "trade_count": float(best.trade_count),
            "score": best.score,
        },
    )
    return recommendation, results, features


def _confidence(best: StrategyResult, results: list[StrategyResult]) -> float:
    if len(results) == 1:
        return 0.5
    runner_up = results[1]
    gap = max(0.0, best.score - runner_up.score)
    base = 0.45 + min(gap * 0.4, 0.3)
    if best.trade_count < 3:
        base -= 0.15
    if best.max_drawdown > 0.2:
        base -= 0.1
    return max(0.0, min(1.0, base))


def _build_rationale(best: StrategyResult, regime: str, features: FeatureSet) -> list[str]:
    rationale = [
        f"Detected regime: {regime}",
        f"Best-ranked strategy: {best.name}",
        f"Recent price sits at {features.price_vs_sma * 100:.2f}% vs slow average",
        f"Latest volume is {features.volume_ratio:.2f}x its 20-bar average",
    ]
    if best.trade_count == 0:
        rationale.append("Strategy is favored by the score model even though no trades were triggered in the sample")
    return rationale


def _build_risks(best: StrategyResult, features: FeatureSet) -> list[str]:
    risks = [
        "Historical performance is a proxy, not a guarantee",
        "Regime shifts can invalidate chart-based setups quickly",
    ]
    if features.atr / features.close > 0.03:
        risks.append("Volatility is elevated, so position sizing should be conservative")
    if best.max_drawdown > 0.15:
        risks.append("Drawdown profile is material and should be stress tested further")
    if best.trade_count < 3:
        risks.append("Few trades were generated, so the estimate is statistically weak")
    return risks
