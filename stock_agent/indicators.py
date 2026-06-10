from __future__ import annotations

from dataclasses import dataclass
import math

from .data import Bar


@dataclass(frozen=True)
class FeatureSet:
    close: float
    sma_fast: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    macd: float
    macd_signal: float
    macd_hist: float
    upper_band: float
    lower_band: float
    rolling_high: float
    rolling_low: float
    average_volume: float
    volume_ratio: float
    trend_slope: float
    price_vs_sma: float


def compute_features(bars: list[Bar]) -> FeatureSet:
    if len(bars) < 30:
        raise ValueError("Need at least 30 bars to compute features")

    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    volumes = [bar.volume for bar in bars]

    sma_fast = sma(closes, 10)
    sma_slow = sma(closes, 30)
    ema_fast = ema(closes, 12)
    ema_slow = ema(closes, 26)
    rsi_value = rsi(closes, 14)
    atr_value = atr(highs, lows, closes, 14)
    macd_line, macd_signal, macd_hist = macd(closes)
    upper_band, lower_band = bollinger_bands(closes, 20, 2.0)
    rolling_high = max(highs[-20:])
    rolling_low = min(lows[-20:])
    average_volume = sum(volumes[-20:]) / 20.0
    volume_ratio = volumes[-1] / average_volume if average_volume else 0.0
    trend_slope = _slope(closes[-20:])
    price_vs_sma = (closes[-1] - sma_slow) / sma_slow if sma_slow else 0.0

    return FeatureSet(
        close=closes[-1],
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi=rsi_value,
        atr=atr_value,
        macd=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        upper_band=upper_band,
        lower_band=lower_band,
        rolling_high=rolling_high,
        rolling_low=rolling_low,
        average_volume=average_volume,
        volume_ratio=volume_ratio,
        trend_slope=trend_slope,
        price_vs_sma=price_vs_sma,
    )


def sma(values: list[float], length: int) -> float:
    window = values[-length:]
    return sum(window) / len(window)


def ema(values: list[float], length: int) -> float:
    alpha = 2.0 / (length + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def rsi(values: list[float], length: int) -> float:
    gains = 0.0
    losses = 0.0
    for previous, current in zip(values[-length - 1 : -1], values[-length:]):
        change = current - previous
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: list[float], lows: list[float], closes: list[float], length: int) -> float:
    trs: list[float] = []
    for index in range(1, len(closes)):
        high = highs[index]
        low = lows[index]
        previous_close = closes[index - 1]
        trs.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return sma(trs, min(length, len(trs)))


def macd(values: list[float]) -> tuple[float, float, float]:
    fast = ema(values, 12)
    slow = ema(values, 26)
    line = fast - slow
    signal = line * 0.2
    hist = line - signal
    return line, signal, hist


def bollinger_bands(values: list[float], length: int, multiplier: float) -> tuple[float, float]:
    window = values[-length:]
    mean = sum(window) / len(window)
    variance = sum((value - mean) ** 2 for value in window) / len(window)
    deviation = math.sqrt(variance)
    return mean + multiplier * deviation, mean - multiplier * deviation


def _slope(values: list[float]) -> float:
    count = len(values)
    x_mean = (count - 1) / 2.0
    y_mean = sum(values) / count
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    if denominator == 0:
        return 0.0
    return numerator / denominator
