from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import math
import random
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _seed(symbol: str, timeframe: str) -> int:
    digest = hashlib.sha256(f"{symbol}:{timeframe}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_bars(symbol: str, timeframe: str, csv_path: str | None = None, count: int = 260) -> list[Bar]:
    """Load OHLCV data from CSV, or generate deterministic demo data."""
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV path not found: {path}")
        return list(_load_csv(path))

    default_path = Path("data") / f"{symbol.upper()}_{timeframe}.csv"
    if default_path.exists():
        return list(_load_csv(default_path))

    return _generate_demo_bars(symbol, timeframe, count=count)


def _load_csv(path: Path) -> Iterable[Bar]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield Bar(
                timestamp=_parse_timestamp(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )


def _parse_timestamp(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _generate_demo_bars(symbol: str, timeframe: str, count: int) -> list[Bar]:
    rng = random.Random(_seed(symbol, timeframe))
    start = datetime.now(timezone.utc) - timedelta(days=count)
    price = 100.0 + rng.uniform(-8.0, 8.0)
    drift = rng.uniform(-0.04, 0.08)
    volatility = rng.uniform(0.8, 2.4)
    bars: list[Bar] = []

    for index in range(count):
        timestamp = start + timedelta(days=index)
        cycle = math.sin(index / 16.0) * 0.8
        noise = rng.gauss(0.0, volatility)
        change = drift + cycle + noise * 0.15
        open_price = price
        close = max(1.0, open_price * (1.0 + change / 100.0))
        spread = abs(rng.gauss(0.0, volatility * 0.6)) + 0.25
        high = max(open_price, close) * (1.0 + spread / 200.0)
        low = min(open_price, close) * (1.0 - spread / 220.0)
        volume = max(100_000.0, 1_000_000.0 * (1.0 + rng.random() * 0.7))
        bars.append(
            Bar(
                timestamp=timestamp,
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=round(volume, 0),
            )
        )
        price = close

    return bars
