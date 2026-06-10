from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .agent import recommend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze chart structure and rank trading strategies.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to analyze.")
    parser.add_argument("--timeframe", default="1d", help="Timeframe label, such as 1d or 1h.")
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV file with OHLCV bars.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    recommendation, strategies, features = recommend(
        symbol=args.symbol,
        timeframe=args.timeframe,
        csv_path=args.csv_path,
    )

    if args.json:
        output = {
            "recommendation": asdict(recommendation),
            "strategies": [asdict(strategy) for strategy in strategies],
            "features": asdict(features),
        }
        print(json.dumps(output, indent=2, default=str))
        return 0

    print(f"Symbol: {recommendation.symbol}")
    print(f"Timeframe: {recommendation.timeframe}")
    print(f"Regime: {recommendation.regime}")
    print(f"Top strategy: {recommendation.strategy}")
    print(f"Confidence: {recommendation.confidence:.2f}")
    print("Rationale:")
    for item in recommendation.rationale:
        print(f"- {item}")
    print("Risks:")
    for item in recommendation.risks:
        print(f"- {item}")
    print("Strategy ranking:")
    for strategy in strategies:
        print(
            f"- {strategy.name}: score={strategy.score:.3f}, return={strategy.total_return:.3f}, "
            f"drawdown={strategy.max_drawdown:.3f}, sharpe={strategy.sharpe:.3f}, trades={strategy.trade_count}"
        )
    return 0
