from __future__ import annotations

import unittest

from stock_agent.agent import recommend


class RecommendationSmokeTest(unittest.TestCase):
    def test_recommendation_smoke(self) -> None:
        recommendation, strategies, features = recommend("AAPL", "1d")
        self.assertEqual(recommendation.symbol, "AAPL")
        self.assertTrue(recommendation.strategy)
        self.assertTrue(strategies)
        self.assertGreater(features.close, 0)


if __name__ == "__main__":
    unittest.main()
