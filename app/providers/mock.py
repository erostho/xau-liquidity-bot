from __future__ import annotations
from typing import List
import random, time
from .base import MarketDataProvider, Candle

class MockProvider(MarketDataProvider):
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 220) -> List[Candle]:
        base = 4600.0
        candles: List[Candle] = []
        price = base + random.uniform(-25, 25)
        for i in range(limit):
            o = price
            drift = random.uniform(-4, 4)
            c = o + drift
            h = max(o, c) + random.uniform(0, 3)
            l = min(o, c) - random.uniform(0, 3)
            v = random.uniform(100, 1500)
            candles.append({"open": o, "high": h, "low": l, "close": c, "volume": v, "ts": time.time() - (limit-i)*60})
            price = c
        return candles
