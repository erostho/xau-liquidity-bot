from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict

Candle = Dict[str, float]  # open, high, low, close, volume(optional), ts(optional)

class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 220) -> List[Candle]:
        raise NotImplementedError
