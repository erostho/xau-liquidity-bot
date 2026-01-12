from __future__ import annotations
from typing import List
import requests
from .base import MarketDataProvider, Candle
from ..config import TWELVEDATA_API_KEY

_TIMEFRAME_MAP = {
    "1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min",
    "45m":"45min","1h":"1h","2h":"2h","4h":"4h","1d":"1day"
}

class TwelveDataProvider(MarketDataProvider):
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 220) -> List[Candle]:
        if not TWELVEDATA_API_KEY:
            raise RuntimeError("TWELVEDATA_API_KEY missing. Set env or switch DATA_PROVIDER=mock.")
        interval = _TIMEFRAME_MAP.get(timeframe.lower(), timeframe)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": str(limit),
            "apikey": TWELVEDATA_API_KEY,
            "format": "JSON"
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "values" not in data:
            raise RuntimeError(f"TwelveData error: {data}")
        out: List[Candle] = []
        for row in reversed(data["values"]):  # oldest-first
            out.append({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume") or 0.0),
                "ts": row.get("datetime")
            })
        return out
