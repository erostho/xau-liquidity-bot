from typing import List, Tuple
import time
import requests

from app.pro_analysis import Candle

# =========================
# Abstract Data Source
# =========================
class MarketDataSource:
    name: str = "BASE"

    def is_online(self) -> bool:
        raise NotImplementedError

    def get_candles(self, symbol: str, tf: str, limit: int) -> List[Candle]:
        raise NotImplementedError


# =========================
# MT5 EXNESS DATA SOURCE
# =========================
class MT5DataSource(MarketDataSource):
    name = "EXNESS_MT5"

    def __init__(self):
        self._mt5 = None

    def _init_mt5(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            if not mt5.initialize():
                return False
            return True
        except Exception:
            return False

    def is_online(self) -> bool:
        return self._init_mt5()

    def _tf_map(self, tf: str):
        if tf in ["15m", "15min", "M15"]:
            return self._mt5.TIMEFRAME_M15
        if tf in ["1h", "H1"]:
            return self._mt5.TIMEFRAME_H1
        raise ValueError(f"Unsupported timeframe: {tf}")

    def get_candles(self, symbol: str, tf: str, limit: int) -> List[Candle]:
        mt5 = self._mt5
        tf_mt5 = self._tf_map(tf)

        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, limit)
        if rates is None or len(rates) == 0:
            raise RuntimeError("MT5 returned no data")

        candles = []
        for i, r in enumerate(rates):
            candles.append(
                Candle(
                    ts=int(r["time"]),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r.get("tick_volume", 0)),
                )
            )
        return candles


# =========================
# TWELVEDATA FALLBACK
# =========================
class TwelveDataSource(MarketDataSource):
    name = "TWELVEDATA"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def is_online(self) -> bool:
        return True  # fallback luôn available

    def _tf_map(self, tf: str):
        if tf in ["15m", "15min", "M15"]:
            return "15min"
        if tf in ["1h", "H1"]:
            return "1h"
        raise ValueError(f"Unsupported timeframe: {tf}")

    def get_candles(self, symbol: str, tf: str, limit: int) -> List[Candle]:
        interval = self._tf_map(tf)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": limit,
            "apikey": self.api_key,
            "format": "JSON",
        }

        r = requests.get(url, params=params, timeout=20)
        data = r.json()

        if "status" in data and data["status"] == "error":
            raise RuntimeError(data.get("message"))

        values = data.get("values", [])
        if not values:
            raise RuntimeError("TwelveData returned no data")

        # TwelveData newest first → reverse
        values = list(reversed(values))

        candles = []
        for i, v in enumerate(values):
            candles.append(
                Candle(
                    ts=i,
                    open=float(v["open"]),
                    high=float(v["high"]),
                    low=float(v["low"]),
                    close=float(v["close"]),
                )
            )
        return candles


# =========================
# AUTO SELECTOR
# =========================
def get_best_data_source(twelvedata_api_key: str) -> Tuple[MarketDataSource, str]:
    """
    Returns:
        (data_source, source_name)
    """
    mt5_src = MT5DataSource()
    if mt5_src.is_online():
        return mt5_src, mt5_src.name

    return TwelveDataSource(twelvedata_api_key), "TWELVEDATA_FALLBACK"
