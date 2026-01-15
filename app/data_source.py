import os
import json
import time
import requests
from typing import List, Tuple, Optional
from dataclasses import dataclass

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MT5_DATA_DIR = os.path.join(BASE_DIR, "data", "mt5")

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

# =========================
# DATA MODEL
# =========================

@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# =========================
# MT5 (PUSH DATA)
# =========================

def _mt5_file(symbol: str, tf: str) -> str:
    fname = f"{symbol}_{tf}.json"
    return os.path.join(MT5_DATA_DIR, fname)


def load_mt5_candles(symbol: str, tf: str, limit: int) -> Optional[List[Candle]]:
    """
    Đọc dữ liệu MT5 đã được push từ PC lên Render
    """
    path = _mt5_file(symbol, tf)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not raw or len(raw) < limit:
            return None

        candles = [
            Candle(
                ts=int(c["ts"]),
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c.get("volume", 0)),
            )
            for c in raw[-limit:]
        ]

        return candles

    except Exception as e:
        print(f"[MT5] load error {symbol} {tf}: {e}")
        return None


# =========================
# TWELVEDATA (FALLBACK)
# =========================

_TD_TF_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1day",
}


def load_twelvedata(symbol: str, tf: str, limit: int) -> List[Candle]:
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    interval = _TD_TF_MAP.get(tf)
    if not interval:
        raise ValueError(f"Unsupported TF {tf}")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": limit,
        "apikey": TWELVEDATA_API_KEY,
    }

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"TwelveData error: {data}")

    candles = []
    for v in reversed(data["values"]):
        candles.append(
            Candle(
                ts=int(time.mktime(time.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S"))),
                open=float(v["open"]),
                high=float(v["high"]),
                low=float(v["low"]),
                close=float(v["close"]),
                volume=float(v.get("volume", 0)),
            )
        )

    return candles


# =========================
# MAIN API (DUY NHẤT DÙNG)
# =========================

def get_candles(
    symbol: str,
    tf: str,
    limit: int,
) -> Tuple[List[Candle], str]:
    """
    Ưu tiên MT5 → fallback TwelveData
    Trả về (candles, source)
    """

    mt5 = load_mt5_candles(symbol, tf, limit)
    if mt5:
        return mt5, "MT5"

    td = load_twelvedata(symbol, tf, limit)
    return td, "TWELVEDATA_FALLBACK"
