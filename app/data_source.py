# app/data_source.py
import os
import time
import logging
import requests
from typing import Dict, Any, List, Tuple, Optional
from app.pro_analysis import Candle

logger = logging.getLogger("uvicorn.error")

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

# MT5 cache in-memory (Render process memory)
# key: (symbol, tf) -> {"candles": [..], "ts": unix_time}
_MT5_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

# MT5 symbol mapping (so XAU/USD can find XAUUSDm, etc.)
MT5_SYMBOL_XAU = os.getenv("MT5_SYMBOL_XAU", "XAUUSDm")
MT5_SYMBOL_BTC = os.getenv("MT5_SYMBOL_BTC", "BTCUSDm")

# How "fresh" MT5 data must be to be trusted (seconds)
MT5_MAX_AGE_SEC = int(os.getenv("MT5_MAX_AGE_SEC", "1200"))  # 20 minutes default

def _tf_seconds(tf: str) -> int:
    tf2 = _tf_alias(tf)
    if tf2 == "15min":
        return 15 * 60
    if tf2 == "5min":
        return 5 * 60
    if tf2 == "1h":
        return 60 * 60
    return 0

def _norm_symbol(s: str) -> str:
    return (s or "").strip()


def _symbol_variants(symbol: str) -> List[str]:
    """
    Try multiple forms so cache lookup works:
    - XAU/USD, BTC/USD (telegram)
    - XAUUSDm, BTCUSDm (MT5)
    - XAUUSD, BTCUSD
    - remove '/' etc.
    """
    s = _norm_symbol(symbol)
    out = [s]

    low = s.lower()
    if "xau" in low:
        out.append(MT5_SYMBOL_XAU)
        out.append("XAUUSD")
        out.append("XAUUSDm")
        out.append("XAU/USD")
    if "btc" in low:
        out.append(MT5_SYMBOL_BTC)
        out.append("BTCUSD")
        out.append("BTCUSDm")
        out.append("BTC/USD")

    if "/" in s:
        out.append(s.replace("/", ""))
        out.append(s.replace("/", "") + "m")

    # unique keep order
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _tf_alias(tf: str) -> str:
    t = (tf or "").strip().lower()
    # accept "15min", "15m", "m15"...
    if t in ["15", "15m", "m15", "15min", "15mins", "15minute", "15minutes"]:
        return "15min"
    if t in ["60", "1h", "h1", "60m", "60min", "1hour", "1hr"]:
        return "1h"
    if t in ["5", "5m", "m5", "5min"]:
        return "5min"
    return t


def ingest_mt5_candles(symbol: str, tf: str, candles: List[Dict[str, Any]]) -> int:
    """
    Called by FastAPI endpoint to store MT5 candles in memory.
    candles item format expected:
    [{"time": 173..., "open":..., "high":..., "low":..., "close":..., "tick_volume":...}, ...]
    or already with keys: ts/open/high/low/close/volume
    """
    sym = _norm_symbol(symbol)
    tf2 = _tf_alias(tf)

    parsed: List[Candle] = []
    for c in candles or []:
        # support both formats
        ts = int(c.get("time", c.get("ts", 0)))
        o = float(c.get("open"))
        h = float(c.get("high"))
        l = float(c.get("low"))
        cl = float(c.get("close"))
        vol = float(c.get("tick_volume", c.get("volume", 0.0)) or 0.0)
        parsed.append(Candle(ts=ts, open=o, high=h, low=l, close=cl, volume=vol))

    if not parsed:
        raise ValueError("No candles to ingest")

    _MT5_CACHE[(sym, tf2)] = {"candles": parsed, "ts": int(time.time())}

    logger.info(f"[MT5] Received {len(parsed)} candles {sym} {tf2}")
    return len(parsed)


def _get_mt5_cached(symbol: str, tf: str, limit: int) -> Optional[List[Candle]]:
    tf2 = _tf_alias(tf)
    now = int(time.time())

    for sym in _symbol_variants(symbol):
        key = (sym, tf2)
        item = _MT5_CACHE.get(key)
        if not item:
            continue
        age = now - int(item.get("ts", 0))
        if age > MT5_MAX_AGE_SEC:
            continue

        candles = item.get("candles") or []
        if len(candles) < 20:
            continue

        # ✅ NEW: check cây nến cuối có quá cũ không
        tf_sec = _tf_seconds(tf2)
        last_ts = int(getattr(candles[-1], "ts", 0) or 0)

        # cho phép trễ ~ 2 cây nến
        if tf_sec and last_ts > 0:
            if now - last_ts > (2 * tf_sec + 60):
                continue

        return candles[-limit:]

    return None


def _fetch_twelvedata(symbol: str, interval: str, outputsize: int = 200) -> List[Candle]:
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"No candle data from TwelveData for {symbol} {interval}")

    values = list(reversed(values))  # oldest -> newest

    candles: List[Candle] = []
    for i, v in enumerate(values):
        candles.append(
            Candle(
                ts=i,
                open=float(v["open"]),
                high=float(v["high"]),
                low=float(v["low"]),
                close=float(v["close"]),
                volume=0.0,
            )
        )
    return candles


def get_candles(symbol: str, tf: str, limit: int = 220) -> Tuple[List[Candle], str]:
    """
    Returns candles + source_name
    Priority:
      1) MT5 cache if fresh
      2) TwelveData fallback
    """
    # 1) MT5
    mt5 = _get_mt5_cached(symbol, tf, limit)
    if mt5:
        return mt5, "MT5"

    # 2) TwelveData
    interval = _tf_alias(tf)
    td = _fetch_twelvedata(symbol, interval, limit)
    return td, "TWELVEDATA_FALLBACK"


def get_data_source(symbol="XAU/USD", tf="15min") -> str:
    mt5 = _get_mt5_cached(symbol, tf, 50)
    return "MT5" if mt5 else "TWELVEDATA_FALLBACK"

