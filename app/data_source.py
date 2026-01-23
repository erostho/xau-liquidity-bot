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
_TD_CACHE = {}  # key -> (ts_minute, candles)
# MT5 symbol mapping (so XAU/USD can find XAUUSDm, etc.)
MT5_SYMBOL_XAU = os.getenv("MT5_SYMBOL_XAU", "XAUUSDm")
MT5_SYMBOL_BTC = os.getenv("MT5_SYMBOL_BTC", "BTCUSDm")
MT5_SYMBOL_XAG = os.getenv("MT5_SYMBOL_XAG", "XAGUSDm")
# How "fresh" MT5 data must be to be trusted (seconds)
MT5_MAX_AGE_SEC = int(os.getenv("MT5_MAX_AGE_SEC", "1200"))  # 20 minutes default

def _tf_seconds(tf: str) -> int:
    tf2 = _tf_alias(tf)
    if tf2 == "5min":
        return 5 * 60
    if tf2 == "15min":
        return 15 * 60
    if tf2 == "30min":
        return 30 * 60
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
    if "xag" in low:
        out.append(MT5_SYMBOL_XAG)
        out.append("XAGUSD")
        out.append("XAGUSDm")
        out.append("XAG/USD")
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

    # 30m / m30
    if t in ["30", "30m", "m30", "30min", "30mins", "30minute", "30minutes"]:
        return "30min"

    if t in ["15", "15m", "m15", "15min", "15mins", "15minute", "15minutes"]:
        return "15min"
    if t in ["60", "1h", "h1", "60m", "60min", "1hour", "1hr"]:
        return "1h"
    if t in ["5", "5m", "m5", "5min"]:
        return "5min"
    return

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
        ts = int(c.get("time", c.get("ts", 0)) or 0)
        # FIX: nếu MT5 gửi millisecond → đổi về second
        if ts > 10_000_000_000:   # > year 2286 (seconds) => chắc chắn là ms
            ts = ts // 1000
        o = float(c.get("open"))
        h = float(c.get("high"))
        l = float(c.get("low"))
        cl = float(c.get("close"))
        vol = float(c.get("tick_volume", c.get("volume", 0.0)) or 0.0)
        parsed.append(Candle(ts=ts, open=o, high=h, low=l, close=cl, volume=vol))
    if not parsed:
        raise ValueError("No candles to ingest")
    now_ts = int(time.time())
    for symv in _symbol_variants(sym):
        _MT5_CACHE[(symv, tf2)] = {"candles": parsed, "ts": now_ts}
    logger.info(f"[MT5] Received {len(parsed)} candles {sym} {tf2}")
    logger.info(f"[MT5][CACHE_SET] sym={sym} tf2={tf2} first_ts={parsed[0].ts} last_ts={parsed[-1].ts} n={len(parsed)}")
    return len(parsed)
def _get_mt5_cached(symbol: str, tf: str, limit: int):
    tf2 = _tf_alias(tf)
    now = int(time.time())
    logger.info(f"[MT5][LOOKUP] symbol={symbol} tf={tf} alias={tf2} variants={_symbol_variants(symbol)} now={now}")

    for sym in _symbol_variants(symbol):
        key = (sym, tf2)
        item = _MT5_CACHE.get(key)
        if not item:
            logger.info(f"[MT5][MISS] key={key}")
            continue

        age = now - int(item.get("ts", 0))
        if age > MT5_MAX_AGE_SEC:
            logger.info(f"[MT5][STALE_CACHE] key={key} age={age}s > {MT5_MAX_AGE_SEC}")
            continue

        candles = item.get("candles") or []
        if len(candles) < 20:
            logger.info(f"[MT5][TOO_FEW] key={key} n={len(candles)}")
            continue

        tf_sec = _tf_seconds(tf2)
        last_ts = int(getattr(candles[-1], "ts", 0) or 0)
        if tf_sec and last_ts > 0:
            if now - last_ts > (2 * tf_sec + 60):
                logger.info(f"[MT5][LAST_CANDLE_OLD] key={key} now-last_ts={now-last_ts}s tf_sec={tf_sec}")
                continue

        logger.info(f"[MT5][HIT] key={key} n={len(candles)} last_ts={last_ts} age={age}s")
        return candles[-limit:]

    return None


def _fetch_twelvedata(symbol: str, interval: str, outputsize: int = 200) -> List[Candle]:
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("Missing TWELVEDATA_API_KEY")
    minute = int(time.time() // 60)
    key = (symbol, interval, int(outputsize))
    
    cached = _TD_CACHE.get(key)
    if cached and cached[0] == minute:
        return cached[1]

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
        msg = (data.get("message") or "").lower()
        # TwelveData rate limit/credits: don't crash the whole analysis
        if "out of api credits" in msg or "limit" in msg or "too many" in msg:
            logger.warning(f"[twelvedata] rate-limit/credits: {data.get('message')}")
            return None
        
        # other errors: still don't crash hard, just return None to allow other sources
        logger.warning(f"[twelvedata] error: {data.get('message')}")
        return None


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
    _TD_CACHE[key] = (minute, candles)
    return candles


def get_candles(symbol: str, tf: str, limit: int = 220) -> Tuple[List[Candle], str]:
    """
    Returns candles + source_name
    Priority:
      1) MT5 cache if fresh
      2) TwelveData fallback
    """
    # 1) MT5
    USE_TWELVEDATA = os.getenv("USE_TWELVEDATA", "1") == "1"
    mt5 = _get_mt5_cached(symbol, tf, limit)
    if mt5:
        return mt5, "MT5"

    # 2) TwelveData
    interval = _tf_alias(tf)
    if not USE_TWELVEDATA:
        return None, "NO_TWELVEDATA"
    td = _fetch_twelvedata(symbol, interval, limit)
    return td, "TWELVEDATA_FALLBACK"


def get_data_source(symbol="XAU/USD", tf="15min") -> str:
    mt5 = _get_mt5_cached(symbol, tf, 50)
    return "MT5" if mt5 else "TWELVEDATA_FALLBACK"

