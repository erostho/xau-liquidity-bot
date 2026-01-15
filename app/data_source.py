# app/data_source.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import time
import requests

from app.pro_analysis import Candle  # dùng đúng Candle của bạn

# =========================
# In-memory MT5 candle store
# =========================
# MT5_STORE[symbol_key][tf_key] = {"candles": List[Candle], "ts": epoch_seconds}
MT5_STORE: Dict[str, Dict[str, Dict[str, object]]] = {}

# =========================
# Symbol normalization + alias
# =========================
def norm_symbol(s: str) -> str:
    # "XAU/USD" -> "XAUUSD", "xauusdm" -> "XAUUSDM"
    return (s or "").replace("/", "").replace("-", "").replace(" ", "").upper()

def norm_tf(tf: str) -> str:
    t = (tf or "").strip().upper()
    # Accept: "15min", "M15", "1h", "H1"
    if t in ["15MIN", "M15"]:
        return "M15"
    if t in ["1H", "H1", "60MIN", "60MINUTES"]:
        return "H1"
    return t

# Map chat symbols -> MT5 symbols (bạn chỉnh theo MarketWatch Exness)
MT5_SYMBOL_MAP = {
    "XAU/USD": os.getenv("MT5_SYMBOL_XAU", "XAUUSDm"),
    "BTC/USD": os.getenv("MT5_SYMBOL_BTC", "BTCUSDm"),
}

# =========================
# TwelveData fallback
# =========================
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

def fetch_twelvedata_candles(symbol: str, interval: str, outputsize: int = 220) -> List[Candle]:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,   # "15min" / "1h"
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

    # newest first -> reverse -> oldest -> newest
    values = list(reversed(values))

    candles: List[Candle] = []
    for i, v in enumerate(values):
        candles.append(
            Candle(
                ts=i,
                open=float(v["open"]),
                high=float(v["high"]),
                low=float(v["low"]),
                close=float(v["close"]),
                volume=float(v.get("volume", 0.0) or 0.0),
            )
        )
    return candles

# =========================
# MT5 push receiver helpers
# =========================
def store_mt5_candles(symbol: str, tf: str, candles: List[dict]) -> int:
    """
    candles: list of dict {ts, open, high, low, close, volume?}
    Return number of candles stored.
    """
    sym_key = norm_symbol(symbol)
    tf_key = norm_tf(tf)

    parsed: List[Candle] = []
    for c in candles:
        parsed.append(
            Candle(
                ts=int(c.get("ts", 0)),
                open=float(c.get("open")),
                high=float(c.get("high")),
                low=float(c.get("low")),
                close=float(c.get("close")),
                volume=float(c.get("volume", 0.0) or 0.0),
            )
        )

    # ensure oldest->newest
    parsed.sort(key=lambda x: x.ts)

    if sym_key not in MT5_STORE:
        MT5_STORE[sym_key] = {}
    MT5_STORE[sym_key][tf_key] = {
        "candles": parsed,
        "ts": time.time(),
    }
    return len(parsed)

def mt5_available(symbol: str, tf: str, min_candles: int = 60, max_age_sec: int = 15 * 60) -> bool:
    sym_key = norm_symbol(symbol)
    tf_key = norm_tf(tf)
    slot = MT5_STORE.get(sym_key, {}).get(tf_key)
    if not slot:
        return False
    age = time.time() - float(slot.get("ts", 0))
    if age > max_age_sec:
        return False
    candles = slot.get("candles") or []
    return isinstance(candles, list) and len(candles) >= min_candles

def get_mt5_candles_for_chat_symbol(chat_symbol: str, tf: str, limit: int) -> Optional[List[Candle]]:
    """
    chat_symbol: "XAU/USD" or "BTC/USD" (the thing your analyzer uses)
    We map to MT5 symbol like XAUUSDm then read from MT5_STORE.
    """
    tf_key = norm_tf(tf)

    mt5_sym = MT5_SYMBOL_MAP.get(chat_symbol, chat_symbol)
    mt5_key = norm_symbol(mt5_sym)

    slot = MT5_STORE.get(mt5_key, {}).get(tf_key)
    if not slot:
        return None

    candles: List[Candle] = slot.get("candles") or []
    if not candles:
        return None

    # return last limit candles
    return candles[-limit:] if len(candles) > limit else candles

def get_candles(chat_symbol: str, tf: str, limit: int) -> Tuple[List[Candle], str]:
    """
    Returns (candles, source_name)
    source_name: "MT5" or "TWELVEDATA_FALLBACK"
    """
    # 1) try MT5
    mt5 = get_mt5_candles_for_chat_symbol(chat_symbol, tf, limit)
    if mt5 and len(mt5) >= 60:
        return mt5, "MT5"

    # 2) fallback TwelveData
    interval = "15min" if norm_tf(tf) == "M15" else "1h"
    td = fetch_twelvedata_candles(chat_symbol, interval, outputsize=max(limit, 220))
    td = td[-limit:] if len(td) > limit else td
    return td, "TWELVEDATA_FALLBACK"
