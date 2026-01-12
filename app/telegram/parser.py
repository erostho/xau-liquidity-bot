from __future__ import annotations
import re
from dataclasses import dataclass
from ..config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_RISK_USD

@dataclass
class UserQuery:
    symbol: str
    timeframe: str
    risk_usd: float

_TF_RE = re.compile(r"\b(1m|3m|5m|15m|30m|45m|1h|2h|4h|1d)\b", re.I)
_RISK_RE = re.compile(r"risk\s*=\s*(\d+(?:\.\d+)?)", re.I)

def parse_text(text: str) -> UserQuery:
    t = text.strip()
    symbol = DEFAULT_SYMBOL
    timeframe = DEFAULT_TIMEFRAME
    risk = DEFAULT_RISK_USD

    # Extract symbol if present (e.g., XAUUSD, BTCUSDT, XAU/USD)
    m = re.search(r"\b([A-Za-z]{3,10}(?:/[A-Za-z]{3,10})?)\b", t)
    if m:
        cand = m.group(1).upper()
        if cand not in {"NEN","BUY","SELL","HAY","PHAN","TICH","MARKET","NOW","RISK","ANALYZE"}:
            symbol = cand.replace("XAU/USD","XAUUSD")

    tfm = _TF_RE.search(t)
    if tfm:
        timeframe = tfm.group(1).lower()

    rm = _RISK_RE.search(t)
    if rm:
        risk = float(rm.group(1))

    return UserQuery(symbol=symbol, timeframe=timeframe, risk_usd=risk)
