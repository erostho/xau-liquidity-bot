# app/pro_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

Number = Union[int, float]

@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _get(c: Any, key: str, default: float = 0.0) -> float:
    """Support both dict candles and Candle objects."""
    if isinstance(c, dict):
        return _f(c.get(key, default), default)
    return _f(getattr(c, key, default), default)

def _ts(c: Any) -> int:
    if isinstance(c, dict):
        v = c.get("ts", c.get("time", 0))
    else:
        v = getattr(c, "ts", getattr(c, "time", 0))
    try:
        return int(v)
    except Exception:
        return 0

def _normalize(candles: Sequence[Any]) -> List[Candle]:
    out: List[Candle] = []
    for c in candles or []:
        out.append(
            Candle(
                ts=_ts(c),
                open=_get(c, "open"),
                high=_get(c, "high"),
                low=_get(c, "low"),
                close=_get(c, "close"),
                volume=_get(c, "volume", 0.0),
            )
        )
    out.sort(key=lambda x: x.ts)
    return out

def ema(values: Sequence[Number], period: int) -> List[float]:
    vals = [float(v) for v in values]
    if period <= 1 or len(vals) == 0:
        return vals[:]
    k = 2.0 / (period + 1.0)
    out: List[float] = []
    e = vals[0]
    for v in vals:
        e = v * k + e * (1.0 - k)
        out.append(e)
    return out

def atr(candles: Sequence[Any], period: int = 14) -> float:
    cs = _normalize(candles)
    if len(cs) < 2:
        return 0.0
    trs: List[float] = []
    prev_close = cs[0].close
    for c in cs[1:]:
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        prev_close = c.close
    if not trs:
        return 0.0
    # Wilder ATR (RMA)
    p = max(1, int(period))
    a = trs[0]
    alpha = 1.0 / p
    for tr in trs[1:]:
        a = a * (1.0 - alpha) + tr * alpha
    return float(a)

def rsi(candles: Sequence[Any], period: int = 14) -> float:
    cs = _normalize(candles)
    if len(cs) < period + 1:
        return 50.0
    closes = [c.close for c in cs]
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    # Wilder smoothing
    p = period
    avg_gain = sum(gains[:p]) / p
    avg_loss = sum(losses[:p]) / p
    for i in range(p, len(gains)):
        avg_gain = (avg_gain * (p - 1) + gains[i]) / p
        avg_loss = (avg_loss * (p - 1) + losses[i]) / p
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def trend_ema(candles: Sequence[Any], fast: int = 20, slow: int = 50) -> str:
    cs = _normalize(candles)
    if len(cs) < slow + 2:
        return "unknown"
    closes = [c.close for c in cs]
    ef = ema(closes, fast)[-1]
    es = ema(closes, slow)[-1]
    a = atr(cs, 14)
    # deadzone based on ATR to avoid flip
    dead = max(1e-9, 0.15 * a)
    if ef - es > dead:
        return "bullish"
    if es - ef > dead:
        return "bearish"
    return "sideways"

def _recent_range(candles: Sequence[Any], n: int = 30, exclude_last: bool = True) -> Tuple[float, float, int]:
    cs = _normalize(candles)
    if exclude_last and len(cs) > 0:
        cs = cs[:-1]
    cs = cs[-max(1, int(n)):]
    if not cs:
        return 0.0, 0.0, 0
    lo = min(c.low for c in cs)
    hi = max(c.high for c in cs)
    return float(lo), float(hi), len(cs)

def _is_higher_low(m15: Sequence[Any]) -> bool:
    cs = _normalize(m15)
    if len(cs) < 4:
        return False
    # use last 3 CLOSED candles
    a, b, c = cs[-4], cs[-3], cs[-2]
    return (b.low > a.low) and (c.low > b.low)

def _fmt_price(symbol: str, x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "..."
    # XAU tends to be 3 decimals on Exness, crypto 2
    if "XAU" in symbol.upper():
        return f"{x:.3f}"
    return f"{x:.2f}"

def build_short_hint(symbol: str, m15: Sequence[Any], m30: Sequence[Any], h1: Sequence[Any]) -> List[str]:
    m15c = _normalize(m15)
    if len(m15c) < 10:
        return ["- Chưa đủ dữ liệu M15 → CHỜ KÈO"]
    current_price = m15c[-1].close
    h1_trend = trend_ema(h1, 20, 50)
    m30_trend = trend_ema(m30, 20, 50)

    lo, hi, used = _recent_range(m15c, n=30, exclude_last=True)
    a15 = atr(m15c, 14)
    if used == 0 or hi <= lo:
        return ["- Chưa đủ dữ liệu để ra vùng → CHỜ KÈO"]

    rng = hi - lo
    # define "near-term" zones inside last 30 M15 candles (≈7.5h)
    buy_zone_lo = lo + 0.25 * rng
    buy_zone_hi = lo + 0.45 * rng
    sell_zone_lo = lo + 0.55 * rng
    sell_zone_hi = lo + 0.75 * rng

    # triggers & invalidation close-by (not xa lắc)
    buy_trigger = buy_zone_lo + 0.10 * max(a15, 1e-9)
    buy_invalid = lo - 0.15 * max(a15, 1e-9)
    sell_trigger = sell_zone_hi - 0.10 * max(a15, 1e-9)
    sell_invalid = hi + 0.15 * max(a15, 1e-9)

    lines: List[str] = []
    if h1_trend == "bullish":
        lines.append("- Ưu tiên BUY theo xu hướng H1.")
        lines.append(f"- Vùng quan sát BUY (8h gần nhất, M15): {_fmt_price(symbol, buy_zone_lo)} – {_fmt_price(symbol, buy_zone_hi)}.")
        lines.append(f"- BUY khi M15 tạo higher-low và đóng trên {_fmt_price(symbol, buy_trigger)}.")
        lines.append(f"- Nếu M15 đóng dưới {_fmt_price(symbol, buy_invalid)} → bỏ kèo, chờ cấu trúc mới.")
        if m30_trend == "bearish":
            lines.append("- ⚠️ M30 đang bearish → giảm khối lượng / chờ M30 confirm rồi mới BUY.")
    elif h1_trend == "bearish":
        lines.append("- Ưu tiên SELL theo xu hướng H1.")
        lines.append(f"- Vùng quan sát SELL (8h gần nhất, M15): {_fmt_price(symbol, sell_zone_lo)} – {_fmt_price(symbol, sell_zone_hi)}.")
        lines.append(f"- SELL khi M15 tạo lower-high và đóng dưới {_fmt_price(symbol, sell_trigger)}.")
        lines.append(f"- Nếu M15 đóng trên {_fmt_price(symbol, sell_invalid)} → bỏ kèo, chờ cấu trúc mới.")
        if m30_trend == "bullish":
            lines.append("- ⚠️ M30 đang bullish → giảm khối lượng / chờ M30 confirm rồi mới SELL.")
    else:
        lines.append("- H1 sideways → ưu tiên CHỜ hoặc đánh nhanh theo range.")
        lines.append(f"- Range 8h (M15): {_fmt_price(symbol, lo)} – {_fmt_price(symbol, hi)}.")
        lines.append(f"- Nếu M15 đóng > {_fmt_price(symbol, lo + 0.70*rng)} → canh BUY; nếu đóng < {_fmt_price(symbol, lo + 0.30*rng)} → canh SELL; ở giữa → CHỜ.")

    # Make sure hint is not "xa lắc" vs current price:
    # if suggested trigger is too far (> 2.5 ATR) from current price, tighten using current price +/- ATR band.
    tightened: List[str] = []
    max_far = 2.5 * max(a15, 1e-9)
    for ln in lines:
        tightened.append(ln)
    # no extra tightening text; the zone is already bounded by recent range
    return tightened

def analyze_pro(symbol: str, m15: Sequence[Any], m30: Sequence[Any], h1: Sequence[Any]) -> Dict[str, Any]:
    m15c = _normalize(m15)
    m30c = _normalize(m30)
    h1c = _normalize(h1)

    if len(m15c) < 50 or len(h1c) < 60:
        return {
            "symbol": symbol,
            "recommendation": "CHỜ",
            "stars": 1,
            "reason": "Chưa đủ dữ liệu nến",
            "short_hint": ["- Chưa đủ dữ liệu → CHỜ KÈO"],
        }

    h1_tr = trend_ema(h1c, 20, 50)
    m30_tr = trend_ema(m30c, 20, 50) if len(m30c) >= 60 else "unknown"
    rsi15 = rsi(m15c, 14)
    atr15 = atr(m15c, 14)

    # Simple quality score
    stars = 1
    if h1_tr in ("bullish", "bearish"):
        stars += 1
    if m30_tr == h1_tr:
        stars += 1
    if atr15 > 0:
        stars += 0  # keep for future
    stars = int(max(1, min(5, stars)))

    recommendation = "CHỜ"
    if stars >= 3:
        recommendation = "BUY" if h1_tr == "bullish" else ("SELL" if h1_tr == "bearish" else "CHỜ")

    out: Dict[str, Any] = {
        "symbol": symbol,
        "tf": "M30",
        "h1_trend": h1_tr,
        "m30_trend": m30_tr,
        "rsi15": float(rsi15),
        "atr15": float(atr15),
        "recommendation": recommendation,
        "stars": stars,
        "short_hint": build_short_hint(symbol, m15c, m30c, h1c),
    }
    return out

def format_signal(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "XAUUSD")
    tf = sig.get("tf", "M15")
    session = sig.get("session", "Phiên Mỹ")

    context_lines = sig.get("context_lines", [])
    position_lines = sig.get("position_lines", [])
    short_hint = sig.get("short_hint", [])
    if isinstance(short_hint, str):
        short_hint = [short_hint]
    liquidity_lines = sig.get("liquidity_lines", [])
    quality_lines = sig.get("quality_lines", [])

    rec = sig.get("recommendation", "CHỜ")
    stars = int(sig.get("stars", 1))
    stars_txt = "⭐️" * max(1, min(5, stars))

    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")

    notes = sig.get("notes", [])
    levels = sig.get("levels", [])
    levels_info = sig.get("levels_info", [])
    observation = sig.get("observation", {})

    def nf(x):
        if x is None:
            return "..."
        try:
            x = float(x)
            return f"{x:.3f}".rstrip("0").rstrip(".")
        except Exception:
            return "..."

    lines: List[str] = []
    lines.append(f"📊 {symbol} | {tf} | {session}")
    lines.append("TF: Signal=M15 | Entry=M30 | Confirm=H1")
    lines.append("")
    lines.append("Context:")
    for s in context_lines:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("GỢI Ý NGẮN HẠN:")
    for ln in sig.get("short_hint", []) or []:
        lines.append(ln)
    lines.append("")
    lines.append("Thanh khoản:")
    for s in liquidity_lines:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("Chất lượng setup:")
    for s in quality_lines:
        lines.append(f"- {s}")
    lines.append("")
    lines.append(f"🎯 Khuyến nghị: {rec}")
    lines.append(f"Độ tin cậy: {stars_txt} ({max(1, min(5, stars))}/5)")
    lines.append("")
    lines.append(f"ENTRY: {nf(entry)}")
    lines.append(f"SL: {nf(sl)} | TP1: {nf(tp1)} | TP2: {nf(tp2)}")
    lines.append("")
    lines.append("⚠️ Lưu ý:")
    if notes:
        for s in notes:
            lines.append(f"- {s}")
    else:
        lines.append("- Luôn chờ nến xác nhận.")
    lines.append("")
    lines.append("Mốc giá quan trọng:")
    if levels_info:
        for price, label in levels_info[:8]:
            lines.append(f"- {nf(price)} — {label}")
    elif levels:
        for lv in levels[:6]:
            lines.append(f"- {nf(lv)}")
    else:
        lines.append("- (chưa có mốc)")

    # Extra hint below levels: what M15 close would trigger
    try:
        b = observation.get("buy")
        s = observation.get("sell")
        buf = float(observation.get("buffer", 0.4))
        tf_obs = observation.get("tf", "M15")
        if b is not None and s is not None:
            lines.append("")
            lines.append("Gợi ý quan sát vào lệnh:")
            lines.append(f"- Nếu {tf_obs} đóng > {nf(float(b)+buf)} → ưu tiên canh BUY (theo H1 + chờ M30 confirm)")
            lines.append(f"- Nếu {tf_obs} đóng < {nf(float(s)-buf)} → ưu tiên canh SELL (theo H1 + chờ M30 confirm)")
            lines.append(f"- Nếu đóng giữa 2 mốc → CHỜ KÈO")
    except Exception:
        pass

    return "\n".join(lines)

