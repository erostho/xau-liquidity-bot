# app/pro_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional
import math

Number = Union[int, float]


# ----------------------------
# Data model + helpers
# ----------------------------
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


def _ts(c: Any) -> int:
    if isinstance(c, dict):
        v = c.get("ts", c.get("time", 0))
    else:
        v = getattr(c, "ts", getattr(c, "time", 0))
    try:
        return int(v)
    except Exception:
        return 0


def _get(c: Any, key: str, default: float = 0.0) -> float:
    """Support dict candles, Candle objects, or objects with attributes."""
    if isinstance(c, dict):
        return _f(c.get(key, default), default)
    return _f(getattr(c, key, default), default)


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


def _fmt_price(symbol: str, x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "..."
    if "XAU" in (symbol or "").upper():
        return f"{float(x):.3f}"
    return f"{float(x):.2f}"


# ----------------------------
# Indicators
# ----------------------------
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

    p = int(period)
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
    dead = max(1e-9, 0.15 * a)  # anti-flip deadzone
    if ef - es > dead:
        return "bullish"
    if es - ef > dead:
        return "bearish"
    return "sideways"


# ----------------------------
# Short-hint logic (30 candles M15 ≈ 7.5-8h)
# ----------------------------
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
    if len(cs) < 5:
        return False
    # last 3 CLOSED candles
    a, b, c = cs[-4], cs[-3], cs[-2]
    return (b.low > a.low) and (c.low > b.low)


def _is_lower_high(m15: Sequence[Any]) -> bool:
    cs = _normalize(m15)
    if len(cs) < 5:
        return False
    a, b, c = cs[-4], cs[-3], cs[-2]
    return (b.high < a.high) and (c.high < b.high)


def build_short_hint(symbol: str, m15: Sequence[Any], m30: Sequence[Any], h1: Sequence[Any]) -> List[str]:
    m15c = _normalize(m15)
    if len(m15c) < 10:
        return ["- Chưa đủ dữ liệu M15 → CHỜ KÈO"]

    current_price = m15c[-1].close
    h1_trend = trend_ema(h1, 20, 50)
    m30_trend = trend_ema(m30, 20, 50) if len(_normalize(m30)) >= 60 else "unknown"

    lo, hi, used = _recent_range(m15c, n=30, exclude_last=True)
    a15 = atr(m15c, 14)

    if used < 10 or hi <= lo:
        return ["- Chưa đủ dữ liệu để ra vùng → CHỜ KÈO"]

    rng = hi - lo

    # Zones derived ONLY from the last 30 M15 candles (≈8h)
    # buy zone: lower-mid of range, sell zone: upper-mid of range
    buy_zone_lo = lo + 0.20 * rng
    buy_zone_hi = lo + 0.45 * rng
    sell_zone_lo = lo + 0.55 * rng
    sell_zone_hi = lo + 0.80 * rng

    # triggers & invalidation near-by (avoid xa lắc)
    buy_trigger = buy_zone_lo + 0.10 * max(a15, 1e-9)
    buy_invalid = lo - 0.20 * max(a15, 1e-9)
    sell_trigger = sell_zone_hi - 0.10 * max(a15, 1e-9)
    sell_invalid = hi + 0.20 * max(a15, 1e-9)

    # Clamp triggers near current price if too far
    max_far = 2.2 * max(a15, 1e-9)
    if abs(buy_trigger - current_price) > max_far:
        buy_trigger = current_price + 0.35 * a15
    if abs(sell_trigger - current_price) > max_far:
        sell_trigger = current_price - 0.35 * a15

    lines: List[str] = []

    if h1_trend == "bullish":
        lines.append("- Ưu tiên BUY theo xu hướng H1.")
        lines.append(f"- Vùng quan sát BUY (8h gần nhất, M15): {_fmt_price(symbol, buy_zone_lo)} – {_fmt_price(symbol, buy_zone_hi)}.")
        lines.append(f"- BUY khi M15 tạo higher-low và đóng trên {_fmt_price(symbol, buy_trigger)}.")
        lines.append(f"- Nếu M15 đóng dưới {_fmt_price(symbol, buy_invalid)} → bỏ kèo, chờ cấu trúc mới.")
        if m30_trend == "bearish":
            lines.append("- ⚠️ M30 đang bearish → chờ M30 confirm hoặc giảm khối lượng.")
    elif h1_trend == "bearish":
        lines.append("- Ưu tiên SELL theo xu hướng H1.")
        lines.append(f"- Vùng quan sát SELL (8h gần nhất, M15): {_fmt_price(symbol, sell_zone_lo)} – {_fmt_price(symbol, sell_zone_hi)}.")
        lines.append(f"- SELL khi M15 tạo lower-high và đóng dưới {_fmt_price(symbol, sell_trigger)}.")
        lines.append(f"- Nếu M15 đóng trên {_fmt_price(symbol, sell_invalid)} → bỏ kèo, chờ cấu trúc mới.")
        if m30_trend == "bullish":
            lines.append("- ⚠️ M30 đang bullish → chờ M30 confirm hoặc giảm khối lượng.")
    else:
        lines.append("- H1 sideways → ưu tiên CHỜ hoặc đánh nhanh theo range.")
        lines.append(f"- Range 8h (M15): {_fmt_price(symbol, lo)} – {_fmt_price(symbol, hi)}.")
        lines.append(f"- Nếu M15 đóng > {_fmt_price(symbol, lo + 0.70*rng)} → canh BUY; nếu đóng < {_fmt_price(symbol, lo + 0.30*rng)} → canh SELL; ở giữa → CHỜ.")

    return lines


# ----------------------------
# Main analysis (keeps your existing fields for Telegram FULL)
# ----------------------------
def _market_context(m15: Sequence[Any]) -> str:
    cs = _normalize(m15)
    if len(cs) < 20:
        return "SIDEWAY / HỒI NHẸ"
    a = atr(cs, 14)
    last = cs[-2].close  # last CLOSED candle
    prev = cs[-6].close
    move = abs(last - prev)
    if a > 0 and move > 2.2 * a:
        return "SPIKE → HỒI"
    return "SIDEWAY / HỒI NHẸ"


def _session_tag() -> str:
    # Keep simple (you can later map by timezone)
    return "Phiên Mỹ"


def _build_levels(symbol: str, m15: Sequence[Any], m30: Sequence[Any], h1: Sequence[Any]) -> List[Tuple[float, str]]:
    # Lightweight levels: recent swing-ish extremes (NOT too xa) + H1 extreme.
    m15c = _normalize(m15)
    m30c = _normalize(m30)
    h1c = _normalize(h1)
    levels: List[Tuple[float, str]] = []

    if len(h1c) >= 40:
        h1_hi = max(c.high for c in h1c[-80:])
        h1_lo = min(c.low for c in h1c[-80:])
        levels.append((h1_hi, "H1 High gần đây"))
        levels.append((h1_lo, "H1 Low gần đây"))

    if len(m30c) >= 30:
        m30_hi = max(c.high for c in m30c[-60:])
        m30_lo = min(c.low for c in m30c[-60:])
        levels.append((m30_hi, "M30 High gần đây"))
        levels.append((m30_lo, "M30 Low gần đây"))

    if len(m15c) >= 30:
        lo, hi, _ = _recent_range(m15c, n=30, exclude_last=True)
        levels.append((hi, "M15 High (8h)"))
        levels.append((lo, "M15 Low (8h)"))

    # Deduplicate by rounded price
    seen = set()
    out: List[Tuple[float, str]] = []
    for p, lbl in levels:
        key = round(float(p), 3 if "XAU" in symbol.upper() else 2)
        if key in seen:
            continue
        seen.add(key)
        out.append((float(p), lbl))
    return out[:8]


def analyze_pro(symbol: str, m15: Sequence[Any], m30: Sequence[Any], h1: Sequence[Any]) -> Dict[str, Any]:
    m15c = _normalize(m15)
    m30c = _normalize(m30)
    h1c = _normalize(h1)

    if len(m15c) < 60 or len(h1c) < 60:
        return {
            "symbol": symbol,
            "tf": "M30",
            "session": _session_tag(),
            "context_lines": ["Chưa đủ dữ liệu nến"],
            "liquidity_lines": ["Chưa đủ dữ liệu sweep/rejection"],
            "quality_lines": ["RSI/ATR chưa đủ"],
            "recommendation": "CHỜ",
            "stars": 1,
            "short_hint": ["- Chưa đủ dữ liệu → CHỜ KÈO"],
            "notes": ["Nguồn dữ liệu: EXNESS_MT5_PUSH hoặc TwelveData"],
        }

    h1_tr = trend_ema(h1c, 20, 50)
    m30_tr = trend_ema(m30c, 20, 50) if len(m30c) >= 60 else "unknown"
    rsi15 = rsi(m15c, 14)
    atr15 = atr(m15c, 14)

    # Stars (simple + stable)
    stars = 1
    if h1_tr in ("bullish", "bearish"):
        stars += 1
    if m30_tr != "unknown" and m30_tr == h1_tr:
        stars += 1
    stars = int(max(1, min(5, stars)))

    recommendation = "CHỜ"
    if stars >= 3:
        recommendation = "BUY" if h1_tr == "bullish" else ("SELL" if h1_tr == "bearish" else "CHỜ")

    # Telegram FULL content
    context_lines = [
        f"Thị trường: {_market_context(m15c)}",
        f"H1: {h1_tr} (EMA20 vs EMA50)",
    ]
    if m30_tr != "unknown":
        context_lines.append(f"M30: {m30_tr} (EMA20 vs EMA50)")

    liquidity_lines = [
        "Chưa thấy sweep/rejection rõ (liquidity proxy).",
    ]

    quality_lines = [
        "Nến từ chối rõ" if stars >= 3 else "Chưa rõ nến xác nhận",
        f"RSI(14) M15: {float(rsi15):.3f}",
        f"ATR(14) M15: ~{float(atr15):.3f}",
        "RR ~ 1:2 (mục tiêu)",
    ]

    # Entry/SL/TP placeholder: keep existing format, fill later by your risk engine if any.
    entry = m15c[-1].close
    # Basic ATR-based placeholders (safe)
    sl = entry - 1.0 * atr15 if recommendation == "BUY" else (entry + 1.0 * atr15 if recommendation == "SELL" else None)
    tp1 = entry + 1.0 * atr15 if recommendation == "BUY" else (entry - 1.0 * atr15 if recommendation == "SELL" else None)
    tp2 = entry + 2.0 * atr15 if recommendation == "BUY" else (entry - 2.0 * atr15 if recommendation == "SELL" else None)

    notes = [
        "Nguồn dữ liệu: EXNESS_MT5_PUSH",
        "Entry M30: chỉ vào khi M30 đóng xác nhận (anti-flip).",
        "Luôn chờ nến xác nhận/retest theo M15.",
    ]

    levels_info = _build_levels(symbol, m15c, m30c, h1c)

    out: Dict[str, Any] = {
        "symbol": symbol,
        "tf": "M30",
        "session": _session_tag(),
        "context_lines": context_lines,
        "position_lines": [],  # kept for compatibility (optional)
        "short_hint": build_short_hint(symbol, m15c, m30c, h1c),
        "liquidity_lines": liquidity_lines,
        "quality_lines": quality_lines,
        "recommendation": recommendation,
        "stars": stars,
        "entry": float(entry),
        "sl": float(sl) if sl is not None else None,
        "tp1": float(tp1) if tp1 is not None else None,
        "tp2": float(tp2) if tp2 is not None else None,
        "notes": notes,
        "levels_info": levels_info,
    }
    return out


# ----------------------------
# Telegram formatting (FULL)
# ----------------------------
def format_signal(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "XAU/USD")
    tf = sig.get("tf", "M30")
    session = sig.get("session", "Phiên Mỹ")

    context_lines = sig.get("context_lines", []) or []
    short_hint = sig.get("short_hint", []) or []
    liquidity_lines = sig.get("liquidity_lines", []) or []
    quality_lines = sig.get("quality_lines", []) or []
    notes = sig.get("notes", []) or []
    levels_info = sig.get("levels_info", []) or []

    rec = sig.get("recommendation", "CHỜ")
    stars = int(sig.get("stars", 1))
    stars_txt = "⭐️" * max(1, min(5, stars))

    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")

    def nf(x):
        if x is None:
            return "..."
        try:
            x = float(x)
            return _fmt_price(symbol, x)
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
    for ln in short_hint:
        # already formatted with leading "-"
        lines.append(ln if ln.strip().startswith("-") else f"- {ln}")

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
    else:
        lines.append("- (chưa có mốc)")

    return "\n".join(lines)
