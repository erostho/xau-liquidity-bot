# app/pro_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math


# =========================
# Data model
# =========================
@dataclass
class Candle:
    ts: int          # unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# =========================
# Helpers: indicators
# =========================
def _ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 1:
        return values[:]
    k = 2.0 / (period + 1.0)
    out = []
    ema = values[0]
    out.append(ema)
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
        out.append(ema)
    return out


def _rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 2:
        return 50.0
    gains = 0.0
    losses = 0.0
    # Wilder smoothing initial
    for i in range(1, period + 1):
        chg = values[i] - values[i - 1]
        if chg >= 0:
            gains += chg
        else:
            losses -= chg
    avg_gain = gains / period
    avg_loss = losses / period if losses != 0 else 1e-9
    # Continue smoothing
    for i in range(period + 1, len(values)):
        chg = values[i] - values[i - 1]
        gain = chg if chg > 0 else 0.0
        loss = -chg if chg < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    rs = avg_gain / (avg_loss if avg_loss != 0 else 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(max(0.0, min(100.0, rsi)))


def _true_range(c: Candle, prev_close: float) -> float:
    return max(
        c.high - c.low,
        abs(c.high - prev_close),
        abs(c.low - prev_close),
    )


def _atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 2:
        # fallback: average range
        if not candles:
            return 0.0
        ranges = [c.high - c.low for c in candles[-min(len(candles), period):]]
        return float(sum(ranges) / max(1, len(ranges)))
    trs = []
    for i in range(1, len(candles)):
        trs.append(_true_range(candles[i], candles[i - 1].close))
    # Wilder ATR
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return float(atr)


def _swing_high_low(candles: List[Candle], lookback: int = 80) -> Tuple[float, float]:
    if not candles:
        return (0.0, 0.0)
    window = candles[-min(len(candles), lookback):]
    hi = max(c.high for c in window)
    lo = min(c.low for c in window)
    return float(hi), float(lo)


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


# =========================
# Helpers: price action cues
# =========================
def _body(c: Candle) -> float:
    return abs(c.close - c.open)


def _range(c: Candle) -> float:
    return max(1e-9, c.high - c.low)


def _upper_wick(c: Candle) -> float:
    return c.high - max(c.open, c.close)


def _lower_wick(c: Candle) -> float:
    return min(c.open, c.close) - c.low


def _is_rejection(c: Candle, direction: str, wick_ratio: float = 0.45) -> bool:
    # direction: "sell" => upper wick heavy; "buy" => lower wick heavy
    r = _range(c)
    if direction == "sell":
        return (_upper_wick(c) / r) >= wick_ratio and (c.close <= c.open)
    if direction == "buy":
        return (_lower_wick(c) / r) >= wick_ratio and (c.close >= c.open)
    return False


def _is_engulfing(prev: Candle, cur: Candle, direction: str) -> bool:
    # direction: "bull" or "bear"
    if direction == "bull":
        return (cur.close > cur.open) and (cur.open <= prev.close) and (cur.close >= prev.open)
    if direction == "bear":
        return (cur.close < cur.open) and (cur.open >= prev.close) and (cur.close <= prev.open)
    return False


def _spike_then_retrace(candles: List[Candle], atr: float) -> str:
    """
    crude regime label:
    - SPIKE → HỒI if last 8 bars include one bar range > 1.8*ATR and then price retraces > 0.35*ATR
    - TREND if EMAs aligned strongly later (handled elsewhere)
    - RANGE otherwise
    """
    if len(candles) < 12 or atr <= 0:
        return "RANGE"

    last = candles[-8:]
    spike_idx = None
    for i, c in enumerate(last):
        if _range(c) >= 1.8 * atr:
            spike_idx = i
            break
    if spike_idx is None:
        return "RANGE"

    spike_c = last[spike_idx]
    # after spike, did we retrace meaningfully?
    after = last[spike_idx + 1:]
    if not after:
        return "RANGE"

    spike_dir_up = spike_c.close > spike_c.open
    if spike_dir_up:
        peak = spike_c.high
        trough_after = min(c.low for c in after)
        retr = peak - trough_after
    else:
        low = spike_c.low
        peak_after = max(c.high for c in after)
        retr = peak_after - low

    if retr >= 0.35 * atr:
        return "SPIKE → HỒI"
    return "RANGE"


def _sideways_count(candles: List[Candle], n: int = 6, band: float = 0.25) -> bool:
    """
    detect chop: last n candles closes in tight band relative to ATR-ish proxy (avg range).
    band = 0.25 => 25% of avg range
    """
    if len(candles) < n:
        return False
    last = candles[-n:]
    closes = [c.close for c in last]
    hi = max(closes)
    lo = min(closes)
    avg_r = sum((_range(c) for c in last)) / n
    if avg_r <= 0:
        return True
    return (hi - lo) <= band * avg_r


def _session_label() -> str:
    # Keep stable label; caller can override.
    return "Phiên Mỹ"


# =========================
# Core PRO analysis
# =========================
def analyze_pro(
    m15: List[Candle],
    h1: List[Candle],
    symbol: str = "XAUUSD",
    session: Optional[str] = None,
) -> Dict:
    """
    Returns a dict used by format_signal().
    Assumes m15/h1 are sorted oldest->newest.
    """
    session = session or _session_label()

    if len(m15) < 50 or len(h1) < 50:
        return {
            "symbol": symbol,
            "tf": "M15",
            "session": session,
            "context_lines": ["- Thiếu dữ liệu nến để phân tích (cần >=50 candles mỗi TF)."],
            "position_lines": [],
            "liquidity_lines": [],
            "quality_lines": [],
            "recommendation": "CHỜ",
            "bias": "NEUTRAL",
            "stars": 1,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "notes": ["- Hãy thử lại sau ~5–10 phút."],
            "levels": [],
        }

    # ---------- Indicators
    m15_closes = [c.close for c in m15]
    h1_closes = [c.close for c in h1]

    ema20_h1 = _ema(h1_closes, 20)
    ema50_h1 = _ema(h1_closes, 50)
    ema20_m15 = _ema(m15_closes, 20)
    ema50_m15 = _ema(m15_closes, 50)

    rsi_m15 = _rsi(m15_closes, 14)
    atr_m15 = _atr(m15, 14)
    atr_h1 = _atr(h1, 14)

    last_m15 = m15[-1]
    prev_m15 = m15[-2]

    # ---------- Trend context (H1)
    h1_trend_up = ema20_h1[-1] > ema50_h1[-1]
    h1_trend_down = ema20_h1[-1] < ema50_h1[-1]

    # Trend strength (slope)
    def slope(arr: List[float], k: int = 8) -> float:
        if len(arr) < k + 1:
            return 0.0
        return (arr[-1] - arr[-k-1]) / max(1e-9, abs(arr[-k-1]))

    s20 = slope(ema20_h1, 8)
    s50 = slope(ema50_h1, 8)

    h1_strength = "mạnh"
    if abs(s20) < 0.0006 and abs(s50) < 0.0004:
        h1_strength = "yếu dần"

    # ---------- Alignment (M15 with H1)
    m15_up = ema20_m15[-1] > ema50_m15[-1]
    m15_down = ema20_m15[-1] < ema50_m15[-1]
    alignment = (h1_trend_up and m15_up) or (h1_trend_down and m15_down)

    # ---------- Market regime (M15)
    regime = _spike_then_retrace(m15, atr_m15)
    if regime == "RANGE" and _sideways_count(m15, 8, 0.22):
        regime = "RANGE (đi ngang)"

    # ---------- Structure levels
    m15_swh, m15_swl = _swing_high_low(m15, 80)
    h1_swh, h1_swl = _swing_high_low(h1, 80)
    price = last_m15.close

    # ---------- Positioning
    # where are we in the recent range?
    m15_pos = 0.5 if (m15_swh - m15_swl) <= 1e-9 else (price - m15_swl) / (m15_swh - m15_swl)
    near_top = m15_pos >= 0.78
    near_bottom = m15_pos <= 0.22

    position_lines: List[str] = []
    if near_top:
        position_lines.append("- Giá gần đỉnh phiên / vùng kháng cự ngắn hạn")
    elif near_bottom:
        position_lines.append("- Giá gần đáy phiên / vùng hỗ trợ ngắn hạn")
    else:
        position_lines.append("- Giá đang ở giữa cấu trúc (không quá đẹp)")

    # retest zone heuristic: use last impulse pivot on M15
    # build a soft "zone" from last 12 bars: local swing high/low
    last12 = m15[-12:]
    z_hi = max(c.high for c in last12)
    z_lo = min(c.low for c in last12)
    # compress zone a bit to avoid too wide
    zone_mid = (z_hi + z_lo) / 2.0
    zone_half = max(0.15 * atr_m15, 0.25 * (z_hi - z_lo))
    supply_lo = zone_mid
    supply_hi = zone_mid + zone_half
    demand_hi = zone_mid
    demand_lo = zone_mid - zone_half

    # Are we retesting supply/demand?
    if near_top and (supply_lo <= price <= supply_hi):
        position_lines.append(f"- Retest vùng phân phối {supply_lo:.3f}–{supply_hi:.3f}")
    elif near_bottom and (demand_lo <= price <= demand_hi):
        position_lines.append(f"- Retest vùng cầu {demand_lo:.3f}–{demand_hi:.3f}")

    # ---------- Liquidity cues (price-action inference)
    liquidity_lines: List[str] = []
    # 1) failed break / no follow-through:
    broke_high = last_m15.high > m15_swh * 0.9995  # loose
    # Actually we use last 6 bars: if we made a new high then closed back under prior high -> possible sell-limits
    last6 = m15[-6:]
    prev_range_high = max(c.high for c in m15[-20:-6])
    made_new_high = max(c.high for c in last6) > prev_range_high
    close_below = last_m15.close < prev_range_high
    if made_new_high and close_below:
        liquidity_lines.append("- 🔴 Dấu hiệu SELL limit lớn phía trên (break high nhưng đóng lại dưới vùng)")
        liquidity_lines.append("- Break trước đó không follow-through")

    # 2) sweep: wick above recent high then close down (bear sweep)
    recent_high = max(c.high for c in m15[-20:-1])
    if last_m15.high > recent_high and last_m15.close < recent_high:
        liquidity_lines.append("- Sweep đỉnh (liquidity grab) rồi đóng ngược lại")

    # 3) absorption / rejection candle
    if _is_rejection(last_m15, "sell", wick_ratio=0.42):
        liquidity_lines.append("- Nến từ chối phía trên rõ (upper wick dài)")

    if not liquidity_lines:
        # provide neutral liquidity read
        liquidity_lines.append("- Chưa thấy dấu hiệu thanh khoản quá rõ (ưu tiên chờ nến xác nhận)")

    # ---------- Setup quality + scoring
    quality_lines: List[str] = []
    stars = 1
    # alignment
    if alignment:
        stars += 1
        quality_lines.append("- Đồng pha M15 với H1 (trend alignment)")
    else:
        quality_lines.append("- M15 chưa đồng pha hoàn toàn với H1 (cẩn trọng)")

    # liquidity cue present?
    liq_strong = any("SELL limit lớn" in x or "Sweep" in x or "từ chối" in x for x in liquidity_lines)
    if liq_strong:
        stars += 1

    # candle confirmation:
    bearish_engulf = _is_engulfing(prev_m15, last_m15, "bear")
    bullish_engulf = _is_engulfing(prev_m15, last_m15, "bull")
    if bearish_engulf or bullish_engulf or _is_rejection(last_m15, "sell", 0.42) or _is_rejection(last_m15, "buy", 0.42):
        stars += 1
        if bearish_engulf:
            quality_lines.append("- Nến bearish engulfing / đảo chiều ngắn hạn")
        elif bullish_engulf:
            quality_lines.append("- Nến bullish engulfing / đảo chiều ngắn hạn")
        else:
            quality_lines.append("- Nến từ chối rõ")

    # positioning bonus
    if near_top or near_bottom:
        stars += 1
        quality_lines.append("- Vị trí đẹp (gần cực trị range)")

    stars = max(1, min(5, stars))

    # RR planning + ATR SL budget
    # We'll propose both BUY and SELL candidates, then select based on context/liquidity/position.
    # ---------- Bias decision
    # baseline bias from H1 trend:
    bias = "NEUTRAL"
    if h1_trend_up:
        bias = "BUY"
    elif h1_trend_down:
        bias = "SELL"

    # But if near_top + strong sell-liquidity cues => favor SELL even if H1 bullish but weak.
    favor_sell = near_top and liq_strong
    favor_buy = near_bottom and liq_strong

    # if H1 bullish but weak + near_top => more willing to sell retrace
    if h1_trend_up and h1_strength == "yếu dần" and near_top:
        favor_sell = True

    recommendation = "CHỜ"
    rec_side = None  # "SELL" or "BUY"

    if favor_sell and not favor_buy:
        recommendation = "🔴 SELL"
        rec_side = "SELL"
    elif favor_buy and not favor_sell:
        recommendation = "🟢 BUY"
        rec_side = "BUY"
    else:
        # fallback: follow H1 if alignment and RSI supports
        if alignment and bias == "BUY" and rsi_m15 >= 52:
            recommendation = "🟢 BUY"
            rec_side = "BUY"
        elif alignment and bias == "SELL" and rsi_m15 <= 48:
            recommendation = "🔴 SELL"
            rec_side = "SELL"
        else:
            recommendation = "CHỜ"
            rec_side = None

    # ---------- Entry / SL / TP (ATR + structure)
    entry = None
    sl = None
    tp1 = None
    tp2 = None
    notes: List[str] = []

    # ATR budget in "$" (XAU points)
    # We set a "professional" stop: between 1.2–1.6 ATR but capped by nearest structure.
    sl_atr = 1.35 * atr_m15
    # If user wants "$12" style, their broker digits differ; still ok as points.
    quality_lines.append(f"- ATR cho phép SL ~ {sl_atr:.3f}")

    if rec_side == "SELL":
        entry = price  # market/close
        # SL above recent swing high + buffer
        recent_local_high = max(c.high for c in m15[-10:])
        sl_struct = recent_local_high + 0.18 * atr_m15
        # choose tighter of (struct) but ensure not too tight
        sl = max(entry + 0.85 * atr_m15, sl_struct)
        # TP targets: toward mid/low structure
        tp1 = entry - (sl - entry) * 1.2  # ~RR 1:1.2
        tp2 = entry - (sl - entry) * 2.0  # ~RR 1:2
        # clamp TP to swing low if closer for realism
        tp2 = min(tp2, m15_swl + 0.10 * atr_m15)

        # Notes / invalidation
        invalidate = recent_local_high + 0.05 * atr_m15
        notes.append(f"- Không SELL nếu M15 đóng > {invalidate:.3f}")
        if _sideways_count(m15, 8, 0.20):
            notes.append("- Bỏ kèo nếu giá đi ngang thêm (chop tăng rủi ro quét SL)")
    elif rec_side == "BUY":
        entry = price
        recent_local_low = min(c.low for c in m15[-10:])
        sl_struct = recent_local_low - 0.18 * atr_m15
        sl = min(entry - 0.85 * atr_m15, sl_struct)
        tp1 = entry + (entry - sl) * 1.2
        tp2 = entry + (entry - sl) * 2.0
        tp2 = max(tp2, m15_swh - 0.10 * atr_m15)

        invalidate = recent_local_low - 0.05 * atr_m15
        notes.append(f"- Không BUY nếu M15 đóng < {invalidate:.3f}")
        if _sideways_count(m15, 8, 0.20):
            notes.append("- Bỏ kèo nếu giá đi ngang thêm (chop tăng rủi ro quét SL)")
    else:
        notes.append("- Chờ nến M15 xác nhận (engulfing/rejection rõ) trước khi vào.")

    # RR line (if plan exists)
    if entry is not None and sl is not None and tp2 is not None:
        risk = abs(sl - entry)
        reward = abs(tp2 - entry)
        rr = (reward / risk) if risk > 0 else 0.0
        quality_lines.append(f"- RR ~ {rr:.2f}")

    # ---------- Context lines (exact style you want)
    context_lines: List[str] = []
    context_lines.append(f"- Thị trường: {regime}")
    if h1_trend_up:
        context_lines.append(f"- H1: bullish nhưng lực {h1_strength}")
    elif h1_trend_down:
        context_lines.append(f"- H1: bearish nhưng lực {h1_strength}")
    else:
        context_lines.append("- H1: sideway / chưa rõ xu hướng")

    # Add one more “pro” supporting stats (kept short)
    # RSI mention optional but useful
    context_lines.append(f"- RSI(14) M15: {rsi_m15:.1f}")

    # ---------- Levels
    levels = [
        f"- Swing High (M15 ~80): {m15_swh:.3f}",
        f"- Swing Low  (M15 ~80): {m15_swl:.3f}",
        f"- Swing High (H1  ~80): {h1_swh:.3f}",
        f"- Swing Low  (H1  ~80): {h1_swl:.3f}",
        f"- Giá hiện tại: {price:.3f} | ATR(14)~{atr_m15:.3f}",
    ]

    return {
        "symbol": symbol,
        "tf": "M15",
        "session": session,

        "context_lines": context_lines,
        "position_lines": position_lines,
        "liquidity_lines": liquidity_lines,
        "quality_lines": quality_lines,

        "recommendation": recommendation,
        "bias": (rec_side or bias),
        "stars": stars,

        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,

        "notes": notes,
        "levels": levels,
    }


# =========================
# Formatting (100% layout you agreed)
# =========================
def format_signal(sig: Dict) -> str:
    symbol = sig.get("symbol", "XAUUSD")
    tf = sig.get("tf", "M15")
    session = sig.get("session", "Phiên Mỹ")

    context_lines = sig.get("context_lines", [])
    position_lines = sig.get("position_lines", [])
    liquidity_lines = sig.get("liquidity_lines", [])
    quality_lines = sig.get("quality_lines", [])

    recommendation = sig.get("recommendation", "CHỜ")
    stars_n = int(sig.get("stars", 1))
    stars_n = max(1, min(5, stars_n))
    stars = "⭐️" * stars_n + "☆" * (5 - stars_n)

    entry = sig.get("entry", None)
    sl = sig.get("sl", None)
    tp1 = sig.get("tp1", None)
    tp2 = sig.get("tp2", None)

    notes = sig.get("notes", [])
    levels = sig.get("levels", [])

    def f(x: Optional[float]) -> str:
        return "..." if x is None else f"{x:.3f}"

    # EXACT layout (as you wrote)
    out = []
    out.append(f"📊 {symbol} | {tf} | {session}")
    out.append("")
    out.append("Context:")
    for line in context_lines:
        out.append(line)
    out.append("")
    out.append("Vị trí:")
    for line in position_lines:
        out.append(line)
    out.append("")
    out.append("Thanh khoản:")
    for line in liquidity_lines:
        out.append(line)
    out.append("")
    out.append("Chất lượng setup:")
    for line in quality_lines:
        out.append(line)
    out.append("")
    out.append(f"🎯 Khuyến nghị: {recommendation}")
    out.append(f"Độ tin cậy: {stars} ({stars_n}/5)")
    out.append("")
    out.append(f"ENTRY: {f(entry)}")
    out.append(f"SL: {f(sl)} | TP1: {f(tp1)} | TP2: {f(tp2)}")
    out.append("")
    out.append("⚠️ Lưu ý:")
    for line in notes:
        out.append(line)
    out.append("")
    out.append("Mốc giá quan trọng:")
    for line in levels:
        out.append(line)

    return "\n".join(out)
