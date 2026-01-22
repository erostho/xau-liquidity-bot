# app/pro_analysis.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import math
import os
from app.risk import calc_smart_sl_tp
from dataclasses import dataclass

# --- Safe candle access helpers (dict / dataclass / object) ---
def _c_val(c, key: str, default=None):
    try:
        if isinstance(c, dict):
            return c.get(key, default)
        # dataclass or plain object
        return getattr(c, key, default)
    except Exception:
        return default

def _series(candles, key: str):
    return [float(_c_val(c, key, 0.0) or 0.0) for c in (candles or [])]

def _close_series(candles):
    return _series(candles, "close")

def _hl_series(candles):
    return _series(candles, "high"), _series(candles, "low")
def _closes(candles):
    out = []
    for c in candles or []:
        if isinstance(c, dict):
            out.append(float(c.get("close", 0.0)))
        else:
            out.append(float(getattr(c, "close", 0.0)))
    return out

def _trend_label(candles):
    """
    Return: 'bullish' / 'bearish' / 'sideways'
    candles: list[dict] hoặc list[Candle]
    """
    closes = _closes(candles)
    if not closes or len(closes) < 60:
        return "sideways"

    # _ema() trong file của mày đang trả về LIST (chuỗi EMA),
    # nên phải lấy EMA cuối cùng để so sánh.
    ema_f_series = _ema(closes, 20)
    ema_s_series = _ema(closes, 50)

    if not ema_f_series or not ema_s_series:
        return "sideways"

    ema_f = float(ema_f_series[-1])
    ema_s = float(ema_s_series[-1])

    # tránh chia/so sánh kiểu “rất sát nhau”
    if abs(ema_f - ema_s) <= 1e-9:
        return "sideways"

    return "bullish" if ema_f > ema_s else "bearish"

# =========================
# Data model (MUST exist for import in main.py)
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
# Candle normalization (dict/object -> Candle)
# =========================
def _safe_candles(raw) -> List[Candle]:
    """Normalize candles coming from TwelveData/MT5 push into List[Candle].
    Accepts: Candle, dict, or objects with OHLC attrs. Returns sorted by ts."""
    if not raw:
        return []
    out: List[Candle] = []
    for c in raw:
        if c is None:
            continue
        if isinstance(c, Candle):
            out.append(c)
            continue
        # dict payloads (MT5 push or TwelveData)
        if isinstance(c, dict):
            ts = c.get("ts") or c.get("time") or c.get("t") or 0
            try:
                ts_i = int(float(ts))
            except Exception:
                ts_i = 0
            def _f(x, default=0.0):
                try:
                    return float(x)
                except Exception:
                    return float(default)
            out.append(Candle(
                ts=ts_i,
                open=_f(c.get("open")),
                high=_f(c.get("high")),
                low=_f(c.get("low")),
                close=_f(c.get("close")),
                volume=_f(c.get("volume") if c.get("volume") is not None else c.get("tick_volume"), 0.0),
            ))
            continue
        # object payloads
        ts = getattr(c, "ts", None) or getattr(c, "time", None) or getattr(c, "t", None) or 0
        try:
            ts_i = int(float(ts))
        except Exception:
            ts_i = 0
        def _fa(attr, default=0.0):
            try:
                return float(getattr(c, attr))
            except Exception:
                return float(default)
        vol = getattr(c, "volume", None)
        if vol is None:
            vol = getattr(c, "tick_volume", 0.0)
        try:
            vol_f = float(vol)
        except Exception:
            vol_f = 0.0
        out.append(Candle(ts=ts_i, open=_fa("open"), high=_fa("high"), low=_fa("low"), close=_fa("close"), volume=vol_f))
    out.sort(key=lambda x: x.ts)
    return out


# =========================
# Indicators
# =========================
def _ema(values: List[float], period: int) -> List[float]:
    if period <= 0 or len(values) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append(ema[-1] + k * (v - ema[-1]))
    pad = [ema[0]] * (period - 1)
    return pad + ema

def _rsi(values, period: int = 14):
    # Accept: list[float] OR list[candle]
    if not values:
        return None
    if not isinstance(values[0], (int, float)):
        values = _close_series(values)
    values = [float(x) for x in values if x is not None]
    if len(values) < period + 2:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    # Wilder's smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _atr(candles, period: int = 14):
    if not candles:
        return None
    # Accept list[candle] only
    highs = _series(candles, "high")
    lows = _series(candles, "low")
    closes = _series(candles, "close")
    if len(highs) < period + 2:
        return None
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    # Wilder
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr

def _build_short_hint_m15(m15: list[Candle], h1_trend: str, m30_trend: str) -> list[str]:
    """
    Gợi ý NGẮN HẠN dựa trên 30 nến M15 (~7.5h) gần nhất để tránh mốc quá xa.
    Dùng H1/M30 chỉ để confirm hướng (bullish/bearish/sideway).
    """
    if not m15 or len(m15) < 10:
        return ["- Chưa đủ dữ liệu M15 để gợi ý ngắn hạn → CHỜ KÈO"]

    # dùng 30 nến đã đóng gần nhất (bỏ nến cuối để tránh nến đang chạy)
    use = m15[-31:-1] if len(m15) >= 31 else m15[:-1]
    if not use:
        use = m15[-10:]

    lo = min(c.low for c in use)
    hi = max(c.high for c in use)
    last = m15[-1].close

    atr15 = _atr(m15, 14) or ((hi - lo) / 10.0 if hi > lo else 0.0)
    if atr15 <= 0:
        atr15 = max(1e-6, abs(last) * 0.001)

    # Ưu tiên theo H1, M30 dùng để tránh ngược hoàn toàn
    prefer = "neutral"
    if h1_trend == "bullish" and m30_trend != "bearish":
        prefer = "buy"
    elif h1_trend == "bearish" and m30_trend != "bullish":
        prefer = "sell"

    lines: list[str] = []

    if prefer == "buy":
        zone_lo = max(lo, last - 1.0 * atr15)
        zone_hi = min(hi, last - 0.25 * atr15)
        if zone_hi <= zone_lo:
            zone_lo = max(lo, last - 0.8 * atr15)
            zone_hi = min(hi, last - 0.2 * atr15)

        trigger = max(zone_lo, last - 0.4 * atr15)
        invalid = max(lo, zone_lo - 0.2 * atr15)

        lines.append("- Ưu tiên BUY theo xu hướng H1.")
        lines.append(f"- Vùng quan sát BUY (30 nến M15): {zone_lo:.3f} – {zone_hi:.3f}.")
        lines.append(f"- BUY khi M15 tạo higher-low và đóng trên {trigger:.3f}.")
        lines.append(f"- Nếu M15 đóng dưới {invalid:.3f} → bỏ kèo, chờ cấu trúc mới.")
        return lines

    if prefer == "sell":
        zone_hi = min(hi, last + 1.0 * atr15)
        zone_lo = max(lo, last + 0.25 * atr15)
        if zone_hi <= zone_lo:
            zone_lo = max(lo, last + 0.2 * atr15)
            zone_hi = min(hi, last + 0.8 * atr15)

        trigger = min(zone_hi, last + 0.4 * atr15)
        invalid = min(hi, zone_hi + 0.2 * atr15)

        lines.append("- Ưu tiên SELL theo xu hướng H1.")
        lines.append(f"- Vùng quan sát SELL (30 nến M15): {zone_lo:.3f} – {zone_hi:.3f}.")
        lines.append(f"- SELL khi M15 tạo lower-high và đóng dưới {trigger:.3f}.")
        lines.append(f"- Nếu M15 đóng trên {invalid:.3f} → bỏ kèo, chờ cấu trúc mới.")
        return lines

    # neutral
    lines.append("- Chưa có gợi ý ngắn hạn rõ ràng → CHỜ KÈO")
    lines.append(f"- Range 30 nến M15: {lo:.3f} – {hi:.3f}.")
    return lines
def _swing_high(candles: List[Candle], lookback: int = 80) -> Optional[float]:
    if len(candles) < 5:
        return None
    lb = candles[-lookback:] if len(candles) > lookback else candles
    return max(c.high for c in lb)

def _swing_low(candles: List[Candle], lookback: int = 80) -> Optional[float]:
    if len(candles) < 5:
        return None
    lb = candles[-lookback:] if len(candles) > lookback else candles
    return min(c.low for c in lb)

def _is_rejection(c: Candle) -> Dict[str, bool]:
    body = abs(c.close - c.open)
    rng = max(1e-9, c.high - c.low)
    upper_wick = c.high - max(c.close, c.open)
    lower_wick = min(c.close, c.open) - c.low
    return {
        "upper_reject": upper_wick / rng >= 0.45 and body / rng <= 0.45,
        "lower_reject": lower_wick / rng >= 0.45 and body / rng <= 0.45,
        "doji_like": body / rng <= 0.20,
    }

def _build_short_hint(
    symbol: str,
    current_price: float,
    h1_trend: str,
    atr15: float,
    m15c: List[Candle],
    m30_swing_low: Optional[float],
    m30_swing_high: Optional[float],
    m15_swing_low: Optional[float],
    m15_swing_high: Optional[float],
    entry_price: Optional[float],
) -> List[str]:
    """Return concise, actionable short-term guidance lines.

    This is NOT a signal generator. It's a readability layer for the Telegram message.
    """
    # Cushion: small buffer above/below levels to avoid "touch by spread"
    cushion = max(0.15, atr15 * 0.01) if atr15 else 0.15

    # crude "higher-low" proxy on M15:
    # compare the minimum low of last 5 CLOSED candles vs the prior 5.
    higher_low = False
    if len(m15c) >= 12:
        last5 = m15c[-6:-1]   # exclude current forming candle
        prev5 = m15c[-11:-6]
        low1 = min(c.low for c in last5)
        low0 = min(c.low for c in prev5)
        higher_low = low1 > low0 + cushion

    lines: List[str] = []

    if h1_trend == "bullish":
        lines.append("Ưu tiên BUY theo xu hướng H1.")
        zone_low = m30_swing_low or m15_swing_low
        zone_high = entry_price or m30_swing_high or m15_swing_high
        if zone_low and zone_high:
            lo = min(zone_low, zone_high)
            hi = max(zone_low, zone_high)
            lines.append(f"Vùng quan sát BUY: {lo:.2f} – {hi:.2f} (hồi M30).")
            trigger = (zone_low + cushion) if zone_low else (current_price + cushion)
            if higher_low:
                lines.append(f"BUY khi M15 tạo higher-low và đóng trên {trigger:.2f}.")
            else:
                lines.append(f"Chờ M15 tạo higher-low rồi đóng trên {trigger:.2f} để BUY an toàn hơn.")
            if zone_low:
                lines.append(f"Nếu M15 đóng dưới {zone_low:.2f} → bỏ kèo, chờ cấu trúc mới.")
        else:
            lines.append("Chưa đủ dữ liệu để xác định vùng M30 rõ ràng → chờ thêm nến.")
    elif h1_trend == "bearish":
        lines.append("Ưu tiên SELL theo xu hướng H1.")
        zone_high = m30_swing_high or m15_swing_high
        zone_low = entry_price or m30_swing_low or m15_swing_low
        if zone_low and zone_high:
            lo = min(zone_low, zone_high)
            hi = max(zone_low, zone_high)
            lines.append(f"Vùng quan sát SELL: {lo:.2f} – {hi:.2f} (hồi M30).")
            trigger = (zone_high - cushion) if zone_high else (current_price - cushion)
            lines.append(f"SELL khi M15 hồi lên yếu và đóng dưới {trigger:.2f}.")
            if zone_high:
                lines.append(f"Nếu M15 đóng trên {zone_high:.2f} → bỏ kèo, chờ cấu trúc mới.")
        else:
            lines.append("Chưa đủ dữ liệu để xác định vùng M30 rõ ràng → chờ thêm nến.")
    else:
        lines.append("H1 đang SIDEWAY → ưu tiên CHỜ (đợi phá range hoặc tín hiệu rõ hơn).")

    return lines

def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# =========================
# PRO Analyzer (MUST be named analyze_pro for main.py import)
# =========================
def analyze_pro(symbol: str, m15: Sequence[dict], m30: Sequence[dict], h1: Sequence[dict]) -> dict:
    """PRO analysis: Signal=M15, Entry=M30, Confirm=H1.

    NOTE: Phần chấm sao/logic entry/SLTP giữ nguyên như bản gốc.
    Chỉ bổ sung/ổn định 'GỢI Ý NGẮN HẠN' dựa trên 30 nến M15 (~7.5h).
    """
    base = {
        "symbol": symbol,
        "tf": "M30",
        "session": "Phiên Mỹ",
        "context_lines": [],
        "short_hint": [],
        "liquidity_lines": [],
        "quality_lines": [],
        "recommendation": "CHỜ",
        "stars": 1,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "note_lines": [],
        "key_levels": [],
        "meta": {},
    }

    # ---- Safety / normalize candles
    if not m15 or not m30 or not h1:
        base["note_lines"].append("⚠️ Thiếu dữ liệu M15/M30/H1 → không phân tích được.")
        base["short_hint"] = ["- Chưa đủ dữ liệu → CHỜ KÈO"]
        return base

    m15c = _safe_candles(m15)
    m30c = _safe_candles(m30)
    h1c = _safe_candles(h1)

    if len(m15c) < 20 or len(m30c) < 5 or len(h1c) < 5:
        base["note_lines"].append("⚠️ Dữ liệu candles chưa đủ → kết quả có thể thiếu chính xác.")
        # vẫn tiếp tục, vì có thể đủ để hiển thị thông tin

    last15 = m15c[-1]
    last_close_15 = last15.close

    # Indicators (M15)
    m15_closes = [c.close for c in m15c]
    atr15 = _atr(m15c, 14) or 0.0
    rsi15 = _rsi(m15_closes, 14) or 50.0

    # Trends (H1 + M30)
    h1_trend = _trend_label(h1c)   # bullish / bearish / sideways
    m30_trend = _trend_label(m30c) # bullish / bearish / sideways

    # --- GỢI Ý NGẮN HẠN (dựa 30 nến M15 gần nhất)
    try:
        base["short_hint"] = _build_short_hint_m15(m15c, h1_trend, m30_trend)
    except Exception:
        base["short_hint"] = []

    last30 = m30c[-1]

    last_close_15 = last15.close
    last_close_30 = last30.close

    m15_closes = [c.close for c in m15c]
    h1_closes  = [c.close for c in h1c]

    ema20_h1 = _ema(h1_closes, 20)
    ema50_h1 = _ema(h1_closes, 50)
    rsi15 = _rsi(m15_closes, 14)
    atr15 = _atr(m15c, 14)

    # --- Trend H1
    h1_trend = "NEUTRAL"
    m30_trend = "sideway"
    if ema20_h1 and ema50_h1:
        if ema20_h1[-1] > ema50_h1[-1]:
            h1_trend = "bullish"
        elif ema20_h1[-1] < ema50_h1[-1]:
            h1_trend = "bearish"

        # M30 trend (confirm): EMA20 vs EMA50
        m30_closes = [c.close for c in m30c]
        ema20_m30 = _ema(m30_closes, 20)
        ema50_m30 = _ema(m30_closes, 50)
        if ema20_m30 and ema50_m30 and len(ema20_m30) > 0 and len(ema50_m30) > 0:
            m30_trend = "bullish" if ema20_m30[-1] > ema50_m30[-1] else "bearish"
        else:
            m30_trend = "sideway"

    weakening = False
    if ema20_h1 and ema50_h1 and len(ema20_h1) >= 6 and len(ema50_h1) >= 6:
        sep_now = ema20_h1[-1] - ema50_h1[-1]
        sep_prev = ema20_h1[-6] - ema50_h1[-6]
        if h1_trend == "bullish" and sep_now < sep_prev:
            weakening = True
        if h1_trend == "bearish" and sep_now > sep_prev:
            weakening = True

    # Key levels (important prices)
    sh15 = _swing_high(m15c, 80)
    sl15 = _swing_low(m15c, 80)
    sh30 = _swing_high(m30c, 80)
    sl30 = _swing_low(m30c, 80)
    sh1  = _swing_high(h1c, 80)
    sl1  = _swing_low(h1c, 80)

    levels_info: List[Tuple[float, str]] = []
    if sh15 is not None: levels_info.append((float(sh15), "M15 Swing High (đỉnh gần)") )
    if sl15 is not None: levels_info.append((float(sl15), "M15 Swing Low (đáy gần)") )
    if sh30 is not None: levels_info.append((float(sh30), "M30 Swing High (kháng cự)") )
    if sl30 is not None: levels_info.append((float(sl30), "M30 Swing Low (hỗ trợ)") )
    if sh1 is not None:  levels_info.append((float(sh1),  "H1 Swing High (kháng cự lớn)") )
    if sl1 is not None:  levels_info.append((float(sl1),  "H1 Swing Low (hỗ trợ lớn)") )

    # unique by rounded price, keep the most informative label (first seen)
    seen = set()
    levels_unique: List[Tuple[float, str]] = []
    for price, label in sorted(levels_info, key=lambda x: x[0], reverse=True):
        key = round(price, 3)
        if key in seen:
            continue
        seen.add(key)
        levels_unique.append((price, label))

    base["levels_info"] = levels_unique[:8]
    base["levels"] = [round(p, 3) for p, _ in levels_unique[:8]]

    # Observation triggers for M15 close (thuần range 30 nến M15 gần nhất ~8h, giữ logic gốc: breakout/swing-based))
    # NOTE: Phần "GỢI Ý NGẮN HẠN" phía trên dùng range 30 nến M15, nhưng "Gợi ý quan sát vào lệnh" phải giữ theo các mốc swing/HTF như bản cũ.
    # Giá hiện tại dùng close nến M15 mới nhất (không cần biến current_price)
    cur = float(_c_float(m15c[-1], "close", 0.0)) if m15c else float(entry_price or 0.0)
   
    # levels_unique = [(price, label), ...] đã được build ở phần mốc giá quan trọng
    lv_prices = [float(p) for (p, _lbl) in (levels_unique or []) if p is not None]
    
    if lv_prices:
        above = [p for p in lv_prices if p > cur]
        below = [p for p in lv_prices if p < cur]
        buy_level = min(above) if above else max(lv_prices)
        sell_level = max(below) if below else min(lv_prices)
    else:
        buy_level = sell_level = cur
    
    obs_buffer = float((atr15 or 0.0) * 0.10)
    
    base["observation"] = {
        "tf": "M15",
        "buy": float(buy_level),
        "sell": float(sell_level),
        "buffer": obs_buffer,
    }


    # Market state spike
    ranges20 = [c.high - c.low for c in m15c[-20:]]
    ranges80 = [c.high - c.low for c in m15c[-80:]]
    spike = (sum(ranges20) / max(1, len(ranges20))) > 1.35 * (sum(ranges80) / max(1, len(ranges80)))

    # Lower-high-ish
    lower_highish = False
    if len(m15c) >= 30:
        recent_high = max(c.high for c in m15c[-10:])
        prev_high   = max(c.high for c in m15c[-30:-10])
        if recent_high <= prev_high:
            lower_highish = True

    rej = _is_rejection(last15)

    # Liquidity proxy
    liq_sell = False
    liq_buy = False
    if sh15 is not None and last15.high >= sh15 * 0.999 and rej["upper_reject"]:
        liq_sell = True
    if sl15 is not None and last15.low <= sl15 * 1.001 and rej["lower_reject"]:
        liq_buy = True

    # Build lines + score
    score = 0
    context_lines: List[str] = []
    position_lines: List[str] = []
    liquidity_lines: List[str] = []
    quality_lines: List[str] = []
    notes: List[str] = []

    if spike:
        context_lines.append("Thị trường: SPIKE → HỒI")
        score += 1
    else:
        context_lines.append("Thị trường: SIDEWAY / HỒI NHẸ")

    if h1_trend == "bullish":
        context_lines.append("H1: bullish (EMA20 > EMA50)" + (" nhưng lực suy yếu" if weakening else ""))
        score += 1
    elif h1_trend == "bearish":
        context_lines.append("H1: bearish (EMA20 < EMA50)" + (" nhưng lực suy yếu" if weakening else ""))
        score += 1
    else:
        context_lines.append("H1: neutral")

    if atr15 is None:
        # fallback ATR = range cây vừa đóng
        atr15 = max(1e-6, last15.high - last15.low)

    if sh15 is not None and abs(sh15 - last_close_15) <= atr15 * 0.8:
        position_lines.append("Giá gần đỉnh phiên")
        score += 1
    if sl15 is not None and abs(last_close_15 - sl15) <= atr15 * 0.8:
        position_lines.append("Giá gần đáy phiên")
        score += 1

    if liq_sell:
        liquidity_lines.append("🔴 Dấu hiệu SELL limit lớn phía trên (proxy: sweep + rejection)")
        score += 1
    if liq_buy:
        liquidity_lines.append("🟢 Dấu hiệu BUY limit lớn phía dưới (proxy: sweep + rejection)")
        score += 1
    if not liquidity_lines:
        liquidity_lines.append("Chưa thấy sweep/rejection rõ (liquidity proxy).")

    if rej["upper_reject"] or rej["lower_reject"]:
        quality_lines.append("Nến từ chối rõ")
        score += 1

    if rsi15 is not None:
        quality_lines.append(f"RSI(14) M15: {_fmt(rsi15)}")

    quality_lines.append(f"ATR(14) M15: ~{_fmt(atr15)}")
    score += 1

    # Decide bias
    sell_ok = (rej["upper_reject"] or liq_sell) and (lower_highish or spike or weakening) and (sh15 is not None)
    buy_ok  = (rej["lower_reject"] or liq_buy)  and (spike or weakening) and (sl15 is not None)

    bias: Optional[str] = None
    if sell_ok and rsi15 is not None and rsi15 >= 52:
        bias = "SELL"
    elif buy_ok and rsi15 is not None and rsi15 <= 48:
        bias = "BUY"
    else:
        if sell_ok:
            bias = "SELL"
        elif buy_ok:
            bias = "BUY"

    if bias is None:
        base.update({
            "context_lines": context_lines,
            "position_lines": position_lines,
            "short_hint": [],
            "liquidity_lines": liquidity_lines,
            "quality_lines": quality_lines + ["RR ~ 1:2 (mục tiêu)"],
            "recommendation": "CHỜ",
            "stars": 1,
            "notes": ["Chưa đủ điều kiện vào kèo. Chờ thêm nến xác nhận/retest."],
        })
        return base

    # ---- H1 confirm (hard filter)
    STRICT_H1_CONFIRM = os.getenv("STRICT_H1_CONFIRM", "1") == "1"  # default ON
    if STRICT_H1_CONFIRM:
        if bias == "BUY" and h1_trend != "bullish":
            base.update({
                "context_lines": context_lines,
                "position_lines": position_lines,
                "liquidity_lines": liquidity_lines,
                "quality_lines": quality_lines + ["RR ~ 1:2 (mục tiêu)"],
                "recommendation": "CHỜ",
                "stars": 1,
                "notes": ["H1 chưa bullish → không BUY. Chờ H1 confirm hoặc kèo rõ hơn."],
            })
            return base
        if bias == "SELL" and h1_trend != "bearish":
            base.update({
                "context_lines": context_lines,
                "position_lines": position_lines,
                "liquidity_lines": liquidity_lines,
                "quality_lines": quality_lines + ["RR ~ 1:2 (mục tiêu)"],
                "recommendation": "CHỜ",
                "stars": 1,
                "notes": ["H1 chưa bearish → không SELL. Chờ H1 confirm hoặc kèo rõ hơn."],
            })
            return base

    recommendation = "🔴 SELL" if bias == "SELL" else "🟢 BUY"

    # Entry logic:
    # - Signal + retest zone from M15
    # - Entry trigger only when M30 candle CLOSE confirms direction
    RETEST_K = float(os.getenv("RETEST_K", "0.35"))
    RETEST_K = max(0.15, min(0.80, RETEST_K))

    # Default XAU zone padding (small) to tolerate spread/rung
    ZONE_PAD_K = float(os.getenv("ENTRY_ZONE_PAD_K", "0.20"))
    ZONE_PAD_K = max(0.05, min(0.60, ZONE_PAD_K))

    def _m30_confirm(side: str, c30: Candle) -> bool:
        if side == "BUY":
            return c30.close > c30.open
        if side == "SELL":
            return c30.close < c30.open
        return False

    confirm_m30 = _m30_confirm(bias, last30)

    if bias == "SELL":
        entry_center = last_close_15 + RETEST_K * atr15
        liq_level = sh15
        zone_pad = max(1e-9, ZONE_PAD_K * atr15)
        entry_zone_low = entry_center - zone_pad
        entry_zone_high = entry_center + zone_pad
        notes.append("Entry M30: chỉ SELL khi giá hồi vào vùng entry và M30 đóng xác nhận (nến giảm).")
        notes.append(f"Vùng SELL (retest từ M15): {_fmt(entry_zone_low)} – {_fmt(entry_zone_high)}")
        if sh15 is not None:
            notes.append(f"Không SELL nếu M15 đóng > {_fmt(sh15)}")
    else:
        entry_center = last_close_15 - RETEST_K * atr15
        liq_level = sl15
        zone_pad = max(1e-9, ZONE_PAD_K * atr15)
        entry_zone_low = entry_center - zone_pad
        entry_zone_high = entry_center + zone_pad
        notes.append("Entry M30: chỉ BUY khi giá hồi vào vùng entry và M30 đóng xác nhận (nến tăng).")
        notes.append(f"Vùng BUY (retest từ M15): {_fmt(entry_zone_low)} – {_fmt(entry_zone_high)}")
        if sl15 is not None:
            notes.append(f"Không BUY nếu M15 đóng < {_fmt(sl15)}")

    # Nếu M30 chưa confirm: trả về trạng thái CHỜ nhưng vẫn show zone quan sát
    if not confirm_m30:
        base.update({
            "context_lines": context_lines,
            "position_lines": position_lines,
            "liquidity_lines": liquidity_lines,
            "quality_lines": quality_lines + [f"M30: chưa đóng xác nhận ({'nến tăng' if bias=='BUY' else 'nến giảm'}) → CHỜ"],
            "recommendation": "CHỜ",
            "stars": 2,
            "entry": float(entry_center),
            "sl": None,
            "tp1": None,
            "tp2": None,
            "lot": None,
            "notes": notes + ["Chờ nến M30 đóng xác nhận rồi mới vào lệnh."],
        })
        return base

    # Confirmed: use entry_center as entry
    entry = float(entry_center)

    # ---- SMART SL/TP (CÁCH ĐÚNG: không bỏ kèo, chỉ warn + clamp trong risk.py)
    equity_usd = float(os.getenv("EQUITY_USD", "1000"))
    risk_pct   = float(os.getenv("RISK_PCT", "0.0075"))  # 0.005..0.01

    plan: Dict[str, Any]
    try:
        plan = calc_smart_sl_tp(
            symbol=symbol,
            side=bias,  # BUY/SELL
            entry=float(entry),
            atr=float(atr15),
            liquidity_level=float(liq_level) if liq_level is not None else None,
            equity_usd=equity_usd,
            risk_pct=risk_pct,
        )
    except Exception as e:
        plan = {"ok": False, "reason": f"risk_engine_error: {e}"}

    # nếu plan không ok: vẫn trả kèo nhưng set SL/TP fallback theo ATR để bot vẫn “báo”
    sl: Optional[float] = _safe_float(plan.get("sl"))
    tp1: Optional[float] = _safe_float(plan.get("tp1"))
    tp2: Optional[float] = _safe_float(plan.get("tp2"))
    lot: Optional[float] = _safe_float(plan.get("lot"))
    rdist: Optional[float] = _safe_float(plan.get("r"))

    if not plan.get("ok", True):
        quality_lines.append(f"⚠️ Risk warn: {plan.get('reason', 'risk check failed')}")
        # Fallback SL/TP theo ATR (an toàn, không crash)
        fallback_r = max(0.6, float(atr15) * 1.0)
        if bias == "SELL":
            sl = float(entry) + fallback_r
            tp1 = float(entry) - 1.0 * fallback_r
            tp2 = float(entry) - 1.6 * fallback_r
        else:
            sl = float(entry) - fallback_r
            tp1 = float(entry) + 1.0 * fallback_r
            tp2 = float(entry) + 1.6 * fallback_r
        rdist = fallback_r
        lot = lot or 0.01
        notes.append("⚠️ SL/TP dùng fallback theo ATR do risk engine báo không hợp lệ.")

    # đảm bảo vẫn có số
    if sl is None or tp1 is None or tp2 is None:
        fallback_r = max(0.6, float(atr15) * 1.0)
        if bias == "SELL":
            sl = float(entry) + fallback_r
            tp1 = float(entry) - 1.0 * fallback_r
            tp2 = float(entry) - 1.6 * fallback_r
        else:
            sl = float(entry) - fallback_r
            tp1 = float(entry) + 1.0 * fallback_r
            tp2 = float(entry) + 1.6 * fallback_r
        rdist = fallback_r if rdist is None else rdist
        lot = lot or 0.01

    # Stars
    stars = 1
    if score >= 6:
        stars = 5
    elif score >= 5:
        stars = 4
    elif score >= 3:
        stars = 3
    elif score >= 2:
        stars = 2

    quality_lines.append("RR ~ 1:2")
    if rdist is not None:
        quality_lines.append(f"R~{rdist:.2f} | SL=MIN(Liq, ATR, Risk) (risk engine)")

    base.update({
        "context_lines": context_lines,
        "position_lines": position_lines,
        "liquidity_lines": liquidity_lines,
        "quality_lines": quality_lines,
        "recommendation": recommendation,
        "stars": stars,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "lot": float(lot) if lot is not None else None,
        "notes": notes,
    })
    return base


# =========================
# Formatter (MUST be named format_signal for main.py import)
# =========================
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
    if short_hint:
        for s in short_hint:
            lines.append(f"- {s}")
    elif position_lines:
        for s in position_lines:
            lines.append(f"- {s}")
    else:
        lines.append("- Chưa có gợi ý ngắn hạn rõ ràng → CHỜ KÈO")

    # Add 1 guidance line right under Vị trí (as requested)
    try:
        b = observation.get("buy")
        s = observation.get("sell")
        buf = float(observation.get("buffer", 0.4))
        tf_obs = observation.get("tf", "M15")
        if b is not None and s is not None:
            lines.append(
                f"- QUAN SÁT: {tf_obs} đóng > {nf(float(b)+buf)} → BUY | {tf_obs} đóng < {nf(float(s)-buf)} → SELL | ngoài vùng → CHỜ KÈO"
            )
    except Exception:
        pass
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
