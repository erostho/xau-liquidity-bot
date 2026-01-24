# app/pro_analysis.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Sequence
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

# =========================
# SYMBOL PROFILES (XAG vs XAU/BTC)
# =========================
SYMBOL_PROFILE = {
    "XAG": {
        # wick must be big (XAG hay quét sâu)
        "sweep_wick_min": 0.48,
        # nến sweep phải đóng lại vào trong range tối thiểu bao nhiêu %
        "close_back_ratio": 0.28,
        # volume confirm (tick_volume cũng ok nhưng chỉ "bonus")
        "vol_spike_k": 1.7,
        # buffer để tránh "chạm nhẹ/spread"
        "buf_atr_k": 0.25,
        # spring follow-through tối thiểu
        "spring_follow_k": 0.60,
        # cooldown ý nghĩa là: chỉ coi là spring nếu có follow-through trong N nến sau
        "spring_lookahead": 3,
    },
    "XAU": {
        "sweep_wick_min": 0.42,
        "close_back_ratio": 0.22,
        "vol_spike_k": 1.4,
        "buf_atr_k": 0.20,
        "spring_follow_k": 0.50,
        "spring_lookahead": 3,
    },
    "BTC": {
        "sweep_wick_min": 0.36,
        "close_back_ratio": 0.18,
        "vol_spike_k": 1.3,
        "buf_atr_k": 0.18,
        "spring_follow_k": 0.45,
        "spring_lookahead": 3,
    },
}

def _get_profile(symbol: str) -> dict:
    s = (symbol or "").upper()
    if "XAG" in s:
        return SYMBOL_PROFILE["XAG"]
    if "XAU" in s:
        return SYMBOL_PROFILE["XAU"]
    if "BTC" in s:
        return SYMBOL_PROFILE["BTC"]
    return SYMBOL_PROFILE["XAU"]


def _median(xs: List[float]) -> float:
    xs = [float(x) for x in (xs or []) if x is not None]
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _vol_spike(candles: List[Candle], n: int = 20, k: float = 1.5) -> bool:
    """Volume spike so với median volume của n nến gần nhất (đã đóng)."""
    if not candles or len(candles) < max(6, n + 1):
        return False
    closed = candles[:-1]  # bỏ nến đang chạy
    use = closed[-n:]
    vols = [max(0.0, float(getattr(c, "volume", 0.0) or 0.0)) for c in use]
    base = _median(vols)
    if base <= 0:
        return False
    return vols[-1] >= k * base

def detect_sweep(
    candles: List[Candle],
    side: str,  # "SELL" (sweep high) hoặc "BUY" (sweep low)
    level: float,
    atr: Optional[float],
    symbol: str,
) -> Dict[str, Any]:
    """
    Sweep liquidity:
    - SELL sweep: giá chọc lên trên level rồi đóng xuống lại (rejection upper wick)
    - BUY sweep : giá chọc xuống dưới level rồi đóng lên lại (rejection lower wick)

    Trả về dict: {"ok":bool, "type": "...", "reason": "...", "score": int, ...}
    """
    cfg = _get_profile(symbol)
    if not candles or len(candles) < 6 or level is None:
        return {"ok": False, "reason": "not_enough_candles"}

    c = candles[-2] if len(candles) >= 2 else candles[-1]  # dùng nến đã đóng gần nhất
    a = float(atr or 0.0)
    buf = max(1e-9, (cfg["buf_atr_k"] * a) if a > 0 else abs(level) * 0.0002)

    rng = max(1e-9, c.high - c.low)
    body = abs(c.close - c.open)
    upper_wick = c.high - max(c.close, c.open)
    lower_wick = min(c.close, c.open) - c.low

    wick_min = float(cfg["sweep_wick_min"])
    close_back_ratio = float(cfg["close_back_ratio"])

    score = 0
    vol_ok = _vol_spike(candles, n=20, k=float(cfg["vol_spike_k"]))

    if side.upper() == "SELL":
        # 1) phá đỉnh
        pierced = c.high >= (level + buf)
        # 2) đóng lại xuống (nằm trong range cũ)
        close_back = (c.close <= (level - buf * 0.15))  # đóng dưới level một chút cho “thật”
        # 3) wick trên đủ dài
        wick_ok = (upper_wick / rng) >= wick_min and (body / rng) <= 0.65

        # 4) đóng lại vào thân dưới của nến (để tránh phá rồi kéo tiếp)
        # close_back_ratio: vị trí close tính từ đáy lên
        pos_close = (c.close - c.low) / rng  # 0..1
        close_in_lower = pos_close <= (1.0 - close_back_ratio)

        if pierced: score += 1
        if close_back: score += 1
        if wick_ok: score += 1
        if close_in_lower: score += 1
        if vol_ok: score += 1

        ok = (pierced and close_back and wick_ok and close_in_lower)
        return {
            "ok": bool(ok),
            "type": "sweep_high",
            "score": int(score),
            "vol_ok": bool(vol_ok),
            "level": float(level),
            "buf": float(buf),
        }

    else:  # BUY
        pierced = c.low <= (level - buf)
        close_back = (c.close >= (level + buf * 0.15))
        wick_ok = (lower_wick / rng) >= wick_min and (body / rng) <= 0.65

        pos_close = (c.close - c.low) / rng
        close_in_upper = pos_close >= close_back_ratio

        if pierced: score += 1
        if close_back: score += 1
        if wick_ok: score += 1
        if close_in_upper: score += 1
        if vol_ok: score += 1

        ok = (pierced and close_back and wick_ok and close_in_upper)
        return {
            "ok": bool(ok),
            "type": "sweep_low",
            "score": int(score),
            "vol_ok": bool(vol_ok),
            "level": float(level),
            "buf": float(buf),
        }


def detect_spring(
    candles: List[Candle],
    side: str,  # "BUY" spring (phá đáy giả) hoặc "SELL" upthrust (phá đỉnh giả)
    range_low: float,
    range_high: float,
    atr: Optional[float],
    symbol: str,
) -> Dict[str, Any]:
    """
    Spring / Upthrust:
    - BUY spring: chọc thủng range_low rồi kéo lên đóng lại trong range + có follow-through
    - SELL upthrust: chọc thủng range_high rồi kéo xuống đóng lại trong range + follow-through

    Đây là “cú phá vỡ cuối cùng” (false break + reversal confirmation).
    """
    cfg = _get_profile(symbol)
    if not candles or len(candles) < 10:
        return {"ok": False, "reason": "not_enough_candles"}
    if range_low is None or range_high is None or range_high <= range_low:
        return {"ok": False, "reason": "bad_range"}

    closed = candles[:-1] if len(candles) > 1 else candles
    c0 = closed[-1]  # nến đã đóng gần nhất (nến spring)
    a = float(atr or 0.0)
    buf = max(1e-9, (cfg["buf_atr_k"] * a) if a > 0 else abs(c0.close) * 0.0002)

    lookahead = int(cfg.get("spring_lookahead", 3))
    follow_k = float(cfg.get("spring_follow_k", 0.5))

    # follow-through: trong lookahead nến sau spring phải đi theo hướng reversal ít nhất k*ATR
    # (dùng max/min close để đơn giản, tránh nhiễu wick)
    after = closed[-lookahead:] if len(closed) >= lookahead else closed
    max_after_close = max(x.close for x in after)
    min_after_close = min(x.close for x in after)

    vol_ok = _vol_spike(candles, n=20, k=float(cfg["vol_spike_k"]))

    if side.upper() == "BUY":
        # phá đáy
        pierced = c0.low <= (range_low - buf)
        # đóng lại trong range
        close_back_in = c0.close >= (range_low + buf * 0.15)
        # follow-through lên
        need = (follow_k * a) if a > 0 else (range_high - range_low) * 0.20
        follow = (max_after_close - c0.close) >= max(1e-9, need)

        score = int(pierced) + int(close_back_in) + int(follow) + int(vol_ok)
        ok = pierced and close_back_in and follow
        return {
            "ok": bool(ok),
            "type": "spring_buy",
            "score": int(score),
            "vol_ok": bool(vol_ok),
            "range_low": float(range_low),
            "range_high": float(range_high),
            "buf": float(buf),
            "need": float(need),
        }

    else:  # SELL upthrust
        pierced = c0.high >= (range_high + buf)
        close_back_in = c0.close <= (range_high - buf * 0.15)
        need = (follow_k * a) if a > 0 else (range_high - range_low) * 0.20
        follow = (c0.close - min_after_close) >= max(1e-9, need)

        score = int(pierced) + int(close_back_in) + int(follow) + int(vol_ok)
        ok = pierced and close_back_in and follow
        return {
            "ok": bool(ok),
            "type": "spring_sell",
            "score": int(score),
            "vol_ok": bool(vol_ok),
            "range_low": float(range_low),
            "range_high": float(range_high),
            "buf": float(buf),
            "need": float(need),
        }


def _build_short_hint_m15(m15: list[Candle], h1_trend: str, m30_trend: str) -> list[str]:
    """
    Gợi ý NGẮN HẠN: thuần 30 nến M15 gần nhất (~7.5h), bỏ nến đang chạy.
    Micro-bias dùng SLOPE: avg(close last10) vs avg(close prev10).
    (h1_trend/m30_trend chỉ để hiển thị/nhắc thêm, không dùng làm ngưỡng.)
    """
    if not m15 or len(m15) < 12:
        return ["- Chưa đủ dữ liệu M15 để gợi ý ngắn hạn → CHỜ KÈO"]

    # dùng 30 nến đã đóng (bỏ nến cuối nếu muốn tránh nến đang chạy)
    closed = m15[:-1] if len(m15) > 1 else m15
    use = closed[-30:] if len(closed) >= 30 else closed
    if len(use) < 12:
        return ["- Chưa đủ dữ liệu M15 (sau khi bỏ nến đang chạy) → CHỜ KÈO"]

    lo = min(c.low for c in use)
    hi = max(c.high for c in use)

    # giá tham chiếu = close nến đã đóng gần nhất
    cur = use[-1].close

    # ATR dùng trên chuỗi M15 đã normalize
    atr15 = _atr(closed, 14)
    if atr15 is None or atr15 <= 0:
        # fallback nhẹ nếu thiếu ATR
        atr15 = max(1e-6, (hi - lo) / 10.0 if hi > lo else abs(cur) * 0.001)

    rng = max(1e-9, hi - lo)
    pos = (cur - lo) / rng  # 0..1
    pos_pct = max(0.0, min(1.0, pos)) * 100.0

    # === SLOPE micro-bias (A) ===
    # last10 vs prev10 (đều là nến đã đóng)
    micro = "neutral"
    micro_txt = "Micro-bias: NEUTRAL"
    slope = 0.0
    if len(use) >= 20:
        last10 = [c.close for c in use[-10:]]
        prev10 = [c.close for c in use[-20:-10]]
        avg_last = sum(last10) / 10.0
        avg_prev = sum(prev10) / 10.0
        slope = avg_last - avg_prev

        # ngưỡng “đáng kể” theo ATR
        thr = 0.15 * atr15
        if slope > thr:
            micro = "buy"
            micro_txt = f"Micro-bias: BUY nhẹ (slope +{_fmt(slope)} > { _fmt(thr) })"
        elif slope < -thr:
            micro = "sell"
            micro_txt = f"Micro-bias: SELL nhẹ (slope {_fmt(slope)} < -{ _fmt(thr) })"
        else:
            micro = "neutral"
            micro_txt = f"Micro-bias: NEUTRAL (slope {_fmt(slope)} ~ nhỏ)"

    # Range/ATR để biết đang nén hay đang giãn
    rng_atr = rng / max(1e-9, atr15)
    if rng_atr <= 2.0:
        state = "NÉN (range chặt)"
    elif rng_atr >= 3.5:
        state = "GIÃN (biến động cao)"
    else:
        state = "BÌNH THƯỜNG"

    # breakout trigger theo range30 + buffer
    buf = 0.20 * atr15
    buy_trig = hi + buf
    sell_trig = lo - buf

    lines: list[str] = []
    lines.append(f"- Range 30 nến M15: {_fmt(lo)} – {_fmt(hi)}.")
    lines.append(f"- Giá hiện tại: {_fmt(cur)} (~{pos_pct:.0f}% trong range) | {state} | Range≈{rng_atr:.1f} ATR.")
    lines.append(f"- {micro_txt}.")
    lines.append(f"- QUAN SÁT (breakout): M15 đóng > {_fmt(buy_trig)} → canh BUY | M15 đóng < {_fmt(sell_trig)} → canh SELL.")
    lines.append("- Nếu vẫn nằm trong range → ưu tiên CHỜ (đợi chạm biên + nến từ chối/retest rồi mới tính).")

    # optional: nhắc trend HTF (không dùng làm điều kiện)
    if h1_trend in ("bullish", "bearish"):
        lines.append(f"- (Tham khảo) H1: {h1_trend} | M30: {m30_trend}.")

    return lines
def _pick_trade_method_m30(m30c: List[Candle], atr30: Optional[float]) -> Dict[str, Any]:
    """
    Dựa 20 nến M30 đã đóng → gợi ý METHOD + entry/SL/TP dạng hướng dẫn.
    Return dict: {"method": str, "lines": list[str]}
    """
    if not m30c or len(m30c) < 25:
        return {"method": "UNKNOWN", "lines": ["Chưa đủ dữ liệu M30 để gợi ý phương pháp trade."]}

    closed = m30c[:-1] if len(m30c) > 1 else m30c
    use = closed[-20:] if len(closed) >= 20 else closed
    if len(use) < 20:
        return {"method": "UNKNOWN", "lines": ["Chưa đủ 20 nến M30 đã đóng → CHỜ."]}

    hi = max(c.high for c in use)
    lo = min(c.low for c in use)
    rng = max(1e-9, hi - lo)

    # atr30 fallback
    a = atr30 if (atr30 is not None and atr30 > 0) else max(1e-6, rng / 8.0)

    cur = use[-1].close
    pos = (cur - lo) / rng  # 0..1

    # slope: avg last5 - avg prev5
    last5 = [c.close for c in use[-5:]]
    prev5 = [c.close for c in use[-10:-5]]
    slope = (sum(last5)/5.0) - (sum(prev5)/5.0)
    thr = 0.20 * a

    # range/atr: biết đang nén hay giãn
    rng_atr = rng / max(1e-9, a)

    # --- Detect RANGE trading (điều kiện: slope nhỏ + range vừa phải)
    is_range = abs(slope) <= thr and rng_atr <= 3.2

    # --- Detect BREAKOUT-RETEST (điều kiện: có nén trước đó + close vượt biên rõ)
    # nén: 10 nến đầu range nhỏ hơn 10 nến sau (đang bung)
    first10 = use[:10]
    last10  = use[10:]
    r1 = max(c.high for c in first10) - min(c.low for c in first10)
    r2 = max(c.high for c in last10)  - min(c.low for c in last10)
    was_compress = (r1 / max(1e-9, a)) <= 1.8
    breakout_up = cur > hi - 0.05 * a and slope > thr
    breakout_dn = cur < lo + 0.05 * a and slope < -thr
    is_breakout = was_compress and (breakout_up or breakout_dn)

    # --- Detect IPC (Impulse–Pullback–Continuation)
    # impulse: 1-2 nến range lớn; pullback: 2-4 nến range nhỏ ngược hướng; continuation: close quay lại theo hướng impulse
    ranges = [c.high - c.low for c in use]
    big = [r for r in ranges if r >= 1.3 * a]
    has_impulse = len(big) >= 1
    is_ipc = (has_impulse and abs(slope) > thr and rng_atr >= 2.0)

    lines: List[str] = []
    # ưu tiên chọn method theo tính “rõ”
    if is_breakout:
        method = "BREAKOUT-RETEST"
        direction = "BUY" if breakout_up else "SELL"
        # entry: chờ retest về biên range
        entry = hi - 0.30 * a if direction == "BUY" else lo + 0.30 * a
        sl = entry - 1.1 * a if direction == "BUY" else entry + 1.1 * a
        tp1 = entry + 1.2 * a if direction == "BUY" else entry - 1.2 * a
        tp2 = entry + 2.0 * a if direction == "BUY" else entry - 2.0 * a

        lines.append(f"Method: {method} ({direction}).")
        lines.append(f"Vị trí: giá đang {'gần biên trên' if pos>0.75 else 'gần biên dưới' if pos<0.25 else 'giữa range'} của 20 nến M30.")
        lines.append(f"Entry gợi ý: chờ RETEST về ~{_fmt(entry)} rồi mới vào ({direction}).")
        lines.append(f"SL gợi ý: {_fmt(sl)} | TP1: {_fmt(tp1)} | TP2: {_fmt(tp2)}.")
        lines.append("Trigger: ưu tiên có nến M30/M15 từ chối tại vùng retest (đuôi/wick) rồi mới bấm.")
        return {"method": method, "lines": lines}

    if is_range:
        method = "RANGE"
        # range trading: buy near lo, sell near hi
        buy_zone = lo + 0.20 * a
        sell_zone = hi - 0.20 * a
        buy_sl = lo - 0.80 * a
        sell_sl = hi + 0.80 * a
        buy_tp = lo + 1.0 * a
        sell_tp = hi - 1.0 * a

        lines.append(f"Method: {method} (đánh trong biên).")
        lines.append(f"Range20 M30: {_fmt(lo)} – {_fmt(hi)} | Range≈{rng_atr:.1f} ATR | slope nhỏ.")
        lines.append(f"BUY gần đáy range: ~{_fmt(buy_zone)} | SL: {_fmt(buy_sl)} | TP: {_fmt(buy_tp)}.")
        lines.append(f"SELL gần đỉnh range: ~{_fmt(sell_zone)} | SL: {_fmt(sell_sl)} | TP: {_fmt(sell_tp)}.")
        lines.append("Trigger: chờ nến từ chối (rejection) ở biên range, không FOMO giữa range.")
        return {"method": method, "lines": lines}

    if is_ipc:
        method = "IPC"
        direction = "BUY" if slope > 0 else "SELL"
        # IPC: entry pullback 0.5-0.8 ATR từ điểm hiện tại
        entry = cur - 0.6 * a if direction == "BUY" else cur + 0.6 * a
        sl = entry - 1.2 * a if direction == "BUY" else entry + 1.2 * a
        tp1 = entry + 1.2 * a if direction == "BUY" else entry - 1.2 * a
        tp2 = entry + 2.1 * a if direction == "BUY" else entry - 2.1 * a

        lines.append(f"Method: {method} ({direction}) – xung lực mạnh, chờ hồi.")
        lines.append(f"Vị trí: giá ~{pos*100:.0f}% trong range 20 nến M30 | Range≈{rng_atr:.1f} ATR.")
        lines.append(f"Entry gợi ý: chờ PULLBACK về ~{_fmt(entry)} rồi canh {direction} (ưu tiên có HL/LH).")
        lines.append(f"SL gợi ý: {_fmt(sl)} | TP1: {_fmt(tp1)} | TP2: {_fmt(tp2)}.")
        lines.append("Trigger: M15 tạo cấu trúc (HL cho BUY / LH cho SELL) tại vùng pullback.")
        return {"method": method, "lines": lines}

    # default
    method = "WAIT"
    lines.append("Method: CHỜ – 20 nến M30 chưa ra mẫu rõ (không range đẹp, không breakout rõ, không IPC sạch).")
    lines.append(f"Range20 M30: {_fmt(lo)} – {_fmt(hi)} | Range≈{rng_atr:.1f} ATR | slope={_fmt(slope)}.")
    lines.append("Chờ: hoặc nén thêm (range/ATR giảm) rồi breakout, hoặc chạm biên rồi rejection rõ.")
    return {"method": method, "lines": lines}


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
    # --- Trade method suggestion (20 nến M30)
    m30_closed = m30c[:-1] if len(m30c) > 1 else m30c
    atr30 = _atr(m30_closed, 14)
    base["trade_method"] = _pick_trade_method_m30(m30c, atr30)

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
   # ===== SHORT-TERM HINT (30 candles M15 ONLY) =====
   
    # levels_unique = [(price, label), ...] đã được build ở phần mốc giá quan trọng
    cur = float(last_close_15)
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
    # =========================
    # LIQUIDITY: sweep / spring (profile-based)
    # =========================

    # Range 30 nến M15 đã đóng để làm "range" cho spring
    closed15 = m15c[:-1] if len(m15c) > 1 else m15c
    use15 = closed15[-30:] if len(closed15) >= 30 else closed15
    range15_low = min(c.low for c in use15) if use15 else (sl15 or last_close_15)
    range15_high = max(c.high for c in use15) if use15 else (sh15 or last_close_15)

    # Sweep theo swing levels (m15 swing)
    sweep_sell = detect_sweep(m15c, side="SELL", level=float(sh15) if sh15 else float(range15_high), atr=atr15, symbol=symbol)
    sweep_buy  = detect_sweep(m15c, side="BUY",  level=float(sl15) if sl15 else float(range15_low),  atr=atr15, symbol=symbol)

    # Spring / Upthrust theo range 30 nến M15
    spring_buy  = detect_spring(m15c, side="BUY",  range_low=float(range15_low),  range_high=float(range15_high), atr=atr15, symbol=symbol)
    spring_sell = detect_spring(m15c, side="SELL", range_low=float(range15_low),  range_high=float(range15_high), atr=atr15, symbol=symbol)

    liq_sell = bool(sweep_sell.get("ok")) or bool(spring_sell.get("ok"))
    liq_buy  = bool(sweep_buy.get("ok"))  or bool(spring_buy.get("ok"))

    # Build liquidity_lines with explanations
    #liquidity_lines = []
    context_lines: List[str] = []
    position_lines: List[str] = []
    liquidity_lines: List[str] = []
    quality_lines: List[str] = []
    notes: List[str] = []
    if sweep_sell.get("ok"):
        vtxt = " +VOL" if sweep_sell.get("vol_ok") else ""
        liquidity_lines.append(f"🔴 Sweep HIGH (quét đỉnh){vtxt}: chọc { _fmt(sweep_sell['level']) } rồi đóng xuống lại.")
        score += 1

    if sweep_buy.get("ok"):
        vtxt = " +VOL" if sweep_buy.get("vol_ok") else ""
        liquidity_lines.append(f"🟢 Sweep LOW (quét đáy){vtxt}: chọc { _fmt(sweep_buy['level']) } rồi đóng lên lại.")
        score += 1

    if spring_buy.get("ok"):
        vtxt = " +VOL" if spring_buy.get("vol_ok") else ""
        liquidity_lines.append(f"🟢 SPRING (false break đáy){vtxt}: phá range_low rồi kéo lên + follow-through.")
        score += 1

    if spring_sell.get("ok"):
        vtxt = " +VOL" if spring_sell.get("vol_ok") else ""
        liquidity_lines.append(f"🔴 UPTHRUST (false break đỉnh){vtxt}: phá range_high rồi kéo xuống + follow-through.")
        score += 1

    if not liquidity_lines:
        liquidity_lines.append("Chưa thấy sweep/spring rõ (liquidity proxy).")

    # (Gợi ý thêm) Nếu spring/sweep có nhưng H1 ngược trend -> note cho bot “đừng vội”
    if (spring_buy.get("ok") or sweep_buy.get("ok")) and h1_trend == "bearish":
        notes.append("⚠️ Có dấu hiệu hút thanh khoản dưới nhưng H1 đang bearish → ưu tiên CHỜ confirm M30.")
    if (spring_sell.get("ok") or sweep_sell.get("ok")) and h1_trend == "bullish":
        notes.append("⚠️ Có dấu hiệu hút thanh khoản trên nhưng H1 đang bullish → ưu tiên CHỜ confirm M30.")

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
    symbol = sig.get("symbol", "XAUUSD,XAGUSD")
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
    lines.append("Thanh khoản:")
    for s in liquidity_lines:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("Chất lượng setup:")
    for s in quality_lines:
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
        lines.append("- Chưa có gợi ý ngắn hạn rõ ràng → CHỜ TÍN HIỆU")
    lines.append("")
    tm = sig.get("trade_method", None)
    if isinstance(tm, dict):
        tm_lines = tm.get("lines", [])
        if tm_lines:
            lines.append("")
            lines.append("📌 Phương pháp trade gợi ý (M30/20 nến):")
            for s in tm_lines:
                lines.append(f"- {s}")
    lines.append("")
    lines.append(f"🎯 Khuyến nghị: {rec}")
    lines.append(f"Độ tin cậy: {stars_txt} ({max(1, min(5, stars))}/5)")
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
