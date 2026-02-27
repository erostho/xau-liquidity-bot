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
    spread: float = 0.0


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
                spread=_f(c.get("spread"), 0.0),
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
        out.append(Candle(ts=ts_i, open=_fa("open"), high=_fa("high"), low=_fa("low"), close=_fa("close"), volume=vol_f, spread=_fa("spread", 0.0)))
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
# EXTRA MODULES: Volume / Candle / Divergence
# =========================

def _sma(xs: List[float], n: int) -> Optional[float]:
    xs = [float(x) for x in (xs or []) if x is not None]
    if len(xs) < n or n <= 0:
        return None
    return sum(xs[-n:]) / float(n)

def _vol_quality(candles: List[Candle], n: int = 20) -> Dict[str, Any]:
    """
    Volume quality: so sánh volume nến đã đóng gần nhất với SMA/Median của n nến trước đó.
    Return: {"state": "HIGH"/"NORMAL"/"LOW"/"N/A", "ratio": float|None}
    """
    if not candles or len(candles) < n + 3:
        return {"state": "N/A", "ratio": None}

    closed = candles[:-1]  # bỏ nến đang chạy
    use = closed[-(n+1):]  # n+1 để lấy last + n trước
    vols = [max(0.0, float(getattr(c, "volume", 0.0) or 0.0)) for c in use]
    v_last = vols[-1]
    base = _sma(vols[:-1], n)  # SMA của n nến trước đó
    if base is None or base <= 0:
        return {"state": "N/A", "ratio": None}

    ratio = v_last / base
    if ratio >= 1.4:
        st = "HIGH"
    elif ratio <= 0.75:
        st = "LOW"
    else:
        st = "NORMAL"
    return {"state": st, "ratio": ratio}

def _candle_patterns(candles: List[Candle]) -> Dict[str, Any]:
    """
    Nhận diện 1 số mẫu nến quan trọng để review/confirm:
    - engulfing (bull/bear)
    - strong rejection (pinbar kiểu chuẩn)
    Dùng 2 nến đã đóng gần nhất.
    """
    if not candles or len(candles) < 4:
        return {"engulf": None, "rejection": None, "txt": "N/A"}

    c1 = candles[-2]  # nến đã đóng gần nhất
    c0 = candles[-3]  # nến trước đó

    def _rng(c: Candle) -> float:
        return max(1e-9, c.high - c.low)

    # Engulfing (body engulf body)
    b0_hi = max(c0.open, c0.close)
    b0_lo = min(c0.open, c0.close)
    b1_hi = max(c1.open, c1.close)
    b1_lo = min(c1.open, c1.close)

    bull_engulf = (c1.close > c1.open) and (b1_hi >= b0_hi) and (b1_lo <= b0_lo)
    bear_engulf = (c1.close < c1.open) and (b1_hi >= b0_hi) and (b1_lo <= b0_lo)

    engulf = "BULL" if bull_engulf else ("BEAR" if bear_engulf else None)

    # Rejection / pinbar “chuẩn” (wick dài, thân nhỏ)
    rng1 = _rng(c1)
    body1 = abs(c1.close - c1.open)
    up_w = c1.high - max(c1.open, c1.close)
    lo_w = min(c1.open, c1.close) - c1.low

    upper_reject = (up_w / rng1 >= 0.50) and (body1 / rng1 <= 0.35)
    lower_reject = (lo_w / rng1 >= 0.50) and (body1 / rng1 <= 0.35)

    rejection = "UPPER" if upper_reject else ("LOWER" if lower_reject else None)

    txt_parts = []
    if engulf:
        txt_parts.append(f"Engulfing={engulf}")
    if rejection:
        txt_parts.append(f"Rejection={rejection}")
    txt = " | ".join(txt_parts) if txt_parts else "None"

    return {"engulf": engulf, "rejection": rejection, "txt": txt}

def _find_swings(values: List[float], left: int = 2, right: int = 2) -> Dict[str, List[int]]:
    """
    Trả về index swing highs/lows đơn giản trên chuỗi values.
    """
    n = len(values)
    highs, lows = [], []
    if n < (left + right + 3):
        return {"highs": highs, "lows": lows}

    for i in range(left, n - right):
        v = values[i]
        if all(v > values[i - j] for j in range(1, left + 1)) and all(v > values[i + j] for j in range(1, right + 1)):
            highs.append(i)
        if all(v < values[i - j] for j in range(1, left + 1)) and all(v < values[i + j] for j in range(1, right + 1)):
            lows.append(i)
    return {"highs": highs, "lows": lows}

def _rsi_series(values: List[float], period: int = 14) -> List[float]:
    """
    RSI series để detect divergence (tránh chỉ 1 số).
    """
    if not values or len(values) < period + 2:
        return []
    values = [float(x) for x in values]
    gains, losses = [], []
    for i in range(1, len(values)):
        ch = values[i] - values[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out = [50.0] * (period)  # pad
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        out.append(rsi)
    # align length to closes
    out = [out[0]] + out  # because gains len = closes-1
    return out[:len(values)]

def _divergence_rsi(candles: List[Candle], period: int = 14, lookback: int = 50) -> Dict[str, Any]:
    """
    Divergence RSI đơn giản:
    - Bearish div: price HH nhưng RSI LH (tín hiệu nên chốt BUY / cẩn thận SELL reversal)
    - Bullish div: price LL nhưng RSI HL (tín hiệu nên chốt SELL / cẩn thận BUY reversal)
    Dùng closes trong lookback nến đã đóng.
    """
    if not candles or len(candles) < lookback + period + 5:
        return {"bear": False, "bull": False, "txt": "N/A"}

    closed = candles[:-1]
    use = closed[-lookback:]
    closes = [c.close for c in use]
    rsis = _rsi_series(closes, period=period)
    if not rsis or len(rsis) != len(closes):
        return {"bear": False, "bull": False, "txt": "N/A"}

    swings = _find_swings(closes, left=2, right=2)
    hs = swings["highs"]
    ls = swings["lows"]

    bear = False
    bull = False

    # bearish: last 2 swing highs
    if len(hs) >= 2:
        i1, i2 = hs[-2], hs[-1]
        p1, p2 = closes[i1], closes[i2]
        r1, r2 = rsis[i1], rsis[i2]
        if p2 > p1 and r2 < r1:
            bear = True

    # bullish: last 2 swing lows
    if len(ls) >= 2:
        i1, i2 = ls[-2], ls[-1]
        p1, p2 = closes[i1], closes[i2]
        r1, r2 = rsis[i1], rsis[i2]
        if p2 < p1 and r2 > r1:
            bull = True

    if bear and bull:
        txt = "RSI divergence: MIXED"
    elif bear:
        txt = "RSI divergence: BEARISH (đà lên yếu dần)"
    elif bull:
        txt = "RSI divergence: BULLISH (đà xuống yếu dần)"
    else:
        txt = "RSI divergence: None"

    return {"bear": bear, "bull": bull, "txt": txt}

# =========================
# SYMBOL PROFILES (XAG vs XAU/BTC)
# =========================
SYMBOL_PROFILE = {
    "XAG": {
        # wick must be big (XAG hay quét sâu)
        "sweep_wick_min": 0.42,
        # nến sweep phải đóng lại vào trong range tối thiểu bao nhiêu %
        "close_back_ratio": 0.22,
        # volume confirm (tick_volume cũng ok nhưng chỉ "bonus")
        "vol_spike_k": 1.5,
        # buffer để tránh "chạm nhẹ/spread"
        "buf_atr_k": 0.20,
        # spring follow-through tối thiểu
        "spring_follow_k": 0.50,
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
    GỢI Ý NGẮN HẠN (M15):
    - Quan sát breakout / chờ kèo chính
    - + SCALE NHANH (hớt sóng) nếu đủ điều kiện
    """

    if not m15 or len(m15) < 30:
        return ["- Chưa đủ dữ liệu M15 → CHỜ"]

    lines: list[str] = []

    # ====== PREP DATA ======
    closed = m15[:-1] if len(m15) > 1 else m15
    use = closed[-30:]
    cur = use[-1].close

    hi = max(c.high for c in use)
    lo = min(c.low for c in use)
    rng = max(1e-9, hi - lo)

    atr15 = _atr(closed, 14)
    if atr15 is None or atr15 <= 0:
        return ["- ATR M15 chưa sẵn sàng → CHỜ"]

    rsi15 = _rsi([c.close for c in closed], 14) or 50.0
    rej = _is_rejection(use[-1])

    # ====== PHẦN 1: GỢI Ý QUAN SÁT (LOGIC CŨ – GIỮ) ======
    pos = (cur - lo) / rng * 100.0
    buf = 0.20 * atr15

    lines.append(f"- Range 30 nến M15: {_fmt(lo)} – {_fmt(hi)}.")
    lines.append(f"- Giá hiện tại: {_fmt(cur)} (~{pos:.0f}% trong range).")

    buy_trig = hi + buf
    sell_trig = lo - buf

    lines.append(
        f"- Quan sát breakout: M15 đóng > {_fmt(buy_trig)} → canh BUY | "
        f"M15 đóng < {_fmt(sell_trig)} → canh SELL."
    )

    # ====== PHẦN 2: SCALE NHANH (HỚT SÓNG) ======
    # Điều kiện: chỉ scale khi KHÔNG có trend mạnh
    allow_scale = False  # DISABLED: scale/scalp branch turned off

    scale_buf = 0.15 * atr15

    # ---- SCALE BUY ----
    if allow_scale and cur <= lo + scale_buf and rej["lower_reject"] and rsi15 < 45:
        entry = cur
        sl = cur - 0.4 * atr15
        tp = cur + 0.7 * atr15

        lines.append("")
        lines.append("⚡ GỢI Ý SCALE NHANH (M15 – hớt sóng):")
        lines.append(f"- BUY quanh {_fmt(entry)}")
        lines.append(f"- SL: {_fmt(sl)} | TP nhanh: {_fmt(tp)}")
        lines.append("- Lệnh ngắn, vào ra nhanh, KHÔNG gồng.")

    # ---- SCALE SELL ----
    elif allow_scale and cur >= hi - scale_buf and rej["upper_reject"] and rsi15 > 55:
        entry = cur
        sl = cur + 0.4 * atr15
        tp = cur - 0.7 * atr15

        lines.append("")
        lines.append("⚡ GỢI Ý SCALE NHANH (M15 – hớt sóng):")
        lines.append(f"- SELL quanh {_fmt(entry)}")
        lines.append(f"- SL: {_fmt(sl)} | TP nhanh: {_fmt(tp)}")
        lines.append("- Lệnh ngắn, vào ra nhanh, KHÔNG gồng.")

    # ====== PHẦN 3: NHẮC TREND LỚN (THAM KHẢO) ======
    if h1_trend in ("bullish", "bearish"):
        lines.append("")
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


def _sma(vals, n: int):
    if not vals or n <= 0 or len(vals) < n:
        return None
    return sum(vals[-n:]) / float(n)

def _trend_tag(candles, n_fast=20, n_slow=50):
    """Return a simple structure tag: HH-HL (bull), LL-LH (bear), or TRANSITION.
    Uses SMA fast/slow + fast slope as a lightweight proxy (robust when swing logic is missing)."""
    try:
        if not candles or len(candles) < n_fast + 2:
            return "n/a"
        closes = [float(c.close) for c in candles if c is not None]
        if len(closes) < n_fast + 2:
            return "n/a"
        sma_f = _sma(closes, n_fast)
        sma_s = _sma(closes, n_slow) if len(closes) >= n_slow else None
        # slope of fast SMA (last 3 points approx)
        sma_f_prev = sum(closes[-(n_fast+3):-(3)]) / float(n_fast) if len(closes) >= n_fast + 3 else None
        slope_up = (sma_f is not None and sma_f_prev is not None and sma_f > sma_f_prev)
        slope_dn = (sma_f is not None and sma_f_prev is not None and sma_f < sma_f_prev)
        last = closes[-1]
        if sma_f is None:
            return "n/a"
        bull = last >= sma_f and (sma_s is None or sma_f >= sma_s) and slope_up
        bear = last <= sma_f and (sma_s is None or sma_f <= sma_s) and slope_dn
        if bull:
            return "HH-HL"
        if bear:
            return "LL-LH"
        return "TRANSITION"
    except Exception:
        return "n/a"

def _range_high_low(candles, n: int):
    if not candles:
        return (None, None)
    cc = candles[-n:] if len(candles) > n else candles
    try:
        hi = max(float(c.high) for c in cc)
        lo = min(float(c.low) for c in cc)
        return (lo, hi)
    except Exception:
        return (None, None)

def _inject_meta_structure_and_levels(base: dict, m15, m30, h1, h4):
    """Ensure base['meta']['structure'] + base['meta']['key_levels'] exist for Telegram template,
    even when stars are low / trade plan is suppressed."""
    meta = base.get("meta") or {}
    base["meta"] = meta

    # Structure tags
    struct = meta.get("structure") or {}
    struct.setdefault("H4", _trend_tag(h4))
    struct.setdefault("H1", _trend_tag(h1))
    struct.setdefault("M15", _trend_tag(m15))
    meta["structure"] = struct

    # Key levels (flattened keys expected by format_signal)
    kl = meta.get("key_levels") or {}

    lo_h1, hi_h1 = _range_high_low(h1, 48)
    lo_h4, hi_h4 = _range_high_low(h4, 60)
    if hi_h1 is not None:
        kl.setdefault("H1_HH", hi_h1)
    elif hi_h4 is not None:
        kl.setdefault("H1_HH", hi_h4)

    if lo_h1 is not None:
        kl.setdefault("H1_HL", lo_h1)
    elif lo_h4 is not None:
        kl.setdefault("H1_HL", lo_h4)

    lo_m15, hi_m15 = _range_high_low(m15, 32)
    if lo_m15 is not None and hi_m15 is not None:
        kl.setdefault("M15_RANGE_LOW", lo_m15)
        kl.setdefault("M15_RANGE_HIGH", hi_m15)
        kl.setdefault("M15_BOS_LEVEL", hi_m15)
        kl.setdefault("M15_PB_EXTREME", lo_m15)

    meta["key_levels"] = kl

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _fmt2(x: float) -> str:
    """Format price like MT5 mobile (2 decimals)."""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "..."

def _structure_from_swings(candles: Sequence[Any], lookback: int = 220) -> Dict[str, Any]:
    """Return structure tag + key swing levels (HH/HL/LH/LL) from candles.
    Uses simple swing detection on HIGHs and LOWs.
    """
    if not candles:
        return {"tag": "n/a", "hh": None, "hl": None, "lh": None, "ll": None,
                "last_high": None, "prev_high": None, "last_low": None, "prev_low": None}

    c = list(candles)[-lookback:]
    highs = [float(getattr(x, "high", 0.0)) for x in c]
    lows  = [float(getattr(x, "low", 0.0)) for x in c]

    sh = _find_swings(highs, left=2, right=2).get("highs", [])
    sl = _find_swings(lows,  left=2, right=2).get("lows", [])

    # take last 2 swing highs/lows if possible
    last_hi = prev_hi = None
    last_lo = prev_lo = None

    if len(sh) >= 1:
        last_hi = highs[sh[-1]]
    if len(sh) >= 2:
        prev_hi = highs[sh[-2]]

    if len(sl) >= 1:
        last_lo = lows[sl[-1]]
    if len(sl) >= 2:
        prev_lo = lows[sl[-2]]

    # classify current swing high/low relative to previous
    hh = lh = hl = ll = None
    if last_hi is not None and prev_hi is not None:
        if last_hi > prev_hi:
            hh = last_hi
        elif last_hi < prev_hi:
            lh = last_hi

    if last_lo is not None and prev_lo is not None:
        if last_lo > prev_lo:
            hl = last_lo
        elif last_lo < prev_lo:
            ll = last_lo

    # overall tag
    tag = "TRANSITION"
    if (hh is not None) and (hl is not None):
        tag = "HH–HL"
    elif (lh is not None) and (ll is not None):
        tag = "LH–LL"
    elif (last_hi is not None and prev_hi is not None and last_lo is not None and prev_lo is not None):
        # mixed
        tag = "TRANSITION"
    else:
        tag = "n/a"

    return {
        "tag": tag,
        "hh": hh, "hl": hl, "lh": lh, "ll": ll,
        "last_high": last_hi, "prev_high": prev_hi,
        "last_low": last_lo, "prev_low": prev_lo,
    }

def _m15_key_levels(m15c: Sequence[Any], bias_side: str, lookback: int = 80) -> Dict[str, Any]:
    """Key M15 levels for BOS + pullback."""
    if not m15c or len(m15c) < 30:
        return {"bos_level": None, "pullback_extreme": None, "tag": "n/a"}
    c = list(m15c)[-(lookback+5):]
    closed = c[:-2]  # avoid current forming candles
    highs = [float(x.high) for x in closed]
    lows  = [float(x.low) for x in closed]

    # swing-based structure tag for M15
    struct = _structure_from_swings(closed, lookback=min(len(closed), 160))
    tag = struct.get("tag", "n/a")

    # BOS level: swing level that should be broken in the direction of bias
    bos_level = None
    if bias_side == "BUY":
        sh = _find_swings(highs, left=2, right=2).get("highs", [])
        if sh:
            bos_level = highs[sh[-1]]
    elif bias_side == "SELL":
        sl = _find_swings(lows, left=2, right=2).get("lows", [])
        if sl:
            bos_level = lows[sl[-1]]

    # pullback extreme (for SL / "forming HL/LH")
    pullback_extreme = None
    if bias_side == "BUY":
        pullback_extreme = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    elif bias_side == "SELL":
        pullback_extreme = max(highs[-20:]) if len(highs) >= 20 else max(highs)

    return {"bos_level": bos_level, "pullback_extreme": pullback_extreme, "tag": tag}


# =========================
# PRO Analyzer (MUST be named analyze_pro for main.py import)
# =========================

def _range_levels(candles: Sequence[Any], n: int = 20) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (range_low, range_high, last_close) for last n candles."""
    if not candles or len(candles) < max(10, n):
        return None, None, None
    c = list(candles)[-n:]
    hi = max(float(_c_val(x, "high", 0.0) or 0.0) for x in c)
    lo = min(float(_c_val(x, "low", 0.0) or 0.0) for x in c)
    last = float(_c_val(c[-1], "close", 0.0) or 0.0)
    return lo, hi, last

def _where_wait_text(m15c: Sequence[Any], bias_side: str) -> Tuple[str, str]:
    lo, hi, last = _range_levels(m15c, n=20)
    if lo is None or hi is None:
        return ("Không đủ nến M15 để định vị.", "Chờ có thêm dữ liệu.")
    span = max(hi - lo, 1e-9)
    pos = (last - lo) / span  # 0..1

    if pos <= 0.25:
        where = f"Đang gần hỗ trợ (range low) {last:.2f} ~ {lo:.2f}"
    elif pos >= 0.75:
        where = f"Đang gần kháng cự (range high) {last:.2f} ~ {hi:.2f}"
    else:
        where = f"Đang ở giữa range {lo:.2f}–{hi:.2f} (nhiễu)"

    if bias_side == "BUY":
        wait = f"Chờ BOS↑: M15 đóng trên {hi:.2f} (tốt nhất retest giữ được)."
    elif bias_side == "SELL":
        wait = f"Chờ BOS↓: M15 đóng dưới {lo:.2f} (tốt nhất retest giữ được)."
    else:
        wait = f"Chờ break range: trên {hi:.2f} hoặc dưới {lo:.2f}."
    return where, wait


def analyze_pro(symbol: str, m15: Sequence[dict], m30: Sequence[dict], h1: Sequence[dict], h4: Sequence[dict]) -> dict:
    """PRO analysis: Signal=M15, Entry=M30, Confirm=H1.

    Patch:
    - Context luôn có: Thị trường (TĂNG MẠNH/GIẢM MẠNH/SIDEWAY) + H1 trend
    - Liquidity WARNING (chưa quét nhưng nguy cơ quét) -> đẩy vào context_lines (để main.py dễ ưu tiên gửi)
    - Quét xong -> POST-SWEEP -> CHỜ CẤU TRÚC (HL/LH + BOS) rồi mới cho BUY/SELL
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

    # IMPORTANT: define these early so any early-return or exception path won't crash the caller
    # (Python 3.11+ can raise UnboundLocalError if referenced before assignment).
    entry_major = sl_major = tp1_major = tp2_major = None
    entry_minor = sl_minor = tp1_minor = tp2_minor = None

    context_lines = base["context_lines"]
    position_lines = base.get("position_lines", [])
    liquidity_lines = base["liquidity_lines"]
    quality_lines = base["quality_lines"]
    notes = base.setdefault("notes", [])
    score = 0

    # ---- Safety / normalize candles
    if not m15 or not m30 or not h1:
        base["note_lines"].append("⚠️ Thiếu dữ liệu M15/M30/H1 → không phân tích được.")
        base["short_hint"] = ["- Chưa đủ dữ liệu → CHỜ KÈO"]
        # Context vẫn phải có để telegram không bị n/a trống
        base["context_lines"] = ["Thị trường: n/a", "H1: n/a"]
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
        return base

    m15c = _safe_candles(m15)
    m30c = _safe_candles(m30)
    h1c = _safe_candles(h1)
    h4c = _safe_candles(h4)

    if len(m15c) < 20 or len(m30c) < 5 or len(h1c) < 5 or len(h4c) < 5:
        base["note_lines"].append("⚠️ Dữ liệu candles chưa đủ → kết quả có thể thiếu chính xác (vẫn hiển thị).")

    last15 = m15c[-1]
    last30 = m30c[-1]
    last_close_15 = float(last15.close)

    # Indicators (M15)
    m15_closes = [c.close for c in m15c]
    atr15 = _atr(m15c, 14) or 0.0
    rsi15 = _rsi(m15_closes, 14) or 50.0

    # Trends (H1 + M30) (dùng _trend_label có sẵn)
    h1_trend = _trend_label(h1c)    # bullish / bearish / sideways
    m30_trend = _trend_label(m30c)  # bullish / bearish / sideways
    # ===== EXTRA: Volume / Candle / Divergence (for PRO review) =====
    volq = _vol_quality(m15c, n=20)
    cpat = _candle_patterns(m15c)
    div  = _divergence_rsi(m15c, period=14, lookback=50)

    # nhét vào meta để main.py / review lệnh dùng lại
    base["meta"]["volq"] = volq
    base["meta"]["candle"] = cpat
    base["meta"]["div"] = div

    # show vào quality_lines (đọc phát hiểu)
    if volq["state"] != "N/A":
        quality_lines.append(f"Volume: {volq['state']} (x{volq['ratio']:.2f} vs SMA20)")
        # volume cao → thêm điểm chất lượng (đỡ fake breakout)
        if volq["state"] == "HIGH":
            score += 1
        elif volq["state"] == "LOW":
            notes.append("⚠️ Volume thấp → dễ fake move (ưu tiên TP nhanh / không add).")

    # candle reaction
    if cpat.get("engulf") or cpat.get("rejection"):
        quality_lines.append(f"Candle: {cpat['txt']}")
        score += 1

    # divergence
    if div.get("bear") or div.get("bull"):
        quality_lines.append(div["txt"])
        score += 1

    # ====== Market state: chỉ 3 trạng thái (đúng ý mày) ======
    # spike volatility (M15): range 20 > 1.35 * range 80
    ranges20 = [c.high - c.low for c in m15c[-20:]] if len(m15c) >= 20 else [c.high - c.low for c in m15c]
    ranges80 = [c.high - c.low for c in m15c[-80:]] if len(m15c) >= 80 else [c.high - c.low for c in m15c]
    avg20 = sum(ranges20) / max(1, len(ranges20))
    avg80 = sum(ranges80) / max(1, len(ranges80))
    spike = (avg20 > 1.35 * avg80) if avg80 > 0 else False

    # weakening trend trên H1 dựa EMA20-EMA50 (đã có đoạn dưới, nhưng ta cần dùng sớm)
    h1_closes = [c.close for c in h1c]
    ema20_h1 = _ema(h1_closes, 20)
    ema50_h1 = _ema(h1_closes, 50)

    weakening = False
    if ema20_h1 and ema50_h1 and len(ema20_h1) >= 6 and len(ema50_h1) >= 6:
        sep_now = ema20_h1[-1] - ema50_h1[-1]
        sep_prev = ema20_h1[-6] - ema50_h1[-6]
        if h1_trend == "bullish" and sep_now < sep_prev:
            weakening = True
        if h1_trend == "bearish" and sep_now > sep_prev:
            weakening = True

    # chỉ 3 nhãn:
    if h1_trend == "bullish" and spike and not weakening:
        market_state = "TĂNG MẠNH"
    elif h1_trend == "bearish" and spike and not weakening:
        market_state = "GIẢM MẠNH"
    else:
        market_state = "SIDEWAY"

    # ====== Context luôn có (không còn n/a vô nghĩa) ======
    context_lines.append(f"Thị trường: {market_state}")
    context_lines.append(f"H1: {h1_trend}")

    # --- GỢI Ý NGẮN HẠN (dựa 30 nến M15 gần nhất)
    try:
        base["short_hint"] = _build_short_hint_m15(m15c, h1_trend, m30_trend)
    except Exception:
        base["short_hint"] = []

    # --- Trade method suggestion (20 nến M30)
    m30_closed = m30c[:-1] if len(m30c) > 1 else m30c
    atr30 = _atr(m30_closed, 14)
    base["trade_method"] = _pick_trade_method_m30(m30c, atr30)

    # ===== Key levels =====
    sh15 = _swing_high(m15c, 80)
    sl15 = _swing_low(m15c, 80)
    sh30 = _swing_high(m30c, 80)
    sl30 = _swing_low(m30c, 80)
    sh1  = _swing_high(h1c, 80)
    sl1  = _swing_low(h1c, 80)

    levels_info: List[Tuple[float, str]] = []
    if sh15 is not None: levels_info.append((float(sh15), "M15 Swing High (đỉnh gần)"))
    if sl15 is not None: levels_info.append((float(sl15), "M15 Swing Low (đáy gần)"))
    if sh30 is not None: levels_info.append((float(sh30), "M30 Swing High (kháng cự)"))
    if sl30 is not None: levels_info.append((float(sl30), "M30 Swing Low (hỗ trợ)"))
    if sh1 is not None:  levels_info.append((float(sh1),  "H1 Swing High (kháng cự lớn)"))
    if sl1 is not None:  levels_info.append((float(sl1),  "H1 Swing Low (hỗ trợ lớn)"))

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

    # ===== Observation triggers =====
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

    # ===== Helper: slope micro (dùng cho WARNING) =====
    slope = 0.0
    closed15 = m15c[:-1] if len(m15c) > 1 else m15c
    use = closed15[-20:] if len(closed15) >= 20 else closed15
    if len(use) >= 20:
        last10 = [c.close for c in use[-10:]]
        prev10 = [c.close for c in use[-20:-10]]
        slope = (sum(last10)/10.0) - (sum(prev10)/10.0)

    # ===== 1) LIQUIDITY WARNING (chưa quét nhưng nguy hiểm) =====
    # Nguy cơ quét khi giá tiệm cận swing + có động lượng (rsi/slope) -> in vào context_lines
    def _liquidity_warning_lines(cur_price: float) -> List[str]:
        out = []
        if atr15 <= 0:
            return out
        buf = 0.30 * atr15  # vùng "dễ quét"

        # WARNING quét đỉnh
        if sh15 is not None and cur_price >= float(sh15) - buf:
            if rsi15 >= 60 or slope > 0.20 * atr15:
                out.append(f"⚠️ Liquidity WARNING: gần đỉnh {float(sh15):.2f} → dễ QUÉT ĐỈNH rồi đảo chiều.")

        # WARNING quét đáy
        if sl15 is not None and cur_price <= float(sl15) + buf:
            if rsi15 <= 40 or slope < -0.20 * atr15:
                out.append(f"⚠️ Liquidity WARNING: gần đáy {float(sl15):.2f} → dễ QUÉT ĐÁY rồi bật lại.")

        return out

    lw = _liquidity_warning_lines(cur)
    if lw:
        context_lines.extend(lw)

    # ===== Rejection =====
    rej = _is_rejection(last15)

    # ===== LIQUIDITY: sweep / spring =====
    use15 = closed15[-30:] if len(closed15) >= 30 else closed15
    range15_low = min(c.low for c in use15) if use15 else (sl15 or last_close_15)
    range15_high = max(c.high for c in use15) if use15 else (sh15 or last_close_15)

    sweep_sell = detect_sweep(m15c, side="SELL", level=float(sh15) if sh15 else float(range15_high), atr=atr15, symbol=symbol)
    sweep_buy  = detect_sweep(m15c, side="BUY",  level=float(sl15) if sl15 else float(range15_low),  atr=atr15, symbol=symbol)

    spring_buy  = detect_spring(m15c, side="BUY",  range_low=float(range15_low),  range_high=float(range15_high), atr=atr15, symbol=symbol)
    spring_sell = detect_spring(m15c, side="SELL", range_low=float(range15_low),  range_high=float(range15_high), atr=atr15, symbol=symbol)

    liq_sell = bool(sweep_sell.get("ok")) or bool(spring_sell.get("ok"))
    liq_buy  = bool(sweep_buy.get("ok"))  or bool(spring_buy.get("ok"))

    liquidity_lines = []
    if sweep_sell.get("ok"):
        vtxt = " +VOL" if sweep_sell.get("vol_ok") else ""
        liquidity_lines.append(f"🔴 Sweep HIGH (quét đỉnh){vtxt}: chọc {_fmt(sweep_sell['level'])} rồi đóng xuống lại.")
        score += 1

    if sweep_buy.get("ok"):
        vtxt = " +VOL" if sweep_buy.get("vol_ok") else ""
        liquidity_lines.append(f"🟢 Sweep LOW (quét đáy){vtxt}: chọc {_fmt(sweep_buy['level'])} rồi đóng lên lại.")
        score += 1

    if spring_buy.get("ok"):
        vtxt = " +VOL" if spring_buy.get("vol_ok") else ""
        liquidity_lines.append("🟢 SPRING (false break đáy){vtxt}: phá range_low rồi kéo lên + follow-through.")
        score += 1

    if spring_sell.get("ok"):
        vtxt = " +VOL" if spring_sell.get("vol_ok") else ""
        liquidity_lines.append("🔴 UPTHRUST (false break đỉnh){vtxt}: phá range_high rồi kéo xuống + follow-through.")
        score += 1

    if not liquidity_lines:
        liquidity_lines.append("Chưa thấy sweep/spring rõ (liquidity proxy).")

    # ===== 2) POST-SWEEP: Quét xong -> CHỜ CẤU TRÚC =====
    # Cấu trúc rõ để biết khi nào vào:
    # BUY: HL + BOS_UP
    # SELL: LH + BOS_DN
    def _post_sweep_structure_state(side: str) -> Tuple[bool, List[str]]:
        """
        Return (ok_to_trade, explain_lines)
        """
        explain = []
        if atr15 <= 0 or len(closed15) < 16:
            explain.append("POST-SWEEP: thiếu dữ liệu để xác nhận cấu trúc → CHỜ.")
            return False, explain

        # dùng 10 nến đóng gần nhất để kiểm tra cấu trúc
        # prev5 = 5 nến trước, last5 = 5 nến sau
        prev5 = closed15[-11:-6]
        last5 = closed15[-6:-1]
        if len(prev5) < 5 or len(last5) < 5:
            explain.append("POST-SWEEP: thiếu đủ 10 nến đóng để xét HL/LH → CHỜ.")
            return False, explain

        buf = 0.10 * atr15  # buffer nhỏ để tránh nhiễu

        prev_low = min(c.low for c in prev5)
        last_low = min(c.low for c in last5)
        prev_high = max(c.high for c in prev5)
        last_high = max(c.high for c in last5)

        last_close = float(closed15[-1].close)

        if side == "BUY":
            hl = (last_low > prev_low + buf)
            bos_up = (last_close > prev_high + buf)

            explain.append(f"POST-SWEEP BUY: chờ HL + BOS.")
            explain.append(f"- HL (Higher-Low): đáy 5 nến mới > đáy 5 nến trước (buf~{_fmt(buf)}).")
            explain.append(f"- BOS: M15 đóng > đỉnh 5 nến trước.")
            explain.append(f"Trạng thái: HL={'OK' if hl else 'NO'} | BOS={'OK' if bos_up else 'NO'}.")

            return (hl and bos_up), explain

        else:  # SELL
            lh = (last_high < prev_high - buf)
            bos_dn = (last_close < prev_low - buf)

            explain.append(f"POST-SWEEP SELL: chờ LH + BOS.")
            explain.append(f"- LH (Lower-High): đỉnh 5 nến mới < đỉnh 5 nến trước (buf~{_fmt(buf)}).")
            explain.append(f"- BOS: M15 đóng < đáy 5 nến trước.")
            explain.append(f"Trạng thái: LH={'OK' if lh else 'NO'} | BOS={'OK' if bos_dn else 'NO'}.")

            return (lh and bos_dn), explain

    # Nếu có quét -> vào POST-SWEEP mode (báo context) + KHÓA vào lệnh cho tới khi có cấu trúc
    post_sweep_buy = bool(sweep_buy.get("ok")) or bool(spring_buy.get("ok"))
    post_sweep_sell = bool(sweep_sell.get("ok")) or bool(spring_sell.get("ok"))
    if post_sweep_buy or post_sweep_sell:
        context_lines.append("POST-SWEEP: Đã xảy ra QUÉT thanh khoản → KHÔNG vào ngay, chờ cấu trúc.")
        # (để telegram đọc là biết)
        if post_sweep_buy:
            ok_struct, explain = _post_sweep_structure_state("BUY")
            notes.extend(explain)
            if not ok_struct:
                base.update({
                    "context_lines": context_lines,
                    "position_lines": position_lines,
                    "liquidity_lines": liquidity_lines,
                    "quality_lines": quality_lines + [f"RSI(14) M15: {_fmt(rsi15)}", f"ATR(14) M15: ~{_fmt(atr15)}", "RR ~ 1:2 (mục tiêu)"],
                    "recommendation": "CHỜ",
                    "stars": 2,
                    "notes": notes + ["➡️ Khi HL + BOS xuất hiện (trạng thái OK/OK) → mới canh BUY theo H1 + chờ M30 confirm."],
                })
                _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
                return base

        if post_sweep_sell:
            ok_struct, explain = _post_sweep_structure_state("SELL")
            notes.extend(explain)
            if not ok_struct:
                base.update({
                    "context_lines": context_lines,
                    "position_lines": position_lines,
                    "liquidity_lines": liquidity_lines,
                    "quality_lines": quality_lines + [f"RSI(14) M15: {_fmt(rsi15)}", f"ATR(14) M15: ~{_fmt(atr15)}", "RR ~ 1:2 (mục tiêu)"],
                    "recommendation": "CHỜ",
                    "stars": 2,
                    "notes": notes + ["➡️ Khi LH + BOS xuất hiện (trạng thái OK/OK) → mới canh SELL theo H1 + chờ M30 confirm."],
                })
                _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
                return base

    # ===== Quality =====
    if rej["upper_reject"] or rej["lower_reject"]:
        quality_lines.append("Nến từ chối rõ")
        score += 1

    quality_lines.append(f"RSI(14) M15: {_fmt(rsi15)}")
    quality_lines.append(f"ATR(14) M15: ~{_fmt(atr15)}")
    score += 1

    # ===== Lower-high-ish (giữ logic cũ) =====
    lower_highish = False
    if len(m15c) >= 30:
        recent_high = max(c.high for c in m15c[-10:])
        prev_high   = max(c.high for c in m15c[-30:-10])
        if recent_high <= prev_high:
            lower_highish = True

    # ===== Bias decision (giữ logic cũ) =====
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

    # ---- defaults (avoid UnboundLocalError when structure/bias is n/a) ----
    trade_mode = "HALF"  # default; will be overwritten once scoring is computed
    entry_major = sl_major = tp1_major = tp2_major = None
    entry_minor = sl_minor = tp1_minor = tp2_minor = None

    if bias is None:
        # --------------------
        # Mode-based plan selection:
        # - FULL  => MAJOR plan (H1 + H4 confluence): wider levels, wait for MAJOR BOS then retest
        # - HALF  => MINOR plan (M15): shorter TP/SL, only if H1 HL/LH (major trend guard) is not broken
        # --------------------
        entry_frame = "M15"
        plan_tag = "MINOR"
        # defaults (HALF / WAIT): use minor plan
        entry, sl, tp1, tp2 = entry_minor, sl_minor, tp1_minor, tp2_minor

        if trade_mode == "FULL":
            entry_frame = "H1"
            plan_tag = "MAJOR"

            # Override "wait_for" for MAJOR: wait M15 close to break H1 HH/LL (and ideally retest).
            major_level = h1_struct.get("hh") if bias_side == "BUY" else h1_struct.get("ll")
            if major_level is not None:
                base.setdefault("meta", {})["wait_for"] = (
                    "Chờ BOS MAJOR: M15 đóng "
                    + ("trên " if bias_side == "BUY" else "dưới ")
                    + f"{float(major_level):.2f}"
                    + " (tốt nhất retest giữ được)."
                )

            atr_h1 = _atr(h1c, 14) or (atr15 * 4.0)  # fallback: ~4x M15 ATR
            major_bos = h1_struct.get("hh") if bias_side == "BUY" else h1_struct.get("ll")
            major_sl_level = h1_struct.get("hl") if bias_side == "BUY" else h1_struct.get("lh")

            # If structure fields are missing, fall back to recent H1 extremes
            if major_bos is None:
                major_bos = (max(c.high for c in h1c[-50:]) if (bias_side == "BUY" and len(h1c) >= 10) else
                             min(c.low for c in h1c[-50:]) if (bias_side == "SELL" and len(h1c) >= 10) else last_close)
            if major_sl_level is None:
                major_sl_level = (min(c.low for c in h1c[-50:]) if bias_side == "BUY" else
                                  max(c.high for c in h1c[-50:]) if bias_side == "SELL" else last_close)

            # Entry for MAJOR = retest around MAJOR BOS (safer than chasing)
            entry_major = float(major_bos)
            buf = float(atr_h1 * 0.35)
            if bias_side == "BUY":
                sl_major = float(major_sl_level) - buf
                risk = max(atr_h1 * 0.8, entry_major - sl_major)
                tp1_major = entry_major + risk * 1.0
                tp2_major = entry_major + risk * 1.6
            else:
                sl_major = float(major_sl_level) + buf
                risk = max(atr_h1 * 0.8, sl_major - entry_major)
                tp1_major = entry_major - risk * 1.0
                tp2_major = entry_major - risk * 1.6

            entry, sl, tp1, tp2 = entry_major, sl_major, tp1_major, tp2_major

        base["meta"]["plan_tag"] = plan_tag
        base["meta"]["entry_frame"] = entry_frame

        base.update({
            "context_lines": context_lines,
            "position_lines": position_lines,
            "liquidity_lines": liquidity_lines,
            "quality_lines": quality_lines + ["RR ~ 1:2 (mục tiêu)"],
            "recommendation": "CHỜ",
            "stars": 1,
            "notes": ["Chưa đủ điều kiện vào kèo. Chờ thêm nến xác nhận/retest."],
        })
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
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
            _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
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
            _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
            return base

    recommendation = "🔴 SELL" if bias == "SELL" else "🟢 BUY"

    # Entry logic (giữ logic cũ)
    RETEST_K = float(os.getenv("RETEST_K", "0.35"))
    RETEST_K = max(0.15, min(0.80, RETEST_K))

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
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
        return base
    # ===== PRO adjustments: divergence/candle/volume affect confidence & management =====
    # 1) Divergence: nếu đánh ngược divergence → warn mạnh
    if bias == "BUY" and base["meta"]["div"].get("bear"):
        notes.append("⚠️ Bearish divergence → BUY dễ bị hụt hơi: ưu tiên TP1 nhanh, dời SL sớm.")
    if bias == "SELL" and base["meta"]["div"].get("bull"):
        notes.append("⚠️ Bullish divergence → SELL dễ bị hụt hơi: ưu tiên TP1 nhanh, dời SL sớm.")

    # 2) Candle phản công ngay trước entry: nếu bias=SELL mà có BULL engulfing/lower rejection → cảnh báo
    engulf = base["meta"]["candle"].get("engulf")
    rej = base["meta"]["candle"].get("rejection")
    if bias == "SELL" and (engulf == "BULL" or rej == "LOWER"):
        notes.append("⚠️ Nến phản công (bull engulf / lower rejection) → SELL nên chờ confirm thêm, tránh vào sớm.")
    if bias == "BUY" and (engulf == "BEAR" or rej == "UPPER"):
        notes.append("⚠️ Nến phản công (bear engulf / upper rejection) → BUY nên chờ confirm thêm, tránh vào sớm.")

    # 3) Volume LOW: giảm “máu chiến”
    if base["meta"]["volq"].get("state") == "LOW":
        notes.append("⚠️ Volume thấp → nếu vào, ưu tiên đánh NGẮN + TP1, không gồng.")

    # Confirmed -> SL/TP bằng risk engine (giữ logic cũ)
    entry = float(entry_center)
    equity_usd = float(os.getenv("EQUITY_USD", "1000"))
    risk_pct   = float(os.getenv("RISK_PCT", "0.0075"))

    try:
        plan = calc_smart_sl_tp(
            symbol=symbol,
            side=bias,
            entry=float(entry),
            atr=float(atr15),
            liquidity_level=float(liq_level) if liq_level is not None else None,
            equity_usd=equity_usd,
            risk_pct=risk_pct,
        )
    except Exception as e:
        plan = {"ok": False, "reason": f"risk_engine_error: {e}"}

    sl: Optional[float] = _safe_float(plan.get("sl"))
    tp1: Optional[float] = _safe_float(plan.get("tp1"))
    tp2: Optional[float] = _safe_float(plan.get("tp2"))
    lot: Optional[float] = _safe_float(plan.get("lot"))
    rdist: Optional[float] = _safe_float(plan.get("r"))

    if not plan.get("ok", True):
        quality_lines.append(f"⚠️ Risk warn: {plan.get('reason', 'risk check failed')}")
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


    # =========================
    # SYSTEMATIC CONTINUATION SCORING ENGINE (Bias-Pullback-Momentum)
    # - Bias = H1 + H4 confluence + EMA200 slope/distance
    # - Pullback = M15 pullback into EMA20-EMA50 zone + basic HL/LH structure
    # - Momentum = M15 BOS (break minor structure) OR engulfing in direction
    # Bonus/Filter:
    # - Volume quality + candle patterns add stars (bonus)
    # - Liquidity warning can downgrade / require stronger score
    # =========================

    def _ema_last(closes: list[float], n: int) -> float | None:
        s = _ema(closes, n)
        return float(s[-1]) if s and len(s) else None

    def _ema_slope_pct(closes: list[float], n: int, slope_n: int) -> float:
        s = _ema(closes, n)
        if not s or len(s) < slope_n + 5:
            return 0.0
        last = float(s[-1])
        prev = float(s[-1 - slope_n])
        cur = float(closes[-1]) if closes else last
        return ((last - prev) / max(1e-9, cur)) * 100.0

    # ---- Bias (H1 + H4 confluence) ----
    # NORMAL mục tiêu: tín hiệu "trung bình" (không gắt quá, không rác quá)
    # - Bias phải có (không cho phép thiếu Bias)
    # - Confluence H1+H4 quyết định FULL (lệch nhau thì tối đa HALF)
    mode = str(os.getenv("SIGNAL_MODE", "NORMAL") or "NORMAL").upper()

    h1_tr = _trend_label(h1c)
    h4_tr = _trend_label(h4c)

    confluence_ok = int(h1_tr in ("bullish", "bearish") and h1_tr == h4_tr)
    bias_side = "BUY" if h1_tr == "bullish" else ("SELL" if h1_tr == "bearish" else "NONE")
    # ---- Structure + key levels (for Telegram: HH/HL/LH/LL + BOS level) ----
    h1_struct = _structure_from_swings(h1c, lookback=260)
    h4_struct = _structure_from_swings(h4c, lookback=260)
    m15_struct = _m15_key_levels(m15c, bias_side=bias_side, lookback=120)

    base.setdefault("meta", {})["structure"] = {
        "H4": h4_struct.get("tag"),
        "H1": h1_struct.get("tag"),
        "M15": m15_struct.get("tag"),
    }
    base["meta"]["key_levels"] = {
        "H1_HH": h1_struct.get("hh") or h1_struct.get("last_high"),
        "H1_HL": h1_struct.get("hl") or h1_struct.get("last_low"),
        "H1_LH": h1_struct.get("lh"),
        "H1_LL": h1_struct.get("ll"),
        "M15_BOS": m15_struct.get("bos_level"),
        "M15_PB_EXT": m15_struct.get("pullback_extreme"),
        # Always-available context levels (even when HH/HL/LH/LL are n/a)
        "M15_RANGE_LOW": _range_levels(m15c, n=20)[0],
        "M15_RANGE_HIGH": _range_levels(m15c, n=20)[1],
        "M15_LAST": float(m15c[-1].close) if m15c else None,
    }

    # build levels_info list for rendering (2 decimals)
    levels_info = []
    kh = base["meta"]["key_levels"]
    if kh.get("H1_HH") is not None:
        levels_info.append((kh["H1_HH"], "H1 HH (đỉnh cấu trúc)"))
    if kh.get("H1_HL") is not None:
        levels_info.append((kh["H1_HL"], "H1 HL (trend giữ; thủng là yếu)"))
    if kh.get("H1_LH") is not None:
        levels_info.append((kh["H1_LH"], "H1 LH (đỉnh hồi; vượt là fail SELL)"))
    if kh.get("H1_LL") is not None:
        levels_info.append((kh["H1_LL"], "H1 LL (đáy cấu trúc)"))
    if kh.get("M15_BOS") is not None:
        levels_info.append((kh["M15_BOS"], f"M15 BOS{'↑' if bias_side=='BUY' else ('↓' if bias_side=='SELL' else '')} level (mốc phá cấu trúc)"))
    if kh.get("M15_PB_EXT") is not None:
        levels_info.append((kh["M15_PB_EXT"], f"M15 Pullback {'Low' if bias_side=='BUY' else ('High' if bias_side=='SELL' else 'extreme')} (mốc giữ HL/LH)"))
    base["levels_info"] = levels_info

    where, wait_for = _where_wait_text(m15c, bias_side=bias_side)
    base["where"] = where
    base["wait_for"] = wait_for
    base.setdefault("meta", {})["where"] = where
    base["meta"]["wait_for"] = wait_for



    # Bias base: chỉ cần H1 có trend rõ (bull/bear)
    # Keep a MINOR plan (M15 execution) as default suggestion
    entry_minor, sl_minor, tp1_minor, tp2_minor = entry, sl, tp1, tp2

    bias_ok = int(bias_side in ("BUY", "SELL"))

    # EMA200 slope/distance (để né đoạn "lẹt đẹt quanh EMA200")
    # NORMAL: dùng H1 là chính; nếu confluence_ok thì check thêm H4 (được cộng strength)
    slope_n = 20
    xau_priority = str(symbol).upper().startswith("XAU")

    # Ngưỡng NORMAL: XAU hơi siết hơn BTC nhưng không "gắt"
    slope_min_h1 = 0.025 if xau_priority else 0.02
    dist_min_h1  = 0.06  if xau_priority else 0.05
    slope_min_h4 = 0.02
    dist_min_h4  = 0.05

    h1_cl = _closes(h1c)
    h4_cl = _closes(h4c)
    m15_cl = _closes(m15c)

    ema200_h1 = _ema_last(h1_cl, 200) if h1_cl else None
    ema200_h4 = _ema_last(h4_cl, 200) if h4_cl else None

    slope200_h1 = _ema_slope_pct(h1_cl, 200, slope_n) if h1_cl else 0.0
    slope200_h4 = _ema_slope_pct(h4_cl, 200, slope_n) if h4_cl else 0.0

    bias_strength = "UNKNOWN"
    if bias_ok and ema200_h1 and h1_cl:
        last_h1 = float(h1_cl[-1])
        dist_h1 = abs(last_h1 - float(ema200_h1)) / max(1e-9, last_h1) * 100.0
        slope_ok_h1 = abs(slope200_h1) >= slope_min_h1
        dist_ok_h1  = dist_h1 >= dist_min_h1
        if slope_ok_h1 and dist_ok_h1:
            bias_strength = "STRONG"
        else:
            bias_strength = "WEAK"

    # Nếu có confluence, có thể nâng strength khi H4 cũng đạt
    if confluence_ok and bias_ok and ema200_h4 and h4_cl:
        last_h4 = float(h4_cl[-1])
        dist_h4 = abs(last_h4 - float(ema200_h4)) / max(1e-9, last_h4) * 100.0
        slope_ok_h4 = abs(slope200_h4) >= slope_min_h4
        dist_ok_h4  = dist_h4 >= dist_min_h4
        if bias_strength == "STRONG" and slope_ok_h4 and dist_ok_h4:
            bias_strength = "STRONG"
        else:
            # confluence nhưng H4 yếu => vẫn cho trade, nhưng FULL sẽ bị kiểm soát bởi confluence/score
            bias_strength = "WEAK"

    base.setdefault("meta", {}).setdefault("score_detail", {})["bias_strength"] = bias_strength
    base.setdefault("meta", {}).setdefault("score_detail", {})["confluence_ok"] = int(confluence_ok)

    # ---- Pullback (M15) ----
    pullback_ok = 0
    if bias_side != "NONE" and m15_cl and len(m15_cl) >= 80:
        e20 = _ema_last(m15_cl, 20)
        e50 = _ema_last(m15_cl, 50)
        if e20 and e50:
            band_lo = min(e20, e50)
            band_hi = max(e20, e50)
            last_close = float(m15c[-2].close)  # last closed candle
            # NORMAL: XAU chặt hơn chút, BTC thoáng hơn
            tol_pct = 0.10 if xau_priority else 0.12
            tol = tol_pct / 100.0
            in_zone = (band_lo * (1 - tol)) <= last_close <= (band_hi * (1 + tol))

            # basic HL/LH structure check using swing blocks
            look = 10
            a = m15c[-(2*look+5):-5]  # older block
            b = m15c[-(look+5):-5]   # recent block
            struct_ok = True
            if len(a) >= look and len(b) >= look:
                a_low = min(c.low for c in a)
                b_low = min(c.low for c in b)
                a_high = max(c.high for c in a)
                b_high = max(c.high for c in b)

                if bias_side == "BUY":
                    struct_ok = b_low >= a_low  # HL-ish
                else:
                    struct_ok = b_high <= a_high  # LH-ish

            pullback_ok = int(in_zone and struct_ok)

    # ---- Momentum (M15): BOS + (micro) retest ----
    momentum_ok = 0
    bos_retest = False
    bos_micro_retest = False
    engulf_aligned = False

    if bias_side != "NONE" and len(m15c) >= 50:
        # Use closed candles: -3 (BOS), -2 (retest/micro-retest)
        look = 14 if xau_priority else 12
        base_prev = m15c[-(look+6):-6]  # before BOS window
        bos_c = m15c[-3]
        ret_c = m15c[-2]

        prev_high = max(c.high for c in base_prev) if base_prev else None
        prev_low  = min(c.low for c in base_prev) if base_prev else None

        # ATR-based tolerance (micro-retest)
        atr15_local = _atr(m15c[:-1], 14) or (atr15 or 0.0) or 0.0
        tol_atr = 0.35 if xau_priority else 0.45
        micro_tol = float(tol_atr) * float(atr15_local)

        if bias_side == "BUY" and prev_high is not None:
            bos_ok = bos_c.close > prev_high
            # retest: wick touches near level, close holds above
            ret_ok = (ret_c.low <= prev_high + micro_tol) and (ret_c.close >= prev_high)
            ret_bull = ret_c.close >= ret_c.open
            bos_retest = bool(bos_ok and ret_ok and ret_bull)

            # micro-retest (không chạm đúng level): quay về 30-70% thân nến BOS
            bos_body_low = min(bos_c.open, bos_c.close)
            bos_body_high = max(bos_c.open, bos_c.close)
            zone_lo = bos_body_low + 0.30 * (bos_body_high - bos_body_low)
            zone_hi = bos_body_low + 0.70 * (bos_body_high - bos_body_low)
            micro_ok = (ret_c.low <= zone_hi + micro_tol) and (ret_c.close >= zone_lo)
            bos_micro_retest = bool(bos_ok and micro_ok and ret_bull)

        if bias_side == "SELL" and prev_low is not None:
            bos_ok = bos_c.close < prev_low
            ret_ok = (ret_c.high >= prev_low - micro_tol) and (ret_c.close <= prev_low)
            ret_bear = ret_c.close <= ret_c.open
            bos_retest = bool(bos_ok and ret_ok and ret_bear)

            bos_body_low = min(bos_c.open, bos_c.close)
            bos_body_high = max(bos_c.open, bos_c.close)
            zone_hi = bos_body_high - 0.30 * (bos_body_high - bos_body_low)
            zone_lo = bos_body_high - 0.70 * (bos_body_high - bos_body_low)
            micro_ok = (ret_c.high >= zone_lo - micro_tol) and (ret_c.close <= zone_hi)
            bos_micro_retest = bool(bos_ok and micro_ok and ret_bear)

        cpat = _candle_patterns(m15c)
        engulf_aligned = (cpat.get("engulf") == "bull" and bias_side == "BUY") or (cpat.get("engulf") == "bear" and bias_side == "SELL")

        # NORMAL:
        # - XAU: ưu tiên BOS+retest, cho BOS+micro-retest (để không đói kèo), engulf chỉ là "bonus" không đủ 1 mình
        # - BTC: BOS+retest / BOS+micro-retest, engulf là fallback
        if xau_priority:
            momentum_ok = int(bool(bos_retest or bos_micro_retest))
        else:
            momentum_ok = int(bool(bos_retest or bos_micro_retest or engulf_aligned))

    base.setdefault("meta", {}).setdefault("score_detail", {})["bos_retest"] = int(bool(bos_retest))
    base.setdefault("meta", {}).setdefault("score_detail", {})["bos_micro_retest"] = int(bool(bos_micro_retest))
    base.setdefault("meta", {}).setdefault("score_detail", {})["engulf_aligned"] = int(bool(engulf_aligned))

    # =========================
    # SCORING (v8): Bias + Pullback(PB) + Momentum(MOM)
    #
    # - HALF: mặc định cần >=2/3 điều kiện (Bias + PB + MOM), và *bắt buộc có Bias* (tránh trade ngược xu hướng lớn).
    # - FULL: bắt buộc đủ 3/3 + Confluence(H4)=1 (tức H4/H1 đồng hướng/ủng hộ).
    # =========================

    score3 = int(bias_ok) + int(pullback_ok) + int(momentum_ok)


    # HL/LH hold filter (trend must not be broken for HALF):
    # - If bias is BUY: require last M15 close >= H1_HL (close-based, ignore wicks).
    # - If bias is SELL: require last M15 close <= H1_LH.
    last_close = float(m15c[-1].close) if m15c else float('nan')
    h1_hl_lv = kh.get('H1_HL')
    h1_lh_lv = kh.get('H1_LH')
    if bias_side == 'BUY' and h1_hl_lv is not None:
        hl_hold_ok = (last_close >= float(h1_hl_lv))
    elif bias_side == 'SELL' and h1_lh_lv is not None:
        hl_hold_ok = (last_close <= float(h1_lh_lv))
    else:
        hl_hold_ok = True

    half_ok = (bias_ok == 1) and (score3 >= 2) and hl_hold_ok and (not major_bearish)

    # FULL must be a MAJOR signal: confirm M15 close breaks the MAJOR level (H1 HH/LL), not just a minor swing.
    major_bos_level = h1_struct.get("hh") if bias_side == "BUY" else h1_struct.get("ll")
    try:
        major_bos_confirmed = (major_bos_level is not None) and (
            (m15_last_close > float(major_bos_level)) if bias_side == "BUY" else (m15_last_close < float(major_bos_level))
        )
    except Exception:
        major_bos_confirmed = False

    full_ok = (score3 == 3) and (confluence_ok == 1) and major_bos_confirmed

    if full_ok:
        trade_mode = "FULL"
        stars = 4
    elif half_ok:
        trade_mode = "HALF"
        stars = 3
    else:
        trade_mode = "WAIT"
        stars = 1

    # if liquidity warning, cap to HALF at most (still allow observation)
    if liq_warn and trade_mode == "FULL":
        trade_mode = "HALF"
        stars = min(stars, 3)

    # ---- Spread filter (NORMAL: không bóp nghẹt cơ hội) ----
        # MT5 bars có thể có field 'spread' (points). Nếu không có, bỏ qua.
        spread_now = getattr(m15c[-2], "spread", 0.0) if len(m15c) >= 3 else 0.0
        spread_list = [float(getattr(c, "spread", 0.0) or 0.0) for c in m15c[-(60+2):-2]]
        spread_list = [s for s in spread_list if s and s > 0]

        spread_ratio = None
        spread_warn = False
        spread_block = False
        if spread_now and spread_now > 0 and spread_list:
            med = float(sorted(spread_list)[len(spread_list)//2])
            if med > 0:
                spread_ratio = float(spread_now) / med

                # NORMAL thresholds:
                # - HIGH: downgrade FULL -> HALF
                # - BLOCK: WAIT (chỉ khi "bất thường" thật sự)
                warn_k = 1.8 if xau_priority else 1.7
                block_k = 2.6 if xau_priority else 2.9

                if spread_ratio >= warn_k:
                    spread_warn = True
                if spread_ratio >= block_k:
                    spread_block = True

        if spread_ratio is not None:
            base.setdefault("meta", {}).setdefault("spread", {})["ratio"] = spread_ratio
            base.setdefault("meta", {}).setdefault("spread", {})["now"] = spread_now

        if spread_block:
            score3 = 1  # WAIT
            base.setdefault("meta", {}).setdefault("spread", {})["state"] = "BLOCK"
        elif spread_warn:
            base.setdefault("meta", {}).setdefault("spread", {})["state"] = "HIGH"
            # nếu vẫn đủ 3/3 thì downgrade FULL -> HALF
            if score3 == 3:
                score3 = 2

        # ---- Reversal warning layer (cảnh báo đảo chiều) ----
        reversal_flags: list[str] = []
        severe_reversal = False

        div = base.get("meta", {}).get("div", {}) or {}
        if bias_side == "BUY" and div.get("bear"):
            reversal_flags.append("Bearish divergence (M15) chống BUY")
        if bias_side == "SELL" and div.get("bull"):
            reversal_flags.append("Bullish divergence (M15) chống SELL")

        h1_pat = _candle_patterns(h1c) if h1c else {}
        h4_pat = _candle_patterns(h4c) if h4c else {}

        if bias_side == "BUY" and (h1_pat.get("engulf") == "bear" or h1_pat.get("rejection") == "upper"):
            reversal_flags.append("H1 có nến phản công (bear engulf / upper rejection)")
            severe_reversal = True
        if bias_side == "SELL" and (h1_pat.get("engulf") == "bull" or h1_pat.get("rejection") == "lower"):
            reversal_flags.append("H1 có nến phản công (bull engulf / lower rejection)")
            severe_reversal = True

        if bias_side == "BUY" and (h4_pat.get("engulf") == "bear" or h4_pat.get("rejection") == "upper"):
            reversal_flags.append("H4 có nến phản công (bear engulf / upper rejection)")
            severe_reversal = True
        if bias_side == "SELL" and (h4_pat.get("engulf") == "bull" or h4_pat.get("rejection") == "lower"):
            reversal_flags.append("H4 có nến phản công (bull engulf / lower rejection)")
            severe_reversal = True

        if len(h1c) >= 30:
            look = 10
            prev_h1 = h1c[-(look+3):-3]
            last_h1 = h1c[-2]  # last closed
            prev_high = max(c.high for c in prev_h1) if prev_h1 else None
            prev_low = min(c.low for c in prev_h1) if prev_h1 else None
            if bias_side == "BUY" and prev_low is not None and last_h1.close < prev_low:
                reversal_flags.append("H1 CHoCH nhỏ: close phá đáy gần nhất (nguy cơ đảo chiều)")
                severe_reversal = True
            if bias_side == "SELL" and prev_high is not None and last_h1.close > prev_high:
                reversal_flags.append("H1 CHoCH nhỏ: close phá đỉnh gần nhất (nguy cơ đảo chiều)")
                severe_reversal = True

        if reversal_flags:
            base.setdefault("meta", {})["reversal_warnings"] = reversal_flags

        # Nếu đảo chiều mạnh: FULL -> HALF, HALF -> WAIT
        if severe_reversal:
            if score3 == 3:
                score3 = 2
            elif score3 == 2:
                score3 = 1

        # Confluence rule: nếu H1/H4 lệch nhau => tối đa HALF (không FULL)
        if base.get("meta", {}).get("score_detail", {}).get("confluence_ok", 0) != 1 and score3 == 3:
            score3 = 2

        trade_mode = "WAIT"
        if score3 == 3:
            trade_mode = "FULL"
        elif score3 == 2:
            trade_mode = "HALF"


        # Bonus stars from vol + candle pattern, and penalty from liquidity warning
        bonus = 0
        volq = _vol_quality(m15c, 20)
        if volq.get("state") == "HIGH":
            bonus += 1
        cpat2 = _candle_patterns(m15c)
        if (cpat2.get("engulf") == "bull" and bias_side == "BUY") or (cpat2.get("engulf") == "bear" and bias_side == "SELL"):
            bonus += 1
        if liq_warn:
            bonus -= 1

        base.setdefault("meta", {}).setdefault("score_detail", {}).update({
            "bias_ok": int(bias_ok),
            "pullback_ok": int(pullback_ok),
            "momentum_ok": int(momentum_ok),
            "bias_tf": "H1+H4",
            "bias_side": bias_side,
            "score": int(score3),
            "liq_warn": bool(liq_warn),
            "volq": volq,
            "candle": cpat2,
        })
        base["trade_mode"] = trade_mode

        stars = 1
        if score >= 6:
            stars = 5
        elif score >= 5:
            stars = 4
        elif score >= 3:
            stars = 3
        elif score >= 2:
            stars = 2


        # ---- Override recommendation/stars by scoring engine (FULL/HALF/WAIT) ----
        try:
            sm = base.get("trade_mode", "WAIT")
            sd = (base.get("meta", {}) or {}).get("score_detail", {}) or {}
            side2 = sd.get("bias_side", None)

            if sm in ("FULL", "HALF") and side2 in ("BUY", "SELL") and entry is not None and sl is not None and tp1 is not None:
                recommendation = "🟢 BUY" if side2 == "BUY" else "🔴 SELL"

                # base stars by mode
                stars_mode = 4 if sm == "FULL" else 3
                b = 0
                # bonus from meta (already computed)
                if (sd.get("volq", {}) or {}).get("state") == "HIGH":
                    b += 1
                c = sd.get("candle", {}) or {}
                if (c.get("engulf") == "bull" and side2 == "BUY") or (c.get("engulf") == "bear" and side2 == "SELL"):
                    b += 1
                if sd.get("liq_warn"):
                    b -= (2 if str(symbol).upper().startswith("XAU") else 1)

                stars = max(stars, stars_mode + b)
            else:
                # WAIT or missing components -> no trade
                base["trade_mode"] = "WAIT"
                recommendation = "CHỜ"
                stars = min(stars, 2)
        except Exception:
            pass

        quality_lines.append("RR ~ 1:2")
        if rdist is not None:
            quality_lines.append(f"R~{rdist:.2f} | SL=MIN(Liq, ATR, Risk) (risk engine)")

        stars = max(1, min(5, int(stars)))
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
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
        return base

    # =========================
    # Formatter (MUST be named format_signal for main.py import)
    # =========================
def format_signal(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "XAUUSD")
    tf = sig.get("tf", "M15")
    session = sig.get("session", "")
    rec = sig.get("recommendation", "CHỜ")
    action = "BUY" if rec=="BUY" else "SELL" if rec=="SELL" else "CHỜ"
    stars = int(sig.get("stars", 1))
    stars_txt = "⭐️" * max(1, min(5, stars))

    meta = sig.get("meta", {}) or {}
    sd = (meta.get("score_detail", {}) or {})
    trade_mode = (sig.get("trade_mode") or "").upper()

    def nf2(x):
        if x is None:
            return "..."
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "..."

    # Structure + key levels
    st = meta.get("structure", {}) or {}
    kl = meta.get("key_levels", {}) or {}

    h4_tag = st.get("H4", "n/a")
    h1_tag = st.get("H1", "n/a")
    m15_tag = st.get("M15", "n/a")

    # Spread
    sp = meta.get("spread", {}) or {}
    spread_state = sp.get("state")
    spread_ratio = sp.get("ratio")

    # Reversal warnings
    revs = meta.get("reversal_warnings", []) or []
    liq_lines = sig.get("liquidity_lines", []) or []
    qual_lines = sig.get("quality_lines", []) or []

    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")

    lines: List[str] = []
    head = f"📊 {symbol} | {tf}"
    if session:
        head += f" | {session}"
    lines.append(head)

    # Recommendation line
    mode_txt = f" | Mode: {trade_mode}" if trade_mode in ("FULL", "HALF") else ""
    score_txt = ""
    if trade_mode in ("FULL", "HALF") and sd.get("score") is not None:
        score_txt = f" ({sd.get('score')}/3)"
    lines.append(f"{stars_txt}  <b>{rec}</b>{mode_txt}{score_txt}")

    # Structures
    lines.append(f"Structure (Major): H4 {h4_tag} | H1 {h1_tag}")
    lines.append(f"Structure (Minor): M15 {m15_tag}")

    where = meta.get("where")
    wait_for = meta.get("wait_for")
    if where:
        lines.append(f"Now: {where}")
    if wait_for:
        lines.append(f"Wait: {wait_for}")


    # Key levels (prices)
    lines.append("Key levels:")
    lines.append("Major (H4/H1):")
    if kl.get("H1_HH") is not None:
        lines.append(f"- H1 HH: {nf2(kl.get('H1_HH'))}")
    if kl.get("H1_HL") is not None:
        lines.append(f"- H1 HL (trend giữ): {nf2(kl.get('H1_HL'))}")
    if kl.get("H1_LH") is not None:
        lines.append(f"- H1 LH: {nf2(kl.get('H1_LH'))}")
    if kl.get("H1_LL") is not None:
        lines.append(f"- H1 LL: {nf2(kl.get('H1_LL'))}")

    lines.append("Minor (M15):")
    if kl.get("M15_RANGE_LOW") is not None and kl.get("M15_RANGE_HIGH") is not None:
        lines.append(f"- M15 Range: {nf2(kl.get('M15_RANGE_LOW'))} – {nf2(kl.get('M15_RANGE_HIGH'))}")
    if kl.get("M15_LAST") is not None:
        lines.append(f"- M15 Last close: {nf2(kl.get('M15_LAST'))}")
    if kl.get("M15_BOS") is not None:
        lines.append(f"- M15 BOS level: {nf2(kl.get('M15_BOS'))}")
    if kl.get("M15_PB_EXT") is not None:
        lines.append(f"- M15 pullback extreme: {nf2(kl.get('M15_PB_EXT'))}")

    show_plan = bool(sig.get("show_trade_plan", True))
    if show_plan:
        lines.append(f"Entry: {nf2(entry)}")
        lines.append(f"SL: {nf2(sl)} | TP1: {nf2(tp1)} | TP2: {nf2(tp2)}")

    # SL focus: watch the level that matches the structure you're trading.
    # - FULL  => major (H1) invalidation
    # - HALF  => minor (M15) invalidation
    sl_focus_level = None
    sl_focus_label = None
    if trade_mode == "FULL":
        if action == "BUY":
            sl_focus_level = kl.get("H1_HL") or kl.get("H1_LL")
            sl_focus_label = "H1 HL (major)"
        elif action == "SELL":
            sl_focus_level = kl.get("H1_LH") or kl.get("H1_HH")
            sl_focus_label = "H1 LH (major)"
    elif trade_mode == "HALF":
        if action in ("BUY", "SELL"):
            sl_focus_level = kl.get("M15_PB_EXT") or kl.get("M15_RANGE_LOW") or kl.get("M15_RANGE_HIGH")
            sl_focus_label = "M15 pullback extreme (minor)"

    if sl_focus_level is not None and sl_focus_label:
        lines.append(f"SL focus ({trade_mode}): {sl_focus_label} @ {nf2(sl_focus_level)}")

    # Quick reasons (score detail)
    if trade_mode in ("FULL", "HALF") and sd:
        lines.append("")
        lines.append(f"Score detail: Bias:{sd.get('bias_ok')} PB:{sd.get('pullback_ok')} MOM:{sd.get('momentum_ok')} | Confluence:{sd.get('confluence_ok')}")

    # Spread info
    if spread_state in ("HIGH", "BLOCK"):
        rr = f"x{float(spread_ratio):.2f}" if spread_ratio is not None else ""
        lines.append(f"⚠️ Spread: <b>{spread_state}</b> {rr}".strip())

    # Liquidity warnings (keep short)
    if liq_lines:
        # keep only top 2 lines
        lines.append("")
        lines.append("Liquidity:")
        for s in liq_lines[:2]:
            lines.append(f"- {s}")

    # Reversal warnings (short)
    if revs:
        lines.append("⚠️ Reversal watch:")
        for r in revs[:2]:
            lines.append(f"- {r}")

    # =========================
    # Signal block (B / A / A+)
    # =========================
    def _bool(v):
        return bool(v) if v is not None else False

    # derive core checks
    bias_ok = _bool(sd.get("bias_ok"))
    pullback_ok = _bool(sd.get("pullback_ok"))
    momentum_ok = _bool(sd.get("momentum_ok"))
    confluence_ok = _bool(sd.get("confluence_ok"))
    liq_warn = ("Liquidity WARNING" in " | ".join(sig.get("context_lines", []) or [])) or _bool(sd.get("liq_warn"))
    liquidity_ok = not liq_warn

    # Spread: nếu không có dữ liệu bid/ask hoặc ratio -> để None (hiển thị …)
    if spread_state is None and spread_ratio is None:
        spread_ok = None
    else:
        spread_ok = (spread_state not in ("HIGH", "BLOCK"))
    tradable = (rec in ("BUY", "SELL")) and (trade_mode in ("FULL", "HALF"))

    # Location (within M15 range) -> compute position % and zone
    m15_lo = kl.get("M15_RANGE_LOW")
    m15_hi = kl.get("M15_RANGE_HIGH")
    m15_last = kl.get("M15_LAST") or entry
    loc_ok = None
    loc_txt = "thiếu dữ liệu range"
    try:
        if m15_lo is not None and m15_hi is not None and m15_last is not None:
            lo = float(m15_lo); hi = float(m15_hi); last = float(m15_last)
            rng = hi - lo
            if abs(rng) < 1e-9:
                loc_ok = None
                loc_txt = "range quá hẹp"
            else:
                pos = (last - lo) / rng  # 0..1 (có thể vượt nếu phá range)
                pos_clamped = max(0.0, min(1.0, pos))
                pos_pct = pos_clamped * 100.0

                # Zone classification
                if pos < 0.0:
                    zone = "Dưới range (breakdown)"
                elif pos > 1.0:
                    zone = "Trên range (breakout)"
                elif 0.20 <= pos_clamped <= 0.80:
                    zone = "Giữa range – No trade zone"
                elif pos_clamped < 0.20:
                    zone = "Gần đáy range"
                else:
                    zone = "Gần đỉnh range"

                # For checklist pass/fail:
                # - BUY tốt nhất khi gần đáy
                # - SELL tốt nhất khi gần đỉnh
                if action == "BUY":
                    loc_ok = pos_clamped <= 0.20
                elif action == "SELL":
                    loc_ok = pos_clamped >= 0.80
                else:
                    # Nếu đang CHỜ: coi location OK nếu ở sát biên (có edge), còn giữa range là không OK
                    loc_ok = not (0.20 <= pos_clamped <= 0.80)

                loc_txt = f"{zone} (pos={pos_pct:.0f}%)"
    except Exception:
        loc_ok, loc_txt = None, "lỗi tính location"

    # Grade logic
    score_val = sd.get("score")
    score_val = int(score_val) if score_val is not None else None

    grade = "B"
    if not tradable:
        grade = "B"
    else:
        if trade_mode == "FULL" and score_val == 3 and liquidity_ok and spread_state != "BLOCK":
            grade = "A+"
        elif score_val is not None and score_val >= 2 and liquidity_ok and spread_state != "BLOCK":
            grade = "A"
        else:
            grade = "B"

    # Priority hint (based on bias)
    if bias_ok:
        priority = f"Ưu tiên chờ {rec}" if rec in ("BUY", "SELL") else "Ưu tiên CHỜ theo bias"
    else:
        priority = "Ưu tiên CHỜ (bias chưa rõ)"

    # What is missing?
    missing = []
    if not bias_ok: missing.append("Bias (H4/H1)")
    if tradable and not pullback_ok: missing.append("Pullback/Location")
    if tradable and (loc_ok is False): missing.append("Location (đúng biên range)")
    if tradable and not momentum_ok: missing.append("Trigger/Momentum")
    if tradable and not confluence_ok: missing.append("Confluence (H1/H4)")
    if tradable and not liquidity_ok: missing.append("Liquidity sạch (no WARNING)")
    if tradable and (spread_ok is False): missing.append("Spread ổn")

    lines.append("")
    lines.append("📌 TÍN HIỆU (chuẩn hoá):")
    lines.append(f"- {priority}")
    lines.append(f"- Grade: <b>{grade}</b>  | Trade: {('YES' if tradable else 'NO')}")

    # 5-checklist
    def okno(x):
        if x is None:
            return "…"
        return "✅" if x else "❌"

    lines.append("Checklist (5):")
    lines.append(f"1) Bias (H4/H1): {okno(bias_ok)}  ({h4_tag} / {h1_tag})")
    lines.append(f"2) Location: {okno(loc_ok)}  ({loc_txt})")
    lines.append(f"3) Liquidity: {okno(liquidity_ok)}  ({'WARN' if liq_warn else 'OK'})")
    lines.append(f"4) Trigger/Momentum: {okno(momentum_ok)}")
        sc_ok = None if spread_ok is None else bool(spread_ok and confluence_ok)
    spread_label = spread_state if spread_state is not None else "n/a (thiếu bid/ask)"
    if spread_ratio is not None:
        try:
            spread_label = f"{spread_label} (ratio={float(spread_ratio):.2f})"
        except Exception:
            pass
    lines.append(f"5) Spread/Confluence: {okno(sc_ok)}  (Spread={spread_label})")

    if missing:
        lines.append("Thiếu / chờ thêm:")
        for x in missing[:6]:
            lines.append(f"- {x}")
    else:
        lines.append("✅ Đủ điều kiện theo grade hiện tại.")

    return "\n".join(lines)
