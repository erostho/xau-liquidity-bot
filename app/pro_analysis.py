# app/pro_analysis.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Sequence
from datetime import datetime, timedelta
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



def _sweep_grade(sw: Dict[str, Any]) -> str:
    """Classify sweep/spring strength for V6 rendering."""
    if not isinstance(sw, dict) or not sw.get("ok"):
        return "NONE"
    score = int(sw.get("score") or 0)
    vol_ok = bool(sw.get("vol_ok"))
    if score >= 5 and vol_ok:
        return "STRONG"
    if score >= 4:
        return "MEDIUM"
    if score >= 3:
        return "WEAK"
    return "NONE"


def _entry_zone_v6(side: str, k: Dict[str, Any], atr15: float) -> Tuple[Optional[float], Optional[float]]:
    """Refine entry zone around BOS + pullback extreme / wick area."""
    bos = _safe_float((k or {}).get("M15_BOS"))
    pbx = _safe_float((k or {}).get("M15_PB_EXT"))
    hi = _safe_float((k or {}).get("M15_RANGE_HIGH"))
    lo = _safe_float((k or {}).get("M15_RANGE_LOW"))
    a = max(float(atr15 or 0.0), 1e-9)
    pad = 0.18 * a

    side = str(side or "").upper()
    if side == "SELL":
        anchors = [x for x in [bos, pbx, hi] if x is not None]
        if not anchors:
            return None, None
        zl = min(anchors) - pad
        zh = max(anchors) + pad
        return float(zl), float(zh)

    if side == "BUY":
        anchors = [x for x in [bos, pbx, lo] if x is not None]
        if not anchors:
            return None, None
        zl = min(anchors) - pad
        zh = max(anchors) + pad
        return float(zl), float(zh)

    return None, None


def _grade_v6(meta: Dict[str, Any], trade_mode: str, sweep_grade: str, close_confirm: Dict[str, Any]) -> str:
    """Translate current engine state to A/B/C/SKIP real-edge grading."""
    meta = meta or {}
    sd = meta.get("score_detail", {}) or {}
    spread = meta.get("spread", {}) or {}
    revs = meta.get("reversal_warnings", []) or []
    playbook_v4 = meta.get("playbook_v4", {}) or {}

    if spread.get("state") == "BLOCK":
        return "SKIP"
    if str(playbook_v4.get("quality") or "").upper() == "LOW" and str(trade_mode or "").upper() == "WAIT":
        return "SKIP"

    mode = str(trade_mode or "").upper()
    sweep_grade = str(sweep_grade or "NONE").upper()
    cc_strength = str((close_confirm or {}).get("strength") or "NO").upper()

    if mode == "FULL":
        if sweep_grade == "STRONG" or cc_strength == "STRONG":
            return "A"
        return "A-"

    if mode == "HALF":
        if revs:
            return "C"
        if sweep_grade in ("MEDIUM", "STRONG") or cc_strength in ("WEAK", "STRONG"):
            return "B"
        return "B-"

    if sd.get("bias_ok") and not sd.get("momentum_ok"):
        return "C"
    return "SKIP"

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



def _vn_session_label(now_utc: datetime | None = None) -> str:
    """Rough FX-style session label in Vietnam time (UTC+7).
    This is only for display, not scoring."""
    try:
        now_utc = now_utc or datetime.utcnow()
        vn = now_utc + timedelta(hours=7)
        h = vn.hour
        # Vietnam time blocks
        # Asia ~07-14, EU ~14-20, US ~20-03 (approx, overlaps included)
        if 14 <= h < 15:
            return "Giao phiên Á-Âu"
        if 20 <= h < 22:
            return "Giao phiên Âu-Mỹ"
        if 7 <= h < 14:
            return "Phiên Á"
        if 15 <= h < 20:
            return "Phiên Âu"
        if h >= 22 or h < 3:
            return "Phiên Mỹ"
        return "Phiên Á"  # 03-07
    except Exception:
        return ""

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

    # Guard: missing candle data (e.g., MT5 not pushed and TwelveData disabled/unavailable)
    if not m15 or not m30 or not h1 or not h4:
        sd = meta.get("score_detail") or {}
        sd.setdefault("grade", "B")
        sd.setdefault("trade", "NO")
        sd.setdefault("checklist", [])
        meta["score_detail"] = sd
        meta["error"] = "MISSING_CANDLES"
        base.setdefault("recommendation", "CHỜ")
        base.setdefault("stars", 1)
        # _inject_meta_structure_and_levels(base, m15 or [], m30 or [], h1 or [], h4 or [])
        m15 = m15 or []
        m30 = m30 or []
        h1 = h1 or []
        h4 = h4 or []
        if m15:
            try:
                base["last_price"] = float(m15[-1].close)
                base["current_price"] = float(m15[-1].close)
            except Exception:
                pass
        base["meta"] = meta
        return base

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


def _detect_liquidation_v2(
    m15c: Sequence[Any],
    atr15: float,
    sweep_buy: Dict[str, Any],
    sweep_sell: Dict[str, Any],
    spring_buy: Dict[str, Any],
    spring_sell: Dict[str, Any],
) -> Dict[str, Any]:
    if not m15c or len(m15c) < 3 or not atr15:
        return {"ok": False}
    c = list(m15c)[-2]  # last closed candle
    rng = max(1e-9, float(c.high) - float(c.low))
    body = abs(float(c.close) - float(c.open))
    upper = float(c.high) - max(float(c.open), float(c.close))
    lower = min(float(c.open), float(c.close)) - float(c.low)
    close_pos = (float(c.close) - float(c.low)) / rng
    range_atr = rng / max(1e-9, float(atr15))
    body_atr = body / max(1e-9, float(atr15))

    if sweep_buy.get("ok") or spring_buy.get("ok"):
        return {
            "ok": True, "side": "BUY LIQUIDATION", "kind": "sweep_to_buy",
            "body_atr": body_atr, "range_atr": range_atr,
            "severity": "HIGH" if range_atr >= 2.0 else "MEDIUM",
        }
    if sweep_sell.get("ok") or spring_sell.get("ok"):
        return {
            "ok": True, "side": "SELL LIQUIDATION", "kind": "sweep_to_sell",
            "body_atr": body_atr, "range_atr": range_atr,
            "severity": "HIGH" if range_atr >= 2.0 else "MEDIUM",
        }

    if range_atr >= 2.2 and body_atr >= 1.1:
        if close_pos <= 0.22:
            return {"ok": True, "side": "SELL LIQUIDATION", "kind": "panic_dump", "body_atr": body_atr, "range_atr": range_atr, "severity": "HIGH"}
        if close_pos >= 0.78:
            return {"ok": True, "side": "BUY LIQUIDATION", "kind": "panic_pump", "body_atr": body_atr, "range_atr": range_atr, "severity": "HIGH"}

    if range_atr >= 1.5 and (upper / rng >= 0.55 or lower / rng >= 0.55):
        side = "SELL SWEEP" if upper > lower else "BUY SWEEP"
        return {"ok": True, "side": side, "kind": "stop_hunt", "body_atr": body_atr, "range_atr": range_atr, "severity": "LOW"}

    return {"ok": False, "body_atr": body_atr, "range_atr": range_atr}

def _detect_market_state_v2(
    h1_trend: str,
    h4_trend: str,
    range_pos: Optional[float],
    atr15: float,
    avg20: float,
    avg80: float,
    div: Dict[str, Any],
    liquidation_evt: Dict[str, Any],
) -> str:
    spike = bool(avg80 > 0 and avg20 > 1.35 * avg80)
    if liquidation_evt.get("ok"):
        side = str(liquidation_evt.get("side") or "")
        if "SELL" in side:
            return "POST_LIQUIDATION_BOUNCE"
        if "BUY" in side:
            return "POST_SHORT_COVER"
    if h1_trend == "bearish" and h4_trend == "bearish":
        if div.get("bull"):
            return "EXHAUSTION_DOWN"
        if range_pos is not None and range_pos >= 0.45:
            return "BOUNCE_TO_SELL"
        return "TREND_DOWN" if spike else "PULLBACK_DOWN"
    if h1_trend == "bullish" and h4_trend == "bullish":
        if div.get("bear"):
            return "EXHAUSTION_UP"
        if range_pos is not None and range_pos <= 0.55:
            return "DIP_TO_BUY"
        return "TREND_UP" if spike else "PULLBACK_UP"
    if range_pos is not None and 0.20 <= range_pos <= 0.80:
        return "CHOP"
    return "TRANSITION"

def _detect_flow_state_v2(
    symbol: str,
    h1_trend: str,
    h4_trend: str,
    market_state_v2: str,
    range_pos: Optional[float],
) -> Dict[str, Any]:
    sym = str(symbol or "").upper()
    if h1_trend == "bearish" and h4_trend == "bearish":
        state = "OUTFLOW" if "XAU" in sym or "XAG" in sym else "RISK_OFF"
        favored = "SELL"
        note = "smart money selling rally" if market_state_v2 in ("BOUNCE_TO_SELL", "PULLBACK_DOWN") else "trend pressure still down"
    elif h1_trend == "bullish" and h4_trend == "bullish":
        state = "INFLOW" if "BTC" in sym else "RISK_ON"
        favored = "BUY"
        note = "smart money buying dip" if market_state_v2 in ("DIP_TO_BUY", "PULLBACK_UP") else "trend pressure still up"
    else:
        state = "NEUTRAL"
        favored = "NONE"
        note = "flow chưa rõ"
    if range_pos is not None and 0.25 <= range_pos <= 0.75 and state != "NEUTRAL":
        note += " | mid-range"
    return {"state": state, "favored_side": favored, "note": note}

def _detect_no_trade_zone_v2(
    bias_side: Optional[str],
    market_state_v2: str,
    range_pos: Optional[float],
    liq_warn_flag: bool,
    liquidation_evt: Dict[str, Any],
    confirmation_ok: Optional[bool] = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    if market_state_v2 in ("CHOP", "TRANSITION"):
        reasons.append("market state nhiễu")
    if liq_warn_flag:
        reasons.append("liquidity warning")
    if liquidation_evt.get("ok"):
        reasons.append("vừa có liquidation")
    if range_pos is not None and 0.25 <= range_pos <= 0.75:
        reasons.append("mid-range")
    if not bias_side or bias_side == "NONE":
        reasons.append("bias chưa rõ")
    if confirmation_ok is False:
        reasons.append("chưa có confirm")
    active = False
    if market_state_v2 in ("CHOP", "POST_LIQUIDATION_BOUNCE", "POST_SHORT_COVER"):
        active = True
    if len(reasons) >= 2 and (range_pos is None or 0.15 <= range_pos <= 0.85):
        active = True
    return {"active": active, "reasons": reasons}

def _detect_phase_369_v2(
    bias_side: Optional[str],
    market_state_v2: str,
    playbook: Dict[str, Any],
    range_pos: Optional[float],
    liquidation_evt: Dict[str, Any],
    no_trade_zone: Dict[str, Any],
) -> Dict[str, Any]:
    plan = str(playbook.get("plan") or "")
    if liquidation_evt.get("ok"):
        return {"phase": 3, "label": "EXTREME", "meaning": "liquidation / panic zone", "reason": str(liquidation_evt.get("kind") or liquidation_evt.get("side") or "liquidation")}
    if no_trade_zone.get("active") and market_state_v2 in ("CHOP", "TRANSITION"):
        return {"phase": 3, "label": "EARLY", "meaning": "nhiễu / chưa có lợi thế", "reason": "; ".join(no_trade_zone.get("reasons") or []) or "no-trade"}
    if plan in ("BOUNCE_TO_SELL", "SELL_RALLY"):
        return {"phase": 6, "label": "BOUNCE_TO_SELL", "meaning": "hồi để bán", "reason": f"range-pos={int((range_pos or 0)*100)}%"}
    if plan in ("DIP_TO_BUY", "BUY_DIP"):
        return {"phase": 6, "label": "DIP_TO_BUY", "meaning": "hồi để mua", "reason": f"range-pos={int((range_pos or 0)*100)}%"}
    if bias_side == "SELL" and range_pos is not None and range_pos <= 0.18:
        return {"phase": 9, "label": "LATE", "meaning": "đang sát đáy, dễ sell trễ", "reason": f"range-pos={int(range_pos*100)}%"}
    if bias_side == "BUY" and range_pos is not None and range_pos >= 0.82:
        return {"phase": 9, "label": "LATE", "meaning": "đang sát đỉnh, dễ buy trễ", "reason": f"range-pos={int(range_pos*100)}%"}
    return {"phase": 6, "label": "READY", "meaning": "đợi đủ bias + confirm", "reason": f"range-pos={int((range_pos or 0)*100)}%"}

def _attach_gd2_meta(
    base: Dict[str, Any],
    flow_state: Dict[str, Any],
    market_state_v2: str,
    liquidation_evt: Dict[str, Any],
    no_trade_zone: Dict[str, Any],
    phase_369: Dict[str, Any],
    playbook_v2: Dict[str, Any],
) -> None:
    meta = base.setdefault("meta", {})
    meta["flow_state"] = flow_state
    meta["market_state_v2"] = market_state_v2
    meta["liquidation"] = liquidation_evt
    meta["no_trade_zone"] = no_trade_zone
    meta["phase_369"] = phase_369
    meta["playbook_v2"] = playbook_v2


def _build_narrative_v3(symbol: str, bias_side: Optional[str], market_state_v2: str,
                        flow_state: Dict[str, Any], liquidation_evt: Dict[str, Any],
                        playbook_v2: Dict[str, Any], no_trade_zone: Dict[str, Any]) -> Dict[str, Any]:
    sym = str(symbol or "").upper()
    asset = "BTC" if "BTC" in sym else ("XAU" if "XAU" in sym else ("XAG" if "XAG" in sym else sym))
    plan = str(playbook_v2.get("plan") or "OBSERVE")
    flow = str(flow_state.get("state") or "NEUTRAL")
    ms = str(market_state_v2 or "")
    headline = "Quan sát thêm"
    summary = "Thị trường chưa cho lợi thế rõ."
    danger = []
    if liquidation_evt.get("ok"):
        headline = "Sau liquidation"
        summary = f"{asset} vừa có liquidation move; ưu tiên chờ phản ứng sau panic, không follow ngay."
        danger.append("biến động 2 đầu")
    elif plan in ("BOUNCE_TO_SELL", "SELL_RALLY", "WAIT_BOUNCE_TO_SELL"):
        headline = "Hồi để bán"
        summary = f"{asset} đang hồi kỹ thuật trong bối cảnh giảm; ưu tiên sell-the-rally, tránh sell đáy."
        danger.append("fake reversal")
    elif plan in ("DIP_TO_BUY", "BUY_DIP", "WAIT_PULLBACK_TO_BUY"):
        headline = "Hồi để mua"
        summary = f"{asset} đang pullback trong xu hướng tăng; ưu tiên buy-the-dip, tránh buy đuổi."
        danger.append("fake breakdown")
    elif ms in ("CHOP", "TRANSITION"):
        headline = "Nhiễu / chuyển pha"
        summary = f"{asset} đang ở trạng thái nhiễu hoặc chuyển pha; edge thấp, nên đứng ngoài là chính."
        danger.append("stop-hunt")
    elif flow in ("OUTFLOW", "RISK_OFF"):
        headline = "Dòng tiền chưa ủng hộ"
        summary = f"Flow hiện tại chưa ủng hộ {asset}; thuận flow vẫn ưu tiên SELL nếu có setup rõ."
    elif flow in ("INFLOW", "RISK_ON"):
        headline = "Dòng tiền đang ủng hộ"
        summary = f"Flow hiện tại ủng hộ {asset}; ưu tiên BUY khi có pullback hoặc break xác nhận."
    if no_trade_zone.get("active"):
        danger.extend(no_trade_zone.get("reasons") or [])
    return {
        "headline": headline,
        "summary": summary,
        "danger": [str(x) for x in danger if x],
    }


def _build_scenario_v3(bias_side: Optional[str], playbook_v2: Dict[str, Any], key_levels: Dict[str, Any],
                       flow_state: Dict[str, Any], market_state_v2: str, no_trade_zone: Dict[str, Any]) -> Dict[str, Any]:
    plan = str(playbook_v2.get("plan") or "OBSERVE")
    zl = _safe_float(playbook_v2.get("zone_low"))
    zh = _safe_float(playbook_v2.get("zone_high"))
    favored = str(flow_state.get("favored_side") or "NONE")
    base_case = "Quan sát thêm"
    alt_case = "Chờ break xác nhận"
    invalid = "Mất cấu trúc hiện tại"
    best_zone = None
    if zl is not None and zh is not None:
        best_zone = f"{zl:.2f} – {zh:.2f}"
    if plan in ("BOUNCE_TO_SELL", "SELL_RALLY", "WAIT_BOUNCE_TO_SELL"):
        base_case = f"Base case: hồi để SELL{' ở vùng ' + best_zone if best_zone else ''}"
        alt_level = _safe_float(key_levels.get("H1_HH") or key_levels.get("M15_RANGE_HIGH"))
        alt_case = f"Alt case: nếu giữ trên {alt_level:.2f} → chuyển sang reversal candidate" if alt_level is not None else "Alt case: nếu break đỉnh mạnh → reversal candidate"
        inv = _safe_float(key_levels.get("H1_LH") or key_levels.get("M15_RANGE_HIGH"))
        invalid = f"Invalid if: M15/H1 giữ trên {inv:.2f}" if inv is not None else invalid
    elif plan in ("DIP_TO_BUY", "BUY_DIP", "WAIT_PULLBACK_TO_BUY"):
        base_case = f"Base case: hồi để BUY{' ở vùng ' + best_zone if best_zone else ''}"
        alt_level = _safe_float(key_levels.get("H1_LL") or key_levels.get("M15_RANGE_LOW"))
        alt_case = f"Alt case: nếu thủng {alt_level:.2f} → breakdown / reversal risk" if alt_level is not None else "Alt case: nếu thủng đáy mạnh → breakdown risk"
        inv = _safe_float(key_levels.get("H1_HL") or key_levels.get("M15_RANGE_LOW"))
        invalid = f"Invalid if: M15/H1 mất {inv:.2f}" if inv is not None else invalid
    elif no_trade_zone.get("active"):
        base_case = "Base case: NO TRADE / đứng ngoài"
        alt_case = "Alt case: chờ displacement thật + follow-through"
        invalid = "Invalid if: market state thoát khỏi CHOP/TRANSITION"
    elif favored in ("BUY", "SELL") and bias_side in ("BUY", "SELL"):
        base_case = f"Base case: ưu tiên {favored} theo flow"
        alt_case = "Alt case: chỉ đổi kịch bản nếu break major level"
    return {
        "base_case": base_case,
        "alt_case": alt_case,
        "invalid_if": invalid,
        "best_zone": best_zone,
    }


def _attach_gd3_meta(base: Dict[str, Any], narrative_v3: Dict[str, Any], scenario_v3: Dict[str, Any]) -> None:
    meta = base.setdefault("meta", {})
    meta["narrative_v3"] = narrative_v3
    meta["scenario_v3"] = scenario_v3

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



def _detect_playbook_v2(symbol: str, bias_side: Optional[str], h1_trend: str, market_state_v2: str,
                        m15c: Sequence[Any], flow_state: Dict[str, Any], no_trade_zone: Dict[str, Any],
                        liquidation_evt: Dict[str, Any]) -> Dict[str, Any]:
    """Human-readable plan for NOW/REVIEW. Not an auto-entry engine."""
    lo, hi, last = _range_levels(m15c[:-1] if len(m15c) > 1 else m15c, n=20)
    pos = None
    if lo is not None and hi is not None and hi > lo and last is not None:
        pos = (last - lo) / max(1e-9, hi - lo)

    favored = flow_state.get("favored_side")
    plan = "OBSERVE"
    why = []
    zone_low = zone_high = None

    if no_trade_zone.get("active"):
        plan = "NO_TRADE"
        why.extend(no_trade_zone.get("reasons") or [])
    elif liquidation_evt.get("ok"):
        plan = "POST_LIQUIDATION_WAIT"
        why.append("vừa có liquidation move")
    elif market_state_v2 in ("BOUNCE_TO_SELL", "PULLBACK_DOWN", "TREND_DOWN", "EXHAUSTION_DOWN") or (bias_side == "SELL" and h1_trend == "bearish"):
        if pos is not None and pos <= 0.22:
            plan = "WAIT_BOUNCE_TO_SELL"
            why.append("đang sát đáy range, không sell đáy")
        elif pos is not None and pos <= 0.55:
            plan = "BOUNCE_TO_SELL"
            why.append("hồi kỹ thuật trong xu hướng giảm")
        else:
            plan = "SELL_RALLY"
            why.append("đang ở vùng hồi thuận trend giảm")
        if lo is not None and hi is not None:
            span = hi - lo
            zone_low = lo + 0.52 * span
            zone_high = lo + 0.82 * span
    elif market_state_v2 in ("DIP_TO_BUY", "PULLBACK_UP", "TREND_UP", "EXHAUSTION_UP") or (bias_side == "BUY" and h1_trend == "bullish"):
        if pos is not None and pos >= 0.78:
            plan = "WAIT_PULLBACK_TO_BUY"
            why.append("đang sát đỉnh range, không buy đuổi")
        elif pos is not None and pos >= 0.45:
            plan = "DIP_TO_BUY"
            why.append("pullback trong xu hướng tăng")
        else:
            plan = "BUY_DIP"
            why.append("đang ở vùng hồi đẹp thuận trend tăng")
        if lo is not None and hi is not None:
            span = hi - lo
            zone_low = lo + 0.18 * span
            zone_high = lo + 0.48 * span
    else:
        plan = "RANGE_WAIT"
        why.append("thị trường đi ngang / chuyển pha")

    if favored and bias_side and favored not in ("NONE", bias_side):
        why.append(f"flow proxy ưu tiên {favored}")

    return {
        "plan": plan,
        "why": why,
        "range_pos": pos,
        "zone_low": zone_low,
        "zone_high": zone_high,
    }



# =========================
# GD4 MODULES (append-only on V3)
# =========================
def _session_engine_v4(m15c: Sequence[Any], market_state_v2: str) -> Dict[str, Any]:
    if not m15c or len(m15c) < 16:
        return {"session_tag": "N/A", "follow_through": "N/A", "fake_move_risk": "MEDIUM", "bias": "NONE"}
    closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
    recent = closed[-8:]
    first = recent[:4]
    last = recent[4:]
    move1 = (float(first[-1].close) - float(first[0].open)) if first else 0.0
    move2 = (float(last[-1].close) - float(last[0].open)) if last else 0.0
    hi = max(float(c.high) for c in recent)
    lo = min(float(c.low) for c in recent)
    span = max(1e-9, hi - lo)
    if abs(move1) / span >= 0.45:
        tag = "OPEN_IMPULSE_UP" if move1 > 0 else "OPEN_IMPULSE_DOWN"
    else:
        tag = "SESSION_CHOP"
    same_dir = (move1 > 0 and move2 > 0) or (move1 < 0 and move2 < 0)
    follow = "YES" if same_dir and abs(move2) >= 0.20 * span else "NO"
    fake = "HIGH" if tag != "SESSION_CHOP" and follow == "NO" else ("LOW" if follow == "YES" else "MEDIUM")
    bias = "BUY" if "UP" in tag and follow == "YES" else ("SELL" if "DOWN" in tag and follow == "YES" else "NONE")
    if market_state_v2 in ("CHOP", "TRANSITION"):
        fake = "HIGH"
    return {"session_tag": tag, "follow_through": follow, "fake_move_risk": fake, "bias": bias}

def _htf_pressure_v4(h1c: Sequence[Any], h4c: Sequence[Any]) -> Dict[str, Any]:
    def _close_bias(candles, n=4):
        if not candles or len(candles) < n + 1:
            return "NEUTRAL", 0
        cs = [float(_c_val(x, "close", 0.0) or 0.0) for x in candles[-(n+1):]]
        diffs = [cs[i] - cs[i-1] for i in range(1, len(cs))]
        up = sum(1 for d in diffs if d > 0)
        dn = sum(1 for d in diffs if d < 0)
        if dn >= max(3, n-1):
            return "DOWN", dn
        if up >= max(3, n-1):
            return "UP", up
        return "MIXED", max(up, dn)
    h1_bias, h1_cnt = _close_bias(h1c, 4)
    h4_bias, h4_cnt = _close_bias(h4c, 4)
    if h1_bias == "DOWN" and h4_bias == "DOWN":
        state = "BEARISH_STRONG"
    elif h1_bias == "UP" and h4_bias == "UP":
        state = "BULLISH_STRONG"
    elif "DOWN" in (h1_bias, h4_bias):
        state = "BEARISH_WEAK"
    elif "UP" in (h1_bias, h4_bias):
        state = "BULLISH_WEAK"
    else:
        state = "NEUTRAL"
    stability = "HIGH" if state.endswith("STRONG") else ("MEDIUM" if state != "NEUTRAL" else "LOW")
    return {"state": state, "h1_close_bias": h1_bias, "h4_close_bias": h4_bias, "stability": stability}

def _close_confirmation_v4(m15c: Sequence[Any], bias_side: Optional[str], bos_level: Optional[float]) -> Dict[str, Any]:
    if not m15c or len(m15c) < 4 or bos_level is None or bias_side not in ("BUY", "SELL"):
        return {"break_valid": False, "strength": "N/A", "hold": "N/A"}
    c1 = list(m15c)[-3]
    c2 = list(m15c)[-2]
    level = float(bos_level)
    if bias_side == "BUY":
        break_valid = float(c1.close) > level and float(c2.close) >= level
        strength = "STRONG" if break_valid and float(c1.close) > max(float(c1.open), level) else ("WEAK" if float(c1.high) > level else "NO")
        hold = "YES" if float(c2.low) >= level or float(c2.close) >= level else "NO"
    else:
        break_valid = float(c1.close) < level and float(c2.close) <= level
        strength = "STRONG" if break_valid and float(c1.close) < min(float(c1.open), level) else ("WEAK" if float(c1.low) < level else "NO")
        hold = "YES" if float(c2.high) <= level or float(c2.close) <= level else "NO"
    return {"break_valid": bool(break_valid), "strength": strength, "hold": hold, "level": level}

def _macro_intermarket_v4(symbol: str, flow_state: Dict[str, Any], h1_trend: str, market_state_v2: str) -> Dict[str, Any]:
    sym = str(symbol or "").upper()
    if "XAU" in sym or "XAG" in sym:
        if flow_state.get("state") == "OUTFLOW":
            headline = "USD headwind / metals outflow"
            bias = "SELL"
        elif flow_state.get("state") == "INFLOW":
            headline = "metals supported / USD soft"
            bias = "BUY"
        else:
            headline = "macro mixed"
            bias = "NONE"
    else:
        if flow_state.get("state") in ("INFLOW", "RISK_ON"):
            headline = "risk appetite supportive"
            bias = "BUY"
        elif flow_state.get("state") in ("OUTFLOW", "RISK_OFF"):
            headline = "risk-off / pressure remains"
            bias = "SELL"
        else:
            headline = "macro mixed"
            bias = "NONE"
    note = "environment aligns with trend" if ((bias == "BUY" and h1_trend == "bullish") or (bias == "SELL" and h1_trend == "bearish")) else "macro chưa thật sự rõ"
    if market_state_v2 in ("CHOP", "TRANSITION"):
        note = "market regime nhiễu, macro edge thấp"
    return {"headline": headline, "bias": bias, "note": note}

def _refine_playbook_v4(playbook_v2: Dict[str, Any], close_confirm: Dict[str, Any], session_v4: Dict[str, Any],
                        htf_pressure_v4: Dict[str, Any], macro_v4: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(playbook_v2 or {})
    plan = str(out.get("plan") or "OBSERVE")
    triggers = []
    if close_confirm.get("strength") not in (None, "N/A", "NO"):
        triggers.append(f"close-confirm {close_confirm.get('strength')}")
    if session_v4.get("follow_through") == "YES":
        triggers.append("session follow-through")
    if htf_pressure_v4.get("state") in ("BEARISH_STRONG", "BULLISH_STRONG"):
        triggers.append(f"HTF {htf_pressure_v4.get('state')}")
    if macro_v4.get("bias") in ("BUY", "SELL"):
        triggers.append(f"macro {macro_v4.get('bias')}")
    out["trigger_pack"] = triggers
    out["quality"] = "HIGH" if len(triggers) >= 3 else ("MEDIUM" if len(triggers) >= 2 else "LOW")
    return out

def _attach_gd4_meta(base: Dict[str, Any], session_v4: Dict[str, Any], htf_pressure_v4: Dict[str, Any],
                     close_confirm_v4: Dict[str, Any], macro_v4: Dict[str, Any], playbook_v4: Dict[str, Any]) -> None:
    meta = base.setdefault("meta", {})
    meta["session_v4"] = session_v4
    meta["htf_pressure_v4"] = htf_pressure_v4
    meta["close_confirm_v4"] = close_confirm_v4
    meta["macro_v4"] = macro_v4
    meta["playbook_v4"] = playbook_v4

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
        "session": _vn_session_label(),
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
    last_close_15 = None
    last_close_30 = None
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

    # derive major_bearish for gating logic
    try:
        h1_tag = ((base.get("meta") or {}).get("structure") or {}).get("H1")
        major_bearish = str(h1_tag) in ("LL-LH", "LH–LL", "LH-LL")
    except Exception:
        major_bearish = False
        return base

    m15c = _safe_candles(m15)
    m30c = _safe_candles(m30)
    h1c = _safe_candles(h1)
    h4c = _safe_candles(h4)
    if not m15c or not m30c:
        base["note_lines"].append("⚠️ Không đọc được nến M15/M30 sau khi chuẩn hoá dữ liệu.")
        base["short_hint"] = ["- Dữ liệu nến lỗi / thiếu → CHỜ KÈO"]
        base["context_lines"] = ["Thị trường: n/a", "H1: n/a"]
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
        return base
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
    liq_warn = bool(lw)

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

    sweep_buy["grade"] = _sweep_grade(sweep_buy)
    sweep_sell["grade"] = _sweep_grade(sweep_sell)
    spring_buy["grade"] = "STRONG" if spring_buy.get("ok") else "NONE"
    spring_sell["grade"] = "STRONG" if spring_sell.get("ok") else "NONE"

    liq_sell = bool(sweep_sell.get("ok")) or bool(spring_sell.get("ok"))
    liq_buy  = bool(sweep_buy.get("ok"))  or bool(spring_buy.get("ok"))

    liquidity_lines = []
    if sweep_sell.get("ok"):
        vtxt = " +VOL" if sweep_sell.get("vol_ok") else ""
        liquidity_lines.append(f"🔴 Sweep HIGH (quét đỉnh){vtxt}: chọc {_fmt(sweep_sell['level'])} rồi đóng xuống lại | lực={sweep_sell.get('grade','NONE')}")
        score += 1

    if sweep_buy.get("ok"):
        vtxt = " +VOL" if sweep_buy.get("vol_ok") else ""
        liquidity_lines.append(f"🟢 Sweep LOW (quét đáy){vtxt}: chọc {_fmt(sweep_buy['level'])} rồi đóng lên lại | lực={sweep_buy.get('grade','NONE')}")
        score += 1

    if spring_buy.get("ok"):
        vtxt = " +VOL" if spring_buy.get("vol_ok") else ""
        liquidity_lines.append(f"🟢 SPRING (false break đáy){vtxt}: phá range_low rồi kéo lên + follow-through | lực={spring_buy.get('grade','NONE')}")
        score += 1

    if spring_sell.get("ok"):
        vtxt = " +VOL" if spring_sell.get("vol_ok") else ""
        liquidity_lines.append(f"🔴 UPTHRUST (false break đỉnh){vtxt}: phá range_high rồi kéo xuống + follow-through | lực={spring_sell.get('grade','NONE')}")
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

    # ===== GD2 initial context (available for all early returns) =====
    lo20, hi20, last20 = _range_levels(closed15, n=20)
    range_pos = None
    if lo20 is not None and hi20 is not None and hi20 > lo20 and last20 is not None:
        range_pos = (last20 - lo20) / max(1e-9, hi20 - lo20)

    h4_trend = _trend_label(h4c)
    liquidation_evt = _detect_liquidation_v2(m15c, atr15, sweep_buy, sweep_sell, spring_buy, spring_sell)
    bias_guess = "BUY" if h1_trend == "bullish" else ("SELL" if h1_trend == "bearish" else None)
    market_state_v2 = _detect_market_state_v2(h1_trend, h4_trend, range_pos, atr15, avg20, avg80, div, liquidation_evt)
    flow_state = _detect_flow_state_v2(symbol, h1_trend, h4_trend, market_state_v2, range_pos)
    no_trade_zone = _detect_no_trade_zone_v2(bias_guess, market_state_v2, range_pos, liq_warn, liquidation_evt, confirmation_ok=None)
    playbook_v2 = _detect_playbook_v2(symbol, bias_guess, h1_trend, market_state_v2, m15c, flow_state, no_trade_zone, liquidation_evt)
    phase_369_v2 = _detect_phase_369_v2(bias_guess, market_state_v2, playbook_v2, range_pos, liquidation_evt, no_trade_zone)
    _attach_gd2_meta(base, flow_state, market_state_v2, liquidation_evt, no_trade_zone, phase_369_v2, playbook_v2)
    _attach_gd3_meta(
        base,
        _build_narrative_v3(symbol, bias_guess, market_state_v2, flow_state, liquidation_evt, playbook_v2, no_trade_zone),
        _build_scenario_v3(bias_guess, playbook_v2, base.get("meta", {}).get("key_levels", {}), flow_state, market_state_v2, no_trade_zone),
    )

    if flow_state.get("state"):
        context_lines.append(f"Flow proxy: {flow_state.get('state')} | Ưu tiên {flow_state.get('favored_side') or 'n/a'}")
    if market_state_v2:
        context_lines.append(f"State: {market_state_v2}")
    if liquidation_evt.get("ok"):
        notes.append(
            f"⚠️ Liquidation move: {liquidation_evt.get('side')} | body~{liquidation_evt.get('body_atr', 0):.1f} ATR | range~{liquidation_evt.get('range_atr', 0):.1f} ATR"
        )
    if no_trade_zone.get("active"):
        notes.append("⛔ No-trade zone: " + "; ".join(no_trade_zone.get("reasons") or []))
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
                base["last_price"] = float(last_close_15)
                base["current_price"] = float(last_close_15)
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
                base["last_price"] = float(last_close_15)
                base["current_price"] = float(last_close_15)
                return base

    # ===== Quality =====
    if rej.get("upper_reject"):
        txt = "Nến từ chối tăng rõ (râu trên dài)"
        if range_pos is not None and float(range_pos) > 0.70:
            txt += " → xuất hiện ở vùng cao, dễ xảy ra pullback"
        quality_lines.append(txt)
        score += 1
    
    elif rej.get("lower_reject"):
        txt = "Nến từ chối giảm rõ (râu dưới dài)"
        if range_pos is not None and float(range_pos) < 0.30:
            txt += " → xuất hiện ở vùng thấp, có thể bật lên"
        quality_lines.append(txt)
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


    # ===== GD2 recompute after bias decision =====
    bias_for_gd2 = bias if bias in ("BUY", "SELL") else bias_guess
    market_state_v2 = _detect_market_state_v2(h1_trend, h4_trend, range_pos, atr15, avg20, avg80, div, liquidation_evt)
    flow_state = _detect_flow_state_v2(symbol, h1_trend, h4_trend, market_state_v2, range_pos)
    no_trade_zone = _detect_no_trade_zone_v2(bias_for_gd2, market_state_v2, range_pos, liq_warn, liquidation_evt, confirmation_ok=None)
    playbook_v2 = _detect_playbook_v2(symbol, bias_for_gd2, h1_trend, market_state_v2, m15c, flow_state, no_trade_zone, liquidation_evt)
    phase_369_v2 = _detect_phase_369_v2(bias_for_gd2, market_state_v2, playbook_v2, range_pos, liquidation_evt, no_trade_zone)
    _attach_gd2_meta(base, flow_state, market_state_v2, liquidation_evt, no_trade_zone, phase_369_v2, playbook_v2)
    _attach_gd3_meta(
        base,
        _build_narrative_v3(symbol, bias_for_gd2, market_state_v2, flow_state, liquidation_evt, playbook_v2, no_trade_zone),
        _build_scenario_v3(bias_for_gd2, playbook_v2, base.get("meta", {}).get("key_levels", {}), flow_state, market_state_v2, no_trade_zone),
    )
    session_v4 = _session_engine_v4(m15c, market_state_v2)
    htf_pressure_v4 = _htf_pressure_v4(h1c, h4c)
    close_confirm_v4 = _close_confirmation_v4(m15c, bias_for_gd2, (base.get('meta', {}).get('key_levels', {}) or {}).get('M15_BOS'))
    macro_v4 = _macro_intermarket_v4(symbol, flow_state, h1_trend, market_state_v2)
    playbook_v4 = _refine_playbook_v4(playbook_v2, close_confirm_v4, session_v4, htf_pressure_v4, macro_v4)
    _attach_gd4_meta(base, session_v4, htf_pressure_v4, close_confirm_v4, macro_v4, playbook_v4)

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
        base["last_price"] = float(last_close_15)
        base["current_price"] = float(last_close_15)
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
            base["last_price"] = float(last_close_15)
            base["current_price"] = float(last_close_15)
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
            base["last_price"] = float(last_close_15)
            base["current_price"] = float(last_close_15)
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
        base["last_price"] = float(last_close_15)
        base["current_price"] = float(last_close_15)
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

    ez_low_v6, ez_high_v6 = _entry_zone_v6(bias_side, base["meta"]["key_levels"], atr15)
    base["meta"]["entry_zone_v6"] = {
        "low": ez_low_v6,
        "high": ez_high_v6,
        "side": bias_side,
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
    major_bearish = False

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
    liq_warn = ("Liquidity WARNING" in " | ".join(context_lines or []))
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

    # ===== GD2 final attach with scoring-aware no-trade =====
    no_trade_zone = _detect_no_trade_zone_v2(
        bias_side,
        market_state_v2,
        range_pos,
        liq_warn,
        liquidation_evt,
        confirmation_ok=bool(momentum_ok),
    )
    playbook_v2 = _detect_playbook_v2(symbol, bias_side, h1_trend, market_state_v2, m15c, flow_state, no_trade_zone, liquidation_evt)
    phase_369_v2 = _detect_phase_369_v2(bias_side, market_state_v2, playbook_v2, range_pos, liquidation_evt, no_trade_zone)
    _attach_gd2_meta(base, flow_state, market_state_v2, liquidation_evt, no_trade_zone, phase_369_v2, playbook_v2)
    _attach_gd3_meta(
        base,
        _build_narrative_v3(symbol, bias_side, market_state_v2, flow_state, liquidation_evt, playbook_v2, no_trade_zone),
        _build_scenario_v3(bias_side, playbook_v2, base.get("meta", {}).get("key_levels", {}), flow_state, market_state_v2, no_trade_zone),
    )
    session_v4 = _session_engine_v4(m15c, market_state_v2)
    htf_pressure_v4 = _htf_pressure_v4(h1c, h4c)
    close_confirm_v4 = _close_confirmation_v4(m15c, bias_side, (base.get('meta', {}).get('key_levels', {}) or {}).get('M15_BOS'))
    macro_v4 = _macro_intermarket_v4(symbol, flow_state, h1_trend, market_state_v2)
    playbook_v4 = _refine_playbook_v4(playbook_v2, close_confirm_v4, session_v4, htf_pressure_v4, macro_v4)
    _attach_gd4_meta(base, session_v4, htf_pressure_v4, close_confirm_v4, macro_v4, playbook_v4)

    sweep_grade_v6 = "NONE"
    if bias_side == "BUY":
        sweep_grade_v6 = spring_buy.get("grade") if spring_buy.get("ok") else sweep_buy.get("grade")
    elif bias_side == "SELL":
        sweep_grade_v6 = spring_sell.get("grade") if spring_sell.get("ok") else sweep_sell.get("grade")
    base.setdefault("meta", {})["grade_v6"] = _grade_v6(base.get("meta", {}), trade_mode, sweep_grade_v6, close_confirm_v4)
    base["meta"]["sweep_grade_v6"] = sweep_grade_v6

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
    base["last_price"] = float(last_close_15)
    base["current_price"] = float(last_close_15)
    return base

    # =========================
    # Formatter (MUST be named format_signal for main.py import)
    # =========================

def _safe_float(x):
    """Convert various numeric formats to float safely.
    Supports strings like '67,123.45' or '67.123,45'. Returns None if cannot parse.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip()
        if not s:
            return None
        # handle formats with both '.' and ','
        if '.' in s and ',' in s:
            # if comma appears after dot => likely decimal comma, dot thousands
            if s.rfind(',') > s.rfind('.'):
                s = s.replace('.', '').replace(',', '.')
            else:
                s = s.replace(',', '')
        else:
            # only comma: could be decimal comma
            if ',' in s and '.' not in s:
                s = s.replace(',', '.')
        return float(s)
    except Exception:
        return None

def format_signal(sig: Dict[str, Any]) -> str:
    """Telegram formatter V5: dễ đọc hơn, giữ nguyên logic/meta hiện có."""
    def sf(x):
        try:
            return _safe_float(x)
        except Exception:
            try:
                return float(x)
            except Exception:
                return None

    def nf(x, nd: int = 2, default: str = "...") -> str:
        v = sf(x)
        if v is None:
            return default
        return f"{v:.{nd}f}"

    if not isinstance(sig, dict):
        return "❌ Invalid signal payload."

    meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
    symbol = sig.get("symbol") or "n/a"
    tf = sig.get("tf") or "n/a"
    session = sig.get("session") or ""
    data_source = sig.get("data_source") or meta.get("data_source") or ""
    trade_mode = str(sig.get("trade_mode") or "MANUAL").upper()
    stars = max(0, min(5, int(sig.get("stars", 0) or 0)))
    rec_raw = str(sig.get("recommendation") or "CHỜ")
    rec = "MUA" if "BUY" in rec_raw.upper() else ("BÁN" if "SELL" in rec_raw.upper() else "CHỜ")

    narrative = meta.get("narrative_v3") if isinstance(meta.get("narrative_v3"), dict) else {}
    scenario = meta.get("scenario_v3") if isinstance(meta.get("scenario_v3"), dict) else {}
    phase = meta.get("phase_369") if isinstance(meta.get("phase_369"), dict) else {}
    flow = meta.get("flow_state") if isinstance(meta.get("flow_state"), dict) else {}
    liq_evt = meta.get("liquidation") if isinstance(meta.get("liquidation"), dict) else {}
    ntz = meta.get("no_trade_zone") if isinstance(meta.get("no_trade_zone"), dict) else {}
    playbook = meta.get("playbook_v2") if isinstance(meta.get("playbook_v2"), dict) else {}
    k = meta.get("key_levels") if isinstance(meta.get("key_levels"), dict) else {}
    struct = meta.get("structure") if isinstance(meta.get("structure"), dict) else {}

    session_v4 = meta.get("session_v4") if isinstance(meta.get("session_v4"), dict) else {}
    htf_pressure_v4 = meta.get("htf_pressure_v4") if isinstance(meta.get("htf_pressure_v4"), dict) else {}
    close_confirm_v4 = meta.get("close_confirm_v4") if isinstance(meta.get("close_confirm_v4"), dict) else {}
    macro_v4 = meta.get("macro_v4") if isinstance(meta.get("macro_v4"), dict) else {}
    playbook_v4 = meta.get("playbook_v4") if isinstance(meta.get("playbook_v4"), dict) else {}

    ctx_lines = sig.get("context_lines") or []
    liq_lines = sig.get("liquidity_lines") or []
    q_lines = sig.get("quality_lines") or []
    notes = sig.get("notes") or sig.get("note_lines") or []

    def add(buf: list[str], x: str = ""):
        if x is None:
            return
        s = str(x).rstrip()
        if s:
            buf.append(s)
        elif buf and buf[-1] != "":
            buf.append("")

    def phase_text(pobj: dict) -> str:
        label = str((pobj or {}).get("label") or "").upper()
        mapping = {
            "EARLY": "Giai đoạn sớm",
            "READY": "Có thể chuẩn bị",
            "LATE": "Đang ở đoạn muộn",
            "EXTREME": "Biến động quá mạnh",
            "BOUNCE_TO_SELL": "Hồi để bán",
            "DIP_TO_BUY": "Hồi để mua",
        }
        return mapping.get(label, label or "Chưa rõ")

    def state_text(state: str, narrative_obj: dict, struct: dict, htf_pressure_v4: dict) -> str:
        state = str(state or "").upper()
        summary = str((narrative_obj or {}).get("summary") or "").strip()
    
        # ===== Lấy structure =====
        m15_struct = str(struct.get("M15") or "").upper()
        h1_struct = str(struct.get("H1") or "").upper()
        htf_state = str(htf_pressure_v4.get("state") or "").upper()
    
        # ===== Xác định hướng chính =====
        is_bear = (
            "BEARISH" in htf_state
            or m15_struct in ("LL-LH", "DOWN")
            or h1_struct in ("LL-LH", "DOWN")
        )
    
        is_bull = (
            "BULLISH" in htf_state
            or m15_struct in ("HH-HL", "UP")
            or h1_struct in ("HH-HL", "UP")
        )
    
        # ===== Ưu tiên structure trước =====
        if is_bear and not is_bull:
            return "Đang hồi kỹ thuật trong xu hướng giảm; ưu tiên sell-the-rally, tránh sell đuổi."
    
        if is_bull and not is_bear:
            return "Đang điều chỉnh trong xu hướng tăng; ưu tiên buy-the-dip, tránh buy đuổi."
    
        # ===== Nếu không rõ (chuyển pha / nhiễu) =====
        mapping = {
            "CHOP": "Thị trường nhiễu, dễ quét hai đầu",
            "TRANSITION": "Thị trường đang chuyển trạng thái, chưa nên vội vào lệnh",
            "POST_LIQUIDATION_BOUNCE": "Sau cú quét mạnh, thị trường dễ hồi kỹ thuật",
            "POST_SHORT_COVER": "Sau cú ép thoát lệnh bán, thị trường dễ hồi mạnh",
            "EXHAUSTION_DOWN": "Đà giảm có dấu hiệu yếu dần",
            "EXHAUSTION_UP": "Đà tăng có dấu hiệu yếu dần",
        }
    
        # fallback cuối
        return summary or mapping.get(state, "Thị trường chưa rõ hướng ưu tiên")

    def flow_text(flow_obj: dict) -> str:
        st = str((flow_obj or {}).get("state") or "").upper()
        favored = str((flow_obj or {}).get("favored_side") or "").upper()
        mapping = {
            "INFLOW": "Dòng tiền đang vào",
            "OUTFLOW": "Dòng tiền đang ra",
            "RISK_ON": "Dòng tiền đang nghiêng về tài sản rủi ro",
            "RISK_OFF": "Dòng tiền đang rời tài sản rủi ro",
            "NEUTRAL": "Dòng tiền trung tính",
        }
        base = mapping.get(st, st or "Chưa rõ")
        if favored == "BUY":
            return f"{base} | Ưu tiên BUY"
        if favored == "SELL":
            return f"{base} | Ưu tiên SELL"
        return f"{base} | Ưu tiên NONE"

    def _final_score_now(sig, meta, struct, playbook, ntz, session_v4, htf_pressure_v4):
        score = 50
        reasons = []
    
        # ===== HTF =====
        htf_state = str(htf_pressure_v4.get("state") or "").upper()
        if "BULLISH_STRONG" in htf_state or "BEARISH_STRONG" in htf_state:
            score += 12
            reasons.append("HTF mạnh")
        elif "BULLISH_WEAK" in htf_state or "BEARISH_WEAK" in htf_state:
            score += 6
            reasons.append("HTF hơi nghiêng")
        else:
            score -= 4
            reasons.append("HTF chưa đồng thuận")
    
        # ===== Structure =====
        m15 = str((struct or {}).get("M15") or "").upper()
        h1 = str((struct or {}).get("H1") or "").upper()
    
        if m15 in ("HH-HL", "LL-LH"):
            score += 8
            reasons.append("M15 rõ cấu trúc")
        else:
            score -= 4
            reasons.append("M15 chưa rõ")
    
        if h1 in ("HH-HL", "LL-LH"):
            score += 6
            reasons.append("H1 rõ cấu trúc")
        elif h1 == "TRANSITION":
            score -= 3
            reasons.append("H1 đang chuyển pha")
    
        # ===== Position / range =====
        rp = playbook.get("range_pos")
        try:
            rp = float(rp)
            if rp <= 0.20 or rp >= 0.80:
                score += 8
                reasons.append("đang ở vùng biên")
            elif 0.30 <= rp <= 0.70:
                score -= 10
                reasons.append("đang ở giữa biên độ")
        except Exception:
            pass
    
        # ===== Session =====
        sess = str(session_v4.get("session_tag") or "").upper()
        ft = str(session_v4.get("follow_through") or "").upper()
        fake = str(session_v4.get("fake_move_risk") or "").upper()
    
        if "CHOP" in sess:
            score -= 6
            reasons.append("session nhiễu")
        if ft == "YES":
            score += 5
            reasons.append("có follow-through")
        elif ft == "NO":
            score -= 5
            reasons.append("thiếu follow-through")
    
        if fake == "HIGH":
            score -= 8
            reasons.append("fake risk cao")
        elif fake == "MEDIUM":
            score -= 3
            reasons.append("fake risk trung bình")
    
        # ===== No trade zone =====
        if ntz.get("active"):
            score -= 8
            reasons.append("đang ở no-trade zone")
    
        # ===== Entry zone =====
        has_zone = bool(playbook.get("zone_low") is not None and playbook.get("zone_high") is not None)
        if has_zone:
            score += 6
            reasons.append("có vùng entry")
        else:
            score -= 8
            reasons.append("chưa có vùng entry")
    
        # ===== Trigger gần =====
        action_text = " ".join(str(x) for x in (sig.get("action_lines") or []))
        trigger_text = " ".join(str(x) for x in (sig.get("quality_lines") or []))
        merged = f"{action_text} {trigger_text}".upper()
    
        # nếu bạn render trigger ở dưới, có thể đổi nguồn sang plan_pack trigger lines
        if "BUY GẦN" in merged or "SELL GẦN" in merged or "BUY NEAR" in merged or "SELL NEAR" in merged:
            score += 8
            reasons.append("có trigger gần")
        else:
            score -= 6
            reasons.append("chưa có trigger gần")
    
        # ===== Quality lines =====
        q_lines = sig.get("quality_lines") or []
        q_text = " | ".join(str(x) for x in q_lines).upper()
    
        if "VOLUME: HIGH" in q_text or "VOLUME CAO" in q_text:
            score += 5
            reasons.append("volume ủng hộ")
        elif "VOLUME: LOW" in q_text or "VOLUME THẤP" in q_text:
            score -= 4
            reasons.append("volume yếu")
    
        if "ENGULFING=BULL" in q_text or "ENGULFING=BEAR" in q_text or "REJECTION=UPPER" in q_text or "REJECTION=LOWER" in q_text:
            score += 4
            reasons.append("nến có phản ứng rõ")
    
        if "RSI DIVERGENCE: BEARISH" in q_text or "RSI DIVERGENCE: BULLISH" in q_text or "DIVERGENCE: BEARISH" in q_text or "DIVERGENCE: BULLISH" in q_text:
            score += 4
            reasons.append("có phân kỳ hỗ trợ")
    
        # clamp
        score = max(0, min(100, score))
    
        # tradeable logic cứng
        tradeable = True
        tradeable_reasons = []
    
        if not has_zone:
            tradeable = False
            tradeable_reasons.append("chưa có vùng entry rõ")
    
        if ntz.get("active"):
            tradeable = False
            tradeable_reasons.append("đang ở vùng no-trade")
    
        try:
            rp = float(playbook.get("range_pos"))
            if 0.30 <= rp <= 0.70:
                tradeable = False
                tradeable_reasons.append("đang ở giữa biên độ")
        except Exception:
            pass
    
        if score < 60:
            tradeable = False
            tradeable_reasons.append("edge chưa đủ mạnh")
    
        dedup = []
        seen = set()
        for r in reasons:
            if r not in seen:
                seen.add(r)
                dedup.append(r)
    
        dedup_trade = []
        seen_trade = set()
        for r in tradeable_reasons:
            if r not in seen_trade:
                seen_trade.add(r)
                dedup_trade.append(r)
    
        return score, ("YES" if tradeable else "NO"), dedup[:5], dedup_trade[:3]

    def _market_summary_line(score, tradeable_label, session_v4, htf_pressure_v4):
        htf = str(htf_pressure_v4.get("state") or "").upper()
        sess = str(session_v4.get("session_tag") or "").upper()
    
        if tradeable_label == "NO":
            if "CHOP" in sess:
                return "Môi trường đang nhiễu → chưa tradeable"
            if "BULLISH" in htf:
                return "Khung lớn nghiêng tăng nhưng hiện chưa có điểm vào đẹp"
            if "BEARISH" in htf:
                return "Khung lớn nghiêng giảm nhưng hiện chưa có điểm vào đẹp"
            return "Môi trường hiện tại chưa đủ lợi thế để vào lệnh"
    
        if score >= 75:
            return "Bối cảnh và điểm vào đang khá đồng thuận"
        if score >= 60:
            return "Có edge nhưng cần chọn điểm vào kỹ"
        return "Kèo có ý tưởng nhưng rủi ro còn cao"
    def _tradeability_engine(sig, playbook, ntz, htf_pressure_v4):
        reasons = []
        tradeable = True
    
        zlo = playbook.get("zone_low")
        zhi = playbook.get("zone_high")
    
        if not zlo or not zhi:
            tradeable = False
            reasons.append("chưa có vùng entry rõ")
    
        if ntz.get("active"):
            tradeable = False
            reasons.append("đang ở vùng no-trade")
    
        try:
            rp = float(playbook.get("range_pos"))
            if 0.3 <= rp <= 0.7:
                tradeable = False
                reasons.append("đang ở giữa biên độ")
        except:
            pass
    
        action_text = " ".join(sig.get("action_lines") or []).lower()
        if "buy gần" not in action_text and "sell gần" not in action_text:
            tradeable = False
            reasons.append("chưa có trigger gần")
    
        return "YES" if tradeable else "NO", reasons[:3]
    def _signal_weighting_now(sig: dict, meta: dict, struct: dict, playbook: dict, htf_pressure_v4: dict, ntz: dict):
        score = 0
        reasons = []
    
        # 1) HTF
        htf_state = str(htf_pressure_v4.get("state") or "").upper()
        m15_struct = str(struct.get("M15") or "").upper()
    
        if "BULLISH" in htf_state or "BEARISH" in htf_state:
            score += 1
            reasons.append("khung lớn có thiên hướng")
    
        # 2) structure ngắn hạn
        if m15_struct in ("HH-HL", "LL-LH"):
            score += 1
            reasons.append("cấu trúc ngắn hạn rõ")
    
        # 3) position
        rp = playbook.get("range_pos")
        try:
            rp = float(rp)
            if rp <= 0.2 or rp >= 0.8:
                score += 2
                reasons.append("đang ở vùng biên có lợi")
            elif 0.3 <= rp <= 0.7:
                score -= 2
                reasons.append("đang ở giữa biên độ")
        except Exception:
            pass
    
        # 4) no-trade zone
        if ntz.get("active"):
            score -= 2
            reasons.append("đang ở vùng no-trade")
    
        # 5) quality lines / chi tiết phụ
        q_lines = sig.get("quality_lines") or []
        q_text = " | ".join(str(x) for x in q_lines).upper()
    
        if "VOLUME: HIGH" in q_text or "VOLUME CAO" in q_text:
            score += 1
            reasons.append("volume ủng hộ")
        if "VOLUME: LOW" in q_text or "VOLUME THẤP" in q_text:
            score -= 1
            reasons.append("volume chưa ủng hộ")
    
        if "REJECTION=UPPER" in q_text or "REJECTION=LOWER" in q_text or "ENGULFING=BULL" in q_text or "ENGULFING=BEAR" in q_text:
            score += 1
            reasons.append("nến có phản ứng rõ")
    
        if "DIVERGENCE: BEARISH" in q_text or "DIVERGENCE: BULLISH" in q_text or "RSI DIVERGENCE: BEARISH" in q_text or "RSI DIVERGENCE: BULLISH" in q_text:
            score += 1
            reasons.append("có phân kỳ hỗ trợ")
    
        has_zone = bool(playbook.get("zone_low") and playbook.get("zone_high"))
        if score > 3 and has_zone:
            label = "MẠNH"
        elif score > 3 and not has_zone:
            label = "MẠNH (CHỜ ENTRY)"
        elif 2 <= score <= 3:
            label = "TRUNG BÌNH"
        else:
            label = "YẾU"
    
        # loại trùng
        dedup = []
        seen = set()
        for r in reasons:
            if r not in seen:
                seen.add(r)
                dedup.append(r)
        # chống mâu thuẫn: nếu đang ở vùng biên có lợi mà label vẫn YẾU thì nâng tối thiểu lên TRUNG BÌNH
        if "đang ở vùng biên có lợi" in dedup and label == "YẾU":
            label = "TRUNG BÌNH"
        return label, dedup[:4]
    def position_note_from_range(range_pos_val) -> str:
        try:
            rp = float(range_pos_val)
        except Exception:
            return "chưa xác định rõ vị trí"
        if rp >= 0.80:
            return "đang sát vùng cao"
        if rp <= 0.20:
            return "đang sát vùng thấp"
        return "đang ở giữa biên độ"

    def no_trade_reason(range_pos_val, ntz_obj: dict, state: str) -> str:
        tags = []
    
        for r in (ntz_obj.get("reasons") or []):
            rs = str(r).lower()
            if "mid" in rs and "mid" not in tags:
                tags.append("mid")
            elif ("nhiễu" in rs or "chop" in rs or "transition" in rs) and "chop" not in tags:
                tags.append("chop")
            elif "confirm" in rs and "confirm" not in tags:
                tags.append("confirm")
    
        st = str(state or "").upper()
        if st in ("CHOP", "TRANSITION") and "chop" not in tags:
            tags.append("chop")
    
        try:
            rp = float(range_pos_val)
            if 0.30 <= rp <= 0.70 and "mid" not in tags:
                tags.append("mid")
        except Exception:
            pass
    
        mapping = {
            "chop": "thị trường đang nhiễu",
            "mid": "đang ở giữa biên độ",
            "confirm": "chưa có xác nhận rõ",
        }
    
        reasons = [mapping[t] for t in tags if t in mapping]
        return "; ".join(reasons) if reasons else "chưa có xác nhận đủ mạnh"

    def trigger_lines_v2(rec_text: str, key_levels: dict, playbook_obj: dict):
        hi = key_levels.get("M15_RANGE_HIGH")
        lo = key_levels.get("M15_RANGE_LOW")
        bos = key_levels.get("M15_BOS")
    
        zone_lo = playbook_obj.get("zone_low")
        zone_hi = playbook_obj.get("zone_high")
    
        # ===== trigger gần =====
        buy_near = "Chưa có trigger BUY gần"
        sell_near = "Chưa có trigger SELL gần"
    
        if bos:
            buy_near = f"BUY gần: nếu reclaim {nf(bos)} và giữ được"
            sell_near = f"SELL gần: nếu bị từ chối dưới {nf(bos)}"
    
        if zone_lo and zone_hi:
            if rec_text == "BÁN":
                sell_near = f"SELL gần: nếu bị từ chối tại vùng {nf(zone_lo)} – {nf(zone_hi)}"
                buy_near = f"BUY sớm: nếu phá lên và giữ trên vùng {nf(zone_hi)}"
            else:
                buy_near = f"BUY gần: nếu giữ được vùng {nf(zone_lo)} – {nf(zone_hi)}"
                sell_near = f"SELL sớm: nếu thủng vùng {nf(zone_lo)}"
    
        # ===== trigger mạnh =====
        buy_strong = "Chưa có trigger BUY mạnh"
        sell_strong = "Chưa có trigger SELL mạnh"
    
        if hi:
            buy_strong = f"BUY mạnh: M15 đóng trên {nf(hi)} và giữ được"
        if lo:
            sell_strong = f"SELL mạnh: M15 đóng dưới {nf(lo)} với follow-through"
    
        return buy_near, sell_near, buy_strong, sell_strong
    def _trend_context_for_now(htf_state: str, struct_obj: dict, rec_text: str) -> str:
        m15 = str((struct_obj or {}).get("M15") or "").upper()
        h1 = str((struct_obj or {}).get("H1") or "").upper()
        htf = str(htf_state or "").upper()
    
        is_bear = ("BEARISH" in htf) or (m15 in ("LL-LH", "DOWN")) or (h1 in ("LL-LH", "DOWN"))
        is_bull = ("BULLISH" in htf) or (m15 in ("HH-HL", "UP")) or (h1 in ("HH-HL", "UP"))
    
        if is_bear and not is_bull:
            return "BEAR"
        if is_bull and not is_bear:
            return "BULL"
        return "MIXED"
    
    
    def _state_text_v7(symbol: str, trend_ctx: str, market_state: str) -> str:
        st = str(market_state or "").upper()
    
        if trend_ctx == "BEAR":
            if st in ("CHOP", "TRANSITION"):
                return f"{symbol} đang hồi kỹ thuật trong bối cảnh giảm nhưng ngắn hạn còn nhiễu; ưu tiên sell-the-rally, tránh sell đuổi."
            return f"{symbol} đang hồi kỹ thuật trong xu hướng giảm; ưu tiên sell-the-rally, tránh sell đuổi."
    
        if trend_ctx == "BULL":
            if st in ("CHOP", "TRANSITION"):
                return f"{symbol} đang điều chỉnh trong bối cảnh tăng nhưng ngắn hạn còn nhiễu; ưu tiên buy-the-dip, tránh buy đuổi."
            return f"{symbol} đang điều chỉnh trong xu hướng tăng; ưu tiên buy-the-dip, tránh buy đuổi."
    
        return f"{symbol} đang ở trạng thái nhiễu hoặc chuyển pha; edge thấp, nên đứng ngoài là chính."
    
    
    def _now_plan_and_triggers_v7(trend_ctx: str, key_levels: dict, playbook_obj: dict, last_px) -> dict:
        zone_lo = playbook_obj.get("zone_low")
        zone_hi = playbook_obj.get("zone_high")
        range_lo = key_levels.get("M15_RANGE_LOW")
        range_hi = key_levels.get("M15_RANGE_HIGH")
    
        def _f(v):
            return nf(v)
    
        out = {
            "main_plan": "Ưu tiên đứng ngoài quan sát, chưa có lợi thế rõ để mở lệnh mới",
            "alt_plan": "Chờ displacement thật + follow-through",
            "invalid_lines": [],
            "trigger_lines": [],
        }
    
        in_zone = False
        below_zone = False
        above_zone = False
    
        try:
            if zone_lo is not None and zone_hi is not None and last_px is not None:
                px = float(last_px)
                zlo = float(zone_lo)
                zhi = float(zone_hi)
                in_zone = zlo <= px <= zhi
                below_zone = px < zlo
                above_zone = px > zhi
        except Exception:
            pass
    
        if trend_ctx == "BEAR":
            if zone_lo is not None and zone_hi is not None:
                if below_zone:
                    out["main_plan"] = f"Nếu giá hồi lại vùng {_f(zone_lo)} – {_f(zone_hi)} thì mới canh SELL; hiện tại đã ở dưới vùng, không nên SELL đuổi"
                elif in_zone:
                    out["main_plan"] = f"Đang ở trong vùng {_f(zone_lo)} – {_f(zone_hi)}; chỉ canh SELL nếu xuất hiện lực từ chối rõ"
                elif above_zone:
                    out["main_plan"] = f"Chờ phản ứng tại vùng {_f(zone_lo)} – {_f(zone_hi)} rồi canh SELL nếu bị từ chối"
    
                out["trigger_lines"].append(f"SELL gần: nếu giá hồi lên vùng {_f(zone_lo)} – {_f(zone_hi)} và bị từ chối")
                out["trigger_lines"].append(f"BUY sớm: chỉ xét nếu phá lên và giữ trên {_f(zone_hi)}")
    
            if range_hi is not None:
                out["trigger_lines"].append(f"BUY mạnh: M15 đóng trên {_f(range_hi)} và giữ được")
                out["invalid_lines"].append(f"Nếu M15 vượt rõ {_f(range_hi)} và giữ được → kịch bản SELL yếu đi rõ")
            if range_lo is not None:
                out["trigger_lines"].append(f"SELL mạnh: M15 đóng dưới {_f(range_lo)} với follow-through")
                out["invalid_lines"].append(f"Nếu M15 không giữ được đà xuống dưới {_f(range_lo)} → tránh SELL đuổi")
    
        elif trend_ctx == "BULL":
            if zone_lo is not None and zone_hi is not None:
                if above_zone:
                    out["main_plan"] = f"Nếu giá điều chỉnh lại vùng {_f(zone_lo)} – {_f(zone_hi)} thì mới canh BUY; hiện tại đã ở trên vùng, không nên BUY đuổi"
                elif in_zone:
                    out["main_plan"] = f"Đang ở trong vùng {_f(zone_lo)} – {_f(zone_hi)}; chỉ canh BUY nếu giữ được và bật lên rõ"
                elif below_zone:
                    out["main_plan"] = f"Chờ phản ứng tại vùng {_f(zone_lo)} – {_f(zone_hi)} rồi canh BUY nếu giữ được"
    
                out["trigger_lines"].append(f"BUY gần: nếu giữ được vùng {_f(zone_lo)} – {_f(zone_hi)} và bật lên rõ")
                out["trigger_lines"].append(f"SELL sớm: nếu thủng vùng {_f(zone_lo)}")
    
            if range_hi is not None:
                out["trigger_lines"].append(f"BUY mạnh: M15 đóng trên {_f(range_hi)} và giữ được")
            if range_lo is not None:
                out["trigger_lines"].append(f"SELL mạnh: M15 đóng dưới {_f(range_lo)} với follow-through")
                out["invalid_lines"].append(f"Nếu M15 thủng rõ {_f(range_lo)} → kịch bản BUY yếu đi rõ")
    
        else:
            out["main_plan"] = "Ưu tiên đứng ngoài quan sát, chờ thị trường rõ hướng hơn"
            if range_hi is not None:
                out["trigger_lines"].append(f"BUY mạnh: M15 đóng trên {_f(range_hi)} và giữ được")
            if range_lo is not None:
                out["trigger_lines"].append(f"SELL mạnh: M15 đóng dưới {_f(range_lo)} với follow-through")
            out["invalid_lines"].append("Nếu thị trường thoát khỏi trạng thái nhiễu và có break rõ kèm follow-through → bắt đầu xét vào lệnh")
    
        return out

    def _htf_summary(htf_obj: dict, side: str) -> str:
        state = str((htf_obj or {}).get("state") or "")
        h1 = str((htf_obj or {}).get("h1_close_bias") or "")
        if "BEARISH" in state and h1 == "UP":
            return "Khung lớn vẫn nghiêng giảm, nhưng H1 đang hồi lên → lệnh SELL chỉ nên đánh ngắn" if side == "SELL" else "Khung lớn nghiêng giảm, không thuận cho BUY mạnh"
        if "BULLISH" in state and h1 == "DOWN":
            return "Khung lớn vẫn nghiêng tăng, nhưng H1 đang điều chỉnh xuống → lệnh BUY chỉ nên đánh ngắn" if side == "BUY" else "Khung lớn nghiêng tăng, không thuận cho SELL mạnh"
        if "BEARISH" in state:
            return "Khung lớn đang nghiêng giảm"
        if "BULLISH" in state:
            return "Khung lớn đang nghiêng tăng"
        return "Khung lớn chưa thật sự đồng thuận"
    def grade_from_mode(mode: str, stars_val: int) -> str:
        mode = str(mode or "").upper()
        if mode == "FULL":
            return "A"
        if mode == "HALF":
            return "B"
        if stars_val >= 4:
            return "A-"
        if stars_val == 3:
            return "B"
        if stars_val == 2:
            return "C"
        return "SKIP"
    def _session_htf_comment(session_obj, htf_pressure_v4):
        session = str(session_obj.get("name") or "").upper()
        htf = str(htf_pressure_v4.get("state") or "").upper()
    
        if "IMPULSE_DOWN" in session and "BULLISH" in htf:
            return "Impulse giảm ngược xu hướng lớn → dễ là pullback, cần thận trọng"
        if "IMPULSE_UP" in session and "BEARISH" in htf:
            return "Impulse tăng ngược xu hướng lớn → dễ là hồi kỹ thuật"
    
        return None
    def extract_gap_lines(ctxs: list[str], nts: list[str]) -> list[str]:
        out = []
        seen = set()
        for raw in list(ctxs or []) + list(nts or []):
            s = str(raw or "").strip()
            low = s.lower()
            if any(k in low for k in ["gap", "mở cửa", "biên độ đầu phiên", "đầu phiên", "mất cân bằng"]):
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out[:3]

    gap_lines = extract_gap_lines(ctx_lines, notes)

    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")

    range_lo = k.get("M15_RANGE_LOW")
    range_hi = k.get("M15_RANGE_HIGH")
    range_pos = playbook.get("range_pos")
    if range_pos is None and range_lo is not None and range_hi is not None and last_px is not None:
        lo_v = sf(range_lo)
        hi_v = sf(range_hi)
        cur_v = sf(last_px)
        if lo_v is not None and hi_v is not None and cur_v is not None and hi_v > lo_v:
            range_pos = (cur_v - lo_v) / max(1e-9, hi_v - lo_v)

    grade = str(meta.get("grade_v6") or grade_from_mode(trade_mode, stars))
    ez_v6 = meta.get("entry_zone_v6") if isinstance(meta.get("entry_zone_v6"), dict) else {}
    sweep_grade_v6 = str(meta.get("sweep_grade_v6") or "NONE")

    verdict_quick = "Chưa đẹp để vào ngay"
    reason_text = no_trade_reason(range_pos, ntz, meta.get("market_state_v2"))

    if trade_mode == "FULL":
        verdict_quick = "Có thể theo kịch bản chính"
    elif trade_mode == "HALF":
        verdict_quick = "Có thể canh nhưng chưa nên quá quyết liệt"
    elif ntz.get("active"):
        verdict_quick = "Ưu tiên đứng ngoài"

    if trade_mode == "WAIT":
        verdict_quick = f"{verdict_quick}; {reason_text}"

    lines: List[str] = []
    head = f"📌 {symbol} NOW"
    if tf:
        head += f" | {tf}"
    if session:
        head += f" | {session}"
    add(lines, head)
    if data_source:
        add(lines, f"📡 Dữ liệu: {data_source}")

    add(lines, "")   
    # ===== Giá hiện tại: fallback nhiều lớp =====
    last_px = k.get("M15_LAST")
    if last_px is None:
        last_px = sig.get("last_price")
    if last_px is None:
        last_px = sig.get("current_price")
    if last_px is None:
        last_px = sig.get("entry")
    price_now = nf(last_px)
    # ===== Kết luận nhanh: lý do ngắn gọn, không lặp =====
    reason = ""
    if range_pos is not None:
        try:
            rp = float(range_pos)
            if rp > 0.8:
                reason = "đang sát vùng cao, không nên BUY đuổi"
            elif rp < 0.2:
                reason = "đang sát vùng thấp, không nên SELL đuổi"
            else:
                reason = "đang ở giữa biên độ, dễ nhiễu"
        except Exception:
            reason = ""
    
    verdict_full = verdict_quick
    if reason and reason not in verdict_quick:
        verdict_full += f"; {reason}"
    trend_ctx = _trend_context_for_now(str(htf_pressure_v4.get("state") or ""), struct, rec)
    state_text_final = _state_text_v7(symbol, trend_ctx, meta.get("market_state_v2"))
    
    # chỉ override rec khi đang CHỜ, không đè mạnh lên engine hiện có
    if rec == "CHỜ":
        if trend_ctx == "BEAR":
            rec = "CHỜ"
        elif trend_ctx == "BULL":
            rec = "CHỜ"
    
    reason_text = no_trade_reason(range_pos, ntz, meta.get("market_state_v2"))
    verdict_quick = "Chưa đẹp để vào ngay"
    if ntz.get("active") or trade_mode == "WAIT":
        verdict_quick = "Ưu tiên đứng ngoài"
    verdict_full = f"{verdict_quick}; {reason_text}" if reason_text else verdict_quick
    add(lines, f"💵 Giá hiện tại: {price_now}")
    add(lines, f"📌 Kết luận nhanh: {verdict_full}")
    add(lines, f"🧭 Hướng ưu tiên: {rec}")
    if phase:
        add(lines, f"🪜 Giai đoạn: {phase.get('phase', 'n/a')} | {phase_text(phase)}")
    add(lines, f"🌡 Trạng thái: {state_text(meta.get('market_state_v2'), narrative, struct, htf_pressure_v4)}")
    if flow:
        add(lines, f"💰 Dòng tiền: {flow_text(flow)}")

    add(lines, "")
    add(lines, "📍 Vị trí giá:")
    if last_px is not None:
        add(lines, f"- Giá hiện tại: {nf(last_px)}")
    
    if range_lo is not None and range_hi is not None:
        add(lines, f"- Biên độ M15: {nf(range_lo)} – {nf(range_hi)}")
    
    if range_pos is not None:
        pos_pct = int(round(float(range_pos) * 100))
    
        if pos_pct >= 80:
            pos_note = "→ sát vùng cao, không nên BUY đuổi, chờ phản ứng rồi quyết định theo xu hướng"
        elif pos_pct >= 60:
            pos_note = "→ vùng cao, ưu tiên chờ tín hiệu SELL"
        elif pos_pct >= 40:
            pos_note = "→ giữa biên độ, dễ nhiễu, nên đứng ngoài"
        elif pos_pct >= 20:
            pos_note = "→ vùng thấp, ưu tiên chờ tín hiệu BUY"
        else:
            pos_note = "→ sát vùng thấp, không nên SELL đuổi, chờ phản ứng rồi quyết định theo xu hướng"
    
        add(lines, f"- Vị trí trong biên độ: ~{pos_pct}% {pos_note}")

    add(lines, "")
    add(lines, "💧 Thanh khoản:")
    if liq_lines:
        for s in liq_lines[:4]:
            add(lines, f"- {str(s).replace(chr(10), ' ').strip()}")
    else:
        add(lines, "- Chưa thấy vùng quét thanh khoản rõ")
    if liq_evt.get("ok"):
        add(lines, f"- Vừa có quét mạnh: {liq_evt.get('side')} | {liq_evt.get('kind')}")
    if sweep_grade_v6 and sweep_grade_v6 != "NONE":
        add(lines, f"- Độ mạnh sweep hiện tại: {sweep_grade_v6}")

    add(lines, "")
    add(lines, "✅ Xác nhận:")
    if struct:
        add(lines, f"- Cấu trúc lớn: H4 {struct.get('H4', 'n/a')} | H1 {struct.get('H1', 'n/a')}")
        add(lines, f"- Cấu trúc ngắn hạn: M15 {struct.get('M15', 'n/a')}")
    bos = k.get("M15_BOS")
    if bos is not None:
        add(lines, f"- Mốc xác nhận gần: {nf(bos)}")
    if close_confirm_v4 and close_confirm_v4.get("strength") not in (None, "N/A"):
        hold_txt = "đã giữ được mốc" if close_confirm_v4.get("hold") == "YES" else "chưa giữ mốc rõ"
        add(lines, f"- Đóng nến xác nhận: {close_confirm_v4.get('strength')} | {hold_txt}")
    if k.get("M15_BOS"):
        add(lines, f"- Mốc xác nhận gần: {nf(k.get('M15_BOS'))}")
    add(lines, "")
    add(lines, "🕳 GAP:")
    if gap_lines:
        for s in gap_lines:
            add(lines, f"- {s}")
    else:
        add(lines, "- Chưa có dấu hiệu GAP / mở cửa bất thường rõ")
    buy_near, sell_near, buy_strong, sell_strong = trigger_lines_v2(rec, k, playbook)
    plan_pack = _now_plan_and_triggers_v7(trend_ctx, k, playbook, last_px)
    add(lines, "")
    add(lines, "🎯 Kịch bản chính:")
    add(lines, f"- {plan_pack['main_plan']}")
    
    add(lines, "")
    add(lines, "🪄 Kịch bản phụ:")
    add(lines, f"- {plan_pack['alt_plan']}")
    
    add(lines, "")
    add(lines, "🧯 Điểm sai kịch bản:")
    if plan_pack["invalid_lines"]:
        for s in plan_pack["invalid_lines"]:
            add(lines, f"- {s}")
    else:
        add(lines, "- Nếu cấu trúc hiện tại bị phá thì bỏ kịch bản")
    
    add(lines, "")
    add(lines, "🧯 Trigger quan trọng:")
    add(lines, f"- {buy_near}")
    add(lines, f"- {sell_near}")
    add(lines, f"- {buy_strong}")
    add(lines, f"- {sell_strong}")

    add(lines, "")
    add(lines, f"📊 Chất lượng cơ hội: {grade}")
    
    final_score, tradeable_label, score_reasons, tradeable_reasons = _final_score_now(
        sig, meta, struct, playbook, ntz, session_v4, htf_pressure_v4
    )
    
    add(lines, f"🔥 Final Score: {final_score}/100")
    add(lines, f"→ Tradeable: {tradeable_label}")
    
    summary_line = _market_summary_line(final_score, tradeable_label, session_v4, htf_pressure_v4)
    add(lines, f"- {summary_line}")
    
    if score_reasons:
        add(lines, f"- Điểm cộng/trừ chính: {', '.join(score_reasons)}")
    if tradeable_reasons:
        add(lines, f"- Lý do chưa trade: {', '.join(tradeable_reasons)}")
    
    if playbook_v4.get("quality"):
        add(lines, f"- Độ sạch theo playbook: {playbook_v4.get('quality')}")
    if ntz.get("active"):
        rs = "; ".join(str(x) for x in (ntz.get("reasons") or []) if x)
        add(lines, f"- Cảnh báo: {rs or 'đang là vùng nên đứng ngoài'}")
    
    # Grade explanation
    if grade == "A":
        add(lines, "- Edge mạnh: có thể theo nếu giá vào đúng vùng")
    elif grade in ("A", "B", "B-"):
        add(lines, "- Có edge nhưng cần chọn điểm vào kỹ, không đuổi giá")
    elif grade == "C":
        add(lines, "- Ý tưởng có thể đúng nhưng rủi ro còn cao")
        
    add(lines, "")
    add(lines, "⚙️ Hành động:")
    if trade_mode == "FULL":
        add(lines, "- Có thể mở lệnh nếu giá vào đúng vùng và có xác nhận rõ")
    elif trade_mode == "HALF":
        add(lines, "- Có thể canh nhưng không nên đuổi giá")
    else:
        add(lines, "- Chưa nên mở lệnh mới")
    
    try:
        rp = float(range_pos)
        if 0.30 <= rp <= 0.70:
            add(lines, "- Tránh trade ở giữa biên độ")
        elif rp > 0.80:
            add(lines, "- Không BUY đuổi ở vùng cao, chờ phản ứng rồi mới quyết định theo xu hướng")
        elif rp < 0.20:
            add(lines, "- Không SELL đuổi ở vùng thấp, chờ phản ứng hồi rồi mới quyết định theo xu hướng")
    except Exception:
        pass

    if q_lines:
        add(lines, "")
        add(lines, "🧪 Chi tiết bổ sung:")
        for s in q_lines[:5]:
            add(lines, f"- {str(s).replace(chr(10), ' ').strip()}")
            
    if session_v4 or htf_pressure_v4 or macro_v4 or playbook_v4:
        add(lines, "")
        add(lines, "🧩 Toàn Cảnh Thị Trường:")
        if session_v4.get("session_tag"):
            add(lines, f"- Session: {session_v4.get('session_tag')} | Follow-through: {session_v4.get('follow_through')} | Fake risk: {session_v4.get('fake_move_risk')}")
        # FIX: define trước
        htf_pressure_v4 = sig.get("htf_pressure_v4") or {}
        if htf_pressure_v4.get("state"):
            add(lines, f"- HTF Pressure: {htf_pressure_v4.get('state')} | H1 close: {htf_pressure_v4.get('h1_close_bias')} | H4 close: {htf_pressure_v4.get('h4_close_bias')}")
            htf_state = str(htf_pressure_v4.get("state") or "").upper()
            side = str(rec).upper()
            if "BULLISH" in htf_state and side in ("SELL", "BÁN"):
                add(lines, "- ⚠️ SELL đang ngược khung lớn → chỉ nên đánh ngắn, không gồng")
            if "BEARISH" in htf_state and side in ("BUY", "MUA"):
                add(lines, "- ⚠️ BUY đang ngược khung lớn → chỉ nên đánh ngắn, không gồng")
        # session vs HTF
        comment = _session_htf_comment(session_v4 or {}, htf_pressure_v4)
        if comment:
            add(lines, f"- {comment}")
        if macro_v4.get("headline"):
            add(lines, f"- Macro: {macro_v4.get('headline')} | Bias: {macro_v4.get('bias')} | {macro_v4.get('note')}")
        if playbook_v4.get("quality"):
            trig = ", ".join(playbook_v4.get("trigger_pack") or [])
            add(lines, f"- Playbook V4: quality={playbook_v4.get('quality')}" + (f" | triggers: {trig}" if trig else ""))

    out = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    return "\n".join(out).strip()
