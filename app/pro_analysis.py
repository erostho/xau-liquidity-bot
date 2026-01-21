# app/pro_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import os

from app.risk import calc_smart_sl_tp


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

def _rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        ch = values[i] - values[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses += -ch
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def _atr(candles: List[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs: List[float] = []
    for i in range(-period, 0):
        c = candles[i]
        prev = candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
        trs.append(tr)
    return sum(trs) / period

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

def _ema(values, period: int):
    if not values:
        return None
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = (v * k) + (ema * (1 - k))
    return ema

def _atr(candles: List[Candle], period: int = 14):
    if len(candles) < period + 2:
        return None
    trs = []
    for i in range(1, len(candles)):
        c = candles[i]
        p = candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        trs.append(tr)
    # ATR = EMA(TR)
    return _ema(trs[-(period * 3):], period)  # lấy dư data cho ổn định

def _build_short_hint(
    bias: str,
    h1_trend: str,
    current_price: float,
    atr15: float,
    m15c: List[Candle],
    m30c: List[Candle],
) -> str:
    """
    GỢI Ý NGẮN HẠN (chỉ dùng 30 nến M30 gần nhất để ra vùng hồi).
    H1 chỉ confirm hướng. Không dùng swing high/low khung lớn.
    """

    # --- buffer cho XAU (default)
    # nếu atr15 có thì buffer theo rung, tối thiểu 0.15
    buffer = max(0.15, float(atr15) * 0.12) if atr15 else 0.20

    # --- higher-low proxy (M15)
    higher_low = False
    lower_high = False
    if len(m15c) >= 12:
        last5 = m15c[-6:-1]   # 5 nến đã đóng gần nhất
        prev5 = m15c[-11:-6]
        low1 = min(c.low for c in last5)
        low0 = min(c.low for c in prev5)
        high1 = max(c.high for c in last5)
        high0 = max(c.high for c in prev5)

        higher_low = low1 > low0 + buffer
        lower_high = high1 < high0 - buffer

    # --- 30 nến M30 gần nhất (đã đóng)
    m30_closed = m30c[:-1] if len(m30c) > 1 else m30c
    m30_last30 = m30_closed[-30:] if len(m30_closed) >= 30 else m30_closed

    if len(m30_last30) < 10:
        return "Chưa đủ dữ liệu M30 (cần ~30 nến) → CHỜ."

    closes = [c.close for c in m30_last30]
    ema30 = _ema(closes, 30) or closes[-1]
    atr30 = _atr(m30_last30, 14)

    # nếu ATR30 thiếu thì dùng range thay thế
    hi = max(c.high for c in m30_last30)
    lo = min(c.low for c in m30_last30)
    rng = max(hi - lo, 1e-6)
    if atr30 is None:
        atr30 = rng * 0.25

    # vùng hồi quanh EMA (pullback zone) — chỉ dựa 30 nến M30
    # bullish: canh BUY khi hồi xuống dưới EMA
    # bearish: canh SELL khi hồi lên trên EMA
    k_lo = float(os.getenv("M30_ZONE_K_LO", "0.80"))  # xa EMA hơn
    k_hi = float(os.getenv("M30_ZONE_K_HI", "0.30"))  # gần EMA hơn

    lines = ["GỢI Ý NGẮN HẠN:"]

    # dùng H1 confirm hướng
    if h1_trend == "bullish":
        z1 = ema30 - (k_lo * atr30)
        z2 = ema30 - (k_hi * atr30)
        zone_low, zone_high = (min(z1, z2), max(z1, z2))

        lines.append("- Ưu tiên BUY theo xu hướng H1.")
        lines.append(f"- Vùng quan sát BUY: {zone_low:.2f} – {zone_high:.2f} (hồi M30, 30 nến gần nhất).")
        # điều kiện “M15 higher-low + đóng trên”
        trigger = zone_low + buffer
        lines.append(f"- BUY khi M15 tạo higher-low và đóng trên {trigger:.2f}.")
        lines.append(f"- Nếu M15 đóng dưới {zone_low:.2f} → bỏ kèo, chờ cấu trúc mới.")
        # nếu muốn “đúng ý” hơn: chỉ gợi ý BUY khi bias cũng BUY
        if bias == "SELL":
            lines.append("- Lưu ý: M15 bias đang nghiêng SELL, BUY chỉ nên ưu tiên khi có dấu hiệu đảo rõ.")

    elif h1_trend == "bearish":
        z1 = ema30 + (k_hi * atr30)
        z2 = ema30 + (k_lo * atr30)
        zone_low, zone_high = (min(z1, z2), max(z1, z2))

        lines.append("- Ưu tiên SELL theo xu hướng H1.")
        lines.append(f"- Vùng quan sát SELL: {zone_low:.2f} – {zone_high:.2f} (hồi M30, 30 nến gần nhất).")
        trigger = zone_high - buffer
        lines.append(f"- SELL khi M15 tạo lower-high và đóng dưới {trigger:.2f}.")
        lines.append(f"- Nếu M15 đóng trên {zone_high:.2f} → bỏ kèo, chờ cấu trúc mới.")
        if bias == "BUY":
            lines.append("- Lưu ý: M15 bias đang nghiêng BUY, SELL chỉ nên ưu tiên khi có dấu hiệu đảo rõ.")

    else:
        lines.append("- H1 sideway → ưu tiên CHỜ (đợi phá range hoặc tín hiệu rõ hơn).")

    return "\n".join(lines)

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
def analyze_pro(symbol: str, m15: List[Candle], m30: List[Candle], h1: List[Candle], session_name: str = "Phiên Mỹ") -> Dict[str, Any]:
    # ---- default return skeleton (never crash)
    base: Dict[str, Any] = {
        "symbol": symbol,
        "tf": "M30",
        "session": session_name,
        "context_lines": [],
        "position_lines": [],
        "liquidity_lines": [],
        "quality_lines": [],
        "recommendation": "CHỜ",
        "stars": 1,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "lot": None,
        "notes": [],
        "levels": [],
        "levels_info": [],  # list[(price, label)] for Telegram
        "observation": {},  # {"buy": x, "sell": y, "buffer": b, "tf": "M15"}
    }

    # Basic validation
    if len(m15) < 50 or len(m30) < 50 or len(h1) < 50:
        base["context_lines"] = ["Thiếu dữ liệu nến để phân tích (cần >=50 candles mỗi TF: M15/M30/H1)."]
        base["notes"] = ["Hãy thử lại sau ~5–10 phút."]
        return base

    # Use CLOSED candles only
    m15c = m15[:-1] if len(m15) > 1 else m15
    m30c = m30[:-1] if len(m30) > 1 else m30
    h1c  = h1[:-1] if len(h1) > 1 else h1

    last15 = m15c[-1]
    last30 = m30c[-1]

    last_close_15 = last15.close
    last_close_30 = last30.close

    m15_closes = [c.close for c in m15c]
    h1_closes  = [c.close for c in h1c]
    # --- ALWAYS init atr15 to avoid UnboundLocalError
    atr15 = None
    try:
        atr15 = _atr(m15c, 14)
    except Exception:
        atr15 = None
    
    # fallback ATR nếu vẫn None
    if atr15 is None:
        # dùng range cây M15 vừa đóng làm ATR tạm
        last15_tmp = m15c[-1]
        atr15 = max(1e-6, last15_tmp.high - last15_tmp.low)

    ema20_h1 = _ema(h1_closes, 20)
    ema50_h1 = _ema(h1_closes, 50)
    
    # normalize: _ema có thể trả list hoặc float (do bị định nghĩa trùng)
    def _ema_last(v):
        if v is None:
            return None
        return v[-1] if isinstance(v, list) else v
    
    ema20_last = _ema_last(ema20_h1)
    ema50_last = _ema_last(ema50_h1)
    
    # --- Trend H1
    h1_trend = "NEUTRAL"
    if ema20_last is not None and ema50_last is not None:
        if ema20_last > ema50_last:
            h1_trend = "bullish"
        elif ema20_last < ema50_last:
            h1_trend = "bearish"

    weakening = False
    if isinstance(ema20_h1, list) and isinstance(ema50_h1, list) and len(ema20_h1) >= 6 and len(ema50_h1) >= 6:
        sep_now = ema20_h1[-1] - ema50_h1[-1]
        sep_prev = ema20_h1[-6] - ema50_h1[-6]
        if h1_trend == "bullish" and sep_now < sep_prev:
            weakening = True
        if h1_trend == "bearish" and sep_now > sep_prev:
            weakening = True

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

    # Observation triggers for M15 close (simple, trader-friendly)
    try:
        obs_buffer = float(os.getenv("OBS_BUFFER", "0.40"))  # default XAU
    except Exception:
        obs_buffer = 0.40
    cur = float(last_close_15)
    above = [p for p, _ in levels_unique if p > cur]
    below = [p for p, _ in levels_unique if p < cur]
    buy_level = min(above) if above else (max([p for p, _ in levels_unique]) if levels_unique else None)
    sell_level = max(below) if below else (min([p for p, _ in levels_unique]) if levels_unique else None)
    base["observation"] = {
        "tf": "M15",
        "buffer": obs_buffer,
        "buy": float(buy_level) if buy_level is not None else None,
        "sell": float(sell_level) if sell_level is not None else None,
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
            "liquidity_lines": liquidity_lines,
            "quality_lines": quality_lines + ["RR ~ 1:2 (mục tiêu)"],
            "recommendation": "CHỜ",
            "stars": 1,
            "notes": ["Chưa đủ điều kiện vào kèo. Chờ thêm nến xác nhận/retest."],
        })
        return base
        # =========================
        # GỢI Ý NGẮN HẠN (30 nến M30, H1 confirm)
        # =========================
        try:
            short_hint_text = _build_short_hint(
                bias=bias,
                h1_trend=h1_trend,
                current_price=cur,      # last_close_15
                atr15=atr15,
                m15c=m15c,
                m30c=m30c,
            )
            base["short_hint"] = [
                ln.strip() for ln in short_hint_text.split("\n") if ln.strip()
            ]
        except Exception as e:
            base["short_hint"] = [f"Lỗi tạo gợi ý ngắn hạn: {e}"]

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
