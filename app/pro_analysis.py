from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math
import os

from app.risk import calc_smart_sl_tp
def analyze_pro(symbol: str, m15: List[Candle], h1: List[Candle], session_name: str = "Phiên Mỹ") -> Dict[str, Any]:
    # =========================
    # Basic validation
    # =========================
    if len(m15) < 50 or len(h1) < 50:
        return {
            "symbol": symbol,
            "tf": "M15",
            "session": session_name,
            "context_lines": ["Thiếu dữ liệu nến để phân tích (cần >=50 candles mỗi TF)."],
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
            "notes": ["Hãy thử lại sau ~5–10 phút."],
            "levels": [],
        }

    # =========================
    # USE CLOSED CANDLES ONLY
    # =========================
    m15_closed = m15[:-1] if len(m15) > 1 else m15
    h1_closed  = h1[:-1] if len(h1) > 1 else h1

    last15 = m15_closed[-1]
    last_close = last15.close

    m15_closes = [c.close for c in m15_closed]
    h1_closes  = [c.close for c in h1_closed]

    ema20_h1 = _ema(h1_closes, 20)
    ema50_h1 = _ema(h1_closes, 50)
    rsi15 = _rsi(m15_closes, 14)
    atr15 = _atr(m15_closed, 14)
    if atr15 is None:
        atr15 = max(1e-6, (last15.high - last15.low))

    # =========================
    # Trend H1 + weakening
    # =========================
    h1_trend = "neutral"
    if ema20_h1 and ema50_h1:
        if ema20_h1[-1] > ema50_h1[-1]:
            h1_trend = "bullish"
        elif ema20_h1[-1] < ema50_h1[-1]:
            h1_trend = "bearish"

    weakening = False
    if ema20_h1 and ema50_h1 and len(ema20_h1) >= 6 and len(ema50_h1) >= 6:
        sep_now = ema20_h1[-1] - ema50_h1[-1]
        sep_prev = ema20_h1[-6] - ema50_h1[-6]
        if h1_trend == "bullish" and sep_now < sep_prev:
            weakening = True
        if h1_trend == "bearish" and sep_now > sep_prev:
            weakening = True

    # =========================
    # Key levels
    # =========================
    sh15 = _swing_high(m15_closed, 80)
    sl15 = _swing_low(m15_closed, 80)
    sh1  = _swing_high(h1_closed, 80)
    sl1  = _swing_low(h1_closed, 80)

    levels = []
    for v in [sh15, sl15, sh1, sl1]:
        if v is not None:
            levels.append(float(v))
    levels = sorted(list({round(x, 3) for x in levels}), reverse=True)[:6]

    # =========================
    # Market state (spike/pullback) + rejection
    # =========================
    ranges20 = [c.high - c.low for c in m15_closed[-20:]]
    ranges80 = [c.high - c.low for c in m15_closed[-80:]]
    spike = (sum(ranges20) / len(ranges20)) > 1.35 * (sum(ranges80) / len(ranges80))

    lower_highish = False
    if len(m15_closed) >= 30:
        recent_high = max(c.high for c in m15_closed[-10:])
        prev_high   = max(c.high for c in m15_closed[-30:-10])
        if recent_high <= prev_high:
            lower_highish = True

    rej = _is_rejection(last15)

    # =========================
    # Liquidity proxy
    # =========================
    liq_sell = False
    liq_buy = False
    if sh15 is not None and last15.high >= sh15 * 0.999 and rej["upper_reject"]:
        liq_sell = True
    if sl15 is not None and last15.low <= sl15 * 1.001 and rej["lower_reject"]:
        liq_buy = True

    # =========================
    # Score + text lines
    # =========================
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

    if sh15 is not None and abs(sh15 - last_close) <= atr15 * 0.8:
        position_lines.append("Giá gần đỉnh phiên")
        score += 1
    if sl15 is not None and abs(last_close - sl15) <= atr15 * 0.8:
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

    # =========================
    # Decide bias
    # =========================
    bias: Optional[str] = None

    sell_ok = (rej["upper_reject"] or liq_sell) and (lower_highish or spike or weakening) and (sh15 is not None)
    buy_ok  = (rej["lower_reject"] or liq_buy)  and (spike or weakening) and (sl15 is not None)

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
        # chờ
        return {
            "symbol": symbol,
            "tf": "M15",
            "session": session_name,
            "context_lines": context_lines,
            "position_lines": position_lines,
            "liquidity_lines": liquidity_lines,
            "quality_lines": quality_lines + ["RR ~ 1:2 (mục tiêu)"],
            "recommendation": "CHỜ",
            "stars": 1,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "lot": None,
            "notes": ["Hãy chờ thêm xác nhận nến."],
            "levels": levels,
        }

    recommendation = "🔴 SELL" if bias == "SELL" else "🟢 BUY"

    # =========================
    # Entry retest
    # =========================
    RETEST_K = float(os.getenv("RETEST_K", "0.35"))
    BUF_K    = float(os.getenv("BUF_K", "0.25"))
    TP2_R    = float(os.getenv("TP2_R", "1.6"))
    MIN_RISK_ATR = float(os.getenv("MIN_RISK_ATR", "1.2"))

    def _min_buf(sym: str) -> float:
        s = (sym or "").upper()
        if "XAU" in s:
            return 0.30
        if "BTC" in s:
            return 30.0
        return 0.0

    buf = max(BUF_K * atr15, _min_buf(symbol))

    swing_hi = sh15 if sh15 is not None else last15.high
    swing_lo = sl15 if sl15 is not None else last15.low

    if bias == "SELL":
        entry = last_close + RETEST_K * atr15

        sl_liq = max(swing_hi, last15.high) + buf
        sl_atr = entry + (MIN_RISK_ATR * atr15)
        sl = max(sl_liq, sl_atr)

        r0 = abs(sl - entry)
        tp1 = entry - 1.0 * r0
        tp2 = entry - TP2_R * r0

        notes.append("Entry RETEST: chờ giá hồi lên vùng entry rồi mới SELL.")
        notes.append("Confirm nhanh: upper-wick từ chối HOẶC phá đáy nhỏ của 3 nến gần nhất.")
        if sh15 is not None:
            notes.append(f"Không SELL nếu M15 đóng > {_fmt(sh15)}")

        liq_level = float(swing_hi)
    else:
        entry = last_close - RETEST_K * atr15

        sl_liq = min(swing_lo, last15.low) - buf
        sl_atr = entry - (MIN_RISK_ATR * atr15)
        sl = min(sl_liq, sl_atr)

        r0 = abs(entry - sl)
        tp1 = entry + 1.0 * r0
        tp2 = entry + TP2_R * r0

        notes.append("Entry RETEST: chờ giá hồi xuống vùng entry rồi mới BUY.")
        notes.append("Confirm nhanh: lower-wick từ chối HOẶC phá đỉnh nhỏ của 3 nến gần nhất.")
        if sl15 is not None:
            notes.append(f"Không BUY nếu M15 đóng < {_fmt(sl15)}")

        liq_level = float(swing_lo)

    # =========================
    # Risk engine (NO DROP) + safe override
    # =========================
    equity_usd = float(os.getenv("EQUITY_USD", os.getenv("EQUITY", "1000")))
    risk_pct   = float(os.getenv("RISK_PCT", "0.0075"))

    plan: Dict[str, Any] = {}
    try:
        # Gọi theo signature "ổn định" nhất: side/bias + liquidity_level + atr
        plan = calc_smart_sl_tp(
            symbol=symbol,
            side=bias,  # "BUY"/"SELL"
            entry=float(entry),
            atr=float(atr15),
            liquidity_level=float(liq_level) if liq_level is not None else None,
            equity_usd=float(equity_usd),
            risk_pct=float(risk_pct),
        ) or {}
    except Exception as e:
        plan = {"ok": True, "warn": f"risk_engine_error: {e}"}

    # Warn only (không return)
    if plan.get("ok") is False:
        quality_lines.append(f"⚠️ Risk warn: {plan.get('reason', 'risk check failed')}")

    if plan.get("warn"):
        quality_lines.append(f"⚠️ {plan.get('warn')}")

    # Override SL/TP nếu plan trả đủ key
    plan_sl  = plan.get("sl")
    plan_tp1 = plan.get("tp1")
    plan_tp2 = plan.get("tp2")
    if plan_sl is not None and plan_tp1 is not None and plan_tp2 is not None:
        try:
            sl  = float(plan_sl)
            tp1 = float(plan_tp1)
            tp2 = float(plan_tp2)
            quality_lines.append("✅ SL/TP theo Risk Engine")
        except Exception:
            quality_lines.append("⚠️ Risk engine trả SL/TP không parse được → dùng SL/TP mặc định (liq/ATR).")
    else:
        quality_lines.append("⚠️ Risk engine thiếu SL/TP → dùng SL/TP mặc định (liq/ATR).")

    # R an toàn (không dùng plan['r'])
    r = abs(float(entry) - float(sl))
    if r <= 0:
        r = max(1e-6, atr15 * 1.2)

    quality_lines.append("RR ~ 1:2")
    quality_lines.append(f"SL theo liquidity + buffer ~{_fmt(buf)} | RETEST {RETEST_K}*ATR | R~{r:.2f}")

    # Lot (nếu plan có)
    lot = plan.get("lot")
    try:
        lot = float(lot) if lot is not None else None
    except Exception:
        lot = None

    # =========================
    # Stars from score
    # =========================
    stars = 1
    if score >= 6:
        stars = 5
    elif score >= 5:
        stars = 4
    elif score >= 3:
        stars = 3
    elif score >= 2:
        stars = 2

    return {
        "symbol": symbol,
        "tf": "M15",
        "session": session_name,
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
        "lot": lot,
        "notes": notes,
        "levels": levels,
    }
