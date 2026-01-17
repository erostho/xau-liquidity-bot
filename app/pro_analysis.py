from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math
from app.risk import calc_smart_sl_tp

# =========================
# Data model
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
    if len(values) < period:
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
    trs = []
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

    # rejection strong if wick >= 45% range and body small-ish
    return {
        "upper_reject": upper_wick / rng >= 0.45 and body / rng <= 0.45,
        "lower_reject": lower_wick / rng >= 0.45 and body / rng <= 0.45,
        "doji_like": body / rng <= 0.20,
    }

def _fmt(x: float) -> str:
    # XAU decimals tuỳ broker; cứ giữ 3 số để ổn
    return f"{x:.3f}".rstrip("0").rstrip(".")

def _stars(n: int) -> str:
    n = max(1, min(5, n))
    return "⭐️" * n + "☆" * (5 - n)


# =========================
# PRO Analyzer
# =========================
def analyze_pro(symbol: str, m15: List[Candle], h1: List[Candle], session_name: str = "Phiên Mỹ") -> Dict[str, Any]:
    # Basic validation
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
            "notes": ["Hãy thử lại sau ~5–10 phút."],
            "levels": [],
        }

    # =========================
    # USE CLOSED CANDLES ONLY
    # =========================
    # Bỏ cây đang chạy để tránh tín hiệu ảo
    m15_closed = m15[:-1] if len(m15) > 1 else m15
    h1_closed  = h1[:-1] if len(h1) > 1 else h1
    
    last15 = m15_closed[-1]          # cây M15 đã đóng
    last_close = last15.close
    
    m15_closes = [c.close for c in m15_closed]
    h1_closes  = [c.close for c in h1_closed]


    ema20_h1 = _ema(h1_closes, 20)
    ema50_h1 = _ema(h1_closes, 50)
    rsi15 = _rsi(m15_closes, 14)
    atr15 = _atr(m15_closed, 14)
    

    # H1 trend
    h1_trend = "NEUTRAL"
    if ema20_h1 and ema50_h1:
        if ema20_h1[-1] > ema50_h1[-1]:
            h1_trend = "bullish"
        elif ema20_h1[-1] < ema50_h1[-1]:
            h1_trend = "bearish"

    # Weakening (ema20 slope down or smaller separation)
    weakening = False
    if ema20_h1 and ema50_h1 and len(ema20_h1) >= 6 and len(ema50_h1) >= 6:
        sep_now = ema20_h1[-1] - ema50_h1[-1]
        sep_prev = ema20_h1[-6] - ema50_h1[-6]
        # bullish but losing momentum if separation shrinks
        if h1_trend == "bullish" and sep_now < sep_prev:
            weakening = True
        if h1_trend == "bearish" and sep_now > sep_prev:
            weakening = True

    # Key levels
    sh15 = _swing_high(m15_closed, 80)
    sl15 = _swing_low(m15_closed, 80)
    sh1  = _swing_high(h1_closed, 80)
    sl1  = _swing_low(h1_closed, 80)


    levels = []
    for v in [sh15, sl15, sh1, sl1]:
        if v is not None:
            levels.append(float(v))
    # unique-ish
    levels = sorted(list({round(x, 3) for x in levels}), reverse=True)[:6]

    # Market state: spike -> pullback heuristic
    # "Spike" if last 20 candles range bigger than last 80 avg range
    ranges20 = [c.high - c.low for c in m15_closed[-20:]]
    ranges80 = [c.high - c.low for c in m15_closed[-80:]]

    spike = sum(ranges20) / len(ranges20) > 1.35 * (sum(ranges80) / len(ranges80))

    # Pullback if last 6 closes are not making new highs and wicks appear
    lower_highish = False
    if len(m15_closed) >= 30:
        recent_high = max(c.high for c in m15_closed[-10:])
        prev_high   = max(c.high for c in m15_closed[-30:-10])
        if recent_high <= prev_high:
            lower_highish = True
    # Candle rejection
    rej = _is_rejection(last15)

    # Liquidity proxy:
    # - "SELL limit lớn phía trên" => price tapping swing high then rejecting
    # - "BUY limit lớn phía dưới" => price tapping swing low then rejecting
    liq_sell = False
    liq_buy = False
    if sh15 is not None:
        if last15.high >= sh15 * 0.999 and rej["upper_reject"]:
            liq_sell = True
    if sl15 is not None:
        if last15.low <= sl15 * 1.001 and rej["lower_reject"]:
            liq_buy = True

    # Recommendation logic
    # Score factors
    score = 0
    context_lines = []
    position_lines = []
    liquidity_lines = []
    quality_lines = []
    notes = []

    # Context
    if spike:
        context_lines.append("Thị trường: SPIKE → HỒI")
        score += 1
    else:
        context_lines.append("Thị trường: SIDEWAY / HỒI NHẸ")
    if h1_trend == "bullish":
        if weakening:
            context_lines.append("H1: bullish nhưng lực suy yếu")
            score += 1
        else:
            context_lines.append("H1: bullish (EMA20 > EMA50)")
            score += 1
    elif h1_trend == "bearish":
        if weakening:
            context_lines.append("H1: bearish nhưng lực suy yếu")
            score += 1
        else:
            context_lines.append("H1: bearish (EMA20 < EMA50)")
            score += 1
    else:
        context_lines.append("H1: neutral")

    # Position
    if sh15 is not None:
        dist_to_high = (sh15 - last_close)
        if abs(dist_to_high) <= (atr15 or 0) * 0.8:
            position_lines.append("Giá gần đỉnh phiên")
            score += 1
    if sl15 is not None:
        dist_to_low = (last_close - sl15)
        if abs(dist_to_low) <= (atr15 or 0) * 0.8:
            position_lines.append("Giá gần đáy phiên")
            score += 1

    # Liquidity lines (proxy)
    if liq_sell:
        liquidity_lines.append("🔴 Dấu hiệu SELL limit lớn phía trên (proxy: sweep + rejection)")
        score += 1
    if liq_buy:
        liquidity_lines.append("🟢 Dấu hiệu BUY limit lớn phía dưới (proxy: sweep + rejection)")
        score += 1
    if not liquidity_lines:
        liquidity_lines.append("Chưa thấy sweep/rejection rõ (liquidity proxy).")

    # Quality
    if rej["upper_reject"] or rej["lower_reject"]:
        quality_lines.append("Nến từ chối rõ")
        score += 1
    if rsi15 is not None:
        quality_lines.append(f"RSI(14) M15: {_fmt(rsi15)}")
        # Overbought + near high => favor sell; Oversold + near low => favor buy
    if atr15 is not None:
        quality_lines.append(f"ATR(14) M15: ~{_fmt(atr15)}")
        score += 1

    # Decide bias
    recommendation = "CHỜ"
    bias = None

    # SELL setup: near high + upper rejection + bullish weakening or spike pullback
    sell_ok = (
        (rej["upper_reject"] or liq_sell)
        and (lower_highish or spike or weakening)
        and (sh15 is not None)
    )
    # BUY setup: near low + lower rejection + bearish weakening or spike pullback
    buy_ok = (
        (rej["lower_reject"] or liq_buy)
        and (spike or weakening)
        and (sl15 is not None)
    )

    # Tie-breaker using RSI
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
        recommendation = "CHỜ"
        notes.append("Bỏ kèo nếu giá đi ngang thêm.")
        stars = 1
        return {
            "symbol": symbol,
            "tf": "M15",
            "session": session_name,
            "context_lines": context_lines,
            "position_lines": position_lines,
            "liquidity_lines": liquidity_lines,
            "quality_lines": quality_lines + ["RR ~ 1:2 (mục tiêu)"],
            "recommendation": recommendation,
            "stars": stars,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "notes": notes or ["Hãy chờ thêm xác nhận nến."],
            "levels": levels,
        }

    recommendation = "🔴 SELL" if bias == "SELL" else "🟢 BUY"

    # Entry RETEST (giữ logic vào lệnh của mày)
    # =========================
    if atr15 is None:
        atr15 = max(1e-6, (last15.high - last15.low))

    RETEST_K = 0.35  # 0.25..0.55 (mục tiêu 5-10 kèo/ngày)

    if bias == "SELL":
        entry = last_close + RETEST_K * atr15
        liq_level = sh15  # SELL dùng swing high làm liquidity line
        notes.append("Entry RETEST: chờ giá hồi lên vùng entry rồi mới SELL.")
        notes.append("Confirm nhanh: upper-wick từ chối HOẶC phá đáy nhỏ của 3 nến gần nhất.")
        if sh15 is not None:
            notes.append(f"Không SELL nếu M15 đóng > {_fmt(sh15)}")
    else:
        entry = last_close - RETEST_K * atr15
        liq_level = sl15  # BUY dùng swing low làm liquidity line
        notes.append("Entry RETEST: chờ giá hồi xuống vùng entry rồi mới BUY.")
        notes.append("Confirm nhanh: lower-wick từ chối HOẶC phá đỉnh nhỏ của 3 nến gần nhất.")
        if sl15 is not None:
            notes.append(f"Không BUY nếu M15 đóng < {_fmt(sl15)}")

    # =========================
    # SMART SL/TP (SL = MIN(Liq, ATR, Risk))
    # =========================
    import os
    equity_usd = float(os.getenv("EQUITY_USD", "1000"))
    risk_pct   = float(os.getenv("RISK_PCT", "0.0075"))  # 0.005..0.01

    plan = calc_smart_sl_tp(
        symbol=symbol,
        side=bias,  # "SELL"/"BUY"
        entry=float(entry),
        atr=float(atr15),
        liquidity_level=float(liq_level) if liq_level is not None else None,
        equity_usd=equity_usd,
        risk_pct=risk_pct,
        # atr_k=1.0, max_atr_k=1.25, buf_atr_k=0.25,
        # contract_size=100.0
    )
    # Nếu risk engine báo không ok
    # =========================
    # RISK ENGINE (KHÔNG BỎ KÈO) + CLAMP
    # =========================
    try:
        plan = calc_smart_sl_tp(
            symbol=symbol,
            entry=float(entry),
            bias=bias,                 # "BUY" / "SELL"
            atr=float(atr15 or 0.0),
            swing_hi=float(swing_hi),
            swing_lo=float(swing_lo),
            equity=float(os.getenv("EQUITY", "0") or 0),     # nếu chưa dùng equity thì để 0
            risk_pct=float(os.getenv("RISK_PCT", "0.0075")), # 0.5–1% => 0.005..0.01
            max_sl_atr=float(os.getenv("MAX_SL_ATR", "2.2")),# clamp SL theo ATR
        )
    except Exception as _e:
        plan = {"ok": True, "warn": f"risk_engine_error: {_e}"}
    
    # Nếu risk engine báo không ok -> KHÔNG return nữa, chỉ cảnh báo
    if not plan.get("ok", True):
        quality_lines.append(f"⚠️ Risk warn: {plan.get('reason', 'risk check failed')}")
    
    # Lấy giá trị an toàn (không bao giờ crash)
    plan_sl  = plan.get("sl", None)
    plan_tp1 = plan.get("tp1", None)
    plan_tp2 = plan.get("tp2", None)
    
    if plan_sl is not None:
        sl = float(plan_sl)
    if plan_tp1 is not None:
        tp1 = float(plan_tp1)
    if plan_tp2 is not None:
        tp2 = float(plan_tp2)
    
    # Tính R an toàn, KHÔNG dùng plan['r']
    r = abs(float(entry) - float(sl)) if (entry is not None and sl is not None) else None
    
    # Nếu vẫn không có r (trường hợp cực lỗi) thì tự set tối thiểu để khỏi nổ
    if not r or r <= 0:
        # fallback: lấy theo ATR
        r = float(max(0.6, (atr15 or 0) * 1.2))  # tuỳ bạn
    
    quality_lines.append("RR ~ 1:2")
    quality_lines.append(f"SL = MIN(Liq, ATR, Risk) | R~{r:.2f}")
    if plan.get("warn"):
        quality_lines.append(f"⚠️ {plan['warn']}")
    
        # Rating stars from score
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
        "lot": float(lot),
        "notes": notes,
        "levels": levels,
    }

def format_signal(sig: Dict[str, Any]) -> str:
    symbol = sig.get("symbol", "XAUUSD")
    tf = sig.get("tf", "M15")
    session = sig.get("session", "Phiên Mỹ")

    context_lines = sig.get("context_lines", [])
    position_lines = sig.get("position_lines", [])
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

    # Format numbers
    def nf(x):
        if x is None:
            return "..."
        return f"{x:.3f}".rstrip("0").rstrip(".")

    lines: List[str] = []
    lines.append(f"📊 {symbol} | {tf} | {session}")
    lines.append("")
    lines.append("Context:")
    for s in context_lines:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("Vị trí:")
    for s in position_lines:
        lines.append(f"- {s}")
    if not position_lines:
        lines.append("- (chưa rõ vị trí đẹp)")
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
    if levels:
        # In ra tối đa 6 mốc
        for lv in levels[:6]:
            lines.append(f"- {nf(lv)}")
    else:
        lines.append("- (chưa có mốc)")

    return "\n".join(lines)
