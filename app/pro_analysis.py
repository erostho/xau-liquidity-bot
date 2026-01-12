# app/analysis.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import statistics

@dataclass
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float

@dataclass
class Signal:
    symbol: str
    tf: str
    htf: str
    bias: str          # "BUY" | "SELL" | "WAIT"
    stars: int         # 1..5
    entry: str
    sl: str
    tp1: str
    tp2: str
    summary: str
    reasons: List[str]
    levels: List[str]
    risk_notes: List[str]


# ---------- basic indicator helpers ----------

def _ema(values: List[float], period: int) -> List[float]:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    ema = []
    prev = sum(values[:period]) / period
    ema.append(prev)
    for v in values[period:]:
        prev = v * k + prev * (1 - k)
        ema.append(prev)
    # align length to input (pad front with None conceptually)
    pad = [ema[0]] * (period - 1)
    return pad + ema

def _rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return []
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50.0] * (period)  # padding
    def rsi_from(ag, al):
        if al == 0:
            return 100.0
        rs = ag / al
        return 100 - (100 / (1 + rs))
    rsis.append(rsi_from(avg_gain, avg_loss))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsis.append(rsi_from(avg_gain, avg_loss))
    return rsis

def _atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        c = candles[i]
        p = candles[i-1]
        tr = max(c.h - c.l, abs(c.h - p.c), abs(c.l - p.c))
        trs.append(tr)
    # Wilder ATR
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr

def _swing_levels(candles: List[Candle], lookback: int = 60) -> Tuple[float, float]:
    window = candles[-lookback:] if len(candles) >= lookback else candles[:]
    hi = max(c.h for c in window)
    lo = min(c.l for c in window)
    return hi, lo

def _last(candles: List[Candle], n: int = 1) -> List[Candle]:
    return candles[-n:] if len(candles) >= n else candles[:]

def _is_bull_engulf(prev: Candle, cur: Candle) -> bool:
    return (cur.c > cur.o) and (prev.c < prev.o) and (cur.c >= prev.o) and (cur.o <= prev.c)

def _is_bear_engulf(prev: Candle, cur: Candle) -> bool:
    return (cur.c < cur.o) and (prev.c > prev.o) and (cur.o >= prev.c) and (cur.c <= prev.o)

def _wick_ratio(c: Candle) -> Tuple[float, float, float]:
    body = abs(c.c - c.o)
    upper = c.h - max(c.c, c.o)
    lower = min(c.c, c.o) - c.l
    rng = max(c.h - c.l, 1e-9)
    return upper / rng, lower / rng, body / rng

def _fmt(x: float) -> str:
    # XAU on your broker looks like 4xxx.xxx -> keep 3 decimals
    return f"{x:.3f}"


# ---------- PRO logic ----------

def analyze_pro(symbol: str, m15: List[Candle], h1: List[Candle]) -> Signal:
    closes15 = [c.c for c in m15]
    closes60 = [c.c for c in h1]

    ema20_15 = _ema(closes15, 20)
    ema50_15 = _ema(closes15, 50)
    ema20_60 = _ema(closes60, 20)
    ema50_60 = _ema(closes60, 50)

    rsi15 = _rsi(closes15, 14)
    atr15 = _atr(m15, 14)

    last15 = m15[-1]
    prev15 = m15[-2] if len(m15) >= 2 else last15

    # Levels
    hi15, lo15 = _swing_levels(m15, 80)
    hi60, lo60 = _swing_levels(h1, 80)

    # Trend bias (HTF first)
    htf_trend = "UP" if (ema20_60 and ema50_60 and ema20_60[-1] > ema50_60[-1]) else "DOWN"
    ltf_trend = "UP" if (ema20_15 and ema50_15 and ema20_15[-1] > ema50_15[-1]) else "DOWN"

    # Liquidity sweep detection (simple but effective)
    # "sweep high": current high breaks recent swing high but closes back below
    recent_hi = max(c.h for c in m15[-20:])
    recent_lo = min(c.l for c in m15[-20:])
    sweep_high = (last15.h > recent_hi) and (last15.c < recent_hi)
    sweep_low = (last15.l < recent_lo) and (last15.c > recent_lo)

    # Candle quality
    up_w, lo_w, body = _wick_ratio(last15)
    two_sided_wick = (up_w > 0.30 and lo_w > 0.30 and body < 0.35)  # indecision
    strong_bull = (last15.c > last15.o) and (body > 0.55)
    strong_bear = (last15.c < last15.o) and (body > 0.55)

    bull_engulf = _is_bull_engulf(prev15, last15)
    bear_engulf = _is_bear_engulf(prev15, last15)

    # "Structure" proxy: last 6 closes staircase
    last6 = closes15[-6:] if len(closes15) >= 6 else closes15
    lower_highs = sum(1 for i in range(1, len(last6)) if last6[i] <= last6[i-1]) >= 4
    higher_lows = sum(1 for i in range(1, len(last6)) if last6[i] >= last6[i-1]) >= 4

    # RSI context
    rsi_now = rsi15[-1] if rsi15 else 50.0
    rsi_overbought = rsi_now >= 70
    rsi_oversold = rsi_now <= 30

    # Score components (0..1 each)
    score = 0
    reasons: List[str] = []
    levels: List[str] = []
    risk_notes: List[str] = []

    # HTF alignment
    score += 1
    reasons.append(f"H1 trend: **{htf_trend}** (EMA20 vs EMA50)")
    if (htf_trend == "UP" and ltf_trend == "UP") or (htf_trend == "DOWN" and ltf_trend == "DOWN"):
        score += 1
        reasons.append("M15 đồng pha với H1 (trend alignment)")
    else:
        risk_notes.append("M15 lệch pha H1 → dễ bị giật ngược")

    # Liquidity behavior
    if sweep_high:
        score += 1
        reasons.append("M15 có dấu hiệu **sweep đỉnh** (break high rồi đóng lại dưới) → thiên về SELL")
    if sweep_low:
        score += 1
        reasons.append("M15 có dấu hiệu **sweep đáy** (break low rồi đóng lại trên) → thiên về BUY")

    # Candle confirmation
    if bear_engulf:
        score += 1
        reasons.append("Có **bearish engulfing** → lực bán xác nhận")
    if bull_engulf:
        score += 1
        reasons.append("Có **bullish engulfing** → lực mua xác nhận")

    if strong_bear:
        score += 1
        reasons.append("Nến đỏ thân lớn (momentum sell)")
    if strong_bull:
        score += 1
        reasons.append("Nến xanh thân lớn (momentum buy)")
    if two_sided_wick:
        risk_notes.append("Nến râu 2 đầu → lưỡng lự/giằng co, dễ quét SL hai đầu")

    # RSI filter
    reasons.append(f"RSI(14) M15: **{rsi_now:.1f}**")
    if rsi_overbought:
        score += 1
        reasons.append("RSI quá mua → SELL có lợi thế nếu có setup đảo chiều")
    if rsi_oversold:
        score += 1
        reasons.append("RSI quá bán → BUY có lợi thế nếu có setup đảo chiều")

    # Structure proxy
    if lower_highs:
        score += 1
        reasons.append("Giá tạo chuỗi đóng cửa yếu dần (lower-high-ish) → nghiêng SELL")
    if higher_lows:
        score += 1
        reasons.append("Giá tạo chuỗi đóng cửa mạnh dần (higher-low-ish) → nghiêng BUY")

    # Core levels
    levels.append(f"Swing High (M15 ~80): { _fmt(hi15) }")
    levels.append(f"Swing Low  (M15 ~80): { _fmt(lo15) }")
    levels.append(f"Swing High (H1  ~80): { _fmt(hi60) }")
    levels.append(f"Swing Low  (H1  ~80): { _fmt(lo60) }")

    # Decide bias
    # Rule: if sweep_high or bear signals dominate => SELL; if sweep_low or bull dominate => BUY; else WAIT.
    sell_votes = 0
    buy_votes = 0

    if sweep_high: sell_votes += 2
    if bear_engulf: sell_votes += 2
    if strong_bear: sell_votes += 1
    if lower_highs: sell_votes += 1
    if htf_trend == "DOWN": sell_votes += 1
    if rsi_overbought: sell_votes += 1

    if sweep_low: buy_votes += 2
    if bull_engulf: buy_votes += 2
    if strong_bull: buy_votes += 1
    if higher_lows: buy_votes += 1
    if htf_trend == "UP": buy_votes += 1
    if rsi_oversold: buy_votes += 1

    if abs(sell_votes - buy_votes) <= 1 and two_sided_wick:
        bias = "WAIT"
    else:
        bias = "SELL" if sell_votes > buy_votes else "BUY"

    # Stars: map score -> 1..5
    # score roughly 1..9+, clamp
    stars = int(max(1, min(5, round(score / 2))))  # 2 points ~ 1 star
    # enforce: WAIT => at most 3 stars
    if bias == "WAIT":
        stars = min(stars, 3)

    # Entry/SL/TP logic (ATR + structure)
    px = last15.c
    if atr15 <= 0:
        atr15 = max((hi15 - lo15) / 20, 1.0)

    # adaptive ATR multiple by confidence
    k_sl = 1.2 if stars <= 2 else (1.0 if stars == 3 else 0.9)
    sl_dist = atr15 * k_sl

    # Entry style
    # - If reversal setup (sweep + rejection), prefer limit at 50% of signal candle
    # - If momentum (strong body), prefer market on close
    sig = last15
    mid_sig = (sig.o + sig.c) / 2

    if bias == "SELL":
        prefer_limit = sweep_high or (two_sided_wick and rsi_overbought)
        entry_px = mid_sig if prefer_limit else px
        sl_px = max(sig.h, entry_px + sl_dist)
        # TP: TP1=1R, TP2=2R but not beyond swing low
        r = sl_px - entry_px
        tp1_px = entry_px - r * 1.0
        tp2_px = entry_px - r * 2.0
        # clamp to meaningful supports
        tp2_px = max(tp2_px, lo15)  # don’t demand too deep beyond local low
        entry = f"{_fmt(entry_px)} ({'SELL limit @ 50% nến' if prefer_limit else 'SELL market/close'})"
        sl = _fmt(sl_px)
        tp1 = _fmt(tp1_px)
        tp2 = _fmt(tp2_px)
        summary = "Ưu tiên SELL khi hồi lên vùng kháng cự/supply, tránh SELL đuổi khi nến râu dài."
        if prefer_limit:
            risk_notes.append("Nếu giá không hồi về entry limit → bỏ lệnh, tránh FOMO")
        if htf_trend == "UP":
            risk_notes.append("H1 đang UP: SELL chỉ là pullback/counter-trend → giảm lot, chốt nhanh TP1")
    elif bias == "BUY":
        prefer_limit = sweep_low or (two_sided_wick and rsi_oversold)
        entry_px = mid_sig if prefer_limit else px
        sl_px = min(sig.l, entry_px - sl_dist)
        r = entry_px - sl_px
        tp1_px = entry_px + r * 1.0
        tp2_px = entry_px + r * 2.0
        tp2_px = min(tp2_px, hi15)
        entry = f"{_fmt(entry_px)} ({'BUY limit @ 50% nến' if prefer_limit else 'BUY market/close'})"
        sl = _fmt(sl_px)
        tp1 = _fmt(tp1_px)
        tp2 = _fmt(tp2_px)
        summary = "Ưu tiên BUY ở demand/đáy quét (sweep), hạn chế BUY ngay sát đỉnh cũ."
        if prefer_limit:
            risk_notes.append("Nếu giá không hồi về entry limit → bỏ lệnh, tránh FOMO")
        if htf_trend == "DOWN":
            risk_notes.append("H1 đang DOWN: BUY chỉ là hồi kỹ thuật → giảm lot, chốt nhanh TP1")
    else:
        entry = f"{_fmt(px)} (WAIT)"
        sl = "-"
        tp1 = "-"
        tp2 = "-"
        summary = "Thị trường đang giằng co/không đủ hợp lưu. Chờ nến xác nhận (engulfing mạnh) hoặc break cấu trúc."
        risk_notes.append("WAIT ưu tiên hơn vì nến râu 2 đầu / sideway dễ quét SL")

    # Add current key zone suggestion
    levels.append(f"Giá hiện tại: {_fmt(px)} | ATR(14)~{atr15:.3f}")

    return Signal(
        symbol=symbol,
        tf="M15",
        htf="H1",
        bias=bias,
        stars=stars,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        summary=summary,
        reasons=reasons[:10],
        levels=levels[:8],
        risk_notes=risk_notes[:6],
    )


def format_signal(sig: Signal) -> str:
    stars = "⭐" * sig.stars + "☆" * (5 - sig.stars)
    head = f"📊 *{sig.symbol}* — *{sig.tf}* (HTF: {sig.htf})\n"
    bias = f"🎯 *Bias:* *{sig.bias}*   {stars}\n"
    plan = (
        f"\n*📌 Kế hoạch lệnh*\n"
        f"• Entry: `{sig.entry}`\n"
        f"• SL: `{sig.sl}`\n"
        f"• TP1: `{sig.tp1}`\n"
        f"• TP2: `{sig.tp2}`\n"
    )
    reasons = "\n*🧠 Hợp lưu chính*\n" + "\n".join([f"• {r}" for r in sig.reasons])
    lv = "\n\n*🧱 Mốc giá quan trọng*\n" + "\n".join([f"• {x}" for x in sig.levels])
    risk = "\n\n*⚠️ Lưu ý rủi ro*\n" + "\n".join([f"• {x}" for x in sig.risk_notes]) if sig.risk_notes else ""
    tail = f"\n\n*📝 Tóm tắt:* {sig.summary}"
    return head + bias + plan + reasons + lv + risk + tail
