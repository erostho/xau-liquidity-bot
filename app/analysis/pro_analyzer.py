from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from zoneinfo import ZoneInfo
from datetime import datetime

from .indicators import atr, candle_features, swing_levels, find_liquidity_pools
from .scoring import ScoreItem, summarize, stars_from_score, clamp
from ..config import TIMEZONE, MIN_RR

@dataclass
class Plan:
    action: str
    score: float
    stars: str
    context: Dict[str, Any]
    liquidity: Dict[str, Any]
    plan: Dict[str, Any]
    score_breakdown: List[Dict[str, Any]]

def _session_tag(now: datetime) -> Tuple[str, float]:
    h = now.hour + now.minute/60
    if 7 <= h < 14:
        return "Asia (nhiễu)", 0.6
    if 14 <= h < 19:
        return "Europe (tốt)", 0.85
    if 19 <= h < 23:
        return "US (mạnh)", 0.95
    return "Late (dễ trap)", 0.7

def _trend_state(df: pd.DataFrame) -> str:
    n=30
    w=df.tail(n)
    highs=w["high"].values
    lows=w["low"].values
    mid=n//2
    hh = highs[mid:].max() > highs[:mid].max()
    hl = lows[mid:].min() > lows[:mid].min()
    ll = lows[mid:].min() < lows[:mid].min()
    lh = highs[mid:].max() < highs[:mid].max()
    if hh and hl:
        return "BULL"
    if ll and lh:
        return "BEAR"
    return "RANGE"

def _trap_risk(feat: pd.DataFrame) -> Tuple[float, str]:
    last = feat.iloc[-1]
    prev = feat.iloc[-2]
    if last["two_sided_wick"] and last["body_ratio"] < 0.35:
        return 0.8, "Nến râu 2 đầu → chop, dễ quét"
    if (last["body_ratio"] > 0.6) and (prev["body_ratio"] > 0.6) and (last["dir"] != prev["dir"]):
        return 0.7, "Đổi màu nến mạnh liên tục → whipsaw"
    return 0.35, "Trap risk thấp-vừa"

def make_plan(candles: list[dict], tf: str, risk_usd: float, symbol: str) -> Plan:
    df = pd.DataFrame(candles).copy()
    df = df.rename(columns={c:c.lower() for c in df.columns})
    for c in ["open","high","low","close"]:
        df[c]=df[c].astype(float)

    feat = candle_features(df)
    feat["atr14"] = atr(feat, 14)
    last = feat.iloc[-1]
    last_close=float(last["close"])
    atr14=float(last["atr14"]) if not np.isnan(last["atr14"]) else float(last["range"])

    # Levels
    swing_res, swing_sup = swing_levels(feat, lookback=80)
    micro_res = float(feat.tail(12)["high"].max())
    micro_sup = float(feat.tail(12)["low"].min())

    trend = _trend_state(feat)
    now = datetime.now(ZoneInfo(TIMEZONE))
    session, session_weight = _session_tag(now)

    # Volatility regime
    atr_med = float(feat["atr14"].tail(120).median()) if feat["atr14"].notna().any() else atr14
    vol_ratio = (atr14 / atr_med) if atr_med>0 else 1.0
    vol_state = "HIGH" if vol_ratio>1.4 else ("LOW" if vol_ratio<0.8 else "NORMAL")

    # Chop detection (range too tight)
    range_width = (micro_res - micro_sup)
    chop = range_width < max(1e-6, atr14*0.7)
    pos_pct = (last_close - micro_sup) / max(1e-9, (micro_res - micro_sup))

    trap_risk, trap_note = _trap_risk(feat)

    # Liquidity pools (Method #5)
    pools = find_liquidity_pools(feat, atr_val=atr14, lookback=140)
    above = pools["above"]
    below = pools["below"]

    # Inferred zones (text)
    buy_zone = (micro_sup - max(1.0, 0.2*atr14), micro_sup + max(1.0, 0.2*atr14))
    sell_zone = (micro_res - max(1.0, 0.2*atr14), micro_res + max(1.0, 0.2*atr14))

    # Setup detection (simple but robust)
    # Rejection at micro_res for SELL
    rejection = (last["high"] >= micro_res - atr14*0.1) and (last["close"] < micro_res) and (last["upper_wick"] > last["range"]*0.30)
    # Breakdown for SELL
    broke_down = (last["close"] < micro_sup) and (last["dir"] == -1) and (last["body_ratio"] > 0.55)

    # Bounce/breakout for BUY
    bounce = (last["low"] <= micro_sup + atr14*0.15) and (last["dir"]==1) and (last["body_ratio"] > 0.45)
    broke_up = (last["close"] > micro_res) and (last["dir"] == 1) and (last["body_ratio"] > 0.55)

    # Decide action
    action="WAIT"
    if chop and (0.25 < pos_pct < 0.75):
        action="WAIT"
    else:
        if trend=="BEAR" and (rejection or broke_down):
            action="SELL"
        elif trend=="BULL" and (bounce or broke_up):
            action="BUY"
        else:
            # range trading only near extremes with confirmation
            if trend=="RANGE" and rejection and pos_pct>0.8:
                action="SELL"
            elif trend=="RANGE" and bounce and pos_pct<0.2:
                action="BUY"
            else:
                action="WAIT"

    # Scoring (0..100)
    items=[]
    # Position (0..20)
    if chop and (0.25 < pos_pct < 0.75):
        items.append(ScoreItem("Vị trí giá", 4, "Giữa hộp chop → NO TRADE"))
    else:
        items.append(ScoreItem("Vị trí giá", 14, "Gần biên / vùng quyết định"))
    # Session (0..10)
    items.append(ScoreItem("Phiên giao dịch", 10*session_weight, session))
    # Trend (0..15)
    items.append(ScoreItem("Xu hướng M15", 14 if trend!="RANGE" else 8, trend))
    # Volatility (0..10)
    items.append(ScoreItem("Biến động (ATR)", 9 if vol_state=="NORMAL" else 6, f"{vol_state} (ATR≈{atr14:.1f})"))
    # Trap risk (0..15)
    items.append(ScoreItem("Rủi ro trap", (1-trap_risk)*15, trap_note))
    # Setup quality (0..20)
    setup_pts=0
    setup_notes=[]
    if rejection: setup_pts += 10; setup_notes.append("Từ chối kháng cự")
    if broke_down: setup_pts += 12; setup_notes.append("Breakdown mạnh")
    if bounce: setup_pts += 10; setup_notes.append("Bounce mạnh")
    if broke_up: setup_pts += 12; setup_notes.append("Breakout mạnh")
    setup_pts=min(20, setup_pts)
    items.append(ScoreItem("Chất lượng setup", setup_pts, "; ".join(setup_notes) if setup_notes else "Chưa rõ"))

    # Liquidity target quality (0..10)
    # If we have clear pools in direction, score higher
    liq_pts=4
    liq_note="Pool chưa rõ"
    if action=="SELL" and len(below)>=1:
        liq_pts=9 if below[0][1]>=2 else 7
        liq_note=f"Có sell-side liquidity dưới: ~{below[0][0]:.1f} (strength {below[0][1]})"
    if action=="BUY" and len(above)>=1:
        liq_pts=9 if above[0][1]>=2 else 7
        liq_note=f"Có buy-side liquidity trên: ~{above[0][0]:.1f} (strength {above[0][1]})"
    items.append(ScoreItem("Liquidity target", liq_pts, liq_note))

    score, breakdown = summarize(items)
    stars = stars_from_score(score)

    # SL distance: structure + ATR breathing, but cap by risk_usd (price dollars)
    sl_dist = min(risk_usd, max(4.0, atr14*0.9))

    plan = {"entry_zone": None, "sl": None, "tp1": None, "tp2": None, "rr": None, "time_stop": "3 nến M15 không đi đúng hướng → thoát"}
    notes=[]

    if action in {"SELL","BUY"}:
        if action=="SELL":
            entry_zone = (max(micro_res - atr14*0.35, last_close-1.0), micro_res + atr14*0.15)
            sl = entry_zone[1] + sl_dist

            # TP by liquidity pools (Method #5)
            tp1 = below[0][0] if len(below)>=1 else (last_close - 1.3*sl_dist)
            tp2 = below[1][0] if len(below)>=2 else (last_close - 2.0*sl_dist)

            # Put TP slightly before the pool to increase fill probability
            tp1_adj = tp1 + min(0.6, 0.08*atr14)
            tp2_adj = tp2 + min(0.6, 0.08*atr14)

            rr = (entry_zone[0] - tp1_adj) / max(1e-9, (sl - entry_zone[1]))
            plan.update({"entry_zone": entry_zone, "sl": sl, "tp1": tp1_adj, "tp2": tp2_adj, "rr": rr})
            if rr < MIN_RR:
                action="WAIT"
                notes.append(f"RR thấp ({rr:.2f}) < {MIN_RR} → bỏ kèo")
        else:
            entry_zone = (micro_sup - atr14*0.15, min(micro_sup + atr14*0.35, last_close+1.0))
            sl = entry_zone[0] - sl_dist
            tp1 = above[0][0] if len(above)>=1 else (last_close + 1.3*sl_dist)
            tp2 = above[1][0] if len(above)>=2 else (last_close + 2.0*sl_dist)
            tp1_adj = tp1 - min(0.6, 0.08*atr14)
            tp2_adj = tp2 - min(0.6, 0.08*atr14)
            rr = (tp1_adj - entry_zone[1]) / max(1e-9, (entry_zone[0] - sl))
            plan.update({"entry_zone": entry_zone, "sl": sl, "tp1": tp1_adj, "tp2": tp2_adj, "rr": rr})
            if rr < MIN_RR:
                action="WAIT"
                notes.append(f"RR thấp ({rr:.2f}) < {MIN_RR} → bỏ kèo")

    # If WAIT, cap score
    if action=="WAIT":
        score = min(score, 65)
        stars = stars_from_score(score)

    context = {
        "symbol": symbol,
        "timeframe": tf,
        "last_close": last_close,
        "trend": trend,
        "session": session,
        "vol_state": vol_state,
        "atr14": round(atr14,2),
        "micro_range": (round(micro_sup,2), round(micro_res,2)),
        "chop": bool(chop),
    }

    liquidity = {
        "buy_zone": buy_zone,
        "sell_zone": sell_zone,
        "tp_pools": {
            "below": [(round(lv,2), int(st)) for lv,st in below[:3]],
            "above": [(round(lv,2), int(st)) for lv,st in above[:3]],
            "tol": round(pools["tol"],2)
        },
        "notes": notes
    }

    return Plan(action=action, score=score, stars=stars, context=context, liquidity=liquidity, plan=plan, score_breakdown=breakdown)

def format_plan(p: Plan, risk_usd: float) -> str:
    c=p.context
    lines=[]
    lines.append(f"📊 {c['symbol']} | {c['timeframe']} | Risk ${risk_usd:.0f}")
    lines.append(f"Khuyến nghị: **{p.action}**  {p.stars}  (Score {p.score:.0f}/100)")
    lines.append("")
    lines.append("🧭 Context")
    lines.append(f"- Trend M15: {c['trend']} | Phiên: {c['session']} | Vol: {c['vol_state']} | ATR14≈{c['atr14']}")
    lines.append(f"- Micro range: {c['micro_range'][0]} ↔ {c['micro_range'][1]} | Chop: {c['chop']}")
    lines.append("")
    lines.append("💧 Liquidity (Method #5: liquidity pools)")
    pools=p.liquidity['tp_pools']
    if pools["below"]:
        lines.append(f"- Sell-side pools (dưới giá): {pools['below']}")
    if pools["above"]:
        lines.append(f"- Buy-side pools (trên giá): {pools['above']}")
    lines.append(f"- Tolerance cluster≈{pools['tol']}")
    lines.append("")
    lines.append("🎯 Kế hoạch")
    if p.plan["entry_zone"] is None:
        lines.append("- WAIT: Chờ giá về biên + xác nhận (rejection/bounce) hoặc break mạnh có follow-through.")
    else:
        ez=p.plan["entry_zone"]
        lines.append(f"- Entry: {ez[0]:.1f} – {ez[1]:.1f}")
        lines.append(f"- SL: {p.plan['sl']:.1f}")
        lines.append(f"- TP1: {p.plan['tp1']:.1f} (trước liquidity pool)")
        lines.append(f"- TP2: {p.plan['tp2']:.1f} (trước liquidity pool)")
        lines.append(f"- RR (ước lượng): {p.plan['rr']:.2f}")
        lines.append(f"- Time-stop: {p.plan['time_stop']}")
    lines.append("")
    lines.append("⭐ Vì sao ra score")
    top=sorted(p.score_breakdown, key=lambda x: x["score"], reverse=True)[:5]
    for it in top:
        lines.append(f"- {it['name']}: {it['score']} — {it['note']}")
    if p.liquidity.get("notes"):
        for n in p.liquidity["notes"]:
            lines.append(f"- ⚠️ {n}")
    return "\n".join(lines)
