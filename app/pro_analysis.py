# app/pro_analysis.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Sequence
from datetime import datetime, timedelta
import math
import os
import re
from app.risk import calc_smart_sl_tp
from dataclasses import dataclass
import logging
#from app.news_fetcher_v1 import build_news_items_safe
from app.news_fetcher_v1 import build_news_items
from app.macro_engine_v2 import (
    build_macro_engine_v2,
    explain_tags_v1,
    explain_macro_reason_v1,
)
logger = logging.getLogger("app.pro_analysis")

# --- Safe candle access helpers (dict / dataclass / object) ---
def _dbg(msg: str): #_dbg
    try:
        logger.info(msg)
    except Exception:
        pass
def _c_val(c, key: str, default=None):
    try:
        if isinstance(c, dict):
            return c.get(key, default)
        # dataclass or plain object
        return getattr(c, key, default)
    except Exception:
        return default
def nf(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "n/a"
        
def _as_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def add(buf, s):
    if s is None:
        return
    s = str(s).strip()
    if s:
        buf.append(s)

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _build_setup_plan_v1(sig: dict, cls: str) -> dict:
    meta = (sig.get("meta") or {})
    sce1 = meta.get("signal_consistency_v1") or {}
    side = str(sce1.get("final_side") or sig.get("recommendation") or "BUY").upper()
    if side not in ("BUY", "SELL"):
        side = "BUY"

    def _as_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _fmt(x, nd=3, default="n/a"):
        try:
            if x is None:
                return default
            return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
        except Exception:
            return default

    if cls == "D":
        return {"show": False, "entry": None, "sl": None, "tp": None, "entry_status": None}

    if cls == "C":
        return {"show": True, "entry": "WAIT_TRIGGER", "sl": "n/a", "tp": "n/a", "entry_status": "WAIT_CONFIRM"}

    # ===== A/B bắt buộc có plan =====
    playbook = meta.get("playbook_v2") or {}
    fib1 = meta.get("fib_confluence_v1") or {}
    pb1 = meta.get("pullback_engine_v1") or {}
    sp = meta.get("scale_plan_v2") or {}
    cc1 = meta.get("close_confirm_v4") or {}
    tg3 = meta.get("trigger_engine_v3") or {}

    entry = _as_float(sig.get("entry"))
    sl = _as_float(sig.get("sl"))
    tp1 = _as_float(sig.get("tp1"))
    tp2 = _as_float(sig.get("tp2"))

    current_price = (
        _as_float(sig.get("current_price"))
        or _as_float(sig.get("last_price"))
        or _as_float(sig.get("price"))
    )

    m15_raw = meta.get("_m15_raw") or []
    if current_price is None and m15_raw:
        try:
            current_price = float(_c_val(m15_raw[-1], "close", 0.0) or 0.0)
        except Exception:
            current_price = None

    atr15 = _as_float(meta.get("atr15"))
    if atr15 is None or atr15 <= 0:
        try:
            atr15 = _atr(m15_raw, 14) if m15_raw else None
        except Exception:
            atr15 = None
    if atr15 is None or atr15 <= 0:
        atr15 = max(1e-9, abs(float(current_price or 1.0)) * 0.003)

    # 1) playbook zone
    if entry is None:
        zlo = _as_float(playbook.get("zone_low"))
        zhi = _as_float(playbook.get("zone_high"))
        if zlo is not None and zhi is not None:
            entry = (zlo + zhi) / 2.0

    # 2) scale plan
    if entry is None:
        orders = sp.get("orders") or []
        if orders:
            zlo = _as_float(orders[0].get("zone_lo"))
            zhi = _as_float(orders[0].get("zone_hi"))
            if zlo is not None and zhi is not None:
                entry = (zlo + zhi) / 2.0

    if sl is None:
        sl = _as_float(sp.get("invalid"))
    if tp1 is None:
        tp1 = _as_float(sp.get("tp1"))
    if tp2 is None:
        tp2 = _as_float(sp.get("tp2"))

    # 3) fib zone
    if entry is None:
        fz_lo = _as_float(fib1.get("zone_low"))
        fz_hi = _as_float(fib1.get("zone_high"))
        if fz_lo is not None and fz_hi is not None:
            entry = (fz_lo + fz_hi) / 2.0

    # 4) pullback anchors
    a_lo = _as_float(pb1.get("anchor_low"))
    a_hi = _as_float(pb1.get("anchor_high"))

    if entry is None and a_lo is not None and a_hi is not None:
        lo = min(a_lo, a_hi)
        hi = max(a_lo, a_hi)
        rng = max(1e-9, hi - lo)
        entry = hi - 0.50 * rng if side == "BUY" else lo + 0.50 * rng

    # 5) HARD FALLBACK CUỐI CÙNG
    if entry is None:
        entry = current_price

    if sl is None:
        if side == "BUY":
            sl = a_lo if a_lo is not None else (entry - atr15)
        else:
            sl = a_hi if a_hi is not None else (entry + atr15)

    if tp1 is None:
        if side == "BUY":
            tp1 = a_hi if a_hi is not None else (entry + atr15)
        else:
            tp1 = a_lo if a_lo is not None else (entry - atr15)

    if tp2 is None:
        rr = abs(entry - sl) if sl is not None else atr15
        tp2 = entry + 1.6 * rr if side == "BUY" else entry - 1.6 * rr

    cc_strength = str(cc1.get("strength") or "NO").upper()
    tg_state = str(tg3.get("state") or "WAIT").upper()

    if tg_state in ("TRIGGERED", "READY") and cc_strength in ("WEAK", "STRONG"):
        entry_status = "READY"
    elif cc_strength in ("WEAK", "STRONG"):
        entry_status = "WAIT_CONFIRM"
    else:
        entry_status = "AGGRESSIVE"

    return {
        "show": True,
        "entry": _fmt(entry),
        "sl": _fmt(sl),
        "tp": f"{_fmt(tp1)} / {_fmt(tp2)}",
        "entry_status": entry_status,
    }


def _setup_class_score_v3(sig: dict) -> tuple[str, float, list[str]]:
    """
    SETUP CLASS V4:
    - Context Grade: thị trường có đúng hướng không?
    - Entry Grade: hiện tại có điểm vào đẹp chưa?
    => Không để trend day SELL bị chấm D chỉ vì chưa có close confirm.
    """
    meta = sig.get("meta") or {}
    reasons = []
    score = 0.0

    sce1 = meta.get("signal_consistency_v1") or {}
    mm1 = meta.get("market_mode_v1") or {}
    pb1 = meta.get("pullback_engine_v1") or {}
    cc1 = meta.get("close_confirm_v4") or {}
    tg3 = meta.get("trigger_engine_v3") or {}
    pbc = meta.get("post_break_continuity_v1") or {}
    ema = meta.get("ema") or sig.get("ema") or {}
    ntz = meta.get("no_trade_zone") or {}

    final_side = str(
        sce1.get("final_side")
        or sce1.get("context_side")
        or mm1.get("side")
        or "NONE"
    ).upper()

    market_mode = str(mm1.get("mode") or "").upper()
    action_mode = str(mm1.get("action_mode") or "").upper()
    ema_trend = str(ema.get("trend") or "").upper()
    ema_align = str(ema.get("alignment") or "").upper()
    cc_strength = str(cc1.get("strength") or "NO").upper()
    tg_state = str(tg3.get("state") or "WAIT").upper()
    pbc_side = str(pbc.get("side") or "").upper()
    pbc_state = str(pbc.get("state") or "").upper()

    # =========================
    # 1) CONTEXT SCORE
    # =========================
    if final_side in ("BUY", "SELL"):
        score += 20
        reasons.append(f"context side = {final_side}")
    else:
        reasons.append("context side = NONE")

    if market_mode in ("TREND_DAY_DOWN", "TREND_DAY_UP"):
        score += 25
        reasons.append(f"market mode = {market_mode}")

    if final_side == "SELL" and ema_trend == "BEARISH" and ema_align == "YES":
        score += 15
        reasons.append("EMA ủng hộ SELL")
    elif final_side == "BUY" and ema_trend == "BULLISH" and ema_align == "YES":
        score += 15
        reasons.append("EMA ủng hộ BUY")
    elif ema_align == "YES":
        score += 5
        reasons.append("EMA có alignment nhưng chưa cùng side")

    if pbc_side == final_side and "HOLD" in pbc_state:
        score += 15
        reasons.append("post-break giữ đúng hướng")

    # Pullback: trong trend day, pullback nông không được phạt nặng
    try:
        pb_pct = float(pb1.get("pullback_pct") or 0.0)
    except Exception:
        pb_pct = 0.0

    pb_label = str(pb1.get("label") or "CHƯA RÕ")
    if pb1.get("ok"):
        if market_mode in ("TREND_DAY_DOWN", "TREND_DAY_UP") and pb_pct <= 0.25:
            score += 10
            reasons.append(f"pullback nông ({pb1.get('pullback_pct_text', 'n/a')}) = đặc điểm trend day")
        elif 0.35 <= pb_pct <= 0.70:
            score += 18
            reasons.append(f"pullback đẹp ({pb1.get('pullback_pct_text', 'n/a')})")
        else:
            score += 8
            reasons.append(f"pullback {pb_label.lower()} ({pb1.get('pullback_pct_text', 'n/a')})")

    # =========================
    # 2) ENTRY / TIMING SCORE
    # =========================
    entry_bonus = 0

    if cc_strength == "STRONG":
        entry_bonus += 20
        reasons.append("close confirm = STRONG")
    elif cc_strength == "WEAK":
        entry_bonus += 10
        reasons.append("close confirm = WEAK")
    else:
        reasons.append("entry chưa có close confirm rõ")

    if tg_state == "TRIGGERED":
        entry_bonus += 15
        reasons.append("trigger = TRIGGERED")
    elif tg_state == "READY":
        entry_bonus += 10
        reasons.append("trigger = READY")
    else:
        reasons.append("trigger chưa sẵn sàng")

    score += entry_bonus

    # No-trade zone chỉ kéo Entry, không giết Context
    if ntz.get("active"):
        score -= 8
        reasons.append("timing bị chặn bởi no-trade zone")

    score = max(0.0, min(100.0, score))

    if score >= 80:
        cls = "A"
    elif score >= 65:
        cls = "B"
    elif score >= 45:
        cls = "C"
    else:
        cls = "D"

    # Trend day có context rõ thì không cho rớt D.
    if market_mode in ("TREND_DAY_DOWN", "TREND_DAY_UP") and final_side in ("BUY", "SELL"):
        if cls == "D":
            cls = "C"
            score = max(score, 45.0)
            reasons.append("context trend day rõ → không xếp D, chỉ là entry chưa tới")

    out = []
    seen = set()
    for r in reasons:
        r = str(r).strip()
        if r and r not in seen:
            seen.add(r)
            out.append(r)

    return cls, round(score, 1), out[:6]
# ============================================================
# FINAL SCORE V2 - Context + Entry + Probe
# Analysis-only: KHÔNG auto trade, chỉ sửa điểm/output cho đồng bộ
# ============================================================

def _probe_status_v1(meta: dict) -> dict:
    pe = (meta or {}).get("probe_engine_v1") or {}

    result = str(pe.get("result") or pe.get("status") or "INACTIVE").upper()
    main_ok = bool(pe.get("main_entry_ok"))

    # Normalize SUCCESS
    success = (
        result in ("SUCCESS", "OK", "PASS")
        or main_ok is True
        or "CÓ THỂ" in str(pe.get("summary") or "").upper()
    )

    side = str(pe.get("side") or "NONE").upper()
    entry_status = str(pe.get("entry_status") or "N/A").upper()

    return {
        "raw": pe,
        "success": success,
        "result": result,
        "side": side,
        "entry_status": entry_status,
        "summary": str(pe.get("summary") or ""),
    }


def _entry_score_v2(meta: dict) -> tuple[int, str, list[str]]:
    """
    Entry Grade riêng:
    - Close confirm / trigger vẫn là chính
    - Probe SUCCESS nâng entry tối thiểu lên C
    - Không để Probe SUCCESS mà Entry Grade = D nữa
    """
    meta = meta or {}
    cc1 = meta.get("close_confirm_v4") or {}
    tg3 = meta.get("trigger_engine_v3") or {}
    ntz = meta.get("no_trade_zone") or {}
    pe = _probe_status_v1(meta)

    score = 0.0
    reasons = []

    cc_strength = str(cc1.get("strength") or "NO").upper()
    tg_state = str(tg3.get("state") or "WAIT").upper()
    tg_quality = str(tg3.get("quality") or "LOW").upper()

    if cc_strength == "STRONG":
        score += 40
        reasons.append("close confirm mạnh")
    elif cc_strength == "WEAK":
        score += 25
        reasons.append("close confirm yếu")
    else:
        reasons.append("chưa có close confirm")

    if tg_state == "TRIGGERED":
        score += 40
        reasons.append("trigger đã kích hoạt")
    elif tg_state == "READY":
        score += 25
        reasons.append("trigger gần sẵn sàng")
    else:
        reasons.append("trigger chưa sẵn sàng")

    if tg_quality == "HIGH":
        score += 10
        reasons.append("trigger quality cao")
    elif tg_quality == "MEDIUM":
        score += 5
        reasons.append("trigger quality trung bình")

    # Probe SUCCESS = có phản ứng vùng / vùng được bảo vệ
    if pe["success"]:
        score = max(score, 35)
        reasons.append("probe xác nhận vùng / có phản ứng thuận hướng")
        if pe["entry_status"] == "AGGRESSIVE":
            reasons.append("entry type = AGGRESSIVE, cần giảm size / đợi nến đóng chắc hơn")
        elif pe["entry_status"] not in ("", "N/A", "NONE"):
            reasons.append(f"entry status = {pe['entry_status']}")

    # no-trade zone chỉ kéo entry, không giết context
    if ntz.get("active"):
        score -= 10
        reasons.append("no-trade zone còn active")

    score = max(0, min(100, int(round(score))))

    if score >= 70:
        grade = "A"
    elif score >= 50:
        grade = "B"
    elif score >= 30:
        grade = "C"
    else:
        grade = "D"

    # dedupe
    out = []
    seen = set()
    for r in reasons:
        r = str(r).strip()
        if r and r not in seen:
            seen.add(r)
            out.append(r)

    return score, grade, out[:6]


def _compute_final_score_v2(sig: dict) -> dict:
    """
    Final Score mới:
    - Context Score: lấy từ Setup Class / Market Mode
    - Entry Score: close confirm + trigger + probe
    - Risk penalty: no-trade zone, SELL quá thấp / BUY quá cao
    => Không còn cảnh Context A nhưng Final Score 35 vô lý.
    """
    sig = sig or {}
    meta = sig.get("meta") or {}
    score = 0.0
    # Context score từ setup class hiện có
    try:
        context_cls, context_score, context_reasons = _setup_class_score_v3(sig)
        context_score = float(context_score or 0)
    except Exception:
        context_cls, context_score, context_reasons = "D", 0.0, []

    entry_score, entry_grade, entry_reasons = _entry_score_v2(meta)
    final_side = str(
        sig.get("final_side")
        or sig.get("side")
        or (sig.get("meta") or {}).get("context_side")
        or (sig.get("meta") or {}).get("bias_side")
        or "NONE"
    ).upper()
    mm1 = meta.get("market_mode_v1") or {}
    ntz = meta.get("no_trade_zone") or {}
    flow_filter = meta.get("smart_filter_v1") or meta.get("fvg_range_plugin_v1") or {}
    range_filter = flow_filter.get("range_filter") or {}
    mode = str(mm1.get("mode") or "").upper()
    side = str(mm1.get("side") or "NONE").upper()
    action_mode = str(mm1.get("action_mode") or "").upper()
    risk_penalty = 0
    risk_reasons = []
    macro = meta.get("macro_v2") or {}
    if macro.get("macro_mode") == "STRONG_THEME":
        if macro.get("gold_bias") == final_side or macro.get("btc_bias") == final_side:
            score += 5
        elif macro.get("gold_bias") not in ("NEUTRAL", final_side):
            score -= 7
    # no-trade zone không còn cap 35, nhưng vẫn trừ risk
    if ntz.get("active"):
        risk_penalty += 8
        risk_reasons.append("no-trade zone active")

    rf_tag = str(range_filter.get("tag") or "").upper()
    rf_state = str(range_filter.get("state") or "").upper()

    if side == "SELL" and ("SELL_TOO_LOW" in rf_tag or rf_state == "BLOCK"):
        risk_penalty += 7
        risk_reasons.append("SELL đang thấp trong range")
    elif side == "BUY" and ("BUY_TOO_HIGH" in rf_tag or rf_state == "BLOCK"):
        risk_penalty += 7
        risk_reasons.append("BUY đang cao trong range")

    # Trend day: context quan trọng hơn entry, vì entry đẹp thường không xuất hiện
    if mode in ("TREND_DAY_DOWN", "TREND_DAY_UP"):
        raw_score = context_score * 0.60 + entry_score * 0.40
    else:
        raw_score = context_score * 0.50 + entry_score * 0.50

    final_score = max(0.0, min(100.0, raw_score - risk_penalty))
    final_score = round(final_score, 1)

    # Tradeable label
    pe = _probe_status_v1(meta)
    if final_score >= 70 and entry_score >= 50 and not ntz.get("active"):
        tradeable = "YES"
    elif (
        mode in ("TREND_DAY_DOWN", "TREND_DAY_UP")
        and side in ("BUY", "SELL")
        and final_score >= 45
    ):
        tradeable = "CONDITIONAL"
    elif pe["success"] and final_score >= 45:
        tradeable = "CONDITIONAL"
    else:
        tradeable = "NO"

    if final_score >= 80:
        grade = "A"
    elif final_score >= 65:
        grade = "B"
    elif final_score >= 50:
        grade = "C"
    else:
        grade = "D"

    reasons = []
    reasons.extend(context_reasons[:3])
    reasons.extend(entry_reasons[:3])
    reasons.extend(risk_reasons[:3])

    # dedupe
    clean_reasons = []
    seen = set()
    for r in reasons:
        r = str(r).strip()
        if r and r not in seen:
            seen.add(r)
            clean_reasons.append(r)

    return {
        "context_class": context_cls,
        "context_score": round(context_score, 1),
        "entry_grade": entry_grade,
        "entry_score": entry_score,
        "final_score": final_score,
        "grade": grade,
        "tradeable": tradeable,
        "risk_penalty": risk_penalty,
        "reasons": clean_reasons[:6],
        "entry_reasons": entry_reasons,
        "risk_reasons": risk_reasons,
        "action_mode": action_mode,
        "mode": mode,
        "side": side,
    }
def _render_setup_class_block_v4(sig: dict, final_score, tradeable_label: str) -> list[str]:
    """
    Render mới:
    - Context Grade: đúng hướng / đúng market chưa
    - Entry Grade: hiện tại có bấm được chưa
    """
    meta = sig.get("meta") or {}
    mm1 = meta.get("market_mode_v1") or {}
    sce1 = meta.get("signal_consistency_v1") or {}
    cc1 = meta.get("close_confirm_v4") or {}
    tg3 = meta.get("trigger_engine_v3") or {}
    ntz = meta.get("no_trade_zone") or {}

    cls, setup_score, reasons = _setup_class_score_v3(sig)

    mode = str(mm1.get("mode") or "UNKNOWN").upper()
    side = str(mm1.get("side") or sce1.get("final_side") or "NONE").upper()
    action_mode = str(mm1.get("action_mode") or sce1.get("action_mode") or "WAIT").upper()

    cc_strength = str(cc1.get("strength") or "NO").upper()
    tg_state = str(tg3.get("state") or "WAIT").upper()


    # Entry grade riêng - lấy từ V2, có tính Probe SUCCESS
    fs2 = _compute_final_score_v2(sig)
    entry_score = int(fs2.get("entry_score") or 0)
    entry_grade = str(fs2.get("entry_grade") or "D")
    entry_reasons = list(fs2.get("entry_reasons") or [])

    if cc_strength == "STRONG":
        entry_score += 40
        entry_reasons.append("close confirm mạnh")
    elif cc_strength == "WEAK":
        entry_score += 25
        entry_reasons.append("close confirm yếu")
    else:
        entry_reasons.append("chưa có close confirm")

    if tg_state == "TRIGGERED":
        entry_score += 40
        entry_reasons.append("trigger đã kích hoạt")
    elif tg_state == "READY":
        entry_score += 25
        entry_reasons.append("trigger gần sẵn sàng")
    else:
        entry_reasons.append("trigger chưa sẵn sàng")

    if ntz.get("active"):
        entry_score -= 15
        entry_reasons.append("no-trade zone còn active")

    entry_score = max(0, min(100, entry_score))

    if entry_score >= 70:
        entry_grade = "A"
    elif entry_score >= 50:
        entry_grade = "B"
    elif entry_score >= 30:
        entry_grade = "C"
    else:
        entry_grade = "D"

    lines = []
    lines.append("")
    lines.append(f"📊 SETUP CLASS: {cls} ({setup_score:.1f}/100)")
    lines.append(f"- Context side: {side}")
    lines.append(f"- Market mode: {mode}")
    lines.append(f"- Action mode: {action_mode}")

    lines.append("Lý do context:")
    for s in reasons[:5]:
        lines.append(f"- {s}")

    lines.append("")
    lines.append(f"🎯 ENTRY GRADE NOW: {entry_grade} ({entry_score}/100)")
    for s in entry_reasons[:4]:
        lines.append(f"- {s}")

    pe = _probe_status_v1(meta)

    if mode in ("TREND_DAY_DOWN", "TREND_DAY_UP") and entry_grade in ("C", "D"):
        lines.append("📌 Diễn giải:")
        if pe["success"]:
            lines.append("- Context đúng hướng và probe đã xác nhận vùng, nhưng entry vẫn cần quản trị rủi ro.")
            lines.append("- Đây là kèo continuation/retest có điều kiện, không phải entry sạch tuyệt đối.")
        else:
            lines.append("- Context đúng hướng nhưng timing chưa đẹp.")
            lines.append("- Không phải kèo rác; đây là kèo chờ trigger continuation/retest.")

    return lines

def should_send_now_alert_v2(sig: dict) -> tuple[bool, str]:
    """
    Gửi NOW khi:
    - SETUP CLASS là A hoặc B
    - score >= 50
    """
    try:
        cls, score, _ = _setup_class_score_v3(sig)
        score = float(score or 0)
    except Exception:
        return False, "ERR"

    if cls in ("A", "B") and score >= 50:
        return True, f"{cls}/{int(score)}"

    return False, f"{cls}/{int(score)}"
    
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

def _calc_ema_pack(candles: Sequence[Any]) -> Dict[str, Any]:
    """EMA 34-89-200 pack on M15 for context/filter only."""
    if not candles or len(candles) < 210:
        return {}
    closes = [float(_c_val(c, "close", 0.0) or 0.0) for c in candles]
    e34 = _ema(closes, 34)
    e89 = _ema(closes, 89)
    e200 = _ema(closes, 200)
    if not e34 or not e89 or not e200:
        return {}
    ema34 = float(e34[-1])
    ema89 = float(e89[-1])
    ema200 = float(e200[-1])
    last = float(closes[-1])
    trend = "MIXED"
    alignment = "NO"
    if ema34 > ema89 > ema200:
        trend = "BULLISH"
        alignment = "YES"
    elif ema34 < ema89 < ema200:
        trend = "BEARISH"
        alignment = "YES"
    zone = "MIXED"
    if last > ema34 > ema89:
        zone = "TRÊN EMA34/89"
    elif ema89 < last <= ema34:
        zone = "GIỮA EMA34-89"
    elif last <= ema89:
        zone = "DƯỚI EMA89"
    return {
        "ema34": ema34,
        "ema89": ema89,
        "ema200": ema200,
        "trend": trend,
        "alignment": alignment,
        "zone": zone,
    }

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
    
def _absorption_v1(
    m15c: list | None,
    volq: dict | None,
    range_low: float | None,
    range_high: float | None,
) -> dict:
    out = {
        "active": False,
        "side": "NONE",
        "type": "NONE",
        "strength": "LOW",
        "location": "UNKNOWN",
        "reason": []
    }

    if not m15c or len(m15c) < 5:
        return out

    try:
        last = m15c[-1]

        o = float(_c_val(last, "open", 0.0) or 0.0)
        h = float(_c_val(last, "high", 0.0) or 0.0)
        l = float(_c_val(last, "low", 0.0) or 0.0)
        c = float(_c_val(last, "close", 0.0) or 0.0)

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        vol_spike = False
        vol_strength = "LOW"

        if isinstance(volq, dict):
            vol_spike = bool(volq.get("spike")) or str(volq.get("state") or "").upper() == "HIGH"
            vol_strength = str(volq.get("strength") or volq.get("state") or "LOW").upper()

        location = "MID"
        if range_low is not None and c <= float(range_low) * 1.002:
            location = "LOW"
        elif range_high is not None and c >= float(range_high) * 0.998:
            location = "HIGH"

        if vol_spike and lower_wick > max(body * 1.5, 1e-9) and c > l:
            out.update({
                "active": True,
                "side": "BUY",
                "type": "SELL_ABSORPTION",
                "strength": vol_strength,
                "location": location,
                "reason": ["volume spike", "lower wick dài", "giá không đóng sâu"],
            })
            return out

        if vol_spike and upper_wick > max(body * 1.5, 1e-9) and c < h:
            out.update({
                "active": True,
                "side": "SELL",
                "type": "BUY_ABSORPTION",
                "strength": vol_strength,
                "location": location,
                "reason": ["volume spike", "upper wick dài", "giá không giữ đỉnh"],
            })
            return out

        return out
    except Exception:
        return out
        
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
    # FIX: key_levels phải luôn được khởi tạo trước khi dùng
    # FIX: key_levels phải luôn được khởi tạo trước khi dùng
    k = k or {}
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
    ema_pack = meta.get("ema") if isinstance(meta.get("ema"), dict) else {}

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

def _fmt(x, nd=3, default="n/a"):
    try:
        if x is None:
            return default
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return default



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
def _build_fvg_range_plugin_v1(
    m15c,
    bias_side: str | None,
    range_pos,
    atr15,
    ema_pack=None,
) -> dict:
    """
    SMART ENTRY FILTER plug-in.
    Always returns a stable payload so renderer never falls back to UNKNOWN/N/A
    just because meta was attached from a different code path.
    """
    side = str(bias_side or "NONE").upper()
    ep = ema_pack if isinstance(ema_pack, dict) else {}

    # ---------- normalize range_pos ----------
    rp = None
    try:
        if range_pos is not None:
            rp = float(range_pos)
            # internal engine usually uses 0..1, renderer wants %
            if 0.0 <= rp <= 1.0:
                rp = rp * 100.0
    except Exception:
        rp = None

    range_filter = {
        "state": "UNKNOWN",
        "position": None,
        "tag": "N/A",
        "reason": ["không có range_pos"],
    }

    if rp is not None:
        if side == "BUY":
            if rp >= 90.0:
                range_filter = {
                    "state": "BLOCK",
                    "position": rp,
                    "tag": "BUY_TOO_HIGH",
                    "reason": ["⚠️ Đang sát đỉnh range → cấm BUY"],
                }
            elif rp >= 80.0:
                range_filter = {
                    "state": "WARN",
                    "position": rp,
                    "tag": "BUY_HIGH",
                    "reason": ["BUY đang ở vùng cao → ưu tiên chờ hồi/FVG"],
                }
            else:
                range_filter = {
                    "state": "OK",
                    "position": rp,
                    "tag": "IN_RANGE",
                    "reason": [],
                }
        elif side == "SELL":
            if rp <= 10.0:
                range_filter = {
                    "state": "BLOCK",
                    "position": rp,
                    "tag": "SELL_TOO_LOW",
                    "reason": ["⚠️ Đang sát đáy range → cấm SELL"],
                }
            elif rp <= 20.0:
                range_filter = {
                    "state": "WARN",
                    "position": rp,
                    "tag": "SELL_LOW",
                    "reason": ["SELL đang ở vùng thấp → ưu tiên chờ hồi/FVG"],
                }
            else:
                range_filter = {
                    "state": "OK",
                    "position": rp,
                    "tag": "IN_RANGE",
                    "reason": [],
                }
        else:
            range_filter = {
                "state": "UNKNOWN",
                "position": rp,
                "tag": "NO_SIDE",
                "reason": ["chưa có bias rõ cho range filter"],
            }

    # ---------- EMA ----------
    ema_block = {
        "trend": str(ep.get("trend") or "N/A"),
        "alignment": str(ep.get("alignment") or "NO"),
        "zone": str(ep.get("zone") or "N/A"),
        "ema34": ep.get("ema34"),
        "ema89": ep.get("ema89"),
        "ema200": ep.get("ema200"),
    }

    # ---------- FVG ----------
    fvg_payload = {
        "ok": False,
        "type": side if side in ("BUY", "SELL") else "NONE",
        "low": None,
        "high": None,
        "text": "chưa có vùng rõ",
    }
    entry = sl = tp1 = tp2 = None
    entry_mode = None
    try:
        fvg = _detect_fvg_v1(m15c=m15c, side=side, atr15=atr15)
        if isinstance(fvg, dict) and fvg.get("ok"):
            zl = _safe_float(fvg.get("zone_low"))
            zh = _safe_float(fvg.get("zone_high"))
            entry = _safe_float(fvg.get("entry"))
            sl = _safe_float(fvg.get("sl"))
            tp1 = _safe_float(fvg.get("tp1"))
            tp2 = _safe_float(fvg.get("tp2"))
            entry_mode = "FVG_LIMIT"
            fvg_payload = {
                "ok": True,
                "type": str(fvg.get("side") or side or "NONE").upper(),
                "low": zl,
                "high": zh,
                "text": f"{str(fvg.get('side') or side).upper()} FVG: {_fmt(zl)} – {_fmt(zh)}" if zl is not None and zh is not None else "FVG hợp lệ",
            }
    except Exception:
        pass

    smart_state = "NEUTRAL"
    if range_filter.get("state") == "BLOCK":
        smart_state = "BLOCK"
    elif fvg_payload.get("ok") and ema_block.get("alignment") == "YES":
        smart_state = "READY"
    elif fvg_payload.get("ok"):
        smart_state = "WAIT"

    return {
        "range_filter": range_filter,
        "ema": ema_block,
        "fvg": fvg_payload,
        "smart_state": smart_state,
        # keep plan fields for setup-plan fallback
        "entry_mode": entry_mode,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
    }

def _pf_zone_tuple(z):
    try:
        if isinstance(z, (list, tuple)) and len(z) == 2:
            a = float(z[0])
            b = float(z[1])
            return (min(a, b), max(a, b))
    except Exception:
        pass
    return None

def _path_forecast_v1(
    current_price: float | None,
    atr15: float | None,
    h1_trend: str | None,
    h4_trend: str | None,
    m15_struct_tag: str | None,
    range_low: float | None,
    range_high: float | None,
    playbook_v2: dict | None,
    liquidity_map_v1: dict | None,
    ema_pack: dict | None,
    smart_filter_v1: dict | None,
    m15c,
) -> dict:
    out = {
        "down_bias": "KHÔNG RÕ",
        "up_bias": "KHÔNG RÕ",
        "sideway_bars": "2-4",
        "res_near": None,
        "res_far": None,
        "sup_near": None,
        "sup_far": None,
        "priority_action": "ƯU TIÊN ĐỨNG NGOÀI / CHỜ NẾN M15 RÕ",
        "action_note": "",
        "reason": [],
    }

    playbook_v2 = playbook_v2 or {}
    liquidity_map_v1 = liquidity_map_v1 or {}
    ema_pack = ema_pack or {}
    smart_filter_v1 = smart_filter_v1 or {}
    m15c = m15c or []

    sf_range = smart_filter_v1.get("range_filter") or {}
    sf_state = str(sf_range.get("state") or smart_filter_v1.get("smart_state") or "").upper()
    sf_tag = str(sf_range.get("tag") or "").upper()

    cp = _safe_float(current_price)
    a = _safe_float(atr15)

    if cp is None:
        return out
    if a is None or a <= 0:
        a = max(1e-9, abs(cp) * 0.003)

    tag = str(m15_struct_tag or "").upper()
    h1 = str(h1_trend or "").lower()
    h4 = str(h4_trend or "").lower()
    ema_trend = str(ema_pack.get("trend") or "MIXED").upper()
    ema_zone = str(ema_pack.get("zone") or "")

    try:
        rlo = float(range_low) if range_low is not None else None
        rhi = float(range_high) if range_high is not None else None
    except Exception:
        rlo, rhi = None, None

    down_score = 0
    up_score = 0
    reasons = []

    # ===== xu hướng lớn =====
    if h1 == "bearish":
        down_score += 2
        reasons.append("H1 giảm")
    elif h1 == "bullish":
        up_score += 2
        reasons.append("H1 tăng")

    if h4 == "bearish":
        down_score += 1
    elif h4 == "bullish":
        up_score += 1

    # ===== structure M15 =====
    if "LH" in tag or "LL" in tag:
        down_score += 2
        reasons.append("M15 yếu")
    elif "HL" in tag or "HH" in tag:
        up_score += 2
        reasons.append("M15 khỏe")

    # ===== EMA =====
    if ema_trend == "BEARISH":
        down_score += 1
    elif ema_trend == "BULLISH":
        up_score += 1

    if "DƯỚI EMA89" in ema_zone:
        down_score += 1
    elif "TRÊN EMA34/89" in ema_zone:
        up_score += 1
    elif "GIỮA EMA34-89" in ema_zone or "GIỮA EMA34-89" in ema_zone.replace("/", "-"):
        up_score += 0.5
        down_score += 0.5

    # ===== liquidity =====
    def _pf_zone_tuple(z):
        try:
            if isinstance(z, (list, tuple)) and len(z) == 2:
                a1 = float(z[0])
                b1 = float(z[1])
                return (min(a1, b1), max(a1, b1))
        except Exception:
            pass
        return None

    above_zone = _pf_zone_tuple(liquidity_map_v1.get("above_zone"))
    below_zone = _pf_zone_tuple(liquidity_map_v1.get("below_zone"))

    above_strength = str(liquidity_map_v1.get("above_strength") or "LOW").upper()
    below_strength = str(liquidity_map_v1.get("below_strength") or "LOW").upper()

    if above_strength in ("MEDIUM", "HIGH"):
        down_score += 1
        reasons.append("phía trên có liquidity")
    if below_strength in ("MEDIUM", "HIGH"):
        up_score += 1
        reasons.append("phía dưới có liquidity")

    sweep_bias = str(liquidity_map_v1.get("sweep_bias") or "").upper()
    if "UP → DOWN" in sweep_bias:
        down_score += 2
        reasons.append("dễ quét lên rồi đạp")
    elif "DOWN → UP" in sweep_bias:
        up_score += 2
        reasons.append("dễ quét xuống rồi kéo")

    # ===== sideway bars =====
    try:
        if m15c and len(m15c) >= 20:
            closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
            recent = closed[-20:]
            highs = [float(_c_val(x, "high", 0.0) or 0.0) for x in recent]
            lows = [float(_c_val(x, "low", 0.0) or 0.0) for x in recent]
            rng20 = max(highs) - min(lows)

            if rng20 <= 2.2 * a:
                out["sideway_bars"] = "4-8"
            elif rng20 <= 3.2 * a:
                out["sideway_bars"] = "3-6"
            else:
                out["sideway_bars"] = "1-3"

            if abs(down_score - up_score) <= 1:
                out["sideway_bars"] = "4-8"
    except Exception:
        pass

    # ===== build zones =====
    zones_res = []
    zones_sup = []

    # playbook zone
    pz_lo = _safe_float(playbook_v2.get("zone_low"))
    pz_hi = _safe_float(playbook_v2.get("zone_high"))
    if pz_lo is not None and pz_hi is not None:
        z = (min(pz_lo, pz_hi), max(pz_lo, pz_hi))
        mid = (z[0] + z[1]) / 2.0
        if mid >= cp:
            zones_res.append(z)
        else:
            zones_sup.append(z)

    # liquidity zones
    if above_zone:
        zones_res.append(above_zone)
    if below_zone:
        zones_sup.append(below_zone)

    # range fallback
    if rhi is not None:
        zones_res.append((rhi - 0.15 * a, rhi + 0.15 * a))
    if rlo is not None:
        zones_sup.append((rlo - 0.15 * a, rlo + 0.15 * a))

    # swing fallback
    try:
        if m15c and len(m15c) >= 12:
            closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
            recent = closed[-12:]
            sh = max(float(_c_val(x, "high", 0.0) or 0.0) for x in recent)
            sl = min(float(_c_val(x, "low", 0.0) or 0.0) for x in recent)
            zones_res.append((sh - 0.12 * a, sh + 0.12 * a))
            zones_sup.append((sl - 0.12 * a, sl + 0.12 * a))
    except Exception:
        pass

    def _dedupe(zs):
        outz = []
        for z in zs:
            if not z:
                continue
            lo, hi = float(z[0]), float(z[1])
            found = False
            for e in outz:
                if abs(lo - e[0]) <= 0.10 * a and abs(hi - e[1]) <= 0.10 * a:
                    found = True
                    break
            if not found:
                outz.append((lo, hi))
        return outz

    zones_res = _dedupe(zones_res)
    zones_sup = _dedupe(zones_sup)

    def _dist(z):
        if not z:
            return 999999999.0
        lo, hi = z
        if lo <= cp <= hi:
            return 0.0
        return min(abs(cp - lo), abs(cp - hi))

    zones_res = sorted(zones_res, key=_dist)
    zones_sup = sorted(zones_sup, key=_dist)

    out["res_near"] = zones_res[0] if len(zones_res) >= 1 else None
    out["res_far"] = zones_res[1] if len(zones_res) >= 2 else None
    out["sup_near"] = zones_sup[0] if len(zones_sup) >= 1 else None
    out["sup_far"] = zones_sup[1] if len(zones_sup) >= 2 else None

    # ===== bias text =====
    if down_score >= up_score + 2:
        out["down_bias"] = "NGHIÊNG"
    elif down_score > up_score:
        out["down_bias"] = "CÓ THỂ"

    if up_score >= down_score + 2:
        out["up_bias"] = "NGHIÊNG"
    elif up_score > down_score:
        out["up_bias"] = "CÓ THỂ"


    # ===== ACTION ENGINE V2: sync theo vị trí =====
    dist_res = _dist(out["res_near"])
    dist_sup = _dist(out["sup_near"])

    near_res = dist_res <= 0.35 * a
    near_sup = dist_sup <= 0.35 * a
    mid_zone = (not near_res and not near_sup)

    buy_blocked = ("BUY_TOO_HIGH" in sf_tag) or (sf_state == "BLOCK" and near_res)
    sell_blocked = ("SELL_TOO_LOW" in sf_tag) or (sf_state == "BLOCK" and near_sup)

    # Bias chính
    is_buy_context = (
        up_score >= down_score
        and (h1 == "bullish" or h4 == "bullish" or ema_trend == "BULLISH")
    )
    is_sell_context = (
        down_score >= up_score
        and (h1 == "bearish" or h4 == "bearish" or ema_trend == "BEARISH")
    )

    # Breakout chỉ hợp lệ khi:
    # 1) đang gần kháng cự / hỗ trợ thật
    # 2) bias cùng hướng
    # 3) không bị smart filter block sai phía
    allow_breakout_buy = is_buy_context and near_res and not buy_blocked
    allow_breakout_sell = is_sell_context and near_sup and not sell_blocked

    out["priority_action"] = "ƯU TIÊN ĐỨNG NGOÀI / CHỜ NẾN M15 RÕ"
    out["action_note"] = "Chưa có lợi thế rõ."

    # =========================
    # BUY CONTEXT
    # =========================
    if is_buy_context:

        # A. Gần hỗ trợ -> đúng pha buy dip
        if near_sup:
            out["priority_action"] = "ƯU TIÊN canh BUY vùng hỗ trợ M15"
            out["action_note"] = (
                "Đúng pha buy-the-dip: giá đang gần hỗ trợ / đáy range. "
                "Ưu tiên chờ sweep low, giữ đáy hoặc nến xác nhận rồi BUY."
            )

        # B. Giữa range -> chưa phải breakout phase
        elif mid_zone:
            pos_note = "giá chưa ở điểm buy-dip đẹp"
            try:
                if rlo is not None and rhi is not None and rhi > rlo:
                    rp_val = (cp - rlo) / max(1e-9, (rhi - rlo))
                    if rp_val >= 0.80:
                        pos_note = "giá đang ở vùng cao / gần kháng cự"
                    elif rp_val > 0.60:
                        pos_note = "giá đang ở nửa trên range"
                    elif 0.40 <= rp_val <= 0.60:
                        pos_note = "giá đang ở giữa biên độ"
                    elif rp_val <= 0.20:
                        pos_note = "giá đang ở vùng thấp / gần hỗ trợ"
                    else:
                        pos_note = "giá đang ở nửa dưới range"
            except Exception:
                pass

            out["priority_action"] = "CHỜ về hỗ trợ để BUY hoặc break rõ rồi mới BUY"
            out["action_note"] = (
                f"Context vẫn BUY nhưng {pos_note}, chưa có điểm vào đẹp. "
                "Không BUY giữa đường, ưu tiên chờ hồi về support."
            )

        # C. Gần kháng cự
        elif near_res:
            if buy_blocked:
                out["priority_action"] = "KHÔNG BUY đuổi; chờ break hẳn rồi retest để BUY"
                out["action_note"] = (
                    "Giá đang sát kháng cự / quá cao so với điểm vào đẹp. "
                    "Không BUY đuổi ở vùng này."
                )
            elif allow_breakout_buy:
                out["priority_action"] = "CHỜ break kháng cự + follow-through rồi mới BUY"
                out["action_note"] = (
                    "Đây mới là pha breakout hợp lệ: giá đã áp sát kháng cự trong context BUY. "
                    "Chỉ BUY khi break rõ và nến sau giữ được."
                )
            else:
                out["priority_action"] = "CHỜ phản ứng tại kháng cự rồi quyết định"
                out["action_note"] = (
                    "Đã gần kháng cự nhưng chưa đủ điều kiện breakout sạch."
                )

    # =========================
    # SELL CONTEXT
    # =========================
    elif is_sell_context:

        # A. Gần kháng cự -> đúng pha sell rally
        if near_res:
            out["priority_action"] = "ƯU TIÊN canh SELL vùng kháng cự M15"
            out["action_note"] = (
                "Đúng pha sell-the-rally: giá đang gần kháng cự / đỉnh range. "
                "Ưu tiên chờ fail break, rejection hoặc nến xác nhận rồi SELL."
            )
        # B. Giữa range -> chưa phải breakdown phase
        elif mid_zone:
            pos_note = "giá chưa ở điểm sell-rally đẹp"
            try:
                if rlo is not None and rhi is not None and rhi > rlo:
                    rp_val = (cp - rlo) / max(1e-9, (rhi - rlo))
                    if rp_val >= 0.80:
                        pos_note = "giá đang ở vùng cao / gần kháng cự"
                    elif rp_val > 0.60:
                        pos_note = "giá đang ở nửa trên range"
                    elif 0.40 <= rp_val <= 0.60:
                        pos_note = "giá đang ở giữa biên độ"
                    elif rp_val <= 0.20:
                        pos_note = "giá đang ở vùng thấp / gần hỗ trợ"
                    else:
                        pos_note = "giá đang ở nửa dưới range"
            except Exception:
                pass

            out["priority_action"] = "CHỜ lên kháng cự để SELL hoặc break rõ rồi mới SELL"
            out["action_note"] = (
                f"Context vẫn SELL nhưng {pos_note}, chưa có điểm vào đẹp. "
                "Không SELL giữa đường, ưu tiên chờ hồi lên resistance."
            )
        # C. Gần hỗ trợ
        elif near_sup:
            if sell_blocked:
                out["priority_action"] = "KHÔNG SELL đuổi; chờ break hẳn rồi retest để SELL"
                out["action_note"] = (
                    "Giá đang sát hỗ trợ / quá thấp so với điểm vào đẹp. "
                    "Không SELL đuổi ở vùng này."
                )
            elif allow_breakout_sell:
                out["priority_action"] = "CHỜ break hỗ trợ + follow-through rồi mới SELL"
                out["action_note"] = (
                    "Đây mới là pha breakdown hợp lệ: giá đã áp sát hỗ trợ trong context SELL. "
                    "Chỉ SELL khi break rõ và nến sau giữ được."
                )
            else:
                out["priority_action"] = "CHỜ phản ứng tại hỗ trợ rồi quyết định"
                out["action_note"] = (
                    "Đã gần hỗ trợ nhưng chưa đủ điều kiện breakdown sạch."
                )

    # =========================
    # CONTEXT KHÔNG RÕ
    # =========================
    else:
        if near_sup and not sell_blocked:
            out["priority_action"] = "CHỜ phản ứng tại hỗ trợ rồi quyết định"
            out["action_note"] = "Giá gần hỗ trợ nhưng bias chưa đủ rõ."
        elif near_res and not buy_blocked:
            out["priority_action"] = "CHỜ phản ứng tại kháng cự rồi quyết định"
            out["action_note"] = "Giá gần kháng cự nhưng bias chưa đủ rõ."
        else:
            out["priority_action"] = "ƯU TIÊN ĐỨNG NGOÀI / CHỜ NẾN M15 RÕ"
            out["action_note"] = "Thị trường đang ở vùng khó, chưa có lợi thế rõ."


    # ===== Reasoning bổ sung cho forecast =====
    rp_hint = None
    try:
        if rlo is not None and rhi is not None and rhi > rlo:
            rp_val = (cp - rlo) / max(1e-9, (rhi - rlo))
            if rp_val >= 0.80:
                rp_hint = "đang sát vùng cao / gần kháng cự"
            elif rp_val <= 0.20:
                rp_hint = "đang sát vùng thấp / gần hỗ trợ"
            elif 0.40 <= rp_val <= 0.60:
                rp_hint = "đang giữa biên độ"
            elif rp_val > 0.60:
                rp_hint = "đang ở nửa trên range"
            else:
                rp_hint = "đang ở nửa dưới range"
    except Exception:
        rp_hint = None

    if near_sup:
        reasons.append("gần hỗ trợ")
    elif near_res:
        reasons.append("gần kháng cự")
    elif rp_hint:
        reasons.append(rp_hint)
    else:
        reasons.append("đang giữa biên độ")

    out["reason"] = reasons[:4]
    return out

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
def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _compression_label(m15c: Sequence[Any], volq: Dict[str, Any], atr15: float) -> Dict[str, Any]:
    """
    Đo độ nén đơn giản:
    - ATR ngắn co lại
    - range 6 nến gần nhất nhỏ hơn range 24 nến
    - volume thấp
    """
    if not m15c or len(m15c) < 30:
        return {
            "score": 0,
            "label": "LOW",
            "timing": "CHƯA RÕ",
            "reasons": ["thiếu dữ liệu nến"],
        }

    closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
    last6 = closed[-6:]
    last24 = closed[-24:]

    def _avg_range(cs):
        vals = [abs(float(c.high) - float(c.low)) for c in cs]
        return sum(vals) / max(1, len(vals))

    rng6 = _avg_range(last6)
    rng24 = _avg_range(last24)

    score = 0
    reasons = []

    # 1) range co
    if rng24 > 0 and rng6 <= 0.72 * rng24:
        score += 1
        reasons.append("range ngắn đang co lại")

    # 2) ATR hiện tại không lớn
    if atr15 and rng6 <= 0.85 * float(atr15):
        score += 1
        reasons.append("biên độ gần nhỏ hơn ATR")

    # 3) volume thấp
    vol_state = str((volq or {}).get("state") or "").upper()
    vol_ratio = float((volq or {}).get("ratio") or 1.0)
    if vol_state == "LOW" or vol_ratio <= 0.85:
        score += 1
        reasons.append("volume cạn dần")

    if score >= 3:
        label = "HIGH"
        timing = "SẮP XẢY RA"
    elif score == 2:
        label = "MEDIUM"
        timing = "ĐANG TÍCH LŨY"
    else:
        label = "LOW"
        timing = "CHƯA RÕ"

    return {
        "score": score,
        "label": label,
        "timing": timing,
        "reasons": reasons,
        "rng6": rng6,
        "rng24": rng24,
    }


def _liquidity_side_hint(range_pos: Optional[float], m15_struct_tag: str) -> Dict[str, Any]:
    """
    Gợi ý thanh khoản nằm đâu theo vị trí hiện tại + cấu trúc.
    Không cần quá phức tạp ở bản đầu.
    """
    above = False
    below = False
    reasons = []

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    tag = str(m15_struct_tag or "").upper()

    if rp is not None:
        if rp <= 0.30:
            below = True
            reasons.append("liquidity nằm dưới")
        elif rp >= 0.70:
            above = True
            reasons.append("liquidity nằm trên")

    if "LL" in tag or "LH" in tag:
        below = True
    if "HH" in tag or "HL" in tag:
        above = True

    return {
        "above": above,
        "below": below,
        "reasons": reasons,
    }


def _predict_pump_dump_v1(
    symbol: str,
    m15c: Sequence[Any],
    h1_trend: str,
    htf_pressure_v4: Dict[str, Any],
    market_state_v2: str,
    flow_state: Dict[str, Any],
    range_pos: Optional[float],
    volq: Dict[str, Any],
    atr15: float,
    m15_struct_tag: str,
    liquidation_evt: Dict[str, Any],
    liquidity_map_v1: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Trả về:
    - Compression
    - Bias bung
    - Xác suất
    - Thời điểm
    - Lý do
    """
    comp = _compression_label(m15c, volq, atr15)
    liq = _liquidity_side_hint(range_pos, m15_struct_tag)

    pump_score = 0
    dump_score = 0
    reasons = []

    # 1) Structure M15
    tag = str(m15_struct_tag or "").upper()
    if "LL" in tag or "LH" in tag:
        dump_score += 2
        reasons.append("cấu trúc giảm")
    elif "HH" in tag or "HL" in tag:
        pump_score += 2
        reasons.append("cấu trúc tăng")

    # 2) HTF bias
    htf_state = str((htf_pressure_v4 or {}).get("state") or "").upper()
    if "BEARISH" in htf_state or str(h1_trend).lower() == "bearish":
        dump_score += 1
        reasons.append("H1/HTF yếu")
    elif "BULLISH" in htf_state or str(h1_trend).lower() == "bullish":
        pump_score += 1
        reasons.append("H1/HTF mạnh")

    # 3) Range position
    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if rp is not None:
        if rp <= 0.30:
            dump_score += 1
            reasons.append("nén sát đáy range")
        elif rp >= 0.70:
            pump_score += 1
            reasons.append("nén sát đỉnh range")

    # 4) Liquidity side
    if liq.get("below"):
        dump_score += 1
        reasons.append("liquidity nằm dưới")
    if liq.get("above"):
        pump_score += 1
        reasons.append("liquidity nằm trên")

    # 5) Flow
    favored = str((flow_state or {}).get("favored_side") or "").upper()
    if favored == "SELL":
        dump_score += 1
    elif favored == "BUY":
        pump_score += 1

    # 6) liquidation vừa xảy ra thì nghiêng theo hướng quét gần nhất nhưng hạ độ tin cậy
    just_liquidated = bool((liquidation_evt or {}).get("ok"))
    if just_liquidated:
        side = str((liquidation_evt or {}).get("side") or "").upper()
        if "SELL" in side:
            dump_score += 1
            reasons.append("vừa có sell liquidation")
        elif "BUY" in side:
            pump_score += 1
            reasons.append("vừa có buy liquidation")

    # Kết luận hướng
    if dump_score > pump_score:
        bias = "DUMP nghiêng hơn"
        edge = dump_score - pump_score
    elif pump_score > dump_score:
        bias = "PUMP nghiêng hơn"
        edge = pump_score - dump_score
    else:
        bias = "NEUTRAL"
        edge = 0

    # Xác suất
    comp_label = comp.get("label", "LOW")
    if bias == "NEUTRAL":
        probability = "LOW"
    elif comp_label == "HIGH" and edge >= 2:
        probability = "HIGH"
    elif comp_label in ("HIGH", "MEDIUM"):
        probability = "MEDIUM"
    else:
        probability = "LOW"

    # liquidation vừa xảy ra => hạ 1 nấc để tránh quá tự tin
    if just_liquidated and probability == "HIGH":
        probability = "MEDIUM"

    timing = comp.get("timing", "CHƯA RÕ")

    final_reasons = []
    for x in reasons:
        if x not in final_reasons:
            final_reasons.append(x)
    liquidity_map_v1 = liquidity_map_v1 or {}
    sweep = str(liquidity_map_v1.get("sweep_bias") or "")
    if "UP → DOWN" in sweep:
        pump_bias = "FAKE PUMP → DUMP"
    
    elif "DOWN → UP" in sweep:
        pump_bias = "FAKE DUMP → PUMP"
    return {
        "compression": comp_label,
        "bias": bias,
        "probability": probability,
        "timing": timing,
        "reasons": final_reasons[:4],
        "pump_score": pump_score,
        "dump_score": dump_score,
    }


def _entry_sniper_v1(
    m15c,
    m15_struct: dict | None,
    atr15: float | None,
    volq: dict | None = None,
) -> dict:
    """
    Entry Sniper:
    - Cây chỉ hướng
    - Điểm nổ
    - Trạng thái
    """

    out = {
        "direction": "NONE",
        "strength": "-",
        "trigger": "NONE",
        "state": "KHÔNG CÓ SETUP",
        "reason": "",
    }

    if not m15c or len(m15c) < 25:
        out["reason"] = "thiếu dữ liệu M15"
        return out

    closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
    if len(closed) < 10:
        out["reason"] = "thiếu nến đã đóng"
        return out

    last = closed[-1]
    prev = closed[-2]

    def _body(c):
        return abs(float(c.close) - float(c.open))

    def _range(c):
        return max(1e-9, float(c.high) - float(c.low))

    # ===== 1) Cây chỉ hướng =====
    body_last = _body(last)
    range_last = _range(last)

    avg_body = sum(_body(x) for x in closed[-6:]) / max(1, len(closed[-6:]))
    big_body = body_last >= 0.55 * range_last
    displacement = body_last >= 1.35 * max(avg_body, 1e-9)

    direction = "NONE"
    strength = "-"

    # dựa thêm vào cấu trúc ngắn hạn
    m15_tag = str((m15_struct or {}).get("tag") or "").upper()

    if float(last.close) < float(last.open) and big_body and displacement:
        direction = "SELL"
        strength = "STRONG" if body_last >= 1.6 * max(avg_body, 1e-9) else "MEDIUM"

    elif float(last.close) > float(last.open) and big_body and displacement:
        direction = "BUY"
        strength = "STRONG" if body_last >= 1.6 * max(avg_body, 1e-9) else "MEDIUM"

    # cấu trúc hỗ trợ tăng độ tin cậy
    if direction == "SELL" and ("LL" in m15_tag or "LH" in m15_tag):
        if strength == "MEDIUM":
            strength = "STRONG"
    if direction == "BUY" and ("HH" in m15_tag or "HL" in m15_tag):
        if strength == "MEDIUM":
            strength = "STRONG"

    # nếu chưa có cây chỉ hướng
    if direction == "NONE":
        out["reason"] = "chưa có cây chỉ hướng rõ"
        return out

    out["direction"] = direction
    out["strength"] = strength

    # ===== 2) Điểm nổ =====
    # Ý tưởng đơn giản:
    # - READY: sau cây chỉ hướng có hồi yếu / giữ hướng
    # - TRIGGERED: phá tiếp theo hướng đó
    trigger = "NONE"
    state = "CHỜ"

    recent_high = max(float(x.high) for x in closed[-6:-1]) if len(closed) >= 6 else float(last.high)
    recent_low = min(float(x.low) for x in closed[-6:-1]) if len(closed) >= 6 else float(last.low)

    vol_state = str((volq or {}).get("state") or "").upper()

    if direction == "SELL":
        # READY: nến trước đó đã là cây chỉ hướng, nến hiện tại hồi yếu / không vượt nổi
        weak_pullback = float(last.close) <= (float(last.open) + 0.35 * range_last)
        # TRIGGERED: phá đáy gần + có follow-through
        triggered = float(last.close) < recent_low

        if triggered:
            trigger = "TRIGGERED"
            state = "CÓ THỂ VÀO"
        elif weak_pullback or vol_state == "HIGH":
            trigger = "READY"
            state = "SẮP NỔ"

    elif direction == "BUY":
        weak_pullback = float(last.close) >= (float(last.open) - 0.35 * range_last)
        triggered = float(last.close) > recent_high

        if triggered:
            trigger = "TRIGGERED"
            state = "CÓ THỂ VÀO"
        elif weak_pullback or vol_state == "HIGH":
            trigger = "READY"
            state = "SẮP NỔ"

    out["trigger"] = trigger
    out["state"] = state
    out["reason"] = f"body={body_last:.2f} | avg_body={avg_body:.2f} | m15={m15_tag or 'n/a'}"

    return out

def _liquidity_strength_label(score: int) -> str:
    if score >= 4:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    if score >= 1:
        return "LOW"
    return "LOW"


def _fmt_price_range(a, b) -> str:
    try:
        if a is None and b is None:
            return "n/a"
        if b is None or abs(float(a) - float(b)) < 1e-9:
            return f"{float(a):.2f}"
        lo = min(float(a), float(b))
        hi = max(float(a), float(b))
        return f"{lo:.2f} – {hi:.2f}"
    except Exception:
        return "n/a"

def _liq_tol_from_context(recent_hi: float, recent_lo: float, atr15: float | None) -> float:
    span = max(1e-9, float(recent_hi) - float(recent_lo))
    a = float(atr15 or 0.0)
    return max(span * 0.015, a * 0.12 if a > 0 else span * 0.02)


def _count_hits_near_level(values: list[float], level: float, tol: float) -> int:
    return sum(1 for v in values if abs(float(v) - float(level)) <= tol)


def _equal_cluster_strength(values: list[float], tol: float, min_hits: int = 2) -> tuple[int, tuple[float, float] | None]:
    """
    Tìm cụm equal highs / equal lows đơn giản.
    Return:
    - strength score (0..3)
    - zone (lo, hi) hoặc None
    """
    if not values:
        return 0, None

    vals = sorted(float(v) for v in values)
    best_hits = 0
    best_center = None

    for v in vals:
        hits = [x for x in vals if abs(x - v) <= tol]
        if len(hits) > best_hits:
            best_hits = len(hits)
            best_center = sum(hits) / len(hits)

    if best_hits < min_hits or best_center is None:
        return 0, None

    if best_hits >= 4:
        score = 3
    elif best_hits == 3:
        score = 2
    else:
        score = 1

    return score, (best_center - tol, best_center + tol)


def _distance_score(cur: float, zone: tuple[float, float] | None, recent_span: float) -> int:
    """
    Giá càng gần pool thanh khoản thì khả năng quét đầu đó càng cao hơn.
    Score 0..3
    """
    if zone is None:
        return 0

    zlo, zhi = zone
    if zlo <= cur <= zhi:
        return 3

    dist = min(abs(cur - zlo), abs(cur - zhi))
    span = max(1e-9, recent_span)

    if dist <= 0.08 * span:
        return 3
    if dist <= 0.16 * span:
        return 2
    if dist <= 0.28 * span:
        return 1
    return 0


def _untouched_bonus(cur: float, highs: list[float], lows: list[float], recent_hi: float, recent_lo: float, tol: float) -> tuple[int, int]:
    """
    Pool nào ít bị chạm gần đây hơn thì hay còn stop hơn.
    Return: (above_bonus, below_bonus)
    """
    hi_hits = _count_hits_near_level(highs[-8:], recent_hi, tol) if highs else 0
    lo_hits = _count_hits_near_level(lows[-8:], recent_lo, tol) if lows else 0

    above_bonus = 1 if hi_hits <= 1 else 0
    below_bonus = 1 if lo_hits <= 1 else 0
    return above_bonus, below_bonus
    
def _build_liquidity_map_v1(
    symbol: str,
    m15c,
    h1_trend: str,
    htf_pressure_v4: dict | None,
    flow_state: dict | None,
    range_pos: float | None,
    market_state_v2: str | None,
    playbook_v2: dict | None,
    liquidation_evt: dict | None,
    m15_struct_tag: str | None,
    range_low: float | None,
    range_high: float | None,
) -> dict:
    """
    Liquidity map upgraded:
    - vẫn giữ output keys cũ
    - đọc rõ hơn pool thanh khoản trên / dưới
    - cố gắng trả lời đầu nào dễ bị MM quét trước
    """
    out = {
        "above_strength": "LOW",
        "below_strength": "LOW",
        "above_zone": None,
        "below_zone": None,
        "state_text": "Chưa thấy sweep/spring rõ",
        "sweep_bias": "NEUTRAL",
        "story": "Chưa có câu chuyện liquidity rõ",
        "equal_highs": False,
        "equal_lows": False,
        "reasons": [],
    }

    if not m15c or len(m15c) < 20:
        out["state_text"] = "Thiếu dữ liệu để đọc thanh khoản"
        return out

    closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
    use = closed[-24:] if len(closed) >= 24 else closed
    if len(use) < 12:
        out["state_text"] = "Thiếu dữ liệu để đọc thanh khoản"
        return out

    highs = [float(c.high) for c in use]
    lows = [float(c.low) for c in use]
    closes = [float(c.close) for c in use]

    recent_hi = max(highs)
    recent_lo = min(lows)
    cur = float(closes[-1])
    span = max(1e-9, recent_hi - recent_lo)
    atr15 = _atr(closed, 14) or 0.0
    tol = _liq_tol_from_context(recent_hi, recent_lo, atr15)

    # ===== 1) cụm liquidity cơ bản =====
    hi_hits = _count_hits_near_level(highs, recent_hi, tol)
    lo_hits = _count_hits_near_level(lows, recent_lo, tol)

    above_score = 0
    below_score = 0
    reasons = []

    if hi_hits >= 2:
        above_score += 2
        reasons.append("phía trên có cụm đỉnh gần nhau")
    if hi_hits >= 4:
        above_score += 1

    if lo_hits >= 2:
        below_score += 2
        reasons.append("phía dưới có cụm đáy gần nhau")
    if lo_hits >= 4:
        below_score += 1

    # ===== 2) equal highs / equal lows =====
    eh_score, eh_zone = _equal_cluster_strength(highs, tol, min_hits=2)
    el_score, el_zone = _equal_cluster_strength(lows, tol, min_hits=2)

    if eh_score > 0:
        above_score += eh_score
        out["above_zone"] = eh_zone
        out["equal_highs"] = True
        reasons.append("equal highs phía trên")
    else:
        out["above_zone"] = (recent_hi - tol, recent_hi)

    if el_score > 0:
        below_score += el_score
        out["below_zone"] = el_zone
        out["equal_lows"] = True
        reasons.append("equal lows phía dưới")
    else:
        out["below_zone"] = (recent_lo, recent_lo + tol)

    # ===== 3) structure phụ trợ =====
    tag = str(m15_struct_tag or "").upper()
    if "HH" in tag or "HL" in tag:
        above_score += 1
    if "LL" in tag or "LH" in tag:
        below_score += 1

    # ===== 4) pool nào gần giá hơn → dễ sweep trước hơn =====
    above_dist_score = _distance_score(cur, out["above_zone"], span)
    below_dist_score = _distance_score(cur, out["below_zone"], span)
    above_score += above_dist_score
    below_score += below_dist_score

    if above_dist_score > below_dist_score:
        reasons.append("giá đang gần pool phía trên hơn")
    elif below_dist_score > above_dist_score:
        reasons.append("giá đang gần pool phía dưới hơn")

    # ===== 5) pool nào còn untouched hơn =====
    above_bonus, below_bonus = _untouched_bonus(cur, highs, lows, recent_hi, recent_lo, tol)
    above_score += above_bonus
    below_score += below_bonus

    if above_bonus > 0:
        reasons.append("pool phía trên còn tương đối untouched")
    if below_bonus > 0:
        reasons.append("pool phía dưới còn tương đối untouched")

    # ===== 6) trạng thái / liquidation text =====
    liq_evt = liquidation_evt or {}
    if liq_evt.get("ok"):
        side = str(liq_evt.get("side") or "")
        kind = str(liq_evt.get("kind") or "")
        out["state_text"] = f"Vừa có quét mạnh: {side} | {kind}"
    else:
        out["state_text"] = "Chưa thấy sweep/spring rõ"

    # ===== 7) bias tổng hợp như cũ, nhưng cho liquidity weight lớn hơn =====
    pump_score = 0
    dump_score = 0

    htf_state = str((htf_pressure_v4 or {}).get("state") or "").upper()
    flow_favored = str((flow_state or {}).get("favored_side") or "").upper()
    plan = str((playbook_v2 or {}).get("plan") or "").upper()
    ms = str(market_state_v2 or "").upper()

    # liquidity actual pools
    pump_score += below_score
    dump_score += above_score

    # HTF
    if "BEARISH" in htf_state or str(h1_trend).lower() == "bearish":
        dump_score += 2
        reasons.append("HTF nghiêng giảm")
    elif "BULLISH" in htf_state or str(h1_trend).lower() == "bullish":
        pump_score += 2
        reasons.append("HTF nghiêng tăng")

    # Flow
    if flow_favored == "SELL":
        dump_score += 1
        reasons.append("flow ưu tiên SELL")
    elif flow_favored == "BUY":
        pump_score += 1
        reasons.append("flow ưu tiên BUY")

    # Range position
    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if rp is not None:
        if rp <= 0.20:
            pump_score += 1
            reasons.append("đang ở vùng thấp")
        elif rp >= 0.80:
            dump_score += 1
            reasons.append("đang ở vùng cao")

    # Playbook
    if plan in ("BOUNCE_TO_SELL", "SELL_RALLY", "WAIT_BOUNCE_TO_SELL"):
        dump_score += 2
        reasons.append("playbook nghiêng SELL")
    elif plan in ("DIP_TO_BUY", "BUY_DIP", "WAIT_PULLBACK_TO_BUY"):
        pump_score += 2
        reasons.append("playbook nghiêng BUY")

    if ms in ("CHOP", "TRANSITION"):
        reasons.append("market nhiễu")

    # ===== 8) output strength =====
    out["above_strength"] = _liquidity_strength_label(above_score)
    out["below_strength"] = _liquidity_strength_label(below_score)

    # ===== 9) quyết định quét đầu nào trước =====
    # Logic:
    # - nếu price đang cao + pool trên mạnh + HTF bearish -> UP → DOWN
    # - nếu price đang thấp + pool dưới mạnh + HTF bullish -> DOWN → UP
    # - còn lại: nghiêng quét theo phía có score mạnh hơn
    if rp is not None and rp >= 0.75 and above_score >= below_score:
        if dump_score >= pump_score:
            out["sweep_bias"] = "UP → DOWN"
            reasons.append("ở vùng cao + pool trên rõ → dễ quét lên trước")
        else:
            out["sweep_bias"] = "UP"
    elif rp is not None and rp <= 0.25 and below_score >= above_score:
        if pump_score >= dump_score:
            out["sweep_bias"] = "DOWN → UP"
            reasons.append("ở vùng thấp + pool dưới rõ → dễ quét xuống trước")
        else:
            out["sweep_bias"] = "DOWN"
    else:
        if dump_score - pump_score >= 2:
            out["sweep_bias"] = "UP → DOWN" if above_score >= below_score else "DOWN"
        elif pump_score - dump_score >= 2:
            out["sweep_bias"] = "DOWN → UP" if below_score >= above_score else "UP"
        else:
            out["sweep_bias"] = "NEUTRAL"

    # liquidation vừa xảy ra → hạ độ chắc chắn
    if liq_evt.get("ok") and out["sweep_bias"] in ("UP", "DOWN", "UP → DOWN", "DOWN → UP"):
        out["sweep_bias"] = out["sweep_bias"] + " (cẩn thận quét 2 đầu)"


    # ===== 10) liquidity story =====
    if out["equal_highs"] and out["equal_lows"]:
        out["story"] = "2 đầu đều có liquidity: equal highs phía trên, equal lows phía dưới"
    elif out["equal_highs"]:
        out["story"] = "Phía trên có equal highs / buy-side liquidity rõ hơn"
    elif out["equal_lows"]:
        out["story"] = "Phía dưới có equal lows / sell-side liquidity rõ hơn"
    elif above_score > below_score:
        out["story"] = "Pool phía trên gần và dày hơn → dễ quét lên trước"
    elif below_score > above_score:
        out["story"] = "Pool phía dưới gần và dày hơn → dễ quét xuống trước"
    else:
        out["story"] = "Liquidity hai đầu khá cân bằng, dễ quét 2 đầu"

    dedup = []
    for r in reasons:
        if r not in dedup:
            dedup.append(r)
    out["reasons"] = dedup[:5]

    return out

def _detect_session_gap_v1(m15c, atr15: float | None = None) -> dict:
    """
    Detect session/open gap thật:
    - có time gap lớn giữa 2 nến liên tiếp
    - open nến sau lệch close nến trước đủ lớn so với ATR
    """
    out = {
        "active": False,
        "type": "NONE",
        "gap_low": None,
        "gap_high": None,
        "size": 0.0,
        "fill_pct": 0.0,
        "text": "Chưa có dấu hiệu GAP / mở cửa bất thường rõ",
    }

    if not m15c or len(m15c) < 3:
        return out

    closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
    if len(closed) < 2:
        return out

    try:
        # median delta để nhận ra nến mở phiên / nhảy session
        deltas = []
        for i in range(1, len(closed)):
            t1 = int(_c_val(closed[i - 1], "ts", 0) or 0)
            t2 = int(_c_val(closed[i], "ts", 0) or 0)
            if t1 > 0 and t2 > t1:
                deltas.append(t2 - t1)

        if not deltas:
            return out

        med_delta = sorted(deltas)[len(deltas) // 2]
        a = float(atr15 or 0.0)
        gap_min = max(0.25 * a, 1e-9) if a > 0 else 0.0

        idx_found = None
        for i in range(1, len(closed)):
            t1 = int(_c_val(closed[i - 1], "ts", 0) or 0)
            t2 = int(_c_val(closed[i], "ts", 0) or 0)
            if med_delta > 0 and (t2 - t1) >= 1.5 * med_delta:
                prev_close = float(_c_val(closed[i - 1], "close", 0.0) or 0.0)
                cur_open = float(_c_val(closed[i], "open", 0.0) or 0.0)
                if abs(cur_open - prev_close) >= gap_min:
                    idx_found = i
                    break

        if idx_found is None:
            return out

        prev_close = float(_c_val(closed[idx_found - 1], "close", 0.0) or 0.0)
        cur_open = float(_c_val(closed[idx_found], "open", 0.0) or 0.0)
        cur_price = float(_c_val(closed[-1], "close", 0.0) or 0.0)

        gap_low = min(prev_close, cur_open)
        gap_high = max(prev_close, cur_open)
        gap_size = gap_high - gap_low

        if cur_open > prev_close:
            gap_type = "GAP_UP"
            fill_pct = _clamp((gap_high - cur_price) / max(gap_size, 1e-9), 0.0, 1.0)
        else:
            gap_type = "GAP_DOWN"
            fill_pct = _clamp((cur_price - gap_low) / max(gap_size, 1e-9), 0.0, 1.0)

        out.update({
            "active": True,
            "type": gap_type,
            "gap_low": gap_low,
            "gap_high": gap_high,
            "size": gap_size,
            "fill_pct": fill_pct * 100.0,
            "text": f"{gap_type} | {_fmt(gap_low)} – {_fmt(gap_high)} | fill ~{fill_pct * 100.0:.0f}%",
        })
        return out

    except Exception:
        return out


def _build_flow_engine_v1(
    symbol: str,
    m15c,
    current_price: float | None,
    atr15: float | None,
    liquidity_map_v1: dict | None,
    fvg_range_plugin_v1: dict | None,
    gap_info_v1: dict | None = None,
) -> dict:
    """
    Merge 3 thứ lại:
    - Liquidity
    - GAP
    - FVG / imbalance

    Output dùng chung cho NOW / REVIEW / ALERT.
    """
    liq = liquidity_map_v1 if isinstance(liquidity_map_v1, dict) else {}
    fvgp = fvg_range_plugin_v1 if isinstance(fvg_range_plugin_v1, dict) else {}
    fvg = fvgp.get("fvg") or {}
    cp = _safe_float(current_price)

    if gap_info_v1 and isinstance(gap_info_v1, dict):
        gap1 = gap_info_v1
    else:
        gap1 = _detect_session_gap_v1(m15c=m15c, atr15=atr15)

    out = {
        "state": "NEUTRAL",
        "displacement": "NONE",
        "liquidity_state": liq.get("state_text", "Chưa thấy sweep/spring rõ"),
        "liquidity_done": bool(liq.get("done")),
        "liquidity_bias": str(liq.get("sweep_bias") or "NEUTRAL"),
        "above_strength": str(liq.get("above_strength") or "LOW"),
        "below_strength": str(liq.get("below_strength") or "LOW"),
        "gap_active": bool(gap1.get("active")),
        "gap_text": str(gap1.get("text") or "Chưa có dấu hiệu GAP / mở cửa bất thường rõ"),
        "fvg_active": bool(fvg.get("ok")),
        "fvg_text": str(fvg.get("text") or "chưa có vùng rõ"),
        "fvg_side": str(fvg.get("type") or "NONE").upper(),
        "narrative": "Chưa có câu chuyện flow rõ",
        "action_hint": "WAIT",
        "reasons": [],
    }

    reasons = []

    # ---- displacement sơ bộ ----
    try:
        closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
        if len(closed) >= 3:
            c1 = closed[-1]
            rng = abs(float(_c_val(c1, "high", 0.0) or 0.0) - float(_c_val(c1, "low", 0.0) or 0.0))
            body = abs(float(_c_val(c1, "close", 0.0) or 0.0) - float(_c_val(c1, "open", 0.0) or 0.0))
            a = float(atr15 or 0.0)
            if a > 0 and rng >= 1.2 * a:
                if float(_c_val(c1, "close", 0.0) or 0.0) > float(_c_val(c1, "open", 0.0) or 0.0):
                    out["displacement"] = "STRONG_UP"
                elif float(_c_val(c1, "close", 0.0) or 0.0) < float(_c_val(c1, "open", 0.0) or 0.0):
                    out["displacement"] = "STRONG_DOWN"
            elif body > 0 and rng > 0 and (body / max(rng, 1e-9)) >= 0.55:
                if float(_c_val(c1, "close", 0.0) or 0.0) > float(_c_val(c1, "open", 0.0) or 0.0):
                    out["displacement"] = "UP"
                elif float(_c_val(c1, "close", 0.0) or 0.0) < float(_c_val(c1, "open", 0.0) or 0.0):
                    out["displacement"] = "DOWN"
    except Exception:
        pass

    if out["gap_active"]:
        reasons.append("có session gap")
    if out["fvg_active"]:
        reasons.append("có imbalance / FVG")
    if not out["liquidity_done"]:
        reasons.append("thanh khoản chưa hoàn tất")

    # ---- narrative / action ----
    if out["gap_active"] and out["fvg_active"] and not out["liquidity_done"]:
        out["state"] = "IMBALANCED"
        out["narrative"] = "Có gap + imbalance nhưng thanh khoản chưa hoàn tất → dễ còn nhịp fill / quét thêm"
        out["action_hint"] = "WAIT"
    elif out["fvg_active"] and not out["liquidity_done"]:
        out["state"] = "FVG_OPEN"
        out["narrative"] = "Có FVG nhưng chưa hoàn tất thanh khoản → ưu tiên chờ giá phản ứng / fill"
        out["action_hint"] = "WAIT_REACTION"
    elif out["gap_active"] and not out["fvg_active"]:
        out["state"] = "SESSION_GAP"
        out["narrative"] = "Có gap đầu phiên → dễ nhiễu đầu phiên, theo dõi khả năng fill gap"
        out["action_hint"] = "WAIT_FILL"
    elif out["liquidity_done"] and out["liquidity_bias"] != "NEUTRAL":
        out["state"] = "FLOW_READY"
        out["narrative"] = "Thanh khoản đã xử lý xong, có thể chuẩn bị follow nếu có confirm"
        out["action_hint"] = "WAIT_CONFIRM"
    else:
        out["state"] = "NEUTRAL"
        out["narrative"] = "Flow chưa rõ, chưa có displacement / imbalance đủ mạnh"
        out["action_hint"] = "WAIT"

    # ---- thêm reason cụ thể ----
    if out["gap_active"]:
        reasons.append(out["gap_text"])
    if out["fvg_active"]:
        reasons.append(out["fvg_text"])
    if liq.get("state_text"):
        reasons.append(str(liq.get("state_text")))
    
    out["reasons"] = reasons[:4]
    # ===== FLOW STATE FIX V2 =====
    # Nếu displacement mạnh thì không được để state = NEUTRAL.
    try:
        disp = str(out.get("displacement") or "NONE").upper()
        cur_state = str(out.get("state") or "NEUTRAL").upper()

        if disp == "STRONG_DOWN" and cur_state in ("NEUTRAL", "NONE", "N/A", ""):
            out["state"] = "TRENDING_DOWN"
            out["action_hint"] = "FOLLOW_SELL_CONDITIONAL"
            out["narrative"] = (
                "Có displacement giảm mạnh → flow thực tế nghiêng SELL; "
                "không SELL đuổi đáy, chỉ canh continuation/retest."
            )
            reasons = list(out.get("reasons") or [])
            reasons.insert(0, "displacement giảm mạnh")
            out["reasons"] = reasons[:4]

        elif disp == "STRONG_UP" and cur_state in ("NEUTRAL", "NONE", "N/A", ""):
            out["state"] = "TRENDING_UP"
            out["action_hint"] = "FOLLOW_BUY_CONDITIONAL"
            out["narrative"] = (
                "Có displacement tăng mạnh → flow thực tế nghiêng BUY; "
                "không BUY đuổi đỉnh, chỉ canh continuation/retest."
            )
            reasons = list(out.get("reasons") or [])
            reasons.insert(0, "displacement tăng mạnh")
            out["reasons"] = reasons[:4]
    except Exception:
        pass
    return out

def _build_mm_real_play_v1(
    liq_map: dict | None,
    range_pos: float | None,
    htf_pressure_v4: dict | None,
    playbook_v2: dict | None,
    ema_pack: dict | None,
) -> dict:

    out = {
        "headline": "Chưa có kịch bản MM rõ",
        "path": "NEUTRAL",
        "entry_hint": "Chờ thêm xác nhận",
        "reason": [],
    }

    liq_map = liq_map or {}
    htf_pressure_v4 = htf_pressure_v4 or {}
    playbook_v2 = playbook_v2 or {}
    ema_pack = ema_pack or {}

    sweep_bias = str(liq_map.get("sweep_bias") or "NEUTRAL").upper()
    above_strength = str(liq_map.get("above_strength") or "LOW").upper()
    below_strength = str(liq_map.get("below_strength") or "LOW").upper()
    htf_state = str(htf_pressure_v4.get("state") or "").upper()
    plan = str(playbook_v2.get("plan") or "").upper()
    ema_trend = str(ema_pack.get("trend") or "MIXED").upper()

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    reasons = []

    # ===== UP → DOWN =====
    if sweep_bias.startswith("UP → DOWN") or (
        rp is not None and rp >= 0.80 and above_strength in ("HIGH", "MEDIUM")
    ):
        reasons.append("pool phía trên rõ")
        if rp and rp >= 0.80:
            reasons.append("giá ở vùng cao")
        if "BEARISH" in htf_state:
            reasons.append("HTF nghiêng giảm")

        out["headline"] = "Quét đỉnh rồi đạp xuống"
        out["path"] = "UP → DOWN"
        out["entry_hint"] = "Chờ sweep đỉnh → fail giữ → SELL"
        out["reason"] = reasons[:4]
        return out

    # ===== DOWN → UP =====
    if sweep_bias.startswith("DOWN → UP") or (
        rp is not None and rp <= 0.20 and below_strength in ("HIGH", "MEDIUM")
    ):
        reasons.append("pool phía dưới rõ")
        if rp and rp <= 0.20:
            reasons.append("giá ở vùng thấp")
        if "BULLISH" in htf_state:
            reasons.append("HTF nghiêng tăng")

        out["headline"] = "Quét đáy rồi kéo lên"
        out["path"] = "DOWN → UP"
        out["entry_hint"] = "Chờ sweep đáy → bật mạnh → BUY"
        out["reason"] = reasons[:4]
        return out

    # ===== SIDEWAY =====
    if above_strength == "HIGH" and below_strength == "HIGH":
        reasons.append("2 đầu đều có liquidity")

    out["headline"] = "Market dễ quét 2 đầu"
    out["path"] = "2-SIDE"
    out["entry_hint"] = "Đứng ngoài chờ clear 1 phía"
    out["reason"] = reasons[:4]

    return out
def _nf2(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "n/a"


def _zone_pad_from_atr(atr15: float | None, ratio: float = 0.18) -> float:
    a = float(atr15 or 0.0)
    if a <= 0:
        return 0.0
    return max(1e-9, a * ratio)


def _scale_stage_v2(
    direction: str,
    sniper_dir: str,
    sniper_trigger: str,
) -> tuple[int, str]:
    """
    Stage V2:
    1 = chưa có setup
    2 = chuẩn bị
    3 = vào lệnh
    """
    direction = str(direction or "NONE").upper()
    sniper_dir = str(sniper_dir or "NONE").upper()
    sniper_trigger = str(sniper_trigger or "NONE").upper()

    if direction not in ("BUY", "SELL"):
        return 1, "Chưa có setup"

    if sniper_dir == "NONE":
        return 1, "Chưa có setup"

    if sniper_dir == direction and sniper_trigger in ("READY", "TRIGGERED"):
        return 3, "Vào lệnh"

    return 2, "Chuẩn bị"


def _scale_readiness_v2(
    direction: str,
    sniper_dir: str,
    sniper_trigger: str,
    late_move: bool,
) -> str:
    direction = str(direction or "NONE").upper()
    sniper_dir = str(sniper_dir or "NONE").upper()
    sniper_trigger = str(sniper_trigger or "NONE").upper()

    if direction not in ("BUY", "SELL"):
        return "LOW"

    if sniper_dir == "NONE":
        return "LOW"

    if late_move:
        return "LOW"

    if sniper_dir == direction and sniper_trigger in ("READY", "TRIGGERED"):
        return "HIGH"

    return "MEDIUM"

def _find_last_impulse_leg(m15c, side: str, lookback: int = 40) -> dict:
    """
    Tìm nhịp impulse gần nhất:
    - SELL: nhịp giảm mạnh gần nhất
    - BUY: nhịp tăng mạnh gần nhất

    Return:
    {
        "ok": bool,
        "leg_low": float|None,
        "leg_high": float|None,
        "start_idx": int|None,
        "end_idx": int|None,
        "reason": str,
    }
    """
    out = {
        "ok": False,
        "leg_low": None,
        "leg_high": None,
        "start_idx": None,
        "end_idx": None,
        "reason": "not_found",
    }

    if not m15c or len(m15c) < 15:
        out["reason"] = "not_enough_candles"
        return out

    closed = list(m15c[:-1] if len(m15c) > 1 else m15c)
    use = closed[-lookback:] if len(closed) >= lookback else closed

    atr15 = _atr(closed, 14) or 0.0
    if atr15 <= 0:
        out["reason"] = "atr_not_ready"
        return out

    best_score = -1.0
    best = None

    for i in range(3, len(use)):
        c = use[i]
        prev = use[i - 1]

        body = abs(float(c.close) - float(c.open))
        rng = max(1e-9, float(c.high) - float(c.low))

        # impulse candle phải đủ mạnh
        strong_body = body >= 0.55 * rng
        big_enough = body >= 0.9 * atr15

        if side == "SELL":
            direction_ok = float(c.close) < float(c.open)
            break_ok = float(c.close) < min(float(x.low) for x in use[max(0, i-4):i])
        else:
            direction_ok = float(c.close) > float(c.open)
            break_ok = float(c.close) > max(float(x.high) for x in use[max(0, i-4):i])

        if not (strong_body and big_enough and direction_ok and break_ok):
            continue

        # gom thêm 1-2 nến quanh impulse để lấy whole leg
        left = max(0, i - 2)
        right = min(len(use) - 1, i + 1)
        leg = use[left:right + 1]

        leg_low = min(float(x.low) for x in leg)
        leg_high = max(float(x.high) for x in leg)

        # score = độ mạnh thân + độ rộng leg
        score = body / max(1e-9, atr15) + (leg_high - leg_low) / max(1e-9, atr15)

        if score > best_score:
            best_score = score
            best = {
                "ok": True,
                "leg_low": leg_low,
                "leg_high": leg_high,
                "start_idx": left,
                "end_idx": right,
                "reason": "ok",
            }

    if best:
        return best

    return out

def _build_scale_zones_from_impulse(
    current_price: float,
    side: str,
    leg_low: float,
    leg_high: float,
    atr15: float,
    lot1: float,
    lot2: float,
    lot3: float,
) -> dict:
    """
    Dựng vùng scale từ impulse:
    - SELL: vùng scale nằm phía trên current price
    - BUY : vùng scale nằm phía dưới current price
    """
    rng = max(1e-9, float(leg_high) - float(leg_low))
    pad = max(1e-9, float(atr15 or 0.0) * 0.12)

    if side == "SELL":
        # hồi lên trong nhịp giảm: 38 / 50 / 61.8 tính từ đáy lên
        z1_mid = leg_low + 0.382 * rng
        z2_mid = leg_low + 0.500 * rng
        z3_mid = leg_low + 0.618 * rng

        orders = [
            {"name": "Lệnh 1", "zone_lo": z1_mid - pad, "zone_hi": z1_mid + pad, "lot": float(lot1)},
            {"name": "Lệnh 2", "zone_lo": z2_mid - pad, "zone_hi": z2_mid + pad, "lot": float(lot2)},
            {"name": "Lệnh 3", "zone_lo": z3_mid - pad, "zone_hi": z3_mid + pad, "lot": float(lot3)},
        ]

        # chỉ giữ các vùng nằm trên current_price, nếu không thì coi là đã lỡ
        valid_orders = [o for o in orders if float(o["zone_hi"]) > float(current_price)]
        invalid = leg_high + max(pad, 0.35 * atr15)

        return {
            "orders": valid_orders,
            "invalid": invalid,
            "tp1": leg_low,
            "tp2": leg_low - 0.8 * rng,
        }

    else:
        # hồi xuống trong nhịp tăng: 38 / 50 / 61.8 tính từ đỉnh xuống
        z1_mid = leg_high - 0.382 * rng
        z2_mid = leg_high - 0.500 * rng
        z3_mid = leg_high - 0.618 * rng

        orders = [
            {"name": "Lệnh 1", "zone_lo": z1_mid - pad, "zone_hi": z1_mid + pad, "lot": float(lot1)},
            {"name": "Lệnh 2", "zone_lo": z2_mid - pad, "zone_hi": z2_mid + pad, "lot": float(lot2)},
            {"name": "Lệnh 3", "zone_lo": z3_mid - pad, "zone_hi": z3_mid + pad, "lot": float(lot3)},
        ]

        valid_orders = [o for o in orders if float(o["zone_lo"]) < float(current_price)]
        invalid = leg_low - max(pad, 0.35 * atr15)

        return {
            "orders": valid_orders,
            "invalid": invalid,
            "tp1": leg_high,
            "tp2": leg_high + 0.8 * rng,
        }
        
def build_scale_plan_v2(
    symbol: str,
    m15,
    m30,
    h1,
    h4,
    total_tp_cent: float = 500.0,
    lot1: float = 0.30,
    lot2: float = 0.30,
    lot3: float = 0.50,
) -> dict:
    """
    SCALE V2:
    - Stage 1: chưa có setup
    - Stage 2: chuẩn bị
    - Stage 3: vào lệnh
    """
    base = {
        "symbol": symbol,
        "tf": "M15/H1",
        "direction": "NONE",
        "condition": "CHƯA ĐỦ",
        "stage_num": 1,
        "stage_text": "Chưa có setup",
        "readiness": "LOW",
        "logic_lines": [],
        "orders": [],
        "tp_total_cent": float(total_tp_cent),
        "tp1": None,
        "tp2": None,
        "invalid": None,
        "notes": [],
        "meta": {},
    }

    m15c = _safe_candles(m15)
    m30c = _safe_candles(m30)
    h1c = _safe_candles(h1)
    h4c = _safe_candles(h4)

    if not m15c or not h1c or not h4c:
        base["notes"].append("Thiếu dữ liệu M15/H1/H4")
        return base
    ema_pack = _calc_ema_pack(m15c)
    if ema_pack:
        base["ema"] = ema_pack
    atr15 = _atr(m15c, 14) or 0.0
    h1_trend = _trend_label(h1c)
    h4_trend = _trend_label(h4c)

    # ===== Bias lớn =====
    if h1_trend == "bearish":
        direction = "SELL"
    elif h1_trend == "bullish":
        direction = "BUY"
    else:
        direction = "NONE"

    base["direction"] = direction
    base["logic_lines"].append(f"Bias lớn: {direction if direction != 'NONE' else 'CHƯA RÕ'}")

    # ===== Structure / Sniper =====
    m15_struct = _m15_key_levels(
        m15c,
        bias_side=direction if direction in ("BUY", "SELL") else "BUY",
        lookback=120,
    )

    try:
        sniper = _entry_sniper_v1(
            m15c=m15c,
            m15_struct=m15_struct,
            atr15=atr15,
            volq=_vol_quality(m15c, n=20),
        )
    except Exception:
        sniper = {"direction": "NONE", "strength": "-", "trigger": "NONE", "state": "KHÔNG CÓ SETUP"}

    sniper_dir = str(sniper.get("direction") or "NONE").upper()
    sniper_strength = str(sniper.get("strength") or "-")
    sniper_trigger = str(sniper.get("trigger") or "NONE").upper()

    base["logic_lines"].append(f"Cây chỉ hướng: {sniper_dir} ({sniper_strength})")
    base["logic_lines"].append(f"Điểm nổ: {sniper_trigger if sniper_trigger != 'NONE' else 'NONE'}")

    # ===== Range / position =====
    lo, hi, last_px = _range_levels(m15c[:-1] if len(m15c) > 1 else m15c, n=20)
    range_pos = None
    if lo is not None and hi is not None and hi > lo and last_px is not None:
        range_pos = (last_px - lo) / max(1e-9, hi - lo)

    late_move = False
    if range_pos is not None:
        if range_pos <= 0.15 or range_pos >= 0.85:
            late_move = True

    # ===== Stage V2 =====
    stage_num, stage_text = _scale_stage_v2(
        direction=direction,
        sniper_dir=sniper_dir,
        sniper_trigger=sniper_trigger,
    )

    # nếu đang late move thì vẫn không nên vào, dù có bias
    if late_move and stage_num == 3:
        stage_num, stage_text = 2, "Chuẩn bị"

    base["stage_num"] = stage_num
    base["stage_text"] = stage_text
    base["logic_lines"].append(f"Giai đoạn: {stage_num} | {stage_text}")

    # ===== Readiness =====
    readiness = _scale_readiness_v2(
        direction=direction,
        sniper_dir=sniper_dir,
        sniper_trigger=sniper_trigger,
        late_move=late_move,
    )

    if stage_num == 1:
        readiness = "LOW"

    base["readiness"] = readiness
    base["logic_lines"].append(f"Scale readiness: {readiness}")

    # ===== Condition =====
    condition_ok = (
        stage_num == 3
        and direction in ("BUY", "SELL")
        and sniper_dir == direction
        and sniper_trigger in ("READY", "TRIGGERED")
        and not late_move
    )
    base["condition"] = "ĐỦ" if condition_ok else "CHƯA ĐỦ"

    # ===== Build scale zones =====
    if lo is None or hi is None or last_px is None or direction == "NONE":
        base["notes"].append("Chưa dựng được vùng scale")
        return base

    # ===== Build zones from impulse leg instead of raw range =====
    impulse = _find_last_impulse_leg(m15c, side=direction, lookback=40)

    if not impulse.get("ok"):
        base["notes"].append("Chưa tìm được impulse leg rõ để dựng vùng scale")
        return base

    leg_low = float(impulse["leg_low"])
    leg_high = float(impulse["leg_high"])

    zone_pack = _build_scale_zones_from_impulse(
        current_price=float(last_px),
        side=direction,
        leg_low=leg_low,
        leg_high=leg_high,
        atr15=float(atr15),
        lot1=float(lot1),
        lot2=float(lot2),
        lot3=float(lot3),
    )

    base["orders"] = zone_pack.get("orders") or []
    base["invalid"] = zone_pack.get("invalid")
    base["tp1"] = zone_pack.get("tp1")
    base["tp2"] = zone_pack.get("tp2")

    if not base["orders"]:
        base["notes"].append("Vùng scale hợp lệ đã ở sau lưng giá hiện tại → chờ nhịp mới")
        return base

    # note đúng hướng
    if direction == "SELL":
        base["notes"].append("SELL scale = chờ giá hồi lên vùng rồi mới bán")
        if range_pos is not None and range_pos < 0.20:
            base["notes"].append("Đang ở vùng thấp → không SELL đuổi")
    else:
        base["notes"].append("BUY scale = chờ giá hồi xuống vùng rồi mới mua")
        if range_pos is not None and range_pos > 0.80:
            base["notes"].append("Đang ở vùng cao → không BUY đuổi")

    # ===== Notes V2 =====
    if sniper_dir == "NONE":
        base["notes"].append("Không có cây chỉ hướng → KHÔNG SCALE")

    base["notes"].append("Chỉ scale khi giá hồi vào vùng + có phản ứng rõ")
    base["notes"].append("KHÔNG scale đuổi")
    # ===== PRO DESK SCALE META =====
    meta = base.setdefault("meta", {})

    ntz_reasons = []
    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if rp is not None and 0.35 <= rp <= 0.65:
        ntz_reasons.append("giữa biên độ")

    if stage_num == 1:
        ntz_reasons.append("chưa có setup")

    if sniper_dir == "NONE":
        ntz_reasons.append("chưa có cây chỉ hướng")

    if readiness == "LOW":
        ntz_reasons.append("readiness thấp")

    no_trade_zone_v3 = {
        "active": len(ntz_reasons) >= 2,
        "reasons": ntz_reasons[:4],
    }

    if stage_num == 3 and readiness == "HIGH":
        decision_engine_v1 = {"decision": "MANUAL STRIKE", "reason": "Scale có thể bắt đầu"}
    elif stage_num == 2:
        decision_engine_v1 = {"decision": "WAIT", "reason": "Chuẩn bị vùng scale"}
    else:
        decision_engine_v1 = {"decision": "STAND ASIDE", "reason": "Chưa đủ điều kiện scale"}

    wait_for_v1 = {
        "lines": [
            "Giá phải chạm vùng scale",
            "Có phản ứng rõ (wick / giữ giá)",
            "Entry Sniper phải READY/TRIGGERED",
        ]
    }

    meta["no_trade_zone_v3"] = no_trade_zone_v3
    meta["decision_engine_v1"] = decision_engine_v1
    meta["wait_for_v1"] = wait_for_v1
    return base


def format_scale_plan_v2(plan: dict) -> str:
    symbol = str(plan.get("symbol") or "n/a")
    tf = str(plan.get("tf") or "M15/H1")
    direction = str(plan.get("direction") or "NONE")
    condition = str(plan.get("condition") or "CHƯA ĐỦ")
    readiness = str(plan.get("readiness") or "LOW")
    stage_num = int(plan.get("stage_num") or 1)
    stage_text = str(plan.get("stage_text") or "Chưa có setup")

    lines = []
    lines.append(f"📌 {symbol} SCALE | {tf}")
    lines.append("")
    lines.append(f"🧭 Hướng scale: {direction}")
    lines.append(f"📌 Điều kiện hiện tại: {condition}")
    lines.append("")
    lines.append("🎯 Logic scale:")
    for s in (plan.get("logic_lines") or []):
        lines.append(f"- {s}")

    lines.append("")
    lines.append("📍 Vùng scale (pullback zone):")
    for od in (plan.get("orders") or []):
        lines.append(f"- {od.get('name')}: {_nf2(od.get('zone_lo'))} – {_nf2(od.get('zone_hi'))} | lot {float(od.get('lot') or 0):.2f}")

    lines.append("")
    lines.append("🎯 Mục tiêu:")
    lines.append(f"- TP tổng: +{int(float(plan.get('tp_total_cent') or 0))} cent")
    lines.append("- TP theo cấu trúc M15")
    if plan.get("tp1") is not None:
        lines.append(f"- TP1 tham khảo: {_nf2(plan.get('tp1'))}")
    if plan.get("tp2") is not None:
        lines.append(f"- TP2 tham khảo: {_nf2(plan.get('tp2'))}")
    
    ema_pack = plan.get("ema") or {}
    if ema_pack:
        lines.append("")
        lines.append("📉 EMA FILTER:")
        if ema_pack.get("ema34") is not None:
            lines.append(f"- EMA34: {_nf2(ema_pack.get('ema34'))}")
        if ema_pack.get("ema89") is not None:
            lines.append(f"- EMA89: {_nf2(ema_pack.get('ema89'))}")
        if ema_pack.get("ema200") is not None:
            lines.append(f"- EMA200: {_nf2(ema_pack.get('ema200'))}")
        lines.append(f"- Trend: {ema_pack.get('trend', 'N/A')}")
        lines.append(f"- Alignment: {ema_pack.get('alignment', 'NO')}")
        if ema_pack.get("zone"):
            lines.append(f"- Vị trí giá vs EMA: {ema_pack.get('zone')}")
    lines.append("")
    lines.append("🧯 Invalidation:")
    if direction == "SELL":
        lines.append(f"- Nếu M15 đóng trên {_nf2(plan.get('invalid'))} → bỏ kịch bản SELL")
    elif direction == "BUY":
        lines.append(f"- Nếu M15 đóng dưới {_nf2(plan.get('invalid'))} → bỏ kịch bản BUY")
    else:
        lines.append("- Chưa có invalid rõ")

    notes = plan.get("notes") or []
    if notes:
        lines.append("")
        lines.append("⚠️ Lưu ý:")
        for s in notes[:4]:
            lines.append(f"- {s}")

    lines.append("")
    lines.append("🧠 Kết luận:")
    if stage_num == 1:
        lines.append("- Giai đoạn 1 → chưa có setup")
        lines.append("- KHÔNG vào lệnh")
    elif stage_num == 2:
        lines.append("- Giai đoạn 2 → chỉ chuẩn bị vùng")
        lines.append("- Chưa có điểm vào → KHÔNG vào lệnh")
    else:
        lines.append("- Giai đoạn 3 → có thể bắt đầu scale")
        lines.append(f"- Hiện tại {condition} ĐIỀU KIỆN SCALE")

    # ===== PRO DESK SCALE =====
    try:
        meta = plan.get("meta", {}) or {}
        ntz3 = meta.get("no_trade_zone_v3") or {}
        de1 = meta.get("decision_engine_v1") or {}
        wf1 = meta.get("wait_for_v1") or {}

        lines.append("")
        lines.append("🧠 ===== PRO DESK SCALE =====")

        if ntz3.get("active"):
            lines.append("🚫 SCALE BLOCKED")
            for r in ntz3.get("reasons", [])[:3]:
                lines.append(f"- {r}")

        lines.append("🎯 SCALE DECISION:")
        lines.append(f"- {de1.get('decision', 'WAIT')}")
        if de1.get("reason"):
            lines.append(f"- {de1.get('reason')}")

        if wf1.get("lines"):
            lines.append("⏳ SCALE TRIGGER:")
            for s in wf1.get("lines", [])[:3]:
                lines.append(f"- {s}")
            # ===== SCALE SUGGESTION =====
        lines.append("📌 SCALE SUGGESTION:")
        if ntz3.get("active"):
            lines.append("- NO SCALE")
            lines.append("- Chỉ quan sát vùng scale, chưa được vào")
        else:
            lines.append(f"- {de1.get('decision', 'WAIT')}")
            lines.append("- Chỉ scale khi chạm vùng + có phản ứng rõ")

        # ===== SCALE TRIGGER V2 =====
        tg2 = plan.get("meta", {}).get("trigger_engine_v2", {}) if isinstance(plan.get("meta", {}), dict) else {}
        if tg2:
            lines.append("🎯 SCALE TRIGGER V2:")
            lines.append(f"- State: {tg2.get('state', 'WAIT')}")
            lines.append(f"- Quality: {tg2.get('quality', 'LOW')}")
            if tg2.get("reason"):
                for s in tg2.get("reason", [])[:3]:
                    lines.append(f"- {s}")

        # ===== MASTER ENGINE SCALE =====
        try:
            meta = plan.get("meta", {}) or {}
            me1 = meta.get("master_engine_v1") or {}
            if me1:
                lines.append("🧠 MASTER ENGINE:")
                lines.append(f"- State: {me1.get('state', 'WAIT')}")
                lines.append(f"- Tradeable final: {'YES' if me1.get('tradeable_final') else 'NO'}")
                lines.append(f"- Confidence: {me1.get('confidence', 'LOW')}")
        except Exception:
            pass
    except Exception:
        pass
    return "\n".join(lines).strip()

# =========================
# VNEXT ADD-ON (append-only)
# Context / RSI / Fib / Trap / Liquidity completion / Manual likelihood / Manual guidance
# =========================

def _context_verdict_v1(
    bias_side: str | None,
    h1_trend: str,
    h4_trend: str,
    market_state_v2: str,
    flow_state: dict | None,
    range_pos: float | None,
    no_trade_zone: dict | None,
    liquidation_evt: dict | None,
    m15_struct_tag: str | None,
) -> dict:
    out = {
        "verdict": "CHƯA CÓ CẢNH",
        "state": "NEUTRAL",
        "reason": [],
    }

    flow_side = str((flow_state or {}).get("favored_side") or "NONE").upper()
    tag = str(m15_struct_tag or "").upper()
    ntz = bool((no_trade_zone or {}).get("active"))
    liq = bool((liquidation_evt or {}).get("ok"))

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if (
        bias_side == "SELL"
        and h1_trend == "bearish"
        and h4_trend == "bearish"
        and flow_side in ("SELL", "NONE")
        and not ntz
    ):
        if rp is not None and rp >= 0.45:
            out["verdict"] = "ĐÚNG CẢNH SELL"
            out["state"] = "SELL_CONTINUATION"
            out["reason"] = ["khung lớn giảm", "đang hồi trong xu hướng giảm"]
            return out

    if (
        bias_side == "BUY"
        and h1_trend == "bullish"
        and h4_trend == "bullish"
        and flow_side in ("BUY", "NONE")
        and not ntz
    ):
        if rp is not None and rp <= 0.55:
            out["verdict"] = "ĐÚNG CẢNH BUY"
            out["state"] = "BUY_CONTINUATION"
            out["reason"] = ["khung lớn tăng", "đang điều chỉnh trong xu hướng tăng"]
            return out

    if bias_side == "BUY" and h1_trend == "bearish":
        out["verdict"] = "SAI CẢNH BUY"
        out["state"] = "COUNTERTREND_BUY"
        out["reason"] = ["đang ngược H1", "dễ bắt đáy hỏng"]
        return out

    if bias_side == "SELL" and h1_trend == "bullish":
        out["verdict"] = "SAI CẢNH SELL"
        out["state"] = "COUNTERTREND_SELL"
        out["reason"] = ["đang ngược H1", "dễ bắt đỉnh hỏng"]
        return out

    if liq:
        out["verdict"] = "CHỜ SAU LIQUIDATION"
        out["state"] = "POST_LIQUIDATION"
        out["reason"] = ["vừa có quét mạnh", "chưa nên follow ngay"]
        return out

    if ntz or market_state_v2 in ("CHOP", "TRANSITION"):
        out["verdict"] = "CHƯA CÓ CẢNH"
        out["state"] = "NO_EDGE"
        out["reason"] = ["market nhiễu", "chưa có edge rõ"]
        return out

    if "LH" in tag or "LL" in tag:
        out["verdict"] = "NGHIÊNG CẢNH SELL"
        out["state"] = "SELL_BIAS"
        out["reason"] = ["M15 đang yếu", "thiên về hướng giảm"]
        return out

    if "HL" in tag or "HH" in tag:
        out["verdict"] = "NGHIÊNG CẢNH BUY"
        out["state"] = "BUY_BIAS"
        out["reason"] = ["M15 đang khỏe", "thiên về hướng tăng"]
        return out

    return out


def _rsi_context_v1(
    rsi15: float | None,
    bias_side: str | None,
    h1_trend: str,
    market_state_v2: str,
    div: dict | None,
    liquidation_evt: dict | None,
) -> dict:
    out = {
        "state": "NEUTRAL",
        "message": "RSI trung tính",
        "meaning": "RSI chỉ là nhịp của giá, chưa nói lên điểm vào",
    }

    rsi = float(rsi15 or 50.0)
    bear_div = bool((div or {}).get("bear"))
    bull_div = bool((div or {}).get("bull"))
    just_liq = bool((liquidation_evt or {}).get("ok"))

    if rsi >= 75:
        if h1_trend == "bullish" and not bear_div and not just_liq:
            out["state"] = "ACCELERATION_UP"
            out["message"] = "RSI cao = đà tăng mạnh, chưa phải tín hiệu SELL"
            out["meaning"] = "thị trường đang tăng tốc"
            return out
        if bear_div or market_state_v2 in ("EXHAUSTION_UP", "POST_SHORT_COVER"):
            out["state"] = "EXHAUSTION_UP"
            out["message"] = "RSI cao + dấu hiệu kiệt sức"
            out["meaning"] = "cẩn thận mua đuổi / dễ bị phân phối"
            return out

    if rsi <= 25:
        if h1_trend == "bearish" and not bull_div and not just_liq:
            out["state"] = "ACCELERATION_DOWN"
            out["message"] = "RSI thấp = đà giảm mạnh, chưa phải tín hiệu BUY"
            out["meaning"] = "thị trường đang giảm tốc mạnh theo xu hướng"
            return out
        if bull_div or market_state_v2 in ("EXHAUSTION_DOWN", "POST_LIQUIDATION_BOUNCE"):
            out["state"] = "EXHAUSTION_DOWN"
            out["message"] = "RSI thấp + dấu hiệu kiệt sức"
            out["meaning"] = "cẩn thận bán đuổi / dễ bật hồi"
            return out

    if bear_div:
        out["state"] = "BEARISH_DIVERGENCE"
        out["message"] = "RSI đang yếu đi so với giá"
        out["meaning"] = "đà tăng không còn sạch"
        return out

    if bull_div:
        out["state"] = "BULLISH_DIVERGENCE"
        out["message"] = "RSI đang mạnh lên so với giá"
        out["meaning"] = "đà giảm đang yếu dần"
        return out

    return out


def _fib_confluence_v1(
    m15c,
    bias_side: str | None,
    atr15: float | None,
    liquidity_map_v1: dict | None,
    ema_pack: dict | None,
    playbook_v2: dict | None,
) -> dict:
    out = {
        "ok": False,
        "level": None,
        "zone_low": None,
        "zone_high": None,
        "score": 0,
        "reason": [],
    }

    if not m15c or len(m15c) < 20 or bias_side not in ("BUY", "SELL"):
        return out

    impulse = _find_last_impulse_leg(m15c, side=bias_side, lookback=40)
    if not impulse.get("ok"):
        return out

    leg_low = float(impulse["leg_low"])
    leg_high = float(impulse["leg_high"])
    rng = max(1e-9, leg_high - leg_low)
    pad = max(1e-9, float(atr15 or 0.0) * 0.12)

    if bias_side == "BUY":
        fib50 = leg_high - 0.500 * rng
        fib618 = leg_high - 0.618 * rng
        fib705 = leg_high - 0.705 * rng
    else:
        fib50 = leg_low + 0.500 * rng
        fib618 = leg_low + 0.618 * rng
        fib705 = leg_low + 0.705 * rng

    zone_low = min(fib50, fib618, fib705) - pad
    zone_high = max(fib50, fib618, fib705) + pad

    score = 0
    reason = []

    if ema_pack:
        e34 = _safe_float(ema_pack.get("ema34"))
        e89 = _safe_float(ema_pack.get("ema89"))
        if e34 is not None and zone_low <= e34 <= zone_high:
            score += 1
            reason.append("fib trùng EMA34")
        if e89 is not None and zone_low <= e89 <= zone_high:
            score += 1
            reason.append("fib trùng EMA89")

    pz_lo = _safe_float((playbook_v2 or {}).get("zone_low"))
    pz_hi = _safe_float((playbook_v2 or {}).get("zone_high"))
    if pz_lo is not None and pz_hi is not None:
        if not (pz_hi < zone_low or pz_lo > zone_high):
            score += 1
            reason.append("fib trùng vùng playbook")

    liq_zone = None
    if bias_side == "BUY":
        liq_zone = (liquidity_map_v1 or {}).get("below_zone")
    else:
        liq_zone = (liquidity_map_v1 or {}).get("above_zone")

    if isinstance(liq_zone, (list, tuple)) and len(liq_zone) == 2:
        l1, l2 = float(liq_zone[0]), float(liq_zone[1])
        if not (l2 < zone_low or l1 > zone_high):
            score += 1
            reason.append("fib trùng liquidity zone")

    best = fib618
    out.update({
        "ok": score >= 2,
        "level": best,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "score": score,
        "reason": reason[:4],
    })
    return out
    
def _pullback_engine_v1(
    bias_side: str | None,
    current_price: float | None,
    key_levels: dict | None,
    ema_pack: dict | None,
    h1_trend: str | None,
    h4_trend: str | None,
    m15_struct_tag: str | None,
    close_confirm_v4: dict | None,
    liquidity_completion_v1: dict | None,
    trap_warning_v1: dict | None,
    atr15: float | None = None,
) -> dict:
    """
    Pullback engine:
    - Đo % hồi trong xu hướng
    - Phân biệt: hồi nông / hồi đẹp / hồi sâu / nguy cơ đảo chiều
    - Không tạo lệnh, chỉ phục vụ phân tích + guidance

    BUY context:
        anchor_low  = vùng giữ xu hướng
        anchor_high = đỉnh cấu trúc / swing high gần
        pullback_pct = (anchor_high - current) / (anchor_high - anchor_low)

    SELL context:
        anchor_low  = đáy cấu trúc / swing low gần
        anchor_high = vùng giữ xu hướng giảm
        pullback_pct = (current - anchor_low) / (anchor_high - anchor_low)
    """
    out = {
        "ok": False,
        "side": str(bias_side or "NONE").upper(),
        "anchor_low": None,
        "anchor_high": None,
        "pullback_pct": None,      # 0..1
        "pullback_pct_text": "n/a",
        "stage": "NONE",
        "label": "CHƯA RÕ",
        "action": "WAIT",
        "reversal_risk": "LOW",
        "enough_for_entry": False,
        "reason": [],
        "message": "Chưa đủ dữ liệu để đánh giá pullback",
    }

    side = str(bias_side or "NONE").upper()
    if side not in ("BUY", "SELL"):
        return out

    cp = _safe_float(current_price)
    if cp is None:
        return out

    k = key_levels or {}
    ema = ema_pack or {}
    cc = close_confirm_v4 or {}
    ld = liquidity_completion_v1 or {}
    tw = trap_warning_v1 or {}

    h1 = str(h1_trend or "").lower()
    h4 = str(h4_trend or "").lower()
    m15_tag = str(m15_struct_tag or "").upper()

    ema34 = _safe_float(ema.get("ema34"))
    ema89 = _safe_float(ema.get("ema89"))
    ema200 = _safe_float(ema.get("ema200"))

    reasons = []

    # ===== Anchor selection =====
    if side == "BUY":
        # Ưu tiên low = H1_HL / M15_PB_EXT / M15_RANGE_LOW
        anchor_low = (
            _safe_float(k.get("H1_HL"))
            or _safe_float(k.get("M15_PB_EXT"))
            or _safe_float(k.get("M15_RANGE_LOW"))
        )
        # Ưu tiên high = H1_HH / M15_RANGE_HIGH
        anchor_high = (
            _safe_float(k.get("H1_HH"))
            or _safe_float(k.get("M15_RANGE_HIGH"))
        )
    else:
        # SELL
        anchor_low = (
            _safe_float(k.get("H1_LL"))
            or _safe_float(k.get("M15_RANGE_LOW"))
        )
        anchor_high = (
            _safe_float(k.get("H1_LH"))
            or _safe_float(k.get("M15_PB_EXT"))
            or _safe_float(k.get("M15_RANGE_HIGH"))
        )

    if anchor_low is None or anchor_high is None or anchor_high <= anchor_low:
        return out

    span = max(1e-9, anchor_high - anchor_low)

    # ===== Pullback % =====
    if side == "BUY":
        pullback_pct = (anchor_high - cp) / span
    else:
        pullback_pct = (cp - anchor_low) / span

    pullback_pct = _clamp(pullback_pct, 0.0, 1.5)

    # ===== Base stage =====
    if pullback_pct < 0.25:
        stage = "SHALLOW"
        label = "Hồi nông"
        action = "WAIT"
        enough_for_entry = False
        reasons.append("độ hồi còn nông")
    elif pullback_pct < 0.40:
        stage = "EARLY_OK"
        label = "Hồi sớm"
        action = "WAIT_CONFIRM"
        enough_for_entry = False
        reasons.append("đã bắt đầu hồi nhưng chưa đủ sâu")
    elif pullback_pct <= 0.62:
        stage = "HEALTHY"
        label = "Hồi đẹp"
        action = "LOOK_FOR_TRIGGER"
        enough_for_entry = True
        reasons.append("độ hồi nằm trong vùng đẹp")
    elif pullback_pct <= 0.78:
        stage = "DEEP"
        label = "Hồi sâu"
        action = "CAREFUL"
        enough_for_entry = True
        reasons.append("độ hồi khá sâu")
    else:
        stage = "EXTREME"
        label = "Hồi quá sâu"
        action = "AVOID_OR_WAIT"
        enough_for_entry = False
        reasons.append("độ hồi quá sâu")

    reversal_risk_score = 0

    # ===== HTF context =====
    if side == "BUY":
        if h1 == "bullish" and h4 == "bullish":
            reasons.append("khung lớn vẫn tăng")
        else:
            reversal_risk_score += 2
            reasons.append("khung lớn không đồng thuận BUY")
    else:
        if h1 == "bearish" and h4 == "bearish":
            reasons.append("khung lớn vẫn giảm")
        else:
            reversal_risk_score += 2
            reasons.append("khung lớn không đồng thuận SELL")

    # ===== EMA filter =====
    if side == "BUY":
        if ema34 is not None and cp >= ema34:
            reasons.append("giá vẫn giữ trên EMA34")
        elif ema89 is not None and cp >= ema89:
            reasons.append("giá đã hồi về EMA89")
        elif ema200 is not None and cp < ema200:
            reversal_risk_score += 3
            reasons.append("giá thủng EMA200")
    else:
        if ema34 is not None and cp <= ema34:
            reasons.append("giá vẫn dưới EMA34")
        elif ema89 is not None and cp <= ema89:
            reasons.append("giá hồi lên EMA89")
        elif ema200 is not None and cp > ema200:
            reversal_risk_score += 3
            reasons.append("giá vượt EMA200")

    # ===== M15 structure =====
    if side == "BUY":
        if "LL" in m15_tag or "LH" in m15_tag:
            reversal_risk_score += 2
            reasons.append("M15 đang nghiêng yếu cho BUY")
        elif "HL" in m15_tag or "HH" in m15_tag:
            reasons.append("M15 chưa phá cấu trúc tăng")
        elif "TRANSITION" in m15_tag:
            reversal_risk_score += 1
            reasons.append("M15 đang chuyển pha")
    else:
        if "HH" in m15_tag or "HL" in m15_tag:
            reversal_risk_score += 2
            reasons.append("M15 đang nghiêng yếu cho SELL")
        elif "LH" in m15_tag or "LL" in m15_tag:
            reasons.append("M15 chưa phá cấu trúc giảm")
        elif "TRANSITION" in m15_tag:
            reversal_risk_score += 1
            reasons.append("M15 đang chuyển pha")

    # ===== confirm / liquidity =====
    cc_strength = str(cc.get("strength") or "NO").upper()
    if cc_strength in ("NO", "N/A"):
        reversal_risk_score += 1
        reasons.append("chưa có close confirmation")
    else:
        reasons.append(f"đã có close confirm {cc_strength}")

    liq_state = str(ld.get("state") or "NO").upper()
    if liq_state == "YES":
        reasons.append("thanh khoản đã hoàn tất")
    elif liq_state == "PARTIAL":
        reversal_risk_score += 0.5   # 👈 giảm nhẹ
        reasons.append("thanh khoản mới hoàn tất một phần")
    else:
        # ❌ KHÔNG tăng risk mạnh nữa nếu HTF đang đẹp
        if not (h1 == "bullish" and h4 == "bullish" and side == "BUY"):
            reversal_risk_score += 1
        reasons.append("thanh khoản chưa hoàn tất")

    if (tw or {}).get("active"):
        # chỉ tăng mạnh nếu đã hồi sâu
        if pullback_pct > 0.6:
            reversal_risk_score += 1
        else:
            reversal_risk_score += 0.5
        reasons.append("trap risk đang hiện diện")
        
    # ===== ADJUST FOR SHALLOW PULLBACK =====
    if side == "BUY" and pullback_pct < 0.25:
        # hồi nông → không nên coi là reversal risk
        reversal_risk_score = min(reversal_risk_score, 2)
        
    # ===== ATR sanity =====
    a = _safe_float(atr15)
    if a is not None and a > 0:
        # check distance to invalid side
        if side == "BUY" and anchor_low is not None:
            if (cp - anchor_low) <= 0.25 * a:
                reversal_risk_score += 1
                reasons.append("giá đang quá gần vùng invalid BUY")
        if side == "SELL" and anchor_high is not None:
            if (anchor_high - cp) <= 0.25 * a:
                reversal_risk_score += 1
                reasons.append("giá đang quá gần vùng invalid SELL")

    # ===== final reversal risk =====
    if reversal_risk_score >= 6:
        reversal_risk = "HIGH"
    elif reversal_risk_score >= 3:
        reversal_risk = "MEDIUM"
    else:
        reversal_risk = "LOW"

    # ===== tighten enough_for_entry =====
    if enough_for_entry:
        if reversal_risk == "HIGH":
            enough_for_entry = False
        if cc_strength in ("NO", "N/A") and liq_state == "NO":
            enough_for_entry = False

    # ===== final message =====
    if side == "BUY":
        if enough_for_entry and stage == "HEALTHY":
            message = "Hồi đẹp trong xu hướng tăng; có thể chờ trigger BUY"
        elif enough_for_entry and stage == "DEEP":
            message = "Hồi sâu nhưng vẫn còn giữ logic tăng; chỉ BUY khi có xác nhận mạnh"
        elif stage == "SHALLOW":
            message = "Hồi còn nông; chưa đẹp để BUY ngay"
        elif stage == "EXTREME":
            message = "Hồi quá sâu; nguy cơ không còn là pullback sạch"
        else:
            message = "Đang hồi trong xu hướng tăng nhưng chưa đủ điều kiện vào"
    else:
        if enough_for_entry and stage == "HEALTHY":
            message = "Hồi đẹp trong xu hướng giảm; có thể chờ trigger SELL"
        elif enough_for_entry and stage == "DEEP":
            message = "Hồi sâu nhưng vẫn còn giữ logic giảm; chỉ SELL khi có xác nhận mạnh"
        elif stage == "SHALLOW":
            message = "Hồi còn nông; chưa đẹp để SELL ngay"
        elif stage == "EXTREME":
            message = "Hồi quá sâu; nguy cơ không còn là pullback sạch"
        else:
            message = "Đang hồi trong xu hướng giảm nhưng chưa đủ điều kiện vào"

    out.update({
        "ok": True,
        "anchor_low": anchor_low,
        "anchor_high": anchor_high,
        "pullback_pct": pullback_pct,
        "pullback_pct_text": f"{int(round(pullback_pct * 100))}%",
        "stage": stage,
        "label": label,
        "action": action,
        "reversal_risk": reversal_risk,
        "enough_for_entry": bool(enough_for_entry),
        "reason": reasons[:6],
        "message": message,
    })
    return out

def _trap_warning_v1(
    bias_side: str | None,
    context_verdict: dict | None,
    rsi_ctx: dict | None,
    no_trade_zone: dict | None,
    liquidation_evt: dict | None,
    range_pos: float | None,
    div: dict | None,
    close_confirm_v4: dict | None,
) -> dict:
    warns = []

    verdict = str((context_verdict or {}).get("state") or "")
    rsi_state = str((rsi_ctx or {}).get("state") or "")

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if verdict == "COUNTERTREND_BUY":
        warns.append("BUY đang ngược xu hướng chính → dễ bắt đáy hỏng")
    if verdict == "COUNTERTREND_SELL":
        warns.append("SELL đang ngược xu hướng chính → dễ bắt đỉnh hỏng")

    if (no_trade_zone or {}).get("active"):
        warns.append("market đang ở no-trade zone → edge thấp")

    if (liquidation_evt or {}).get("ok"):
        warns.append("vừa có liquidation → dễ quét 2 đầu / hồi giả")

    if rsi_state == "ACCELERATION_UP" and bias_side == "SELL":
        warns.append("RSI đang tăng tốc chứ chưa kiệt → SELL dễ chết")
    if rsi_state == "ACCELERATION_DOWN" and bias_side == "BUY":
        warns.append("RSI đang giảm mạnh chứ chưa cạn → BUY dễ chết")

    if bias_side == "BUY" and (div or {}).get("bear"):
        warns.append("bearish divergence đang chống BUY")
    if bias_side == "SELL" and (div or {}).get("bull"):
        warns.append("bullish divergence đang chống SELL")

    if rp is not None:
        if bias_side == "BUY" and rp >= 0.85:
            warns.append("BUY vùng cao → dễ mua đuổi")
        if bias_side == "SELL" and rp <= 0.15:
            warns.append("SELL vùng thấp → dễ bán đuổi")

    if str((close_confirm_v4 or {}).get("strength") or "").upper() in ("NO", "N/A"):
        warns.append("chưa có close confirmation rõ")

    return {
        "active": len(warns) > 0,
        "warnings": warns[:6],
    }


def _liquidity_completion_v1(
    sweep_buy: dict | None,
    sweep_sell: dict | None,
    spring_buy: dict | None,
    spring_sell: dict | None,
    close_confirm_v4: dict | None,
    entry_sniper: dict | None,
    bias_side: str | None,
) -> dict:
    out = {
        "state": "NO",
        "message": "Thanh khoản chưa hoàn tất",
    }

    sb = bool((sweep_buy or {}).get("ok"))
    ss = bool((sweep_sell or {}).get("ok"))
    spb = bool((spring_buy or {}).get("ok"))
    sps = bool((spring_sell or {}).get("ok"))

    confirm = str((close_confirm_v4 or {}).get("strength") or "NO").upper()
    sniper_trigger = str((entry_sniper or {}).get("trigger") or "NONE").upper()

    if bias_side == "BUY":
        if (sb or spb) and confirm in ("WEAK", "STRONG"):
            out["state"] = "YES"
            out["message"] = "Đã quét đáy + có xác nhận giữ giá"
            return out
        if (sb or spb) and sniper_trigger in ("READY", "TRIGGERED"):
            out["state"] = "PARTIAL"
            out["message"] = "Đã quét đáy nhưng xác nhận chưa hoàn tất"
            return out

    if bias_side == "SELL":
        if (ss or sps) and confirm in ("WEAK", "STRONG"):
            out["state"] = "YES"
            out["message"] = "Đã quét đỉnh + có xác nhận đạp xuống"
            return out
        if (ss or sps) and sniper_trigger in ("READY", "TRIGGERED"):
            out["state"] = "PARTIAL"
            out["message"] = "Đã quét đỉnh nhưng xác nhận chưa hoàn tất"
            return out

    return out


def _manual_likelihood_v1(
    bias_side: str | None,
    context_verdict: dict | None,
    trap_warning: dict | None,
    fib_conf: dict | None,
    liq_done: dict | None,
    close_confirm_v4: dict | None,
    entry_sniper: dict | None,
    playbook_v4: dict | None,
) -> dict:
    buy_score = 35
    sell_score = 35
    trap_risk = 20

    verdict = str((context_verdict or {}).get("state") or "")
    liq_state = str((liq_done or {}).get("state") or "NO")
    fib_ok = bool((fib_conf or {}).get("ok"))
    confirm = str((close_confirm_v4 or {}).get("strength") or "NO").upper()
    sniper_dir = str((entry_sniper or {}).get("direction") or "NONE").upper()
    sniper_trigger = str((entry_sniper or {}).get("trigger") or "NONE").upper()
    pbq = str((playbook_v4 or {}).get("quality") or "LOW").upper()

    if verdict == "BUY_CONTINUATION":
        buy_score += 20
    if verdict == "SELL_CONTINUATION":
        sell_score += 20
    if verdict == "COUNTERTREND_BUY":
        buy_score -= 15
        trap_risk += 20
    if verdict == "COUNTERTREND_SELL":
        sell_score -= 15
        trap_risk += 20

    if liq_state == "YES":
        if bias_side == "BUY":
            buy_score += 12
        elif bias_side == "SELL":
            sell_score += 12
    elif liq_state == "PARTIAL":
        if bias_side == "BUY":
            buy_score += 5
        elif bias_side == "SELL":
            sell_score += 5

    if fib_ok:
        if bias_side == "BUY":
            buy_score += 8
        elif bias_side == "SELL":
            sell_score += 8

    if confirm == "STRONG":
        if bias_side == "BUY":
            buy_score += 10
        elif bias_side == "SELL":
            sell_score += 10
    elif confirm == "WEAK":
        if bias_side == "BUY":
            buy_score += 4
        elif bias_side == "SELL":
            sell_score += 4

    if sniper_dir == bias_side and sniper_trigger in ("READY", "TRIGGERED"):
        if bias_side == "BUY":
            buy_score += 10
        elif bias_side == "SELL":
            sell_score += 10

    if pbq == "HIGH":
        if bias_side == "BUY":
            buy_score += 6
        elif bias_side == "SELL":
            sell_score += 6

    if (trap_warning or {}).get("active"):
        trap_risk += 18

    buy_score = max(0, min(100, buy_score))
    sell_score = max(0, min(100, sell_score))
    trap_risk = max(0, min(100, trap_risk))

    return {
        "buy_likelihood": buy_score,
        "sell_likelihood": sell_score,
        "trap_risk": trap_risk,
        "best_side": "BUY" if buy_score > sell_score else ("SELL" if sell_score > buy_score else "NONE"),
    }


def _manual_guidance_v1(
    bias_side: str | None,
    context_verdict: dict | None,
    liq_done: dict | None,
    fib_conf: dict | None,
    close_confirm_v4: dict | None,
    entry_sniper: dict | None,
    playbook_v2: dict | None,
) -> dict:
    lines = []

    verdict = str((context_verdict or {}).get("verdict") or "")
    liq_state = str((liq_done or {}).get("state") or "NO")
    confirm = str((close_confirm_v4 or {}).get("strength") or "NO").upper()
    sniper_trigger = str((entry_sniper or {}).get("trigger") or "NONE").upper()

    lines.append(f"Cảnh hiện tại: {verdict}")

    if liq_state == "NO":
        lines.append("Chờ thị trường hút thanh khoản xong rồi đánh giá tiếp.")
    elif liq_state == "PARTIAL":
        lines.append("Đã có quét nhưng chưa hoàn tất → chờ close confirm / BOS rõ hơn.")
    else:
        lines.append("Liquidity đã tương đối hoàn tất, chỉ còn chờ xác nhận cuối.")

    if fib_conf.get("ok"):
        lines.append(
            f"Vùng Fib đáng chú ý: {_fmt(fib_conf.get('zone_low'))} – {_fmt(fib_conf.get('zone_high'))}"
        )

    if confirm in ("NO", "N/A"):
        lines.append("Chờ nến đóng xác nhận rõ hơn, tránh vào vì cảm giác.")
    elif confirm == "WEAK":
        lines.append("Đã có xác nhận yếu, nên quan sát thêm follow-through.")
    elif confirm == "STRONG":
        lines.append("Đã có close confirm mạnh hơn, có thể theo dõi sát để tự trade tay.")

    if sniper_trigger == "READY":
        lines.append("Setup đang ở trạng thái READY: chỉ chờ điểm nổ.")
    elif sniper_trigger == "TRIGGERED":
        lines.append("Setup đã TRIGGERED: trader tay có thể tự đánh giá để bấm.")
    else:
        lines.append("Chưa có điểm nổ rõ.")

    return {"lines": lines[:6]}
# =========================
# PROBE ENGINE V1 (append-only)
# =========================

def _probe_zone_strength_label(v: int) -> str:
    if v >= 4:
        return "HIGH"
    if v >= 2:
        return "MEDIUM"
    return "LOW"


def _probe_pick_side(sig: dict, bias_side: str | None) -> str:
    meta = (sig.get("meta") or {})
    sce1 = meta.get("signal_consistency_v1") or {}
    side = str(sce1.get("final_side") or bias_side or sig.get("recommendation") or "NONE").upper()
    if side in ("BUY", "SELL"):
        return side
    return "NONE"


def _probe_detect_zone_v1(
    sig: dict,
    bias_side: str | None,
    current_price: float | None,
    range_pos: float | None,
    atr15: float | None,
) -> dict:
    """
    Tìm 5 zone dò:
    1) liquidity
    2) break & retest
    3) pullback
    4) EMA cluster
    5) range boundary

    Trả về best zone duy nhất theo priority.
    """
    meta = (sig.get("meta") or {})
    side = _probe_pick_side(sig, bias_side)
    if side not in ("BUY", "SELL"):
        return {"ok": False, "reason": "no_side"}

    cp = _safe_float(current_price)
    a = float(atr15 or 0.0)
    if cp is None:
        cp = _safe_float(sig.get("last_price")) or _safe_float(sig.get("current_price")) or _safe_float(sig.get("entry"))
    if cp is None:
        return {"ok": False, "reason": "no_price"}

    liq = meta.get("liquidity_map_v1") or {}
    pb1 = meta.get("pullback_engine_v1") or {}
    ema = meta.get("ema") or {}
    k = meta.get("key_levels") or {}
    ccv4 = meta.get("close_confirm_v4") or {}
    cont1 = meta.get("post_break_continuity_v1") or {}
    fib1 = meta.get("fib_confluence_v1") or {}

    candidates = []

    # -------------------------
    # 1) LIQUIDITY ZONE
    # -------------------------
    above_strength = str(liq.get("above_strength") or "LOW").upper()
    below_strength = str(liq.get("below_strength") or "LOW").upper()
    above_zone = liq.get("above_zone")
    below_zone = liq.get("below_zone")
    sweep_bias = str(liq.get("sweep_bias") or "").upper()

    if side == "BUY":
        z = below_zone if isinstance(below_zone, (list, tuple)) and len(below_zone) == 2 else None
        strong_enough = below_strength in ("MEDIUM", "HIGH") or "DOWN → UP" in sweep_bias or bool(liq.get("equal_lows"))
        if z and strong_enough:
            zlo, zhi = float(z[0]), float(z[1])
            candidates.append({
                "ok": True,
                "type": "LIQUIDITY",
                "priority": 1,
                "strength": below_strength,
                "side": "BUY",
                "zone_low": min(zlo, zhi),
                "zone_high": max(zlo, zhi),
                "reason": ["liquidity pool phía dưới", "equal lows / sweep bias hỗ trợ BUY"],
            })
    elif side == "SELL":
        z = above_zone if isinstance(above_zone, (list, tuple)) and len(above_zone) == 2 else None
        strong_enough = above_strength in ("MEDIUM", "HIGH") or "UP → DOWN" in sweep_bias or bool(liq.get("equal_highs"))
        if z and strong_enough:
            zlo, zhi = float(z[0]), float(z[1])
            candidates.append({
                "ok": True,
                "type": "LIQUIDITY",
                "priority": 1,
                "strength": above_strength,
                "side": "SELL",
                "zone_low": min(zlo, zhi),
                "zone_high": max(zlo, zhi),
                "reason": ["liquidity pool phía trên", "equal highs / sweep bias hỗ trợ SELL"],
            })

    # -------------------------
    # 2) BREAK & RETEST
    # -------------------------
    bos = _safe_float(k.get("M15_BOS"))
    hold = str(ccv4.get("hold") or "NO").upper()
    cont_state = str(cont1.get("state") or "").upper()
    if bos is not None and a > 0:
        near_bos = abs(cp - bos) <= max(1e-9, 0.22 * a)
        if near_bos:
            if side == "BUY" and cont_state in ("POST_BREAK_HOLD", "POST_BREAK_CHOP", "POST_BREAK_FAIL", "NONE"):
                candidates.append({
                    "ok": True,
                    "type": "BREAK_RETEST",
                    "priority": 2,
                    "strength": "MEDIUM" if hold == "YES" else "LOW",
                    "side": "BUY",
                    "zone_low": bos - 0.18 * a,
                    "zone_high": bos + 0.18 * a,
                    "reason": ["đang retest BOS", "chờ xem giữ được mốc hay không"],
                })
            if side == "SELL" and cont_state in ("POST_BREAK_HOLD", "POST_BREAK_CHOP", "POST_BREAK_FAIL", "NONE"):
                candidates.append({
                    "ok": True,
                    "type": "BREAK_RETEST",
                    "priority": 2,
                    "strength": "MEDIUM" if hold == "YES" else "LOW",
                    "side": "SELL",
                    "zone_low": bos - 0.18 * a,
                    "zone_high": bos + 0.18 * a,
                    "reason": ["đang retest BOS", "chờ xem mất mốc hay không"],
                })

    # -------------------------
    # 3) PULLBACK ZONE
    # -------------------------
    if pb1.get("ok"):
        pct = _safe_float(pb1.get("pullback_pct")) or 0.0
        a_lo = _safe_float(pb1.get("anchor_low"))
        a_hi = _safe_float(pb1.get("anchor_high"))
        if a_lo is not None and a_hi is not None and 0.25 <= pct <= 0.78:
            lo = min(a_lo, a_hi)
            hi = max(a_lo, a_hi)
            rng = max(1e-9, hi - lo)
            if side == "BUY":
                zlo = hi - 0.62 * rng
                zhi = hi - 0.40 * rng
            else:
                zlo = lo + 0.40 * rng
                zhi = lo + 0.62 * rng
            candidates.append({
                "ok": True,
                "type": "PULLBACK",
                "priority": 3,
                "strength": "HIGH" if 0.40 <= pct <= 0.62 else "MEDIUM",
                "side": side,
                "zone_low": min(zlo, zhi),
                "zone_high": max(zlo, zhi),
                "reason": [f"pullback {pb1.get('pullback_pct_text', 'n/a')}", pb1.get("label", "pullback")],
            })

    # -------------------------
    # 4) EMA CLUSTER
    # -------------------------
    e34 = _safe_float(ema.get("ema34"))
    e89 = _safe_float(ema.get("ema89"))
    e200 = _safe_float(ema.get("ema200"))
    if a > 0:
        ema_candidates = [x for x in [e34, e89] if x is not None]
        if len(ema_candidates) >= 2:
            zlo = min(ema_candidates) - 0.10 * a
            zhi = max(ema_candidates) + 0.10 * a
            if zlo <= cp <= zhi or abs(cp - zlo) <= 0.20 * a or abs(cp - zhi) <= 0.20 * a:
                candidates.append({
                    "ok": True,
                    "type": "EMA_CLUSTER",
                    "priority": 4,
                    "strength": "MEDIUM",
                    "side": side,
                    "zone_low": zlo,
                    "zone_high": zhi,
                    "reason": ["giá đang gần EMA34/89", "cụm EMA có thể là vùng phản ứng"],
                })
        elif e200 is not None and abs(cp - e200) <= 0.20 * a:
            candidates.append({
                "ok": True,
                "type": "EMA_CLUSTER",
                "priority": 4,
                "strength": "LOW",
                "side": side,
                "zone_low": e200 - 0.10 * a,
                "zone_high": e200 + 0.10 * a,
                "reason": ["giá đang touch EMA200", "case đặc biệt"],
            })

    # 5) RANGE BOUNDARY
    # -------------------------
    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None
    
    # FIX: đảm bảo k luôn tồn tại trước khi dùng
    try:
        meta = sig.get("meta") or {}
    except Exception:
        meta = {}
    
    k = meta.get("key_levels") or {}
    rlo = _safe_float(k.get("M15_RANGE_LOW"))
    rhi = _safe_float(k.get("M15_RANGE_HIGH"))
    if rp is not None and rlo is not None and rhi is not None and a > 0:
        if rp <= 0.15 and side == "BUY":
            candidates.append({
                "ok": True,
                "type": "RANGE_BOUNDARY",
                "priority": 5,
                "strength": "MEDIUM",
                "side": "BUY",
                "zone_low": rlo - 0.08 * a,
                "zone_high": rlo + 0.12 * a,
                "reason": ["giá đang sát biên dưới range", "probe BUY tại low boundary"],
            })
        elif rp >= 0.85 and side == "SELL":
            candidates.append({
                "ok": True,
                "type": "RANGE_BOUNDARY",
                "priority": 5,
                "strength": "MEDIUM",
                "side": "SELL",
                "zone_low": rhi - 0.12 * a,
                "zone_high": rhi + 0.08 * a,
                "reason": ["giá đang sát biên trên range", "probe SELL tại high boundary"],
            })

    # pick best by priority first, then strength
    if not candidates:
        return {"ok": False, "reason": "no_probe_zone"}

    strength_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    best = sorted(
        candidates,
        key=lambda x: (int(x.get("priority", 99)), -strength_rank.get(str(x.get("strength", "LOW")).upper(), 1))
    )[0]
    return best




def _probe_fallback_zone_v1(
    sig: dict,
    bias_side: str | None,
    current_price: float | None,
    range_pos: float | None,
    atr15: float | None,
) -> dict:
    """
    Fallback probe zone when strict 5-zone detector misses.
    Keep probe usable for class A/B setups by reusing zones already computed elsewhere.
    Priority:
    1) playbook zone
    2) fib zone
    3) pullback healthy/deep zone
    4) smart-entry FVG zone
    5) entry_zone_v6
    6) range boundary
    """
    meta = (sig.get("meta") or {})
    side = _probe_pick_side(sig, bias_side)
    if side not in ("BUY", "SELL"):
        return {"ok": False, "reason": "no_side"}

    cp = _safe_float(current_price)
    a = float(atr15 or 0.0)
    if cp is None:
        cp = _safe_float(sig.get("last_price")) or _safe_float(sig.get("current_price")) or _safe_float(sig.get("entry"))
    if cp is None:
        return {"ok": False, "reason": "no_price"}
    if a <= 0:
        a = max(1e-9, abs(cp) * 0.003)

    def _mk(zlo, zhi, ztype, strength="MEDIUM", reason=None):
        zlo = _safe_float(zlo)
        zhi = _safe_float(zhi)
        if zlo is None or zhi is None:
            return None
        lo = min(zlo, zhi)
        hi = max(zlo, zhi)
        return {
            "ok": True,
            "type": ztype,
            "priority": 99,
            "strength": strength,
            "side": side,
            "zone_low": lo,
            "zone_high": hi,
            "reason": reason or [],
        }

    playbook = meta.get("playbook_v2") or {}
    z = _mk(
        playbook.get("zone_low"),
        playbook.get("zone_high"),
        "PLAYBOOK_ZONE",
        "MEDIUM",
        ["fallback từ playbook zone"],
    )
    if z:
        return z

    fib1 = meta.get("fib_confluence_v1") or {}
    if fib1.get("ok"):
        z = _mk(
            fib1.get("zone_low"),
            fib1.get("zone_high"),
            "FIB_ZONE",
            "MEDIUM" if int(fib1.get("score") or 0) < 3 else "HIGH",
            ["fallback từ fib confluence zone"],
        )
        if z:
            return z

    pb1 = meta.get("pullback_engine_v1") or {}
    pct = _safe_float(pb1.get("pullback_pct"))
    a_lo = _safe_float(pb1.get("anchor_low"))
    a_hi = _safe_float(pb1.get("anchor_high"))
    if pct is not None and a_lo is not None and a_hi is not None and 0.25 <= pct <= 0.78:
        lo = min(a_lo, a_hi)
        hi = max(a_lo, a_hi)
        rng = max(1e-9, hi - lo)
        if side == "BUY":
            zlo = hi - 0.62 * rng
            zhi = hi - 0.40 * rng
        else:
            zlo = lo + 0.40 * rng
            zhi = lo + 0.62 * rng
        z = _mk(
            zlo, zhi,
            "PULLBACK_ZONE",
            "HIGH" if 0.40 <= pct <= 0.62 else "MEDIUM",
            [f"fallback từ pullback {pb1.get('pullback_pct_text', 'n/a')}"],
        )
        if z:
            return z

    fvgp = meta.get("fvg_range_plugin_v1") or {}
    fvg = fvgp.get("fvg") or {}
    if fvg.get("ok"):
        z = _mk(
            fvg.get("low"),
            fvg.get("high"),
            "FVG_ZONE",
            "MEDIUM",
            ["fallback từ SMART ENTRY FILTER / FVG"],
        )
        if z:
            return z

    ez = meta.get("entry_zone_v6") or {}
    if str(ez.get("side") or "").upper() == side:
        z = _mk(
            ez.get("low"),
            ez.get("high"),
            "ENTRY_ZONE_V6",
            "LOW",
            ["fallback từ entry_zone_v6"],
        )
        if z:
            return z

    k = meta.get("key_levels") or {}
    rlo = _safe_float(k.get("M15_RANGE_LOW"))
    rhi = _safe_float(k.get("M15_RANGE_HIGH"))
    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None
    if rlo is not None and rhi is not None:
        if side == "BUY":
            if rp is None or rp <= 0.55:
                return {
                    "ok": True,
                    "type": "RANGE_LOW_FALLBACK",
                    "priority": 100,
                    "strength": "LOW",
                    "side": side,
                    "zone_low": rlo - 0.10 * a,
                    "zone_high": rlo + 0.15 * a,
                    "reason": ["fallback từ biên dưới range M15"],
                }
        else:
            if rp is None or rp >= 0.45:
                return {
                    "ok": True,
                    "type": "RANGE_HIGH_FALLBACK",
                    "priority": 100,
                    "strength": "LOW",
                    "side": side,
                    "zone_low": rhi - 0.15 * a,
                    "zone_high": rhi + 0.10 * a,
                    "reason": ["fallback từ biên trên range M15"],
                }

    return {"ok": False, "reason": "no_fallback_probe_zone"}

def _build_probe_plan_v1(
    sig: dict,
    zone: dict,
    atr15: float | None,
    current_price: float | None,
) -> dict:
    meta = (sig.get("meta") or {})
    pb1 = meta.get("pullback_engine_v1") or {}
    side = str(zone.get("side") or "NONE").upper()
    a = float(atr15 or 0.0)
    if a <= 0:
        cp = _safe_float(current_price) or _safe_float(sig.get("last_price")) or _safe_float(sig.get("current_price")) or 1.0
        a = max(1e-9, abs(cp) * 0.003)

    zlo = _safe_float(zone.get("zone_low"))
    zhi = _safe_float(zone.get("zone_high"))
    if zlo is None or zhi is None:
        return {"active": False, "reason": ["zone invalid"]}

    lo = min(zlo, zhi)
    hi = max(zlo, zhi)
    entry = (lo + hi) / 2.0

    a_lo = _safe_float(pb1.get("anchor_low"))
    a_hi = _safe_float(pb1.get("anchor_high"))

    # invalidation ưu tiên theo anchor, nếu không có dùng zone +/- ATR
    if side == "BUY":
        sl = a_lo if a_lo is not None else (lo - 0.30 * a)
        if sl >= entry:
            sl = entry - a
        risk = max(1e-9, entry - sl)
        tp1 = entry + 1.0 * risk
        tp2 = entry + 1.6 * risk
    else:
        sl = a_hi if a_hi is not None else (hi + 0.30 * a)
        if sl <= entry:
            sl = entry + a
        risk = max(1e-9, sl - entry)
        tp1 = entry - 1.0 * risk
        tp2 = entry - 1.6 * risk

    ccv4 = meta.get("close_confirm_v4") or {}
    tg3 = meta.get("trigger_engine_v3") or {}
    cc_strength = str(ccv4.get("strength") or "NO").upper()
    tg_state = str(tg3.get("state") or "WAIT").upper()

    if tg_state in ("READY", "TRIGGERED") and cc_strength in ("WEAK", "STRONG"):
        entry_status = "READY"
    elif cc_strength in ("WEAK", "STRONG"):
        entry_status = "WAIT_CONFIRM"
    else:
        entry_status = "AGGRESSIVE"

    # created_ts = last closed M15 ts nếu có
    created_ts = None
    m15_raw = meta.get("_m15_raw") or []
    try:
        if m15_raw:
            c = m15_raw[-2] if len(m15_raw) > 1 else m15_raw[-1]
            created_ts = _c_val(c, "ts", None) or _c_val(c, "time", None)
    except Exception:
        created_ts = None

    return {
        "active": True,
        "probe_side": side,
        "probe_entry": float(entry),
        "probe_sl": float(sl),
        "probe_tp1": float(tp1),
        "probe_tp2": float(tp2),
        "probe_zone_type": str(zone.get("type") or "UNKNOWN"),
        "probe_zone_low": float(lo),
        "probe_zone_high": float(hi),
        "probe_zone_strength": str(zone.get("strength") or "LOW"),
        "probe_created_tf": "M15",
        "probe_created_ts": created_ts,
        "entry_status": entry_status,
        "reason": zone.get("reason") or [],
    }


def _probe_review_tf_v1(candles, side: str, entry: float, sl: float, zone_low: float, zone_high: float) -> dict:
    """
    Đọc phản ứng trên 1 TF (M15/M30/H1) bằng 3 nến đã đóng gần nhất.
    """
    out = {
        "hold_zone": "NO",
        "reaction_speed": "SLOW",
        "close_confirm": "NO",
        "follow_through": "NO",
        "fav_excursion": 0.0,
        "adv_excursion": 0.0,
    }
    if not candles or len(candles) < 4:
        return out

    closed = list(candles[:-1] if len(candles) > 1 else candles)
    use = closed[-3:]
    highs = [float(_c_val(x, "high", 0.0) or 0.0) for x in use]
    lows = [float(_c_val(x, "low", 0.0) or 0.0) for x in use]
    closes = [float(_c_val(x, "close", 0.0) or 0.0) for x in use]

    if side == "BUY":
        hold_zone = min(lows) >= min(zone_low, zone_high) - 1e-9
        fav = max(highs) - entry
        adv = max(0.0, entry - min(lows))
        close_confirm = "STRONG" if closes[-1] > max(zone_low, zone_high) else ("WEAK" if closes[-1] >= entry else "NO")
        follow = "YES" if closes[-1] >= closes[0] and fav > adv else "NO"
    else:
        hold_zone = max(highs) <= max(zone_low, zone_high) + 1e-9
        fav = entry - min(lows)
        adv = max(0.0, max(highs) - entry)
        close_confirm = "STRONG" if closes[-1] < min(zone_low, zone_high) else ("WEAK" if closes[-1] <= entry else "NO")
        follow = "YES" if closes[-1] <= closes[0] and fav > adv else "NO"

    rr = max(1e-9, abs(entry - sl))
    speed = "FAST" if fav >= 0.8 * rr else ("NORMAL" if fav >= 0.35 * rr else "SLOW")

    out.update({
        "hold_zone": "YES" if hold_zone else "NO",
        "reaction_speed": speed,
        "close_confirm": close_confirm,
        "follow_through": follow,
        "fav_excursion": fav,
        "adv_excursion": adv,
    })
    return out


def _finalize_probe_v1(plan: dict, review_m15: dict, review_m30: dict, review_h1: dict) -> dict:
    """
    SUCCESS / WAIT_CONFIRM / FAILED / TIMEOUT
    Timeout = 4 nến M15 gần nhất không follow-through đủ
    """
    if not plan.get("active"):
        return {"result": "INACTIVE", "main_entry_ok": False, "summary": "Chưa có probe active"}

    side = str(plan.get("probe_side") or "NONE").upper()

    m15_hold = str(review_m15.get("hold_zone") or "NO").upper()
    m15_cc = str(review_m15.get("close_confirm") or "NO").upper()
    m15_ft = str(review_m15.get("follow_through") or "NO").upper()

    fav = float(review_m15.get("fav_excursion") or 0.0)
    adv = float(review_m15.get("adv_excursion") or 0.0)

    # fail nhanh
    if m15_hold != "YES":
        return {
            "result": "FAILED",
            "main_entry_ok": False,
            "summary": "Probe sai: vùng không giữ được",
            "market_read": "thị trường không bảo vệ vùng dò",
            "upgrade": "KHÔNG vào lệnh chính",
        }

    if adv > fav * 1.2 and adv > 0:
        return {
            "result": "FAILED",
            "main_entry_ok": False,
            "summary": "Probe sai: phản ứng ngược lấn át",
            "market_read": "thị trường chưa chấp nhận hướng dò",
            "upgrade": "KHÔNG vào lệnh chính",
        }

    # success
    if m15_hold == "YES" and m15_cc in ("WEAK", "STRONG") and m15_ft == "YES" and fav > adv:
        ready = (m15_cc == "STRONG") or (str(review_m30.get("close_confirm") or "NO").upper() in ("WEAK", "STRONG"))
        return {
            "result": "SUCCESS" if ready else "WAIT_CONFIRM",
            "main_entry_ok": bool(ready),
            "summary": "Probe đúng hướng, vùng đang được bảo vệ",
            "market_read": "thị trường phản ứng đúng vùng và có follow-through",
            "upgrade": "CÓ THỂ vào lệnh chính" if ready else "Chưa vào lệnh chính, chờ thêm close confirm",
        }

    # timeout / neutral
    if m15_hold == "YES" and m15_ft == "NO" and str(review_m15.get("reaction_speed") or "SLOW").upper() == "SLOW":
        return {
            "result": "TIMEOUT",
            "main_entry_ok": False,
            "summary": "Probe đứng im quá lâu",
            "market_read": "thị trường chưa cho follow-through trong khung 1h",
            "upgrade": "Thoát lệnh dò, chờ probe mới",
        }

    return {
        "result": "WAIT_CONFIRM",
        "main_entry_ok": False,
        "summary": "Probe tạm đúng nhưng chưa đủ mạnh",
        "market_read": "có phản ứng nhưng xác nhận còn yếu",
        "upgrade": "Chưa vào lệnh chính",
    }


def _build_probe_engine_v1(
    sig: dict,
    symbol: str,
    bias_side: str | None,
    m15c,
    m30c,
    h1c,
    atr15: float | None,
    range_pos: float | None,
    cls: str,
    setup_score: float,
) -> dict:
    """
    Probe V3 (direct 5-condition engine)
    Mục tiêu: Class A/B thì probe KHÔNG bị INACTIVE vô lý.
    Ưu tiên dùng trực tiếp 5 điều kiện thay vì phụ thuộc hoàn toàn vào detector strict.
    5 điều kiện:
    1) liquidity zone
    2) break & retest
    3) pullback zone
    4) EMA cluster
    5) range boundary
    Nếu vẫn chưa đủ thì fallback từ playbook/fib/FVG/entry_zone/setup plan.
    """
    meta = sig.get("meta") or {}
    sc3 = meta.get("setup_class_v3") or {}
    
    if sc3:
        cls = str(sc3.get("class") or cls or "D").upper()
        try:
            setup_score = float(sc3.get("score") or setup_score or 0.0)
        except Exception:
            setup_score = float(setup_score or 0.0)
    side = _probe_pick_side(sig, bias_side)
    current_price = (
        _safe_float(sig.get("current_price"))
        or _safe_float(sig.get("last_price"))
        or _safe_float(sig.get("price"))
        or _safe_float(sig.get("entry"))
    )
    if current_price is None:
        try:
            if m15c:
                last_c = m15c[-1]
                current_price = _safe_float(_c_val(last_c, "close", None))
        except Exception:
            current_price = None

    # ===== DEBUG PROBE START =====

    # ===== DEBUG PROBE END =====

    current_price = (
        _safe_float(sig.get("last_price"))
        or _safe_float(sig.get("current_price"))
        or _safe_float(sig.get("entry"))
    )
    if current_price is None:
        try:
            current_price = float(_c_val((m15c or [])[-1], "close", 0.0) or 0.0)
        except Exception:
            current_price = None

    base = {
        "active": False,
        "class": cls,
        "setup_score": setup_score,
        "zone_type": None,
        "zone_strength": None,
        "side": None,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "entry_status": None,
        "created_tf": None,
        "created_ts": None,
        "review_m15": {},
        "review_m30": {},
        "review_h1": {},
        "result": "INACTIVE",
        "main_entry_ok": False,
        "summary": "Chưa có probe zone hợp lệ",
        "market_read": "",
        "upgrade": "",
        "reason": [],
        "detector": "NONE",
    }

    if cls not in ("A", "B"):
        base["summary"] = "Chỉ tạo probe khi setup class là A/B"
        return base

    side = _probe_pick_side(sig, bias_side)
    if side not in ("BUY", "SELL"):
        base["summary"] = "Chưa xác định được side cho probe"
        return base

    cp = _safe_float(current_price)
    if cp is None:
        base["summary"] = "Không có current price để dựng probe"
        return base

    a = float(atr15 or 0.0)
    if a <= 0:
        a = max(1e-9, abs(cp) * 0.003)

    liq = meta.get("liquidity_map_v1") or {}
    pb1 = meta.get("pullback_engine_v1") or {}
    ema = meta.get("ema") or {}
    k = meta.get("key_levels") or {}
    ccv4 = meta.get("close_confirm_v4") or {}
    cont1 = meta.get("post_break_continuity_v1") or {}
    fib1 = meta.get("fib_confluence_v1") or {}
    playbook = meta.get("playbook_v2") or {}
    fvgp = meta.get("fvg_range_plugin_v1") or {}
    fvg = fvgp.get("fvg") or {}
    ez = meta.get("entry_zone_v6") or {}

    def _mk_zone(ztype, zlo, zhi, strength="MEDIUM", reason=None, priority=99):
        zlo = _safe_float(zlo)
        zhi = _safe_float(zhi)
        if zlo is None or zhi is None:
            return None
        return {
            "ok": True,
            "type": ztype,
            "priority": priority,
            "strength": strength,
            "side": side,
            "zone_low": min(zlo, zhi),
            "zone_high": max(zlo, zhi),
            "reason": list(reason or []),
        }

    zone = None
    detector = "NONE"

    # =====================================================
    # DIRECT 5 CONDITIONS (ưu tiên đúng theo yêu cầu user)
    # =====================================================

    # 1) LIQUIDITY
    if zone is None:
        if side == "BUY":
            z = liq.get("below_zone") if isinstance(liq.get("below_zone"), (list, tuple)) and len(liq.get("below_zone")) == 2 else None
            if z is not None and (str(liq.get("below_strength") or "LOW").upper() in ("MEDIUM", "HIGH") or bool(liq.get("equal_lows"))):
                zone = _mk_zone("LIQUIDITY", z[0], z[1], str(liq.get("below_strength") or "MEDIUM").upper(), ["điều kiện 1: liquidity phía dưới"], 1)
                detector = "DIRECT_LIQUIDITY"
        else:
            z = liq.get("above_zone") if isinstance(liq.get("above_zone"), (list, tuple)) and len(liq.get("above_zone")) == 2 else None
            if z is not None and (str(liq.get("above_strength") or "LOW").upper() in ("MEDIUM", "HIGH") or bool(liq.get("equal_highs"))):
                zone = _mk_zone("LIQUIDITY", z[0], z[1], str(liq.get("above_strength") or "MEDIUM").upper(), ["điều kiện 1: liquidity phía trên"], 1)
                detector = "DIRECT_LIQUIDITY"

    # 2) BREAK & RETEST (nới lỏng hơn bản strict)
    if zone is None:
        bos = _safe_float(k.get("M15_BOS") or k.get("M15_BOS_LEVEL"))
        hold = str(ccv4.get("hold") or "NO").upper()
        cc_strength = str(ccv4.get("strength") or "NO").upper()
        cont_state = str(cont1.get("state") or "NONE").upper()
        if bos is not None:
            near_bos = abs(cp - bos) <= max(1e-9, 0.40 * a)
            if near_bos or hold == "YES" or cc_strength in ("WEAK", "STRONG") or cont_state.startswith("POST_BREAK"):
                zone = _mk_zone("BREAK_RETEST", bos - 0.22 * a, bos + 0.22 * a, "MEDIUM", ["điều kiện 2: đang quanh BOS / retest"], 2)
                detector = "DIRECT_BREAK_RETEST"

    # 3) PULLBACK (ưu tiên rất mạnh cho case 45%)
    if zone is None:
        pct = _safe_float(pb1.get("pullback_pct"))
        a_lo = _safe_float(pb1.get("anchor_low"))
        a_hi = _safe_float(pb1.get("anchor_high"))
        if bool(pb1.get("ok")) and pct is not None and a_lo is not None and a_hi is not None and 0.25 <= pct <= 0.78:
            lo = min(a_lo, a_hi)
            hi = max(a_lo, a_hi)
            rng = max(1e-9, hi - lo)
            if side == "BUY":
                zlo = hi - 0.62 * rng
                zhi = hi - 0.40 * rng
            else:
                zlo = lo + 0.40 * rng
                zhi = lo + 0.62 * rng
            zone = _mk_zone(
                "PULLBACK",
                zlo,
                zhi,
                "HIGH" if 0.40 <= pct <= 0.62 else "MEDIUM",
                [f"điều kiện 3: pullback đẹp {pb1.get('pullback_pct_text', 'n/a')}", str(pb1.get("label") or "pullback")],
                3,
            )
            detector = "DIRECT_PULLBACK"

    # 4) EMA CLUSTER (nới lỏng: chỉ cần EMA34/89 hoặc EMA200 gần giá)
    if zone is None:
        e34 = _safe_float(ema.get("ema34"))
        e89 = _safe_float(ema.get("ema89"))
        e200 = _safe_float(ema.get("ema200"))
        ema_candidates = [x for x in [e34, e89] if x is not None]
        if len(ema_candidates) >= 2:
            zlo = min(ema_candidates) - 0.12 * a
            zhi = max(ema_candidates) + 0.12 * a
            if abs(cp - zlo) <= 0.45 * a or abs(cp - zhi) <= 0.45 * a or (zlo <= cp <= zhi):
                zone = _mk_zone("EMA_CLUSTER", zlo, zhi, "MEDIUM", ["điều kiện 4: giá đang gần EMA34/89"], 4)
                detector = "DIRECT_EMA_CLUSTER"
        elif e200 is not None and abs(cp - e200) <= 0.45 * a:
            zone = _mk_zone("EMA_CLUSTER", e200 - 0.12 * a, e200 + 0.12 * a, "LOW", ["điều kiện 4: giá đang gần EMA200"], 4)
            detector = "DIRECT_EMA_CLUSTER"

    # 5) RANGE BOUNDARY (nới lỏng: có thể dùng low/high boundary theo side)
    if zone is None:
        rlo = _safe_float(k.get("M15_RANGE_LOW"))
        rhi = _safe_float(k.get("M15_RANGE_HIGH"))
        try:
            rp = float(range_pos) if range_pos is not None else None
        except Exception:
            rp = None
        if rlo is not None and rhi is not None:
            if side == "BUY" and (rp is None or rp <= 0.30):
                zone = _mk_zone("RANGE_BOUNDARY", rlo - 0.10 * a, rlo + 0.15 * a, "MEDIUM" if rp is not None and rp <= 0.20 else "LOW", ["điều kiện 5: giá gần biên dưới range"], 5)
                detector = "DIRECT_RANGE_BOUNDARY"
            elif side == "SELL" and (rp is None or rp >= 0.70):
                zone = _mk_zone("RANGE_BOUNDARY", rhi - 0.15 * a, rhi + 0.10 * a, "MEDIUM" if rp is not None and rp >= 0.80 else "LOW", ["điều kiện 5: giá gần biên trên range"], 5)
                detector = "DIRECT_RANGE_BOUNDARY"

    # =====================================================
    # FALLBACKS
    # =====================================================
    if zone is None:
        z = _mk_zone(playbook.get("zone_low"), playbook.get("zone_high"), None)
        if z is None:
            z = _mk_zone("PLAYBOOK_ZONE", playbook.get("zone_low"), playbook.get("zone_high"), "MEDIUM", ["fallback từ playbook zone"], 90)
        zone = z
        if zone is not None:
            detector = "FALLBACK_PLAYBOOK"

    if zone is None and fib1.get("ok"):
        zone = _mk_zone("FIB_ZONE", fib1.get("zone_low"), fib1.get("zone_high"), "MEDIUM" if int(fib1.get("score") or 0) < 3 else "HIGH", ["fallback từ fib zone"], 91)
        if zone is not None:
            detector = "FALLBACK_FIB"

    if zone is None and fvg.get("ok"):
        zone = _mk_zone("FVG_ZONE", fvg.get("low"), fvg.get("high"), "MEDIUM", ["fallback từ FVG"], 92)
        if zone is not None:
            detector = "FALLBACK_FVG"

    if zone is None and str(ez.get("side") or "").upper() == side:
        zone = _mk_zone("ENTRY_ZONE_V6", ez.get("low"), ez.get("high"), "LOW", ["fallback từ entry_zone_v6"], 93)
        if zone is not None:
            detector = "FALLBACK_ENTRY_ZONE_V6"

    if zone is None:
        try:
            plan0 = _build_setup_plan_v1(sig, cls)
        except Exception:
            plan0 = {}
        entry0 = _safe_float(plan0.get("entry"))
        sl0 = _safe_float(plan0.get("sl"))
        if entry0 is not None and sl0 is not None:
            pad = max(1e-9, 0.18 * a)
            zone = {
                "ok": True,
                "type": "SETUP_PLAN_FALLBACK",
                "priority": 999,
                "strength": "LOW",
                "side": side,
                "zone_low": min(entry0, sl0) - pad,
                "zone_high": max(entry0, sl0) + pad,
                "reason": ["fallback từ setup plan A/B"],
            }
            detector = "SETUP_PLAN_FALLBACK"

    if not zone or not zone.get("ok"):
        base["summary"] = "Class A/B nhưng chưa gom được vùng dò từ 5 điều kiện trực tiếp + fallback"
        base["reason"] = [
            f"side={side}",
            f"pullback_ok={bool(pb1.get('ok'))}",
            f"pullback_pct={pb1.get('pullback_pct_text', 'n/a')}",
        ]
        return base

    plan = _build_probe_plan_v1(sig=sig, zone=zone, atr15=atr15, current_price=current_price)
    if not plan.get("active"):
        base["summary"] = "Có zone nhưng không dựng được plan dò"
        base["reason"] = [f"detector={detector}"] + list(zone.get("reason") or [])
        base["detector"] = detector
        return base

    review_m15 = _probe_review_tf_v1(
        m15c,
        side=plan["probe_side"],
        entry=plan["probe_entry"],
        sl=plan["probe_sl"],
        zone_low=plan["probe_zone_low"],
        zone_high=plan["probe_zone_high"],
    )
    review_m30 = _probe_review_tf_v1(
        m30c,
        side=plan["probe_side"],
        entry=plan["probe_entry"],
        sl=plan["probe_sl"],
        zone_low=plan["probe_zone_low"],
        zone_high=plan["probe_zone_high"],
    )
    review_h1 = _probe_review_tf_v1(
        h1c,
        side=plan["probe_side"],
        entry=plan["probe_entry"],
        sl=plan["probe_sl"],
        zone_low=plan["probe_zone_low"],
        zone_high=plan["probe_zone_high"],
    )

    final = _finalize_probe_v1(plan, review_m15, review_m30, review_h1)
    active_flag = bool(plan.get("active"))
    result = final.get("result")
    if result in (None, "INACTIVE"):
        result = "WAIT_CONFIRM"

    summary = final.get("summary") or "Probe đã được tạo"
    reason_lines = [f"detector={detector}"] + list(plan.get("reason") or [])

    base.update({
        "active": active_flag,
        "zone_type": plan.get("probe_zone_type"),
        "zone_strength": plan.get("probe_zone_strength"),
        "side": plan.get("probe_side"),
        "entry": plan.get("probe_entry"),
        "sl": plan.get("probe_sl"),
        "tp1": plan.get("probe_tp1"),
        "tp2": plan.get("probe_tp2"),
        "entry_status": plan.get("entry_status"),
        "created_tf": plan.get("probe_created_tf"),
        "created_ts": plan.get("probe_created_ts"),
        "review_m15": review_m15,
        "review_m30": review_m30,
        "review_h1": review_h1,
        "result": result,
        "main_entry_ok": bool(final.get("main_entry_ok")),
        "summary": summary,
        "market_read": final.get("market_read"),
        "upgrade": final.get("upgrade"),
        "reason": reason_lines[:5],
        "detector": detector,
    })
    return base


def _render_probe_block_v1(sig: dict) -> list[str]:
    meta = (sig.get("meta") or {})
    pe = meta.get("probe_engine_v1") or {}

    lines = []
    lines.append("")
    lines.append("🧪 PROBE ENGINE")

    if not pe:
        lines.append("- Trạng thái: INACTIVE")
        lines.append("- Lý do: chưa có dữ liệu probe")
        return lines

    result = str(pe.get("result") or "INACTIVE").upper()
    side = str(pe.get("side") or "NONE").upper()

    if result == "INACTIVE":
        lines.append("- Trạng thái: INACTIVE")
        lines.append(f"- Lý do: {pe.get('summary', 'chưa có probe zone hợp lệ')}")
        return lines

    lines.append(f"- Trạng thái: {result}")
    lines.append(f"- Zone: {pe.get('zone_type', 'UNKNOWN')} ({pe.get('zone_strength', 'LOW')})")
    lines.append(f"- Side: {side}")
    lines.append(f"- Entry: {nf(pe.get('entry')) if pe.get('entry') is not None else 'n/a'}")
    tp1 = pe.get("tp1")
    tp2 = pe.get("tp2")
    if tp1 is not None and tp2 is not None:
        lines.append(f"- TP: {nf(tp1)} / {nf(tp2)}")
    else:
        lines.append("- TP: n/a")
    lines.append(f"- SL: {nf(pe.get('sl')) if pe.get('sl') is not None else 'n/a'}")
    lines.append(f"- Entry status: {pe.get('entry_status', 'AGGRESSIVE')}")

    r15 = pe.get("review_m15") or {}
    r30 = pe.get("review_m30") or {}
    rh1 = pe.get("review_h1") or {}

    lines.append("- Review M15: "
                 f"Hold={r15.get('hold_zone', 'NO')} | "
                 f"Speed={r15.get('reaction_speed', 'SLOW')} | "
                 f"Confirm={r15.get('close_confirm', 'NO')} | "
                 f"FT={r15.get('follow_through', 'NO')}")
    lines.append("- Review M30: "
                 f"Hold={r30.get('hold_zone', 'NO')} | "
                 f"Confirm={r30.get('close_confirm', 'NO')} | "
                 f"FT={r30.get('follow_through', 'NO')}")
    lines.append("- Review H1: "
                 f"Hold={rh1.get('hold_zone', 'NO')} | "
                 f"Confirm={rh1.get('close_confirm', 'NO')} | "
                 f"FT={rh1.get('follow_through', 'NO')}")

    if pe.get("reason"):
        lines.append("- Lý do tạo probe: " + ", ".join([str(x) for x in (pe.get("reason") or [])[:3]]))

    lines.append(f"- Kết luận: {pe.get('summary', '')}")
    if pe.get("market_read"):
        lines.append(f"- Thị trường: {pe.get('market_read')}")
    if pe.get("upgrade"):
        lines.append(f"- Lệnh chính: {pe.get('upgrade')}")

    return lines    
def _attach_vnext_meta(
    base: dict,
    *,
    symbol: str,
    m15c,
    bias_side,
    h1_trend,
    h4_trend,
    market_state_v2,
    flow_state,
    range_pos,
    no_trade_zone,
    liquidation_evt,
    m15_struct,
    rsi15,
    div,
    atr15,
    liquidity_map_v1,
    ema_pack,
    playbook_v2,
    close_confirm_v4,
    sweep_buy,
    sweep_sell,
    spring_buy,
    spring_sell,
    entry_sniper,
    playbook_v4=None,
):
    try:
        # ===== VNEXT =====
        context_verdict_v1 = _context_verdict_v1(
            bias_side=bias_side,
            h1_trend=h1_trend,
            h4_trend=h4_trend,
            market_state_v2=market_state_v2,
            flow_state=flow_state,
            range_pos=range_pos,
            no_trade_zone=no_trade_zone,
            liquidation_evt=liquidation_evt,
            m15_struct_tag=m15_struct.get("tag") if isinstance(m15_struct, dict) else "n/a",
        )

        rsi_context_v1 = _rsi_context_v1(
            rsi15=rsi15,
            bias_side=bias_side,
            h1_trend=h1_trend,
            market_state_v2=market_state_v2,
            div=div,
            liquidation_evt=liquidation_evt,
        )

        fib_confluence_v1 = _fib_confluence_v1(
            m15c=m15c,
            bias_side=bias_side,
            atr15=atr15,
            liquidity_map_v1=liquidity_map_v1,
            ema_pack=ema_pack,
            playbook_v2=playbook_v2,
        )

        liquidity_completion_v1 = _liquidity_completion_v1(
            sweep_buy=sweep_buy,
            sweep_sell=sweep_sell,
            spring_buy=spring_buy,
            spring_sell=spring_sell,
            close_confirm_v4=close_confirm_v4 or {"strength": "NO"},
            entry_sniper=entry_sniper or {"trigger": "NONE"},
            bias_side=bias_side,
        )

        trap_warning_v1 = _trap_warning_v1(
            bias_side=bias_side,
            context_verdict=context_verdict_v1,
            rsi_ctx=rsi_context_v1,
            no_trade_zone=no_trade_zone,
            liquidation_evt=liquidation_evt,
            range_pos=range_pos,
            div=div,
            close_confirm_v4=close_confirm_v4 or {"strength": "NO"},
        )

        pullback_engine_v1 = _pullback_engine_v1(
            bias_side=bias_side,
            current_price=float(m15c[-1].close) if m15c else None,
            key_levels=(base.get("meta", {}) or {}).get("key_levels", {}),
            ema_pack=ema_pack or {},
            h1_trend=h1_trend,
            h4_trend=h4_trend,
            m15_struct_tag=(m15_struct or {}).get("tag"),
            close_confirm_v4=close_confirm_v4 or {"strength": "NO"},
            liquidity_completion_v1=liquidity_completion_v1 or {"state": "NO"},
            trap_warning_v1=trap_warning_v1 or {"active": False},
            atr15=atr15,
        )

        # ===== SMART ENTRY FILTER / EMA snapshot =====
        meta = base.setdefault("meta", {})
        if isinstance(ema_pack, dict) and ema_pack:
            meta["ema"] = ema_pack
        if m15c:
            meta.setdefault("_m15_raw", m15c)
        if atr15 is not None:
            meta["atr15"] = atr15

        rp_for_filter = range_pos
        try:
            if rp_for_filter is None:
                k_meta = meta.get("key_levels") or {}
                lo_rf = _safe_float(k_meta.get("M15_RANGE_LOW"))
                hi_rf = _safe_float(k_meta.get("M15_RANGE_HIGH"))
                cp_rf = None
                try:
                    cp_rf = float(m15c[-1].close) if m15c else None
                except Exception:
                    cp_rf = None
                if lo_rf is not None and hi_rf is not None and cp_rf is not None and hi_rf > lo_rf:
                    rp_for_filter = (cp_rf - lo_rf) / max(1e-9, hi_rf - lo_rf)
            if rp_for_filter is None and m15c:
                lo_rf2, hi_rf2, last_rf2 = _range_levels(m15c[:-1] if len(m15c) > 1 else m15c, n=20)
                if lo_rf2 is not None and hi_rf2 is not None and last_rf2 is not None and hi_rf2 > lo_rf2:
                    rp_for_filter = (last_rf2 - lo_rf2) / max(1e-9, hi_rf2 - lo_rf2)
        except Exception:
            pass

        try:
            meta["fvg_range_plugin_v1"] = _build_fvg_range_plugin_v1(
                m15c=m15c,
                bias_side=bias_side,
                range_pos=rp_for_filter,
                atr15=atr15,
                ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
            )
        except Exception as _smart_e:
            meta["fvg_range_plugin_v1"] = {
                "range_filter": {"state": "UNKNOWN", "position": None, "tag": "N/A", "reason": [f"smart-filter-error: {_smart_e}"]},
                "ema": {"trend": "N/A", "alignment": "NO", "zone": "N/A", "ema34": None, "ema89": None, "ema200": None},
                "fvg": {"ok": False, "type": "NONE", "low": None, "high": None, "text": "chưa có vùng rõ"},
                "smart_state": "NEUTRAL",
                "entry_mode": None,
                "entry": None,
                "sl": None,
                "tp1": None,
                "tp2": None,
            }
            
        kl0 = (base.get("meta", {}) or {}).get("key_levels", {}) or {}
        cp0 = None
        try:
            cp0 = (
                _safe_float(sig.get("current_price"))
                or _safe_float(sig.get("last_price"))
                or _safe_float(sig.get("price"))
                or _safe_float(sig.get("entry"))
            )
        except Exception:
            cp0 = None
    
        if cp0 is None:
            try:
                if m15c:
                    cp0 = _safe_float(_c_val(m15c[-1], "close", None))
            except Exception:
                cp0 = None
    
        try:    
            pf1 = _path_forecast_v1(
                current_price=cp0,
                atr15=atr15,
                h1_trend=h1_trend,
                h4_trend=h4_trend,
                m15_struct_tag=(m15_struct or {}).get("tag"),
                range_low=kl0.get("M15_RANGE_LOW"),
                range_high=kl0.get("M15_RANGE_HIGH"),
                playbook_v2=playbook_v2,
                liquidity_map_v1=liquidity_map_v1,
                ema_pack=ema_pack,
                smart_filter_v1=(base.get("meta", {}) or {}).get("fvg_range_plugin_v1") or {},
                m15c=m15c,
            )    
        except Exception as e:
            pf1 = {
                "down_bias": "KHÔNG RÕ",
                "up_bias": "KHÔNG RÕ",
                "sideway_bars": "n/a",
                "res_near": None,
                "res_far": None,
                "sup_near": None,
                "sup_far": None,
                "priority_action": "ƯU TIÊN ĐỨNG NGOÀI",
                "reason": [f"path_forecast_error: {e}"],
            }
    
        base.setdefault("meta", {})["path_forecast_v1"] = pf1
        # ===== ZONE + ACTION ENGINE V1 =====
        pf1 = base.setdefault("meta", {}).get("path_forecast_v1") or {}
        priority_action = str(pf1.get("priority_action") or "").upper()
        
        za_side = "NONE"
        if "BUY" in priority_action:
            za_side = "BUY"
        elif "SELL" in priority_action:
            za_side = "SELL"
        else:
            # fallback theo best_side/final_side nếu priority_action chưa rõ
            de1_tmp = base.setdefault("meta", {}).get("decision_engine_v1") or {}
            za_side = str(
                de1_tmp.get("best_side")
                or de1_tmp.get("final_side")
                or "NONE"
            ).upper()
        
        support_zone = pf1.get("sup_near")
        resistance_zone = pf1.get("res_near")
        
        # break level đúng theo side
        za_break = None
        if za_side == "BUY":
            za_break = pf1.get("break_up") or pf1.get("range_high")
        elif za_side == "SELL":
            za_break = pf1.get("break_down") or pf1.get("range_low")
        
        zone_action_v1 = _zone_action_engine_v1(
            current_price=base.get("current_price") or (base.get("meta") or {}).get("current_price"),
            side=za_side,
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            break_level=za_break,
            atr15=atr15,
        )
        
        base.setdefault("meta", {})["zone_action_v1"] = zone_action_v1
        
        # ===== rewrite wait_for_v1 bằng zone action =====
        old_wait = base.setdefault("meta", {}).get("wait_for_v1") or {}
        
        if zone_action_v1.get("ok"):
            base.setdefault("meta", {})["wait_for_v1"] = {
                **old_wait,
                "side": za_side,
                "zone_type": zone_action_v1.get("zone_type"),
                "price_state": zone_action_v1.get("price_state"),
                "action_state": zone_action_v1.get("action_state"),
                "lines": zone_action_v1.get("lines") or [],
                "trigger_hint": zone_action_v1.get("trigger"),
                "invalid": zone_action_v1.get("invalid"),
            }
        
        
        manual_likelihood_v1 = _manual_likelihood_v1(
            bias_side=bias_side,
            context_verdict=context_verdict_v1,
            trap_warning=trap_warning_v1,
            fib_conf=fib_confluence_v1,
            liq_done=liquidity_completion_v1,
            close_confirm_v4=close_confirm_v4 or {"strength": "NO"},
            entry_sniper=entry_sniper or {"direction": "NONE", "trigger": "NONE"},
            playbook_v4=playbook_v4 or {"quality": "LOW"},
        )

        manual_guidance_v1 = _manual_guidance_v1(
            bias_side=bias_side,
            context_verdict=context_verdict_v1,
            liq_done=liquidity_completion_v1,
            fib_conf=fib_confluence_v1,
            close_confirm_v4=close_confirm_v4 or {"strength": "NO"},
            entry_sniper=entry_sniper or {"trigger": "NONE"},
            playbook_v2=playbook_v2 or {},
        )

        meta = base.setdefault("meta", {})
        meta["context_verdict_v1"] = context_verdict_v1
        meta["rsi_context_v1"] = rsi_context_v1
        meta["fib_confluence_v1"] = fib_confluence_v1
        meta["liquidity_completion_v1"] = liquidity_completion_v1
        meta["trap_warning_v1"] = trap_warning_v1
        meta["manual_likelihood_v1"] = manual_likelihood_v1
        meta["manual_guidance_v1"] = manual_guidance_v1
        meta["pullback_engine_v1"] = pullback_engine_v1

        #PROBE_ENGINE
        setup_cls, setup_score, setup_reasons = _setup_class_score_v3(base)
        meta["setup_class_v3"] = {
            "class": setup_cls,
            "score": float(setup_score or 0.0),
            "reasons": list(setup_reasons or []),
        }
        try:
            probe_engine_v1 = _build_probe_engine_v1(
                sig=base,
                symbol=symbol,
                bias_side=bias_side,
                m15c=m15c,
                m30c=(meta.get("_m30_raw") or []),
                h1c=(meta.get("_h1_raw") or []),
                atr15=atr15,
                range_pos=range_pos,
                cls=setup_cls,
                setup_score=setup_score,
            )
        except Exception as e:
            probe_engine_v1 = {
                "active": False,
                "result": "INACTIVE",
                "summary": f"probe error: {e}",
            }

        meta["probe_engine_v1"] = probe_engine_v1
        # ===== PRO DESK =====
        m15_tag = str((m15_struct or {}).get("tag") or "n/a").upper()

        # 1) Market state
        market_state_machine_v1 = _market_state_machine_v1(
            h1_trend=h1_trend,
            h4_trend=h4_trend,
            m15_struct_tag=m15_tag,
            market_state_v2=market_state_v2,
            liquidation_evt=liquidation_evt,
            range_pos=range_pos,
        )

        # 2) Bias layers
        if h1_trend == "bullish" and h4_trend == "bullish":
            htf_bias = "BUY"
        elif h1_trend == "bearish" and h4_trend == "bearish":
            htf_bias = "SELL"
        else:
            htf_bias = "MIXED"

        if "LL" in m15_tag or "LH" in m15_tag:
            mtf_bias = "SELL_PULLBACK"
        elif "HH" in m15_tag or "HL" in m15_tag:
            mtf_bias = "BUY_PULLBACK"
        else:
            mtf_bias = "WAIT"

        cv_state = str((context_verdict_v1 or {}).get("state") or "")
        entry_bias = "READY" if "CONTINUATION" in cv_state else "WAIT"

        bias_layers_v1 = {
            "htf_bias": htf_bias,
            "mtf_bias": mtf_bias,
            "entry_bias": entry_bias,
        }

        # 3) No-trade zone
        ntz_reasons = []
        try:
            rp = float(range_pos) if range_pos is not None else None
        except Exception:
            rp = None

        if rp is not None and 0.35 <= rp <= 0.65:
            ntz_reasons.append("giữa biên độ")

        if str((liquidity_completion_v1 or {}).get("state") or "NO").upper() == "NO":
            ntz_reasons.append("chưa có thanh khoản")

        if str((close_confirm_v4 or {}).get("strength") or "NO").upper() in ("NO", "N/A"):
            ntz_reasons.append("chưa có confirm")

        if m15_tag in ("TRANSITION", "N/A", ""):
            ntz_reasons.append("M15 chưa rõ")

        if (trap_warning_v1 or {}).get("active"):
            ntz_reasons.append("trap risk")

        no_trade_zone_v3 = {
            "active": len(ntz_reasons) >= 2,
            "reasons": ntz_reasons[:5],
        }

        # 4) Decision
        sniper_trigger = str((entry_sniper or {}).get("trigger") or "NONE").upper()
        if no_trade_zone_v3["active"]:
            decision_engine_v1 = {
                "decision": "STAND ASIDE",
                "reason": "No-trade zone",
            }
        elif sniper_trigger in ("READY", "TRIGGERED"):
            decision_engine_v1 = {
                "decision": "MANUAL STRIKE",
                "reason": "Có trigger",
            }
        else:
            decision_engine_v1 = {
                "decision": "WAIT",
                "reason": "Chưa đủ điều kiện",
            }

        # 5) Wait for
        wf_lines = []
        zl = (playbook_v2 or {}).get("zone_low")
        zh = (playbook_v2 or {}).get("zone_high")
        if zl is not None and zh is not None:
            wf_lines.append(f"Chờ vùng {_fmt(zl)} – {_fmt(zh)}")

        k2 = meta.get("key_levels", {}) or {}
        if bias_side == "BUY":
            lv = k2.get("M15_RANGE_HIGH")
            if lv is not None:
                wf_lines.append(f"Hoặc break {_fmt(lv)}")
        elif bias_side == "SELL":
            lv = k2.get("M15_RANGE_LOW")
            if lv is not None:
                wf_lines.append(f"Hoặc break {_fmt(lv)}")

        wait_for_v1 = {"lines": wf_lines[:4]}
        meta["market_state_machine_v1"] = market_state_machine_v1
        meta["bias_layers_v1"] = bias_layers_v1
        meta["no_trade_zone_v3"] = no_trade_zone_v3
        meta["decision_engine_v1"] = decision_engine_v1
        meta["wait_for_v1"] = wait_for_v1
        # ===== CONFLICT ENGINE + SUGGESTION =====
        conflict_engine_v1 = _conflict_engine_v1(
            bias_layers_v1=bias_layers_v1,
            context_verdict_v1=context_verdict_v1,
            liquidity_completion_v1=liquidity_completion_v1,
            trap_warning_v1=trap_warning_v1,
            range_pos=range_pos,
            fib_confluence_v1=fib_confluence_v1,
        )
        
        suggestion_block_v1 = _suggestion_block_v1(
            symbol=symbol,
            bias_side=bias_side,
            decision_engine_v1=decision_engine_v1,
            no_trade_zone_v3=no_trade_zone_v3,
            playbook_v2=playbook_v2,
            entry_sniper=entry_sniper,
            manual_likelihood_v1=manual_likelihood_v1,
            conflict_engine_v1=conflict_engine_v1,
            m15_struct=(m15_struct or {}),
        )
        meta["conflict_engine_v1"] = conflict_engine_v1
        meta["suggestion_block_v1"] = suggestion_block_v1

        # ===== TRIGGER ENGINE V2 =====
        trigger_engine_v2 = _trigger_engine_v2(
            bias_side=bias_side,
            context_verdict_v1=context_verdict_v1,
            conflict_engine_v1=conflict_engine_v1,
            no_trade_zone_v3=no_trade_zone_v3,
            entry_sniper=entry_sniper,
            playbook_v2=playbook_v2,
            fib_confluence_v1=fib_confluence_v1,
            liquidity_completion_v1=liquidity_completion_v1,
            close_confirm_v4=close_confirm_v4,
            m15_struct=(m15_struct or {}),
            range_pos=range_pos,
            current_price=base.get("last_price") or base.get("entry") or base.get("current_price"),
            atr15=atr15,
        )
        meta["trigger_engine_v2"] = trigger_engine_v2

        # ===== EXIT ENGINE V2 =====
        review_side_for_exit = _resolve_review_side(base, bias_side)
        invalidation_level_v2 = None
        try:
            m15s = (m15_struct or {})
    
            if review_side_for_exit == "BUY":
                invalidation_level_v2 = (
                    m15s.get("pullback_invalid")
                    or m15s.get("invalid_level")
                    or base.get("invalid")
                )
            elif review_side_for_exit == "SELL":
                invalidation_level_v2 = (
                    m15s.get("pullback_invalid")
                    or m15s.get("invalid_level")
                    or base.get("invalid")
                )
            else:
                invalidation_level_v2 = (
                    m15s.get("pullback_invalid")
                    or m15s.get("invalid_level")
                    or base.get("invalid")
                )
        except Exception:
            invalidation_level_v2 = None
    
        exit_engine_v2 = _exit_engine_v2(
            review_side=review_side_for_exit,
            current_price=base.get("last_price") or base.get("entry") or base.get("current_price"),
            invalidation_level=invalidation_level_v2,
            hl_ok=bool((m15_struct or {}).get("hl")),
            lh_ok=bool((m15_struct or {}).get("lh")),
            break_up=bool((m15_struct or {}).get("break_up")),
            break_dn=bool((m15_struct or {}).get("break_dn")),
            final_score=base.get("final_score") or base.get("score") or 0,
            conflict_engine_v1=conflict_engine_v1,
            trigger_engine_v2=trigger_engine_v2,
            range_pos=range_pos,
        )
        meta["exit_engine_v2"] = exit_engine_v2
        
        position_quality_v2 = _position_quality_v2(
            review_side=review_side_for_exit,
            final_score=base.get("final_score") or base.get("score") or 0,
            range_pos=range_pos,
            hl_ok=bool((m15_struct or {}).get("hl")),
            lh_ok=bool((m15_struct or {}).get("lh")),
            break_up=bool((m15_struct or {}).get("break_up")),
            break_dn=bool((m15_struct or {}).get("break_dn")),
            conflict_engine_v1=conflict_engine_v1,
            trigger_engine_v2=trigger_engine_v2,
        )
    
        review_conflict_v2 = _review_conflict_v2(
            review_side=review_side_for_exit,
            range_pos=range_pos,
            liquidity_completion_v1=liquidity_completion_v1,
            trap_warning_v1=trap_warning_v1,
            hl_ok=bool((m15_struct or {}).get("hl")),
            lh_ok=bool((m15_struct or {}).get("lh")),
        )
    
        meta["position_quality_v2"] = position_quality_v2
        meta["review_conflict_v2"] = review_conflict_v2
        
        # ===== MASTER ENGINE =====
        master_engine_v1 = _master_engine_v1(
            market_state_machine_v1=market_state_machine_v1,
            bias_layers_v1=bias_layers_v1,
            no_trade_zone_v3=no_trade_zone_v3,
            conflict_engine_v1=conflict_engine_v1,
            decision_engine_v1=decision_engine_v1,
            trigger_engine_v2=trigger_engine_v2,
            manual_likelihood_v1=manual_likelihood_v1,
            context_verdict_v1=context_verdict_v1,
        )
        meta["master_engine_v1"] = master_engine_v1
        
        # ===== SIGNAL CONSISTENCY ENGINE =====
        signal_consistency_v1 = _signal_consistency_engine_v1(
            bias_layers_v1=bias_layers_v1,
            market_state_machine_v1=market_state_machine_v1,
            master_engine_v1=master_engine_v1,
            trigger_engine_v2=trigger_engine_v2,
            no_trade_zone_v3=no_trade_zone_v3,
            context_verdict_v1=context_verdict_v1,
            range_pos=range_pos,
        )
        meta["signal_consistency_v1"] = signal_consistency_v1
        # ===== REVEAL ENGINE V1 =====
        reveal_engine_v1 = _reveal_engine_v1(
            bias_layers_v1=bias_layers_v1,
            market_state_machine_v1=market_state_machine_v1,
            close_confirm_v4=close_confirm_v4,
            m15_struct=(m15_struct or {}),
            liquidity_completion_v1=liquidity_completion_v1,
            trigger_engine_v2=trigger_engine_v2,
            range_pos=range_pos,
            vol_ratio=vol_ratio if 'vol_ratio' in locals() else None,
        )
        meta["reveal_engine_v1"] = reveal_engine_v1
        # ===== TRIGGER ENGINE V3 =====
        trigger_engine_v3 = _trigger_engine_v3(
            signal_consistency_v1=signal_consistency_v1,
            trigger_engine_v2=trigger_engine_v2,
            reveal_engine_v1=reveal_engine_v1,
            no_trade_zone_v3=no_trade_zone_v3,
            close_confirm_v4=close_confirm_v4,
            m15_struct=(m15_struct or {}),
            range_pos=range_pos,
            entry_zone_low=entry_zone_low if 'entry_zone_low' in locals() else None,
            entry_zone_high=entry_zone_high if 'entry_zone_high' in locals() else None,
            break_level=break_level if 'break_level' in locals() else None,
            invalidation_level=invalidation_level if 'invalidation_level' in locals() else None,
        )
        meta["trigger_engine_v3"] = trigger_engine_v3
        # ===== FINAL DECISION ENGINE =====
        final_decision_engine_v1 = _final_decision_engine_v1(
            signal_consistency_v1=signal_consistency_v1,
            reveal_engine_v1=reveal_engine_v1,
            trigger_engine_v3=trigger_engine_v3,
            master_engine_v1=master_engine_v1,
            no_trade_zone_v3=no_trade_zone_v3,
        )
        meta["final_decision_engine_v1"] = final_decision_engine_v1
        # =========================================================
        # ===== ELLIOTT PHASE V1 for EARLY RETURN / VNEXT =====
        try:
            meta = base.setdefault("meta", {})
            k = meta.get("key_levels") or {}

            flow1 = meta.get("flow_engine_v1") or {}
            if not flow1:
                gap1 = _detect_session_gap_v1(m15c=m15c, atr15=atr15)
                flow1 = _build_flow_engine_v1(
                    symbol=symbol,
                    m15c=m15c,
                    current_price=(
                        base.get("current_price")
                        or meta.get("current_price")
                        or (_c_val(m15c[-1], "close", None) if m15c else None)
                    ),
                    atr15=atr15,
                    liquidity_map_v1=liquidity_map_v1 if isinstance(liquidity_map_v1, dict) else {},
                    fvg_range_plugin_v1=meta.get("fvg_range_plugin_v1") or {},
                    gap_info_v1=gap1,
                )
                meta["gap_info_v1"] = gap1
                meta["flow_engine_v1"] = flow1

            elliott_phase_v1 = _elliott_phase_v1(
                h4_struct=h4_trend,
                h1_struct=h1_trend,
                m15_struct=(m15_struct or {}).get("tag"),
                pullback_info=meta.get("pullback_engine_v1") or {},
                ema_filter=ema_pack if isinstance(ema_pack, dict) else {},
                flow_engine_v1=flow1,
                zone_action_v1=meta.get("zone_action_v1") or {},
                current_price=(
                    base.get("current_price")
                    or meta.get("current_price")
                    or (_c_val(m15c[-1], "close", None) if m15c else None)
                ),
                range_low=k.get("M15_RANGE_LOW"),
                range_high=k.get("M15_RANGE_HIGH"),
            )

            meta["elliott_phase_v1"] = elliott_phase_v1


        except Exception as e:

            base.setdefault("meta", {})["elliott_phase_v1"] = {
                "ok": False,
                "main_tf": "H1/H4",
                "phase": "ERROR",
                "confidence": 0,
                "meaning": f"Elliott phase lỗi: {e}",
                "action": "Bỏ qua Elliott context",
                "invalid": "n/a",
                "reason": [],
            }
        
        # ===== POST-BREAK CONTINUITY INPUT FIX =====
        meta = base.setdefault("meta", {}) or {}
        k = (meta.get("key_levels") or {}) if isinstance(meta.get("key_levels"), dict) else {}
        struct_for_pbc = (meta.get("structure") or {}) if isinstance(meta.get("structure"), dict) else {}

        # fallback current price an toàn
        try:
            pbc_current_price = None
        
            if 'last_px' in locals() and last_px is not None:
                pbc_current_price = float(last_px)
        
            elif current_price is not None:
                pbc_current_price = float(current_price)
        
            elif m15c and len(m15c) > 0:
                pbc_current_price = float(_c_val(m15c[-1], "close", 0.0) or 0.0)
        
        except Exception:
            try:
                pbc_current_price = float(_c_val(m15c[-1], "close", 0.0) or 0.0) if m15c else None
            except Exception:
                pbc_current_price = None
        
        # fallback BOS / range levels
        pbc_bos = (
            k.get("M15_BOS")
            if k.get("M15_BOS") is not None
            else k.get("M15_BOS_LEVEL")
        )
        
        pbc_range_low = (
            k.get("M15_RANGE_LOW")
            if k.get("M15_RANGE_LOW") is not None
            else k.get("M15_PB_EXTREME")
        )
        
        pbc_range_high = (
            k.get("M15_RANGE_HIGH")
            if k.get("M15_RANGE_HIGH") is not None
            else k.get("H1_HH")
        )

        continuity_v1 = _post_break_continuity_engine_v1(
            current_price=pbc_current_price,
            bos_level=pbc_bos,
            range_low=pbc_range_low,
            range_high=pbc_range_high,
            struct=struct_for_pbc,
            close_confirm_v4=close_confirm_v4 if 'close_confirm_v4' in locals() and isinstance(close_confirm_v4, dict) else {},
            liquidity_map_v1=liquidity_map_v1 if 'liquidity_map_v1' in locals() and isinstance(liquidity_map_v1, dict) else {},
            trigger_engine_v3=trigger_engine_v3 if 'trigger_engine_v3' in locals() and isinstance(trigger_engine_v3, dict) else {},
            absorption_v1=_absorption_v1,
        )

        # ===== DEBUG POST BREAK =====
        #_dbg("PBC checkpoint: built continuity_v1")
        #_dbg(f"PBC RAW: {continuity_v1}")
        
        base.setdefault("meta", {})["post_break_continuity_v1"] = continuity_v1
        meta["post_break_continuity_v1"] = continuity_v1
        

        # ===== SIGNAL CONSISTENCY SYNC WITH FINAL DECISION =====
        try:
            sce1 = meta.get("signal_consistency_v1") or {}
            fd1 = meta.get("final_decision_engine_v1") or {}
            me1 = meta.get("master_engine_v1") or {}
            rv1 = meta.get("reveal_engine_v1") or {}

            fd_decision = str(fd1.get("decision") or "NO_TRADE").upper()
            master_state = str(me1.get("state") or "WAIT").upper()
            revealed = bool(rv1.get("reveal"))
            final_side_sync = str(sce1.get("final_side") or "NONE").upper()

            if fd_decision == "NO_TRADE" or master_state == "NO_TRADE":
                mm1 = meta.get("market_mode_v1") or {}
                mm_mode = str(mm1.get("mode") or "").upper()
                mm_side = str(mm1.get("side") or "NONE").upper()
            
                if mm_mode in ("TREND_DAY_DOWN", "TREND_DAY_UP") and mm_side in ("BUY", "SELL"):
                    # Trend day rõ: không được ép thành CHOP / NO_TRADE cụt ngủn
                    sce1["action_mode"] = "FOLLOW_TREND_CONDITIONAL"
                    sce1["current_move"] = "TREND"
                    sce1["final_side"] = mm_side
                    sce1["context_side"] = mm_side
                    sce1["market_mode"] = mm_mode
                    sce1["narrative"] = (
                        "Trend day đang chạy; chưa có entry đẹp nhưng vẫn theo dõi continuation có điều kiện"
                    )
                else:
                    sce1["action_mode"] = "NO_TRADE"
                    if str(sce1.get("current_move") or "").upper() == "TREND":
                        sce1["current_move"] = "CHOP"
                    if final_side_sync in ("BUY", "SELL"):
                        sce1["narrative"] = "Đúng hướng nhưng market chưa lộ mặt → ưu tiên chờ xác nhận"
                    else:
                        sce1["narrative"] = "Thị trường đang nhiễu / chưa có edge rõ → ưu tiên đứng ngoài"

            meta["signal_consistency_v1"] = sce1
        except Exception:
            pass

        # ===== MASTER ENGINE OVERRIDE =====
        try:
            me1 = meta.get("master_engine_v1") or {}
            mm1 = meta.get("market_mode_v1") or {}
            mm_mode = str(mm1.get("mode") or "").upper()
            mm_side = str(mm1.get("side") or "NONE").upper()
        
            if mm_mode in ("TREND_DAY_DOWN", "TREND_DAY_UP") and mm_side in ("BUY", "SELL"):
                # Không biến trend day thành NO_TRADE.
                # Ý nghĩa đúng: có context, nhưng timing chưa đẹp.
                me1["state"] = "WAIT_TIMING"
                me1["best_side"] = mm_side
                me1["tradeable_final"] = "CONDITIONAL"
                me1["confidence"] = "MEDIUM"
                old_reasons = list(me1.get("reason") or [])
                me1["reason"] = [
                    "context trend day rõ",
                    "chưa có entry confirmation",
                    "chờ continuation/retest thay vì đứng ngoài tuyệt đối",
                ] + old_reasons[:2]
                meta["master_engine_v1"] = me1
        
                base["tradeable"] = False  # vẫn không auto-entry
                base["final_score"] = max(float(base.get("final_score", 0) or 0), 45.0)
                base["final_score"] = min(float(base.get("final_score", 0) or 0), 68.0)
        
            else:
                if me1.get("tradeable_final") is False:
                    base["tradeable"] = False
        
                if str(me1.get("state") or "").upper() == "NO_TRADE":
                    try:
                        base["final_score"] = min(float(base.get("final_score", 0) or 0), 58.0)
                    except Exception:
                        pass
                elif str(me1.get("state") or "").upper() == "WAIT":
                    try:
                        base["final_score"] = min(float(base.get("final_score", 0) or 0), 68.0)
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception as e:
        base.setdefault("meta", {})["vnext_error"] = str(e)
        logger.exception("VNEXT EXCEPTION STACK")

    return base

# =========================
# PRO DESK ENGINE (APPEND ONLY)
# =========================


def _market_state_machine_v1(h1_trend, h4_trend, m15_struct_tag, market_state_v2, liquidation_evt, range_pos):
    if (liquidation_evt or {}).get("ok"):
        return {"state": "LIQUIDATION", "label": "Biến động thanh khoản mạnh"}

    tag = str(m15_struct_tag or "").upper()
    h1t = str(h1_trend or "").lower()
    h4t = str(h4_trend or "").lower()

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    # regime nhiễu / chuyển pha
    if str(market_state_v2 or "").upper() in ("CHOP", "TRANSITION"):
        if h4t == "bearish":
            return {"state": "CHOP_BEARISH", "label": "Nhiễu nhưng nghiêng giảm"}
        if h4t == "bullish":
            return {"state": "CHOP_BULLISH", "label": "Nhiễu nhưng nghiêng tăng"}
        return {"state": "CHOP", "label": "Nhiễu / chuyển pha"}

    if (h1t not in ("bullish", "bearish") and h4t not in ("bullish", "bearish")):
        return {"state": "CHOP", "label": "Nhiễu / chuyển pha"}

    # giữa biên + M15 chưa rõ => không gọi trend quá sớm
    if tag in ("", "TRANSITION", "N/A"):
        if h4t == "bearish" and h1t == "bearish":
            return {"state": "PULLBACK_SELL", "label": "Hồi trong xu hướng giảm"}
        if h4t == "bullish" and h1t == "bullish":
            return {"state": "PULLBACK_BUY", "label": "Hồi trong xu hướng tăng"}
        if h4t == "bearish":
            return {"state": "CHOP_BEARISH", "label": "Nhiễu nhưng nghiêng giảm"}
        if h4t == "bullish":
            return {"state": "CHOP_BULLISH", "label": "Nhiễu nhưng nghiêng tăng"}
        if rp is not None and 0.35 <= rp <= 0.65:
            return {"state": "CHOP", "label": "Nhiễu / chuyển pha"}

    if h1t == "bullish" and h4t == "bullish":
        if "LL" in tag or "LH" in tag or (rp is not None and rp >= 0.65):
            return {"state": "PULLBACK_BUY", "label": "Hồi trong xu hướng tăng"}
        return {"state": "TREND_UP", "label": "Xu hướng tăng"}

    if h1t == "bearish" and h4t == "bearish":
        if "HH" in tag or "HL" in tag or (rp is not None and rp <= 0.35):
            return {"state": "PULLBACK_SELL", "label": "Hồi trong xu hướng giảm"}
        return {"state": "TREND_DOWN", "label": "Xu hướng giảm"}

    if h4t == "bearish":
        return {"state": "CHOP_BEARISH", "label": "Nhiễu nhưng nghiêng giảm"}
    if h4t == "bullish":
        return {"state": "CHOP_BULLISH", "label": "Nhiễu nhưng nghiêng tăng"}

    return {"state": "CHOP", "label": "Nhiễu / chuyển pha"}


    if (h1t not in ("bullish", "bearish") and h4t not in ("bullish", "bearish")):
        return {"state": "CHOP", "label": "Nhiễu / chuyển pha"}

    if tag in ("", "TRANSITION", "N/A") and rp is not None and 0.35 <= rp <= 0.65:
        return {"state": "CHOP", "label": "Nhiễu / chuyển pha"}

    if h1t == "bullish" and h4t == "bullish":
        if "LL" in tag or "LH" in tag:
            return {"state": "PULLBACK_BUY", "label": "Hồi trong xu hướng tăng"}
        return {"state": "TREND_UP", "label": "Xu hướng tăng"}

    if h1t == "bearish" and h4t == "bearish":
        if "HH" in tag or "HL" in tag:
            return {"state": "PULLBACK_SELL", "label": "Hồi trong xu hướng giảm"}
        return {"state": "TREND_DOWN", "label": "Xu hướng giảm"}

    return {"state": "CHOP", "label": "Nhiễu / chuyển pha"}


def _bias_layers_v1(h1_trend, h4_trend, m15_struct_tag, context_verdict):
    if h1_trend == "bullish" and h4_trend == "bullish":
        htf_bias = "BUY"
    elif h1_trend == "bearish" and h4_trend == "bearish":
        htf_bias = "SELL"
    else:
        htf_bias = "MIXED"

    tag = str(m15_struct_tag or "").upper()
    if "LL" in tag or "LH" in tag:
        mtf_bias = "SELL_PULLBACK"
    elif "HH" in tag or "HL" in tag:
        mtf_bias = "BUY_PULLBACK"
    else:
        mtf_bias = "WAIT"

    verdict_state = str((context_verdict or {}).get("state") or "")
    entry_bias = "READY" if "CONTINUATION" in verdict_state else "WAIT"

    return {
        "htf_bias": htf_bias,
        "mtf_bias": mtf_bias,
        "entry_bias": entry_bias,
    }


def _no_trade_zone_v3(range_pos, liquidity_done, close_confirm_v4, m15_struct_tag, trap_warning):
    reasons = []

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if rp is not None and 0.35 <= rp <= 0.65:
        reasons.append("giữa biên độ")

    if str((liquidity_done or {}).get("state") or "NO") == "NO":
        reasons.append("chưa có thanh khoản")

    if str((close_confirm_v4 or {}).get("strength") or "NO") in ("NO", "N/A"):
        reasons.append("chưa có confirm")

    if str(m15_struct_tag or "").upper() in ("", "TRANSITION", "N/A"):
        reasons.append("M15 chưa rõ cấu trúc")

    if (trap_warning or {}).get("active"):
        reasons.append("trap risk")

    return {"active": len(reasons) >= 2, "reasons": reasons}


def _decision_engine_v1(no_trade_zone, context_verdict, entry_sniper, close_confirm_v4, manual_likelihood):
    if (no_trade_zone or {}).get("active"):
        return {"decision": "STAND ASIDE", "reason": "No-trade zone"}

    if str((entry_sniper or {}).get("trigger")).upper() in ("READY", "TRIGGERED"):
        return {"decision": "MANUAL STRIKE", "reason": "Có trigger"}

    return {"decision": "WAIT", "reason": "Chưa đủ điều kiện"}


def _wait_for_engine_v1(bias_side, playbook_v2, key_levels):
    lines = []

    zl = (playbook_v2 or {}).get("zone_low")
    zh = (playbook_v2 or {}).get("zone_high")

    if zl and zh:
        lines.append(f"Chờ vùng {zl} – {zh}")

    if bias_side == "BUY":
        lv = (key_levels or {}).get("M15_RANGE_HIGH")
        if lv:
            lines.append(f"Hoặc break {lv}")
    elif bias_side == "SELL":
        lv = (key_levels or {}).get("M15_RANGE_LOW")
        if lv:
            lines.append(f"Hoặc break {lv}")

    return {"lines": lines}
# =========================
# CONFLICT ENGINE + SUGGESTION BLOCK
# =========================

def _conflict_engine_v1(
    bias_layers_v1: dict | None,
    context_verdict_v1: dict | None,
    liquidity_completion_v1: dict | None,
    trap_warning_v1: dict | None,
    range_pos: float | None,
    fib_confluence_v1: dict | None,
) -> dict:
    reasons = []
    severity = 0

    bl = bias_layers_v1 or {}
    cv = context_verdict_v1 or {}
    liq = liquidity_completion_v1 or {}
    trap = trap_warning_v1 or {}
    fib = fib_confluence_v1 or {}

    htf_bias = str(bl.get("htf_bias") or "MIXED").upper()
    mtf_bias = str(bl.get("mtf_bias") or "WAIT").upper()
    entry_bias = str(bl.get("entry_bias") or "WAIT").upper()
    cv_state = str(cv.get("state") or "").upper()
    liq_state = str(liq.get("state") or "NO").upper()

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    # 1) HTF vs location
    if htf_bias == "SELL" and rp is not None and rp <= 0.30:
        reasons.append("HTF SELL nhưng giá đang ở vùng thấp")
        severity += 1

    if htf_bias == "BUY" and rp is not None and rp >= 0.70:
        reasons.append("HTF BUY nhưng giá đang ở vùng cao")
        severity += 1

    # 2) Context vs entry
    if "CONTINUATION" in cv_state and entry_bias != "READY":
        reasons.append("Đúng cảnh nhưng entry chưa sẵn sàng")
        severity += 1

    # 3) Liquidity conflict
    if liq_state == "NO":
        reasons.append("thanh khoản chưa hoàn tất")
        severity += 1

    # 4) Trap conflict
    if (trap or {}).get("active"):
        reasons.append("trap risk đang hiện diện")
        severity += 1

    # 5) Fib conflict
    if not bool(fib.get("ok")):
        reasons.append("chưa có fib confluence rõ")
        severity += 1

    # 6) Bias mismatch
    if htf_bias == "SELL" and "BUY_PULLBACK" in mtf_bias:
        reasons.append("HTF SELL nhưng MTF đang hồi tăng")
        severity += 1

    if htf_bias == "BUY" and "SELL_PULLBACK" in mtf_bias:
        reasons.append("HTF BUY nhưng MTF đang hồi giảm")
        severity += 1

    if severity >= 4:
        verdict = "HIGH CONFLICT"
    elif severity >= 2:
        verdict = "MEDIUM CONFLICT"
    else:
        verdict = "LOW CONFLICT"

    return {
        "active": severity >= 2,
        "severity": severity,
        "verdict": verdict,
        "reasons": reasons[:5],
    }


def _suggestion_block_v1(
    symbol: str,
    bias_side: str | None,
    decision_engine_v1: dict | None,
    no_trade_zone_v3: dict | None,
    playbook_v2: dict | None,
    entry_sniper: dict | None,
    manual_likelihood_v1: dict | None,
    conflict_engine_v1: dict | None,
    m15_struct: dict | None,
) -> dict:
    out = {
        "title": "NO TRADE",
        "lines": [],
    }

    de = decision_engine_v1 or {}
    ntz = no_trade_zone_v3 or {}
    pb = playbook_v2 or {}
    sniper = entry_sniper or {}
    ml = manual_likelihood_v1 or {}
    cf = conflict_engine_v1 or {}
    m15s = m15_struct or {}

    decision = str(de.get("decision") or "WAIT").upper()
    conflict_active = bool(cf.get("active"))
    trap_risk = int(ml.get("trap_risk") or 0)
    zone_low = pb.get("zone_low")
    zone_high = pb.get("zone_high")
    sniper_trigger = str(sniper.get("trigger") or "NONE").upper()
    bos = m15s.get("bos_level")
    invalid = m15s.get("invalid_level") or m15s.get("pullback_invalid")

    if ntz.get("active") or conflict_active or decision == "STAND ASIDE":
        out["title"] = "NO TRADE"
        out["lines"] = [
            "Chưa có lợi thế rõ để vào lệnh mới.",
            "Ưu tiên đứng ngoài và chờ market lộ mặt thêm.",
        ]
        if zone_low is not None and zone_high is not None:
            out["lines"].append(f"Chờ phản ứng tại vùng {_fmt(zone_low)} – {_fmt(zone_high)}")
        return out

    side = str(bias_side or "NONE").upper()
    if side not in ("BUY", "SELL"):
        side = str(ml.get("best_side") or "NONE").upper()

    if decision == "MANUAL STRIKE":
        out["title"] = f"{side} SETUP"
    else:
        out["title"] = f"WAIT {side}"

    if zone_low is not None and zone_high is not None:
        out["lines"].append(f"Zone ưu tiên: {_fmt(zone_low)} – {_fmt(zone_high)}")

    if sniper_trigger in ("READY", "TRIGGERED"):
        out["lines"].append(f"Trigger: {sniper_trigger}")
    else:
        out["lines"].append("Trigger: chờ M15 reject / displacement / follow-through")

    if bos is not None:
        out["lines"].append(f"Break xác nhận: {_fmt(bos)}")

    if invalid is not None:
        out["lines"].append(f"Invalidation: {_fmt(invalid)}")

    if trap_risk >= 55:
        out["lines"].append("Risk: HIGH")
    elif trap_risk >= 35:
        out["lines"].append("Risk: MEDIUM")
    else:
        out["lines"].append("Risk: LOW")

    out["lines"] = out["lines"][:5]
    return out    

# =========================
# TRIGGER ENGINE V2
# =========================

def _trigger_engine_v2(
    bias_side: str | None,
    context_verdict_v1: dict | None,
    conflict_engine_v1: dict | None,
    no_trade_zone_v3: dict | None,
    entry_sniper: dict | None,
    playbook_v2: dict | None,
    fib_confluence_v1: dict | None,
    liquidity_completion_v1: dict | None,
    close_confirm_v4: dict | None,
    m15_struct: dict | None,
    range_pos: float | None,
    current_price: float | None,
    atr15: float | None,
) -> dict:
    out = {
        "state": "WAIT",
        "entry_type": "NONE",
        "quality": "LOW",
        "reason": [],
        "trigger_ok": False,
        "location_ok": False,
        "confirm_ok": False,
        "follow_ok": False,
    }

    cv = context_verdict_v1 or {}
    cf = conflict_engine_v1 or {}
    ntz = no_trade_zone_v3 or {}
    sniper = entry_sniper or {}
    pb = playbook_v2 or {}
    fib = fib_confluence_v1 or {}
    liq = liquidity_completion_v1 or {}
    ccv4 = close_confirm_v4 or {}
    m15s = m15_struct or {}

    sniper_dir = str(sniper.get("direction") or "NONE").upper()
    sniper_trigger = str(sniper.get("trigger") or "NONE").upper()
    sniper_candle = str(sniper.get("signal_candle") or sniper.get("direction_candle") or "NONE").upper()

    cv_state = str(cv.get("state") or "").upper()
    liq_state = str(liq.get("state") or "NO").upper()
    confirm_strength = str(ccv4.get("strength") or "NO").upper()
    m15_tag = str(m15s.get("tag") or "N/A").upper()

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    px = None
    try:
        px = float(current_price) if current_price is not None else None
    except Exception:
        px = None

    zl = pb.get("zone_low")
    zh = pb.get("zone_high")
    try:
        zl = float(zl) if zl is not None else None
        zh = float(zh) if zh is not None else None
    except Exception:
        zl, zh = None, None

    pad = 0.0
    try:
        pad = max(0.0, float(atr15 or 0.0) * 0.15)
    except Exception:
        pad = 0.0

    # 1) trigger raw
    if sniper_trigger in ("READY", "TRIGGERED"):
        out["trigger_ok"] = True
    else:
        out["reason"].append("chưa có trigger sniper")

    # 2) context
    context_ok = False
    if bias_side == "BUY" and cv_state in ("BUY_CONTINUATION", "BUY_BIAS", "NEUTRAL", "POST_LIQUIDATION"):
        context_ok = True
    elif bias_side == "SELL" and cv_state in ("SELL_CONTINUATION", "SELL_BIAS", "NEUTRAL", "POST_LIQUIDATION"):
        context_ok = True
    elif "CONTINUATION" in cv_state:
        context_ok = True

    if not context_ok:
        out["reason"].append("context chưa ủng hộ đủ mạnh")

    # 3) location
    location_ok = False
    if zl is not None and zh is not None and px is not None:
        if (zl - pad) <= px <= (zh + pad):
            location_ok = True

    if not location_ok and fib.get("ok") and px is not None:
        fzl = fib.get("zone_low")
        fzh = fib.get("zone_high")
        try:
            fzl = float(fzl) if fzl is not None else None
            fzh = float(fzh) if fzh is not None else None
        except Exception:
            fzl, fzh = None, None
        if fzl is not None and fzh is not None and (fzl - pad) <= px <= (fzh + pad):
            location_ok = True

    # fallback by range position
    if not location_ok and rp is not None:
        if bias_side == "BUY" and rp <= 0.35:
            location_ok = True
        elif bias_side == "SELL" and rp >= 0.65:
            location_ok = True

    out["location_ok"] = location_ok
    if not location_ok:
        out["reason"].append("location chưa đẹp")

    # 4) confirm
    confirm_ok = confirm_strength in ("WEAK", "STRONG")
    out["confirm_ok"] = confirm_ok
    if not confirm_ok:
        out["reason"].append("chưa có close confirm")

    # 5) liquidity follow
    follow_ok = False
    if liq_state in ("YES", "PARTIAL"):
        follow_ok = True
    elif sniper_trigger == "TRIGGERED" and confirm_strength == "STRONG":
        follow_ok = True

    # structure alignment
    if bias_side == "BUY" and ("HL" in m15_tag or "HH" in m15_tag or m15_tag == "TRANSITION"):
        follow_ok = follow_ok or True
    if bias_side == "SELL" and ("LH" in m15_tag or "LL" in m15_tag or m15_tag == "TRANSITION"):
        follow_ok = follow_ok or True

    out["follow_ok"] = follow_ok
    if not follow_ok:
        out["reason"].append("chưa có follow-through đủ rõ")

    # 6) conflict filter
    if (cf.get("severity") or 0) >= 4:
        out["reason"].append("conflict cao")
    if (ntz.get("active") is True):
        out["reason"].append("đang trong no-trade zone")

    # final state
    hard_block = bool(ntz.get("active")) or (cf.get("severity") or 0) >= 4

    if out["trigger_ok"] and location_ok and confirm_ok and follow_ok and not hard_block:
        out["state"] = "TRIGGERED"
        out["entry_type"] = "SNIPER"
        out["quality"] = "HIGH"
    elif out["trigger_ok"] and location_ok and not hard_block:
        out["state"] = "READY"
        out["entry_type"] = "SETUP_FORMING"
        out["quality"] = "MEDIUM"
    else:
        out["state"] = "WAIT"
        out["entry_type"] = "NONE"
        out["quality"] = "LOW"

    out["reason"] = out["reason"][:5]
    return out
# =========================
# EXIT ENGINE V2
# =========================

def _exit_engine_v2(
    review_side: str | None,
    current_price: float | None,
    invalidation_level: float | None,
    hl_ok: bool,
    lh_ok: bool,
    break_up: bool,
    break_dn: bool,
    final_score: float | int | None,
    conflict_engine_v1: dict | None,
    trigger_engine_v2: dict | None,
    range_pos: float | None,
) -> dict:
    out = {
        "state": "HOLD_LIGHT",
        "decision": "Giữ nhẹ, không add",
        "reason": [],
        "invalidation_hit": False,
        "structure_status": "UNKNOWN",
        "risk_level": "MEDIUM",
        "add_allowed": False,
    }

    side = str(review_side or "").upper()
    cf = conflict_engine_v1 or {}
    tg2 = trigger_engine_v2 or {}

    try:
        px = float(current_price) if current_price is not None else None
    except Exception:
        px = None

    try:
        inv = float(invalidation_level) if invalidation_level is not None else None
    except Exception:
        inv = None

    try:
        score = float(final_score or 0)
    except Exception:
        score = 0.0

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    invalidation_hit = False
    if side == "BUY" and px is not None and inv is not None and px < inv:
        invalidation_hit = True
    if side == "SELL" and px is not None and inv is not None and px > inv:
        invalidation_hit = True

    out["invalidation_hit"] = invalidation_hit

    if side == "BUY":
        if hl_ok and break_up:
            out["structure_status"] = "CONFIRMED_BUY"
        elif hl_ok:
            out["structure_status"] = "HL_ONLY"
        else:
            out["structure_status"] = "NO_HL"
    elif side == "SELL":
        if lh_ok and break_dn:
            out["structure_status"] = "CONFIRMED_SELL"
        elif lh_ok:
            out["structure_status"] = "LH_ONLY"
        else:
            out["structure_status"] = "NO_LH"

    conflict_sev = int(cf.get("severity") or 0)
    trigger_state = str(tg2.get("state") or "WAIT").upper()

    risk_points = 0
    if invalidation_hit:
        risk_points += 4
    if conflict_sev >= 4:
        risk_points += 2
    elif conflict_sev >= 2:
        risk_points += 1

    if side == "BUY" and not hl_ok:
        risk_points += 1
    if side == "SELL" and not lh_ok:
        risk_points += 1

    if score < 35:
        risk_points += 1

    if side == "BUY" and rp is not None and rp >= 0.75:
        risk_points += 1
    if side == "SELL" and rp is not None and rp <= 0.25:
        risk_points += 1

    if risk_points >= 5:
        out["risk_level"] = "HIGH"
    elif risk_points >= 3:
        out["risk_level"] = "MEDIUM"
    else:
        out["risk_level"] = "LOW"

    if invalidation_hit:
        out["state"] = "EXIT_NOW"
        out["decision"] = "Thoát ngay / cắt mạnh"
        out["reason"].append("đã chạm invalidation")
        out["add_allowed"] = False
        return out

    if side == "BUY":
        if hl_ok and break_up and conflict_sev < 2:
            out["state"] = "HOLD"
            out["decision"] = "Có thể giữ tiếp"
            out["reason"].append("BUY đã có HL + break_up")
            out["add_allowed"] = True
        elif hl_ok and not break_up:
            out["state"] = "HOLD_LIGHT"
            out["decision"] = "Giữ nhẹ, chưa add"
            out["reason"].append("đã có HL nhưng chưa break_up")
        elif not hl_ok and conflict_sev >= 2:
            out["state"] = "REDUCE_RISK"
            out["decision"] = "Ưu tiên giảm size / giữ rất ngắn"
            out["reason"].append("BUY chưa có HL")
            out["reason"].append("conflict còn hiện diện")
        else:
            out["state"] = "CUT_SOON"
            out["decision"] = "Không add, sẵn sàng thoát nếu không cải thiện"
            out["reason"].append("BUY chưa có HL")
    elif side == "SELL":
        if lh_ok and break_dn and conflict_sev < 2:
            out["state"] = "HOLD"
            out["decision"] = "Có thể giữ tiếp"
            out["reason"].append("SELL đã có LH + break_dn")
            out["add_allowed"] = True
        elif lh_ok and not break_dn:
            out["state"] = "HOLD_LIGHT"
            out["decision"] = "Giữ nhẹ, chưa add"
            out["reason"].append("đã có LH nhưng chưa break_dn")
        elif not lh_ok and conflict_sev >= 2:
            out["state"] = "REDUCE_RISK"
            out["decision"] = "Ưu tiên giảm size / giữ rất ngắn"
            out["reason"].append("SELL chưa có LH")
            out["reason"].append("conflict còn hiện diện")
        else:
            out["state"] = "CUT_SOON"
            out["decision"] = "Không add, sẵn sàng thoát nếu không cải thiện"
            out["reason"].append("SELL chưa có LH")

    if side == "BUY" and rp is not None and rp >= 0.75:
        out["reason"].append("BUY đang ở vùng cao")
    if side == "SELL" and rp is not None and rp <= 0.25:
        out["reason"].append("SELL đang ở vùng thấp")

    if trigger_state == "WAIT":
        out["reason"].append("chưa có trigger hỗ trợ giữ lệnh")

    out["reason"] = out["reason"][:5]
    return out
# =========================
# MASTER ENGINE
# =========================

def _master_engine_v1(
    market_state_machine_v1: dict | None,
    bias_layers_v1: dict | None,
    no_trade_zone_v3: dict | None,
    conflict_engine_v1: dict | None,
    decision_engine_v1: dict | None,
    trigger_engine_v2: dict | None,
    manual_likelihood_v1: dict | None,
    context_verdict_v1: dict | None,
) -> dict:
    out = {
        "state": "WAIT",
        "tradeable_final": False,
        "confidence": "LOW",
        "reason": [],
    }

    ms = market_state_machine_v1 or {}
    bl = bias_layers_v1 or {}
    ntz = no_trade_zone_v3 or {}
    cf = conflict_engine_v1 or {}
    de = decision_engine_v1 or {}
    tg = trigger_engine_v2 or {}
    ml = manual_likelihood_v1 or {}
    cv = context_verdict_v1 or {}

    conflict_sev = int(cf.get("severity") or 0)
    ntz_active = bool(ntz.get("active"))
    trigger_state = str(tg.get("state") or "WAIT").upper()
    trigger_quality = str(tg.get("quality") or "LOW").upper()
    de_state = str(de.get("decision") or "WAIT").upper()
    cv_state = str(cv.get("state") or "").upper()
    htf_bias = str(bl.get("htf_bias") or "MIXED").upper()
    mtf_bias = str(bl.get("mtf_bias") or "WAIT").upper()
    buy_lk = int(ml.get("buy_likelihood") or 0)
    sell_lk = int(ml.get("sell_likelihood") or 0)
    trap = int(ml.get("trap_risk") or 0)

    best_side = "BUY" if buy_lk > sell_lk else ("SELL" if sell_lk > buy_lk else "NONE")

    # 1) hard block
    if ntz_active:
        out["state"] = "NO_TRADE"
        out["tradeable_final"] = False
        out["confidence"] = "HIGH"
        out["reason"].append("no-trade zone đang active")

    if conflict_sev >= 4:
        out["state"] = "NO_TRADE"
        out["tradeable_final"] = False
        out["confidence"] = "HIGH"
        out["reason"].append("conflict quá cao")

    # 2) trigger-led promotion
    if out["state"] != "NO_TRADE":
        if trigger_state == "TRIGGERED" and trigger_quality == "HIGH" and trap < 55:
            out["state"] = "MANUAL_STRIKE"
            out["tradeable_final"] = True
            out["confidence"] = "HIGH"
            out["reason"].append("trigger sniper đã xác nhận")

        elif trigger_state in ("READY", "TRIGGERED") and trigger_quality in ("MEDIUM", "HIGH") and trap < 60:
            out["state"] = "READY"
            out["tradeable_final"] = False
            out["confidence"] = "MEDIUM"
            out["reason"].append("setup đang hình thành")

    # 3) context-led fallback
    if out["state"] == "WAIT":
        if "CONTINUATION" in cv_state and best_side in ("BUY", "SELL"):
            out["state"] = "WAIT"
            out["tradeable_final"] = False
            out["confidence"] = "MEDIUM"
            out["reason"].append("đúng cảnh nhưng chưa đủ trigger")
        elif de_state == "STAND ASIDE":
            out["state"] = "NO_TRADE"
            out["tradeable_final"] = False
            out["confidence"] = "MEDIUM"
            out["reason"].append("decision engine đang chặn")
        else:
            out["state"] = "WAIT"
            out["tradeable_final"] = False
            out["confidence"] = "LOW"
            out["reason"].append("chưa có lợi thế rõ")

    # 4) bias sanity check
    if htf_bias == "BUY" and "SELL_PULLBACK" in mtf_bias:
        out["reason"].append("HTF BUY nhưng MTF đang hồi giảm")
    if htf_bias == "SELL" and "BUY_PULLBACK" in mtf_bias:
        out["reason"].append("HTF SELL nhưng MTF đang hồi tăng")

    # 5) likelihood sanity
    if buy_lk == sell_lk:
        out["reason"].append("buy/sell likelihood cân bằng")
    if trap >= 55:
        out["reason"].append("trap risk cao")

    out["best_side"] = best_side
    out["market_state"] = str(ms.get("state") or "UNKNOWN")
    out["reason"] = out["reason"][:6]
    return out
def _normalize_trade_side(v) -> str:
    s = str(v or "").strip().upper()
    if s in ("BUY", "LONG", "MUA"):
        return "BUY"
    if s in ("SELL", "SHORT", "BÁN", "BAN"):
        return "SELL"
    return ""

def _resolve_review_side(sig: dict | None, fallback_bias: str | None = None) -> str:
    sig = sig or {}

    # 1) explicit review side
    side = _normalize_trade_side(sig.get("review_side"))
    if side:
        return side

    # 2) common generic side fields
    side = _normalize_trade_side(sig.get("side"))
    if side:
        return side

    # 3) sometimes recommendation is used as side
    side = _normalize_trade_side(sig.get("recommendation"))
    if side:
        return side

    # 4) try nested meta
    meta = sig.get("meta") or {}
    side = _normalize_trade_side(meta.get("review_side"))
    if side:
        return side

    # 5) final fallback only
    return _normalize_trade_side(fallback_bias)
def _position_quality_v2(
    review_side: str | None,
    final_score: float | int | None,
    range_pos: float | None,
    hl_ok: bool,
    lh_ok: bool,
    break_up: bool,
    break_dn: bool,
    conflict_engine_v1: dict | None,
    trigger_engine_v2: dict | None,
) -> dict:
    side = str(review_side or "").upper()
    score = float(final_score or 0)
    cf = conflict_engine_v1 or {}
    tg = trigger_engine_v2 or {}

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    conflict_sev = int(cf.get("severity") or 0)
    trigger_state = str(tg.get("state") or "WAIT").upper()

    pos_score = 0

    # base score
    if score >= 55:
        pos_score += 2
    elif score >= 35:
        pos_score += 1

    # side-sensitive location
    if side == "BUY":
        if rp is not None and rp <= 0.35:
            pos_score += 2
        elif rp is not None and rp >= 0.75:
            pos_score -= 2
    elif side == "SELL":
        if rp is not None and rp >= 0.65:
            pos_score += 2
        elif rp is not None and rp <= 0.25:
            pos_score -= 2

    # structure
    if side == "BUY":
        if hl_ok:
            pos_score += 1
        if break_up:
            pos_score += 1
    elif side == "SELL":
        if lh_ok:
            pos_score += 1
        if break_dn:
            pos_score += 1

    # conflict
    if conflict_sev >= 4:
        pos_score -= 2
    elif conflict_sev >= 2:
        pos_score -= 1

    # trigger support
    if trigger_state == "TRIGGERED":
        pos_score += 1
    elif trigger_state == "WAIT":
        pos_score -= 1

    if pos_score >= 4:
        quality = "STRONG"
        reason = "vị trí tốt + cấu trúc hỗ trợ + conflict thấp"
    elif pos_score >= 1:
        quality = "MID"
        reason = "có một phần lợi thế nhưng chưa đủ sạch để tăng rủi ro"
    else:
        quality = "WEAK"
        reason = "vị trí/cấu trúc chưa đủ đẹp hoặc conflict còn cao"

    return {
        "quality": quality,
        "reason": reason,
        "score_internal": pos_score,
    }

def _review_conflict_v2(
    review_side: str | None,
    range_pos: float | None,
    liquidity_completion_v1: dict | None,
    trap_warning_v1: dict | None,
    hl_ok: bool,
    lh_ok: bool,
) -> dict:
    side = str(review_side or "").upper()
    liq = liquidity_completion_v1 or {}
    trap = trap_warning_v1 or {}

    reasons = []
    sev = 0

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if str((liq.get("state") or "NO")).upper() == "NO":
        reasons.append("thanh khoản chưa hoàn tất")
        sev += 1

    if (trap or {}).get("active"):
        reasons.append("trap risk đang hiện diện")
        sev += 1

    if side == "BUY" and not hl_ok:
        reasons.append("BUY chưa có HL xác nhận")
        sev += 1
    if side == "SELL" and not lh_ok:
        reasons.append("SELL chưa có LH xác nhận")
        sev += 1

    if side == "BUY" and rp is not None and rp >= 0.75:
        reasons.append("BUY đang ở vùng cao")
        sev += 1
    if side == "SELL" and rp is not None and rp <= 0.25:
        reasons.append("SELL đang ở vùng thấp")
        sev += 1

    if sev >= 4:
        verdict = "HIGH CONFLICT"
    elif sev >= 2:
        verdict = "MEDIUM CONFLICT"
    else:
        verdict = "LOW CONFLICT"

    return {
        "active": sev >= 2,
        "severity": sev,
        "verdict": verdict,
        "reasons": reasons[:5],
    }

def _score_to_grade_v2(score) -> str:
    try:
        s = float(score or 0)
    except Exception:
        s = 0.0

    if s >= 85:
        return "A"
    if s >= 70:
        return "B"
    if s >= 50:
        return "C"
    if s >= 30:
        return "D"
    return "E"
def _final_score_review(
    side,
    gate,
    pos,
    actions,
    playbook,
    no_trade_zone,
    htf_pressure_v4,
):
    score = 50
    reasons = []

    # ------------------
    # STRUCTURE
    # ------------------
    if gate.get("strong"):
        score += 10
        reasons.append("structure tốt")
    else:
        score -= 10
        reasons.append("structure yếu")

    # ------------------
    # POSITION
    # ------------------
    if pos.get("zone") == "HIGH":
        if side == "SELL":
            score += 5
        else:
            score -= 10
    elif pos.get("zone") == "LOW":
        if side == "BUY":
            score += 5
        else:
            score -= 10

    # ------------------
    # NO TRADE ZONE
    # ------------------
    if no_trade_zone.get("active"):
        score -= 20
        reasons.append("no-trade zone")

    # ------------------
    # HTF CONFLICT
    # ------------------
    htf_state = str(htf_pressure_v4.get("state") or "").upper()
    if side == "SELL" and "BULLISH" in htf_state:
        score -= 10
        reasons.append("ngược HTF")
    if side == "BUY" and "BEARISH" in htf_state:
        score -= 10
        reasons.append("ngược HTF")

    # clamp
    score = max(0, min(100, score))

    tradeable = "YES" if score >= 60 else "NO"

    return score, tradeable, reasons, []
# =========================
# SIGNAL CONSISTENCY ENGINE
# =========================

def _signal_consistency_engine_v1(
    bias_layers_v1: dict | None,
    market_state_machine_v1: dict | None,
    master_engine_v1: dict | None,
    trigger_engine_v2: dict | None,
    no_trade_zone_v3: dict | None,
    context_verdict_v1: dict | None,
    range_pos: float | None,
) -> dict:
    bl = bias_layers_v1 or {}
    ms = market_state_machine_v1 or {}
    me = master_engine_v1 or {}
    tg = trigger_engine_v2 or {}
    ntz = no_trade_zone_v3 or {}
    cv = context_verdict_v1 or {}

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    htf_bias = str(bl.get("htf_bias") or "MIXED").upper()
    mtf_bias = str(bl.get("mtf_bias") or "WAIT").upper()
    master_state = str(me.get("state") or "WAIT").upper()
    trigger_state = str(tg.get("state") or "WAIT").upper()
    market_state = str(ms.get("state") or "").upper()
    cv_state = str(cv.get("state") or "").upper()

    final_side = "NONE"
    current_move = "CHOP"
    action_mode = "NO_TRADE"
    allow_buy_text = True
    allow_sell_text = True
    reasons = []

    # -------------------------
    # 1) final side = theo HTF trước
    # -------------------------
    if htf_bias in ("BUY", "SELL"):
        final_side = htf_bias
    else:
        final_side = str(me.get("best_side") or "NONE").upper()

    # -------------------------
    # 2) current move
    # -------------------------
    if "CHOP" in market_state:
        current_move = "CHOP"
    elif "TRANSITION" in market_state:
        current_move = "TRANSITION"
    elif final_side == "SELL" and ("BUY_PULLBACK" in mtf_bias or "PULLBACK" in mtf_bias):
        current_move = "PULLBACK"
    elif final_side == "BUY" and ("SELL_PULLBACK" in mtf_bias or "PULLBACK" in mtf_bias):
        current_move = "PULLBACK"
    else:
        current_move = "TREND"

    # fallback theo context/range nếu market state chưa rõ
    if current_move == "CHOP" and rp is not None and 0.40 <= rp <= 0.60:
        reasons.append("đang ở giữa biên")

    # -------------------------
    # 3) action mode
    # -------------------------
    if bool(ntz.get("active")) or master_state == "NO_TRADE":
        action_mode = "NO_TRADE"
        reasons.append("no-trade zone đang active")
    else:
        if final_side == "SELL":
            if trigger_state == "TRIGGERED":
                action_mode = "MANUAL_STRIKE"
            else:
                action_mode = "WAIT_SELL"
        elif final_side == "BUY":
            if trigger_state == "TRIGGERED":
                action_mode = "MANUAL_STRIKE"
            else:
                action_mode = "WAIT_BUY"
        else:
            action_mode = "NO_TRADE"

    # -------------------------
    # 4) text allowance
    # -------------------------
    if final_side == "SELL":
        allow_buy_text = False
        allow_sell_text = True
    elif final_side == "BUY":
        allow_buy_text = True
        allow_sell_text = False
    else:
        allow_buy_text = False
        allow_sell_text = False

    # nhưng nếu action_mode = NO_TRADE thì không được in câu vào lệnh trực tiếp
    if action_mode == "NO_TRADE":
        allow_buy_text = False
        allow_sell_text = False

    # -------------------------
    # 5) narrative
    # -------------------------
    if final_side == "SELL" and current_move == "PULLBACK":
        narrative = "Đây là nhịp hồi trong bối cảnh giảm → chỉ chờ SELL khi có xác nhận"
    elif final_side == "BUY" and current_move == "PULLBACK":
        narrative = "Đây là nhịp điều chỉnh trong bối cảnh tăng → chỉ chờ BUY khi có xác nhận"
    elif final_side == "SELL" and current_move == "TREND":
        narrative = "Xu hướng chính vẫn nghiêng giảm → ưu tiên SELL theo xác nhận"
    elif final_side == "BUY" and current_move == "TREND":
        narrative = "Xu hướng chính vẫn nghiêng tăng → ưu tiên BUY theo xác nhận"
    elif current_move == "CHOP":
        narrative = "Thị trường đang nhiễu / đi giữa biên → ưu tiên đứng ngoài"
    else:
        narrative = "Chưa có hướng đủ rõ → ưu tiên đứng ngoài"

    return {
        "final_side": final_side,
        "current_move": current_move,
        "action_mode": action_mode,
        "allow_buy_text": allow_buy_text,
        "allow_sell_text": allow_sell_text,
        "narrative": narrative,
        "reasons": reasons[:4],
    }

# =========================
# REVEAL ENGINE V1
# =========================
def _reveal_engine_v1(
    bias_layers_v1: dict | None,
    market_state_machine_v1: dict | None,
    close_confirm_v4: dict | None,
    m15_struct: dict | None,
    liquidity_completion_v1: dict | None,
    trigger_engine_v2: dict | None,
    range_pos: float | None,
    vol_ratio: float | None = None,
) -> dict:
    bl = bias_layers_v1 or {}
    ms = market_state_machine_v1 or {}
    cc4 = close_confirm_v4 or {}
    m15s = m15_struct or {}
    liq1 = liquidity_completion_v1 or {}
    tg2 = trigger_engine_v2 or {}

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    try:
        vr = float(vol_ratio) if vol_ratio is not None else None
    except Exception:
        vr = None

    htf_bias = str(bl.get("htf_bias") or "MIXED").upper()
    market_state = str(ms.get("state") or "").upper()
    cc_strength = str(cc4.get("strength") or "NO").upper()
    liq_state = str(liq1.get("state") or "NO").upper()
    tg_state = str(tg2.get("state") or "WAIT").upper()

    break_up = bool(m15s.get("break_up"))
    break_dn = bool(m15s.get("break_dn"))

    reveal = False
    direction = "NONE"
    quality = "LOW"
    reasons = []

    # market chưa lộ mặt nếu đang CHOP mạnh
    if "CHOP" in market_state:
        reasons.append("market còn nhiễu")
    if rp is not None and 0.40 <= rp <= 0.60:
        reasons.append("đang ở giữa biên")

    # reveal SELL
    if htf_bias == "SELL":
        if break_dn and cc_strength in ("WEAK", "STRONG"):
            reveal = True
            direction = "SELL"
            reasons.append("đã break xuống + close confirm")
        elif tg_state == "TRIGGERED" and break_dn:
            reveal = True
            direction = "SELL"
            reasons.append("trigger SELL đã kích hoạt")
        elif liq_state in ("YES", "PARTIAL") and break_dn:
            reveal = True
            direction = "SELL"
            reasons.append("sweep xong và đang follow-through xuống")

    # reveal BUY
    if htf_bias == "BUY":
        if break_up and cc_strength in ("WEAK", "STRONG"):
            reveal = True
            direction = "BUY"
            reasons.append("đã break lên + close confirm")
        elif tg_state == "TRIGGERED" and break_up:
            reveal = True
            direction = "BUY"
            reasons.append("trigger BUY đã kích hoạt")
        elif liq_state in ("YES", "PARTIAL") and break_up:
            reveal = True
            direction = "BUY"
            reasons.append("sweep xong và đang follow-through lên")

    # quality
    if reveal:
        q = 0
        if cc_strength == "STRONG":
            q += 2
        elif cc_strength == "WEAK":
            q += 1

        if liq_state in ("YES", "PARTIAL"):
            q += 1

        if vr is not None and vr >= 1.20:
            q += 1

        if q >= 3:
            quality = "HIGH"
        elif q >= 2:
            quality = "MEDIUM"
        else:
            quality = "LOW"

    return {
        "reveal": reveal,
        "direction": direction,
        "quality": quality,
        "reasons": reasons[:4],
    }

# =========================
# TRIGGER ENGINE V3
# =========================
def _trigger_engine_v3(
    signal_consistency_v1: dict | None,
    trigger_engine_v2: dict | None,
    reveal_engine_v1: dict | None,
    no_trade_zone_v3: dict | None,
    close_confirm_v4: dict | None,
    m15_struct: dict | None,
    range_pos: float | None,
    entry_zone_low: float | None = None,
    entry_zone_high: float | None = None,
    break_level: float | None = None,
    invalidation_level: float | None = None,
) -> dict:
    sce1 = signal_consistency_v1 or {}
    tg2 = trigger_engine_v2 or {}
    rv1 = reveal_engine_v1 or {}
    ntz = no_trade_zone_v3 or {}
    cc4 = close_confirm_v4 or {}
    m15s = m15_struct or {}

    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    final_side = str(sce1.get("final_side") or "NONE").upper()
    action_mode = str(sce1.get("action_mode") or "NO_TRADE").upper()
    ntz_active = bool(ntz.get("active"))
    confirm_ok = bool(tg2.get("confirm_ok"))
    location_ok = bool(tg2.get("location_ok"))
    follow_ok = bool(tg2.get("follow_ok"))
    reveal = bool(rv1.get("reveal"))
    reveal_dir = str(rv1.get("direction") or "NONE").upper()
    cc_strength = str(cc4.get("strength") or "NO").upper()

    break_up = bool(m15s.get("break_up"))
    break_dn = bool(m15s.get("break_dn"))

    state = "WAIT"
    quality = "LOW"
    entry_side = "NONE"
    reasons = []
    trigger_line = ""
    invalidation_line = ""
    close_confirm_line = ""

    if final_side == "SELL":
        entry_side = "SELL"
        trigger_line = "SELL khi sweep high thất bại hoặc break xuống có follow-through"
        invalidation_line = "Vô hiệu SELL nếu M15 break lên và giữ được"
        close_confirm_line = "Close confirm: quét vùng trên rồi đóng lại dưới, hoặc M15 đóng dưới break low và nến sau không reclaim"

        if ntz_active:
            state = "WAIT"
            reasons.append("đang trong no-trade zone")
        elif reveal and reveal_dir == "SELL":
            state = "TRIGGERED"
            quality = rv1.get("quality", "MEDIUM")
            reasons.append("market đã lộ mặt theo hướng SELL")
        elif location_ok and (confirm_ok or cc_strength in ("WEAK", "STRONG")) and (break_dn or follow_ok):
            state = "READY"
            quality = "MEDIUM"
            reasons.append("SELL đang có location + confirm đủ dùng")
        else:
            state = "WAIT"
            reasons.append("SELL chưa có timing đủ rõ")

    elif final_side == "BUY":
        entry_side = "BUY"
        trigger_line = "BUY khi giữ đáy tốt hoặc break lên có follow-through"
        invalidation_line = "Vô hiệu BUY nếu M15 break xuống và giữ dưới"
        close_confirm_line = "Close confirm: quét vùng dưới rồi đóng lại trên, hoặc M15 đóng trên break high và nến sau không rơi lại"

        if ntz_active:
            state = "WAIT"
            reasons.append("đang trong no-trade zone")
        elif reveal and reveal_dir == "BUY":
            state = "TRIGGERED"
            quality = rv1.get("quality", "MEDIUM")
            reasons.append("market đã lộ mặt theo hướng BUY")
        elif location_ok and (confirm_ok or cc_strength in ("WEAK", "STRONG")) and (break_up or follow_ok):
            state = "READY"
            quality = "MEDIUM"
            reasons.append("BUY đang có location + confirm đủ dùng")
        else:
            state = "WAIT"
            reasons.append("BUY chưa có timing đủ rõ")
    else:
        reasons.append("chưa có hướng đủ rõ")

    # quality refinement
    if state == "TRIGGERED":
        if quality not in ("HIGH", "MEDIUM"):
            quality = "HIGH" if cc_strength == "STRONG" else "MEDIUM"
    elif state == "READY":
        if rp is not None and (rp <= 0.30 or rp >= 0.70):
            quality = "MEDIUM"
        else:
            quality = "LOW"

    return {
        "state": state,
        "quality": quality,
        "entry_side": entry_side,
        "trigger_line": trigger_line,
        "invalidation_line": invalidation_line,
        "close_confirm_line": close_confirm_line,
        "reason": reasons[:4],
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "break_level": break_level,
        "invalidation_level": invalidation_level,
    }

# =========================
# FINAL DECISION ENGINE
# =========================
def _final_decision_engine_v1(
    signal_consistency_v1: dict | None,
    reveal_engine_v1: dict | None,
    trigger_engine_v3: dict | None,
    master_engine_v1: dict | None,
    no_trade_zone_v3: dict | None,
) -> dict:
    sce1 = signal_consistency_v1 or {}
    rv1 = reveal_engine_v1 or {}
    tg3 = trigger_engine_v3 or {}
    me1 = master_engine_v1 or {}
    ntz = no_trade_zone_v3 or {}

    final_side = str(sce1.get("final_side") or "NONE").upper()
    action_mode = str(sce1.get("action_mode") or "NO_TRADE").upper()
    master_state = str(me1.get("state") or "WAIT").upper()
    trigger_state = str(tg3.get("state") or "WAIT").upper()
    reveal = bool(rv1.get("reveal"))
    reveal_dir = str(rv1.get("direction") or "NONE").upper()
    ntz_active = bool(ntz.get("active"))

    decision = "NO_TRADE"
    label = "STAND ASIDE"
    reasons = []

    if ntz_active or master_state == "NO_TRADE":
        decision = "NO_TRADE"
        label = "STAND ASIDE"
        reasons.append("edge chưa đủ rõ")
    else:
        if reveal and reveal_dir == final_side and final_side in ("BUY", "SELL"):
            decision = f"EXECUTE_{final_side}"
            label = f"MANUAL STRIKE {final_side}"
            reasons.append("market đã lộ mặt đúng hướng")
        elif trigger_state == "READY" and final_side in ("BUY", "SELL"):
            decision = f"WAIT_TRIGGER_{final_side}"
            label = f"WAIT TRIGGER {final_side}"
            reasons.append("ý tưởng đúng nhưng timing chưa tới")
        elif action_mode in ("WAIT_BUY", "WAIT_SELL"):
            decision = action_mode
            label = action_mode.replace("_", " ")
            reasons.append("đúng hướng nhưng chưa có tín hiệu vào")
        else:
            decision = "NO_TRADE"
            label = "STAND ASIDE"
            reasons.append("chưa có lợi thế rõ")

    return {
        "decision": decision,
        "label": label,
        "reasons": reasons[:3],
    }
# =========================
# LIQUIDITY REACTION ENGINE V1
# Sweep -> Confirm -> Entry
# =========================
def _liquidity_reaction_engine_v1(
    signal_consistency_v1: dict | None,
    liquidity_map: dict | None,
    liquidity_completion_v1: dict | None,
    close_confirm_v4: dict | None,
    m15_struct: dict | None,
    current_price: float | None,
    range_lo: float | None,
    range_hi: float | None,
) -> dict:
    sce1 = signal_consistency_v1 or {}
    liq_map = liquidity_map or {}
    liq1 = liquidity_completion_v1 or {}
    cc4 = close_confirm_v4 or {}
    m15s = m15_struct or {}

    final_side = str(sce1.get("final_side") or "NONE").upper()
    current_move = str(sce1.get("current_move") or "CHOP").upper()

    try:
        px = float(current_price) if current_price is not None else None
    except Exception:
        px = None

    try:
        lo = float(range_lo) if range_lo is not None else None
    except Exception:
        lo = None

    try:
        hi = float(range_hi) if range_hi is not None else None
    except Exception:
        hi = None

    cc_strength = str(cc4.get("strength") or "NO").upper()
    liq_state = str(liq1.get("state") or "NO").upper()

    break_up = bool(m15s.get("break_up"))
    break_dn = bool(m15s.get("break_dn"))
    hl_ok = bool(m15s.get("hl"))
    lh_ok = bool(m15s.get("lh"))

    top_liq = str(liq_map.get("top_liquidity") or liq_map.get("above_level") or "").upper()
    bot_liq = str(liq_map.get("bottom_liquidity") or liq_map.get("below_level") or "").upper()

    # vùng phản ứng gần đúng
    top_zone_low = liq_map.get("top_zone_low")
    top_zone_high = liq_map.get("top_zone_high")
    bot_zone_low = liq_map.get("bottom_zone_low")
    bot_zone_high = liq_map.get("bottom_zone_high")

    state = "WAIT"
    reaction_type = "NONE"
    entry_side = "NONE"
    reasons = []
    wait_lines = []

    # fallback zone nếu liquidity_map chưa có vùng
    if top_zone_low is None and hi is not None:
        top_zone_low = hi
    if top_zone_high is None and hi is not None:
        top_zone_high = hi

    if bot_zone_low is None and lo is not None:
        bot_zone_low = lo
    if bot_zone_high is None and lo is not None:
        bot_zone_high = lo

    # =========================
    # CASE 1: Sweep high -> fail giữ -> SELL
    # =========================
    sell_reaction_possible = (
        final_side == "SELL"
        or (final_side == "NONE" and current_move in ("CHOP", "PULLBACK"))
    )

    if sell_reaction_possible:
        if top_liq in ("HIGH", "MEDIUM") and top_zone_low is not None:
            wait_lines.append(f"Quét vùng trên {_fmt_num(top_zone_low)} – {_fmt_num(top_zone_high)} rồi fail giữ → xét SELL")
        elif hi is not None:
            wait_lines.append(f"Break lên {_fmt_num(hi)} rồi không giữ được → xét SELL")

        if final_side == "SELL":
            entry_side = "SELL"

            # đã có dấu hiệu reject + cấu trúc yếu đi
            if (lh_ok or break_dn) and cc_strength in ("WEAK", "STRONG"):
                state = "READY"
                reaction_type = "SWEEP_FAIL"
                reasons.append("có phản ứng từ chối sau vùng cao")
                reasons.append("đang có xác nhận yếu theo hướng SELL")

            if liq_state in ("YES", "PARTIAL") and break_dn and cc_strength in ("WEAK", "STRONG"):
                state = "TRIGGERED"
                reaction_type = "SWEEP_FAIL"
                reasons.append("đã sweep xong và fail giữ")
                reasons.append("M15 đang break xuống có confirm")

    # =========================
    # CASE 2: Sweep low -> giữ được -> BUY
    # =========================
    buy_reaction_possible = (
        final_side == "BUY"
        or (final_side == "NONE" and current_move in ("CHOP", "PULLBACK"))
    )

    if buy_reaction_possible:
        if bot_liq in ("HIGH", "MEDIUM") and bot_zone_low is not None:
            wait_lines.append(f"Quét vùng dưới {_fmt_num(bot_zone_low)} – {_fmt_num(bot_zone_high)} rồi giữ được → xét BUY")
        elif lo is not None:
            wait_lines.append(f"Break xuống {_fmt_num(lo)} rồi reclaim lại → xét BUY")

        if final_side == "BUY":
            entry_side = "BUY"

            if (hl_ok or break_up) and cc_strength in ("WEAK", "STRONG"):
                state = "READY"
                reaction_type = "SWEEP_HOLD"
                reasons.append("có phản ứng giữ giá sau vùng thấp")
                reasons.append("đang có xác nhận yếu theo hướng BUY")

            if liq_state in ("YES", "PARTIAL") and break_up and cc_strength in ("WEAK", "STRONG"):
                state = "TRIGGERED"
                reaction_type = "SWEEP_HOLD"
                reasons.append("đã quét xong và giữ được")
                reasons.append("M15 đang break lên có confirm")

    # =========================
    # CASE 3: Break -> Hold
    # =========================
    if final_side == "SELL" and break_dn and cc_strength in ("WEAK", "STRONG"):
        state = "TRIGGERED"
        reaction_type = "BREAK_HOLD"
        entry_side = "SELL"
        reasons.append("đã break xuống và giữ dưới")

    if final_side == "BUY" and break_up and cc_strength in ("WEAK", "STRONG"):
        state = "TRIGGERED"
        reaction_type = "BREAK_HOLD"
        entry_side = "BUY"
        reasons.append("đã break lên và giữ trên")

    return {
        "state": state,                 # WAIT / READY / TRIGGERED
        "reaction_type": reaction_type, # SWEEP_FAIL / SWEEP_HOLD / BREAK_HOLD / NONE
        "entry_side": entry_side,       # BUY / SELL / NONE
        "reasons": reasons[:4],
        "wait_lines": wait_lines[:4],
        "top_zone_low": top_zone_low,
        "top_zone_high": top_zone_high,
        "bot_zone_low": bot_zone_low,
        "bot_zone_high": bot_zone_high,
        "range_lo": lo,
        "range_hi": hi,
    }


def _fmt_num(v):
    try:
        return f"{float(v):.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(v)
        
def _f(v):
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)
def _post_break_continuity_engine_v1(
    current_price,
    bos_level,
    range_low,
    range_high,
    struct,
    close_confirm_v4,
    liquidity_map_v1,
    trigger_engine_v3,
    absorption_v1=None,
):
    """
    Continuity engine đơn giản:
    - Xác định: break chưa?
    - Sau break: giữ được hay fail?
    - Absorption dùng để điều chỉnh hành vi

    Output:
    {
        state,
        side,
        reference,
        action,
        reason[]
    }
    """

    out = {
        "state": "NONE",
        "side": "NONE",
        "reference": None,
        "action": "WAIT",
        "reason": []
    }

    # ========= normalize =========
    try:
        cp = float(current_price)
    except:
        return out

    try:
        ref = float(bos_level) if bos_level else None
    except:
        ref = None

    struct = struct if isinstance(struct, dict) else {}
    tg3 = trigger_engine_v3 if isinstance(trigger_engine_v3, dict) else {}
    cc4 = close_confirm_v4 if isinstance(close_confirm_v4, dict) else {}

    # ===== absorption =====
    absorb = absorption_v1 if isinstance(absorption_v1, dict) else {}
    abs_active = bool(absorb.get("active"))
    abs_side = str(absorb.get("side") or "NONE").upper()
    abs_strength = str(absorb.get("strength") or "LOW").upper()
    abs_location = str(absorb.get("location") or "UNKNOWN").upper()

    # ===== side =====
    side = str(
        tg3.get("entry_side")
        or tg3.get("side")
        or ""
    ).upper()

    if side not in ("BUY", "SELL"):
        m15 = str(struct.get("M15") or "").upper()
        if "HH" in m15 or "HL" in m15:
            side = "BUY"
        elif "LH" in m15 or "LL" in m15:
            side = "SELL"
        else:
            side = "NONE"

    # ===== fallback reference =====
    if ref is None:
        if side == "BUY":
            ref = range_high
        elif side == "SELL":
            ref = range_low

    if ref is None:
        out["reason"] = ["không có reference level"]
        return out

    out["reference"] = ref
    out["side"] = side

    # ===== tolerance =====
    tol = abs(ref) * 0.0015

    above = cp > ref + tol
    below = cp < ref - tol
    at_level = not above and not below

    # ============================
    # ===== LOGIC CHÍNH =========
    # ============================

    # ===== BUY SIDE =====
    if side == "BUY":

        # chưa break
        if not above:
            out["state"] = "WAIT_BREAK"
            out["action"] = "WAIT_BREAK_BUY"
            out["reason"] = ["chưa break kháng cự"]

            # absorption chống BUY
            if abs_active and abs_side == "SELL" and abs_location == "HIGH":
                out["action"] = "AVOID_BUY_CHASE"
                out["reason"].append("có SELL absorption ở đỉnh")

            return out

        # đang test lại
        if at_level:
            out["state"] = "AT_LEVEL"
            out["action"] = "WAIT_RETEST"
            out["reason"] = ["đang test lại vùng break BUY"]
            return out

        # break giữ được
        if above:
            out["state"] = "BREAK_HOLD"
            out["action"] = "BUY_PULLBACK"
            out["reason"] = ["đã break và giữ trên"]

            # absorption ngược chiều
            if abs_active and abs_side == "SELL":
                if abs_strength in ("MEDIUM", "HIGH"):
                    out["action"] = "WAIT_RETEST"
                    out["reason"].append("gặp SELL absorption trên đỉnh")

            return out

        # fail
        if below:
            out["state"] = "BREAK_FAIL"
            out["action"] = "WAIT"
            out["reason"] = ["break BUY thất bại"]
            return out

    # ===== SELL SIDE =====
    if side == "SELL":

        # chưa break
        if not below:
            out["state"] = "WAIT_BREAK"
            out["action"] = "WAIT_BREAK_SELL"
            out["reason"] = ["chưa break hỗ trợ"]

            # absorption chống SELL
            if abs_active and abs_side == "BUY" and abs_location == "LOW":
                out["action"] = "AVOID_SELL_CHASE"
                out["reason"].append("có BUY absorption ở đáy")

            return out

        # test lại
        if at_level:
            out["state"] = "AT_LEVEL"
            out["action"] = "WAIT_RETEST"
            out["reason"] = ["đang test lại vùng break SELL"]
            return out

        # giữ dưới
        if below:
            out["state"] = "BREAK_HOLD"
            out["action"] = "SELL_PULLBACK"
            out["reason"] = ["đã break và giữ dưới"]

            # absorption ngược
            if abs_active and abs_side == "BUY":
                if abs_strength in ("MEDIUM", "HIGH"):
                    out["action"] = "WAIT_RETEST"
                    out["reason"].append("gặp BUY absorption dưới đáy")

            return out

        # fail
        if above:
            out["state"] = "BREAK_FAIL"
            out["action"] = "WAIT"
            out["reason"] = ["break SELL thất bại"]
            return out

    # ===== fallback =====
    out["state"] = "NONE"
    out["action"] = "WAIT"
    out["reason"] = ["không xác định được side"]

    return out


def _inject_wait_levels_v1(base: dict, bias_side: str, m15c, m30c, h1c, atr15: float):
    """
    Dùng cho các nhánh WAIT / early return:
    nếu chưa có entry/sl/tp thì nhét fallback levels để formatter đọc được.
    Không đổi recommendation, chỉ bổ sung mức giá tham khảo.
    """
    try:
        # nếu đã có rồi thì thôi
        if base.get("entry") is not None or base.get("sl") is not None or base.get("tp1") is not None:
            return base

        # 1) ưu tiên method M30
        atr30 = _atr(m30c, 14) if m30c else None
        method_pack = _pick_trade_method_m30(m30c, atr30 or atr15)

        lines = method_pack.get("lines") or []
        text = "\n".join(str(x) for x in lines)

        entry = sl = tp1 = tp2 = None

        import re

        # Entry kiểu: Entry gợi ý: chờ RETEST về ~1234
        m_entry = re.search(r"Entry gợi ý:.*?~([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if m_entry:
            try:
                entry = float(m_entry.group(1))
            except Exception:
                pass

        # SL/TP1/TP2 kiểu IPC/Breakout
        m_trip = re.search(
            r"SL gợi ý:\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*TP1:\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*TP2:\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE
        )
        if m_trip:
            try:
                sl = float(m_trip.group(1))
                tp1 = float(m_trip.group(2))
                tp2 = float(m_trip.group(3))
            except Exception:
                pass

        # 2) fallback bằng pullback/fib nếu M30 method không ra
        meta = base.setdefault("meta", {})
        fib1 = meta.get("fib_confluence_v1") or {}
        pb1 = meta.get("pullback_engine_v1") or {}

        if entry is None:
            zlo = fib1.get("zone_low")
            zhi = fib1.get("zone_high")
            if zlo is not None and zhi is not None:
                try:
                    entry = (float(zlo) + float(zhi)) / 2.0
                except Exception:
                    pass

        if entry is None:
            a_lo = pb1.get("anchor_low")
            a_hi = pb1.get("anchor_high")
            if a_lo is not None and a_hi is not None:
                try:
                    lo = float(a_lo)
                    hi = float(a_hi)
                    rng = max(1e-9, hi - lo)

                    if str(bias_side).upper() == "BUY":
                        entry = hi - 0.50 * rng
                        if sl is None:
                            sl = lo
                        if tp1 is None:
                            tp1 = hi
                    elif str(bias_side).upper() == "SELL":
                        entry = lo + 0.50 * rng
                        if sl is None:
                            sl = hi
                        if tp1 is None:
                            tp1 = lo
                except Exception:
                    pass

        # 3) nếu có entry mà thiếu sl/tp thì ATR fallback
        if entry is not None and (sl is None or tp1 is None):
            try:
                fallback_r = max(0.6, float(atr15 or 0.0) * 1.0)
                if str(bias_side).upper() == "SELL":
                    if sl is None:
                        sl = float(entry) + fallback_r
                    if tp1 is None:
                        tp1 = float(entry) - 1.0 * fallback_r
                    if tp2 is None:
                        tp2 = float(entry) - 1.6 * fallback_r
                else:
                    if sl is None:
                        sl = float(entry) - fallback_r
                    if tp1 is None:
                        tp1 = float(entry) + 1.0 * fallback_r
                    if tp2 is None:
                        tp2 = float(entry) + 1.6 * fallback_r
            except Exception:
                pass

        # inject lại vào base nếu có
        if entry is not None:
            base["entry"] = float(entry)
        if sl is not None:
            base["sl"] = float(sl)
        if tp1 is not None:
            base["tp1"] = float(tp1)
        if tp2 is not None:
            base["tp2"] = float(tp2)

        return base

    except Exception:
        return base

def _post_break_continuity_engine_v1(
    current_price,
    bos_level,
    range_low,
    range_high,
    struct,
    close_confirm_v4,
    liquidity_map_v1,
    trigger_engine_v3,
    absorption_v1=None,
):
    """
    POST-BREAK CONTINUITY V1
    ------------------------
    Mục tiêu:
    - xác định trạng thái quanh mốc break
    - tránh kiểu "break rồi lại chờ break tiếp"
    - absorption chỉ dùng để chỉnh action, không tự đảo state chính

    Output keys cố định:
    {
        "state": "WAIT_BREAK" | "AT_LEVEL" | "BREAK_HOLD" | "BREAK_FAIL" | "NONE",
        "side": "BUY" | "SELL" | "NONE",
        "reference": float | None,
        "action": str,
        "reason": list[str],
    }
    """

    out = {
        "state": "NONE",
        "side": "NONE",
        "reference": None,
        "action": "WAIT",
        "reason": [],
    }

    # =========================
    # normalize input
    # =========================
    try:
        cp = float(current_price) if current_price is not None else None
    except Exception:
        cp = None

    try:
        bos = float(bos_level) if bos_level is not None else None
    except Exception:
        bos = None

    try:
        rlo = float(range_low) if range_low is not None else None
    except Exception:
        rlo = None

    try:
        rhi = float(range_high) if range_high is not None else None
    except Exception:
        rhi = None

    struct = struct if isinstance(struct, dict) else {}
    cc4 = close_confirm_v4 if isinstance(close_confirm_v4, dict) else {}
    liq = liquidity_map_v1 if isinstance(liquidity_map_v1, dict) else {}
    tg3 = trigger_engine_v3 if isinstance(trigger_engine_v3, dict) else {}
    abs1 = absorption_v1 if isinstance(absorption_v1, dict) else {}

    if cp is None:
        out["reason"] = ["missing current_price"]
        return out

    # =========================
    # side inference
    # =========================
    tg_side = str(
        tg3.get("entry_side")
        or tg3.get("side")
        or "NONE"
    ).upper()

    m15_tag = str(struct.get("M15") or "").upper()
    h1_tag = str(struct.get("H1") or "").upper()

    inferred_side = "NONE"

    if tg_side in ("BUY", "SELL"):
        inferred_side = tg_side
    elif ("HH" in m15_tag or "HL" in m15_tag or "HH" in h1_tag or "HL" in h1_tag):
        inferred_side = "BUY"
    elif ("LL" in m15_tag or "LH" in m15_tag or "LL" in h1_tag or "LH" in h1_tag):
        inferred_side = "SELL"
    else:
        sweep_bias = str(liq.get("sweep_bias") or "").upper()
        if "DOWN → UP" in sweep_bias:
            inferred_side = "BUY"
        elif "UP → DOWN" in sweep_bias:
            inferred_side = "SELL"

    out["side"] = inferred_side

    # =========================
    # reference level
    # =========================
    ref = bos

    if ref is None:
        if inferred_side == "BUY":
            ref = rhi
        elif inferred_side == "SELL":
            ref = rlo
        else:
            candidates = [x for x in [rlo, rhi] if x is not None]
            if candidates:
                ref = min(candidates, key=lambda x: abs(cp - x))

    out["reference"] = ref

    if ref is None:
        out["reason"] = ["missing reference level"]
        return out

    # =========================
    # tolerance
    # =========================
    span = None
    if rlo is not None and rhi is not None and rhi > rlo:
        span = rhi - rlo

    # nếu có range thì dùng theo range, không thì fallback theo giá tuyệt đối
    tol = max(
        0.000001,
        (span * 0.015) if span is not None else abs(ref) * 0.0015
    )

    above = cp > ref + tol
    below = cp < ref - tol
    at_level = not above and not below

    # =========================
    # confirm / absorption
    # =========================
    cc_strength = str(cc4.get("strength") or "NO").upper()
    cc_hold = str(cc4.get("hold") or "NO").upper()

    abs_active = bool(abs1.get("active"))
    abs_side = str(abs1.get("side") or "NONE").upper()
    abs_strength = str(abs1.get("strength") or "LOW").upper()
    abs_location = str(abs1.get("location") or "UNKNOWN").upper()

    # =========================
    # no clear side
    # =========================
    if inferred_side not in ("BUY", "SELL"):
        out["state"] = "NONE"
        out["action"] = "WAIT"
        out["reason"] = ["không xác định được side"]
        return out

    # =========================
    # BUY logic
    # =========================
    if inferred_side == "BUY":

        # 1) break hold: đã nằm trên mốc
        if above:
            out["state"] = "BREAK_HOLD"
            out["action"] = "BUY_PULLBACK"
            out["reason"] = ["đã break lên và đang giữ trên mốc"]

            # absorption ngược chiều -> hạ action
            if abs_active and abs_side == "SELL" and abs_strength in ("MEDIUM", "HIGH"):
                out["action"] = "WAIT_RETEST"
                out["reason"].append("gặp SELL absorption phía trên")
            elif cc_strength in ("NO", "N/A"):
                out["action"] = "WAIT_RETEST"
                out["reason"].append("break có nhưng close confirm chưa rõ")

            return out

        # 2) test level
        if at_level:
            out["state"] = "AT_LEVEL"
            out["action"] = "WAIT_REACTION"
            out["reason"] = ["đang test lại mốc BUY"]

            if abs_active and abs_side == "SELL":
                out["reason"].append("đang có hấp thụ ngược chiều")
            return out

        # 3) fail break: đã mất lại mốc sau khi đáng lẽ phải giữ
        if below and cc_strength in ("WEAK", "STRONG") and cc_hold == "NO":
            out["state"] = "BREAK_FAIL"
            out["action"] = "WAIT"
            out["reason"] = ["break BUY thất bại, mất lại mốc"]
            return out

        # 4) chưa break
        out["state"] = "WAIT_BREAK"
        out["action"] = "WAIT_BREAK_BUY"
        out["reason"] = ["chưa break được mốc BUY"]

        if abs_active and abs_side == "SELL" and abs_location == "HIGH":
            out["action"] = "AVOID_BUY_CHASE"
            out["reason"].append("có SELL absorption ở đỉnh")
        return out

    # =========================
    # SELL logic
    # =========================
    if inferred_side == "SELL":

        # 1) break hold: đã nằm dưới mốc
        if below:
            out["state"] = "BREAK_HOLD"
            out["action"] = "SELL_PULLBACK"
            out["reason"] = ["đã break xuống và đang giữ dưới mốc"]

            # absorption ngược chiều -> hạ action
            if abs_active and abs_side == "BUY" and abs_strength in ("MEDIUM", "HIGH"):
                out["action"] = "WAIT_RETEST"
                out["reason"].append("gặp BUY absorption phía dưới")
            elif cc_strength in ("NO", "N/A"):
                out["action"] = "WAIT_RETEST"
                out["reason"].append("break có nhưng close confirm chưa rõ")

            return out

        # 2) test level
        if at_level:
            out["state"] = "AT_LEVEL"
            out["action"] = "WAIT_REACTION"
            out["reason"] = ["đang test lại mốc SELL"]

            if abs_active and abs_side == "BUY":
                out["reason"].append("đang có hấp thụ ngược chiều")
            return out

        # 3) fail break
        if above and cc_strength in ("WEAK", "STRONG") and cc_hold == "NO":
            out["state"] = "BREAK_FAIL"
            out["action"] = "WAIT"
            out["reason"] = ["break SELL thất bại, mất lại mốc"]
            return out

        # 4) chưa break
        out["state"] = "WAIT_BREAK"
        out["action"] = "WAIT_BREAK_SELL"
        out["reason"] = ["chưa break được mốc SELL"]

        if abs_active and abs_side == "BUY" and abs_location == "LOW":
            out["action"] = "AVOID_SELL_CHASE"
            out["reason"].append("có BUY absorption ở đáy")
        return out

    return out
def _zone_action_engine_v1(
    current_price,
    side,
    support_zone=None,
    resistance_zone=None,
    break_level=None,
    atr15=None,
):
    """
    ZONE + ACTION ENGINE V1

    Mục tiêu:
    - BUY thì dùng SUPPORT làm vùng chính
    - SELL thì dùng RESISTANCE làm vùng chính
    - biết giá đang ở đâu so với vùng:
        WAIT   = chưa tới vùng
        WATCH  = đang ở vùng
        TRIGGER = đã phản ứng rời vùng theo hướng có lợi
        CANCEL = xuyên vùng sai hướng
    """

    out = {
        "ok": False,
        "side": str(side or "NONE").upper(),
        "zone_type": "NONE",
        "zone_low": None,
        "zone_high": None,
        "price_state": "NONE",
        "action_state": "WAIT",
        "message": "Chưa có vùng hành động rõ",
        "trigger": "",
        "invalid": "",
        "break_level": break_level,
        "lines": [],
    }

    try:
        cp = float(current_price)
    except Exception:
        return out

    side_u = str(side or "NONE").upper()

    # ===== chọn đúng zone theo side =====
    if side_u == "BUY":
        z = support_zone
        out["zone_type"] = "SUPPORT"
    elif side_u == "SELL":
        z = resistance_zone
        out["zone_type"] = "RESISTANCE"
    else:
        return out

    if not isinstance(z, (list, tuple)) or len(z) != 2:
        return out

    try:
        lo = float(min(z[0], z[1]))
        hi = float(max(z[0], z[1]))
    except Exception:
        return out

    out["ok"] = True
    out["zone_low"] = lo
    out["zone_high"] = hi

    a = float(atr15 or 0.0)
    pad = max(a * 0.10, (hi - lo) * 0.10, 1e-9)

    # ===== BUY logic =====
    if side_u == "BUY":
        if cp > hi + pad:
            out["price_state"] = "ABOVE_ZONE"
            out["action_state"] = "WAIT"
            out["message"] = f"Giá còn ở trên vùng BUY {lo:.3f} – {hi:.3f} → chờ hồi về support"
            out["trigger"] = "Chỉ BUY khi giá về vùng, sweep low/giữ đáy/reclaim rõ"
            out["invalid"] = f"M15 đóng dưới {lo:.3f} và giữ dưới → huỷ BUY"

        elif lo - pad <= cp <= hi + pad:
            out["price_state"] = "IN_ZONE"
            out["action_state"] = "WATCH"
            out["message"] = f"Giá đang ở vùng BUY {lo:.3f} – {hi:.3f} → không còn là chờ vùng, chuyển sang WATCH"
            out["trigger"] = "BUY khi sweep low rồi đóng lại trên vùng, hoặc có nến rút chân/giữ đáy rõ"
            out["invalid"] = f"M15 đóng dưới {lo:.3f} và nến sau không reclaim → huỷ BUY"

        else:  # cp < lo - pad
            out["price_state"] = "BELOW_ZONE"
            out["action_state"] = "CANCEL"
            out["message"] = f"Giá đã thủng vùng BUY {lo:.3f} – {hi:.3f} → không BUY dip nữa"
            out["trigger"] = "Chờ reclaim lại vùng hoặc setup mới"
            out["invalid"] = "BUY dip bị vô hiệu ngắn hạn"

    # ===== SELL logic =====
    elif side_u == "SELL":
        if cp < lo - pad:
            out["price_state"] = "BELOW_ZONE"
            out["action_state"] = "WAIT"
            out["message"] = f"Giá còn ở dưới vùng SELL {lo:.3f} – {hi:.3f} → chờ hồi lên resistance"
            out["trigger"] = "Chỉ SELL khi giá hồi lên vùng, fail break/rejection/đóng yếu rõ"
            out["invalid"] = f"M15 đóng trên {hi:.3f} và giữ trên → huỷ SELL"

        elif lo - pad <= cp <= hi + pad:
            out["price_state"] = "IN_ZONE"
            out["action_state"] = "WATCH"
            out["message"] = f"Giá đang ở vùng SELL {lo:.3f} – {hi:.3f} → không còn là chờ vùng, chuyển sang WATCH"
            out["trigger"] = "SELL khi fail break, rejection, hoặc nến đỏ xác nhận rời vùng"
            out["invalid"] = f"M15 đóng trên {hi:.3f} và nến sau giữ trên → huỷ SELL"

        else:  # cp > hi + pad
            out["price_state"] = "ABOVE_ZONE"
            out["action_state"] = "CANCEL"
            out["message"] = f"Giá đã vượt vùng SELL {lo:.3f} – {hi:.3f} → không SELL vùng này nữa"
            out["trigger"] = "Chờ mất lại vùng hoặc setup mới"
            out["invalid"] = "SELL rally bị vô hiệu ngắn hạn"

    # ===== thêm breakout option =====
    if break_level is not None:
        try:
            bl = float(break_level)
            if side_u == "BUY":
                out["lines"] = [
                    out["message"],
                    out["trigger"],
                    f"Hoặc BUY breakout nếu M15 đóng trên {bl:.3f} và nến sau giữ được",
                ]
            elif side_u == "SELL":
                out["lines"] = [
                    out["message"],
                    out["trigger"],
                    f"Hoặc SELL breakdown nếu M15 đóng dưới {bl:.3f} và nến sau giữ được",
                ]
        except Exception:
            out["lines"] = [out["message"], out["trigger"]]
    else:
        out["lines"] = [out["message"], out["trigger"]]

    return out
def _elliott_phase_v1(
    h4_struct=None,
    h1_struct=None,
    m15_struct=None,
    pullback_info=None,
    ema_filter=None,
    flow_engine_v1=None,
    zone_action_v1=None,
    current_price=None,
    range_low=None,
    range_high=None,
):
    """
    Elliott Phase Context V1
    Không đếm sóng tuyệt đối.
    Chỉ phân loại xác suất phase:
    - WAVE_3 / WAVE_4 / WAVE_5
    - WAVE_A / WAVE_B / WAVE_C
    - BUY_DIP / SELL_RALLY context
    """

    out = {
        "ok": True,
        "main_tf": "H1/H4",
        "phase": "UNCLEAR",
        "direction": "NONE",
        "confidence": 40,
        "meaning": "Chưa đủ dữ liệu để xác định phase Elliott rõ",
        "action": "Đứng ngoài, chờ cấu trúc rõ hơn",
        "invalid": "Không có mốc vô hiệu rõ",
        "reason": [],
    }

    try:
        h4 = str(h4_struct or "").upper()
        h1 = str(h1_struct or "").upper()
        m15 = str(m15_struct or "").upper()

        pb = pullback_info if isinstance(pullback_info, dict) else {}
        ema = ema_filter if isinstance(ema_filter, dict) else {}
        flow = flow_engine_v1 if isinstance(flow_engine_v1, dict) else {}
        za = zone_action_v1 if isinstance(zone_action_v1, dict) else {}

        pb_pct = None
        for k in ("pct", "pullback_pct", "retrace_pct", "retracement_pct"):
            if pb.get(k) is not None:
                try:
                    pb_pct = float(pb.get(k))
                    break
                except Exception:
                    pass

        # nếu bot đang lưu dạng 0.81 thì đổi sang 81
        if pb_pct is not None and pb_pct <= 1:
            pb_pct *= 100.0

        reversal_risk = str(pb.get("reversal_risk") or "").upper()
        pb_label = str(pb.get("label") or pb.get("state") or pb.get("text") or "").upper()

        ema_trend = str(ema.get("trend") or "UNKNOWN").upper()
        ema_align = str(ema.get("alignment") or "NO").upper()

        flow_state = str(flow.get("state") or "NEUTRAL").upper()
        displacement = str(flow.get("displacement") or "NONE").upper()

        zone_state = str(za.get("price_state") or "NONE").upper()
        zone_side = str(za.get("side") or "NONE").upper()

        htf_up = ("HH" in h4 or "HL" in h4 or "HH" in h1 or "HL" in h1)
        htf_down = ("LL" in h4 or "LH" in h4 or "LL" in h1 or "LH" in h1)

        m15_up = ("HH" in m15 or "HL" in m15)
        m15_down = ("LL" in m15 or "LH" in m15)

        # vị trí trong range
        range_pos = None
        try:
            cp = float(current_price)
            lo = float(range_low)
            hi = float(range_high)
            if hi > lo:
                range_pos = (cp - lo) / (hi - lo) * 100.0
        except Exception:
            range_pos = None

        # =========================
        # CASE 1: HTF up, LTF đang giảm/hồi sâu
        # Có thể là sóng 4 / A / B tùy mức hồi
        # =========================
        if htf_up and m15_down:
            out["direction"] = "BULLISH_CONTEXT_CORRECTION"

            if pb_pct is not None and pb_pct >= 75:
                out["phase"] = "POSSIBLE WAVE A / DEEP CORRECTION"
                out["confidence"] = 62
                out["meaning"] = "Xu hướng lớn còn thiên tăng nhưng nhịp giảm ngắn hạn đã hồi quá sâu; có nguy cơ không còn là pullback sạch"
                out["action"] = "Không BUY vội; chỉ BUY nếu có sweep low, reclaim hoặc giữ support rõ"
                out["invalid"] = "Nếu M15/H1 đóng dưới đáy correction và giữ dưới → chuyển sang rủi ro sóng C giảm"
                out["reason"] = ["HTF còn tăng", "M15 đã LL-LH", "pullback sâu"]
                return out

            if pb_pct is not None and 35 <= pb_pct < 75:
                out["phase"] = "POSSIBLE WAVE 4 / BUY DIP"
                out["confidence"] = 58
                out["meaning"] = "Có thể đang là nhịp điều chỉnh trong xu hướng tăng, chưa xác nhận đảo chiều"
                out["action"] = "Chờ về support; BUY khi có giữ đáy / sweep low / close confirm"
                out["invalid"] = "Nếu thủng support và không reclaim → huỷ buy-dip"
                out["reason"] = ["HTF tăng", "pullback vừa phải", "chưa có xác nhận đảo chiều"]
                return out

            out["phase"] = "EARLY PULLBACK / WAIT BUY DIP"
            out["confidence"] = 52
            out["meaning"] = "Nhịp hồi còn sớm, chưa đủ sâu để có entry BUY đẹp"
            out["action"] = "Không BUY đuổi; chờ giá về vùng hỗ trợ hoặc break lên có follow-through"
            out["invalid"] = "Nếu phá đáy M15 và giữ dưới → chuyển sang correction sâu"
            out["reason"] = ["HTF tăng", "pullback còn sớm"]
            return out

        # =========================
        # CASE 2: HTF down, LTF hồi lên
        # Thường là sóng B / sell rally
        # =========================
        if htf_down and m15_up:
            out["direction"] = "BEARISH_CONTEXT_CORRECTION"

            if pb_pct is not None and pb_pct >= 50:
                out["phase"] = "POSSIBLE WAVE B / SELL RALLY"
                out["confidence"] = 62
                out["meaning"] = "Đây có thể là nhịp hồi trong correction, chưa phải đảo chiều thật"
                out["action"] = "Không BUY đuổi; chờ fail ở resistance hoặc break low để SELL"
                out["invalid"] = "Nếu break và giữ trên đỉnh sóng 5 / vùng kháng cự chính"
                out["reason"] = ["HTF giảm", "M15 hồi lên", "pullback đủ sâu để nghi sóng B"]
                return out

            out["phase"] = "EARLY WAVE B / WEAK BOUNCE"
            out["confidence"] = 55
            out["meaning"] = "Có thể là nhịp hồi yếu trong xu hướng giảm"
            out["action"] = "Chờ hồi lên resistance; SELL khi có rejection / fail break"
            out["invalid"] = "Nếu M15 break lên và giữ trên resistance → giảm hiệu lực SELL"
            out["reason"] = ["HTF giảm", "M15 hồi nhưng chưa mạnh"]
            return out

        # =========================
        # CASE 3: HTF down + M15 down
        # Có thể là sóng 3 hoặc C
        # =========================
        if htf_down and m15_down:
            out["direction"] = "BEARISH_IMPULSE"

            if displacement in ("DOWN", "STRONG_DOWN") or flow_state in ("IMBALANCED", "FLOW_READY"):
                out["phase"] = "POSSIBLE WAVE C / SELL CONTINUATION"
                out["confidence"] = 66
                out["meaning"] = "Xu hướng giảm đồng thuận, có thể đang vào nhịp C hoặc continuation giảm"
                out["action"] = "Ưu tiên SELL theo pullback hoặc breakdown; không BUY bắt đáy"
                out["invalid"] = "Nếu reclaim mạnh lại vùng breakdown và tạo HL"
                out["reason"] = ["HTF giảm", "M15 giảm", "có dấu hiệu impulse/flow"]
                return out

            out["phase"] = "DOWN IMPULSE / WAIT SELL RALLY"
            out["confidence"] = 60
            out["meaning"] = "Đang cùng chiều giảm nhưng chưa có displacement đủ rõ"
            out["action"] = "Không SELL đuổi; chờ hồi lên resistance để SELL"
            out["invalid"] = "Nếu phá lên tạo HH-HL trên M15"
            out["reason"] = ["HTF giảm", "M15 giảm", "chưa có flow rõ"]
            return out

        # =========================
        # CASE 4: HTF up + M15 up
        # Có thể là sóng 3 hoặc 5
        # =========================
        if htf_up and m15_up:
            out["direction"] = "BULLISH_IMPULSE"

            if range_pos is not None and range_pos >= 80:
                out["phase"] = "POSSIBLE WAVE 5 / LATE BUY"
                out["confidence"] = 60
                out["meaning"] = "Giá đang ở vùng cao của range, có thể là pha đẩy cuối; rủi ro FOMO tăng"
                out["action"] = "Không BUY đuổi; chờ pullback hoặc breakout giữ được"
                out["invalid"] = "Nếu sweep high rồi đóng ngược xuống → nghi bắt đầu ABC"
                out["reason"] = ["HTF tăng", "M15 tăng", "giá ở vùng cao"]
                return out

            if displacement in ("UP", "STRONG_UP"):
                out["phase"] = "POSSIBLE WAVE 3 / BUY CONTINUATION"
                out["confidence"] = 65
                out["meaning"] = "Có thể đang trong pha mở rộng theo xu hướng tăng"
                out["action"] = "Ưu tiên BUY theo pullback; không short ngược trend"
                out["invalid"] = "Nếu phá HL gần nhất và giữ dưới"
                out["reason"] = ["HTF tăng", "M15 tăng", "có impulse lên"]
                return out

            out["phase"] = "BULLISH CONTINUATION / WAIT PULLBACK"
            out["confidence"] = 58
            out["meaning"] = "Xu hướng tăng còn giữ nhưng chưa có điểm vào sạch"
            out["action"] = "Chờ pullback về support hoặc breakout có follow-through"
            out["invalid"] = "Nếu M15 mất HL và chuyển LL-LH"
            out["reason"] = ["HTF tăng", "M15 tăng"]
            return out

        # =========================
        # Transition / mixed
        # =========================
        out["phase"] = "TRANSITION / UNCLEAR WAVE"
        out["confidence"] = 45
        out["meaning"] = "Cấu trúc đang chuyển pha, chưa đủ rõ để gán Elliott wave"
        out["action"] = "Giảm size hoặc đứng ngoài; chờ break structure rõ"
        out["invalid"] = "Không áp dụng Elliott khi cấu trúc nhiễu"
        out["reason"] = ["HTF/M15 chưa đồng thuận"]
        return out

    except Exception as e:
        out["ok"] = False
        out["phase"] = "ERROR"
        out["meaning"] = f"Elliott engine lỗi: {e}"
        out["action"] = "Bỏ qua Elliott context"
        return out   

# ============================================================
# MARKET MODE V1 - Context reader for analysis-only bot
# Purpose:
# - Đọc market theo cụm nến, không đọc từng nến rời rạc
# - Phân biệt: NO SETUP vs TREND DAY nhưng timing chưa đẹp
# - Không auto trade, chỉ tạo meta + output guidance
# ============================================================

def _safe_last_closed(candles):
    try:
        if not candles:
            return []
        return list(candles[:-1] if len(candles) > 1 else candles)
    except Exception:
        return []

def _count_dir_bars(candles):
    bull = bear = 0
    for c in candles or []:
        try:
            o = float(_c_val(c, "open", 0.0) or 0.0)
            cl = float(_c_val(c, "close", 0.0) or 0.0)
            if cl > o:
                bull += 1
            elif cl < o:
                bear += 1
        except Exception:
            pass
    return bull, bear

def _avg_body_range(candles):
    bodies = []
    ranges = []
    for c in candles or []:
        try:
            o = float(_c_val(c, "open", 0.0) or 0.0)
            h = float(_c_val(c, "high", 0.0) or 0.0)
            l = float(_c_val(c, "low", 0.0) or 0.0)
            cl = float(_c_val(c, "close", 0.0) or 0.0)
            bodies.append(abs(cl - o))
            ranges.append(max(1e-9, h - l))
        except Exception:
            pass
    if not bodies or not ranges:
        return 0.0, 0.0
    return sum(bodies) / len(bodies), sum(ranges) / len(ranges)

def _detect_market_mode_v1(
    symbol: str,
    m15c,
    h1c,
    h4c,
    ema_pack: dict | None = None,
    pullback_engine: dict | None = None,
    range_pos=None,
    market_state_v2: str | None = None,
    playbook_v2: dict | None = None,
    post_break_continuity: dict | None = None,
) -> dict:
    """
    Analysis-only market context.
    Return:
    {
      mode, side, action_mode, confidence, summary,
      read_window, warnings, playbook_lines, reasons
    }
    """
    out = {
        "mode": "UNKNOWN",
        "side": "NONE",
        "action_mode": "WAIT_CONTEXT",
        "confidence": 0,
        "summary": "Chưa đủ dữ liệu để đọc market mode.",
        "read_window": {
            "htf": "H1/H4 50-100 nến",
            "mtf": "M15 30-50 nến",
            "momentum": "M15 10-15 nến gần nhất",
        },
        "warnings": [],
        "playbook_lines": [],
        "reasons": [],
        "range_pos": None,
    }

    try:
        ema_pack = ema_pack or {}
        pullback_engine = pullback_engine or {}
        playbook_v2 = playbook_v2 or {}
        post_break_continuity = post_break_continuity or {}

        m15_closed = _safe_last_closed(m15c)
        h1_closed = _safe_last_closed(h1c)
        h4_closed = _safe_last_closed(h4c)

        if len(m15_closed) < 30 or len(h1_closed) < 30:
            return out

        # ===== 1) HTF context =====
        h1_trend = _trend_label(h1_closed)
        h4_trend = _trend_label(h4_closed) if h4_closed else "sideways"

        htf_bear = h1_trend == "bearish" and h4_trend in ("bearish", "sideways")
        htf_bull = h1_trend == "bullish" and h4_trend in ("bullish", "sideways")

        # ===== 2) M15 structure / momentum =====
        recent_40 = m15_closed[-40:] if len(m15_closed) >= 40 else m15_closed
        recent_15 = m15_closed[-15:] if len(m15_closed) >= 15 else m15_closed
        recent_8 = m15_closed[-8:] if len(m15_closed) >= 8 else m15_closed

        bull15, bear15 = _count_dir_bars(recent_15)

        first_close = float(_c_val(recent_15[0], "close", 0.0) or 0.0)
        last_close = float(_c_val(recent_15[-1], "close", 0.0) or 0.0)
        move_15 = last_close - first_close

        atr15 = _atr(m15_closed, 14) or 0.0
        move_atr = abs(move_15) / max(float(atr15 or 0.0), 1e-9)

        avg_body_8, avg_range_8 = _avg_body_range(recent_8)
        avg_body_40, avg_range_40 = _avg_body_range(recent_40)

        body_expand = avg_body_8 > 1.15 * max(avg_body_40, 1e-9)
        range_expand = avg_range_8 > 1.10 * max(avg_range_40, 1e-9)

        ema_trend = str(ema_pack.get("trend") or "").upper()
        ema_zone = str(ema_pack.get("zone") or "").upper()
        ema_bear = ema_trend == "BEARISH" and ("DƯỚI EMA89" in ema_zone or "DƯỚI" in ema_zone)
        ema_bull = ema_trend == "BULLISH" and ("TRÊN" in ema_zone)

        # ===== 3) Pullback depth =====
        pb_pct = None
        try:
            pb_pct = float(pullback_engine.get("pullback_pct") or 0.0)
        except Exception:
            pb_pct = None

        shallow_pullback = pb_pct is not None and pb_pct <= 0.25

        # ===== 4) Range position =====
        rp = None
        try:
            if range_pos is not None:
                rp = float(range_pos)
            elif playbook_v2.get("range_pos") is not None:
                rp = float(playbook_v2.get("range_pos"))
        except Exception:
            rp = None
        out["range_pos"] = rp

        near_low = rp is not None and rp <= 0.20
        near_high = rp is not None and rp >= 0.80
        mid_range = rp is not None and 0.30 <= rp <= 0.70

        # ===== 5) Post-break continuity =====
        pbc_state = str(post_break_continuity.get("state") or "").upper()
        pbc_side = str(post_break_continuity.get("side") or "").upper()
        break_hold_sell = pbc_side == "SELL" and "HOLD" in pbc_state
        break_hold_buy = pbc_side == "BUY" and "HOLD" in pbc_state

        # ===== 6) Scoring =====
        down_score = 0
        up_score = 0
        reasons = []

        if htf_bear:
            down_score += 2
            reasons.append("H1/H4 nghiêng giảm")
        if htf_bull:
            up_score += 2
            reasons.append("H1/H4 nghiêng tăng")

        if ema_bear:
            down_score += 2
            reasons.append("EMA bearish + giá dưới EMA")
        if ema_bull:
            up_score += 2
            reasons.append("EMA bullish + giá trên EMA")

        if bear15 >= 9 and move_15 < 0:
            down_score += 2
            reasons.append("M15 gần đây nhiều nến giảm")
        if bull15 >= 9 and move_15 > 0:
            up_score += 2
            reasons.append("M15 gần đây nhiều nến tăng")

        if move_15 < 0 and move_atr >= 1.5:
            down_score += 2
            reasons.append(f"momentum giảm mạnh ~{move_atr:.1f} ATR")
        if move_15 > 0 and move_atr >= 1.5:
            up_score += 2
            reasons.append(f"momentum tăng mạnh ~{move_atr:.1f} ATR")

        if body_expand or range_expand:
            if move_15 < 0:
                down_score += 1
                reasons.append("biên độ/body đang mở rộng theo hướng giảm")
            elif move_15 > 0:
                up_score += 1
                reasons.append("biên độ/body đang mở rộng theo hướng tăng")

        if break_hold_sell:
            down_score += 2
            reasons.append("post-break đang giữ dưới mốc SELL")
        if break_hold_buy:
            up_score += 2
            reasons.append("post-break đang giữ trên mốc BUY")

        # ===== 7) Classification =====
        if down_score >= 6 and down_score >= up_score + 3:
            out["mode"] = "TREND_DAY_DOWN"
            out["side"] = "SELL"
            out["confidence"] = min(95, 55 + down_score * 5)
            out["summary"] = "Giá đang đi một chiều trong xu hướng giảm; pullback nông là đặc điểm của trend mạnh, không phải lỗi."
            out["action_mode"] = "FOLLOW_TREND_CONDITIONAL"
            out["playbook_lines"] = [
                "Ưu tiên SELL continuation, không phải bắt đáy BUY.",
                "Không SELL ngay đáy cây dump lớn.",
                "Chờ 1 trong 2: hồi nhỏ lên resistance gần rồi fail, hoặc break low + giữ dưới + nến sau không reclaim.",
            ]
            if shallow_pullback:
                out["warnings"].append("Pullback nông → nếu chỉ chờ hồi sâu có thể bỏ lỡ toàn bộ move.")
            if near_low:
                out["warnings"].append("Giá đang sát đáy range → không SELL đuổi; chỉ canh continuation sau nhịp nghỉ/break giữ dưới.")
            if mid_range:
                out["warnings"].append("Giá giữa biên độ → dễ nhiễu, cần trigger rõ.")
            out["reasons"] = reasons[:6]
            return out

        if up_score >= 6 and up_score >= down_score + 3:
            out["mode"] = "TREND_DAY_UP"
            out["side"] = "BUY"
            out["confidence"] = min(95, 55 + up_score * 5)
            out["summary"] = "Giá đang đi một chiều trong xu hướng tăng; pullback nông là đặc điểm của trend mạnh, không phải lỗi."
            out["action_mode"] = "FOLLOW_TREND_CONDITIONAL"
            out["playbook_lines"] = [
                "Ưu tiên BUY continuation, không phải bắt đỉnh SELL.",
                "Không BUY ngay đỉnh cây pump lớn.",
                "Chờ 1 trong 2: hồi nhỏ về support gần rồi giữ, hoặc break high + giữ trên + nến sau không reclaim.",
            ]
            if shallow_pullback:
                out["warnings"].append("Pullback nông → nếu chỉ chờ hồi sâu có thể bỏ lỡ toàn bộ move.")
            if near_high:
                out["warnings"].append("Giá đang sát đỉnh range → không BUY đuổi; chỉ canh continuation sau nhịp nghỉ/break giữ trên.")
            if mid_range:
                out["warnings"].append("Giá giữa biên độ → dễ nhiễu, cần trigger rõ.")
            out["reasons"] = reasons[:6]
            return out

        # Range / chop / reversal risk
        ms = str(market_state_v2 or "").upper()
        if ms in ("CHOP", "TRANSITION") or mid_range:
            out["mode"] = "RANGE_OR_CHOP"
            out["side"] = "NONE"
            out["confidence"] = 55
            out["action_mode"] = "WAIT_EDGE"
            out["summary"] = "Market đang nhiễu/giữa biên độ; ưu tiên chờ về biên hoặc break có giữ."
            out["playbook_lines"] = [
                "Không đánh giữa range.",
                "Chờ chạm hỗ trợ/kháng cự có phản ứng rõ.",
                "Hoặc chờ break + retest rồi mới đánh theo hướng break.",
            ]
            out["reasons"] = reasons[:6] or ["market chưa lệch rõ"]
            return out

        out["mode"] = "NORMAL_WAIT"
        out["side"] = "NONE"
        out["confidence"] = 45
        out["action_mode"] = "WAIT_CONTEXT"
        out["summary"] = "Market chưa đủ điều kiện để xếp trend day/range day rõ."
        out["reasons"] = reasons[:6] or ["thiếu đồng thuận đa khung"]
        return out

    except Exception as e:
        out["mode"] = "ERROR"
        out["summary"] = f"Market mode lỗi: {e}"
        return out
        
def macro_conflict_filter_v1(symbol: str, final_side: str, macro: dict) -> dict:
    symbol = str(symbol or "").upper()
    side = str(final_side or "NONE").upper()
    macro = macro or {}

    out = {
        "conflict": False,
        "severity": "NONE",
        "macro_bias": "NEUTRAL",
        "action": "ALLOW",
        "score_adjust": 0,
        "block_add": False,
        "reason": [],
    }

    if side not in ("BUY", "SELL"):
        out["reason"].append("final_side chưa rõ")
        return out

    mode = str(macro.get("macro_mode") or "NEUTRAL").upper()
    conf = float(macro.get("confidence") or 0)

    if "XAU" in symbol or "GOLD" in symbol:
        mb = str(macro.get("gold_bias") or "NEUTRAL").upper()
    elif "BTC" in symbol:
        mb = str(macro.get("btc_bias") or "NEUTRAL").upper()
    else:
        mb = "NEUTRAL"

    out["macro_bias"] = mb

    if mode not in ("STRONG_THEME", "WEAK_THEME") or mb in ("NEUTRAL", "MIXED", "NONE"):
        out["reason"].append("macro chưa đủ rõ để can thiệp")
        return out

    if mb == side:
        out["severity"] = "ALIGN"
        out["score_adjust"] = 5 if mode == "STRONG_THEME" else 2
        out["reason"].append(f"macro cùng chiều {side}")
        return out

    # macro ngược chart
    out["conflict"] = True
    out["block_add"] = True

    if mode == "STRONG_THEME" and conf >= 75:
        out["severity"] = "HIGH"
        out["action"] = "BLOCK_AGGRESSIVE_ENTRY"
        out["score_adjust"] = -12
        out["reason"].append(f"macro {mb} ngược final_side {side}")
        out["reason"].append("theme vĩ mô mạnh → không add/không vào đuổi")
    else:
        out["severity"] = "MEDIUM"
        out["action"] = "REDUCE_SIZE_WAIT_CONFIRM"
        out["score_adjust"] = -6
        out["reason"].append(f"macro {mb} ngược final_side {side}")
        out["reason"].append("chỉ trade nếu có confirmation rất rõ")

    return out
def build_bollinger_context_v1(candles, period=20, mult=2.0):
    try:
        import numpy as np

        closes = [
            float(_c_val(c, "close", 0.0) or 0.0)
            for c in (candles or [])
        ]
        closes = [x for x in closes if x > 0]

        if len(closes) < period + 20:
            return {"state": "NO_DATA", "reason": "not_enough_data", "len": len(closes)}

        arr = np.array(closes[-period:])
        ma = float(arr.mean())
        std = float(arr.std())

        upper = ma + mult * std
        lower = ma - mult * std
        width = upper - lower

        prev_widths = []
        for i in range(period, len(closes)):
            win = np.array(closes[i-period:i])
            w = (win.mean() + mult * win.std()) - (win.mean() - mult * win.std())
            prev_widths.append(float(w))

        avg_width = float(np.mean(prev_widths[-20:])) if prev_widths else width
        ratio = width / (avg_width or 1e-9)

        price = closes[-1]

        if ratio < 0.70:
            state = "SQUEEZE"
        elif ratio > 1.30:
            state = "EXPANSION"
        else:
            state = "NORMAL"

        if price > upper:
            position = "ABOVE_UPPER"
        elif price < lower:
            position = "BELOW_LOWER"
        elif price >= ma:
            position = "UPPER_HALF"
        else:
            position = "LOWER_HALF"

        return {
            "state": state,
            "position": position,
            "ma": ma,
            "upper": float(upper),
            "lower": float(lower),
            "width": float(width),
            "ratio": float(ratio),
            "len": len(closes),
        }

    except Exception as e:
        return {"state": "ERROR", "error": str(e)}

def build_ichimoku_context_v1(candles):
    try:
        highs = [
            float(_c_val(c, "high", 0.0) or 0.0)
            for c in (candles or [])
        ]
        lows = [
            float(_c_val(c, "low", 0.0) or 0.0)
            for c in (candles or [])
        ]
        closes = [
            float(_c_val(c, "close", 0.0) or 0.0)
            for c in (candles or [])
        ]

        rows = [
            (h, l, c)
            for h, l, c in zip(highs, lows, closes)
            if h > 0 and l > 0 and c > 0 and h >= l
        ]

        if len(rows) < 60:
            return {"state": "NO_DATA", "reason": "not_enough_data", "len": len(rows)}

        highs = [x[0] for x in rows]
        lows = [x[1] for x in rows]
        closes = [x[2] for x in rows]

        def mid(period):
            return (max(highs[-period:]) + min(lows[-period:])) / 2.0

        tenkan = mid(9)
        kijun = mid(26)
        senkou_a = (tenkan + kijun) / 2.0
        senkou_b = mid(52)

        price = closes[-1]
        cloud_top = max(senkou_a, senkou_b)
        cloud_bot = min(senkou_a, senkou_b)

        if price > cloud_top:
            state = "ABOVE_CLOUD"
        elif price < cloud_bot:
            state = "BELOW_CLOUD"
        else:
            state = "IN_CLOUD"

        thickness = abs(senkou_a - senkou_b)
        cloud = "THIN" if thickness < max(price * 0.002, 1e-9) else "THICK"

        tk_bias = "BULLISH" if tenkan > kijun else "BEARISH" if tenkan < kijun else "FLAT"

        return {
            "state": state,
            "cloud": cloud,
            "tk_bias": tk_bias,
            "tenkan": float(tenkan),
            "kijun": float(kijun),
            "senkou_a": float(senkou_a),
            "senkou_b": float(senkou_b),
            "cloud_top": float(cloud_top),
            "cloud_bot": float(cloud_bot),
            "len": len(rows),
        }

    except Exception as e:
        return {"state": "ERROR", "error": str(e)}

def build_momentum_phase_v1(candles):
    try:
        import numpy as np

        closes = [float(_c_val(c, "close", 0.0) or 0.0) for c in (candles or [])]
        closes = [x for x in closes if x > 0]

        if len(closes) < 10:
            return {"phase": "UNKNOWN", "reason": "not_enough_data"}

        arr = np.array(closes)
        moves = np.abs(np.diff(arr))

        if len(moves) < 6:
            return {"phase": "UNKNOWN", "reason": "not_enough_moves"}

        avg_move = np.mean(moves[:-3])
        recent_move = np.mean(moves[-3:])
        ratio = recent_move / (avg_move or 1e-9)

        if ratio < 0.7:
            phase = "EARLY"
        elif ratio < 1.5:
            phase = "MID"
        else:
            phase = "LATE"

        return {"phase": phase, "ratio": float(ratio)}

    except Exception as e:
        return {"phase": "ERROR", "error": str(e)}


def build_volatility_regime_v1(candles):
    try:
        import numpy as np

        highs = [float(_c_val(c, "high", 0.0) or 0.0) for c in (candles or [])]
        lows = [float(_c_val(c, "low", 0.0) or 0.0) for c in (candles or [])]

        pairs = [(h, l) for h, l in zip(highs, lows) if h > 0 and l > 0 and h >= l]

        if len(pairs) < 20:
            return {"state": "UNKNOWN", "reason": "not_enough_data"}

        ranges = np.array([h - l for h, l in pairs[-20:]])
        avg_atr = np.mean(ranges[:-5])
        recent_atr = np.mean(ranges[-5:])
        ratio = recent_atr / (avg_atr or 1e-9)

        if ratio < 0.7:
            state = "QUIET"
        elif ratio < 1.5:
            state = "NORMAL"
        else:
            state = "EXPANSION"

        return {"state": state, "ratio": float(ratio)}

    except Exception as e:
        return {"state": "ERROR", "error": str(e)}


def build_rsi_divergence_v1(candles, period=14):
    try:
        import numpy as np

        closes = [float(_c_val(c, "close", 0.0) or 0.0) for c in (candles or [])]
        closes = [x for x in closes if x > 0]

        if len(closes) < period + 5:
            return {"state": "NO_DATA"}

        rsi_now = _rsi(closes, period) or 50.0
        p1, p2 = closes[-5], closes[-1]

        # dùng RSI series có sẵn trong file cho sạch hơn
        rsis = _rsi_series(closes, period=period)
        rsi_prev = rsis[-5] if rsis and len(rsis) >= 5 else rsi_now

        if p2 > p1 and rsi_now < rsi_prev:
            return {"state": "BEARISH", "rsi": float(rsi_now)}

        if p2 < p1 and rsi_now > rsi_prev:
            return {"state": "BULLISH", "rsi": float(rsi_now)}

        return {"state": "NONE", "rsi": float(rsi_now)}

    except Exception as e:
        return {"state": "ERROR", "error": str(e)}

def analyze_pro(symbol: str, m15: Sequence[dict], m30: Sequence[dict], h1: Sequence[dict], h4: Sequence[dict], current_price: float | None = None) -> dict:
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
    bias_guess = None
    trade_mode = "WAIT"
    context_lines = base["context_lines"]
    position_lines = base.get("position_lines", [])
    liquidity_lines = base["liquidity_lines"]
    quality_lines = base["quality_lines"]
    notes = base.setdefault("notes", [])
    score = 0
    m15_struct = {}
    h1_struct = {}
    h4_struct = {}
    liquidity_map_v1 = {}
    entry_sniper = {}
    playbook_v4 = {}
    close_confirm_v4 = {"strength": "NO"}
    session_v4 = {}
    htf_pressure_v4 = {}
    macro_v4 = {}
    

    # ===== AUTO NEWS + MACRO ENGINE V2 =====
    try:
        meta = base.setdefault("meta", {})
    
        _dbg("[NEWS] START build_news_items()")
    
        try:
            news_items = build_news_items()
            if not isinstance(news_items, list):
                _dbg(f"[NEWS] bad return type: {type(news_items)}")
                news_items = []
    
            _dbg(f"[NEWS] fetched: {len(news_items)}")
    
            for n in news_items[:3]:
                _dbg(
                    f"[NEWS] item: {n.get('title')} | "
                    f"tags={n.get('tags')} | impact={n.get('impact')}"
                )
    
        except Exception as e:
            _dbg(f"[NEWS ERROR] build_news_items failed: {e}")
            news_items = []
    
        _dbg(f"[MACRO] input news count = {len(news_items)}")
    
        try:
            macro_ctx = build_macro_engine_v2(news_items)
            _dbg(f"[MACRO] raw ctx = {macro_ctx}")
        except Exception as e:
            _dbg(f"[MACRO ERROR] build_macro_engine_v2 failed: {e}")
            macro_ctx = {}
    
        if not isinstance(macro_ctx, dict):
            macro_ctx = {}
    
        macro_ctx.setdefault("macro_mode", "NEUTRAL")
        macro_ctx.setdefault("usd_strength", 0)
        macro_ctx.setdefault("risk_mode", "NEUTRAL")
        macro_ctx.setdefault("gold_bias", "NEUTRAL")
        macro_ctx.setdefault("btc_bias", "NEUTRAL")
        macro_ctx.setdefault("confidence", 0)
        macro_ctx.setdefault("drivers", [])
    
        meta["news_items"] = news_items
        meta["macro_v2"] = macro_ctx
    
        try:
            meta["macro_explain_tags_v1"] = explain_tags_v1(news_items)
        except Exception as e:

            meta["macro_explain_tags_v1"] = []
    
        try:
            meta["macro_reason_v1"] = explain_macro_reason_v1(macro_ctx)
        except Exception as e:

            meta["macro_reason_v1"] = []
    
    except Exception as e:

        meta = base.setdefault("meta", {})
        meta["news_items"] = []
        meta["macro_v2"] = {
            "macro_mode": "ERROR",
            "usd_strength": 0,
            "risk_mode": "NEUTRAL",
            "gold_bias": "NEUTRAL",
            "btc_bias": "NEUTRAL",
            "confidence": 0,
            "drivers": [f"macro error: {e}"],
        }
        meta["macro_explain_tags_v1"] = []
        meta["macro_reason_v1"] = []

    
    # ---- Safety / normalize candles
    m15c = _safe_candles(m15)
    m30c = _safe_candles(m30)
    h1c = _safe_candles(h1)
    h4c = _safe_candles(h4)
    # ===== normalize current_price =====
    try:
        if current_price is not None:
            current_price = float(current_price)
        elif m15c:
            current_price = float(_c_val(m15c[-1], "close", 0.0) or 0.0)
        else:
            current_price = None
    except Exception:
        try:
            current_price = float(_c_val(m15c[-1], "close", 0.0) or 0.0) if m15c else None
        except Exception:
            current_price = None
    # define ATR sớm để mọi early-return đều dùng được
    atr15 = _atr(m15c, 14) or 0.0 if m15c else 0.0
    
    if not m15c or not m30c or not h1c:
        base["note_lines"].append("⚠️ Thiếu dữ liệu M15/M30/H1 → không phân tích được.")
        base["short_hint"] = ["- Chưa đủ dữ liệu → CHỜ KÈO"]
        base["context_lines"] = ["Thị trường: n/a", "H1: n/a"]
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
        base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
        return base
    
    # inject structure trước rồi mới derive major_bearish
    _inject_meta_structure_and_levels(base, m15c, m30c, h1c, h4c)
    
    try:
        h1_tag = ((base.get("meta") or {}).get("structure") or {}).get("H1")
        major_bearish = str(h1_tag) in ("LL-LH", "LH–LL", "LH-LL")
    except Exception:
        major_bearish = False
    
    if not m15c or not m30c:
        base["note_lines"].append("⚠️ Không đọc được nến M15/M30 sau khi chuẩn hoá dữ liệu.")
        base["short_hint"] = ["- Dữ liệu nến lỗi / thiếu → CHỜ KÈO"]
        base["context_lines"] = ["Thị trường: n/a", "H1: n/a"]
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
        base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
        return base

    if not m15c or not m30c:
        base["note_lines"].append("⚠️ Không đọc được nến M15/M30 sau khi chuẩn hoá dữ liệu.")
        base["short_hint"] = ["- Dữ liệu nến lỗi / thiếu → CHỜ KÈO"]
        base["context_lines"] = ["Thị trường: n/a", "H1: n/a"]
        _inject_meta_structure_and_levels(base, m15, m30, h1, h4)
        base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
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
    meta = base.setdefault("meta", {})
    base["meta"]["volq"] = volq
    base["meta"]["candle"] = cpat
    base["meta"]["div"] = div
    base["meta"]["rsi14"] = rsi15
    base["meta"]["atr15"] = atr15
    meta["_m15_raw"] = m15c
    meta["_m30_raw"] = m30c
    meta["atr15"] = atr15
    meta["atr30"] = _atr(m30c, 14) if m30c else None
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

    ema_pack = _calc_ema_pack(m15c)
    if ema_pack:
        base["meta"]["ema"] = ema_pack

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

    # ===== MACRO CONFLICT FILTER V1 =====
    try:
        meta = base.setdefault("meta", {})
        macro = meta.get("macro_v2") or {}
    
        fs = str(
            meta.get("final_side")
            or (meta.get("signal_consistency_v1") or {}).get("final_side")
            or (meta.get("master_engine_v1") or {}).get("best_side")
            or base.get("side")
            or "NONE"
        ).upper()
    
        mcf = macro_conflict_filter_v1(symbol, fs, macro)
        meta["macro_conflict_filter_v1"] = mcf
    
        cur_score = float(base.get("final_score") or 0)
        base["final_score"] = max(
            0,
            min(100, cur_score + float(mcf.get("score_adjust") or 0))
        )
    
        if mcf.get("conflict") and mcf.get("severity") == "HIGH":
            base["tradeable"] = False
            meta["macro_block_reason"] = "Macro ngược hướng kỹ thuật mạnh"
    
        _dbg(
            f"[MACRO CONFLICT] fs={fs} "
            f"conflict={mcf.get('conflict')} "
            f"severity={mcf.get('severity')} "
            f"adjust={mcf.get('score_adjust')}"
        )
    
    except Exception as e:
        _dbg(f"[MACRO CONFLICT ERROR] {e}")

    # ===== INDICATOR ENGINE V1 (MERGED) =====
    _dbg("===== INDICATOR DEBUG START =====")
    _dbg(f"[CHECK] m15 exist: {'m15' in locals()}")
    _dbg(f"[CHECK] m15c exist: {'m15c' in locals()}")
    _dbg(f"[CHECK] candles_m15 exist: {'candles_m15' in locals()}")
    
    _dbg(f"[LEN] m15: {len(locals().get('m15', []))}")
    _dbg(f"[LEN] m15c: {len(locals().get('m15c', []))}")
    _dbg(f"[LEN] candles_m15: {len(locals().get('candles_m15', []))}")
    try:
        meta = base.setdefault("meta", {})
    
        m15_src = (
            locals().get("m15c")
            or locals().get("candles_m15")
            or locals().get("m15")
            or []
        )
    
        _dbg(f"[SRC] using: {'m15c' if locals().get('m15c') else 'candles_m15' if locals().get('candles_m15') else 'm15'}")
        _dbg(f"[SRC LEN] = {len(m15_src)}")
    
        try:
            rsi_div = build_rsi_divergence_v1(m15_src)
            meta["rsi_divergence_v1"] = rsi_div
            _dbg(f"[RSI DIV] {rsi_div}")
        except Exception as e:
            _dbg(f"[RSI DIV ERROR] {e}")
    
        try:
            mom = build_momentum_phase_v1(m15_src)
            meta["momentum_phase_v1"] = mom
            _dbg(f"[MOMENTUM] {mom}")
        except Exception as e:
            _dbg(f"[MOMENTUM ERROR] {e}")
    
        try:
            meta["volatility_regime_v1"] = build_volatility_regime_v1(m15_src)
        except Exception as e:
            _dbg(f"[VOL ERROR] {e}")
    
        try:
            bb = build_bollinger_context_v1(m15_src)
            meta["bollinger_context_v1"] = bb
            _dbg(f"[BOLLINGER RESULT] {bb}")
        except Exception as e:
            _dbg(f"[BOLLINGER ERROR] {e}")
    
        try:
            ichi = build_ichimoku_context_v1(m15_src)
            meta["ichimoku_context_v1"] = ichi
            _dbg(f"[ICHIMOKU RESULT] {ichi}")
        except Exception as e:
            _dbg(f"[ICHIMOKU ERROR] {e}")
    
    except Exception as e:
        _dbg(f"[INDICATOR ENGINE FATAL] {e}")
    
    meta.setdefault("rsi_divergence_v1", {"state": "NO_DATA"})
    meta.setdefault("momentum_phase_v1", {"phase": "UNKNOWN"})
    meta.setdefault("volatility_regime_v1", {"state": "UNKNOWN"})
    meta.setdefault("bollinger_context_v1", {"state": "NO_DATA"})
    meta.setdefault("ichimoku_context_v1", {"state": "NO_DATA"})
    _dbg(f"[META FINAL] keys = {list(meta.keys())}")
    _dbg(f"[META BOLL] {meta.get('bollinger_context_v1')}")
    _dbg(f"[META ICHI] {meta.get('ichimoku_context_v1')}")
    _dbg(f"[META MOM] {meta.get('momentum_phase_v1')}")
    # ===== MARKET MODE V1 =====
    # Đọc ngữ cảnh thị trường theo cụm nến:
    # HTF: H1/H4, MTF: M15 30-50 nến, Momentum: M15 10-15 nến
    try:
        meta = base.setdefault("meta", {})
        market_mode_v1 = _detect_market_mode_v1(
            symbol=symbol,
            m15c=m15c,
            h1c=h1c,
            h4c=h4c,
            ema_pack=meta.get("ema") or base.get("ema") or {},
            pullback_engine=meta.get("pullback_engine_v1") or {},
            range_pos=range_pos,
            market_state_v2=market_state_v2,
            playbook_v2=playbook_v2,
            post_break_continuity=meta.get("post_break_continuity_v1") or {},
        )
        meta["market_mode_v1"] = market_mode_v1

        # Sync nhẹ sang signal consistency: không biến WAIT thành lệnh,
        # chỉ đổi cách diễn giải để bot không nói "NO_TRADE" cụt ngủn trong trend day.
        sce1 = meta.get("signal_consistency_v1") or {}
        if market_mode_v1.get("mode") in ("TREND_DAY_DOWN", "TREND_DAY_UP"):
            sce1["market_mode"] = market_mode_v1.get("mode")
            sce1["context_side"] = market_mode_v1.get("side")
            sce1["action_mode"] = market_mode_v1.get("action_mode")
            sce1["narrative"] = (
                "Trend day đang chạy; không có entry đẹp nhưng vẫn có thể theo dõi continuation có điều kiện"
            )
            meta["signal_consistency_v1"] = sce1

    except Exception as e:
        base.setdefault("meta", {})["market_mode_v1_error"] = str(e)
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
                base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
                base["last_price"] = float(last_close_15)
                base["current_price"] = float(last_close_15)
                _attach_vnext_meta(
                    base,
                    symbol=symbol,
                    m15c=m15c,
                    bias_side=bias_guess,
                    h1_trend=h1_trend,
                    h4_trend=h4_trend,
                    market_state_v2=market_state_v2,
                    flow_state=flow_state,
                    range_pos=range_pos,
                    no_trade_zone=no_trade_zone,
                    liquidation_evt=liquidation_evt,
                    m15_struct=(locals().get("m15_struct") or {}),
                    rsi15=rsi15,
                    div=div,
                    atr15=atr15,
                    liquidity_map_v1=(locals().get("liquidity_map_v1") or {}),
                    ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
                    playbook_v2=(playbook_v2 if isinstance(playbook_v2, dict) else {}),
                    close_confirm_v4=(locals().get("close_confirm_v4") or {"strength": "NO"}),
                    sweep_buy=(sweep_buy if isinstance(sweep_buy, dict) else {}),
                    sweep_sell=(sweep_sell if isinstance(sweep_sell, dict) else {}),
                    spring_buy=(spring_buy if isinstance(spring_buy, dict) else {}),
                    spring_sell=(spring_sell if isinstance(spring_sell, dict) else {}),
                    entry_sniper=(locals().get("entry_sniper") or {}),
                    playbook_v4=(locals().get("playbook_v4") or {}),
                )
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
                _attach_vnext_meta(
                    base,
                    symbol=symbol,
                    m15c=m15c,
                    bias_side=bias_guess,
                    h1_trend=h1_trend,
                    h4_trend=h4_trend,
                    market_state_v2=market_state_v2,
                    flow_state=flow_state,
                    range_pos=range_pos,
                    no_trade_zone=no_trade_zone,
                    liquidation_evt=liquidation_evt,
                    m15_struct=(locals().get("m15_struct") or {}),
                    rsi15=rsi15,
                    div=div,
                    atr15=atr15,
                    liquidity_map_v1=(locals().get("liquidity_map_v1") or {}),
                    ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
                    playbook_v2=(playbook_v2 if isinstance(playbook_v2, dict) else {}),
                    close_confirm_v4=(locals().get("close_confirm_v4") or {"strength": "NO"}),
                    sweep_buy=(sweep_buy if isinstance(sweep_buy, dict) else {}),
                    sweep_sell=(sweep_sell if isinstance(sweep_sell, dict) else {}),
                    spring_buy=(spring_buy if isinstance(spring_buy, dict) else {}),
                    spring_sell=(spring_sell if isinstance(spring_sell, dict) else {}),
                    entry_sniper=(locals().get("entry_sniper") or {}),
                    playbook_v4=(locals().get("playbook_v4") or {}),
                )
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
        base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
        base["last_price"] = float(last_close_15)
        base["current_price"] = float(last_close_15)
        _attach_vnext_meta(
            base,
            symbol=symbol,
            m15c=m15c,
            bias_side=bias_guess,
            h1_trend=h1_trend,
            h4_trend=h4_trend,
            market_state_v2=market_state_v2,
            flow_state=flow_state,
            range_pos=range_pos,
            no_trade_zone=no_trade_zone,
            liquidation_evt=liquidation_evt,
            m15_struct=(locals().get("m15_struct") or {}),
            rsi15=rsi15,
            div=div,
            atr15=atr15,
            liquidity_map_v1=(locals().get("liquidity_map_v1") or {}),
            ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
            playbook_v2=(playbook_v2 if isinstance(playbook_v2, dict) else {}),
            close_confirm_v4=(locals().get("close_confirm_v4") or {"strength": "NO"}),
            sweep_buy=(sweep_buy if isinstance(sweep_buy, dict) else {}),
            sweep_sell=(sweep_sell if isinstance(sweep_sell, dict) else {}),
            spring_buy=(spring_buy if isinstance(spring_buy, dict) else {}),
            spring_sell=(spring_sell if isinstance(spring_sell, dict) else {}),
            entry_sniper=(locals().get("entry_sniper") or {}),
            playbook_v4=(locals().get("playbook_v4") or {}),
        )
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
            base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
            base["last_price"] = float(last_close_15)
            base["current_price"] = float(last_close_15)
            _attach_vnext_meta(
                base,
                symbol=symbol,
                m15c=m15c,
                bias_side=bias_guess,
                h1_trend=h1_trend,
                h4_trend=h4_trend,
                market_state_v2=market_state_v2,
                flow_state=flow_state,
                range_pos=range_pos,
                no_trade_zone=no_trade_zone,
                liquidation_evt=liquidation_evt,
                m15_struct=(locals().get("m15_struct") or {}),
                rsi15=rsi15,
                div=div,
                atr15=atr15,
                liquidity_map_v1=(locals().get("liquidity_map_v1") or {}),
                ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
                playbook_v2=(playbook_v2 if isinstance(playbook_v2, dict) else {}),
                close_confirm_v4=(locals().get("close_confirm_v4") or {"strength": "NO"}),
                sweep_buy=(sweep_buy if isinstance(sweep_buy, dict) else {}),
                sweep_sell=(sweep_sell if isinstance(sweep_sell, dict) else {}),
                spring_buy=(spring_buy if isinstance(spring_buy, dict) else {}),
                spring_sell=(spring_sell if isinstance(spring_sell, dict) else {}),
                entry_sniper=(locals().get("entry_sniper") or {}),
                playbook_v4=(locals().get("playbook_v4") or {}),
            )
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
            base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
            base["last_price"] = float(last_close_15)
            base["current_price"] = float(last_close_15)
            _attach_vnext_meta(
                base,
                symbol=symbol,
                m15c=m15c,
                bias_side=bias_guess,
                h1_trend=h1_trend,
                h4_trend=h4_trend,
                market_state_v2=market_state_v2,
                flow_state=flow_state,
                range_pos=range_pos,
                no_trade_zone=no_trade_zone,
                liquidation_evt=liquidation_evt,
                m15_struct=(locals().get("m15_struct") or {}),
                rsi15=rsi15,
                div=div,
                atr15=atr15,
                liquidity_map_v1=(locals().get("liquidity_map_v1") or {}),
                ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
                playbook_v2=(playbook_v2 if isinstance(playbook_v2, dict) else {}),
                close_confirm_v4=(locals().get("close_confirm_v4") or {"strength": "NO"}),
                sweep_buy=(sweep_buy if isinstance(sweep_buy, dict) else {}),
                sweep_sell=(sweep_sell if isinstance(sweep_sell, dict) else {}),
                spring_buy=(spring_buy if isinstance(spring_buy, dict) else {}),
                spring_sell=(spring_sell if isinstance(spring_sell, dict) else {}),
                entry_sniper=(locals().get("entry_sniper") or {}),
                playbook_v4=(locals().get("playbook_v4") or {}),
            )
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
        base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
        base["last_price"] = float(last_close_15)
        base["current_price"] = float(last_close_15)
        _attach_vnext_meta(
            base,
            symbol=symbol,
            m15c=m15c,
            bias_side=bias_guess,
            h1_trend=h1_trend,
            h4_trend=h4_trend,
            market_state_v2=market_state_v2,
            flow_state=flow_state,
            range_pos=range_pos,
            no_trade_zone=no_trade_zone,
            liquidation_evt=liquidation_evt,
            m15_struct=(locals().get("m15_struct") or {}),
            rsi15=rsi15,
            div=div,
            atr15=atr15,
            liquidity_map_v1=(locals().get("liquidity_map_v1") or {}),
            ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
            playbook_v2=(playbook_v2 if isinstance(playbook_v2, dict) else {}),
            close_confirm_v4=(locals().get("close_confirm_v4") or {"strength": "NO"}),
            sweep_buy=(sweep_buy if isinstance(sweep_buy, dict) else {}),
            sweep_sell=(sweep_sell if isinstance(sweep_sell, dict) else {}),
            spring_buy=(spring_buy if isinstance(spring_buy, dict) else {}),
            spring_sell=(spring_sell if isinstance(spring_sell, dict) else {}),
            entry_sniper=(locals().get("entry_sniper") or {}),
            playbook_v4=(locals().get("playbook_v4") or {}),
        )
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
    liquidity_map_v1 = _build_liquidity_map_v1(
        symbol=symbol,
        m15c=m15c,
        h1_trend=h1_trend,
        htf_pressure_v4=(htf_pressure_v4 or {}),
        flow_state=flow_state,
        range_pos=range_pos,
        market_state_v2=market_state_v2,
        playbook_v2=playbook_v2,
        liquidation_evt=liquidation_evt,
        m15_struct_tag=m15_struct.get("tag") if isinstance(m15_struct, dict) else "n/a",
        range_low=base.get("meta", {}).get("key_levels", {}).get("M15_RANGE_LOW"),
        range_high=base.get("meta", {}).get("key_levels", {}).get("M15_RANGE_HIGH"),
    )

    mm_play = _build_mm_real_play_v1(
        liq_map=liquidity_map_v1,
        range_pos=range_pos,
        htf_pressure_v4=(htf_pressure_v4 or {}),
        playbook_v2=playbook_v2,
        ema_pack=ema_pack,
    )
    base.setdefault("meta", {})["mm_real_play_v1"] = mm_play
    
    entry_sniper = _entry_sniper_v1(
        m15c=m15c,
        m15_struct=m15_struct,
        atr15=atr15,
        volq=volq,
    )
    base.setdefault("meta", {})["entry_sniper"] = entry_sniper
    
    pump_dump_v1 = _predict_pump_dump_v1(
        symbol=symbol,
        m15c=m15c,
        h1_trend=h1_trend,
        htf_pressure_v4=(htf_pressure_v4 or {}),
        market_state_v2=market_state_v2,
        flow_state=flow_state,
        range_pos=range_pos,
        volq=volq,
        atr15=atr15,
        m15_struct_tag=m15_struct.get("tag") if isinstance(m15_struct, dict) else "n/a",
        liquidation_evt=liquidation_evt,
        liquidity_map_v1=liquidity_map_v1,
    )
    base.setdefault("meta", {})["pump_dump_v1"] = pump_dump_v1
    
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
            tol_pct = 0.11 if xau_priority else 0.12
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
            momentum_ok = int(bool(bos_retest or bos_micro_retest or engulf_aligned))
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
    m15_last_close = float(m15c[-1].close) if m15c else None
    major_bos_level = h1_struct.get("hh") if bias_side == "BUY" else h1_struct.get("ll")
    try:
        major_bos_confirmed = (
            major_bos_level is not None and m15_last_close is not None and (
                (m15_last_close > float(major_bos_level)) if bias_side == "BUY"
                else (m15_last_close < float(major_bos_level))
            )
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
    try:
        if range_pos is not None:
            if range_pos >= 0.8:
                if str(liquidity_map_v1.get("sweep_bias")).startswith("UP"):
                    recommendation = "SELL"
    
            if range_pos <= 0.2:
                if str(liquidity_map_v1.get("sweep_bias")).startswith("DOWN"):
                    recommendation = "BUY"
    except:
        pass
    quality_lines.append("RR ~ 1:2")
    if rdist is not None:
        quality_lines.append(f"R~{rdist:.2f} | SL=MIN(Liq, ATR, Risk) (risk engine)")
    stars = max(1, min(5, int(stars)))

    # FIX ưu tiên theo liquidity
    # ===== GD2 final attach with scoring-aware no-trade =====
    no_trade_zone = _detect_no_trade_zone_v2(
        bias_side,
        market_state_v2,
        range_pos,
        liq_warn,
        liquidation_evt,
        confirmation_ok=bool(momentum_ok),
    )
    
    # ---- defaults (avoid UnboundLocalError when structure/bias is n/a) ----
    trade_mode = "HALF"  # default; will be overwritten once scoring is computed
    entry_major = sl_major = tp1_major = tp2_major = None
    entry_minor = sl_minor = tp1_minor = tp2_minor = None
    # ===== GD2 recompute after bias decision =====
    bias_for_gd2 = bias if bias in ("BUY", "SELL") else bias_guess
    market_state_v2 = _detect_market_state_v2(h1_trend, h4_trend, range_pos, atr15, avg20, avg80, div, liquidation_evt)
    flow_state = _detect_flow_state_v2(symbol, h1_trend, h4_trend, market_state_v2, range_pos)
    #no_trade_zone = _detect_no_trade_zone_v2(bias_for_gd2, market_state_v2, range_pos, liq_warn, liquidation_evt, confirmation_ok=None)
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

    # ===== VNEXT ADD-ON =====
    context_verdict_v1 = _context_verdict_v1(
        bias_side=bias_side,
        h1_trend=h1_trend,
        h4_trend=h4_trend,
        market_state_v2=market_state_v2,
        flow_state=flow_state,
        range_pos=range_pos,
        no_trade_zone=no_trade_zone,
        liquidation_evt=liquidation_evt,
        m15_struct_tag=m15_struct.get("tag") if isinstance(m15_struct, dict) else "n/a",
    )

    rsi_context_v1 = _rsi_context_v1(
        rsi15=rsi15,
        bias_side=bias_side,
        h1_trend=h1_trend,
        market_state_v2=market_state_v2,
        div=div,
        liquidation_evt=liquidation_evt,
    )

    fib_confluence_v1 = _fib_confluence_v1(
        m15c=m15c,
        bias_side=bias_side,
        atr15=atr15,
        liquidity_map_v1=liquidity_map_v1,
        ema_pack=ema_pack,
        playbook_v2=playbook_v2,
    )

    liquidity_completion_v1 = _liquidity_completion_v1(
        sweep_buy=sweep_buy,
        sweep_sell=sweep_sell,
        spring_buy=spring_buy,
        spring_sell=spring_sell,
        close_confirm_v4=close_confirm_v4,
        entry_sniper=entry_sniper,
        bias_side=bias_side,
    )

    trap_warning_v1 = _trap_warning_v1(
        bias_side=bias_side,
        context_verdict=context_verdict_v1,
        rsi_ctx=rsi_context_v1,
        no_trade_zone=no_trade_zone,
        liquidation_evt=liquidation_evt,
        range_pos=range_pos,
        div=div,
        close_confirm_v4=close_confirm_v4,
    )

    manual_likelihood_v1 = _manual_likelihood_v1(
        bias_side=bias_side,
        context_verdict=context_verdict_v1,
        trap_warning=trap_warning_v1,
        fib_conf=fib_confluence_v1,
        liq_done=liquidity_completion_v1,
        close_confirm_v4=close_confirm_v4,
        entry_sniper=entry_sniper,
        playbook_v4=playbook_v4,
    )

    manual_guidance_v1 = _manual_guidance_v1(
        bias_side=bias_side,
        context_verdict=context_verdict_v1,
        liq_done=liquidity_completion_v1,
        fib_conf=fib_confluence_v1,
        close_confirm_v4=close_confirm_v4,
        entry_sniper=entry_sniper,
        playbook_v2=playbook_v2,
    )
    # ===== LIQUIDITY REACTION ENGINE V1 =====
    meta = base.setdefault("meta", {})
    liquidity_reaction_v1 = _liquidity_reaction_engine_v1(
        signal_consistency_v1=(base.get("meta", {}) or {}).get("signal_consistency_v1"),
        liquidity_map=liquidity_map if 'liquidity_map' in locals() and isinstance(liquidity_map, dict) else {},
        liquidity_completion_v1=liquidity_completion_v1,
        close_confirm_v4=close_confirm_v4,
        m15_struct=(m15_struct or {}),
        current_price=(
            last_px if 'last_px' in locals()
            else (cur if 'cur' in locals() else None)
        ),
        range_lo=(
            range_lo if 'range_lo' in locals()
            else (lo if 'lo' in locals() else None)
        ),
        range_hi=(
            range_hi if 'range_hi' in locals()
            else (hi if 'hi' in locals() else None)
        ),
    )
    meta["liquidity_reaction_v1"] = liquidity_reaction_v1

    # ===== ABSORPTION ENGINE =====
    # FIX: key_levels phải luôn được khởi tạo trước khi dùng
    meta = base.setdefault("meta", {})
    k = meta.get("key_levels") or {}
    absorption_v1 = _absorption_v1(
        m15c=m15c,
        volq=volq if 'volq' in locals() else {},
        range_low=(k or {}).get("M15_RANGE_LOW"),
        range_high=(k or {}).get("M15_RANGE_HIGH"),
    )
    # ===== FLOW ENGINE V1 (merge GAP + FVG + liquidity) =====
    gap_info_v1 = _detect_session_gap_v1(
        m15c=m15c,
        atr15=atr15,
    )
    flow_engine_v1 = _build_flow_engine_v1(
        symbol=symbol,
        m15c=m15c,
        current_price=base.get("current_price") or (base.get("meta") or {}).get("current_price"),
        atr15=atr15,
        liquidity_map_v1=liquidity_map_v1 if isinstance(liquidity_map_v1, dict) else {},
        fvg_range_plugin_v1=(base.get("meta", {}) or {}).get("fvg_range_plugin_v1") or {},
        gap_info_v1=gap_info_v1,
    )
    base.setdefault("meta", {})["gap_info_v1"] = gap_info_v1
    base.setdefault("meta", {})["flow_engine_v1"] = flow_engine_v1

    # ===== AUTO NEWS + MACRO ENGINE V2 =====

        
    # ===== ELLIOTT PHASE V1 =====
    try:
        meta = base.setdefault("meta", {})
    
        k = meta.get("key_levels") or {}
        flow1 = meta.get("flow_engine_v1") or {}
        za1 = meta.get("zone_action_v1") or {}
    
        elliott_phase_v1 = _elliott_phase_v1(
            h4_struct=(struct.get("H4") if isinstance(struct, dict) else None),
            h1_struct=(struct.get("H1") if isinstance(struct, dict) else None),
            m15_struct=(struct.get("M15") if isinstance(struct, dict) else None),
            pullback_info=pb if "pb" in locals() and isinstance(pb, dict) else {},
            ema_filter=ema_filter if "ema_filter" in locals() and isinstance(ema_filter, dict) else {},
            flow_engine_v1=flow1,
            zone_action_v1=za1,
            current_price=current_price,
            range_low=k.get("M15_RANGE_LOW"),
            range_high=k.get("M15_RANGE_HIGH"),
        )
    
        meta["elliott_phase_v1"] = elliott_phase_v1

    except Exception as e:

        base.setdefault("meta", {})["elliott_phase_v1"] = {
            "ok": False,
            "main_tf": "H1/H4",
            "phase": "ERROR",
            "confidence": 0,
            "meaning": f"Elliott phase lỗi: {e}",
            "action": "Bỏ qua Elliott context",
            "invalid": "n/a",
            "reason": [],
        }

    base.setdefault("meta", {})["context_verdict_v1"] = context_verdict_v1
    base.setdefault("meta", {})["rsi_context_v1"] = rsi_context_v1
    base.setdefault("meta", {})["fib_confluence_v1"] = fib_confluence_v1
    base.setdefault("meta", {})["liquidity_completion_v1"] = liquidity_completion_v1
    base.setdefault("meta", {})["trap_warning_v1"] = trap_warning_v1
    base.setdefault("meta", {})["manual_likelihood_v1"] = manual_likelihood_v1
    base.setdefault("meta", {})["manual_guidance_v1"] = manual_guidance_v1    
    base.setdefault("meta", {})["liquidity_map_v1"] = liquidity_map_v1    
    base.setdefault("meta", {})["absorption_v1"] = absorption_v1
    # ===== PRO DESK META ATTACH =====
    meta = base.setdefault("meta", {})

    m15_tag = str((m15_struct or {}).get("tag") or "n/a").upper()

    # 1) Market state
    if (liquidation_evt or {}).get("ok"):
        market_state_machine_v1 = {
            "state": "LIQUIDATION",
            "label": "Biến động thanh khoản mạnh",
        }
    elif str(market_state_v2 or "").upper() in ("CHOP", "TRANSITION"):
        market_state_machine_v1 = {
            "state": "CHOP",
            "label": "Nhiễu / chuyển pha",
        }
    elif h1_trend == "bullish" and h4_trend == "bullish":
        if "LL" in m15_tag or "LH" in m15_tag:
            market_state_machine_v1 = {
                "state": "PULLBACK_BUY",
                "label": "Hồi trong xu hướng tăng",
            }
        else:
            market_state_machine_v1 = {
                "state": "TREND_UP",
                "label": "Xu hướng tăng",
            }
    elif h1_trend == "bearish" and h4_trend == "bearish":
        if "HH" in m15_tag or "HL" in m15_tag:
            market_state_machine_v1 = {
                "state": "PULLBACK_SELL",
                "label": "Hồi trong xu hướng giảm",
            }
        else:
            market_state_machine_v1 = {
                "state": "TREND_DOWN",
                "label": "Xu hướng giảm",
            }
    else:
        market_state_machine_v1 = {
            "state": "TRANSITION",
            "label": "Chuyển pha",
        }

    # 2) Bias layers
    if h1_trend == "bullish" and h4_trend == "bullish":
        htf_bias = "BUY"
    elif h1_trend == "bearish" and h4_trend == "bearish":
        htf_bias = "SELL"
    else:
        htf_bias = "MIXED"

    if "LL" in m15_tag or "LH" in m15_tag:
        mtf_bias = "SELL_PULLBACK"
    elif "HH" in m15_tag or "HL" in m15_tag:
        mtf_bias = "BUY_PULLBACK"
    else:
        mtf_bias = "WAIT"

    cv_state = str((context_verdict_v1 or {}).get("state") or "")
    entry_bias = "READY" if "CONTINUATION" in cv_state else "WAIT"

    bias_layers_v1 = {
        "htf_bias": htf_bias,
        "mtf_bias": mtf_bias,
        "entry_bias": entry_bias,
    }

    # 3) No trade zone
    ntz_reasons = []
    try:
        rp = float(range_pos) if range_pos is not None else None
    except Exception:
        rp = None

    if rp is not None and 0.35 <= rp <= 0.65:
        ntz_reasons.append("giữa biên độ")

    if str((liquidity_completion_v1 or {}).get("state") or "NO").upper() == "NO":
        ntz_reasons.append("chưa có thanh khoản")

    if str((close_confirm_v4 or {}).get("strength") or "NO").upper() in ("NO", "N/A"):
        ntz_reasons.append("chưa có confirm")

    if m15_tag in ("TRANSITION", "N/A", ""):
        ntz_reasons.append("M15 chưa rõ")

    if (trap_warning_v1 or {}).get("active"):
        ntz_reasons.append("trap risk")

    no_trade_zone_v3 = {
        "active": len(ntz_reasons) >= 2,
        "reasons": ntz_reasons[:5],
    }

    # 4) Decision
    sniper_trigger = str((entry_sniper or {}).get("trigger") or "NONE").upper()
    if no_trade_zone_v3["active"]:
        decision_engine_v1 = {
            "decision": "STAND ASIDE",
            "reason": "No-trade zone",
        }
    elif sniper_trigger in ("READY", "TRIGGERED"):
        decision_engine_v1 = {
            "decision": "MANUAL STRIKE",
            "reason": "Có trigger",
        }
    else:
        decision_engine_v1 = {
            "decision": "WAIT",
            "reason": "Chưa đủ điều kiện",
        }

    # 5) Wait for
    wf_lines = []
    zl = (playbook_v2 or {}).get("zone_low")
    zh = (playbook_v2 or {}).get("zone_high")
    if zl is not None and zh is not None:
        wf_lines.append(f"Chờ vùng {_fmt(zl)} – {_fmt(zh)}")

    k2 = meta.get("key_levels", {}) or {}
    if bias_side == "BUY":
        lv = k2.get("M15_RANGE_HIGH")
        if lv is not None:
            wf_lines.append(f"Hoặc break {_fmt(lv)}")
    elif bias_side == "SELL":
        lv = k2.get("M15_RANGE_LOW")
        if lv is not None:
            wf_lines.append(f"Hoặc break {_fmt(lv)}")

    wait_for_v1 = {"lines": wf_lines[:4]}

    meta["market_state_machine_v1"] = market_state_machine_v1
    meta["bias_layers_v1"] = bias_layers_v1
    meta["no_trade_zone_v3"] = no_trade_zone_v3
    meta["decision_engine_v1"] = decision_engine_v1
    meta["wait_for_v1"] = wait_for_v1


    
    # ===== PRO DESK ADD =====
    market_state_machine_v1 = _market_state_machine_v1(
        h1_trend=h1_trend,
        h4_trend=h4_trend,
        m15_struct_tag=(m15_struct or {}).get("tag"),
        market_state_v2=market_state_v2,
        liquidation_evt=liquidation_evt,
        range_pos=range_pos,
    )
    
    bias_layers_v1 = _bias_layers_v1(
        h1_trend=h1_trend,
        h4_trend=h4_trend,
        m15_struct_tag=(m15_struct or {}).get("tag"),
        context_verdict=context_verdict_v1,
    )
    
    no_trade_zone_v3 = _no_trade_zone_v3(
        range_pos=range_pos,
        liquidity_done=liquidity_completion_v1,
        close_confirm_v4=close_confirm_v4,
        m15_struct_tag=(m15_struct or {}).get("tag"),
        trap_warning=trap_warning_v1,
    )
    
    decision_engine_v1 = _decision_engine_v1(
        no_trade_zone=no_trade_zone_v3,
        context_verdict=context_verdict_v1,
        entry_sniper=entry_sniper,
        close_confirm_v4=close_confirm_v4,
        manual_likelihood=manual_likelihood_v1,
    )
    
    wait_for_v1 = _wait_for_engine_v1(
        bias_side=bias_side,
        playbook_v2=playbook_v2,
        key_levels=base.get("meta", {}).get("key_levels", {}),
    )
    
    meta = base.setdefault("meta", {})
    meta["market_state_machine_v1"] = market_state_machine_v1
    meta["bias_layers_v1"] = bias_layers_v1
    meta["no_trade_zone_v3"] = no_trade_zone_v3
    meta["decision_engine_v1"] = decision_engine_v1
    meta["wait_for_v1"] = wait_for_v1


    # ===== VNEXT RENDER APPEND =====
    try:
        cv = context_verdict_v1
        context_lines.append(f"Context verdict: {cv.get('verdict')}")
        if cv.get("reason"):
            context_lines.append(" | ".join(cv.get("reason")[:2]))
    except Exception:
        pass

    try:
        rc = rsi_context_v1
        quality_lines.append(f"RSI context: {rc.get('message')}")
    except Exception:
        pass

    try:
        liqd = liquidity_completion_v1
        quality_lines.append(f"Liquidity done: {liqd.get('state')} | {liqd.get('message')}")
    except Exception:
        pass

    try:
        fibc = fib_confluence_v1
        if fibc.get("ok"):
            quality_lines.append(
                f"Fib confluence: YES | zone {_fmt(fibc.get('zone_low'))} – {_fmt(fibc.get('zone_high'))}"
            )
        else:
            quality_lines.append("Fib confluence: NO")
    except Exception:
        pass

    try:
        tw = trap_warning_v1
        if tw.get("active"):
            for s in tw.get("warnings", [])[:4]:
                notes.append(f"⚠️ Trap: {s}")
    except Exception:
        pass

    try:
        ml = manual_likelihood_v1
        notes.append(
            f"📊 Manual likelihood | BUY={int(ml.get('buy_likelihood', 0))}/100 | "
            f"SELL={int(ml.get('sell_likelihood', 0))}/100 | "
            f"Trap={int(ml.get('trap_risk', 0))}/100"
        )
    except Exception:
        pass

    try:
        mg = manual_guidance_v1
        for s in mg.get("lines", [])[:4]:
            notes.append(f"🧭 {s}")
    except Exception:
        pass
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
    base = _inject_wait_levels_v1(base, bias_guess, m15c, m30c, h1c, atr15)
    base.setdefault("meta", {})["_m15_raw"] = m15c
    base["meta"]["_m30_raw"] = m30c
    base["meta"]["_h1_raw"] = h1c
    base["meta"]["atr15"] = atr15
    base["last_price"] = float(last_close_15)
    base["current_price"] = float(last_close_15)
    _attach_vnext_meta(
        base,
        symbol=symbol,
        m15c=m15c,
        bias_side=bias_guess,
        h1_trend=h1_trend,
        h4_trend=h4_trend,
        market_state_v2=market_state_v2,
        flow_state=flow_state,
        range_pos=range_pos,
        no_trade_zone=no_trade_zone,
        liquidation_evt=liquidation_evt,
        m15_struct=(locals().get("m15_struct") or {}),
        rsi15=rsi15,
        div=div,
        atr15=atr15,
        liquidity_map_v1=(locals().get("liquidity_map_v1") or {}),
        ema_pack=(ema_pack if isinstance(ema_pack, dict) else {}),
        playbook_v2=(playbook_v2 if isinstance(playbook_v2, dict) else {}),
        close_confirm_v4=(locals().get("close_confirm_v4") or {"strength": "NO"}),
        sweep_buy=(sweep_buy if isinstance(sweep_buy, dict) else {}),
        sweep_sell=(sweep_sell if isinstance(sweep_sell, dict) else {}),
        spring_buy=(spring_buy if isinstance(spring_buy, dict) else {}),
        spring_sell=(spring_sell if isinstance(spring_sell, dict) else {}),
        entry_sniper=(locals().get("entry_sniper") or {}),
        playbook_v4=(locals().get("playbook_v4") or {}),
    )
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
        
def get_now_status(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Safe NOW-status snapshot for cron/filtering and Telegram rendering.
    Always returns a dict and never raises.
    """
    try:
        sig = sig if isinstance(sig, dict) else {}
        meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
        playbook = meta.get("playbook_v2") if isinstance(meta.get("playbook_v2"), dict) else {}
        ntz = meta.get("no_trade_zone") if isinstance(meta.get("no_trade_zone"), dict) else {}
        struct = meta.get("structure") if isinstance(meta.get("structure"), dict) else {}
        session_v4 = meta.get("session_v4") if isinstance(meta.get("session_v4"), dict) else {}
        htf_pressure_v4 = meta.get("htf_pressure_v4") if isinstance(meta.get("htf_pressure_v4"), dict) else {}
        liq_evt = meta.get("liquidation") if isinstance(meta.get("liquidation"), dict) else {}
        k = meta.get("key_levels") if isinstance(meta.get("key_levels"), dict) else {}

        reasons = []
        trade_reasons = []
        setup = 50

        htf_state = str(htf_pressure_v4.get("state") or "").upper()
        if "STRONG" in htf_state:
            setup += 12
            reasons.append("HTF mạnh")
        elif "WEAK" in htf_state:
            setup += 6
            reasons.append("HTF hơi nghiêng")
        else:
            setup -= 4
            reasons.append("HTF chưa đồng thuận")

        m15 = str((struct or {}).get("M15") or "").upper()
        h1 = str((struct or {}).get("H1") or "").upper()
        if m15 in ("HH-HL", "LL-LH"):
            setup += 8
            reasons.append("M15 rõ cấu trúc")
        else:
            setup -= 4
            reasons.append("M15 chưa rõ")
        if h1 in ("HH-HL", "LL-LH"):
            setup += 6
            reasons.append("H1 rõ cấu trúc")
        elif h1 == "TRANSITION":
            setup -= 3
            reasons.append("H1 đang chuyển pha")

        sess = str(session_v4.get("session_tag") or "").upper()
        ft = str(session_v4.get("follow_through") or "").upper()
        fake = str(session_v4.get("fake_move_risk") or "").upper()
        if "CHOP" in sess:
            setup -= 6
            reasons.append("session nhiễu")
        if ft == "YES":
            setup += 5
            reasons.append("có follow-through")
        elif ft == "NO":
            setup -= 5
            reasons.append("thiếu follow-through")
        if fake == "HIGH":
            setup -= 8
            reasons.append("fake risk cao")
        elif fake == "MEDIUM":
            setup -= 3
            reasons.append("fake risk trung bình")

        zlo = _safe_float(playbook.get("zone_low"))
        zhi = _safe_float(playbook.get("zone_high"))
        if zlo is not None and zhi is not None:
            setup += 6
            reasons.append("có vùng entry")
        else:
            setup -= 8
            reasons.append("chưa có vùng entry")

        rp = None
        try:
            rp = float(playbook.get("range_pos"))
            if rp <= 0.20 or rp >= 0.80:
                setup += 8
                reasons.append("đang ở vùng biên")
            elif 0.40 <= rp <= 0.60:
                setup -= 10
                reasons.append("đang ở giữa biên độ")
            elif 0.20 < rp < 0.40:
                reasons.append("đang ở nửa dưới range")
            elif 0.60 < rp < 0.80:
                reasons.append("đang ở nửa trên range")
        except Exception:
            rp = None

        if ntz.get("active"):
            setup -= 8
            reasons.append("đang ở no-trade zone")

        setup = max(0, min(100, setup))

        last_px = _safe_float(k.get("M15_LAST"))
        if last_px is None:
            last_px = _safe_float(sig.get("last_price"))
        if last_px is None:
            last_px = _safe_float(sig.get("current_price"))
        entry = 50
        tradeable_now = True

        if zlo is None or zhi is None:
            entry -= 20
            tradeable_now = False
            trade_reasons.append("chưa có vùng entry rõ")
        elif last_px is not None:
            lo, hi = min(zlo, zhi), max(zlo, zhi)
            width = max(hi - lo, 1e-9)
            dist = 0.0
            if last_px < lo:
                dist = (lo - last_px) / width
            elif last_px > hi:
                dist = (last_px - hi) / width
            if lo <= last_px <= hi:
                entry += 25
            elif dist <= 0.20:
                entry += 12
                trade_reasons.append("giá đang gần vùng vào")
            elif dist <= 0.50:
                entry -= 8
                tradeable_now = False
                trade_reasons.append("chưa vào đúng vùng entry")
            else:
                entry -= 22
                tradeable_now = False
                trade_reasons.append("giá đang xa vùng entry")

        rec = str(sig.get("recommendation") or "").upper()
        if rp is not None:
            if rec in ("🔴 SELL", "SELL", "BÁN") and rp < 0.20:
                entry -= 25
                tradeable_now = False
                trade_reasons.append("SELL đang ở vùng thấp")
            if rec in ("🟢 BUY", "BUY", "MUA") and rp > 0.80:
                entry -= 25
                tradeable_now = False
                trade_reasons.append("BUY đang ở vùng cao")
            if 0.40 <= rp <= 0.60:
                entry -= 18
                tradeable_now = False
                trade_reasons.append("đang ở giữa biên độ")

        if ntz.get("active"):
            entry -= 15
            tradeable_now = False
            trade_reasons.append("đang ở vùng no-trade")

        if setup < 60:
            tradeable_now = False
            trade_reasons.append("setup chưa đủ mạnh")
        if entry < 60:
            tradeable_now = False

        entry = max(0, min(100, entry))

        dedup_reasons = list(dict.fromkeys(reasons))[:5]
        dedup_trade = list(dict.fromkeys(trade_reasons))[:4]

        if tradeable_now and setup >= 75 and entry >= 70:
            confidence_level = "HIGH"
        elif setup >= 60 and entry >= 50:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        risk_points = 0
        if ntz.get("active"):
            risk_points += 2
        if liq_evt.get("ok"):
            risk_points += 2
        if fake == "HIGH":
            risk_points += 2
        elif fake == "MEDIUM":
            risk_points += 1
        if entry < 45:
            risk_points += 2
        elif entry < 60:
            risk_points += 1
        if setup < 60:
            risk_points += 1

        if risk_points >= 5:
            risk_level = "HIGH"
        elif risk_points >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "setup_score": int(setup),
            "entry_score_now": int(entry),
            "tradeable_now": "YES" if tradeable_now else "NO",
            "final_score": int(setup),
            "tradeable": "YES" if tradeable_now else "NO",
            "score_reasons": dedup_reasons,
            "tradeable_reasons": dedup_trade,
            "confidence_level": confidence_level,
            "risk_level": risk_level,
        }
    except Exception as e:
        return {
            "setup_score": 0,
            "entry_score_now": 0,
            "tradeable_now": "NO",
            "final_score": 0,
            "tradeable": "NO",
            "score_reasons": [f"get_now_status fallback: {e}"],
            "tradeable_reasons": ["status fallback"],
            "confidence_level": "LOW",
            "risk_level": "HIGH",
        }
def build_view_engine_v1(sig: dict) -> str:
    sig = sig or {}
    meta = sig.get("meta") or {}

    symbol = sig.get("symbol") or "N/A"
    tf = sig.get("tf") or "M30"

    def _sf(x, nd=2, default="n/a"):
        try:
            if x is None:
                return default
            return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
        except Exception:
            return default

    def _zone(a, b):
        if a is None or b is None:
            return "n/a"
        return f"{_sf(a)} – {_sf(b)}"

    price = (
        sig.get("current_price")
        or sig.get("last_price")
        or sig.get("price")
        or sig.get("entry")
    )

    # ===== basic context =====
    h1_trend = str(meta.get("h1_trend") or meta.get("H1_TREND") or "UNKNOWN").upper()
    h4_trend = str(meta.get("h4_trend") or meta.get("H4_TREND") or "UNKNOWN").upper()

    if "BEAR" in h1_trend or "BEAR" in h4_trend or "SELL" in h1_trend or "SELL" in h4_trend:
        big_trend = "DOWN"
    elif "BULL" in h1_trend or "BULL" in h4_trend or "BUY" in h1_trend or "BUY" in h4_trend:
        big_trend = "UP"
    else:
        big_trend = "MIXED"

    mm = meta.get("market_mode_v1") or {}
    market_mode = str(mm.get("mode") or "UNKNOWN").upper()
    action_mode = str(mm.get("action_mode") or "WAIT").upper()
    side_context = str(mm.get("side") or "NONE").upper()
    rsi_div = meta.get("rsi_divergence_v1") or {}
    mom = meta.get("momentum_phase_v1") or {}
    vol = meta.get("volatility_regime_v1") or {}
    phase = (mom or {}).get("phase") or "UNKNOWN"
    vol_state = (vol or {}).get("state") or "UNKNOWN"
    sce = meta.get("signal_consistency_v1") or {}
    final_side = str(
        meta.get("final_side")
        or sce.get("final_side")
        or side_context
        or "NONE"
    ).upper()

    key = meta.get("key_levels") or {}
    lo = key.get("M15_RANGE_LOW")
    hi = key.get("M15_RANGE_HIGH")
    
    # ===== zones source =====
    playbook = meta.get("playbook_v2") or {}
    lo = key.get("M15_RANGE_LOW")
    hi = key.get("M15_RANGE_HIGH")
    zone_low = playbook.get("zone_low")
    zone_high = playbook.get("zone_high")
    range_low_txt = _sf(lo)
    range_high_txt = _sf(hi)
    range_txt = _zone(lo, hi)
    break_up_txt = _sf(hi)
    break_dn_txt = _sf(lo)
    decision_txt = _zone(zone_low, zone_high)

    try:
        range_pos = (float(price) - float(lo)) / max(float(hi) - float(lo), 1e-9)
        range_pct = max(0, min(100, range_pos * 100))
    except Exception:
        range_pct = None

    if range_pct is None:
        near_where = "UNKNOWN"
    elif range_pct <= 25:
        near_where = "support / vùng thấp"
    elif range_pct >= 75:
        near_where = "resistance / vùng cao"
    else:
        near_where = "mid-range / giữa biên"

    # ===== liquidity / trap =====
    flow = meta.get("flow_imbalance_v1") or {}
    liquidity = str(flow.get("liquidity") or "").upper()
    sweep = "YES" if "SWEEP" in liquidity or "SPRING" in liquidity else "NO"

    trap = meta.get("trap_warning_v1") or {}
    trap_level = str(trap.get("level") or trap.get("risk") or "LOW").upper()


    # ===== indicators fallback =====
    ema = sig.get("ema") or meta.get("ema") or meta.get("ema_filter_v1") or {}
    
    ema_trend = str(ema.get("trend") or "UNKNOWN").upper()
    ema_pos = str(
        ema.get("zone")
        or ema.get("position")
        or ema.get("price_position")
        or ""
    ).upper()
    
    # RSI lấy từ meta trước, nếu không có thì parse quality_lines
    rsi_val = meta.get("rsi15") or meta.get("RSI15")
    
    if rsi_val is None:
        for q in sig.get("quality_lines", []) or []:
            s = str(q)
            if "RSI(14)" in s:
                try:
                    rsi_val = float(s.split(":")[-1].strip())
                    break
                except Exception:
                    pass
    
    # Bollinger / Ichimoku nếu chưa có engine thì ghi rõ là chưa có data
    boll = meta.get("bollinger_context_v1") or {}
    ichi = meta.get("ichimoku_context_v1") or {}
    boll_state = (boll or {}).get("state")
    if boll_state is None:
        boll_txt = "NO DATA"
    elif boll_state == "SQUEEZE":
        boll_txt = "SQUEEZE (chuẩn bị biến động)"
    elif boll_state == "EXPANSION":
        boll_txt = "EXPANSION (đang chạy mạnh)"
    else:
        boll_txt = str(boll_state)
        
    ichi_state = ichi.get("state")
    cloud = ichi.get("cloud")
    if ichi_state == "IN_CLOUD":
        ichi_txt = f"TRONG MÂY ({cloud}) → tranh chấp"
    elif ichi_state == "ABOVE_CLOUD":
        ichi_txt = "TRÊN MÂY → bias tăng"
    elif ichi_state == "BELOW_CLOUD":
        ichi_txt = "DƯỚI MÂY → bias giảm"
    else:
        ichi_txt = "n/a"

    # ===== macro =====
    macro = meta.get("macro_v2") or {}
    macro_reasons = meta.get("macro_reason_v1") or []
    macro_exps = meta.get("macro_explain_tags_v1") or []

    macro_reason_lines = []
    for x in (macro_reasons + macro_exps):
        if x and x not in macro_reason_lines:
            macro_reason_lines.append(x)

    # ===== zones =====
    playbook = meta.get("playbook_v2") or {}
    zone_low = playbook.get("zone_low")
    zone_high = playbook.get("zone_high")

    res_near = key.get("M15_RES_NEAR_LOW"), key.get("M15_RES_NEAR_HIGH")
    sup_near = key.get("M15_SUP_NEAR_LOW"), key.get("M15_SUP_NEAR_HIGH")

    balance_zone = _zone(lo, hi)
    decision_zone = _zone(zone_low, zone_high)

    try:
        trap_pad = float(meta.get("atr15") or meta.get("ATR15") or 10)
        if final_side == "SELL" and zone_high is not None:
            trap_zone = _zone(float(zone_high), float(zone_high) + trap_pad)
        elif final_side == "BUY" and zone_low is not None:
            trap_zone = _zone(float(zone_low) - trap_pad, float(zone_low))
        else:
            trap_zone = "n/a"
    except Exception:
        trap_zone = "n/a"

    # ===== meaning =====
    if big_trend == "DOWN" and market_mode in ("RANGE_OR_CHOP", "PULLBACK_DOWN", "TREND_DAY_DOWN"):
        meaning = "Sideway/hồi trong xu hướng giảm → chưa gọi là đảo chiều, ưu tiên sell-the-rally."
    elif big_trend == "UP" and market_mode in ("RANGE_OR_CHOP", "PULLBACK_UP", "TREND_DAY_UP"):
        meaning = "Sideway/hồi trong xu hướng tăng → ưu tiên buy-the-dip khi có trigger."
    else:
        meaning = "Bối cảnh chưa đồng thuận → ưu tiên chờ market lộ mặt."
    # ===== VIEW BIAS VERDICT V1 =====
    boll_state = (boll or {}).get("state") or "NO_DATA"
    ichi_state = (ichi or {}).get("state") or "NO_DATA"
    phase = (mom or {}).get("phase") or "UNKNOWN"
    def _view_bias_verdict():
        sell_pts = 0
        buy_pts = 0
        reasons = []
    
        # Ichimoku
        if ichi_state == "BELOW_CLOUD":
            sell_pts += 2
            reasons.append("Ichimoku dưới mây → SELL pressure")
        elif ichi_state == "ABOVE_CLOUD":
            buy_pts += 2
            reasons.append("Ichimoku trên mây → BUY pressure")
    
        # Bollinger
        if boll_state == "EXPANSION":
            if range_pct is not None and range_pct <= 35:
                sell_pts += 1
                reasons.append("Bollinger expansion + giá vùng thấp → breakdown pressure")
            elif range_pct is not None and range_pct >= 65:
                buy_pts += 1
                reasons.append("Bollinger expansion + giá vùng cao → breakout pressure")
    
        # EMA
        if "BEAR" in ema_trend:
            sell_pts += 1
            reasons.append("EMA bearish")
        elif "BULL" in ema_trend:
            buy_pts += 1
            reasons.append("EMA bullish")
    
        # Momentum
        _phase = str(phase or "UNKNOWN").upper()
        if _phase == "EARLY":
            reasons.append("Momentum EARLY → sóng mới bắt đầu, chưa nên bắt ngược")
        elif phase == "LATE":
            reasons.append("Momentum LATE → dễ trap/đảo chiều, tránh đuổi")
    
        # chọn bias
        if sell_pts >= buy_pts + 2:
            bias = "SELL"
        elif buy_pts >= sell_pts + 2:
            bias = "BUY"
        else:
            bias = "NONE"
    
        # vị trí
        # vị trí
        if bias == "SELL":
            note = "SELL THE RALLY → ưu tiên bán ở vùng cao, không đuổi giá."
        
            if range_pct is not None and range_pct <= 25:
                action = "SELL_BIAS_WAIT_PULLBACK"
                quick = "SELL BIAS (WAIT PULLBACK)"
                action_hint = f"Giá đang sát đáy → KHÔNG SELL. Chờ hồi lên vùng {decision_txt} rồi SELL."
            else:
                action = "SELL_BIAS_WAIT_TRIGGER"
                quick = "SELL BIAS (WAIT TRIGGER)"
                action_hint = f"Canh SELL nếu có rejection/fail break quanh {decision_txt}."
        
        elif bias == "BUY":
            note = "BUY THE DIP → ưu tiên mua ở vùng thấp, không đuổi giá."
        
            if range_pct is not None and range_pct >= 75:
                action = "BUY_BIAS_WAIT_PULLBACK"
                quick = "BUY BIAS (WAIT PULLBACK)"
                action_hint = f"Giá đang vùng cao → KHÔNG BUY. Chờ hồi về vùng {decision_txt} rồi BUY."
            else:
                action = "BUY_BIAS_WAIT_TRIGGER"
                quick = "BUY BIAS (WAIT TRIGGER)"
                action_hint = f"Canh BUY nếu có reclaim/sweep low quanh {decision_txt}."
        
        else:
            note = "Không có edge rõ → đứng ngoài là chính."
            action = "NO_EDGE_WAIT"
            quick = "NO TRADE / WAIT EDGE"
            action_hint = f"Chờ giá về biên hoặc break rõ khỏi range {range_txt}."
    
        return {
            "bias": bias,
            "action": action,
            "quick": quick,
        
            "playbook": note,        # 🔥 dùng cho PLAYBOOK
            "action_hint": action_hint,  # 📌 dùng cho ACTION
        
            "sell_pts": sell_pts,
            "buy_pts": buy_pts,
            "reasons": reasons[:4],
        }
    
    view_verdict = _view_bias_verdict()
    # ===== playbook + scenarios =====
    if final_side == "SELL" or side_context == "SELL" or big_trend == "DOWN":
        quick_decision = "WAIT SELL ZONE / SELL THE RALLY"
        playbook_main = "ƯU TIÊN SELL vùng cao, không SELL giữa range."

        sc1 = {
            "name": "HỒI XONG ĐẠP",
            "prob": "HIGH",
            "condition": f"Giá hồi lên vùng quyết định {decision_zone} rồi yếu lại.",
            "trigger": "Fail break / râu trên / nến đỏ đóng lại dưới vùng.",
            "action": "SELL",
            "target": "Hỗ trợ gần / đáy range / TP theo plan.",
        }
        sc2 = {
            "name": "SIDEWAY GIẾT TIME",
            "prob": "MID",
            "condition": "Giá lắc quanh vùng cân bằng, không phá rõ hai đầu.",
            "trigger": "Không có rejection, không có follow-through.",
            "action": "NO TRADE / WAIT",
            "target": "Không ưu tiên TP, ưu tiên bảo toàn vốn.",
        }
        sc3 = {
            "name": "LỪA BREAKOUT",
            "prob": "MID",
            "condition": "Giá phá lên vùng trap rồi không giữ được.",
            "trigger": "Break lên nhưng đóng lại dưới / quét SL BUY xong đảo chiều.",
            "action": "CHỜ TRAP → SELL",
            "target": "Đạp ngược về vùng cân bằng/hỗ trợ.",
        }
        invalid = f"Nếu giá đóng và giữ trên vùng trap {trap_zone} → bias SELL yếu đi."

    elif final_side == "BUY" or side_context == "BUY" or big_trend == "UP":
        quick_decision = "WAIT BUY ZONE / BUY THE DIP"
        playbook_main = "ƯU TIÊN BUY vùng thấp, không BUY đuổi giữa range."

        sc1 = {
            "name": "HỒI XONG BẬT",
            "prob": "HIGH",
            "condition": f"Giá hồi về vùng quyết định {decision_zone} rồi giữ được.",
            "trigger": "Râu dưới / nến xanh xác nhận / reclaim vùng.",
            "action": "BUY",
            "target": "Kháng cự gần / đỉnh range / TP theo plan.",
        }
        sc2 = {
            "name": "SIDEWAY GIẾT TIME",
            "prob": "MID",
            "condition": "Giá lắc quanh vùng cân bằng, không có break rõ.",
            "trigger": "Không có follow-through.",
            "action": "NO TRADE / WAIT",
            "target": "Không ưu tiên TP, ưu tiên bảo toàn vốn.",
        }
        sc3 = {
            "name": "LỪA BREAKDOWN",
            "prob": "MID",
            "condition": "Giá phá xuống vùng trap rồi không giữ được.",
            "trigger": "Break xuống nhưng đóng lại trên / quét SL SELL xong bật.",
            "action": "CHỜ TRAP → BUY",
            "target": "Bật về vùng cân bằng/kháng cự.",
        }
        invalid = f"Nếu giá đóng và giữ dưới vùng trap {trap_zone} → bias BUY yếu đi."

    else:
        quick_decision = view_verdict["quick"]
        playbook_main = view_verdict["note"]
        
        sc1 = {
            "name": "BREAK THẬT",
            "prob": "MID",
            "condition": (
                f"Giá phá hẳn biên range M15: "
                f"break lên trên {break_up_txt} hoặc break xuống dưới {break_dn_txt}."
            ),
            "trigger": (
                f"M15 đóng ngoài range {range_txt}, sau đó retest giữ được "
                f"trên {break_up_txt} / dưới {break_dn_txt}."
            ),
            "action": "Đánh theo hướng break sau retest, không vào ngay cây phá đầu tiên.",
            "target": "Vùng tiếp theo theo cấu trúc / hỗ trợ-kháng cự kế tiếp.",
        }
        
        sc2 = {
            "name": "SIDEWAY GIẾT TIME",
            "prob": "MID" if view_verdict["bias"] in ("SELL", "BUY") else "HIGH",
            "condition": (
                f"Giá còn nằm trong range {range_txt}, đặc biệt quanh vùng giữa range "
                f"hoặc vùng quyết định {decision_txt}."
            ),
            "trigger": "Không có nến đóng thoát range, không có follow-through.",
            "action": "NO TRADE / WAIT",
            "target": "Không ưu tiên TP; ưu tiên chờ giá về biên trên hoặc biên dưới.",
        }
        
        sc3 = {
            "name": "FALSE BREAK / BẪY BIÊN",
            "prob": "MID",
            "condition": (
                f"Giá quét lên trên {break_up_txt} rồi đóng lại trong range, "
                f"hoặc quét xuống dưới {break_dn_txt} rồi đóng lại trong range."
            ),
            "trigger": "Fake break + đóng lại trong range + nến sau không đi tiếp.",
            "action": "Chờ phản ứng ngược lại sau trap.",
            "target": f"Quay về vùng cân bằng/range giữa {range_txt}.",
        }
        
        invalid = (
            f"Nếu M15/H1 đóng ngoài range {range_txt} và giữ được sau retest "
            f"→ bỏ kịch bản sideway."
        )

    # ===== output =====
    lines = []
    lines.append(f"📊 {symbol} VIEW TOÀN CẢNH | {tf}")
    lines.append("")
    lines.append("🧠 BẢN CHẤT THỊ TRƯỜNG")
    lines.append(f"- Xu hướng lớn (H1/H4): {big_trend}")
    lines.append(f"- Trạng thái hiện tại: {market_mode}")
    lines.append(f"- Cấu trúc: {sce.get('current_move') or 'n/a'}")
    lines.append("- Ý nghĩa:")
    lines.append(f"  → {meaning}")

    lines.append("")
    lines.append("📍 VỊ TRÍ GIÁ")
    lines.append(f"- Giá hiện tại: {_sf(price)}")
    lines.append(f"- Vị trí trong range M15: {_sf(range_pct, 1)}%")
    lines.append(f"- Gần vùng nào: {near_where}")

    lines.append("")
    lines.append("💧 THANH KHOẢN / TRAP")
    lines.append(f"- Sweep: {sweep}")
    lines.append(f"- Trap khả năng: {trap_level}")
    lines.append("- Nhận định:")
    if trap_level in ("HIGH", "MEDIUM"):
        lines.append("  → Dễ có trap/fake break, không vào lệnh khi chưa có xác nhận.")
    else:
        lines.append("  → Chưa thấy trap rõ, vẫn cần chờ trigger tại vùng.")

    lines.append("")
    lines.append("📊 INDICATOR CONTEXT")
    
    # RSI divergence
    div_state = rsi_div.get("state")
    if div_state == "BEARISH":
        div_txt = "PHÂN KỲ GIẢM → đà yếu"
    elif div_state == "BULLISH":
        div_txt = "PHÂN KỲ TĂNG → đà yếu phía dưới"
    else:
        div_txt = "KHÔNG CÓ PHÂN KỲ"
    
    # Momentum
    phase = (mom or {}).get("phase")

    if not phase or phase == "UNKNOWN":
        phase_txt = "UNKNOWN (thiếu dữ liệu)"
    if phase == "EARLY":
        phase_txt = "EARLY (mới bắt đầu)"
    elif phase == "MID":
        phase_txt = "MID (đang chạy)"
    elif phase == "LATE":
        phase_txt = "LATE (dễ đảo / xả)"
    else:
        phase_txt = "UNKNOWN"
    
    # Volatility
    vol_state = vol.get("state")
    if vol_state == "QUIET":
        vol_txt = "QUIET (tích lũy)"
    elif vol_state == "EXPANSION":
        vol_txt = "EXPANSION (đang bung mạnh)"
    else:
        vol_txt = "NORMAL"
    lines.append(f"- EMA: {ema_trend} {ema_pos}".strip())
    lines.append(f"- Bollinger: {boll_txt}")
    lines.append(f"- Ichimoku: {ichi_txt}")
    lines.append(f"- RSI divergence: {div_txt}")
    lines.append(f"- Momentum phase: {phase_txt}")
    lines.append(f"- Volatility: {vol_txt}")
    lines.append("")
    lines.append("🧠 VIEW VERDICT")
    lines.append(f"- Bias đọc được: {view_verdict['bias']}")
    lines.append(f"- Buy/Sell points: BUY={view_verdict['buy_pts']} | SELL={view_verdict['sell_pts']}")
    lines.append(f"- Kết luận: {view_verdict['quick']}")
    for r in view_verdict["reasons"]:
        lines.append(f"- {r}")

    lines.append("")
    lines.append("📍 VÙNG QUAN TRỌNG")
    lines.append(f"- Vùng cân bằng: {balance_zone}")
    lines.append(f"- Vùng quyết định: {decision_zone}")
    lines.append(f"- Vùng trap: {trap_zone}")

    def _scenario_block(icon, title, sc):
        lines.append("")
        lines.append(f"{icon} {title} — {sc['name']}")
        lines.append(f"- Xác suất: {sc['prob']}")
        lines.append("- Điều kiện:")
        lines.append(f"  {sc['condition']}")
        lines.append("- Trigger:")
        lines.append(f"  {sc['trigger']}")
        lines.append("- Hành động:")
        lines.append(f"  → {sc['action']}")
        lines.append("- Target:")
        lines.append(f"  → {sc['target']}")

    lines.append("")
    lines.append("🎯 3 KỊCH BẢN CHÍNH")
    _scenario_block("🔴", "KỊCH BẢN 1", sc1)
    _scenario_block("🟡", "KỊCH BẢN 2", sc2)
    _scenario_block("🔥", "KỊCH BẢN 3", sc3)

    lines.append("")
    lines.append("⚠️ ĐIỀU KIỆN INVALID")
    lines.append(f"- {invalid}")

    lines.append("")
    lines.append("🧠 QUYẾT ĐỊNH NHANH")
    lines.append(f"→ {quick_decision}")
    lines.append("")
    lines.append("🔥 PLAYBOOK CHÍNH")
    lines.append(f"→ {view_verdict.get('playbook')}")
    lines.append("")
    lines.append("📌 Gợi ý hành động:")
    lines.append(f"- {view_verdict.get('action_hint')}")

    lines.append("")
    lines.append("🚫 Không làm:")
    lines.append("- Không FOMO.")
    lines.append("- Không all-in.")
    lines.append("- Không vào giữa range khi chưa có trigger.")
    lines.append("- Không đánh ngược macro mạnh nếu chart chưa xác nhận.")

    return "\n".join(lines)        
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
    ema_pack = meta.get("ema") if isinstance(meta.get("ema"), dict) else {}
    
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
            elif 0.40 <= rp <= 0.60:
                score -= 10
                reasons.append("đang ở giữa biên độ")
            elif 0.20 < rp < 0.40:
                reasons.append("đang ở nửa dưới range")
            elif 0.60 < rp < 0.80:
                reasons.append("đang ở nửa trên range")
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
            if 0.40 <= rp <= 0.60:
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
        if str(tradeable_label).upper() == "CONDITIONAL":
            return "Có context tốt nhưng entry cần trigger/confirmation; chỉ theo dõi kèo có điều kiện, không vào bừa."
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
            if 0.4 <= rp <= 0.6:
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
            if rp <= 0.20 or rp >= 0.80:
                score += 2
                reasons.append("đang ở vùng biên có lợi")
            elif 0.40 <= rp <= 0.60:
                score -= 2
                reasons.append("đang ở giữa biên độ")
            elif 0.20 < rp < 0.40:
                reasons.append("đang ở nửa dưới range")
            elif 0.60 < rp < 0.80:
                reasons.append("đang ở nửa trên range")
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
            return "đang sát vùng cao / gần kháng cự"
        if rp <= 0.20:
            return "đang sát vùng thấp / gần hỗ trợ"
        if 0.40 <= rp <= 0.60:
            return "đang ở giữa biên độ"
        if rp > 0.60:
            return "đang ở nửa trên range"
        return "đang ở nửa dưới range"

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
            if 0.40 <= rp <= 0.60 and "mid" not in tags:
                tags.append("mid")
            elif 0.60 < rp < 0.80 and "upper" not in tags:
                tags.append("upper")
            elif 0.20 < rp < 0.40 and "lower" not in tags:
                tags.append("lower")
        except Exception:
            pass
        mapping = {
            "chop": "thị trường đang nhiễu",
            "mid": "đang ở giữa biên độ",
            "upper": "đang ở nửa trên range",
            "lower": "đang ở nửa dưới range",
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

    
    # =========================
    # ===== FINAL SCORE FALLBACK =====
    try:
        final_score = float(
            sig.get("final_score")
            if sig.get("final_score") is not None
            else sig.get("score")
        )
    except Exception:
        final_score = 0.0

    try:
        tradeable_raw = sig.get("tradeable")
        if tradeable_raw is None:
            tradeable_raw = (meta.get("master_engine_v1") or {}).get("tradeable_final")
        tradeable_label = "YES" if bool(tradeable_raw) else "NO"
    except Exception:
        tradeable_label = "NO"

    try:
        grade = _score_to_grade_v2(final_score)
    except Exception:
        if final_score >= 80:
            grade = "A"
        elif final_score >= 65:
            grade = "B"
        elif final_score >= 50:
            grade = "C"
        else:
            grade = "D"

    final_note = ""
    try:
        summary_line = _market_summary_line(
            final_score,
            tradeable_label,
            session_v4 if 'session_v4' in locals() else {},
            htf_pressure_v4 if 'htf_pressure_v4' in locals() else {},
        )
        final_note = summary_line or ""
    except Exception:
        final_note = ""

    score_reasons = score_reasons if 'score_reasons' in locals() else []
    tradeable_reasons = tradeable_reasons if 'tradeable_reasons' in locals() else []
    
    # OUTPUT V3 - 3 BLOCK
    # Replace from: head = f"{symbol} NOW ...
    # ====== ADD HELPER ======
    lines: List[str] = []
    info_lines: List[str] = []
    decision_lines: List[str] = []
    conclusion_lines: List[str] = []
    
    def push_info(s=""):
        add(info_lines, s)
    
    def push_decision(s=""):
        add(decision_lines, s)
    
    def push_conclusion(s=""):
        add(conclusion_lines, s)
    
    def _pf_zone_text(z):
        try:
            if not z:
                return "n/a"
            lo = float(z[0])
            hi = float(z[1])
            return f"{_fmt(lo)} – {_fmt(hi)}"
        except Exception:
            return "n/a"
    
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
    
    # ===== Kết luận nhanh =====
    reason = ""
    if range_pos is not None:
        try:
            rp = float(range_pos)
            if rp >= 0.80:
                reason = "đang sát vùng cao / gần kháng cự, không nên BUY đuổi"
            elif rp <= 0.20:
                reason = "đang sát vùng thấp / gần hỗ trợ, không nên SELL đuổi"
            elif 0.40 <= rp <= 0.60:
                reason = "đang ở giữa biên độ, dễ nhiễu"
            elif rp > 0.60:
                reason = "đang ở nửa trên của range, rủi ro BUY đuổi cao"
            else:
                reason = "đang ở nửa dưới của range, chưa phải vùng SELL đẹp"
        except Exception:
            reason = ""
    
    verdict_full = verdict_quick
    if reason and reason not in verdict_quick:
        verdict_full += f"; {reason}"
    
    trend_ctx = _trend_context_for_now(str(htf_pressure_v4.get("state") or ""), struct, rec)
    state_text_final = _state_text_v7(symbol, trend_ctx, meta.get("market_state_v2"))
    
    # =========================
    # BLOCK 1: THÔNG TIN & INDICATOR
    # =========================
    
    pb1 = meta.get("pullback_engine_v1") or {}
    if pb1:
        push_info(f"📌 Pullback: {pb1.get('label', 'n/a')} | {pb1.get('message', 'n/a')}")
        push_info(f"📉 Pullback engine: {pb1.get('label', 'n/a')} | hồi ~{pb1.get('pullback_pct_text', 'n/a')}")
        push_info(f"- Đánh giá: {pb1.get('message', 'n/a')}")
        push_info(f"- Reversal risk: {pb1.get('reversal_risk', 'N/A')}")
        push_info(f"- Enough for entry: {'YES' if pb1.get('enough_for_entry') else 'NO'}")
        push_info(f"- Khung đo hồi: {_fmt(pb1.get('anchor_low'))} – {_fmt(pb1.get('anchor_high'))}")
        rs = pb1.get("reason") or []
        if rs:
            push_info(f"- Lý do: {', '.join([str(x) for x in rs[:4]])}")
        push_info("")
    
    push_info(f"💵 Giá hiện tại: {price_now}")
    push_info("📍 Vị trí giá:")
    push_info(f"- Giá hiện tại: {price_now}")
    push_info(f"- Biên độ M15: {_fmt(range_lo)} – {_fmt(range_hi)}")
    if range_pos is not None:
        try:
            rp_pct = float(range_pos) * 100.0 if float(range_pos) <= 1.0 else float(range_pos)
            pos_note = "giữa biên độ, dễ nhiễu"
            if rp_pct >= 80:
                pos_note = "sát vùng cao, không nên BUY đuổi"
            elif rp_pct <= 20:
                pos_note = "sát vùng thấp, không nên SELL đuổi"
            push_info(f"- Vị trí trong biên độ: ~{rp_pct:.0f}% → {pos_note}")
        except Exception:
            push_info("- Vị trí trong biên độ: n/a")
    else:
        push_info("- Vị trí trong biên độ: n/a")
    
    h4_struct = struct.get("H4") if struct.get("H4") is not None else struct.get("h4")
    h1_struct = struct.get("H1") if struct.get("H1") is not None else struct.get("h1")
    m15_struct = struct.get("M15") if struct.get("M15") is not None else struct.get("m15")
    push_info("✅ Xác nhận:")
    push_info(f"- Cấu trúc lớn: H4 {h4_struct if h4_struct is not None else 'n/a'} | H1 {h1_struct if h1_struct is not None else 'n/a'}")
    push_info(f"- Cấu trúc ngắn hạn: M15 {m15_struct if m15_struct is not None else 'n/a'}")

    flow1 = meta.get("flow_engine_v1") or {}
    push_info("🧠 FLOW / IMBALANCE:")
    push_info(f"- State: {flow1.get('state', 'NEUTRAL')}")
    push_info(f"- Displacement: {flow1.get('displacement', 'NONE')}")
    push_info(f"- Liquidity: {flow1.get('liquidity_state', 'Chưa thấy sweep/spring rõ')}")
    push_info(f"- Liquidity bias: {flow1.get('liquidity_bias', 'NEUTRAL')}")
    push_info(
        f"- Gap: {flow1.get('gap_text', 'Chưa có dấu hiệu GAP / mở cửa bất thường rõ')}"
    )
    push_info(
        f"- Imbalance: {flow1.get('fvg_text', 'chưa có vùng rõ')}"
    )
    push_info(f"- Ý nghĩa: {flow1.get('narrative', 'Flow chưa rõ')}")
    push_info(f"- Hành động gợi ý: {flow1.get('action_hint', 'WAIT')}")
    for s in (flow1.get("reasons") or [])[:2]:
        push_info(f"- {s}")

    
    volq = meta.get("volq") or meta.get("vol_quality") or {}
    candle_pat = meta.get("candle") or {}
    div = meta.get("div") or meta.get("divergence") or {}
    rsi_show = (
        meta.get("rsi14")
        if meta.get("rsi14") is not None
        else (rsi15 if 'rsi15' in locals() else meta.get("rsi"))
    )
    push_info("🧪 Chi tiết bổ sung:")
    vol_state = volq.get("state") or "N/A"
    vol_ratio = volq.get("ratio")
    if vol_ratio is None:
        vol_ratio_txt = "n/a"
    else:
        try:
            vol_ratio_txt = f"{float(vol_ratio):.2f}"
        except Exception:
            vol_ratio_txt = str(vol_ratio)
    push_info(f"- Volume: {vol_state} (x{vol_ratio_txt} vs SMA20)")
    candle_txt = candle_pat.get("txt")
    if candle_txt and str(candle_txt).strip().upper() not in ("NONE", "N/A", "NULL"):
        push_info(f"- Candle: {candle_txt}")
    div_txt = div.get("txt")
    if div_txt and str(div_txt).strip().upper() not in ("NONE", "N/A", "NULL"):
        push_info(f"- RSI divergence: {div_txt}")
    push_info(f"- RSI(14) M15: {rsi_show if rsi_show is not None else 'n/a'}")
    push_info(f"- ATR(14) M15: ~{_fmt(meta.get('atr15') if meta.get('atr15') is not None else (atr15 if 'atr15' in locals() else None))}")
    push_info("- RR ~ 1:2 (mục tiêu)")
    push_info("")
    
    rsi_ctx1 = meta.get("rsi_context_v1") or {}
    push_info("📈 RSI context:")
    push_info(f"- {rsi_ctx1.get('message', 'RSI trung tính')}")
    
    fib1 = meta.get("fib_confluence_v1") or {}
    push_info("📐 Fib confluence:")
    push_info(f"- {'YES' if fib1.get('ok') else 'NO'}")
    
    ema_pack = meta.get("ema") or {}
    if ema_pack:
        push_info("📉 EMA FILTER:")
        push_info(f"- EMA34: {_fmt(ema_pack.get('ema34'))}")
        push_info(f"- EMA89: {_fmt(ema_pack.get('ema89'))}")
        push_info(f"- EMA200: {_fmt(ema_pack.get('ema200'))}")
        push_info(f"- Trend: {ema_pack.get('trend', 'N/A')}")
        push_info(f"- Alignment: {ema_pack.get('alignment', 'NO')}")
        push_info(f"- Vị trí giá vs EMA: {ema_pack.get('zone', 'N/A')}")
    
    pump1 = meta.get("pump_dump_v3") or meta.get("pump_dump_v2") or meta.get("pump_dump_v1") or {}
    push_info("🚀 DỰ ĐOÁN PUMP/DUMP V3:")
    push_info(f"- Stage: {pump1.get('stage', 'NONE')}")
    push_info(f"- Bias bung: {pump1.get('bias', 'NEUTRAL')}")
    push_info(f"- Compression: {pump1.get('compression', 'LOW')}")
    push_info(f"- Xác suất: {pump1.get('probability', 'LOW')}")
    push_info(f"- Thời điểm: {pump1.get('timing', 'CHƯA RÕ')}")
    
    reveal1 = meta.get("reveal_engine_v1") or {}
    if reveal1:
        push_info("🧠 REVEAL ENGINE:")
        push_info(f"- State: {reveal1.get('state', 'NOT REVEALED')}")
        push_info(f"- Direction: {reveal1.get('direction', 'NONE')}")
        push_info(f"- Quality: {reveal1.get('quality', 'LOW')}")
        if reveal1.get("message"):
            push_info(f"- {reveal1.get('message')}")
    
    # =========================
    # BLOCK 2: PHÂN TÍCH & QUYẾT ĐỊNH
    # =========================
    
    # FIX conflict BUY/SELL: ưu tiên bám final_side + bias lớn, không nói 2 giọng
    sce1 = meta.get("signal_consistency_v1") or {}
    final_side = str(sce1.get("final_side") or "NONE").upper()
    market_state = str(meta.get("market_state_v2") or "")
    htf_bias = str((meta.get("htf_pressure_v4") or {}).get("state") or "").upper()
    
    state_line = state_text_final or "Market chưa lộ mặt → ưu tiên chờ"
    if final_side == "BUY":
        state_line = f"{symbol} đang hồi trong xu hướng tăng; ưu tiên buy-the-dip, tránh mua đuổi."
    elif final_side == "SELL":
        state_line = f"{symbol} đang hồi trong xu hướng giảm; ưu tiên sell-the-rally, tránh bán đuổi."
    
    push_decision("🌡 Trạng thái:")
    push_decision(f"- {state_line}")
    
    flow_text = "Dòng tiền trung tính"
    if final_side == "BUY":
        flow_text = "smart money buying dip"
    elif final_side == "SELL":
        flow_text = "smart money selling rally"
    push_decision("💰 Dòng tiền:")
    push_decision(f"- {flow_text}")
    
    ctx_verdict = "CHƯA CÓ CẢNH"
    if final_side == "BUY":
        ctx_verdict = "ĐÚNG CẢNH BUY"
    elif final_side == "SELL":
        ctx_verdict = "ĐÚNG CẢNH SELL"
    push_decision("🧠 Context verdict:")
    push_decision(f"- {ctx_verdict}")
    
    ll = meta.get("manual_likelihood_v1") or {}
    buy_like = (
        ll.get("buy_likelihood")
        if ll.get("buy_likelihood") is not None
        else ll.get("buy_score")
    )
    if buy_like is None:
        buy_like = ll.get("buy", 0)
    
    sell_like = (
        ll.get("sell_likelihood")
        if ll.get("sell_likelihood") is not None
        else ll.get("sell_score")
    )
    if sell_like is None:
        sell_like = ll.get("sell", 0)
    
    trap_like = (
        ll.get("trap_risk")
        if ll.get("trap_risk") is not None
        else ll.get("trap_score")
    )
    if trap_like is None:
        trap_like = ll.get("trap", 0)
    push_decision("📊 Manual likelihood:")
    push_decision(f"- BUY={buy_like}/100")
    push_decision(f"- SELL={sell_like}/100")
    push_decision(f"- Trap={trap_like}/100")
    trap_lines = []
    if pb1 and not pb1.get("enough_for_entry"):
        trap_lines.append("chưa có close confirmation rõ")
    if range_pos is not None:
        try:
            rp = float(range_pos)
            if final_side == "BUY" and rp >= 0.8:
                trap_lines.append("BUY vùng cao → dễ mua đuổi")
            elif final_side == "SELL" and rp <= 0.2:
                trap_lines.append("SELL vùng thấp → dễ bán đuổi")
        except Exception:
            pass
    if trap_lines:
        push_decision("⚠️ Trap:")
        for s in trap_lines[:3]:
            push_decision(f"- {s}")
    
    push_decision("")
    push_decision("===== PRO DESK =====")
    push_decision("🧠 Market state:")
    push_decision(f"- {market_state or 'N/A'}")
    
    liq1 = meta.get("liquidity_map_v1") or {}
    playbook_v4 = meta.get("playbook_v2") or {}
    push_decision("🧭 Bias:")
    push_decision(f"- HTF: {playbook_v4.get('htf_bias', final_side or 'WAIT')}")
    push_decision(f"- MTF: {playbook_v4.get('plan', 'WAIT')}")
    
    conflict_lines = []
    if pb1 and not pb1.get("enough_for_entry"):
        conflict_lines.append("pullback chưa đủ đẹp")
    if liq1 and not liq1.get("done"):
        conflict_lines.append("thanh khoản chưa hoàn tất")
    if fib1 and not fib1.get("ok"):
        conflict_lines.append("chưa có fib confluence rõ")
    
    conflict_label = "LOW CONFLICT"
    if len(conflict_lines) >= 3:
        conflict_label = "HIGH CONFLICT"
    elif len(conflict_lines) >= 1:
        conflict_label = "MEDIUM CONFLICT"
    
    push_decision("⚖️ Conflict:")
    push_decision(f"- {conflict_label}")
    for s in conflict_lines[:3]:
        push_decision(f"- {s}")
    
    me1 = meta.get("master_engine_v1") or {}
    if me1:
        push_decision("")
        push_decision("🧠 MASTER ENGINE:")
        push_decision(f"- State: {me1.get('state', 'WAIT')}")
        push_decision(f"- Best side: {me1.get('best_side', 'NONE')}")
        
        tf = me1.get("tradeable_final")
        if isinstance(tf, str):
            tf_txt = tf.upper()
        else:
            tf_txt = "YES" if tf else "NO"
        
        push_decision(f"- Tradeable final: {tf_txt}")
        push_decision(f"- Confidence: {me1.get('confidence', 'LOW')}")
    
    if sce1:
        push_decision("🧠 SIGNAL CONSISTENCY:")
        push_decision(f"- Final side: {sce1.get('final_side', 'NONE')}")
        push_decision(f"- Current move: {sce1.get('current_move', 'CHOP')}")
        push_decision(f"- Action mode: {sce1.get('action_mode', 'NO_TRADE')}")
        push_decision(f"- Narrative: {sce1.get('narrative', '')}")
    
    # =========================
    # BLOCK 3: KẾT LUẬN & HÀNH ĐỘNG
    # =========================
    
    push_conclusion(f"🧭 Hướng ưu tiên: {rec}")
    push_conclusion(f"📌 Kết luận nhanh: {verdict_full}")
    if phase:
        push_conclusion(f"🪜 Giai đoạn: {phase.get('phase', 'n/a')} | {phase.get('meaning', phase.get('label', 'n/a'))}")
    
    # Gộp action: chỉ giữ một action chính, không lặp
    # ===== PATH FORECAST BUILD (SAFE) =====
    meta = meta or {}
    sig = sig or {}
    
    struct0 = meta.get("structure") or {}
    kl0 = meta.get("key_levels") or {}
    playbook0 = meta.get("playbook_v2") or {}
    liq0 = meta.get("liquidity_map_v1") or {}
    ema0 = meta.get("ema") or {}
    sf0 = meta.get("fvg_range_plugin_v1") or {}
    m15c0 = []
    try:
        m15_raw0 = meta.get("_m15_raw") or []
        if m15_raw0:
            m15c0 = _safe_candles(m15_raw0)
    except Exception:
        m15c0 = []
    cp0 = None
    try:
        cp0 = (
            _safe_float(sig.get("current_price"))
            or _safe_float(sig.get("last_price"))
            or _safe_float(sig.get("price"))
            or _safe_float(sig.get("entry"))
        )
    except Exception:
        cp0 = None
    
    if cp0 is None:
        try:
            if m15c0:
                cp0 = _safe_float(_c_val(m15c0[-1], "close", None))
        except Exception:
            cp0 = None
    pf1 = meta.get("path_forecast_v1") or {}
    if not pf1 or (pf1.get("res_near") is None and pf1.get("sup_near") is None):
        try:
            pf1 = _path_forecast_v1(
                current_price=cp0,
                atr15=(atr15 if 'atr15' in locals() else meta.get("atr15")),
                h1_trend=struct0.get("H1"),
                h4_trend=struct0.get("H4"),
                m15_struct_tag=struct0.get("M15"),
                range_low=kl0.get("M15_RANGE_LOW"),
                range_high=kl0.get("M15_RANGE_HIGH"),
                playbook_v2=playbook0,
                liquidity_map_v1=liq0,
                ema_pack=ema0,
                smart_filter_v1=sf0,
                m15c=m15c0,
            ) or {}
        except Exception as e:
            print(f"[PATH_FORECAST_ERROR] {e}")
            pf1 = {
                "down_bias": "KHÔNG RÕ",
                "up_bias": "KHÔNG RÕ",
                "sideway_bars": "n/a",
                "res_near": None,
                "res_far": None,
                "sup_near": None,
                "sup_far": None,
                "priority_action": "ƯU TIÊN ĐỨNG NGOÀI",
                "action_note": "",
                "reason": [],
            }
        meta["path_forecast_v1"] = pf1
        
    push_conclusion("⚙️ Hành động:")
    de1 = meta.get("decision_engine_v1") or {}
    za1 = meta.get("zone_action_v1") or {}
    pf_action = (pf1.get("priority_action") if 'pf1' in locals() and pf1 else None) or ""

    # ===== ưu tiên Zone + Action Engine =====
    if ntz.get("active"):
        action_main = "No-trade zone"

    elif za1.get("ok"):
        zact = za1.get("action_state")
        if zact == "WAIT":
            action_main = za1.get("message") or "Chờ vùng"
        elif zact == "WATCH":
            action_main = za1.get("message") or "Đang trong vùng → chờ phản ứng"
        elif zact == "TRIGGER":
            action_main = za1.get("message") or "Đã rời vùng theo hướng có lợi → chờ follow-through"
        elif zact == "CANCEL":
            action_main = za1.get("message") or "Vùng bị vô hiệu"
        else:
            action_main = pf_action or "Chưa nên mở lệnh mới"

    elif pf_action:
        action_main = pf_action

    elif final_side == "BUY":
        action_main = "Chờ trigger BUY rõ"

    elif final_side == "SELL":
        action_main = "Chờ trigger SELL rõ"

    else:
        action_main = "Chưa nên mở lệnh mới"

    push_conclusion(f"- {action_main}")

    # ===== dòng giải thích phụ =====
    if za1.get("ok"):
        if za1.get("trigger"):
            push_conclusion(f"- {za1.get('trigger')}")
        if za1.get("invalid"):
            push_conclusion(f"- Vô hiệu: {za1.get('invalid')}")

    elif pf1 and pf1.get("action_note"):
        push_conclusion(f"- {pf1.get('action_note')}")

    else:
        push_conclusion(f"- {state_line}")

    push_conclusion("")
        
    push_conclusion("🧯 Điểm sai kịch bản:")
    scenario = meta.get("scenario_v1") or {}
    push_conclusion(f"- {scenario.get('invalid_if', 'Mất cấu trúc hiện tại')}")
    push_conclusion("")
    
    # kịch bản phụ
    push_conclusion("🪄 KỊCH BẢN PHỤ:")
    if scenario.get("alt_case"):
        push_conclusion(f"- {scenario.get('alt_case')}")
    elif scenario.get("alt_plan"):
        push_conclusion(f"- {scenario.get('alt_plan')}")
    else:
        push_conclusion("- Chưa có alt case rõ")
        
    push_conclusion("━━━━━━━━━━━")
    push_conclusion("🎯 TRIGGER QUAN TRỌNG")
    push_conclusion("━━━━━━━━━━━")
    tg3 = (sig.get("meta") or {}).get("trigger_engine_v3") or {}
    if tg3.get("trigger_line"):
        push_conclusion(f"- {tg3.get('trigger_line')}")
    if tg3.get("close_confirm_line"):
        push_conclusion(f"- {tg3.get('close_confirm_line')}")
    if tg3.get("invalidation_line"):
        push_conclusion(f"- {tg3.get('invalidation_line')}")
    if not (tg3.get("trigger_line") or tg3.get("close_confirm_line") or tg3.get("invalidation_line")):
        for s in (tg3.get("reason") or [])[:3]:
            push_conclusion(f"- {s}")
    push_conclusion("🎯 TRIGGER ENGINE V3:")
    push_conclusion(f"- State: {tg3.get('state', 'WAIT')}")
    push_conclusion(f"- Side: {tg3.get('entry_side', 'NONE')}")
    push_conclusion(f"- Quality: {tg3.get('quality', 'LOW')}")
    if tg3.get("trigger_line"):
        push_conclusion(f"- Trigger: {tg3.get('trigger_line')}")
    if tg3.get("close_confirm_line"):
        push_conclusion(f"- {tg3.get('close_confirm_line')}")
    if tg3.get("invalidation_line"):
        push_conclusion(f"- Invalidation: {tg3.get('invalidation_line')}")
    for s in (tg3.get("reason") or [])[:3]:
        push_conclusion(f"- {s}")  
        
    push_conclusion("")
    push_conclusion(f"🎯 Decision: {de1.get('decision', 'STAND ASIDE')}")
    push_conclusion("⏳ Wait for:")
    wf1 = meta.get("wait_for_v1") or {}
    za1 = meta.get("zone_action_v1") or {}
    
    if za1.get("ok"):
        push_conclusion(
            f"- Zone: {za1.get('zone_type')} | {za1.get('price_state')} | Action={za1.get('action_state')}"
        )
    
    wait_lines = wf1.get("lines") or []
    if wait_lines:
        for s in wait_lines[:3]:
            push_conclusion(f"- {s}")
    else:
        push_conclusion("- Chưa có vùng hành động rõ")

    # FLOW FILTER SUMMARY
    fvgp = meta.get("fvg_range_plugin_v1") or {}
    rf1 = fvgp.get("range_filter") or {}
    ema1 = fvgp.get("ema") or {}
    flow1 = meta.get("flow_engine_v1") or {}
    
    push_conclusion("🧩 FLOW FILTER:")
    pos = rf1.get("position")
    state = rf1.get("state", "UNKNOWN")
    tag = rf1.get("tag", "N/A")
    if pos is None:
        push_conclusion(f"- Range: {state} | N/A")
    else:
        try:
            push_conclusion(f"- Range: {state} | {float(pos):.1f}% | {tag}")
        except Exception:
            push_conclusion(f"- Range: {state} | {pos} | {tag}")
    
    push_conclusion(f"- EMA: {ema1.get('trend', 'N/A')} | Align={ema1.get('alignment', 'NO')} | {ema1.get('zone', 'N/A')}")
    push_conclusion(f"- Flow state: {flow1.get('state', 'NEUTRAL')}")
    push_conclusion(f"- Flow hint: {flow1.get('action_hint', 'WAIT')}")
    push_conclusion(f"- Narrative: {flow1.get('narrative', 'Flow chưa rõ')}")
    
    # ===== ELLIOTT PHASE =====
    elli1 = (meta.get("elliott_phase_v1") or {})
    if elli1:
        push_conclusion("")
        push_conclusion("🌊 ELLIOTT PHASE:")
        push_conclusion(f"- Main TF: {elli1.get('main_tf', 'H1/H4')}")
        push_conclusion(f"- Phase: {elli1.get('phase', 'UNCLEAR')}")
        push_conclusion(f"- Confidence: {elli1.get('confidence', 0)}%")
        push_conclusion(f"- Ý nghĩa: {elli1.get('meaning', 'Chưa rõ phase Elliott')}")
        push_conclusion(f"- Hành động: {elli1.get('action', 'Đứng ngoài, chờ rõ hơn')}")
        push_conclusion(f"- Vô hiệu: {elli1.get('invalid', 'n/a')}")
        
    # ===== ACTION: Post-break continuity =====
    pbc1 = ((sig.get("meta") or {}).get("post_break_continuity_v1") or None)
    if pbc1:
        push_conclusion("")
        push_conclusion("🔁 POST-BREAK CONTINUITY:")
        state = pbc1.get("state")
        side = pbc1.get("side")
        ref = pbc1.get("reference")
        action = pbc1.get("action")
        reasons = pbc1.get("reason") or []
    
        # ===== dòng chính =====
        main_line = f"- {state} | {side}"
        if ref is not None:
            main_line += f" @ {nf(ref)}"
        push_conclusion(main_line)
    
        # ===== diễn giải =====
        if state == "WAIT_BREAK":
            push_conclusion("- Chưa phá mốc → đứng ngoài, chờ break rõ")
    
        elif state == "AT_LEVEL":
            push_conclusion("- Đang test lại mốc → chờ phản ứng giữ/mất")
    
        elif state == "BREAK_HOLD":
            if side == "BUY":
                push_conclusion("- Đã break lên và giữ được → ưu tiên BUY pullback")
            elif side == "SELL":
                push_conclusion("- Đã break xuống và giữ được → ưu tiên SELL pullback")
    
        elif state == "BREAK_FAIL":
            push_conclusion("- Break thất bại → tránh follow, chờ kịch bản mới")
    
        # ===== action =====
        push_conclusion(f"- Hành động: {action}")
    
        # ===== reason ngắn =====
        for s in reasons[:2]:
            push_conclusion(f"- {s}")
    push_conclusion("")
    push_conclusion("🔮 PATH FORECAST:")
    push_conclusion(f"- Đi xuống: {pf1.get('down_bias', 'KHÔNG RÕ')}")
    push_conclusion(f"- Hồi lên: {pf1.get('up_bias', 'KHÔNG RÕ')}")
    push_conclusion(f"- Đi ngang: ~{pf1.get('sideway_bars', 'n/a')} nến M15")
    
    pf_reasons = [str(x) for x in (pf1.get("reason") or [])[:3] if x]
    # ===== FORECAST REASONING (PRO CHIẾN) =====
    reason_lines = []
    
    # 1. HTF trend
    try:
        if htf_bias == "BUY":
            reason_lines.append("- HTF giữ trend tăng")
        elif htf_bias == "SELL":
            reason_lines.append("- HTF giữ trend giảm")
    except:
        pass
    
    # 2. Position trong range
    try:
        rp_val = float(playbook.get("range_pos"))
        if rp_val >= 0.8:
            reason_lines.append("- Giá đang ở vùng cao (late phase)")
        elif rp_val <= 0.2:
            reason_lines.append("- Giá đang ở vùng thấp (early phase)")
        elif 0.3 <= rp_val <= 0.7:
            reason_lines.append("- Giá đang ở giữa biên độ (dễ nhiễu)")
    except:
        pass
    
    # 3. Liquidity (rất quan trọng)
    try:
        liq_done = (meta.get("liquidity") or {}).get("done")
        if not liq_done:
            reason_lines.append("- Chưa có liquidity sweep → chưa đủ điều kiện breakout")
    except:
        pass
    
    # 4. Compression (nếu có)
    try:
        pd = meta.get("pump_dump") or {}
        if pd.get("compression") == "LOW":
            reason_lines.append("- Không có nén → khó có breakout mạnh")
    except:
        pass
    
    # 5. Fallback
    if not reason_lines:
        reason_lines.append("- Market chưa có tín hiệu rõ")
    
    # 6. Push ra output
    push_conclusion("- Reasoning:")
    for r in reason_lines[:3]:   # giữ tối đa 3 dòng cho gọn
        push_conclusion(r)
    
    push_conclusion("📍 Vùng kháng cự M15:")
    push_conclusion(f"- Gần: {_pf_zone_text(pf1.get('res_near'))}")
    push_conclusion(f"- Xa: {_pf_zone_text(pf1.get('res_far'))}")
    
    push_conclusion("📍 Vùng hỗ trợ M15:")
    push_conclusion(f"- Gần: {_pf_zone_text(pf1.get('sup_near'))}")
    push_conclusion(f"- Xa: {_pf_zone_text(pf1.get('sup_far'))}")
    
    push_conclusion(f"🎯 Hành động: {pf1.get('priority_action', 'ƯU TIÊN ĐỨNG NGOÀI')}")
    if pf1.get("action_note"):
        push_conclusion(f"- {pf1.get('action_note')}")
    # ===== MARKET MODE V1 OUTPUT - BLOCK 3 =====
    try:
        mm1 = meta.get("market_mode_v1") or {}
        if mm1:
            mode = str(mm1.get("mode") or "UNKNOWN").upper()
            side = str(mm1.get("side") or "NONE").upper()
            action_mode = str(mm1.get("action_mode") or "WAIT_CONTEXT").upper()
            conf = mm1.get("confidence", 0)

            push_conclusion("")
            push_conclusion("🔥 MARKET MODE / NGỮ CẢNH HÔM NAY:")
            push_conclusion(f"- Mode: {mode}")
            push_conclusion(f"- Side context: {side}")
            push_conclusion(f"- Action mode: {action_mode}")
            push_conclusion(f"- Confidence: {conf}%")

            summary = str(mm1.get("summary") or "").strip()
            if summary:
                push_conclusion(f"- Ý nghĩa: {summary}")

            reasons = mm1.get("reasons") or []
            if reasons:
                push_conclusion("- Lý do đọc ngữ cảnh:")
                for r in reasons[:4]:
                    push_conclusion(f"  • {r}")

            warnings = mm1.get("warnings") or []
            if warnings:
                push_conclusion("⚠️ Cảnh báo:")
                for w in warnings[:3]:
                    push_conclusion(f"- {w}")

            playbook_lines = mm1.get("playbook_lines") or []
            if playbook_lines:
                push_conclusion("🎯 Playbook theo market mode:")
                for line in playbook_lines[:4]:
                    push_conclusion(f"- {line}")

            if mode in ("TREND_DAY_DOWN", "TREND_DAY_UP"):
                push_conclusion("📌 Kết luận ngữ cảnh:")
                if side == "SELL":
                    push_conclusion("- Đây là ngày ưu tiên SELL context, nhưng chỉ SELL khi có nhịp nghỉ / retest / break giữ dưới.")
                    push_conclusion("- Không gọi là mất kèo chỉ vì pullback nông; trend mạnh thường không cho hồi đẹp.")
                elif side == "BUY":
                    push_conclusion("- Đây là ngày ưu tiên BUY context, nhưng chỉ BUY khi có nhịp nghỉ / retest / break giữ trên.")
                    push_conclusion("- Không gọi là mất kèo chỉ vì pullback nông; trend mạnh thường không cho hồi đẹp.")

    except Exception as e:
        push_conclusion("")
        push_conclusion(f"🔥 MARKET MODE: lỗi render ({e})")


    # ===== MACRO ENGINE V2 =====
    macro = meta.get("macro_v2") or {}
    news_items = meta.get("news_items") or []
    
    push_conclusion("")
    push_conclusion("🌍 MACRO ENGINE V2:")
    push_conclusion(f"- Mode: {macro.get('macro_mode', 'NEUTRAL')}")
    push_conclusion(f"- USD strength: {macro.get('usd_strength', 0)}")
    push_conclusion(f"- Risk mode: {macro.get('risk_mode', 'NEUTRAL')}")
    push_conclusion(f"- Gold bias: {macro.get('gold_bias', 'NEUTRAL')}")
    push_conclusion(f"- BTC bias: {macro.get('btc_bias', 'NEUTRAL')}")
    push_conclusion(f"- Confidence: {macro.get('confidence', 0)}%")
    push_conclusion(f"- News items: {len(news_items)}")
    
    drivers = macro.get("drivers") or []
    if drivers:
        push_conclusion(f"- Drivers: {', '.join(str(x) for x in drivers[:3])}")
    # ===== MACRO EXPLAIN (FIX HIỂN THỊ) =====
    exps = meta.get("macro_explain_tags_v1") or []
    reasons = meta.get("macro_reason_v1") or []
    
    shown = []
    for x in (reasons + exps):
        if x and x not in shown:
            shown.append(x)
    
    if shown:
        push_conclusion("")
        push_conclusion("🧠 LÝ DO VĨ MÔ:")
        for x in shown[:4]:
            push_conclusion(f"- {x}")
    
    # ===== MACRO CONFLICT FILTER =====
    mcf = meta.get("macro_conflict_filter_v1") or {}
    if mcf:
        push_conclusion("")
        push_conclusion("⚠️ MACRO CONFLICT FILTER:")
        push_conclusion(f"- Conflict: {'YES' if mcf.get('conflict') else 'NO'}")
        push_conclusion(f"- Macro bias: {mcf.get('macro_bias', 'NEUTRAL')}")
        push_conclusion(f"- Severity: {mcf.get('severity', 'NONE')}")
        push_conclusion(f"- Action: {mcf.get('action', 'ALLOW')}")
        push_conclusion(f"- Score adjust: {mcf.get('score_adjust', 0)}")
    
        if mcf.get("block_add"):
            push_conclusion("- Add position: BLOCK")
    
        for r in (mcf.get("reason") or [])[:3]:
            push_conclusion(f"- {r}")
            
    # ===== PRACTICAL SUMMARY - BLOCK 3 =====
    try:
        mm1 = meta.get("market_mode_v1") or {}
        pf1 = meta.get("path_forecast_v1") or {}
        playbook = meta.get("playbook_v2") or {}
        mode = str(mm1.get("mode") or "").upper()
        side = str(mm1.get("side") or "NONE").upper()
    
        zlo = playbook.get("zone_low")
        zhi = playbook.get("zone_high")
    
        def _fmt_zone(a, b):
            try:
                return f"{float(a):.3f} – {float(b):.3f}"
            except Exception:
                return "vùng kháng cự/hỗ trợ gần"
    
        if mode in ("TREND_DAY_DOWN", "TREND_DAY_UP"):
            push_conclusion("")
            push_conclusion("📌 KẾT LUẬN THỰC CHIẾN:")
    
            if side == "SELL":
                if zlo is not None and zhi is not None:
                    push_conclusion(f"- Không SELL đuổi tại giá thấp; vùng SELL đẹp hơn: {_fmt_zone(zlo, zhi)}.")
                else:
                    push_conclusion("- Không SELL đuổi tại giá thấp; ưu tiên chờ hồi về kháng cự gần.")
    
                push_conclusion("- Context hôm nay vẫn ưu tiên SELL.")
                push_conclusion("- Nếu giá hồi lên rồi fail/rejection/đóng yếu → SELL đẹp nhất.")
                push_conclusion("- Nếu không hồi: chỉ xét SELL continuation khi break low và giữ dưới.")
                push_conclusion("- Không chờ hồi sâu quá lâu vì trend day thường không cho pullback đẹp.")
    
            elif side == "BUY":
                if zlo is not None and zhi is not None:
                    push_conclusion(f"- Không BUY đuổi tại giá cao; vùng BUY đẹp hơn: {_fmt_zone(zlo, zhi)}.")
                else:
                    push_conclusion("- Không BUY đuổi tại giá cao; ưu tiên chờ hồi về hỗ trợ gần.")
    
                push_conclusion("- Context hôm nay vẫn ưu tiên BUY.")
                push_conclusion("- Nếu giá hồi xuống rồi giữ/rejection/đóng mạnh → BUY đẹp nhất.")
                push_conclusion("- Nếu không hồi: chỉ xét BUY continuation khi break high và giữ trên.")
                push_conclusion("- Không chờ hồi sâu quá lâu vì trend day thường không cho pullback đẹp.")
    
    except Exception as e:
        push_conclusion(f"📌 KẾT LUẬN THỰC CHIẾN: lỗi render ({e})")
    push_conclusion("━━━━━━━━━━━")
    push_conclusion("🎯 KỊCH BẢN CHÍNH")
    push_conclusion("━━━━━━━━━━━")
    
    scenario = meta.get("scenario_v1") or {}
    wf1 = meta.get("wait_for_v1") or {}
    za1 = meta.get("zone_action_v1") or {}
    kl1 = meta.get("key_levels") or {}

    kb_lines = []

    # ===== Priority 0: ZONE + ACTION ENGINE =====
    if za1.get("ok"):
        msg = str(za1.get("message") or "").strip()
        trg = str(za1.get("trigger") or "").strip()
        brk = _safe_float(za1.get("break_level"))
        side = str(za1.get("side") or "NONE").upper()

        if msg:
            kb_lines.append(f"- {msg}")
        if trg:
            kb_lines.append(f"- {trg}")

        if brk is not None:
            if side == "BUY":
                kb_lines.append(f"- Hoặc BUY breakout nếu M15 đóng trên {_fmt(brk)} và giữ được")
            elif side == "SELL":
                kb_lines.append(f"- Hoặc SELL breakdown nếu M15 đóng dưới {_fmt(brk)} và giữ được")

    # ===== Priority 1: wait_for_v1 đã được rewrite bởi zone_action_v1 =====
    if not kb_lines:
        wait_lines = wf1.get("lines") or []
        for s in wait_lines[:3]:
            if s and str(s).strip():
                kb_lines.append(f"- {s}")

    # ===== Priority 2: scenario.base_case =====
    if not kb_lines:
        base_case = str(scenario.get("base_case") or "").strip()
        best_zone = str(scenario.get("best_zone") or "").strip()

        if base_case:
            try:
                base_case = re.sub(r"^\s*Base case:\s*", "", base_case, flags=re.I).strip()
            except Exception:
                pass

            kb_lines.append(f"- {base_case}")

            if best_zone and "vùng" not in base_case.lower():
                kb_lines.append(f"- Vùng ưu tiên: {best_zone}")

    # ===== Priority 3: fallback từ PATH FORECAST, map đúng BUY->support, SELL->resistance =====
    if not kb_lines:
        final_side_kb = str((sce1 or {}).get("final_side") or final_side or "NONE").upper()

        if final_side_kb == "BUY":
            sup_zone = pf1.get("sup_near") or pf1.get("sup_far")
            brk = _safe_float(kl1.get("M15_RANGE_HIGH"))

            if sup_zone:
                kb_lines.append(f"- Chờ vùng BUY/support {_pf_zone_text(sup_zone)}")
            if brk is not None:
                kb_lines.append(f"- Hoặc BUY breakout nếu M15 đóng trên {_fmt(brk)} và giữ được")

        elif final_side_kb == "SELL":
            res_zone = pf1.get("res_near") or pf1.get("res_far")
            brk = _safe_float(kl1.get("M15_RANGE_LOW"))

            if res_zone:
                kb_lines.append(f"- Chờ vùng SELL/resistance {_pf_zone_text(res_zone)}")
            if brk is not None:
                kb_lines.append(f"- Hoặc SELL breakdown nếu M15 đóng dưới {_fmt(brk)} và giữ được")

    # ===== Final fallback =====
    if not kb_lines:
        kb_lines.append("- Chờ thị trường rõ hơn")

    for s in kb_lines[:3]:
        push_conclusion(s)
    
    # PROBE + SETUP CLASS
    for s in _render_probe_block_v1(sig):
        add(conclusion_lines, s)
    for s in _render_setup_class_block_v4(sig, final_score, tradeable_label):
        add(conclusion_lines, s)
    

    # SCORE LOGIC V2 - Context + Entry + Probe
    fs2 = _compute_final_score_v2(sig)

    final_score = fs2.get("final_score", 0)
    tradeable_label = fs2.get("tradeable", "NO")
    grade = fs2.get("grade", "D")
    score_reasons = fs2.get("reasons", [])
    tradeable_reasons = fs2.get("risk_reasons", [])

    fd1 = meta.get("final_decision_engine_v1") or {}
    me1 = meta.get("master_engine_v1") or {}
    sce1 = meta.get("signal_consistency_v1") or {}

    # Đồng bộ master engine cho output
    try:
        if tradeable_label == "CONDITIONAL":
            me1["state"] = "WAIT_TIMING"
            me1["tradeable_final"] = "CONDITIONAL"
            me1["best_side"] = fs2.get("side") or me1.get("best_side") or "NONE"
            me1["confidence"] = "MEDIUM"
            me1["reason"] = [
                "context có lợi thế",
                "entry cần trigger/confirmation",
                "tradeable dạng conditional, không phải auto-entry",
            ]
            meta["master_engine_v1"] = me1
        elif tradeable_label == "YES":
            me1["state"] = "READY"
            me1["tradeable_final"] = "YES"
            me1["best_side"] = fs2.get("side") or me1.get("best_side") or "NONE"
            me1["confidence"] = "HIGH"
            meta["master_engine_v1"] = me1
    except Exception:
        pass
    
    push_conclusion("")
    push_conclusion(f"📊 Chất lượng cơ hội: {grade} | {symbol}")
    push_conclusion(f"🔥 Final Score: {final_score}/100")
    push_conclusion(f"→ Tradeable: {tradeable_label}")
    push_conclusion(
        f"- Score V2: Context {fs2.get('context_score')}/100 | "
        f"Entry {fs2.get('entry_score')}/100 | "
        f"Risk penalty -{fs2.get('risk_penalty')}"
    )
    if str(tradeable_label).upper() == "CONDITIONAL":
        summary_line = "Có context tốt nhưng entry cần trigger/confirmation; chỉ theo dõi kèo có điều kiện, không vào bừa."
    else:
        summary_line = _market_summary_line(final_score, tradeable_label, session_v4, htf_pressure_v4)
    push_conclusion(f"- {summary_line}")
    
    if score_reasons:
        push_conclusion(f"- Điểm cộng/trừ chính: {', '.join(score_reasons)}")
    if tradeable_reasons:
        push_conclusion(f"- Lý do chưa trade: {', '.join(tradeable_reasons)}")
    
    # thêm 1 dòng EDGE
    edge_line = "Chưa có edge rõ"
    if grade in ("A", "B", "B-"):
        edge_line = "Có edge nhưng cần chọn điểm vào kỹ"
    elif grade == "C":
        edge_line = "Ý tưởng có thể đúng nhưng rủi ro còn cao"
    push_conclusion(f"- EDGE: {edge_line}")
    
    if playbook_v4.get("quality"):
        push_conclusion(f"- Độ sạch theo playbook: {playbook_v4.get('quality')}")
    if ntz.get("active"):
        rs = "; ".join(str(x) for x in (ntz.get("reasons") or []) if x)
        push_conclusion(f"- Cảnh báo: {rs or 'đang là vùng nên đứng ngoài'}")
    
    # final output
    out = []
    out.extend(lines)
    
    out.append("━━━━━━━━━━━━━━━━━━")
    out.append("🧩 BLOCK 1: THÔNG TIN & INDICATOR")
    out.append("━━━━━━━━━━━━━━━━━━")
    out.extend(info_lines)
    
    out.append("")
    out.append("━━━━━━━━━━━━━━━━━━")
    out.append("🎯 BLOCK 2: PHÂN TÍCH & QUYẾT ĐỊNH")
    out.append("━━━━━━━━━━━━━━━━━━")
    out.extend(decision_lines)
    
    out.append("")
    out.append("━━━━━━━━━━━━━━━━━━")
    out.append("🚀 BLOCK 3: KẾT LUẬN & HÀNH ĐỘNG")
    out.append("━━━━━━━━━━━━━━━━━━")
    out.extend(conclusion_lines)
    
    final = []
    for line in out:
        if line == "" and (not final or final[-1] == ""):
            continue
        final.append(line)
    
    return "\n".join(final).strip()
