# app/pro++_analysis.py
# PRO++ upgrade: 4-pillar checklist (Bias/Location/Liquidity/Confirmation)
# - Designed to trigger at the "circled" candle: sweep + rejection/engulf (M15)
# - Adds optional CHoCH/BOS confirmation for higher quality
# - Hardens candle reading so MT5 payload formats (o/h/l/c/t/v) won't produce n/a.

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Sequence
from dataclasses import dataclass
import math
import time

from app.risk import calc_smart_sl_tp


# =========================================================
# Candle helpers (robust MT5 key mapping)
# =========================================================

def _c_val(c, key: str, default=None):
    """Safe accessor supporting dicts, dataclasses and common MT5 payload aliases."""
    try:
        if c is None:
            return default

        if isinstance(c, dict):
            if key in c:
                return c.get(key, default)

            # Common aliases
            alias = {
                "time": ("t", "ts", "Time"),
                "ts": ("t", "time", "Time"),
                "open": ("o", "Open"),
                "high": ("h", "High"),
                "low": ("l", "Low"),
                "close": ("c", "Close"),
                "volume": ("v", "tick_volume", "Volume"),
            }
            for k in alias.get(key, ()):
                if k in c:
                    return c.get(k, default)
            return default

        # dataclass / object
        if hasattr(c, key):
            return getattr(c, key)

        alias_obj = {
            "time": ("t", "ts"),
            "ts": ("t", "time"),
            "open": ("o",),
            "high": ("h",),
            "low": ("l",),
            "close": ("c",),
            "volume": ("v", "tick_volume"),
        }
        for k in alias_obj.get(key, ()):
            if hasattr(c, k):
                return getattr(c, k)

        return default
    except Exception:
        return default


def _to_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _safe_candles(raw: Optional[Sequence[Any]]) -> List[Candle]:
    """Normalize candles into a consistent Candle list.

    Accepts dicts with keys:
      - open/high/low/close/time/ts/volume
      - o/h/l/c/t/v (typical MT5 bridge)
      - Open/High/Low/Close/Time

    Returns sorted list by ts.
    """
    if not raw:
        return []

    out: List[Candle] = []
    for c in raw:
        if isinstance(c, Candle):
            out.append(c)
            continue

        ts = _c_val(c, "ts", None)
        if ts is None:
            ts = _c_val(c, "time", 0)
        ts_i = int(_to_float(ts, 0.0))

        o = _to_float(_c_val(c, "open", 0.0), 0.0)
        h = _to_float(_c_val(c, "high", 0.0), 0.0)
        l = _to_float(_c_val(c, "low", 0.0), 0.0)
        cl = _to_float(_c_val(c, "close", 0.0), 0.0)
        v = _to_float(_c_val(c, "volume", 0.0), 0.0)

        # Guard: if H/L inverted due to bad payload
        if h and l and h < l:
            h, l = l, h

        out.append(Candle(ts=ts_i, open=o, high=h, low=l, close=cl, volume=v))

    out.sort(key=lambda x: x.ts)
    return out


# =========================================================
# Session helper (Asia/Singapore / Vietnam = UTC+7)
# =========================================================

def _session_vn(ts: Optional[int] = None) -> str:
    """Rough trading session for display. Crypto is 24/7 but this helps context."""
    if ts is None or ts <= 0:
        ts = int(time.time())
    # Convert to UTC+7
    h = (time.gmtime(ts).tm_hour + 7) % 24

    # Approx windows (hour VN)
    # Asia: 07-15, Europe: 14-22, US: 19-03
    if 7 <= h < 14:
        return "Phiên Á"
    if 14 <= h < 19:
        return "Giao phiên Á-Âu"
    if 19 <= h < 23:
        return "Phiên Âu"
    if h >= 23 or h < 3:
        return "Giao phiên Âu-Mỹ"
    return "Phiên Mỹ"


# =========================================================
# Core analytics (swings / structure / range)
# =========================================================

def _swing_points(vals: List[float], left: int = 2, right: int = 2) -> Tuple[List[int], List[int]]:
    """Return indices of swing highs and swing lows on a 1D series."""
    highs: List[int] = []
    lows: List[int] = []
    n = len(vals)
    if n < left + right + 3:
        return highs, lows

    for i in range(left, n - right):
        v = vals[i]
        # swing high
        if all(v > vals[i - j] for j in range(1, left + 1)) and all(v > vals[i + j] for j in range(1, right + 1)):
            highs.append(i)
        # swing low
        if all(v < vals[i - j] for j in range(1, left + 1)) and all(v < vals[i + j] for j in range(1, right + 1)):
            lows.append(i)

    return highs, lows


def _structure_from_swings(candles: List[Candle]) -> Tuple[str, Dict[str, float]]:
    """Return structure tag and key swing levels."""
    if not candles or len(candles) < 30:
        return "n/a", {}

    closes = [c.close for c in candles]
    hi_idx, lo_idx = _swing_points(closes, left=2, right=2)

    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return "TRANSITION", {}

    h1 = closes[hi_idx[-1]]
    h2 = closes[hi_idx[-2]]
    l1 = closes[lo_idx[-1]]
    l2 = closes[lo_idx[-2]]

    if h1 > h2 and l1 > l2:
        tag = "HH-HL"
    elif h1 < h2 and l1 < l2:
        tag = "LL-LH"
    else:
        tag = "TRANSITION"

    levels = {
        "LAST_SWING_HIGH": float(h1),
        "PREV_SWING_HIGH": float(h2),
        "LAST_SWING_LOW": float(l1),
        "PREV_SWING_LOW": float(l2),
    }
    return tag, levels


def _range_pos(candles: List[Candle], lookback: int = 50) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not candles:
        return None, None, None
    arr = candles[-lookback:] if len(candles) > lookback else candles
    hi = max(c.high for c in arr)
    lo = min(c.low for c in arr)
    if hi <= 0 or lo <= 0 or hi == lo:
        return 0.5, hi, lo
    pos = (arr[-1].close - lo) / (hi - lo)
    return pos, hi, lo


# =========================================================
# Liquidity + Confirmation (PRO++)
# =========================================================

def _wick_parts(c: Candle) -> Tuple[float, float, float, float]:
    """Returns (range, body, upper_wick, lower_wick)."""
    rng = max(0.0, c.high - c.low)
    body = abs(c.close - c.open)
    upper = max(0.0, c.high - max(c.open, c.close))
    lower = max(0.0, min(c.open, c.close) - c.low)
    return rng, body, upper, lower


def _liq_sweep(candles: List[Candle], lookback: int = 12) -> Optional[Dict[str, Any]]:
    """Detect classic stop-hunt sweep + close back inside on the trigger candle (-2)."""
    if not candles or len(candles) < lookback + 5:
        return None

    trig = candles[-2]  # candle just closed
    prev = candles[-(lookback + 2):-2]

    prev_high = max(c.high for c in prev)
    prev_low = min(c.low for c in prev)

    # Sweep high + close back in
    if trig.high > prev_high and trig.close < prev_high:
        return {"dir": "SELL", "swept": float(prev_high), "note": "sweep_high_close_back_in"}

    # Sweep low + close back in
    if trig.low < prev_low and trig.close > prev_low:
        return {"dir": "BUY", "swept": float(prev_low), "note": "sweep_low_close_back_in"}

    return None


def _confirm_on_trigger(candles: List[Candle], liq: Dict[str, Any]) -> Tuple[bool, str]:
    """Confirmation that fires at the circled candle:

    - For SELL: trigger candle is bearish with strong upper wick OR bearish engulf OR bearish impulse
    - For BUY:  trigger candle is bullish with strong lower wick OR bullish engulf OR bullish impulse
    """
    if not candles or len(candles) < 4:
        return False, "no_data"

    trig = candles[-2]
    prev = candles[-3]

    rng, body, upper, lower = _wick_parts(trig)
    if rng <= 0:
        return False, "bad_range"

    # engulf detection
    bear_engulf = (prev.close > prev.open) and (trig.close < trig.open) and (trig.open >= prev.close) and (trig.close <= prev.open)
    bull_engulf = (prev.close < prev.open) and (trig.close > trig.open) and (trig.open <= prev.close) and (trig.close >= prev.open)

    # impulse: body dominates range
    impulse = (body / rng) >= 0.6

    if liq.get("dir") == "SELL":
        if bear_engulf:
            return True, "bear_engulf"
        if upper > max(1e-9, body) * 1.5 and trig.close < trig.open:
            return True, "bear_reject"
        if impulse and trig.close < trig.open:
            return True, "bear_impulse"
        return False, "no_confirm"

    if liq.get("dir") == "BUY":
        if bull_engulf:
            return True, "bull_engulf"
        if lower > max(1e-9, body) * 1.5 and trig.close > trig.open:
            return True, "bull_reject"
        if impulse and trig.close > trig.open:
            return True, "bull_impulse"
        return False, "no_confirm"

    return False, "no_confirm"


def _confirm_choch_bos(candles: List[Candle], liq: Dict[str, Any], swing_levels: Dict[str, float]) -> Tuple[bool, str, Optional[float]]:
    """Optional higher-grade confirmation using the next candle (-1):

    - After SELL sweep, look for close below last swing low (CHOCH/BOS down)
    - After BUY sweep,  look for close above last swing high (CHOCH/BOS up)

    Returns (ok, reason, bos_level)
    """
    if not candles or len(candles) < 5:
        return False, "no_data", None

    nxt = candles[-1]  # current forming/just closed depending on your feed; we treat as available
    last_sh = swing_levels.get("LAST_SWING_HIGH")
    last_sl = swing_levels.get("LAST_SWING_LOW")

    if liq.get("dir") == "SELL" and last_sl is not None:
        if nxt.close < float(last_sl):
            return True, "bos_down", float(last_sl)
        return False, "no_bos", float(last_sl)

    if liq.get("dir") == "BUY" and last_sh is not None:
        if nxt.close > float(last_sh):
            return True, "bos_up", float(last_sh)
        return False, "no_bos", float(last_sh)

    return False, "no_bos", None


# =========================================================
# PRO++ Checklist 4 pillars
# =========================================================

def _bias_ok(direction: str, h4_tag: str, h1_tag: str) -> Tuple[bool, str]:
    """Bias rule tuned for discretionary sweeps:

    - Allow TRANSITION
    - Block only when HTF is strongly opposite
    """
    bias_txt = f"{h4_tag} / {h1_tag}"

    if direction == "SELL":
        # Block only if both are strong bullish
        if h4_tag == "HH-HL" and h1_tag == "HH-HL":
            return False, bias_txt
        return True, bias_txt

    if direction == "BUY":
        # Block only if both are strong bearish
        if h4_tag == "LL-LH" and h1_tag == "LL-LH":
            return False, bias_txt
        return True, bias_txt

    return False, bias_txt


def _location_ok(direction: str, pos: Optional[float]) -> Tuple[bool, str]:
    """Location is range-based. We keep thresholds slightly looser so it triggers like a trader."""
    if pos is None:
        return False, "thiếu dữ liệu range"

    # SELL near top, BUY near bottom
    if direction == "SELL":
        ok = pos >= 0.65
        return ok, (f"Near range HIGH (pos={pos:.2f})" if ok else f"Giữa range (pos={pos:.2f}) – No trade zone")

    if direction == "BUY":
        ok = pos <= 0.35
        return ok, (f"Near range LOW (pos={pos:.2f})" if ok else f"Giữa range (pos={pos:.2f}) – No trade zone")

    return False, "n/a"


# =========================================================
# Main entry used by bot
# =========================================================

def analyze_pro(symbol: str, m15, m30, h1, h4) -> Optional[Dict[str, Any]]:
    """Return a signal dict consumed by format_signal()."""

    m15c = _safe_candles(m15)
    m30c = _safe_candles(m30)
    h1c = _safe_candles(h1)
    h4c = _safe_candles(h4)

    # If missing candles, return a minimal dict (avoid NoneType.get downstream)
    if not m15c or not h1c or not h4c:
        sess = _session_vn(int(time.time()))
        return {
            "symbol": symbol,
            "tf": "M30",
            "session": sess,
            "recommendation": "CHỜ",
            "stars": 1,
            "trade_mode": "MANUAL",
            "meta": {
                "structure": {"H4": "n/a", "H1": "n/a", "M15": "n/a"},
                "where": "Thiếu dữ liệu nến (MT5 stale / market đóng / source fail)",
                "wait_for": "Đợi có đủ M15/H1/H4",
                "score_detail": {"score": 0},
                "key_levels": {},
            },
            "liquidity_lines": ["missing candles"],
            "quality_lines": ["no analysis"],
            "show_trade_plan": False,
        }

    # --- structure tags
    h4_tag, h4_lv = _structure_from_swings(h4c)
    h1_tag, h1_lv = _structure_from_swings(h1c)
    m15_tag, m15_lv = _structure_from_swings(m15c)

    # --- range/location (M15)
    pos, r_hi, r_lo = _range_pos(m15c, lookback=50)

    # --- liquidity sweep (M15 trigger candle)
    liq = _liq_sweep(m15c, lookback=12)

    direction = liq["dir"] if liq else "CHỜ"

    # --- confirmation
    conf_trig_ok, conf_trig_reason = (False, "no_sweep")
    if liq:
        conf_trig_ok, conf_trig_reason = _confirm_on_trigger(m15c, liq)

    # --- PRO++ extra: CHOCH/BOS on the next candle (optional boost)
    bos_ok, bos_reason, bos_level = (False, "no_sweep", None)
    if liq:
        # Use minor swing levels from M15 for BOS trigger (fast, like your entries)
        bos_ok, bos_reason, bos_level = _confirm_choch_bos(m15c, liq, m15_lv or {})

    # --- checklist 4 pillars
    bias_ok, bias_txt = _bias_ok(direction, h4_tag, h1_tag)
    loc_ok, loc_txt = _location_ok(direction, pos)
    liq_ok = bool(liq)
    liq_txt = liq.get("note") if liq else "no sweep"

    # Confirmation pillar: trigger confirm OR BOS confirm (both acceptable)
    conf_ok = bool(conf_trig_ok or bos_ok)
    conf_txt = conf_trig_reason
    if not conf_trig_ok and bos_ok:
        conf_txt = f"{bos_reason} @ {bos_level:.2f}" if bos_level is not None else bos_reason

    stars = int(bias_ok) + int(loc_ok) + int(liq_ok) + int(conf_ok)

    # Trade mode:
    # - FULL if BOS happened (stronger)
    # - HALF if only trigger confirmation
    trade_mode = "MANUAL"
    if stars >= 3 and direction in ("BUY", "SELL"):
        trade_mode = "FULL" if bos_ok else "HALF"

    # Recommendation
    rec = "CHỜ"
    if stars >= 3 and direction in ("BUY", "SELL"):
        rec = direction

    # Build key levels
    key_levels: Dict[str, Any] = {}
    key_levels.update({
        "H1_HH": h1_lv.get("LAST_SWING_HIGH"),
        "H1_HL": h1_lv.get("LAST_SWING_LOW"),
        "H1_LH": h1_lv.get("LAST_SWING_HIGH"),
        "H1_LL": h1_lv.get("LAST_SWING_LOW"),
        "M15_RANGE_LOW": r_lo,
        "M15_RANGE_HIGH": r_hi,
        "M15_LAST": m15c[-1].close if m15c else None,
        "M15_BOS": bos_level,
    })

    # Where/Wait
    where = ""
    wait_for = ""
    if direction == "CHỜ":
        where = "Chưa có sweep/liquidity trigger"
        wait_for = "Đợi quét thanh khoản + nến rejection"
    else:
        where = loc_txt
        if not conf_ok:
            wait_for = "Đợi nến confirm (rejection/engulf) hoặc BOS"
        elif trade_mode == "HALF":
            wait_for = "Có thể vào theo nến confirm; ưu tiên retest"
        else:
            wait_for = "BOS confirmed: có thể vào theo plan"

    # Liquidity lines
    liq_lines: List[str] = []
    if liq:
        liq_lines.append(f"{liq_txt} (swept={liq.get('swept')})")
    if bos_reason and direction in ("BUY", "SELL"):
        liq_lines.append(f"CHOCH/BOS: {bos_reason}" + (f" @ {bos_level:.2f}" if bos_level is not None else ""))

    # Quality lines
    qual_lines: List[str] = []
    qual_lines.append(f"Checklist: Bias={bias_ok} | Location={loc_ok} | Liquidity={liq_ok} | Confirm={conf_ok}")
    qual_lines.append(f"Confirm: {conf_txt}")

    # Risk plan (keep your existing risk function)
    entry = None
    sl = None
    tp1 = None
    tp2 = None
    show_plan = True

    try:
        # Use your risk helper if available; it should handle BUY/SELL/CHỜ.
        plan = calc_smart_sl_tp(symbol, rec, m15c[-1].close)
        if isinstance(plan, dict):
            entry = plan.get("entry")
            sl = plan.get("sl")
            tp1 = plan.get("tp1")
            tp2 = plan.get("tp2")
    except Exception:
        # fallback simple plan only if trade
        show_plan = False

    sess = _session_vn(m15c[-1].ts if m15c else int(time.time()))

    # score_detail must remain compatible with existing format_signal
    score_detail = {
        "bias_ok": bool(bias_ok),
        "pullback_ok": bool(loc_ok),      # map legacy PB -> Location
        "momentum_ok": bool(conf_ok),     # map legacy MOM -> Confirmation
        "confluence_ok": bool(liq_ok),    # map legacy Confluence -> Liquidity
        "score": int(max(0, min(4, stars))),

        # also store explicit 4 pillars
        "location_ok": bool(loc_ok),
        "liquidity_ok": bool(liq_ok),
        "confirmation_ok": bool(conf_ok),
    }

    sig: Dict[str, Any] = {
        "symbol": symbol,
        "tf": "M30",
        "session": sess,
        "recommendation": rec,
        "stars": int(max(1, min(5, stars))),
        "trade_mode": trade_mode,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "show_trade_plan": show_plan,
        "liquidity_lines": liq_lines,
        "quality_lines": qual_lines,
        "meta": {
            "structure": {"H4": h4_tag, "H1": h1_tag, "M15": m15_tag},
            "key_levels": key_levels,
            "where": where,
            "wait_for": wait_for,
            "score_detail": score_detail,
            "propp": {
                "bias_txt": bias_txt,
                "location_txt": loc_txt,
                "liquidity_txt": liq_txt,
                "confirmation_txt": conf_txt,
                "bos_ok": bos_ok,
                "bos_reason": bos_reason,
            },
        },
    }

    return sig


# =========================================================
# Telegram formatting (keeps your current style/icons)
# =========================================================

def format_signal(sig: Dict[str, Any]) -> str:
    """Format Telegram message (PRO++).
    - Keep old readable layout (like previous versions)
    - Always show Entry/SL/TP when stars>=3 and recommendation is BUY/SELL
    - Never crash if any field is missing/None
    """

    def nf2(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "..."

    symbol = sig.get("symbol", "?")
    tf = sig.get("tf", "?")
    meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
    session = sig.get("session") or (meta.get("session") if isinstance(meta, dict) else None)

    rec = sig.get("recommendation", "CHỜ")
    stars = int(sig.get("stars") or 0)
    trade_mode = sig.get("trade_mode") or "MANUAL"

    sd = sig.get("score_detail") if isinstance(sig.get("score_detail"), dict) else {}

    lines: List[str] = []

    # Header
    head = f"📊 {symbol} | {tf}"
    if session:
        head += f" | {session}"
    lines.append(head)

    # Data source (MT5 / TWELVEDATA_FALLBACK / etc.)
    data_source = sig.get("data_source") or meta.get("data_source")
    if data_source:
        lines.append(f"🛰️ Data: {data_source}")

    # Stars + Recommendation
    stars_txt = "⭐" * max(0, min(stars, 5))
    mode_txt = f" | Mode: {trade_mode}" if trade_mode else ""
    score_txt = ""
    if isinstance(sd.get("score"), (int, float)):
        score_txt = f" ({int(sd.get('score'))}/4)"
    lines.append(f"{stars_txt} {rec}{mode_txt}{score_txt}")

    # Structure tags
    h4_tag = meta.get("h4_tag") or "n/a"
    h1_tag = meta.get("h1_tag") or "n/a"
    m15_tag = meta.get("m15_tag") or "n/a"
    lines.append(f"Structure (Major): H4 {h4_tag} | H1 {h1_tag}")
    lines.append(f"Structure (Minor): M15 {m15_tag}")

    # Key levels
    key_lines = sig.get("key_levels_lines")
    if isinstance(key_lines, list) and key_lines:
        lines.append("Key levels:")
        lines.extend([str(x) for x in key_lines if x is not None])

    # Entry plan (3⭐+ BUY/SELL)
    entry = sig.get("entry")
    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")

    want_plan = (stars >= 3) and (str(rec).upper() in ("BUY", "SELL"))
    has_plan = (entry is not None) and (sl is not None) and (tp1 is not None)

    if want_plan and has_plan:
        lines.append(f"Entry: {nf2(entry)}")
        lines.append(f"SL: {nf2(sl)} | TP1: {nf2(tp1)} | TP2: {nf2(tp2)}")
    else:
        lines.append("Entry: ...")
        lines.append("SL: ... | TP1: ... | TP2: ...")

    # Liquidity
    liq_lines = sig.get("liquidity_lines") if isinstance(sig.get("liquidity_lines"), list) else []
    if liq_lines:
        lines.append("")
        lines.append("Liquidity:")
        lines.extend([str(x) for x in liq_lines if x is not None])

    # Normalized signal + checklist
    lines.append("")
    lines.append("📌 TÍN HIỆU (chuẩn hoá):")

    # Keep existing "standard" lines if present
    std_lines = sig.get("standard_lines") if isinstance(sig.get("standard_lines"), list) else None
    if std_lines:
        lines.extend([str(x) for x in std_lines if x is not None])

    checklist = sd.get("checklist") if isinstance(sd.get("checklist"), dict) else {}
    bias_ok = bool(checklist.get("bias"))
    loc_ok = bool(checklist.get("location"))
    liq_ok = bool(checklist.get("liquidity"))
    conf_ok = bool(checklist.get("confirmation"))

    def mark(ok: bool) -> str:
        return "✅" if ok else "❌"

    # Optional detail strings
    bias_detail = sd.get("bias_detail") or ""
    loc_detail = sd.get("location_detail") or ""
    liq_detail = sd.get("liquidity_detail") or ""
    conf_detail = sd.get("confirmation_detail") or ""

    lines.append("Checklist (4):")
    lines.append(f"1) Bias: {mark(bias_ok)} {bias_detail}".rstrip())
    lines.append(f"2) Location: {mark(loc_ok)} {loc_detail}".rstrip())
    lines.append(f"3) Liquidity: {mark(liq_ok)} {liq_detail}".rstrip())
    lines.append(f"4) Confirmation: {mark(conf_ok)} {conf_detail}".rstrip())

    # Missing / wait hints
    where = meta.get("where") or sig.get("where")
    wait_for = meta.get("wait_for") or sig.get("wait_for")
    if where or wait_for:
        lines.append("Thiếu / chờ thêm:")
        if where:
            lines.append(f"- {where}")
        if wait_for:
            lines.append(f"- {wait_for}")

    return "\n".join(lines)
