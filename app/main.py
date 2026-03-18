# app/main.py
from __future__ import annotations
import json
import os
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from app.data_source import get_candles, ingest_mt5_candles
from app.pro_analysis import analyze_pro, format_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="xau-liquidity-bot")

# ---- Simple concurrency guards for cron ----
CRON_LOCK = asyncio.Lock()
LAST_CRON_TS = 0
MIN_CRON_GAP_SEC = int(os.getenv("MIN_CRON_GAP_SEC", "25"))

# Default symbols (override by env SYMBOLS="XAU/USD,BTC/USD")
DEFAULT_SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "XAU/USD,BTC/USD").split(",") if s.strip()]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # default chat for cron
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", TELEGRAM_CHAT_ID)

REGIME_ALERT_ENABLED=1
REGIME_CHOP_THRESHOLD=6.8
REGIME_ALERT_COOLDOWN_MIN=120
REGIME_ALERT_STATE_PATH = os.getenv("REGIME_ALERT_STATE_PATH", "regime_alert_state.json")


CRON_SECRET = os.getenv("CRON_SECRET", "")

# Send both symbols always by default (you can override)
MIN_STARS = int(os.getenv("MIN_STARS", "1"))

# Telegram hard limit is 4096; keep safe chunk size
TG_CHUNK = int(os.getenv("TG_CHUNK", "3500"))

@app.get("/health")
def health():
    return {"status": "ok"}
def _send_telegram(text: str, chat_id: Optional[str] = None) -> None:
    token = TELEGRAM_TOKEN
    cid = chat_id or TELEGRAM_CHAT_ID
    if not token or not cid:
        logger.info("[TG] missing TELEGRAM_TOKEN/CHAT_ID -> skip sending")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests = __import__("requests")
        payload = {
            "chat_id": cid,
            "text": str(text or ""),
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=15)
        logger.info("[TG] send status=%s body=%s", resp.status_code, resp.text)
    except Exception as e:
        logger.exception("[TG] send failed: %s", e)
        
def _send_long_telegram(text: str, chat_id: str, chunk_size: int = 3500, parse_mode=None):
    text = str(text or "")
    if not text.strip():
        return

    parts = []
    buf = ""

    for line in text.splitlines(True):  # giữ newline
        if len(buf) + len(line) <= chunk_size:
            buf += line
        else:
            if buf:
                parts.append(buf)
            if len(line) <= chunk_size:
                buf = line
            else:
                # line quá dài thì cắt cứng
                for i in range(0, len(line), chunk_size):
                    parts.append(line[i:i + chunk_size])
                buf = ""

    if buf:
        parts.append(buf)

    total = len(parts)
    for i, part in enumerate(parts, start=1):
        header = f"📩 REVIEW ({i}/{total})\n" if total > 1 else ""
        _send_telegram(header + part, chat_id=chat_id)

def _parse_symbol_from_text(text: str) -> str:
    t = text.lower()

    if "xag" in t or "silver" in t:
        return "XAG/USD"

    if "xau" in t or "gold" in t:
        return "XAU/USD"

    if "btc" in t:
        return "BTC/USD"

    return DEFAULT_SYMBOLS[0] if DEFAULT_SYMBOLS else "XAU/USD"


import re

# 1) Chuẩn hoá symbol người dùng gõ -> symbol hệ thống dùng để get_candles()
def normalize_symbol(user_sym: str) -> str:
    s = (user_sym or "").strip().upper()
    s = s.replace("-", "/").replace("_", "/").replace(" ", "")
    # cho phép: BTC, BTCUSD, BTC/USD, BTCUSDT...
    if s in ("BTC", "BTCUSD", "BTC/USD", "BTCUSDT", "BTC/USDT"):
        return "BTC/USD"
    if s in ("XAU", "XAUUSD", "XAU/USD"):
        return "XAU/USD"
    if s in ("XAG", "XAGUSD", "XAG/USD"):
        return "XAG/USD"

    # fallback: nếu user gõ kiểu BTC/USDT -> BTC/USD (mày đang dùng USD)
    if s.endswith("/USDT"):
        s = s.replace("/USDT", "/USD")
    if s.endswith("USDT"):
        s = s.replace("USDT", "/USD")
    if "/" not in s and s.endswith("USD"):
        s = s[:-3] + "/USD"
    return s


# 2) Parse lệnh manual: hỗ trợ entry 1 giá hoặc 1 vùng (a-b), TP/SL có/không
_MANUAL_RE = re.compile(
    r"^\s*(?P<sym>[A-Za-z\/\-_]+)\s+"
    r"(?P<side>BUY|SELL)\s+"
    r"(?P<entry1>\d+(\.\d+)?)"
    r"(?:\s*[-~]\s*(?P<entry2>\d+(\.\d+)?))?"
    r"(?:\s+(?:TP|TP1|TARGET)\s*[:=\s]+(?P<tp>\d+(\.\d+)?))?"
    r"(?:\s+(?:SL|STOP)\s*[:=\s]+(?P<sl>\d+(\.\d+)?))?"
    r"\s*$",
    re.IGNORECASE
)

def parse_manual_trade(text: str):
    m = _MANUAL_RE.match((text or "").strip())
    if not m:
        return None

    symbol = normalize_symbol(m.group("sym"))
    side = m.group("side").upper()

    e1 = float(m.group("entry1"))
    e2 = float(m.group("entry2")) if m.group("entry2") else None
    entry_lo = min(e1, e2) if e2 is not None else e1
    entry_hi = max(e1, e2) if e2 is not None else e1

    tp = float(m.group("tp")) if m.group("tp") else None
    sl = float(m.group("sl")) if m.group("sl") else None

    return {
        "symbol": symbol,
        "side": side,
        "entry_lo": entry_lo,
        "entry_hi": entry_hi,
        "tp": tp,
        "sl": sl,
    }

def _as_list_from_get_candles(res):
    """
    get_candles() của mày có lúc trả:
      - list
      - (list, meta)
    -> normalize về list
    """
    if isinstance(res, tuple) and len(res) >= 1:
        return res[0] or []
    return res or []


def _as_list_and_source_from_get_candles(res):
    """Unwrap return of get_candles(): (candles, source) or candles."""
    source = None
    candles = res
    if isinstance(res, tuple) and len(res) == 2:
        candles, source = res
    if candles is None:
        return [], source
    try:
        return list(candles), source
    except Exception:
        return [], source

def _cget(c, k, default=0.0):
    if isinstance(c, dict):
        v = c.get(k, default)
    else:
        v = getattr(c, k, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _m15_closed(candles):
    # bỏ nến đang chạy (nếu có)
    if not candles:
        return []
    return candles[:-1] if len(candles) > 1 else candles

def _atr14_simple(candles):
    """
    ATR(14) đơn giản (Wilder) để review lệnh.
    candles: list dict/object OHLC (đã đóng là tốt nhất)
    """
    cs = candles or []
    if len(cs) < 16:
        return None
    trs = []
    for i in range(1, len(cs)):
        h = _cget(cs[i], "high")
        l = _cget(cs[i], "low")
        pc = _cget(cs[i-1], "close")
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < 14:
        return None
    atr = sum(trs[:14]) / 14.0
    for j in range(14, len(trs)):
        atr = (atr * 13.0 + trs[j]) / 14.0
    return atr

def _range30_info_m15(m15):
    closed = _m15_closed(m15)
    use = closed[-30:] if len(closed) >= 30 else closed
    if len(use) < 12:
        return None
    lo = min(_cget(x, "low") for x in use)
    hi = max(_cget(x, "high") for x in use)
    cur = _cget(use[-1], "close")
    rng = max(1e-9, hi - lo)
    pos = (cur - lo) / rng  # 0..1
    return {"lo": lo, "hi": hi, "cur": cur, "pos": pos, "use": use}

def _hl_lh_gate(m15, atr_val):
    """
    “CHỜ CẤU TRÚC” nghĩa là gì? -> trả text rõ:
    - BUY: M15 tạo Higher-Low (HL) + sau đó đóng vượt đỉnh gần (break high gần)
    - SELL: M15 tạo Lower-High (LH) + sau đó đóng thủng đáy gần (break low gần)
    """
    closed = _m15_closed(m15)
    if len(closed) < 22:
        return {"hl": False, "lh": False, "break_up": False, "break_dn": False, "txt": "Chưa đủ dữ liệu M15 để xác nhận cấu trúc."}

    # cushion theo ATR để tránh nhiễu
    a = float(atr_val or 0.0)
    cushion = max(1e-9, 0.12 * a) if a > 0 else 0.0

    last5 = closed[-5:]
    prev5 = closed[-10:-5]

    low_last5 = min(_cget(x, "low") for x in last5)
    low_prev5 = min(_cget(x, "low") for x in prev5)
    hl = low_last5 > (low_prev5 + cushion)

    high_last5 = max(_cget(x, "high") for x in last5)
    high_prev5 = max(_cget(x, "high") for x in prev5)
    lh = high_last5 < (high_prev5 - cushion)

    # “break” mốc gần: dùng high/low của 10 nến trước đó (không tính 1-2 nến cuối)
    ref = closed[-12:-2]
    ref_hi = max(_cget(x, "high") for x in ref)
    ref_lo = min(_cget(x, "low") for x in ref)
    cur_close = _cget(closed[-1], "close")

    break_up = cur_close > (ref_hi + 0.05 * a if a > 0 else ref_hi)
    break_dn = cur_close < (ref_lo - 0.05 * a if a > 0 else ref_lo)

    txt = (
        f"Gate cấu trúc:\n"
        f"- BUY chỉ mạnh khi: HL=True và M15 đóng > đỉnh gần ({ref_hi:.2f}).\n"
        f"- SELL chỉ mạnh khi: LH=True và M15 đóng < đáy gần ({ref_lo:.2f})."
    )
    return {"hl": hl, "lh": lh, "break_up": break_up, "break_dn": break_dn, "ref_hi": ref_hi, "ref_lo": ref_lo, "txt": txt}
def review_manual_trade(symbol: str, side: str, entry_lo: float, entry_hi: float, tp: float | None, sl: float | None) -> str:
    symbol = str(symbol or "").strip().upper()
    side = (side or "").upper().strip()

    def _f(x, nd: int = 2, default: str = "n/a") -> str:
        try:
            if x is None:
                return default
            return f"{float(x):.{nd}f}"
        except Exception:
            return default

    def _pct(x, default: str = "n/a") -> str:
        try:
            return f"{int(round(float(x) * 100))}%"
        except Exception:
            return default

    def _safe_float(x, default: float = 0.0) -> float:
        try:
            return float(x) if x is not None else default
        except Exception:
            return default

    try:
        entry_lo = float(entry_lo)
        entry_hi = float(entry_hi)
        entry = (entry_lo + entry_hi) / 2.0
    except Exception as e:
        logger.exception("Invalid entry values for manual review %s %s: %s", symbol, side, e)
        return f"❌ REVIEW lỗi cho {symbol}: entry không hợp lệ."

    try:
        m15, src15 = _as_list_and_source_from_get_candles(get_candles(symbol, "15min", limit=220))
        m30, src30 = _as_list_and_source_from_get_candles(get_candles(symbol, "30min", limit=220))
        h1, src1h = _as_list_and_source_from_get_candles(get_candles(symbol, "1h", limit=220))
        h4, src4h = _as_list_and_source_from_get_candles(get_candles(symbol, "4h", limit=220))
    except Exception as e:
        logger.exception("get_candles failed for %s: %s", symbol, e)
        return f"❌ REVIEW lỗi cho {symbol}: không lấy được dữ liệu nến ({e})."

    try:
        sig = analyze_pro(symbol, m15, m30, h1, h4)
        if not isinstance(sig, dict):
            sig = {
                "symbol": symbol, "tf": "M30", "session": "", "recommendation": "CHỜ",
                "stars": 1, "trade_mode": "MANUAL", "meta": {},
                "context_lines": ["Hệ phân tích chưa trả đủ signal → fallback review."],
                "liquidity_lines": [], "quality_lines": [], "notes": [],
            }
    except Exception as e:
        logger.exception("analyze_pro failed for %s: %s", symbol, e)
        sig = {
            "symbol": symbol, "tf": "M30", "session": "", "recommendation": "CHỜ",
            "stars": 1, "trade_mode": "MANUAL", "meta": {},
            "context_lines": [f"Hệ phân tích lỗi → fallback review ({e})."],
            "liquidity_lines": [], "quality_lines": [], "notes": [],
        }

    try:
        ds = src30 or src15 or src1h or src4h
        if ds:
            sig["data_source"] = ds
            sig.setdefault("meta", {})["data_source"] = ds
    except Exception:
        pass

    meta = sig.get("meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}

    volq = meta.get("volq", {}) or {}
    cpat = meta.get("candle", {}) or {}
    div = meta.get("div", {}) or {}
    phase369 = meta.get("phase_369", {}) or {}
    flow_state = meta.get("flow_state", {}) or {}
    liquidation = meta.get("liquidation", {}) or {}
    no_trade_zone = meta.get("no_trade_zone", {}) or {}
    market_state_v2 = meta.get("market_state_v2")
    playbook = meta.get("playbook_v2", {}) or {}

    ctx = sig.get("context_lines", []) or []
    liq = sig.get("liquidity_lines", []) or []
    qlt = sig.get("quality_lines", []) or []
    notes = sig.get("notes", []) or []

    atr_val = None
    for x in qlt:
        s = str(x)
        if "ATR(" in s and "~" in s:
            try:
                atr_val = float(s.split("~")[-1].strip())
                break
            except Exception:
                atr_val = None
    if atr_val is None:
        atr_val = _atr14_simple(_m15_closed(m15)) or 0.0
    a = _safe_float(atr_val, 0.0)

    rr_txt = "RR: n/a"
    try:
        if tp is not None and sl is not None and abs(entry - float(sl)) > 1e-9:
            risk = abs(entry - float(sl))
            reward = abs(float(tp) - entry)
            rr_txt = f"RR≈{(reward / risk):.2f}"
    except Exception:
        pass

    rinfo = _range30_info_m15(m15)
    cur = rinfo["cur"] if isinstance(rinfo, dict) else None
    pos = rinfo["pos"] if isinstance(rinfo, dict) else None
    lo = rinfo["lo"] if isinstance(rinfo, dict) else None
    hi = rinfo["hi"] if isinstance(rinfo, dict) else None

    try:
        gate = _hl_lh_gate(m15, atr_val) or {}
    except Exception:
        gate = {"txt": "Không đọc được gate cấu trúc.", "hl": False, "lh": False, "break_up": False, "break_dn": False}

    actions = []
    vol_state = str((volq or {}).get("state") or "").upper()
    vol_ratio = _safe_float((volq or {}).get("ratio"), 0.0)
    if vol_state and vol_state != "N/A":
        actions.append(f"📦 Volume: {vol_state} (x{vol_ratio:.2f} vs SMA20)")
        if vol_state == "LOW":
            actions.append("⚠️ Volume thấp → ưu tiên TP nhanh, KHÔNG add.")
        elif vol_state == "HIGH":
            actions.append("✅ Volume cao → move đáng tin hơn, có thể giữ theo plan.")

    cpat_txt = str((cpat or {}).get("txt") or "").strip()
    if cpat_txt and cpat_txt != "N/A":
        actions.append(f"🕯 Candle: {cpat_txt}")

    div_txt = str((div or {}).get("txt") or "").strip()
    if div_txt and div_txt != "N/A":
        actions.append(f"📉 {div_txt}")

    if pos is not None:
        pos_pct = int(max(0, min(1, float(pos))) * 100)
        if side == "BUY" and float(pos) > 0.70:
            actions.append(f"⚠️ Entry đang ~{pos_pct}% range M15 → BUY dễ bị xem là đuổi.")
        if side == "SELL" and float(pos) < 0.30:
            actions.append(f"⚠️ Entry đang ~{pos_pct}% range M15 → SELL dễ bị xem là đuổi.")

    h1_txt = " ".join(str(x).lower() for x in ctx)
    if side == "BUY" and "h1: bearish" in h1_txt:
        actions.append("⚠️ BUY ngược H1 bearish → ưu tiên đánh ngắn, không gồng.")
    if side == "SELL" and "h1: bullish" in h1_txt:
        actions.append("⚠️ SELL ngược H1 bullish → ưu tiên đánh ngắn, không gồng.")

    if a > 0:
        if sl is not None:
            sl_atr = abs(entry - float(sl)) / a
            actions.append((f"⚠️ SL hơi sát: ~{sl_atr:.2f} ATR → dễ dính quét.") if sl_atr < 0.70 else f"✅ SL khoảng ~{sl_atr:.2f} ATR → tạm ổn.")
        if tp is not None:
            tp_atr = abs(float(tp) - entry) / a
            actions.append((f"⚠️ TP hơi ngắn: ~{tp_atr:.2f} ATR.") if tp_atr < 0.70 else f"✅ TP khoảng ~{tp_atr:.2f} ATR.")

    verdict = "TRUNG TÍNH"
    if cur is not None and a > 0:
        move = (float(cur) - entry) if side == "BUY" else (entry - float(cur))
        dd_atr = (-move) / a
        if side == "BUY":
            if gate.get("hl") and gate.get("break_up"):
                verdict = "ĐÚNG (cấu trúc ủng hộ)"
                actions.append("✅ Đã có HL + break đỉnh gần → có thể GIỮ.")
            elif dd_atr >= 0.80 and not gate.get("hl"):
                verdict = "SAI/NGUY HIỂM"
                actions.append("🛑 Đang âm mạnh >0.8 ATR mà chưa có HL → ưu tiên THOÁT hoặc giảm size.")
            else:
                verdict = "CHƯA RÕ"
                actions.append("⏳ Có thể giữ ngắn hạn, KHÔNG add. Chờ break xác nhận.")
        else:
            if gate.get("lh") and gate.get("break_dn"):
                verdict = "ĐÚNG (cấu trúc ủng hộ)"
                actions.append("✅ Đã có LH + break đáy gần → có thể GIỮ.")
            elif dd_atr >= 0.80 and not gate.get("lh"):
                verdict = "SAI/NGUY HIỂM"
                actions.append("🛑 Đang âm mạnh >0.8 ATR mà chưa có LH → ưu tiên THOÁT hoặc giảm size.")
            else:
                verdict = "CHƯA RÕ"
                actions.append("⏳ Có thể giữ ngắn hạn, KHÔNG add. Chờ break xác nhận.")

    if isinstance(no_trade_zone, dict) and no_trade_zone.get("active"):
        actions.append("🚨 Đang ở no-trade zone → không nên add hoặc mở rộng rủi ro.")

    try:
        mgmt = _trade_management_5_10_15(side, entry, cur, a, gate, div, cpat, volq, tp, sl) or {}
    except Exception as e:
        mgmt = {"stage": "n/a", "label": "fallback", "lines": [f"Không tính được 5-10-15: {e}"]}

    tp1_s = tp2_s = sl_s = None
    if a > 0:
        if side == "BUY":
            tp1_s = entry + 0.90 * a
            tp2_s = entry + 1.80 * a
            sl_s = entry - 1.10 * a
        else:
            tp1_s = entry - 0.90 * a
            tp2_s = entry - 1.80 * a
            sl_s = entry + 1.10 * a

    lines = []
    lines.append("🧠 REVIEW LỆNH (Manual)")
    lines.append("")
    lines.append(f"📌 {symbol} | {side} | Kết luận: {verdict}")
    lines.append(f"🎯 Entry: {_f(entry)}")
    lines.append(f"🎯 TP: {_f(tp, 2, '...')} | 🛑 SL: {_f(sl, 2, '...')} | {rr_txt}")
    if phase369:
        lines.append(f"🧭 Phase 369: {phase369.get('phase', 'n/a')} | {phase369.get('label', 'n/a')} | {phase369.get('reason') or phase369.get('note') or ''}".rstrip(" |"))
    if playbook.get("plan"):
        plan = f"🗺 Plan: {playbook.get('plan')}"
        if playbook.get("zone_low") is not None and playbook.get("zone_high") is not None:
            plan += f" | Zone: {_f(playbook.get('zone_low'))} – {_f(playbook.get('zone_high'))}"
        lines.append(plan)
    if isinstance(flow_state, dict) and flow_state.get("state"):
        lines.append(f"💰 Flow: {flow_state.get('state')} | Favored: {flow_state.get('favored_side', 'n/a')}")
    if market_state_v2:
        lines.append(f"🌡 State: {market_state_v2}")
    if isinstance(liquidation, dict) and liquidation.get("ok"):
        lines.append(f"⚠️ Liquidation: {liquidation.get('side')} | body~{float(liquidation.get('body_atr', 0) or 0):.1f} ATR | range~{float(liquidation.get('range_atr', 0) or 0):.1f} ATR")
    if isinstance(no_trade_zone, dict) and no_trade_zone.get("active"):
        reasons = '; '.join(str(x) for x in (no_trade_zone.get('reasons') or []) if x)
        lines.append(f"⛔ No-trade zone: {reasons or 'active'}")
    if a > 0:
        lines.append(f"📐 ATR(14) M15: {_f(a)}")

    if isinstance(rinfo, dict):
        lines.append("")
        lines.append("📏 Range M15:")
        lines.append(f"- {_f(lo)} – {_f(hi)}")
        cur_txt = _f(cur)
        if pos is not None:
            cur_txt += f" (~{_pct(pos)})"
        lines.append(f"- Current: {cur_txt}")

    lines.append("")
    lines.append("🧠 Context:")
    for s in (ctx[:4] if ctx else ["n/a"]):
        lines.append(f"- {s}")

    lines.append("")
    lines.append("💧 Liquidity:")
    for s in (liq[:4] if liq else ["Chưa thấy sweep/spring rõ"]):
        lines.append(f"- {s}")

    lines.append("")
    lines.append("🏗 Structure:")
    lines.append(f"- HL={'✅' if gate.get('hl') else '❌'} | LH={'✅' if gate.get('lh') else '❌'} | BreakUp={'✅' if gate.get('break_up') else '❌'} | BreakDn={'✅' if gate.get('break_dn') else '❌'}")
    lines.append(f"- {gate.get('txt') or 'Chưa đọc được gate cấu trúc.'}")

    lines.append("")
    lines.append("⚙️ Action:")
    seen = set()
    for s in actions:
        ss = str(s).strip()
        if ss and ss not in seen:
            seen.add(ss)
            lines.append(f"- {ss}")

    lines.append("")
    lines.append(f"🛑 Invalidation: {'thủng' if side == 'BUY' else 'vượt'} {_f(sl, 2, 'n/a')} hoặc mất {'HL' if side == 'BUY' else 'LH'} → ưu tiên thoát")

    lines.append("")
    lines.append("🪜 Quản trị 5-10-15:")
    lines.append(f"- Stage: {mgmt.get('stage', 'n/a')} | {mgmt.get('label', 'n/a')}")
    for s in (mgmt.get('lines', []) or [])[:4]:
        lines.append(f"- {s}")

    if a > 0:
        lines.append("")
        lines.append("📌 ATR Plan:")
        lines.append(f"- SL chuẩn: {_f(sl_s)}")
        lines.append(f"- TP1: {_f(tp1_s)}")
        lines.append(f"- TP2: {_f(tp2_s)}")
        lines.append("- +0.8 ATR → BE")
        lines.append("- +1.2 ATR → trailing 3 nến M15")

    return "\n".join(lines)

#def _fetch_triplet(symbol: str, limit: int = 260) -> Dict[str, List[Any]]:
    # M15, M30, H1
    #m15, _ = get_candles(symbol, "15min", limit)
    #m30, _ = get_candles(symbol, "30min", limit)
    #h1, _ = get_candles(symbol, "1h", limit)
    #return {"m15": m15, "m30": m30, "h1": h1}
def _fetch_triplet(symbol: str, limit: int = 260) -> Dict[str, Any]:
    # M15, M30, H1, H4 (H1+H4 confluence for Bias)
    sym = normalize_symbol(symbol)
    m15, src15 = _as_list_and_source_from_get_candles(get_candles(sym, "15min", limit=220))
    m30, src30 = _as_list_and_source_from_get_candles(get_candles(sym, "30min", limit=220))
    h1, src1h = _as_list_and_source_from_get_candles(get_candles(sym, "1h", limit=220))
    h4, src4h = _as_list_and_source_from_get_candles(get_candles(sym, "4h", limit=220))
    return {"m15": m15, "m30": m30, "h1": h1, "h4": h4, "data_source": (src30 or src15 or src1h or src4h)}

def _force_send(sig: dict) -> bool:
    ctx = " | ".join(sig.get("context_lines", []) or [])
    notes = " | ".join(sig.get("notes", []) or [])

    # Liquidity warning
    if "Liquidity WARNING" in ctx:
        return True

    # Post-sweep state
    if "POST-SWEEP" in ctx or "POST-SWEEP" in notes:
        return True

    return False
# =========================
# REGIME ALERT (CHOP / STOP-HUNT) - independent of stars
# =========================

REGIME_ALERT_STATE_PATH = os.getenv("REGIME_ALERT_STATE_PATH", "regime_alert_state.json")
REGIME_ALERT_COOLDOWN_MIN = int(os.getenv("REGIME_ALERT_COOLDOWN_MIN", "120"))  # 2h
REGIME_CHOP_THRESHOLD = float(os.getenv("REGIME_CHOP_THRESHOLD", "6.8"))        # 0..10
REGIME_ALERT_ENABLED = os.getenv("REGIME_ALERT_ENABLED", "1").strip() != "0"


def _load_json(path: str, default: dict):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json_atomic(path: str, data: dict):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _wick_stats(o: float, h: float, l: float, c: float):
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = max(h - l, 1e-9)
    wick_total = max(upper + lower, 0.0)
    wick_body = wick_total / max(body, 1e-9)   # wick/body (rất lớn => nhiễu)
    wick_rng = wick_total / rng                # wick/range (0.6+ => wick chiếm ưu thế)
    return wick_body, wick_rng


def _avg_wickiness(candles, bars: int):
    cs = _m15_closed(candles)  # bỏ nến đang chạy (nếu có)
    if not cs or len(cs) < bars:
        return None
    w = cs[-bars:]
    wb, wr = [], []
    for x in w:
        o = _cget(x, "open"); h = _cget(x, "high"); l = _cget(x, "low"); c = _cget(x, "close")
        a, b = _wick_stats(o, h, l, c)
        wb.append(a); wr.append(b)
    return (sum(wb) / len(wb), sum(wr) / len(wr))


def _netmove_pct(candles, bars: int):
    cs = _m15_closed(candles)
    if not cs or len(cs) < bars:
        return None
    w = cs[-bars:]
    net = abs(_cget(w[-1], "close") - _cget(w[0], "open"))
    hi = max(_cget(x, "high") for x in w)
    lo = min(_cget(x, "low") for x in w)
    rng = max(hi - lo, 1e-9)
    return net / rng  # càng nhỏ => đi nhiều nhưng không tiến => chop


def _count_false_breaks(candles, lookback: int, level_lookback: int, eps: float = 0.0):
    cs = _m15_closed(candles)
    if not cs or len(cs) < level_lookback + 2:
        return 0

    n = len(cs)
    start = max(1, n - lookback)
    fb = 0

    for i in range(start, n):
        window = cs[max(0, i - level_lookback):i]
        hi_lvl = max(_cget(x, "high") for x in window)
        lo_lvl = min(_cget(x, "low") for x in window)

        o = _cget(cs[i], "open")
        h = _cget(cs[i], "high")
        l = _cget(cs[i], "low")
        c = _cget(cs[i], "close")

        # break high rồi đóng lại dưới level => false break up
        if h > hi_lvl + eps and c < hi_lvl:
            fb += 1
        # break low rồi đóng lại trên level => false break down
        if l < lo_lvl - eps and c > lo_lvl:
            fb += 1

    return fb


def score_chop_regime(m15, h1, h2=None):
    """
    Score 0..10: càng cao => càng giống chop/stop-hunt.
    Dùng dữ liệu ~2-3h đầu phiên cũng bắt được.
    """
    details = {}

    fb15 = _count_false_breaks(m15, lookback=24, level_lookback=48)
    fb1h = _count_false_breaks(h1,  lookback=12, level_lookback=36)
    details["false_breaks_15m"] = fb15
    details["false_breaks_1h"] = fb1h

    w15 = _avg_wickiness(m15, bars=12)  # ~3h
    if w15:
        wick_body, wick_rng = w15
    else:
        wick_body, wick_rng = None, None
    details["wick_body_avg_15m"] = wick_body
    details["wick_range_avg_15m"] = wick_rng

    nm15 = _netmove_pct(m15, bars=12)
    nm1h = _netmove_pct(h1,  bars=6)
    details["netmove_pct_15m"] = nm15
    details["netmove_pct_1h"] = nm1h

    nm2h = None
    wr2h = None
    if h2:
        w2 = _avg_wickiness(h2, bars=6)
        wr2h = w2[1] if w2 else None
        nm2h = _netmove_pct(h2, bars=4)
    details["wick_range_avg_2h"] = wr2h
    details["netmove_pct_2h"] = nm2h

    score = 0.0

    # False breaks là dấu hiệu “quét 2 đầu”
    score += min(fb15, 6) * 0.8   # max ~4.8
    score += min(fb1h, 4) * 0.7   # max ~2.8

    # Wick dominance
    if wick_rng is not None:
        if wick_rng >= 0.75: score += 2.2
        elif wick_rng >= 0.65: score += 1.6
        elif wick_rng >= 0.55: score += 1.0

    # Net move nhỏ => chop
    if nm15 is not None:
        if nm15 <= 0.22: score += 1.8
        elif nm15 <= 0.30: score += 1.2
    if nm1h is not None:
        if nm1h <= 0.25: score += 1.2
        elif nm1h <= 0.35: score += 0.8

    # 2H (nếu có) tăng độ chắc
    if nm2h is not None and wr2h is not None:
        if nm2h <= 0.30 and wr2h >= 0.60:
            score += 0.8

    score = max(0.0, min(10.0, score))
    return score, details


def maybe_send_regime_alert(symbol: str, m15, h1, h2=None, chat_id: Optional[str] = None):
    """
    Gửi cảnh báo nếu score >= threshold.
    Không phụ thuộc MIN_STARS.
    Có cooldown để không spam.
    """
    if not REGIME_ALERT_ENABLED:
        return False, 0.0, {}

    score, details = score_chop_regime(m15, h1, h2)

    st = _load_json(REGIME_ALERT_STATE_PATH, default={})
    now = int(time.time())
    key = f"{symbol}_CHOP"
    last_ts = int(st.get(key, 0))

    should_send = (score >= REGIME_CHOP_THRESHOLD) and (now - last_ts >= REGIME_ALERT_COOLDOWN_MIN * 60)
    if not should_send:
        return False, score, details

    msg = (
        f"⚠️ REGIME ALERT: CHOP / STOP-HUNT ({symbol})\n"
        f"Score: {score:.1f}/10 (>= {REGIME_CHOP_THRESHOLD})\n"
        f"- FalseBreak 15m: {details.get('false_breaks_15m')}\n"
        f"- FalseBreak 1h : {details.get('false_breaks_1h')}\n"
        f"- Wick(range)15m: {details.get('wick_range_avg_15m')}\n"
        f"- NetMove% 15m  : {details.get('netmove_pct_15m')}\n"
        f"⛔ Gợi ý: NO TRADE / đợi displacement thật + follow-through / tránh đặt SL ngay sau swing gần."
    )

    _send_telegram(msg, chat_id=chat_id or ADMIN_CHAT_ID)
    st[key] = now
    _save_json_atomic(REGIME_ALERT_STATE_PATH, st)

    return True, score, details

def _ingest_mt5_payload(payload: Dict[str, Any]) -> None:
    """
    payload shape from bridge:
      { "symbol": "...", "tf": "M15"/"M30"/"H1" (or "15"/"30"/"1h"), "candles": [ {ts/open/high/low/close/volume}, ... ] }

    Your app.data_source.ingest_mt5_candles signature varies by version.
    We'll support both:
      ingest_mt5_candles(payload)
      ingest_mt5_candles(symbol, tf, candles)
    """
    try:
        # try old signature (payload)
        ingest_mt5_candles(payload)  # type: ignore
        return
    except TypeError:
        pass

    symbol = (payload or {}).get("symbol")
    tf = (payload or {}).get("tf")
    candles = (payload or {}).get("candles")
    if not symbol or not tf or not isinstance(candles, list):
        raise ValueError("Invalid MT5 payload: require symbol/tf/candles")

    ingest_mt5_candles(symbol, tf, candles)  # type: ignore


@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"


# Accept MT5 push bridge
@app.post("/data/mt5", response_class=PlainTextResponse)
async def data_mt5(token: str = "", request: Request = None):
    secret = os.getenv("MT5_PUSH_SECRET", "")
    if not secret or token != secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    payload = await request.json()
    try:
        _ingest_mt5_payload(payload)
    except Exception as e:
        logger.exception("[MT5] ingest failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    return "OK"


# Telegram webhook handler
@app.post("/telegram/webhook", response_class=PlainTextResponse)
async def telegram_webhook(request: Request):
    try:
        update = await request.json()
    except Exception:
        return "OK"

    msg = update.get("message") or update.get("edited_message") or {}
    text = (msg.get("text") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")

    if not text:
        return "OK"

    # 0) ƯU TIÊN: Manual trade review (không cần "now")
    parsed = parse_manual_trade(text)
    if parsed:
        logger.info("[TG][REVIEW] matched manual trade: text=%r parsed=%s", text, parsed)
        try:
            reply = review_manual_trade(**parsed)
            if not reply:
                reply = "❌ REVIEW lỗi: hàm review không trả nội dung."
        except Exception as e:
            logger.exception("manual review failed: text=%r err=%s", text, e)
            reply = f"❌ REVIEW lỗi: {e}"
    
        try:
            _send_long_telegram(reply, chat_id=chat_id)
        except Exception as send_err:
            logger.exception(
                "send long telegram failed for manual review: chat_id=%s err=%s",
                chat_id, send_err
            )
            try:
                _send_telegram("❌ REVIEW lỗi: không gửi được tin nhắn dài. Xem logs.", chat_id=chat_id)
            except Exception:
                pass
    
        return "OK"

    low = text.lower()

    # 1) "now/scan" -> mới chạy analyze_pro
    if "now" in low or "scan" in low or "all" in low:
        if "scan" in low or "all" in low:
            symbols = DEFAULT_SYMBOLS or ["XAU/USD", "BTC/USD", "XAG/USD"]
        else:
            symbols = [_parse_symbol_from_text(text)]

        for sym in symbols:
            try:
                data = _fetch_triplet(sym, limit=260)
                # (optional) lấy thêm 2H cho regime radar (nếu get_candles hỗ trợ)
                try:
                    h2 = _as_list_from_get_candles(get_candles(sym, "2h", limit=220))
                except Exception:
                    h2 = []
                
                # ✅ Regime alert: independent of stars
                maybe_send_regime_alert(sym, data["m15"], data["h1"], h2=h2, chat_id=ADMIN_CHAT_ID)
                session = ""
                sig = analyze_pro(sym, data["m15"], data["m30"], data["h1"], data["h4"])
                # --- Guard: analyze_pro phải trả dict, nếu không thì fallback để khỏi crash
                if not isinstance(sig, dict):
                    sig = {
                        "symbol": sym,
                        "tf": "M30",
                        "session": session,
                        "recommendation": "CHỜ",
                        "stars": 1,
                        "trade_mode": "MANUAL",
                        "meta": {
                            "data_source": data.get("data_source") if isinstance(data, dict) else None
                        },
                        "context_lines": ["- Context: n/a"],
                        "liquidity_lines": ["- n/a"],
                        "quality_lines": ["- n/a"],
                        "note_lines": ["⚠️ analyze_pro returned None → fallback signal"],
                    }
                # attach data source for Telegram
                try:
                    ds = data.get("data_source")
                    if ds:
                        sig["data_source"] = ds
                        sig.setdefault("meta", {})["data_source"] = ds
                except Exception:
                    pass

                stars = int(sig.get("stars", 0) or 0)
                force_send = _force_send(sig)

                if force_send:
                    prefix = "🚨 CẢNH BÁO THANH KHOẢN / POST-SWEEP\\n\\n"
                    _send_telegram(prefix + format_signal(sig), chat_id=chat_id)
                elif stars < MIN_STARS:
                    # Manual 'NOW/SCAN': always send full analysis, but hide trade plan when under the star gate
                    #sig["show_trade_plan"] = False
                    prefix = f"⚠️ (Manual) Kèo dưới {MIN_STARS}⭐ – tham khảo thôi.\n\n"
                    _send_telegram(prefix + format_signal(sig), chat_id=chat_id)
                else:
                    _send_telegram(format_signal(sig), chat_id=chat_id)

            except Exception as e:
                logger.exception("analysis failed: %s", e)
                _send_telegram(f"❌ Analysis failed ({sym}): {e}", chat_id=chat_id)

    return "OK"



# Cron endpoint
@app.get("/cron/run", response_class=PlainTextResponse)
@app.head("/cron/run", response_class=PlainTextResponse)
async def cron_run(token: str = "", request: Request = None):
    global LAST_CRON_TS

    if CRON_SECRET and token != CRON_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    now = int(time.time())

    # cooldown: return 200 to avoid external retries storm
    if now - LAST_CRON_TS < MIN_CRON_GAP_SEC:
        return "OK cooldown"

    if CRON_LOCK.locked():
        return "OK overlap"

    async with CRON_LOCK:
        LAST_CRON_TS = int(time.time())
        client = getattr(getattr(request, "client", None), "host", "unknown") if request else "unknown"
        logger.info("[CRON] start from=%s", client)

        symbols = DEFAULT_SYMBOLS or ["XAU/USD", "BTC/USD", "XAG/USD"]

        for sym in symbols:
            try:
                data = _fetch_triplet(sym, limit=260)
                session = ""
                sig = analyze_pro(sym, data["m15"], data["m30"], data["h1"], data["h4"])
                # --- Guard: analyze_pro phải trả dict, nếu không thì fallback để khỏi crash
                if not isinstance(sig, dict):
                    sig = {
                        "symbol": sym,
                        "tf": "M30",
                        "session": session,
                        "recommendation": "CHỜ",
                        "stars": 1,
                        "trade_mode": "MANUAL",
                        "meta": {
                            "data_source": data.get("data_source") if isinstance(data, dict) else None
                        },
                        "context_lines": ["- Context: n/a"],
                        "liquidity_lines": ["- n/a"],
                        "quality_lines": ["- n/a"],
                        "note_lines": ["⚠️ analyze_pro returned None → fallback signal"],
                    }
                # attach data source for Telegram
                try:
                    ds = data.get("data_source")
                    if ds:
                        sig["data_source"] = ds
                        sig.setdefault("meta", {})["data_source"] = ds
                except Exception:
                    pass
                stars = int(sig.get("stars", 0) or 0)
                short_hint = sig.get("short_hint") or []
                entry = sig.get("entry")
                sl = sig.get("sl")
                tp1 = sig.get("tp1")
                rec = sig.get("recommendation", "")
                
                # ----- LUỒNG A: KÈO CHÍNH -----
                if stars >= MIN_STARS and rec != "CHỜ":
                    _send_telegram(format_signal(sig), chat_id=ADMIN_CHAT_ID)
                # ----- LUỒNG B (DISABLED): KÈO NGẮN HẠN / SCALE / SCALP -----
                # Đã tắt theo cấu hình chiến lược: chỉ gửi kèo theo scoring engine FULL/HALF.

                # ----- CÒN LẠI: KHÔNG GỬI -----
                else:
                    logger.info("[CRON] %s: only observation, no trade", sym)
            except Exception as e:
                logger.exception("[CRON] %s failed: %s", sym, e)

        return "OK"
