from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.data_source import get_candles, ingest_mt5_candles
from app.pro_analysis import analyze_pro, format_signal, render_review_text, review_manual_order

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="xau-liquidity-bot")

CRON_LOCK = asyncio.Lock()
LAST_CRON_TS = 0
MIN_CRON_GAP_SEC = int(os.getenv("MIN_CRON_GAP_SEC", "25"))
DEFAULT_SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "XAU/USD,BTC/USD").split(",") if s.strip()]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", TELEGRAM_CHAT_ID)
CRON_SECRET = os.getenv("CRON_SECRET", "")
MIN_STARS = int(os.getenv("MIN_STARS", "1"))
TG_CHUNK = int(os.getenv("TG_CHUNK", "3500"))
REGIME_ALERT_ENABLED = os.getenv("REGIME_ALERT_ENABLED", "1").strip() != "0"
REGIME_CHOP_THRESHOLD = float(os.getenv("REGIME_CHOP_THRESHOLD", "6.8"))
REGIME_ALERT_COOLDOWN_MIN = int(os.getenv("REGIME_ALERT_COOLDOWN_MIN", "120"))
REGIME_ALERT_STATE_PATH = os.getenv("REGIME_ALERT_STATE_PATH", "regime_alert_state.json")


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


def _send_long_telegram(text: str, chat_id: str, chunk_size: int = 3500) -> None:
    text = str(text or "")
    if not text.strip():
        return

    parts: List[str] = []
    buf = ""
    for line in text.splitlines(True):
        if len(buf) + len(line) <= chunk_size:
            buf += line
        else:
            if buf:
                parts.append(buf)
            if len(line) <= chunk_size:
                buf = line
            else:
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
    t = (text or "").lower()
    if "xag" in t or "silver" in t:
        return "XAG/USD"
    if "xau" in t or "gold" in t:
        return "XAU/USD"
    if "btc" in t:
        return "BTC/USD"
    return DEFAULT_SYMBOLS[0] if DEFAULT_SYMBOLS else "XAU/USD"


def normalize_symbol(user_sym: str) -> str:
    s = (user_sym or "").strip().upper().replace("-", "/").replace("_", "/").replace(" ", "")
    if s in ("BTC", "BTCUSD", "BTC/USD", "BTCUSDT", "BTC/USDT"):
        return "BTC/USD"
    if s in ("XAU", "XAUUSD", "XAU/USD"):
        return "XAU/USD"
    if s in ("XAG", "XAGUSD", "XAG/USD"):
        return "XAG/USD"
    if s.endswith("/USDT"):
        s = s.replace("/USDT", "/USD")
    if s.endswith("USDT"):
        s = s.replace("USDT", "/USD")
    if "/" not in s and s.endswith("USD"):
        s = s[:-3] + "/USD"
    return s


_MANUAL_RE = re.compile(
    r"^\s*(?P<sym>[A-Za-z\/\-_]+)\s+"
    r"(?P<side>BUY|SELL)\s+"
    r"(?P<entry1>\d+(\.\d+)?)"
    r"(?:\s*[-~]\s*(?P<entry2>\d+(\.\d+)?))?"
    r"(?:\s+(?:TP|TP1|TARGET)\s*[:=\s]+(?P<tp>\d+(\.\d+)?))?"
    r"(?:\s+(?:SL|STOP)\s*[:=\s]+(?P<sl>\d+(\.\d+)?))?"
    r"\s*$",
    re.IGNORECASE,
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
    if isinstance(res, tuple) and len(res) >= 1:
        return res[0] or []
    return res or []


def _as_list_and_source_from_get_candles(res):
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
    if not candles:
        return []
    return candles[:-1] if len(candles) > 1 else candles


def _wick_stats(o: float, h: float, l: float, c: float):
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    rng = max(h - l, 1e-9)
    wick_total = max(upper + lower, 0.0)
    return wick_total / max(body, 1e-9), wick_total / rng


def _avg_wickiness(candles, bars: int):
    cs = _m15_closed(candles)
    if not cs or len(cs) < bars:
        return None
    w = cs[-bars:]
    wb, wr = [], []
    for x in w:
        o = _cget(x, "open")
        h = _cget(x, "high")
        l = _cget(x, "low")
        c = _cget(x, "close")
        a, b = _wick_stats(o, h, l, c)
        wb.append(a)
        wr.append(b)
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
    return net / rng


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
        h = _cget(cs[i], "high")
        l = _cget(cs[i], "low")
        c = _cget(cs[i], "close")
        if h > hi_lvl + eps and c < hi_lvl:
            fb += 1
        if l < lo_lvl - eps and c > lo_lvl:
            fb += 1
    return fb


def score_chop_regime(m15, h1, h2=None):
    details = {}
    fb15 = _count_false_breaks(m15, lookback=24, level_lookback=48)
    fb1h = _count_false_breaks(h1, lookback=12, level_lookback=36)
    details["false_breaks_15m"] = fb15
    details["false_breaks_1h"] = fb1h
    w15 = _avg_wickiness(m15, bars=12)
    wick_rng = w15[1] if w15 else None
    details["wick_range_avg_15m"] = wick_rng
    nm15 = _netmove_pct(m15, bars=12)
    nm1h = _netmove_pct(h1, bars=6)
    details["netmove_pct_15m"] = nm15
    details["netmove_pct_1h"] = nm1h
    score = 0.0
    score += min(fb15, 6) * 0.8
    score += min(fb1h, 4) * 0.7
    if wick_rng is not None:
        if wick_rng >= 0.75:
            score += 2.2
        elif wick_rng >= 0.65:
            score += 1.6
        elif wick_rng >= 0.55:
            score += 1.0
    if nm15 is not None:
        if nm15 <= 0.22:
            score += 1.8
        elif nm15 <= 0.30:
            score += 1.2
    if nm1h is not None:
        if nm1h <= 0.25:
            score += 1.2
        elif nm1h <= 0.35:
            score += 0.8
    return max(0.0, min(10.0, score)), details


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


def maybe_send_regime_alert(symbol: str, m15, h1, h2=None, chat_id: Optional[str] = None):
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
        f"⚠️ CẢNH BÁO THỊ TRƯỜNG NHIỄU ({symbol})\n"
        f"Điểm nhiễu: {score:.1f}/10\n"
        f"- Quét giả M15: {details.get('false_breaks_15m')}\n"
        f"- Quét giả H1: {details.get('false_breaks_1h')}\n"
        f"- Wick/range M15: {details.get('wick_range_avg_15m')}\n"
        f"- Độ tiến của giá M15: {details.get('netmove_pct_15m')}\n"
        f"⛔ Gợi ý: tạm đứng ngoài, chờ cú chạy thật rồi mới theo."
    )
    _send_telegram(msg, chat_id=chat_id or ADMIN_CHAT_ID)
    st[key] = now
    _save_json_atomic(REGIME_ALERT_STATE_PATH, st)
    return True, score, details


def _fetch_triplet(symbol: str, limit: int = 260) -> Dict[str, Any]:
    sym = normalize_symbol(symbol)
    m15, src15 = _as_list_and_source_from_get_candles(get_candles(sym, "15min", limit=limit))
    m30, src30 = _as_list_and_source_from_get_candles(get_candles(sym, "30min", limit=limit))
    h1, src1h = _as_list_and_source_from_get_candles(get_candles(sym, "1h", limit=limit))
    h4, src4h = _as_list_and_source_from_get_candles(get_candles(sym, "4h", limit=limit))
    return {"m15": m15, "m30": m30, "h1": h1, "h4": h4, "data_source": (src30 or src15 or src1h or src4h)}


def _force_send(sig: dict) -> bool:
    ctx = " | ".join(sig.get("context_lines", []) or []).lower()
    notes = " | ".join(sig.get("notes", []) or []).lower()
    return (
        "quét thanh khoản" in ctx
        or "quét thanh khoản" in notes
        or "mở cửa mất cân bằng" in ctx
        or "mở cửa mất cân bằng" in notes
        or "liquidity warning" in ctx
        or "post-sweep" in ctx
    )


def _ingest_mt5_payload(payload: Dict[str, Any]) -> None:
    try:
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


def review_manual_trade(symbol: str, side: str, entry_lo: float, entry_hi: float, tp: float | None, sl: float | None) -> str:
    symbol = str(symbol or "").strip().upper()
    side = (side or "").upper().strip()
    try:
        entry = (float(entry_lo) + float(entry_hi)) / 2.0
        tp_val = float(tp) if tp is not None else entry
        sl_val = float(sl) if sl is not None else entry
    except Exception as e:
        logger.exception("Invalid entry values for manual review %s %s: %s", symbol, side, e)
        return f"❌ REVIEW lỗi cho {symbol}: entry không hợp lệ."

    try:
        data = _fetch_triplet(symbol, limit=220)
        analysis = analyze_pro(symbol, data["m15"], data["m30"], data["h1"], data["h4"])
        if not isinstance(analysis, dict):
            return f"❌ REVIEW lỗi cho {symbol}: bot không trả đủ dữ liệu phân tích."
        ds = data.get("data_source")
        if ds:
            analysis["data_source"] = ds
        review = review_manual_order(symbol, side, entry, sl_val, tp_val, analysis)
        return render_review_text(review)
    except Exception as e:
        logger.exception("manual review failed for %s: %s", symbol, e)
        return f"❌ REVIEW lỗi cho {symbol}: {e}"


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

    parsed = parse_manual_trade(text)
    if parsed:
        reply = review_manual_trade(**parsed)
        _send_long_telegram(reply, chat_id=chat_id)
        return "OK"

    low = text.lower()
    if "now" in low or "scan" in low or "all" in low:
        symbols = (DEFAULT_SYMBOLS or ["XAU/USD", "BTC/USD", "XAG/USD"]) if ("scan" in low or "all" in low) else [_parse_symbol_from_text(text)]
        for sym in symbols:
            try:
                data = _fetch_triplet(sym, limit=260)
                try:
                    h2 = _as_list_from_get_candles(get_candles(sym, "2h", limit=220))
                except Exception:
                    h2 = []
                maybe_send_regime_alert(sym, data["m15"], data["h1"], h2=h2, chat_id=ADMIN_CHAT_ID)
                sig = analyze_pro(sym, data["m15"], data["m30"], data["h1"], data["h4"])
                if not isinstance(sig, dict):
                    sig = {"symbol": sym, "recommendation": "CHỜ", "stars": 1, "context_lines": ["Bot chưa trả đủ dữ liệu"], "liquidity_lines": [], "quality_lines": [], "notes": [], "meta": {}}
                ds = data.get("data_source")
                if ds:
                    sig["data_source"] = ds
                    sig.setdefault("meta", {})["data_source"] = ds
                stars = int(sig.get("stars", 0) or 0)
                if _force_send(sig):
                    _send_telegram("🚨 CẢNH BÁO THỊ TRƯỜNG\n\n" + format_signal(sig), chat_id=chat_id)
                elif stars < MIN_STARS:
                    _send_telegram(f"⚠️ Kèo dưới {MIN_STARS}⭐ – chỉ để tham khảo.\n\n" + format_signal(sig), chat_id=chat_id)
                else:
                    _send_telegram(format_signal(sig), chat_id=chat_id)
            except Exception as e:
                logger.exception("analysis failed: %s", e)
                _send_telegram(f"❌ Analysis failed ({sym}): {e}", chat_id=chat_id)
    return "OK"


@app.get("/cron/run", response_class=PlainTextResponse)
@app.head("/cron/run", response_class=PlainTextResponse)
async def cron_run(token: str = "", request: Request = None):
    global LAST_CRON_TS
    if CRON_SECRET and token != CRON_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    now = int(time.time())
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
                sig = analyze_pro(sym, data["m15"], data["m30"], data["h1"], data["h4"])
                if not isinstance(sig, dict):
                    continue
                ds = data.get("data_source")
                if ds:
                    sig["data_source"] = ds
                    sig.setdefault("meta", {})["data_source"] = ds
                stars = int(sig.get("stars", 0) or 0)
                rec = str(sig.get("recommendation") or "CHỜ")
                if stars >= MIN_STARS and rec != "CHỜ":
                    _send_telegram(format_signal(sig), chat_id=ADMIN_CHAT_ID)
                else:
                    logger.info("[CRON] %s: only observation, no trade", sym)
            except Exception as e:
                logger.exception("[CRON] %s failed: %s", sym, e)
        return "OK"
