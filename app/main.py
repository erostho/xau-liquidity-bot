# app/main.py
from __future__ import annotations
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
DEFAULT_SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "XAU/USD,BTC/USD,XAG/USD").split(",") if s.strip()]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # default chat for cron
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", TELEGRAM_CHAT_ID)

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

        # split long message to avoid truncation
        chunks: List[str] = []
        buf = (text or "").strip()
        while len(buf) > TG_CHUNK:
            cut = buf.rfind("\n", 0, TG_CHUNK)
            if cut < 500:
                cut = TG_CHUNK
            chunks.append(buf[:cut].rstrip())
            buf = buf[cut:].lstrip()
        if buf:
            chunks.append(buf)

        for part in chunks:
            requests.post(url, json={"chat_id": cid, "text": part}, timeout=15)

    except Exception as e:
        logger.exception("[TG] send failed: %s", e)


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


def review_manual_trade(symbol: str, side: str, entry_lo: float, entry_hi: float, tp: float | None, sl: float | None) -> str:
    # 1) lấy candles
    m15, _ = get_candles(symbol, "15min", limit=200)
    m30, _ = get_candles(symbol, "30min", limit=200)
    h1,  _ = get_candles(symbol, "1h",    limit=200)


    sig = analyze_pro(symbol, m15, m30, h1)

    # 2) tính RR đơn giản
    entry = (entry_lo + entry_hi) / 2.0
    rr_txt = "RR: n/a"
    if tp is not None and sl is not None and abs(entry - sl) > 1e-9:
        r = abs(entry - sl)
        rew = abs(tp - entry)
        rr = rew / r
        rr_txt = f"RR≈{rr:.2f}"

    # 3) lấy context/liquidity/quality từ sig
    ctx = sig.get("context_lines", [])
    liq = sig.get("liquidity_lines", [])
    qlt = sig.get("quality_lines", [])
    notes = sig.get("notes", [])

    atr_line = next((x for x in qlt if "ATR(" in x), None)
    atr_val = None
    if atr_line:
        # ATR(14) M15: ~101.231
        try:
            atr_val = float(atr_line.split("~")[-1].strip())
        except:
            atr_val = None

    # 4) gợi ý hành động rất rõ ràng
    actions = []

    # rule cơ bản: nếu H1 ngược hướng thì cảnh báo
    h1_txt = " ".join(ctx).lower()
    if side == "BUY" and "h1: bearish" in h1_txt:
        actions.append("⚠️ BUY ngược H1 bearish → ưu tiên giảm rủi ro (chốt sớm/siết SL) hoặc chờ cấu trúc HL rồi re-entry.")
    if side == "SELL" and "h1: bullish" in h1_txt:
        actions.append("⚠️ SELL ngược H1 bullish → ưu tiên giảm rủi ro (chốt sớm/siết SL) hoặc chờ LH rồi re-entry.")

    # rule ATR: TP/SL có “thoáng” không
    if atr_val and sl is not None:
        dist_sl = abs(entry - sl)
        if dist_sl < 0.6 * atr_val:
            actions.append(f"⚠️ SL hơi sát (<0.6 ATR). Gợi ý: SL ≥ 0.9–1.2 ATR (đặc biệt nếu vừa có sweep).")
    if atr_val and tp is not None:
        dist_tp = abs(tp - entry)
        if dist_tp < 0.6 * atr_val:
            actions.append(f"⚠️ TP hơi ngắn (<0.6 ATR). Nếu muốn chắc: TP1 ~0.8–1.0 ATR, TP2 ~1.6–2.0 ATR.")

    # nếu có liquidity warning / sweep/spring trong sig notes/liq → nhắc “chờ cấu trúc”
    liq_txt = " ".join(liq).lower() + " " + " ".join(notes).lower()
    if "sweep" in liq_txt or "spring" in liq_txt or "liquidity" in liq_txt:
        actions.append("✅ Nếu vừa quét: CHỜ cấu trúc rồi mới add/giữ mạnh (BUY: M15 tạo HL + break đỉnh gần | SELL: M15 tạo LH + break đáy gần).")

    if not actions:
        actions.append("✅ Lệnh không thấy lỗi rõ ràng theo context hiện tại. Ưu tiên: TP1 chốt 30–50%, dời SL về BE khi đạt +0.8 ATR.")

    # 5) build reply
    lines = []
    lines.append("🧠 REVIEW LỆNH (Manual)")
    lines.append(f"📌 {symbol} | {side}")
    lines.append(f"- Entry: {entry_lo:.2f} – {entry_hi:.2f}")
    lines.append(f"- TP: {tp if tp is not None else '...'} | SL: {sl if sl is not None else '...'} | {rr_txt}")
    lines.append("")
    if ctx:
        lines.append("Context:")
        for s in ctx: lines.append(f"- {s}")
        lines.append("")
    lines.append("Liquidity:")
    for s in liq[:4]: lines.append(f"- {s}")
    lines.append("")
    lines.append("Gợi ý hành động:")
    for a in actions: lines.append(f"- {a}")

    return "\n".join(lines)

def _fetch_triplet(symbol: str, limit: int = 260) -> Dict[str, List[Any]]:
    # M15, M30, H1
    m15, _ = get_candles(symbol, "15min", limit)
    m30, _ = get_candles(symbol, "30min", limit)
    h1, _ = get_candles(symbol, "1h", limit)
    return {"m15": m15, "m30": m30, "h1": h1}

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
        reply = review_manual_trade(**parsed)
        _send_telegram(reply, chat_id=chat_id)
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
                sig = analyze_pro(sym, data["m15"], data["m30"], data["h1"])
                stars = int(sig.get("stars", 0) or 0)
                force_send = _force_send(sig)

                if force_send:
                    prefix = "🚨 CẢNH BÁO THANH KHOẢN / POST-SWEEP\n\n"
                    _send_telegram(prefix + format_signal(sig), chat_id=chat_id)
                elif stars < MIN_STARS:
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
                sig = analyze_pro(sym, data["m15"], data["m30"], data["h1"])
                if int(sig.get("stars", 1)) >= MIN_STARS:
                    _send_telegram(format_signal(sig), chat_id=ADMIN_CHAT_ID)
                    logger.info("[CRON] %s sent telegram stars=%s", sym, sig.get("stars"))
                else:
                    logger.info("[CRON] %s skip: stars=%s < %s", sym, sig.get("stars"), MIN_STARS)
            except Exception as e:
                logger.exception("[CRON] %s failed: %s", sym, e)

        return "OK"
