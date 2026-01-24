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

    return DEFAULT_SYMBOL[0] if DEFAULT_SYMBOLS else "XAU/USD"


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

    low = text.lower()

    # Commands:
    #   "xau now" / "btc now"
    #   "now" -> default symbol
    #   "scan" -> analyze BOTH symbols
    if "now" in low:
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
