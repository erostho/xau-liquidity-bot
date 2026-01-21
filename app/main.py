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

# Default symbols (can override by env SYMBOLS="XAU/USD,BTC/USD")
DEFAULT_SYMBOLS = os.getenv("SYMBOLS", "XAU/USD").split(",")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # default chat for cron
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", TELEGRAM_CHAT_ID)

CRON_SECRET = os.getenv("CRON_SECRET", "")

def _send_telegram(text: str, chat_id: Optional[str] = None) -> None:
    token = TELEGRAM_TOKEN
    cid = chat_id or TELEGRAM_CHAT_ID
    if not token or not cid:
        logger.info("[TG] missing TELEGRAM_TOKEN/CHAT_ID -> skip sending")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests = __import__("requests")
        requests.post(url, json={"chat_id": cid, "text": text}, timeout=12)
    except Exception as e:
        logger.exception("[TG] send failed: %s", e)

def _parse_symbol_from_text(text: str) -> str:
    t = (text or "").lower()
    if "xau" in t or "gold" in t:
        return "XAU/USD"
    if "btc" in t:
        return "BTC/USD"
    # fallback default
    return DEFAULT_SYMBOLS[0].strip() if DEFAULT_SYMBOLS else "XAU/USD"

def _fetch_triplet(symbol: str, limit: int = 260) -> Dict[str, List[Any]]:
    # M15, M30, H1
    m15, _ = get_candles(symbol, "15min", limit)
    m30, _ = get_candles(symbol, "30min", limit)
    h1, _  = get_candles(symbol, "1h", limit)
    return {"m15": m15, "m30": m30, "h1": h1}

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
    ingest_mt5_candles(payload)
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

    # Simple commands: "xau now" / "btc now"
    if "now" in text.lower():
        symbol = _parse_symbol_from_text(text)
        try:
            data = _fetch_triplet(symbol, limit=260)
            sig = analyze_pro(symbol, data["m15"], data["m30"], data["h1"])
            _send_telegram(format_signal(sig), chat_id=chat_id)
        except Exception as e:
            logger.exception("analysis failed: %s", e)
            _send_telegram(f"❌ Analysis failed: {e}", chat_id=chat_id)

    return "OK"

# Cron endpoint (used only if you still call via HTTP)
@app.get("/cron/run", response_class=PlainTextResponse)
@app.head("/cron/run", response_class=PlainTextResponse)
async def cron_run(token: str = "", request: Request = None):
    global LAST_CRON_TS

    if CRON_SECRET and token != CRON_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    now = int(time.time())

    # cooldown: return 200 instead of failing to avoid external retries storm
    if now - LAST_CRON_TS < MIN_CRON_GAP_SEC:
        return "OK cooldown"

    if CRON_LOCK.locked():
        return "OK overlap"

    async with CRON_LOCK:
        LAST_CRON_TS = int(time.time())
        client = getattr(getattr(request, "client", None), "host", "unknown") if request else "unknown"
        logger.info("[CRON] start from=%s", client)

        for sym in [s.strip() for s in DEFAULT_SYMBOLS if s.strip()]:
            try:
                data = _fetch_triplet(sym, limit=260)
                sig = analyze_pro(sym, data["m15"], data["m30"], data["h1"])
                # only send if stars >= MIN_STARS (default 3)
                min_stars = int(os.getenv("MIN_STARS", "3"))
                if int(sig.get("stars", 1)) >= min_stars:
                    _send_telegram(format_signal(sig), chat_id=ADMIN_CHAT_ID)
                    logger.info("[CRON] %s sent telegram stars=%s", sym, sig.get("stars"))
                else:
                    logger.info("[CRON] %s skip: stars=%s < %s", sym, sig.get("stars"), min_stars)
            except Exception as e:
                logger.exception("[CRON] %s failed: %s", sym, e)

        return "OK"
