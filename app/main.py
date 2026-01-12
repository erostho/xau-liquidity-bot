from __future__ import annotations
from fastapi import FastAPI, Request, Header, HTTPException
import requests

from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_SECRET_TOKEN
from .providers.factory import get_provider
from .analysis.pro_analyzer import make_plan, format_plan
from .telegram.parser import parse_text

app = FastAPI()

def _send_message(chat_id: int, text: str):
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()

@app.get("/health")
def health():
    return {"ok": True}

import logging
logger = logging.getLogger("uvicorn.error")

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        logger.exception("❌ Cannot parse telegram JSON")
        return {"ok": True}

    logger.info(f"✅ TELEGRAM HIT: keys={list(data.keys())}")

    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    logger.info(f"✅ MSG chat_id={chat_id} text={text}")

    return {"ok": True}



    update = await request.json()
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = (message.get("chat") or {}).get("id")
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return {"ok": True}

    lower = text.lower()
    if not any(k in lower for k in ["nên", "nen", "buy", "sell", "phân tích", "phan tich", "analyze", "market", "xau"]):
        return {"ok": True}

    q = parse_text(text)
    provider = get_provider()
    candles = provider.fetch_ohlcv(q.symbol, q.timeframe, limit=220)

    plan = make_plan(candles, tf=q.timeframe, risk_usd=q.risk_usd, symbol=q.symbol)
    reply = format_plan(plan, risk_usd=q.risk_usd)

    _send_message(chat_id, reply)
    return {"ok": True}
