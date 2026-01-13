# app/main.py
import os
import requests
from fastapi import FastAPI, Request, HTTPException
import logging

from app.pro_analysis import Candle, analyze_pro, format_signal

logger = logging.getLogger("uvicorn.error")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
SYMBOL = os.getenv("SYMBOL", "XAU/USD")

if not TELEGRAM_TOKEN:
    logger.warning("Missing TELEGRAM_BOT_TOKEN")
if not TWELVEDATA_API_KEY:
    logger.warning("Missing TWELVEDATA_API_KEY")

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}


def send_telegram(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code >= 400:
        logger.error(f"sendMessage failed {r.status_code}: {r.text}")

def fetch_twelvedata_candles(symbol: str, interval: str, outputsize: int = 200):
    # TwelveData time_series
    # interval: "15min", "1h"
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"No candle data from TwelveData for {symbol} {interval}")

    # TwelveData returns newest first -> reverse
    values = list(reversed(values))

    candles = []
    # No timestamp numeric in free sometimes; we can just use index
    for i, v in enumerate(values):
        o = float(v["open"]); h = float(v["high"]); l = float(v["low"]); c = float(v["close"])
        candles.append(
            Candle(
                ts=i,
                open=o,
                high=h,
                low=l,
                close=c
            )
        )

    return candles

def should_analyze(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = ["xau", "gold", "buy", "sell", "now", "scalp", "trend", "tp", "sl"]
    return any(k in t for k in keywords)

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"ok": True}
    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    logger.info(f"✅ MSG chat_id={chat_id} text={text}")


    # quick help
    if text.strip().lower() in ["/start", "help", "/help"]:
        send_telegram(chat_id,
            "🤖 *XAU PRO Bot*\n"
            "Gõ: `XAU now` hoặc `SELL hay BUY?`\n"
            "Bot sẽ trả: Bias + ⭐ + Entry/TP/SL + lý do.\n"
        )
        return {"ok": True}

    if not should_analyze(text):
        send_telegram(chat_id, "Gõ `XAU now` hoặc `SELL hay BUY?` để mình phân tích *PRO* nhé.")
        return {"ok": True}

    # Acknowledge quickly (optional)
    send_telegram(chat_id, "⏳ Đang phân tích dữ liệu...")

    try:
        m15 = fetch_twelvedata_candles(SYMBOL, "15min", 220)
        h1 = fetch_twelvedata_candles(SYMBOL, "1h", 220)

        sig = analyze_pro(SYMBOL, m15, h1)
        reply = format_signal(sig)
        send_telegram(chat_id, reply)

    except Exception as e:
        logger.exception("Analysis failed")
        send_telegram(chat_id, f"❌ Lỗi khi phân tích: `{str(e)}`\nKiểm tra `TWELVEDATA_API_KEY` và SYMBOL.")

    return {"ok": True}
