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

SYMBOLS = [
    {"name": "XAU/USD", "tf": "15min"},
    {"name": "BTC/USD", "tf": "15min"},
]
# Map user text -> symbol TwelveData
SYMBOL_MAP = {
    "xau": os.getenv("SYMBOL_XAU", "XAU/USD"),
    "gold": os.getenv("SYMBOL_XAU", "XAU/USD"),
    "btc": os.getenv("SYMBOL_BTC", "BTC/USD"),
    "bitcoin": os.getenv("SYMBOL_BTC", "BTC/USD"),
}

def pick_symbol_from_text(text: str) -> str:
    t = (text or "").strip().lower()
    for k, sym in SYMBOL_MAP.items():
        if k in t:
            return sym
    # mặc định nếu user không nói rõ: dùng XAU (hoặc SYMBOL env cũ)
    return os.getenv("SYMBOL_DEFAULT", SYMBOL)
def detect_symbol_from_text(text: str) -> str:
    t = (text or "").strip().lower()
    # ưu tiên BTC nếu có chữ btc
    if "btc" in t:
        return "BTC/USD"      # hoặc "BTC/USDT" tùy TwelveData bạn dùng cái nào chạy được
    if "xau" in t or "gold" in t:
        return "XAU/USD"
    return SYMBOL  # fallback default

MIN_STARS = 3
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
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    r = requests.post(url, json=payload, timeout=10)
    if r.status_code != 200:
        logger.error(f"[TG] failed {r.status_code}: {r.text}")


def send_telegram_long(chat_id: int, text: str, max_len: int = 3800):
    """
    Telegram limit is 4096 chars. Use 3800 for safety (markdown, emojis, etc).
    Split by lines to keep formatting nice.
    """
    if not text:
        return

    # normalize
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if len(text) <= max_len:
        return send_telegram(chat_id, text)

    lines = text.split("\n")
    chunks = []
    cur = ""

    for line in lines:
        # If one line is insanely long, hard-split it
        if len(line) > max_len:
            # flush current
            if cur:
                chunks.append(cur.rstrip())
                cur = ""
            # split that line
            start = 0
            while start < len(line):
                chunks.append(line[start:start + max_len])
                start += max_len
            continue

        # normal line append
        candidate = (cur + "\n" + line) if cur else line
        if len(candidate) > max_len:
            chunks.append(cur.rstrip())
            cur = line
        else:
            cur = candidate

    if cur:
        chunks.append(cur.rstrip())

    # send chunks with small headers
    total = len(chunks)
    for i, part in enumerate(chunks, start=1):
        prefix = f"({i}/{total})\n" if total > 1 else ""
        send_telegram(chat_id, prefix + part)

@app.get("/cron/run")
async def cron_run(token: str = ""):
    secret = os.getenv("CRON_SECRET", "")
    if not secret or token != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    admin_chat_id = int(os.getenv("ADMIN_CHAT_ID", "0"))
    if admin_chat_id == 0:
        raise HTTPException(status_code=400, detail="Missing ADMIN_CHAT_ID")

    logger.info("[CRON] multi-symbol scan started")

    results = []

    for item in SYMBOLS:
        symbol = item["name"]
        try:
            m15 = fetch_twelvedata_candles(symbol, "15min", 220)
            h1  = fetch_twelvedata_candles(symbol, "1h", 220)
            sig = analyze_pro(symbol, m15, h1)
            stars = int(sig.get("stars", 0))
            if stars < MIN_STARS:
                logger.info(f"[CRON] {symbol} skip: stars={stars} < {MIN_STARS}")
                continue
    
            msg = format_signal(sig)
            send_telegram_long(admin_chat_id, msg)
            logger.info(f"[CRON] {symbol} sent telegram stars={stars}")
    
            results.append({
                "symbol": symbol,
                "stars": stars
            })
    
        except Exception as e:
            logger.exception(f"[CRON] {symbol} failed")
            send_telegram_long(admin_chat_id, f"❌ {symbol} cron error:\n`{str(e)}`")


    return {
        "ok": True,
        "sent": results
    }

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
    keywords = ["xau", "gold", "btc", "bitcoin", "buy", "sell", "now", "scalp", "trend", "tp", "sl"]
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
    send_telegram(chat_id, f"⏳ Đang phân tích..")

    try:
        symbol = detect_symbol_from_text(text)
        #symbol = pick_symbol_from_text(text)
        m15 = fetch_twelvedata_candles(SYMBOL, "15min", 220)
        h1 = fetch_twelvedata_candles(SYMBOL, "1h", 220)
        sig = analyze_pro(SYMBOL, m15, h1)
        reply = format_signal(sig)
        send_telegram_long(chat_id, reply)

    except Exception as e:
        logger.exception("Analysis failed")
        send_telegram_long(chat_id, f"❌ Lỗi khi phân tích: `{str(e)}`\nKiểm tra `TWELVEDATA_API_KEY` và SYMBOL.")

    return {"ok": True}
