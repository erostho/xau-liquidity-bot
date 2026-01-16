# app/main.py
import os
import requests
from fastapi import FastAPI, Request, HTTPException
import logging
from app.pro_analysis import Candle, analyze_pro, format_signal
#from app.data_source import get_best_data_source
from app.data_source import get_data_source
import time
import asyncio
from app.data_source import get_candles, ingest_mt5_candles
from typing import Dict, Any, Optional, List
from app.data_source import get_candles
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
# =========================
# MT5 PUSH CACHE (Exness)
# =========================
MT5_CACHE: Dict[str, Dict[str, Any]] = {}  # key: "SYMBOL:TF" -> {"ts":..., "candles":[...]}
CRON_LOCK = asyncio.Lock()
LAST_CRON_TS = 0
MIN_CRON_GAP_SEC = int(os.getenv("MIN_CRON_GAP_SEC", "120"))  # 2 phút

# ===== Pending / Anti-flip state (in-memory) =====
# key: symbol -> pending dict
_PENDING: dict = {}

# cấu hình (tune nhanh)
PENDING_HOLD_CANDLES = int(os.getenv("PENDING_HOLD_CANDLES", "2"))   # giữ 1-2 nến M15
FLIP_OVERRIDE_STARS  = int(os.getenv("FLIP_OVERRIDE_STARS", "5"))    # chỉ cho đảo kèo nếu >= 5 sao
PENDING_EXPIRE_MIN   = int(os.getenv("PENDING_EXPIRE_MIN", "45"))    # quá 45' thì hủy pending

def _bias_from_rec(rec: str) -> str:
    r = (rec or "").upper()
    if "SELL" in r: return "SELL"
    if "BUY" in r: return "BUY"
    return "WAIT"

def _m15_closed_ts(candles) -> int:
    # candles: list[Candle] từ fetch
    if not candles:
        return 0
    return candles[-2].ts if len(candles) >= 2 else candles[-1].ts

def apply_pending_antiflip(symbol: str, candle_ts: int, sig: dict) -> tuple[dict, str]:
    """
    Returns: (sig_used, action)
      action:
        - "NEW_PENDING"  : tạo pending mới
        - "KEEP_PENDING" : giữ pending, ignore flip
        - "REPLACE"      : replace pending (flip override)
        - "EXPIRED"      : pending hết hạn -> xóa
        - "NONE"         : không có pending, dùng sig bình thường
    """
    now = int(time.time())
    bias = _bias_from_rec(sig.get("recommendation"))
    stars = int(sig.get("stars", 1))

    # nếu signal không phải BUY/SELL -> không tạo pending mới
    if bias not in ("BUY", "SELL"):
        # nếu có pending mà quá hạn -> xóa
        p = _PENDING.get(symbol)
        if p and (now - p.get("created_at", now)) > PENDING_EXPIRE_MIN * 60:
            _PENDING.pop(symbol, None)
            return sig, "EXPIRED"
        return sig, "NONE"

    p = _PENDING.get(symbol)

    # ===== Không có pending -> tạo pending mới =====
    if not p:
        _PENDING[symbol] = {
            "bias": bias,
            "created_at": now,
            "start_ts": candle_ts,
            "last_ts": candle_ts,
            "sig": sig,  # giữ toàn bộ entry/sl/tp/stars...
        }
        # thêm note để biết đang pending
        sig = dict(sig)
        sig.setdefault("notes", [])
        sig["notes"].insert(0, f"🧷 PENDING: giữ kèo {bias} trong {PENDING_HOLD_CANDLES} nến M15 (anti-flip).")
        return sig, "NEW_PENDING"

    # ===== Có pending -> kiểm tra hết hạn =====
    if (now - p.get("created_at", now)) > PENDING_EXPIRE_MIN * 60:
        _PENDING.pop(symbol, None)
        return sig, "EXPIRED"

    pending_bias = p["bias"]
    start_ts = p["start_ts"]

    # đang trong window giữ kèo 1-2 nến?
    in_hold_window = (candle_ts - start_ts) < PENDING_HOLD_CANDLES

    # ===== Nếu tín hiệu mới đảo kèo trong window -> CHẶN (trừ khi sao rất cao) =====
    if in_hold_window and bias != pending_bias and stars < FLIP_OVERRIDE_STARS:
        kept = dict(p["sig"])  # dùng lại pending signal cũ (entry/sl/tp) để khỏi flip
        kept.setdefault("notes", [])
        kept["notes"].insert(0, f"🛡️ Anti-flip: bỏ {bias} (stars={stars}) vì đang giữ kèo {pending_bias}.")
        p["last_ts"] = candle_ts
        _PENDING[symbol] = p
        return kept, "KEEP_PENDING"

    # ===== Nếu đảo kèo nhưng đủ mạnh -> replace pending =====
    if bias != pending_bias and stars >= FLIP_OVERRIDE_STARS:
        _PENDING[symbol] = {
            "bias": bias,
            "created_at": now,
            "start_ts": candle_ts,
            "last_ts": candle_ts,
            "sig": sig,
        }
        sig = dict(sig)
        sig.setdefault("notes", [])
        sig["notes"].insert(0, f"🔁 Flip OVERRIDE: đổi {pending_bias} → {bias} vì stars={stars} >= {FLIP_OVERRIDE_STARS}.")
        return sig, "REPLACE"

    # ===== Cùng bias -> refresh pending (update plan mới, kéo dài tuổi thọ) =====
    p["sig"] = sig
    p["last_ts"] = candle_ts
    _PENDING[symbol] = p

    sig = dict(sig)
    sig.setdefault("notes", [])
    sig["notes"].insert(0, f"🧷 PENDING: bias {bias} vẫn giữ (refresh).")
    return sig, "KEEP_PENDING"

@app.post("/data/mt5")
async def mt5_push(request: Request, token: str = ""):
    secret = os.getenv("MT5_PUSH_SECRET", "")
    if not secret or token != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    payload = await request.json()
    symbol = payload.get("symbol", "")
    tf = payload.get("tf", "")
    candles = payload.get("candles", [])

    n = ingest_mt5_candles(symbol, tf, candles)
    return {"ok": True, "stored": n}


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
def _get_mt5_cached_candles(symbol: str, tf: str, max_age_sec: int = 120) -> Optional[List[Candle]]:
    key = f"{symbol}:{tf}"
    item = MT5_CACHE.get(key)
    if not item:
        return None
    if time.time() - float(item.get("ts", 0)) > max_age_sec:
        return None

    out: List[Candle] = []
    for i, c in enumerate(item.get("candles", [])):
        out.append(Candle(
            ts=int(c.get("ts", i)),
            open=float(c["open"]),
            high=float(c["high"]),
            low=float(c["low"]),
            close=float(c["close"]),
            volume=float(c.get("volume", 0.0)),
        ))
    return out

@app.get("/cron/run")
async def cron_run(token: str = "", request: Request = None):
    global LAST_CRON_TS

    secret = os.getenv("CRON_SECRET", "")
    if not secret or token != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    now = int(time.time())

    # 1️⃣ Cooldown – cron bắn dồn thì skip, trả 200 OK
    if now - LAST_CRON_TS < MIN_CRON_GAP_SEC:
        return {"ok": True, "skipped": True, "reason": "cooldown"}

    # 2️⃣ Anti-overlap – nếu job cũ chưa xong thì skip
    if CRON_LOCK.locked():
        return {"ok": True, "skipped": True, "reason": "overlap"}

    async with CRON_LOCK:
        LAST_CRON_TS = int(time.time())

        # (optional) log để debug
        try:
            client = request.client.host if request and request.client else "unknown"
        except Exception:
            client = "unknown"
        logger.info(f"[CRON] start from={client}")

    results = []

    for item in SYMBOLS:
        symbol = item["name"]
        try:
            # 1) Lấy MT5 cache trước
            m15, src15 = get_candles(symbol, "15min", 220)
            h1,  srcH1 = get_candles(symbol, "1h", 220)
            
            # 2) Nếu MT5 chưa có thì fallback TwelveData
            if not m15 or not h1:
                m15 = fetch_twelvedata_candles(symbol, "15min", 220)
                h1  = fetch_twelvedata_candles(symbol, "1h", 220)
                source = "TWELVEDATA_FALLBACK"
            else:
                source = "EXNESS_MT5_PUSH"
            
            # 3) Phân tích
            sig = analyze_pro(symbol, m15, h1)
            ts = _m15_closed_ts(m15)
            sig, action = apply_pending_antiflip(symbol, ts, sig)
            stars = int(sig.get("stars", 0))
            
            # 4) Gắn nguồn cho message (an toàn)
            sig["source"] = f"{src15}/{srcH1}"  # nếu MT5 có thì sẽ là MT5/MT5, còn không thì có thể None/None
            notes = sig.get("notes") or []
            sig["notes"] = [f"Nguồn dữ liệu: {source}"] + notes
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
        #"sent": results
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

from fastapi import Body

MT5_PUSH_SECRET = os.getenv("MT5_PUSH_SECRET")

@app.post("/data/mt5")
async def receive_mt5_data(payload: dict = Body(...), token: str = ""):
    if token != MT5_PUSH_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")

    symbol = payload.get("symbol")
    tf = payload.get("tf")
    candles = payload.get("candles", [])

    if not symbol or not tf or not candles:
        raise HTTPException(status_code=400, detail="Invalid payload")

    # TODO: lưu candles vào cache / memory / file
    # Ví dụ đơn giản: log cho thấy đã nhận
    logger.info(f"[MT5] Received {len(candles)} candles {symbol} {tf}")

    return {"ok": True, "received": len(candles)}

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
            "🤖 *PRO Bot*\n"
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

        m15, src15 = get_candles(symbol, "15min", 220)
        h1,  srcH1 = get_candles(symbol, "1h", 220)

        sig = analyze_pro(symbol, m15, h1)
        ts = _m15_closed_ts(m15)
        sig, action = apply_pending_antiflip(symbol, ts, sig)
        sig["data_source"] = f"{src15}/{srcH1}"  # để format_signal show ra

        reply = format_signal(sig)
        send_telegram_long(chat_id, reply)

    except Exception as e:
        logger.exception("Analysis failed")
        send_telegram_long(chat_id, f"❌ Lỗi khi phân tích: `{str(e)}`")



    return {"ok": True}
