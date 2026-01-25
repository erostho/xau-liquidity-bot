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
    # 1) lấy candles (đồng bộ với get_candles trả tuple)
    m15, _ = get_candles(symbol, "15min", limit=220)
    m30, _ = get_candles(symbol, "30min", limit=220)
    h1,  _ = get_candles(symbol, "1h",    limit=220)

    sig = analyze_pro(symbol, m15, m30, h1)

    # 2) Giá hiện tại
    try:
        cur = float(m15[-1]["close"]) if isinstance(m15[-1], dict) else float(getattr(m15[-1], "close"))
    except Exception:
        cur = None

    # 3) RR
    entry = (entry_lo + entry_hi) / 2.0
    rr_txt = "RR: n/a"
    rr = None
    if tp is not None and sl is not None and abs(entry - sl) > 1e-9:
        r = abs(entry - sl)
        rew = abs(tp - entry)
        rr = rew / r
        rr_txt = f"RR≈{rr:.2f}"

    # 4) lấy context/liquidity/quality/notes
    ctx = sig.get("context_lines", []) or []
    liq = sig.get("liquidity_lines", []) or []
    qlt = sig.get("quality_lines", []) or []
    notes = sig.get("notes", []) or []
    levels_info = sig.get("levels_info", []) or []

    # 5) ATR M15
    atr_val = None
    for x in qlt:
        if "ATR(" in x and "~" in x:
            try:
                atr_val = float(x.split("~")[-1].strip())
            except Exception:
                atr_val = None
            break

    # 6) các level quan trọng gần nhất (để invalidation)
    # ưu tiên swing gần/kháng cự hỗ trợ
    swing_hi = None
    swing_lo = None
    for price, label in levels_info:
        lb = (label or "").lower()
        if swing_hi is None and ("swing high" in lb or "kháng cự" in lb):
            swing_hi = float(price)
        if swing_lo is None and ("swing low" in lb or "hỗ trợ" in lb):
            swing_lo = float(price)
    # nếu thiếu thì lấy từ observation
    obs = sig.get("observation", {}) or {}
    if swing_hi is None and obs.get("buy") is not None:
        swing_hi = float(obs.get("buy"))
    if swing_lo is None and obs.get("sell") is not None:
        swing_lo = float(obs.get("sell"))

    buf = 0.10 * atr_val if atr_val else 0.0

    # 7) Tính trạng thái lời/lỗ, khoảng cách TP/SL
    pnl_txt = ""
    dist_tp = dist_sl = None
    if cur is not None:
        if side == "BUY":
            pnl = cur - entry
            pnl_txt = f"P/L (ước tính): {pnl:+.2f} điểm"
            if tp is not None: dist_tp = tp - cur
            if sl is not None: dist_sl = cur - sl
        else:
            pnl = entry - cur
            pnl_txt = f"P/L (ước tính): {pnl:+.2f} điểm"
            if tp is not None: dist_tp = cur - tp
            if sl is not None: dist_sl = sl - cur

    # 8) Quyết định GIỮ/THOÁT/CHỈNH (rõ ràng)
    decisions = []

    # 8.1) Invalidation theo swing + buffer
    if cur is not None:
        if side == "SELL" and swing_hi is not None and cur > (swing_hi + buf):
            decisions.append("🛑 KÈO SAI (invalidation): giá vượt vùng swing/high + buffer → ưu tiên THOÁT hoặc giảm mạnh rủi ro.")
        if side == "BUY" and swing_lo is not None and cur < (swing_lo - buf):
            decisions.append("🛑 KÈO SAI (invalidation): giá thủng vùng swing/low + buffer → ưu tiên THOÁT hoặc giảm mạnh rủi ro.")

    # 8.2) H1 ngược hướng -> giảm rủi ro
    h1_txt = " ".join(ctx).lower()
    if side == "BUY" and "h1: bearish" in h1_txt:
        decisions.append("⚠️ BUY ngược H1 bearish → ưu tiên chốt sớm / dời SL chặt / không gồng.")
    if side == "SELL" and "h1: bullish" in h1_txt:
        decisions.append("⚠️ SELL ngược H1 bullish → ưu tiên chốt sớm / dời SL chặt / không gồng.")

    # 8.3) TP/SL quá sát so với ATR
    if atr_val and sl is not None:
        dist = abs(entry - sl)
        if dist < 0.6 * atr_val:
            decisions.append(f"⚠️ SL đang sát (<0.6 ATR). Gợi ý SL ≥ 0.9–1.2 ATR (hoặc đặt sau vùng sweep).")
    if atr_val and tp is not None:
        dist = abs(tp - entry)
        if dist < 0.6 * atr_val:
            decisions.append("⚠️ TP đang ngắn (<0.6 ATR). Gợi ý TP1 ~0.8–1.0 ATR, TP2 ~1.6–2.0 ATR.")

    # 8.4) Quản trị lệnh theo ATR (TP1/BE)
    if atr_val and cur is not None:
        move = (cur - entry) if side == "BUY" else (entry - cur)
        if move >= 0.8 * atr_val:
            decisions.append("✅ Đã đi được ~0.8 ATR: chốt TP1 30–50% + dời SL về BE (hoặc BE+0.1 ATR).")
        elif move <= -0.6 * atr_val:
            decisions.append("⚠️ Đang ngược ~0.6 ATR: nếu không có lý do giữ (HTF/structure) → cân nhắc cắt sớm để tránh hit SL.")

    # 8.5) Fix lỗi “nếu vừa quét” (chỉ nói khi THẬT SỰ có sweep/spring ok)
    liq_join = " ".join(liq).lower()
    has_real_sweep = ("sweep high" in liq_join) or ("sweep low" in liq_join) or ("spring" in liq_join) or ("upthrust" in liq_join)
    if has_real_sweep:
        decisions.append("🧱 Sau quét: CHỜ CẤU TRÚC rồi mới add/giữ mạnh (BUY: M15 tạo HL + break đỉnh gần | SELL: M15 tạo LH + break đáy gần).")

    if not decisions:
        decisions.append("✅ Chưa thấy tín hiệu ‘sai kèo’ rõ ràng → ưu tiên GIỮ theo plan, chia TP1/TP2 và dời SL về BE khi có lợi nhuận.")

    # 9) Build reply gọn mà “ra quyết định”
    lines = []
    lines.append("🧠 REVIEW LỆNH (Manual)")
    lines.append(f"📌 {symbol} | {side}")
    lines.append(f"- Entry: {entry_lo:.2f} – {entry_hi:.2f}")
    lines.append(f"- TP: {tp if tp is not None else '...'} | SL: {sl if sl is not None else '...'} | {rr_txt}")
    if cur is not None:
        lines.append(f"- Giá hiện tại (M15 close): {cur:.2f}")
    if pnl_txt:
        lines.append(f"- {pnl_txt}")
    lines.append("")

    if ctx:
        lines.append("Context:")
        for s in ctx:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("Liquidity (tóm tắt):")
    for s in liq[:3]:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("✅ Kết luận / Gợi ý hành động:")
    for d in decisions[:6]:
        lines.append(f"- {d}")

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
