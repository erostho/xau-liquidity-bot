from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math


# ============================================================
# PRO ANALYSIS V4.1
# - Giữ logic cũ theo 4 lớp: xu hướng / vị trí / thanh khoản / xác nhận
# - Chỉ thêm GAP Context V5
# - Chuẩn hoá đầu ra sang tiếng Việt dễ hiểu hơn
# - Fix lỗi bias_side chưa được khởi tạo
# ============================================================


# -----------------------------
# Helpers an toàn
# -----------------------------


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default



def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))



def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0



def mean(values: List[float]) -> float:
    vals = [safe_float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# -----------------------------
# Cấu trúc dữ liệu
# -----------------------------


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    time: Optional[str] = None

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(0.0, self.high - self.low)


@dataclass
class GapContext:
    gap_size: float
    gap_pct_of_atr: float
    open_range: float
    open_range_pct_of_atr: float
    has_big_gap: bool
    has_wide_open_range: bool
    unbalanced_open: bool
    note: str


# -----------------------------
# Chuẩn hoá từ ngữ đầu ra
# -----------------------------


TERM_MAP = {
    "BUY": "MUA",
    "SELL": "BÁN",
    "SIDEWAY": "ĐI NGANG",
    "TREND": "Xu hướng rõ",
    "SIDEWAY_STATE": "Đi ngang",
    "EXTREME": "Biến động quá mạnh",
    "EXHAUSTION": "Đà yếu dần",
    "TRANSITION": "Đang chuyển trạng thái",
    "POST_LIQUIDATION_BOUNCE": "Sau nhịp quét mạnh, giá đang hồi",
    "BOUNCE_TO_SELL": "Hồi lên để canh bán",
    "BOUNCE_TO_BUY": "Hồi xuống để canh mua",
    "REVERSAL_CANDIDATE": "Có khả năng đổi hướng",
    "NO_TRADE": "Nên đứng ngoài",
    "OUTFLOW": "Dòng tiền đang rút ra",
    "INFLOW": "Dòng tiền đang vào",
    "NEUTRAL": "Trung tính",
}


# -----------------------------
# Chuyển dữ liệu đầu vào
# -----------------------------



def _to_candle(item: Any) -> Candle:
    if isinstance(item, Candle):
        return item
    if isinstance(item, dict):
        return Candle(
            open=safe_float(item.get("open")),
            high=safe_float(item.get("high")),
            low=safe_float(item.get("low")),
            close=safe_float(item.get("close")),
            volume=safe_float(item.get("volume", 0.0)),
            time=item.get("time"),
        )
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return Candle(
            open=safe_float(item[0]),
            high=safe_float(item[1]),
            low=safe_float(item[2]),
            close=safe_float(item[3]),
            volume=safe_float(item[4]) if len(item) > 4 else 0.0,
            time=item[5] if len(item) > 5 else None,
        )
    raise ValueError(f"Không đọc được candle: {item!r}")



def normalize_candles(data: Optional[List[Any]]) -> List[Candle]:
    if not data:
        return []
    out: List[Candle] = []
    for item in data:
        try:
            out.append(_to_candle(item))
        except Exception:
            continue
    return out


# -----------------------------
# Chỉ báo cơ bản
# -----------------------------



def calc_true_range(curr: Candle, prev_close: Optional[float]) -> float:
    if prev_close is None:
        return curr.range
    return max(curr.high - curr.low, abs(curr.high - prev_close), abs(curr.low - prev_close))



def calc_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs: List[float] = []
    prev_close: Optional[float] = None
    for c in candles[-(period + 2):]:
        trs.append(calc_true_range(c, prev_close))
        prev_close = c.close
    return mean(trs[-period:]) if trs else 0.0



def calc_rsi(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 50.0
    closes = [c.close for c in candles[-(period + 1):]]
    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))



def sma(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    return mean(values[-period:])


# -----------------------------
# Logic cũ - 4 lớp
# -----------------------------



def detect_bias(h4: List[Candle], h1: List[Candle]) -> Tuple[str, str, str]:
    # FIX: luôn gán mặc định trước để tránh lỗi bias_side
    bias_side = "SIDEWAY"
    bias_text = "đi ngang"
    bias_detail = "Khung lớn chưa rõ hướng"

    def slope(candles: List[Candle], lookback: int = 8) -> float:
        if len(candles) < lookback:
            return 0.0
        return candles[-1].close - candles[-lookback].close

    h4_slope = slope(h4)
    h1_slope = slope(h1)

    if h4_slope > 0 and h1_slope > 0:
        bias_side = "BUY"
        bias_text = "tăng"
        bias_detail = "H4 và H1 cùng nghiêng tăng"
    elif h4_slope < 0 and h1_slope < 0:
        bias_side = "SELL"
        bias_text = "giảm"
        bias_detail = "H4 và H1 cùng nghiêng giảm"
    elif h4_slope > 0 and h1_slope <= 0:
        bias_side = "BUY"
        bias_text = "tăng nhưng đang hồi"
        bias_detail = "H4 tăng, H1 đang chững hoặc hồi xuống"
    elif h4_slope < 0 and h1_slope >= 0:
        bias_side = "SELL"
        bias_text = "giảm nhưng đang hồi"
        bias_detail = "H4 giảm, H1 đang chững hoặc hồi lên"

    return bias_side, bias_text, bias_detail



def detect_location(m15: List[Candle]) -> Dict[str, Any]:
    if not m15:
        return {
            "range_low": 0.0,
            "range_high": 0.0,
            "current": 0.0,
            "range_pos": 50.0,
            "location_text": "Không đủ dữ liệu vị trí giá",
        }

    recent = m15[-20:] if len(m15) >= 20 else m15
    range_low = min(c.low for c in recent)
    range_high = max(c.high for c in recent)
    current = recent[-1].close
    width = max(range_high - range_low, 1e-9)
    range_pos = clamp((current - range_low) / width * 100.0, 0.0, 100.0)

    if range_pos >= 80:
        location_text = "Giá đang ở vùng cao của biên độ"
    elif range_pos <= 20:
        location_text = "Giá đang ở vùng thấp của biên độ"
    else:
        location_text = "Giá đang ở giữa biên độ"

    return {
        "range_low": round(range_low, 2),
        "range_high": round(range_high, 2),
        "current": round(current, 2),
        "range_pos": round(range_pos, 1),
        "location_text": location_text,
    }



def detect_liquidity(m15: List[Candle], atr: float) -> Dict[str, Any]:
    if len(m15) < 5:
        return {
            "has_sweep": False,
            "sweep_side": None,
            "liquidity_text": "Chưa đủ dữ liệu để đọc thanh khoản",
        }

    recent = m15[-6:]
    prev = recent[-2]
    last = recent[-1]

    up_sweep = last.high > max(c.high for c in recent[:-1]) and last.close < last.high - (last.range * 0.35)
    down_sweep = last.low < min(c.low for c in recent[:-1]) and last.close > last.low + (last.range * 0.35)

    liquidation = atr > 0 and last.range >= 1.8 * atr

    if up_sweep:
        text = "Vừa quét vùng thanh khoản phía trên"
        return {
            "has_sweep": True,
            "sweep_side": "up",
            "is_liquidation": liquidation,
            "liquidity_text": text,
        }
    if down_sweep:
        text = "Vừa quét vùng thanh khoản phía dưới"
        return {
            "has_sweep": True,
            "sweep_side": "down",
            "is_liquidation": liquidation,
            "liquidity_text": text,
        }

    if liquidation:
        return {
            "has_sweep": False,
            "sweep_side": None,
            "is_liquidation": True,
            "liquidity_text": "Vừa có nhịp quét mạnh nhưng chưa rõ hai đầu",
        }

    return {
        "has_sweep": False,
        "sweep_side": None,
        "is_liquidation": False,
        "liquidity_text": "Chưa thấy quét thanh khoản rõ",
    }



def detect_confirmation(m15: List[Candle]) -> Dict[str, Any]:
    if len(m15) < 8:
        return {
            "hl": False,
            "lh": False,
            "break_up": False,
            "break_down": False,
            "confirm_text": "Chưa đủ dữ liệu xác nhận",
        }

    recent = m15[-8:]
    highs = [c.high for c in recent]
    lows = [c.low for c in recent]
    closes = [c.close for c in recent]

    hl = lows[-1] > min(lows[:-1])
    lh = highs[-1] < max(highs[:-1])
    break_up = closes[-1] > max(highs[:-1])
    break_down = closes[-1] < min(lows[:-1])

    if break_up:
        text = "Giá vừa phá lên khỏi vùng gần nhất"
    elif break_down:
        text = "Giá vừa phá xuống khỏi vùng gần nhất"
    elif hl:
        text = "Đã có dấu hiệu giữ đáy cao hơn"
    elif lh:
        text = "Đã có dấu hiệu giữ đỉnh thấp hơn"
    else:
        text = "Chưa có xác nhận cấu trúc rõ"

    return {
        "hl": hl,
        "lh": lh,
        "break_up": break_up,
        "break_down": break_down,
        "confirm_text": text,
    }



def detect_flow(symbol: str, bias_side: str) -> Dict[str, str]:
    symbol = (symbol or "").upper()

    if symbol.startswith("XAU") or symbol.startswith("GOLD"):
        if bias_side == "SELL":
            return {"flow": "OUTFLOW", "flow_text": "Dòng tiền đang rút ra, nghiêng về phía bán", "favored": "SELL"}
        if bias_side == "BUY":
            return {"flow": "INFLOW", "flow_text": "Dòng tiền đang vào, nghiêng về phía mua", "favored": "BUY"}

    if symbol.startswith("BTC") or symbol.startswith("ETH"):
        if bias_side == "BUY":
            return {"flow": "INFLOW", "flow_text": "Dòng tiền đang vào nhóm rủi ro", "favored": "BUY"}
        if bias_side == "SELL":
            return {"flow": "OUTFLOW", "flow_text": "Dòng tiền đang rời nhóm rủi ro", "favored": "SELL"}

    return {"flow": "NEUTRAL", "flow_text": "Dòng tiền chưa rõ ràng", "favored": bias_side if bias_side in ("BUY", "SELL") else "SIDEWAY"}



def detect_market_state(
    bias_side: str,
    location: Dict[str, Any],
    liquidity: Dict[str, Any],
    confirmation: Dict[str, Any],
    atr: float,
    m15: List[Candle],
) -> Tuple[str, str]:
    if not m15:
        return "SIDEWAY_STATE", "Không đủ dữ liệu để đọc trạng thái thị trường"

    last = m15[-1]
    rsi = calc_rsi(m15)

    if liquidity.get("is_liquidation"):
        return "POST_LIQUIDATION_BOUNCE", "Sau nhịp quét mạnh, nên chờ giá ổn định lại"

    if bias_side == "SELL" and location.get("range_pos", 50) >= 70:
        return "BOUNCE_TO_SELL", "Giá đang hồi lên trong bối cảnh giảm; ưu tiên chờ hồi yếu để canh bán"

    if bias_side == "BUY" and location.get("range_pos", 50) <= 30:
        return "BOUNCE_TO_BUY", "Giá đang hồi xuống trong bối cảnh tăng; ưu tiên chờ hồi yếu để canh mua"

    width = safe_float(location.get("range_high")) - safe_float(location.get("range_low"))
    if atr > 0 and width < 2.2 * atr:
        return "SIDEWAY_STATE", "Thị trường đang đi ngang, dễ nhiễu hai đầu"

    if rsi < 28 and bias_side == "SELL":
        return "EXHAUSTION", "Đà giảm đang mạnh nhưng có thể sắp hồi kỹ thuật"
    if rsi > 72 and bias_side == "BUY":
        return "EXHAUSTION", "Đà tăng đang mạnh nhưng có thể sắp điều chỉnh"

    if confirmation.get("break_up") or confirmation.get("break_down"):
        return "TREND", "Giá vừa xác nhận phá cấu trúc, xu hướng đang rõ hơn"

    return "TRANSITION", "Thị trường đang chuyển trạng thái, chưa nên vội vào lệnh"


# -----------------------------
# GAP V5 - chỉ thêm ngữ cảnh, không đổi logic cũ
# -----------------------------



def detect_gap_context(m15: List[Candle], atr: float, bars_for_open: int = 4) -> GapContext:
    if len(m15) < bars_for_open + 2:
        return GapContext(
            gap_size=0.0,
            gap_pct_of_atr=0.0,
            open_range=0.0,
            open_range_pct_of_atr=0.0,
            has_big_gap=False,
            has_wide_open_range=False,
            unbalanced_open=False,
            note="Chưa đủ dữ liệu để đọc GAP đầu phiên",
        )

    prev = m15[-(bars_for_open + 2)]
    first = m15[-(bars_for_open + 1)]
    session_bars = m15[-bars_for_open:]

    gap_size = abs(first.open - prev.close)
    open_high = max(c.high for c in session_bars)
    open_low = min(c.low for c in session_bars)
    open_range = open_high - open_low

    gap_pct_atr = pct(gap_size, atr) if atr > 0 else 0.0
    open_range_pct_atr = pct(open_range, atr) if atr > 0 else 0.0

    has_big_gap = atr > 0 and gap_size >= 0.8 * atr
    has_wide_open_range = atr > 0 and open_range >= 1.6 * atr
    unbalanced_open = has_big_gap and has_wide_open_range

    if unbalanced_open:
        note = "Mở cửa lệch mạnh và biên độ đầu phiên quá rộng; dễ là giai đoạn quét thanh khoản hai đầu"
    elif has_big_gap:
        note = "Có GAP đầu phiên; chưa nên vội xem đây là tín hiệu vào lệnh ngay"
    elif has_wide_open_range:
        note = "Biên độ đầu phiên đang rộng bất thường; ưu tiên chờ ổn định hơn"
    else:
        note = "Mở cửa tương đối bình thường"

    return GapContext(
        gap_size=round(gap_size, 2),
        gap_pct_of_atr=round(gap_pct_atr, 1),
        open_range=round(open_range, 2),
        open_range_pct_of_atr=round(open_range_pct_atr, 1),
        has_big_gap=has_big_gap,
        has_wide_open_range=has_wide_open_range,
        unbalanced_open=unbalanced_open,
        note=note,
    )


# -----------------------------
# Phase / review / management
# -----------------------------



def calc_phase_369(location_pos: float, state_key: str) -> Tuple[str, str]:
    if state_key in ("POST_LIQUIDATION_BOUNCE", "EXHAUSTION"):
        return "3E", "Biến động quá mạnh, ưu tiên chờ"
    if 35 <= location_pos <= 65:
        return "6", "Đang ở giai đoạn chuẩn bị quyết định"
    if location_pos > 80 or location_pos < 20:
        return "9X", "Giá đang đi xa, dễ muộn nhịp"
    return "3", "Giai đoạn sớm, tín hiệu còn yếu"



def atr_plan(entry: float, bias_side: str, atr: float) -> Dict[str, float]:
    if atr <= 0 or entry <= 0:
        return {"sl": 0.0, "tp1": 0.0, "tp2": 0.0, "be_at": 0.0, "trail_at": 0.0}

    if bias_side == "SELL":
        return {
            "sl": round(entry + 1.1 * atr, 2),
            "tp1": round(entry - 0.9 * atr, 2),
            "tp2": round(entry - 1.8 * atr, 2),
            "be_at": round(entry - 0.8 * atr, 2),
            "trail_at": round(entry - 1.2 * atr, 2),
        }
    return {
        "sl": round(entry - 1.1 * atr, 2),
        "tp1": round(entry + 0.9 * atr, 2),
        "tp2": round(entry + 1.8 * atr, 2),
        "be_at": round(entry + 0.8 * atr, 2),
        "trail_at": round(entry + 1.2 * atr, 2),
    }


# -----------------------------
# API chính
# -----------------------------



def analyze_pro(symbol: str, data: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
    # FIX: luôn có mặc định để không bị lỗi bias_side chưa gán
    bias_side = "SIDEWAY"
    bias_text = "đi ngang"

    try:
        data = data or {}
        m15 = normalize_candles(data.get("m15"))
        h1 = normalize_candles(data.get("h1"))
        h4 = normalize_candles(data.get("h4"))

        if len(m15) < 5:
            return {
                "symbol": symbol,
                "bias_side": bias_side,
                "bias_text": bias_text,
                "status": "no_data",
                "message": "Bot chưa đọc đủ dữ liệu M15 để phân tích",
            }

        atr = calc_atr(m15, 14)
        bias_side, bias_text, bias_detail = detect_bias(h4, h1)
        location = detect_location(m15)
        liquidity = detect_liquidity(m15, atr)
        confirmation = detect_confirmation(m15)
        flow = detect_flow(symbol, bias_side)
        state_key, state_text = detect_market_state(bias_side, location, liquidity, confirmation, atr, m15)
        gap = detect_gap_context(m15, atr)
        phase_code, phase_text = calc_phase_369(location.get("range_pos", 50.0), state_key)

        no_trade_zone = bool(liquidity.get("is_liquidation") or gap.unbalanced_open)

        current = safe_float(location.get("current"))
        plan = atr_plan(current, bias_side if bias_side in ("BUY", "SELL") else "BUY", atr)

        main_scenario = "Chờ rõ hơn"
        if bias_side == "SELL":
            main_scenario = "Ưu tiên chờ giá hồi yếu rồi canh bán"
        elif bias_side == "BUY":
            main_scenario = "Ưu tiên chờ giá điều chỉnh nhẹ rồi canh mua"

        caution: List[str] = []
        if gap.unbalanced_open:
            caution.append("Mở cửa mất cân bằng, dễ quét hai đầu")
        if liquidity.get("is_liquidation"):
            caution.append("Vừa có nhịp quét mạnh, không nên đuổi giá")
        if no_trade_zone:
            caution.append("Hiện tại là vùng nên đứng ngoài hoặc giảm khối lượng")

        return {
            "symbol": symbol,
            "bias_side": bias_side,
            "bias_text": bias_text,
            "bias_detail": bias_detail,
            "phase_369": phase_code,
            "phase_text": phase_text,
            "atr_m15": round(atr, 2),
            "location": location,
            "liquidity": liquidity,
            "confirmation": confirmation,
            "flow": flow,
            "state_key": state_key,
            "state_text": state_text,
            "gap_context": asdict(gap),
            "no_trade_zone": no_trade_zone,
            "current_price": round(current, 2),
            "main_scenario": main_scenario,
            "atr_plan": plan,
            "caution": caution,
            "status": "ok",
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "bias_side": bias_side,
            "bias_text": bias_text,
            "status": "error",
            "message": f"Analysis failed ({symbol}): {e}",
        }


# -----------------------------
# Review lệnh thủ công
# -----------------------------



def review_manual_order(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    side = (side or "").upper()
    atr = safe_float(analysis.get("atr_m15"), 0.0)
    current = safe_float(analysis.get("current_price"), entry)
    location = analysis.get("location", {})
    confirmation = analysis.get("confirmation", {})
    liquidity = analysis.get("liquidity", {})
    gap_context = analysis.get("gap_context", {})

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    verdict = "CHƯA RÕ"
    note = "Cần thêm xác nhận"

    if side == analysis.get("bias_side") and rr >= 1.8:
        verdict = "TẠM ỔN"
        note = "Ý tưởng cùng hướng chính, nhưng vẫn cần quản trị chặt"

    if gap_context.get("unbalanced_open"):
        verdict = "THẬN TRỌNG"
        note = "Đầu phiên mở lệch mạnh; tránh thêm lệnh trong lúc nhiễu"

    if liquidity.get("is_liquidation"):
        verdict = "THẬN TRỌNG"
        note = "Vừa có nhịp quét mạnh; không nên đuổi theo nến"

    if side == "SELL" and not confirmation.get("lh") and not confirmation.get("break_down"):
        note = "Ý tưởng bán có thể đúng hướng, nhưng entry còn sớm vì chưa có xác nhận giảm rõ"
    if side == "BUY" and not confirmation.get("hl") and not confirmation.get("break_up"):
        note = "Ý tưởng mua có thể đúng hướng, nhưng entry còn sớm vì chưa có xác nhận tăng rõ"

    manage = "Có thể giữ ngắn hạn, không nên add"
    if analysis.get("no_trade_zone"):
        manage = "Ưu tiên giảm rủi ro, không add, chờ thị trường ổn định hơn"

    return {
        "symbol": symbol,
        "side": side,
        "entry": round(entry, 2),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "rr": rr,
        "phase_369": analysis.get("phase_369"),
        "phase_text": analysis.get("phase_text"),
        "state_text": analysis.get("state_text"),
        "bias_text": analysis.get("bias_text"),
        "current_price": round(current, 2),
        "range_pos": location.get("range_pos"),
        "flow_text": analysis.get("flow", {}).get("flow_text", "Dòng tiền chưa rõ"),
        "liquidity_text": liquidity.get("liquidity_text", "Chưa rõ thanh khoản"),
        "confirm_text": analysis.get("confirmation", {}).get("confirm_text", "Chưa rõ xác nhận"),
        "gap_note": gap_context.get("note", ""),
        "verdict": verdict,
        "note": note,
        "management": manage,
        "atr_m15": atr,
        "atr_plan": analysis.get("atr_plan", {}),
    }


# -----------------------------
# Render output dễ đọc
# -----------------------------



def render_analysis_text(result: Dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return result.get("message", "Bot chưa đọc được dữ liệu")

    location = result["location"]
    liquidity = result["liquidity"]
    gap = result["gap_context"]
    flow = result["flow"]
    confirmation = result["confirmation"]
    caution = result.get("caution", [])
    bias_detail = result.get("bias_detail", "")

    lines = [
        f"📌 {result['symbol']} | NOW",
        f"🧭 Xu hướng chính: {result['bias_text']}",
        f"💵 Giá hiện tại: {result['current_price']}",
        f"🪜 Giai đoạn 3-6-9: {result['phase_369']} | {result['phase_text']}",
        f"🌡 Trạng thái thị trường: {result['state_text']}",
        f"📍 Vị trí giá hiện tại: {location['range_pos']}% trong biên độ M15 ({location['range_low']} – {location['range_high']})",
        f"💧 Thanh khoản: {liquidity['liquidity_text']}",
        f"✅ Xác nhận hiện tại: {confirmation['confirm_text']}",
        f"💰 Dòng tiền: {flow['flow_text']}",
        f"🕳 GAP đầu phiên: {gap['note']}",
        f"🎯 Kịch bản chính: {result['main_scenario']}",
    ]

    if bias_detail:
        lines.insert(2, f"🧠 Bối cảnh lớn: {bias_detail}")

    if result.get("no_trade_zone"):
        lines.append("⛔ Hiện tại là vùng nên đứng ngoài, hoặc chỉ quan sát thêm")

    if caution:
        lines.append("⚠️ Lưu ý: " + " | ".join(caution))

    return "\n".join(lines)



def render_review_text(review: Dict[str, Any]) -> str:
    plan = review.get("atr_plan", {}) or {}
    lines = [
        f"🧠 REVIEW LỆNH | {review['symbol']} | {review['side']}",
        f"🎯 Entry: {review['entry']} | TP: {review['tp']} | SL: {review['sl']} | RR≈{review['rr']}",
        f"🧭 Xu hướng chính: {review.get('bias_text', 'chưa rõ')}",
        f"🪜 Giai đoạn 3-6-9: {review['phase_369']} | {review['phase_text']}",
        f"🌡 Trạng thái hiện tại: {review['state_text']}",
        f"📍 Vị trí giá hiện tại: ~{review['range_pos']}% biên độ",
        f"💰 Dòng tiền: {review['flow_text']}",
        f"💧 Thanh khoản: {review['liquidity_text']}",
        f"✅ Xác nhận hiện tại: {review['confirm_text']}",
        f"🕳 GAP đầu phiên: {review['gap_note']}",
        f"📌 Kết luận: {review['verdict']} — {review['note']}",
        f"⚙️ Hành động lúc này: {review['management']}",
    ]
    if plan:
        lines.extend([
            "📐 Gợi ý ATR Plan:",
            f"- SL chuẩn: {plan.get('sl', 0.0)}",
            f"- TP1: {plan.get('tp1', 0.0)}",
            f"- TP2: {plan.get('tp2', 0.0)}",
            f"- +0.8 ATR → dời về hòa vốn quanh: {plan.get('be_at', 0.0)}",
            f"- +1.2 ATR → bắt đầu trailing quanh: {plan.get('trail_at', 0.0)}",
        ])
    return "\n".join(lines)


__all__ = [
    "analyze_pro",
    "review_manual_order",
    "render_analysis_text",
    "render_review_text",
]
