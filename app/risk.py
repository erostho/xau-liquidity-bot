# app/risk.py
from __future__ import annotations
from typing import Dict, Any, Optional
import math

def calc_smart_sl_tp(
    *,
    symbol: str,
    side: str,                 # "BUY" / "SELL"
    entry: float,              # entry price (market or limit)
    atr: float,                # ATR(M15) in price units
    liquidity_level: Optional[float] = None,   # swing high/low or liquidity line (price)
    equity_usd: float = 1000.0,                # account equity in USD
    risk_pct: float = 0.0075,                  # 0.005 .. 0.01
    leverage: float = 100.0,                   # info only
    rr1: float = 1.0,                          # TP1 = rr1 * R
    rr2: float = 1.6,                          # TP2 = rr2 * R
    # --- caps & buffers ---
    atr_k: float = 1.0,                        # SL_ATR = atr_k * ATR
    max_atr_k: float = 1.25,                   # hard cap: SL <= max_atr_k*ATR
    buf_atr_k: float = 0.25,                   # liquidity buffer = buf_atr_k*ATR
    min_buf_xau: float = 0.30,                 # absolute buffer for XAU
    # --- contract / pip value ---
    contract_size: float = 100.0,              # default for XAU (100 oz/lot). Auto override by symbol below.
) -> Dict[str, Any]:
    """
    Smart SL/TP:
      - Ưu tiên SL "sau vùng sweep": liquidity_level +/- buffer (nếu có).
      - SL cuối cùng = MIN(Liquidity SL distance, ATR SL distance, Risk SL distance at MIN_LOT)
      - Có SL tối thiểu theo ATR để tránh đặt quá sát (dễ bị quét).
      - TP1/TP2 theo RR * R.

    Return keys giữ nguyên để pro_analysis.py dùng:
      ok, reason, entry, sl, tp1, tp2, r, risk_usd, risk_pct, lot, sl_components, notes
    """

    # ---------- validate ----------
    side_u = (side or "").strip().upper()
    if side_u not in ("BUY", "SELL"):
        return {"ok": False, "reason": "side must be BUY/SELL"}

    try:
        entry = float(entry)
    except Exception:
        return {"ok": False, "reason": "invalid entry"}

    if entry <= 0:
        return {"ok": False, "reason": "invalid entry"}

    try:
        atr = float(atr)
    except Exception:
        return {"ok": False, "reason": "invalid ATR"}

    if atr <= 0:
        return {"ok": False, "reason": "invalid ATR (need ATR(M15) > 0)"}

    try:
        equity_usd = float(equity_usd)
    except Exception:
        return {"ok": False, "reason": "invalid equity_usd"}

    if equity_usd <= 0:
        return {"ok": False, "reason": "invalid equity_usd"}

    # ---------- symbol profile (auto sane defaults) ----------
    sym = (symbol or "").upper()

    # contract_size:
    # - XAUUSD thường 100 oz/lot (1$ move = $100/lot)
    # - XAGUSD nhiều broker là 5000 oz/lot (1$ move = $5000/lot) -> rất nhạy, nên min lot quan trọng
    # - BTCUSD CFD tuỳ broker; thường 1 BTC/lot hoặc 0.1. Ở đây default 1.0 để không quá ảo.
    # Nếu mày truyền contract_size từ nơi khác thì vẫn được (param vẫn giữ).
    auto_contract = contract_size
    if "XAG" in sym:
        auto_contract = 5000.0
    elif "XAU" in sym:
        auto_contract = 100.0
    elif "BTC" in sym:
        auto_contract = 1.0

    # MIN_LOT: đa số MT5 là 0.01; BTC CFD có thể 0.01. Giữ 0.01 an toàn.
    min_lot = 0.01

    # buffer tối thiểu theo symbol (để tránh “chạm nhẹ/spread”)
    # XAG hay quét sâu + biến động mạnh => buffer lớn hơn chút
    if "XAG" in sym:
        min_buf_abs = 0.06   # $0.06 trên XAG (tuỳ quote), chỉnh nếu cần
        sl_min_atr_k = 1.15  # SL tối thiểu ~ 1.15 ATR
    elif "XAU" in sym:
        min_buf_abs = float(min_buf_xau)  # default 0.30
        sl_min_atr_k = 1.00
    elif "BTC" in sym:
        min_buf_abs = 0.0
        sl_min_atr_k = 1.00
    else:
        min_buf_abs = 0.0
        sl_min_atr_k = 1.00

    notes = []

    # ---------- risk budget ----------
    risk_pct = max(0.005, min(0.01, float(risk_pct)))
    risk_usd = equity_usd * risk_pct

    # ---------- buffer (atr-based + absolute min) ----------
    buf = max(float(buf_atr_k) * atr, float(min_buf_abs))

    # ---------- (1) ATR-based SL distance ----------
    sl_atr_dist = float(atr_k) * atr
    sl_atr_cap = float(max_atr_k) * atr
    if sl_atr_dist > sl_atr_cap:
        sl_atr_dist = sl_atr_cap

    # ---------- (2) Liquidity-based SL distance ----------
    # Liquidity SL: "sau vùng sweep" => vượt liquidity_level + buffer
    sl_liq_dist = float("inf")
    if liquidity_level is not None:
        try:
            lv = float(liquidity_level)
        except Exception:
            lv = None
        if lv is not None and lv > 0:
            if side_u == "SELL":
                # stop phía trên liquidity
                sl_price_liq = max(lv, entry) + buf
                sl_liq_dist = abs(sl_price_liq - entry)
            else:
                # BUY: stop phía dưới liquidity
                sl_price_liq = min(lv, entry) - buf
                sl_liq_dist = abs(entry - sl_price_liq)
        else:
            notes.append("liquidity_level không hợp lệ -> bỏ qua Liquidity SL.")
    else:
        notes.append("Không có liquidity_level -> bỏ qua Liquidity SL.")

    # ---------- (3) Risk-based SL distance (at min lot) ----------
    # risk_usd = sl_dist * contract_size * lot  => max sl_dist at MIN_LOT:
    # sl_risk_dist = risk_usd / (contract_size * MIN_LOT)
    if auto_contract <= 0:
        return {"ok": False, "reason": "invalid contract_size"}
    sl_risk_dist = risk_usd / (auto_contract * min_lot)

    # ---------- final SL selection: MIN(...) ----------
    sl_dist = min(sl_liq_dist, sl_atr_dist, sl_risk_dist)

    # ---------- enforce minimum SL distance (để tránh SL quá sát) ----------
    # “vào muộn sau quét + chờ cấu trúc” => thường entry ở giữa hồi, nếu SL quá ngắn rất dễ bị quét lại
    sl_min_dist = max(sl_min_atr_k * atr, min_buf_abs * 1.2)
    if sl_dist < sl_min_dist:
        notes.append(f"SL quá ngắn ({sl_dist:.3f}) -> nâng lên tối thiểu {sl_min_dist:.3f} (theo ATR/nhiễu).")
        sl_dist = sl_min_dist

    # cap again
    if sl_dist > sl_atr_cap:
        notes.append(f"SL bị cap về {sl_atr_cap:.3f} (max_atr_k).")
        sl_dist = sl_atr_cap

    # ---------- tradability check at MIN_LOT ----------
    risk_at_min_lot = sl_dist * auto_contract * min_lot
    if risk_at_min_lot > risk_usd * 1.05:
        return {
            "ok": False,
            "reason": "SL tối thiểu theo nhiễu vẫn vượt risk với lot tối thiểu 0.01 -> giảm risk_pct hoặc bỏ kèo.",
            "entry": float(entry),
            "risk_usd": float(risk_usd),
            "risk_pct": float(risk_pct),
            "sl_min_dist": float(sl_min_dist),
            "risk_at_min_lot": float(risk_at_min_lot),
            "contract_size": float(auto_contract),
            "min_lot": float(min_lot),
        }

    # ---------- compute SL/TP prices ----------
    if side_u == "SELL":
        sl = entry + sl_dist
        tp1 = entry - float(rr1) * sl_dist
        tp2 = entry - float(rr2) * sl_dist
    else:
        sl = entry - sl_dist
        tp1 = entry + float(rr1) * sl_dist
        tp2 = entry + float(rr2) * sl_dist

    # ---------- suggested lot size (rounded down 0.01) ----------
    lot_raw = risk_usd / (sl_dist * auto_contract)
    if not math.isfinite(lot_raw) or lot_raw <= 0:
        lot_raw = min_lot

    lot = max(min_lot, math.floor(lot_raw * 100.0) / 100.0)

    if lot > 5:
        notes.append("Lot gợi ý khá lớn; kiểm tra contract_size/broker hoặc giảm risk_pct.")

    components = {
        "sl_liq_dist": None if sl_liq_dist == float("inf") else float(sl_liq_dist),
        "sl_atr_dist": float(sl_atr_dist),
        "sl_risk_dist_at_min_lot": float(sl_risk_dist),
        "sl_final_dist": float(sl_dist),
        "buffer": float(buf),
        "sl_min_dist": float(sl_min_dist),
        "sl_atr_cap": float(sl_atr_cap),
        "min_lot": float(min_lot),
        "contract_size": float(auto_contract),
        "profile": {
            "symbol": sym,
            "min_buf_abs": float(min_buf_abs),
            "sl_min_atr_k": float(sl_min_atr_k),
        }
    }

    return {
        "ok": True,
        "reason": "OK",
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "r": float(sl_dist),
        "risk_usd": float(risk_usd),
        "risk_pct": float(risk_pct),
        "lot": float(lot),
        "sl_components": components,
        "notes": notes,
    }
