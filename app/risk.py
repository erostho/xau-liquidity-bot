# app/risk.py
from __future__ import annotations
from typing import Dict, Any, Optional

def calc_smart_sl_tp(
    *,
    symbol: str,
    side: str,                 # "BUY" / "SELL"
    entry: float,              # entry price (market or limit)
    atr: float,                # ATR(M15) in price units (XAU: dollars)
    liquidity_level: Optional[float] = None,   # swing high/low or liquidity line (price)
    equity_usd: float = 1000.0,                # account equity in USD
    risk_pct: float = 0.0075,                  # 0.005 .. 0.01 (0.5%..1%)
    leverage: float = 100.0,                   # for info only (margin calc optional)
    rr1: float = 1.0,                          # TP1 = rr1 * R
    rr2: float = 1.6,                          # TP2 = rr2 * R
    # --- caps & buffers (tune for 5-10 setups/day) ---
    atr_k: float = 1.0,                        # SL_ATR = atr_k * ATR
    max_atr_k: float = 1.25,                   # hard cap: SL <= max_atr_k*ATR
    buf_atr_k: float = 0.25,                   # liquidity buffer = buf_atr_k*ATR
    min_buf_xau: float = 0.30,                 # absolute buffer for XAU to avoid micro-sweeps
    # --- contract / pip value (Exness MT5 XAUUSD usually 100 oz/lot) ---
    contract_size: float = 100.0,              # XAUUSD: 100 oz per lot (common). Adjust if your broker differs.
) -> Dict[str, Any]:
    """
    Smart SL/TP calculator for XAU M15 (works for BTC too if inputs are correct).
    Requirement: SL = MIN(Liquidity SL, ATR-based SL, Risk-based SL)

    Returns:
      {
        "ok": bool,
        "reason": str,
        "entry": float,
        "sl": float,
        "tp1": float,
        "tp2": float,
        "r": float,                 # risk distance in price units ($ for XAU)
        "risk_usd": float,
        "risk_pct": float,
        "lot": float,               # suggested lot size to respect risk_pct using final SL
        "sl_components": {...},
        "notes": [...]
      }
    """
    side_u = (side or "").strip().upper()
    if side_u not in ("BUY", "SELL"):
        return {"ok": False, "reason": "side must be BUY/SELL"}

    if entry <= 0:
        return {"ok": False, "reason": "invalid entry"}

    if atr is None or atr <= 0:
        return {"ok": False, "reason": "invalid ATR (need ATR(M15) > 0)"}

    if equity_usd <= 0:
        return {"ok": False, "reason": "invalid equity_usd"}

    # Clamp risk_pct to [0.5%, 1%] default intent
    risk_pct = max(0.005, min(0.01, float(risk_pct)))
    risk_usd = equity_usd * risk_pct

    # --- buffer ---
    # Use an ATR buffer + a minimum absolute buffer (for XAU spread/quotes)
    sym = (symbol or "").upper()
    min_buf = min_buf_xau if "XAU" in sym else 0.0
    buf = max(buf_atr_k * atr, min_buf)

    notes = []

    # --- 1) ATR-based SL distance ---
    sl_atr_dist = atr_k * atr
    sl_atr_cap = max_atr_k * atr
    sl_atr_dist = min(sl_atr_dist, sl_atr_cap)

    # --- 2) Liquidity-based SL distance ---
    # Liquidity SL means: beyond liquidity_level +/- buffer.
    # If liquidity_level is missing, treat liquidity SL as +inf (so MIN ignores it)
    sl_liq_dist = float("inf")
    if liquidity_level is not None and liquidity_level > 0:
        if side_u == "SELL":
            # stop above liquidity
            sl_price_liq = max(liquidity_level, entry) + buf
            sl_liq_dist = abs(sl_price_liq - entry)
        else:
            sl_price_liq = min(liquidity_level, entry) - buf
            sl_liq_dist = abs(entry - sl_price_liq)
    else:
        notes.append("Không có liquidity_level -> bỏ qua thành phần Liquidity SL.")

    # --- 3) Risk-based SL distance ---
    # risk_usd = sl_dist * contract_size * lot  => sl_dist = risk_usd/(contract_size*lot)
    # But lot is what we want to compute. For SL selection rule, we convert risk limit into a max allowed SL distance
    # GIVEN a minimum tradable lot. If you don't know your minimum lot, assume 0.01.
    MIN_LOT = 0.01
    sl_risk_dist = risk_usd / (contract_size * MIN_LOT)  # max SL distance if you trade MIN_LOT
    # For XAU: contract_size=100 => 1$ move = $100 per lot. For 0.01 lot: $1 move = $1.
    # So sl_risk_dist becomes "how far SL can be for MIN_LOT" to still keep risk <= risk_usd.

    # Final SL distance per your rule:
    sl_dist = min(sl_liq_dist, sl_atr_dist, sl_risk_dist)

    # Hard sanity: never let SL be unrealistically tiny (will be swept)
    # Use at least 0.45*ATR or min_buf*1.2 (tune)
    sl_min_dist = max(1.0 * atr, min_buf * 1.2)
    if sl_dist < sl_min_dist:
        notes.append(f"SL quá ngắn ({sl_dist:.2f}) -> nâng lên tối thiểu theo nhiễu thị trường.")
        sl_dist = sl_min_dist

    # Also enforce cap again
    sl_dist = min(sl_dist, sl_atr_cap)

    # If after MIN rule we exceed risk_usd for MIN_LOT, we still can reduce lot size below MIN_LOT? usually cannot.
    # So we detect untradeable setups.
    risk_at_min_lot = sl_dist * contract_size * MIN_LOT
    if risk_at_min_lot > risk_usd * 1.05:
        return {
            "ok": False,
            "reason": "SL tối thiểu theo nhiễu thị trường vẫn vượt risk với lot tối thiểu 0.01. Giảm risk_pct hoặc bỏ kèo.",
            "entry": entry,
            "risk_usd": risk_usd,
            "risk_pct": risk_pct,
            "sl_min_dist": sl_min_dist,
            "risk_at_min_lot": risk_at_min_lot,
        }

    # Compute SL price
    if side_u == "SELL":
        sl = entry + sl_dist
        tp1 = entry - rr1 * sl_dist
        tp2 = entry - rr2 * sl_dist
    else:
        sl = entry - sl_dist
        tp1 = entry + rr1 * sl_dist
        tp2 = entry + rr2 * sl_dist

    # Suggested lot size to match risk_usd with this SL
    # lot = risk_usd / (sl_dist * contract_size)
    lot = risk_usd / (sl_dist * contract_size)
    # Round down to 0.01 steps (common). You can adapt.
    lot = max(MIN_LOT, (int(lot * 100) / 100.0))

    # If lot is huge, warn
    if lot > 5:
        notes.append("Lot gợi ý khá lớn; kiểm tra contract_size/broker hoặc giảm risk_pct.")

    components = {
        "sl_liq_dist": None if sl_liq_dist == float("inf") else sl_liq_dist,
        "sl_atr_dist": sl_atr_dist,
        "sl_risk_dist_at_min_lot": sl_risk_dist,
        "sl_final_dist": sl_dist,
        "buffer": buf,
        "sl_min_dist": sl_min_dist,
        "sl_atr_cap": sl_atr_cap,
        "min_lot": MIN_LOT,
        "contract_size": contract_size,
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
