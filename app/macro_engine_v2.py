
def build_macro_engine_v2(news_items: list) -> dict:
    ctx = {
        "macro_mode": "NEUTRAL",
        "usd_strength": 0,
        "risk_mode": "NEUTRAL",
        "gold_bias": "NEUTRAL",
        "btc_bias": "NEUTRAL",
        "confidence": 0,
        "drivers": []
    }

    score = 0

    for n in news_items:
        tags = n.get("tags", [])

        if "FED_HAWKISH" in tags or "RATE_HIKE" in tags:
            ctx["usd_strength"] += 2
            ctx["risk_mode"] = "RISK_OFF"
            ctx["drivers"].append("FED hawkish")
            score += 2

        if "FED_DOVISH" in tags or "RATE_CUT" in tags:
            ctx["usd_strength"] -= 2
            ctx["risk_mode"] = "RISK_ON"
            ctx["drivers"].append("FED dovish")
            score += 2

        if "INFLATION_HIGH" in tags:
            ctx["usd_strength"] += 1
            ctx["drivers"].append("inflation high")
            score += 1

        if "WAR" in tags:
            ctx["risk_mode"] = "RISK_OFF"
            ctx["drivers"].append("geopolitics risk")
            score += 2

    # ===== derive bias =====
    if ctx["usd_strength"] >= 2:
        ctx["gold_bias"] = "SELL"
        ctx["btc_bias"] = "SELL"
    elif ctx["usd_strength"] <= -2:
        ctx["gold_bias"] = "BUY"
        ctx["btc_bias"] = "BUY"

    if ctx["risk_mode"] == "RISK_OFF":
        ctx["gold_bias"] = "BUY"
        if ctx["btc_bias"] != "BUY":
            ctx["btc_bias"] = "MIXED"

    if ctx["risk_mode"] == "RISK_ON":
        ctx["btc_bias"] = "BUY"

    if score >= 4:
        ctx["macro_mode"] = "STRONG_THEME"
        ctx["confidence"] = 85
    elif score >= 2:
        ctx["macro_mode"] = "WEAK_THEME"
        ctx["confidence"] = 60
    else:
        ctx["macro_mode"] = "NEUTRAL"
        ctx["confidence"] = 40
    # ===== dedupe drivers =====
    ctx["drivers"] = list(dict.fromkeys(ctx.get("drivers") or []))
    return ctx

def explain_macro_reason_v1(macro: dict) -> list:
    macro = macro or {}
    reasons = []

    drivers = macro.get("drivers") or []
    risk = macro.get("risk_mode")
    gold = macro.get("gold_bias")
    btc = macro.get("btc_bias")

    # Geopolitics / war
    if any("geo" in d or "war" in d for d in drivers):
        if risk == "RISK_OFF":
            reasons.append("Căng thẳng địa chính trị → market chuyển sang RISK_OFF")
        if gold == "BUY":
            reasons.append("Risk-off → dòng tiền trú ẩn vào GOLD")

    # FED / lãi suất
    if any("fed" in d or "rate" in d for d in drivers):
        if "hawkish" in str(drivers).lower():
            reasons.append("FED hawkish → lãi suất cao → USD mạnh")
        if "dovish" in str(drivers).lower():
            reasons.append("FED dovish → hỗ trợ tài sản rủi ro")

    # Inflation
    if any("inflation" in d or "cpi" in d for d in drivers):
        reasons.append("Lạm phát cao → tăng nhu cầu hedge → hỗ trợ GOLD")

    # Crypto
    if btc == "SELL":
        reasons.append("Macro không ủng hộ crypto → BTC yếu")
    elif btc == "BUY":
        reasons.append("Macro hỗ trợ dòng tiền vào crypto")

    # fallback
    if not reasons:
        reasons.append("Chưa có yếu tố vĩ mô đủ mạnh")

    return reasons
    
TAG_EXPLANATION_MAP = {
    "WAR": "Chiến tranh/căng thẳng → dòng tiền trú ẩn → hỗ trợ GOLD",
    "GEOPOLITICS": "Rủi ro địa chính trị → market chuyển sang RISK_OFF",
    
    "FED_HAWKISH": "FED hawkish → lãi suất cao → USD mạnh → áp lực lên crypto",
    "FED_DOVISH": "FED dovish → nới lỏng → hỗ trợ tài sản rủi ro",

    "INFLATION_HIGH": "Lạm phát cao → nhu cầu hedge tăng → GOLD được hỗ trợ",
    "CPI": "Dữ liệu CPI → ảnh hưởng kỳ vọng lãi suất",

    "RATE_HIKE": "Tăng lãi suất → giảm thanh khoản → tài sản rủi ro bị bán",
    "RATE_CUT": "Giảm lãi suất → tăng thanh khoản → hỗ trợ crypto và stocks",

    "CRYPTO": "Tin tức crypto → ảnh hưởng trực tiếp BTC",
}
def explain_tags_v1(news_items: list) -> list:
    explanations = []

    for item in news_items:
        tags = item.get("tags") or []

        for t in tags:
            exp = TAG_EXPLANATION_MAP.get(t)
            if exp and exp not in explanations:
                explanations.append(exp)

    return explanations
