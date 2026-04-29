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

    return ctx