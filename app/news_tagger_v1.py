# app/news_tagger_v1.py

def tag_news_item(title: str):
    t = (title or "").lower()
    tags = []

    if "fomc" in t:
        tags.append("FOMC")

    if "federal reserve" in t or " fed " in f" {t} ":
        if any(x in t for x in ["hawkish", "raise", "higher rates", "tightening", "rate hike", "hike"]):
            tags.append("FED_HAWKISH")
        elif any(x in t for x in ["cut", "dovish", "easing", "rate cut"]):
            tags.append("FED_DOVISH")
        else:
            tags.append("FED")

    if "interest rate" in t or "rates" in t:
        if any(x in t for x in ["hike", "increase", "raise", "higher"]):
            tags.append("RATE_HIKE")
        if any(x in t for x in ["cut", "reduce", "lower"]):
            tags.append("RATE_CUT")

    if "cpi" in t or "inflation" in t:
        if any(x in t for x in ["hot", "higher", "surge", "rises", "accelerates"]):
            tags.append("INFLATION_HIGH")
        else:
            tags.append("INFLATION")

    if any(x in t for x in ["war", "conflict", "attack", "missile", "strike", "geopolitical"]):
        tags.append("WAR")

    if any(x in t for x in ["bitcoin", "btc", "crypto", "etf"]):
        tags.append("CRYPTO")

    return list(dict.fromkeys(tags))