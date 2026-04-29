# app/news_fetcher_v1.py
from __future__ import annotations
import os
import requests
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from app.news_tagger_v1 import tag_news_item

import logging
logger = logging.getLogger("app.news_fetcher")

def _dbg(msg):
    try:
        logger.info(msg)
    except Exception:
        pass

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()

NEWS_QUERY = (
    "Federal Reserve OR Fed OR FOMC OR CPI OR inflation OR interest rates "
    "OR rate cut OR rate hike OR war OR conflict OR missile OR attack "
    "OR gold OR bitcoin OR crypto"
)


def _parse_dt_utc(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def fetch_news_api(page_size: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch từ NewsAPI.
    Nếu không có NEWS_API_KEY hoặc API lỗi -> trả [] để bot không crash.
    """
    if not NEWS_API_KEY:
        return []

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": NEWS_QUERY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max(1, min(int(page_size), 50)),
            "apiKey": NEWS_API_KEY,
        }

        r = requests.get(url, params=params, timeout=8)
        _dbg(f"[NEWS API] status={res.status_code}")
        _dbg(f"[NEWS API] text={res.text[:300]}")
        data = r.json() if r is not None else {}

        if data.get("status") != "ok":
            return []

        articles = data.get("articles") or []
        return articles if isinstance(articles, list) else []

    except Exception:
        return []


def _impact_from_tags(tags: List[str]) -> str:
    tags = tags or []

    if any(t in tags for t in ["FED_HAWKISH", "FED_DOVISH", "RATE_HIKE", "RATE_CUT", "WAR", "FOMC"]):
        return "HIGH"

    if any(t in tags for t in ["INFLATION_HIGH", "INFLATION", "CPI", "NFP", "CRYPTO"]):
        return "MEDIUM"

    return "LOW"


def build_news_items():
    import requests

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "gold OR bitcoin OR fed OR inflation OR war",
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": os.getenv("NEWS_API_KEY")
    }

    try:
        _dbg(f"[NEWS API] calling...")

        res = requests.get(url, params=params, timeout=10)

        _dbg(f"[NEWS API] status = {res.status_code}")
        _dbg(f"[NEWS API] text = {res.text[:300]}")  # 🔥 QUAN TRỌNG

        data = res.json()

        articles = data.get("articles") or []

        _dbg(f"[NEWS API] articles count = {len(articles)}")

        items = []
        for a in articles:
            items.append({
                "title": a.get("title"),
                "desc": a.get("description"),
                "source": a.get("source", {}).get("name"),
            })

        return items

    except Exception as e:
        _dbg(f"[NEWS API ERROR] {e}")
        return []


def build_news_items_safe() -> List[Dict[str, Any]]:
    """
    Wrapper cực an toàn để gọi trong analyze_pro().
    """
    try:
        return build_news_items()
    except Exception:
        return []
