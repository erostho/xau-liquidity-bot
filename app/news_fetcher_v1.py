# app/news_fetcher_v1.py
from __future__ import annotations

import os
import requests
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from app.news_tagger_v1 import tag_news_item


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


def build_news_items(max_items: int = 8, lookback_hours: int = 12) -> List[Dict[str, Any]]:
    """
    Output chuẩn cho macro_engine_v2:

    [
      {
        "title": "...",
        "source": "...",
        "published_at": "...",
        "url": "...",
        "tags": ["FED_HAWKISH"],
        "impact": "HIGH"
      }
    ]
    """
    raw = fetch_news_api(page_size=30)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1, int(lookback_hours)))

    out: List[Dict[str, Any]] = []
    seen = set()

    for art in raw:
        try:
            title = str(art.get("title") or "").strip()
            if not title:
                continue

            published_at = art.get("publishedAt")
            dt = _parse_dt_utc(published_at)

            # Nếu có timestamp thì lọc theo lookback.
            if dt is not None and dt < cutoff:
                continue

            tags = tag_news_item(title)
            if not tags:
                continue

            key = title.lower()
            if key in seen:
                continue
            seen.add(key)

            source_obj = art.get("source") or {}
            source = source_obj.get("name") if isinstance(source_obj, dict) else ""

            out.append(
                {
                    "title": title,
                    "source": source or "newsapi",
                    "published_at": published_at,
                    "url": art.get("url"),
                    "tags": tags,
                    "impact": _impact_from_tags(tags),
                }
            )

            if len(out) >= int(max_items):
                break

        except Exception:
            continue

    return out


def build_news_items_safe() -> List[Dict[str, Any]]:
    """
    Wrapper cực an toàn để gọi trong analyze_pro().
    """
    try:
        return build_news_items()
    except Exception:
        return []