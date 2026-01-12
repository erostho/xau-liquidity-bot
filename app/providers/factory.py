from __future__ import annotations
from .base import MarketDataProvider
from .mock import MockProvider
from .twelvedata import TwelveDataProvider
from ..config import DATA_PROVIDER

def get_provider() -> MarketDataProvider:
    if DATA_PROVIDER == "twelvedata":
        return TwelveDataProvider()
    return MockProvider()
