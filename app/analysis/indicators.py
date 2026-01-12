from __future__ import annotations
import numpy as np
import pandas as pd

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = (out["high"] - out["low"]).clip(lower=1e-9)
    out["upper_wick"] = (out["high"] - out[["open","close"]].max(axis=1)).clip(lower=0.0)
    out["lower_wick"] = (out[["open","close"]].min(axis=1) - out["low"]).clip(lower=0.0)
    out["dir"] = np.where(out["close"] >= out["open"], 1, -1)
    out["body_ratio"] = out["body"] / out["range"]
    out["two_sided_wick"] = ((out["upper_wick"] > out["range"]*0.25) & (out["lower_wick"] > out["range"]*0.25))
    return out

def swing_levels(df: pd.DataFrame, lookback: int = 60) -> tuple[float,float]:
    window = df.tail(lookback)
    return float(window["high"].max()), float(window["low"].min())

def find_liquidity_pools(df: pd.DataFrame, atr_val: float, lookback: int = 120) -> dict:
    """Infer liquidity pools from price action (Method #5):
    - equal highs / equal lows (clusters within tolerance)
    - recent swing highs/lows
    Returns levels for buy-side liquidity (above price) and sell-side liquidity (below price).
    """
    w = df.tail(lookback).copy()
    tol = max(0.8, 0.25*atr_val)  # $ tolerance for clustering
    highs = w["high"].values
    lows = w["low"].values
    closes = w["close"].values
    last_close = float(closes[-1])

    # Collect candidate pivots using simple fractal rule
    piv_hi=[]
    piv_lo=[]
    for i in range(2, len(w)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            piv_hi.append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            piv_lo.append(float(lows[i]))

    def cluster(levels):
        levels = sorted(levels)
        clusters=[]
        for lv in levels:
            if not clusters or abs(lv - clusters[-1][-1]) > tol:
                clusters.append([lv])
            else:
                clusters[-1].append(lv)
        # return cluster centers with strength
        out=[]
        for cl in clusters:
            center = sum(cl)/len(cl)
            strength = len(cl)
            out.append((center, strength))
        # prefer stronger first
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    hi_clusters = cluster(piv_hi)
    lo_clusters = cluster(piv_lo)

    # Liquidity above (buy-side) usually at equal highs / swing highs
    above = sorted([c for c in hi_clusters if c[0] > last_close], key=lambda x: x[0])
    below = sorted([c for c in lo_clusters if c[0] < last_close], key=lambda x: -x[0])  # nearest below first

    return {
        "tol": tol,
        "above": above,  # [(level, strength)...] ascending
        "below": below,  # [(level, strength)...] descending (nearest first)
    }
